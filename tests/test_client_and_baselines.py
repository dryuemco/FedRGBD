"""CPU tests: FL client fit/evaluate metrics and the local/centralized baselines.

Uses a tiny synthetic image dataset (12 px images, ~20 images per node) and a
randomly initialised MobileNetV3-Small (``pretrained=False``) so no download
and no GPU is required.  The model architecture is the one used in the paper.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from src.evaluation.metrics import METRIC_KEYS  # noqa: E402

IMG = 16


def _make_node(root, n_fire=6, n_nofire=6, seed=0):
    rng = np.random.RandomState(seed)
    for split, k in (("train", 1.0), ("val", 0.5), ("test", 0.5)):
        for cls, n in (("Fire", n_fire), ("No_Fire", n_nofire)):
            d = os.path.join(root, split, cls)
            os.makedirs(d, exist_ok=True)
            for i in range(max(int(n * k), 2)):
                arr = rng.randint(0, 255, (IMG, IMG, 3), dtype=np.uint8)
                if cls == "Fire":
                    arr[..., 0] = 255  # red-ish "fire" images
                Image.fromarray(arr).save(os.path.join(d, f"{split}_{cls}_{i}.png"))
    return root


@pytest.fixture(scope="module")
def nodes(tmp_path_factory):
    base = tmp_path_factory.mktemp("processed")
    return [_make_node(str(base / f"node_{c}"), seed=i) for i, c in enumerate("abc")]


def test_client_fit_and_evaluate_return_full_metrics(nodes):
    from src.fl.client import FedRGBDClient, payload_bytes

    torch.set_num_threads(1)
    client = FedRGBDClient(nodes[0], batch_size=4, lr=1e-3, local_epochs=1, device="cpu",
                           seed=42, pretrained=False, img_size=IMG)
    assert client.node_name == "node_a"
    params = client.get_parameters({})
    n_bytes = payload_bytes(params)
    assert n_bytes == sum(v.numel() * v.element_size() for v in client.model.state_dict().values())

    new_params, n_train, fit_m = client.fit(params, {"server_round": 2})
    assert n_train == len(client.train_ds)
    assert fit_m["server_round"] == 2 and fit_m["node_name"] == "node_a"
    assert fit_m["payload_bytes_down"] == n_bytes and fit_m["payload_bytes_up"] == n_bytes
    assert fit_m["fit_time_s"] > 0 and fit_m["fit_wall_s"] >= fit_m["fit_time_s"]
    assert fit_m["local_epochs"] == 1 and fit_m["batch_size"] == 4 and fit_m["proximal_mu"] == 0.0
    # v2 keys still present
    for k in ("train_loss", "train_time", "hostname", "strategy"):
        assert k in fit_m
    assert len(new_params) == len(params)

    # FedProx path
    _, _, prox_m = client.fit(params, {"server_round": 3, "proximal_mu": 0.01})
    assert prox_m["strategy"].startswith("FedProx") and prox_m["proximal_mu"] == 0.01

    loss, n_eval, ev = client.evaluate(new_params, {"server_round": 2})
    assert n_eval == len(client.val_ds) and ev["num_examples"] == n_eval
    assert isinstance(loss, float) and loss >= 0
    for k in METRIC_KEYS:
        assert k in ev, k
        assert isinstance(ev[k], float)
    assert 0.0 <= ev["accuracy"] <= 1.0 and -1.0 <= ev["mcc"] <= 1.0
    assert ev["cm_0_0"] + ev["cm_0_1"] + ev["cm_1_0"] + ev["cm_1_1"] == n_eval
    assert json.loads(ev["confusion_matrix_json"])[0][0] == ev["cm_0_0"]
    assert ev["server_round"] == 2 and ev["eval_split"] == "val"
    assert ev["eval_time_s"] > 0 and ev["payload_bytes_down"] == n_bytes
    # Flower requires scalar metric values
    assert all(isinstance(v, (bool, int, float, str, bytes)) for v in ev.values())
    assert not any(isinstance(v, float) and np.isnan(v) for v in ev.values())

    # FedBN mode keeps BN params local
    client.fit(params, {"server_round": 4, "fedbn": True})
    assert client._fedbn_mode is True


def test_client_eval_split_test(nodes):
    from src.fl.client import FedRGBDClient

    client = FedRGBDClient(nodes[1], batch_size=4, local_epochs=1, device="cpu", seed=1,
                           pretrained=False, img_size=IMG, eval_split="test", node_name="custom")
    assert client.node_name == "custom"
    _, n, ev = client.evaluate(client.get_parameters({}), {})
    assert n == len(client.test_ds) and ev["eval_split"] == "test" and ev["server_round"] == 0


def test_train_local_batch_writes_full_metrics(nodes, tmp_path):
    from scripts import train_local

    out = tmp_path / "local"
    train_local.main(["--batch", "--cross_eval", "--data_dirs", nodes[0], nodes[1],
                      "--epochs", "1", "--batch_size", "4", "--seed", "42",
                      "--output_dir", str(out), "--no_pretrained", "--img_size", str(IMG)])
    res = json.loads((out / "node_a" / "results.json").read_text())
    # v2 keys
    for k in ("experiment", "node_name", "final_test_accuracy", "final_test_loss", "history", "cross_eval",
              "fl_round_equivalents", "train_class_distribution"):
        assert k in res
    assert res["history"][0]["val_accuracy"] == pytest.approx(res["history"][0]["val_metrics"]["accuracy"])
    # v3 keys
    assert res["results_schema_version"] == 2
    for k in METRIC_KEYS:
        assert k in res["final_test_metrics"]
        assert k in res["history"][0]["val_metrics"]
    assert res["final_test_metrics"]["accuracy"] == pytest.approx(res["final_test_accuracy"])
    assert len(res["final_test_metrics"]["confusion_matrix"]) == 2
    assert res["history"][0]["train_time_s"] > 0 and res["history"][0]["eval_time_s"] > 0
    assert "test_metrics" in res["cross_eval"]["node_b"]
    assert res["device"] in ("cpu", "cuda")   # provenance: which device trained this baseline
    summary = json.loads((out / "summary.json").read_text())
    assert "final_test_metrics" in summary["nodes"]["node_a"] and "mean_test_accuracy" in summary
    assert "NaN" not in (out / "node_a" / "results.json").read_text()


def test_train_centralized_writes_full_metrics(nodes, tmp_path):
    from scripts import train_centralized

    out = tmp_path / "central"
    train_centralized.main(["--data_dirs", nodes[0], nodes[1], "--epochs", "1", "--batch_size", "4",
                            "--seed", "42", "--output_dir", str(out), "--no_pretrained", "--img_size", str(IMG)])
    res = json.loads((out / "results.json").read_text())
    for k in ("experiment", "final_test_accuracy", "final_test_loss", "per_node_test", "history", "fl_round_equivalents"):
        assert k in res
    assert res["results_schema_version"] == 2
    for k in METRIC_KEYS:
        assert k in res["final_test_metrics"] and k in res["history"][0]["val_metrics"]
    assert "test_metrics" in res["per_node_test"]["node_a"]
    assert res["history"][0]["elapsed_s"] > 0
    assert res["device"] in ("cpu", "cuda")   # provenance: which device trained this baseline


@pytest.mark.parametrize("script_name", ["train_local", "train_centralized"])
def test_fl_round_equivalents_cover_every_completed_round(nodes, tmp_path, script_name):
    """``fl_round_equivalents`` must follow ``--epochs``, not a hard-coded 3 rounds.

    The revision's long-horizon block runs FL for 10 rounds, so its matching
    baselines run 50 epochs.  ``analyze_results.load_centralized_run`` prefers
    ``fl_round_equivalents`` over the epoch-derived mapping, so a dict capped at
    round 3 silently truncates the baseline curve to 3 of 10 rounds.
    """
    import importlib

    script = importlib.import_module("scripts." + script_name)
    out = tmp_path / ("equiv_" + script_name)
    common = ["--epochs", "11", "--batch_size", "4", "--seed", "42",
              "--output_dir", str(out), "--no_pretrained", "--img_size", str(IMG)]
    if script_name == "train_local":
        script.main(["--batch", "--data_dirs", nodes[0]] + common)
        res = json.loads((out / "node_a" / "results.json").read_text())
    else:
        script.main(["--data_dirs", nodes[0]] + common)
        res = json.loads((out / "results.json").read_text())

    equiv = res["fl_round_equivalents"]
    assert sorted(equiv) == ["round_1", "round_2"]           # 11 epochs -> 2 full rounds
    assert all(v is not None for v in equiv.values())
    assert equiv["round_1"]["epoch"] == 5 and equiv["round_2"]["epoch"] == 10


def test_fl_round_equivalents_empty_for_short_runs(nodes, tmp_path):
    from scripts import train_centralized

    out = tmp_path / "equiv_short"
    train_centralized.main(["--data_dirs", nodes[0], "--epochs", "2", "--batch_size", "4",
                            "--seed", "42", "--output_dir", str(out), "--no_pretrained",
                            "--img_size", str(IMG)])
    res = json.loads((out / "results.json").read_text())
    # fewer than 5 epochs -> no complete FL round, and no null placeholders that
    # would make analyze_results invent empty rounds 1-3
    assert res["fl_round_equivalents"] == {}
