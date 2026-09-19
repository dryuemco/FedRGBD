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


@pytest.fixture(scope="module")
def uneven_node(tmp_path_factory):
    """A node whose validation and test splits differ in size (4 vs 10 images)."""
    root = str(tmp_path_factory.mktemp("uneven") / "node_u")
    rng = np.random.RandomState(7)
    for split, per_class in (("train", 4), ("val", 2), ("test", 5)):
        for cls in ("Fire", "No_Fire"):
            d = os.path.join(root, split, cls)
            os.makedirs(d, exist_ok=True)
            for i in range(per_class):
                arr = rng.randint(0, 255, (IMG, IMG, 3), dtype=np.uint8)
                Image.fromarray(arr).save(os.path.join(d, f"{split}_{cls}_{i}.png"))
    return root


def test_client_evaluates_val_and_test_with_namespaced_metrics(uneven_node):
    from src.fl.client import FedRGBDClient

    client = FedRGBDClient(uneven_node, batch_size=4, local_epochs=1, device="cpu", seed=1,
                           pretrained=False, img_size=IMG, node_name="custom")
    assert client.node_name == "custom"
    n_val, n_test = len(client.val_ds), len(client.test_ds)
    assert (n_val, n_test) == (4, 10)

    loss, n, ev = client.evaluate(client.get_parameters({}), {"server_round": 3})

    # what Flower aggregates (loss, num_examples) is the validation split
    assert n == n_val and ev["num_examples"] == n_val and ev["eval_split"] == "val"
    assert loss == pytest.approx(ev["val_loss"])
    assert ev["val_n_examples"] == n_val and ev["test_n_examples"] == n_test
    # full metric set in both namespaces; unprefixed keys = validation (v3 compatibility)
    for k in METRIC_KEYS:
        assert isinstance(ev["val_" + k], float), k
        assert isinstance(ev["test_" + k], float), k
        assert ev[k] == ev["val_" + k], k
    for ns, n_split in (("val_", n_val), ("test_", n_test)):
        cm = json.loads(ev[ns + "confusion_matrix_json"])
        assert sum(map(sum, cm)) == n_split
        assert ev[ns + "tp"] + ev[ns + "fp"] + ev[ns + "fn"] + ev[ns + "tn"] == n_split
    assert "test_loss" in ev
    assert ev["server_round"] == 3
    # Flower requires scalar metric values
    assert all(isinstance(v, (bool, int, float, str, bytes)) for v in ev.values())


def test_client_test_pass_is_report_only(uneven_node, monkeypatch):
    """The test split runs after validation, changes nothing, and never reaches fit()."""
    from src.fl.client import FedRGBDClient

    client = FedRGBDClient(uneven_node, batch_size=4, local_epochs=1, device="cpu", seed=1,
                           pretrained=False, img_size=IMG)
    params = client.get_parameters({})

    calls = []
    original = client.evaluate_loader

    def spy(loader):
        calls.append("test" if loader is client.test_loader else
                     "val" if loader is client.val_loader else "other")
        return original(loader)

    monkeypatch.setattr(client, "evaluate_loader", spy)
    loss, n, ev = client.evaluate(params, {"server_round": 1})
    assert calls == ["val", "test"]          # selection inputs are fixed before test runs

    # the test pass leaves the model untouched (eval mode, no grad): same state as loaded
    after = client.get_parameters({})
    assert all(np.array_equal(a, b) for a, b in zip(params, after))

    # the returned loss does not depend on the test split: replacing the test
    # split's metrics with garbage leaves loss / num_examples / val_* unchanged
    def poisoned(loader):
        m = original(loader)
        if loader is client.test_loader:
            m = dict(m, loss=123.0, accuracy=0.0)
        return m

    monkeypatch.setattr(client, "evaluate_loader", poisoned)
    loss2, n2, ev2 = client.evaluate(params, {"server_round": 1})
    assert (loss2, n2) == (pytest.approx(loss), n)
    assert ev2["val_loss"] == pytest.approx(ev["val_loss"])
    assert ev2["test_loss"] == 123.0

    # fit() never touches the test split
    class Boom:
        def __iter__(self):
            raise AssertionError("fit() read the test split")

        def __len__(self):
            return 0

    monkeypatch.setattr(client, "evaluate_loader", original)
    client.test_loader = Boom()
    new_params, n_train, _ = client.fit(params, {"server_round": 1})
    assert n_train == len(client.train_ds)


def test_client_test_pass_leaves_training_unchanged(uneven_node):
    """The logged test pass must not shift the RNG streams used by the next fit()
    (dropout draws from the global torch generator, and so does every DataLoader
    iteration): evaluate-with-test then fit == evaluate-without-test then fit."""
    from src.fl.client import FedRGBDClient

    client = FedRGBDClient(uneven_node, batch_size=4, local_epochs=1, device="cpu", seed=3,
                           pretrained=False, img_size=IMG)
    params = client.get_parameters({})
    cpu_state = torch.get_rng_state()
    shuffle_state = client.train_loader.generator.get_state()

    def evaluate_then_fit():
        torch.set_rng_state(cpu_state)
        client.train_loader.generator.set_state(shuffle_state)
        client.evaluate(params, {"server_round": 1})
        new_params, _, _ = client.fit(params, {"server_round": 2})
        return new_params

    with_test = evaluate_then_fit()
    real_test_loader = client.test_loader
    client.test_loader = []                    # no test pass at all
    without_test = evaluate_then_fit()
    client.test_loader = real_test_loader
    assert all(np.array_equal(a, b) for a, b in zip(with_test, without_test))


def test_client_eval_timers_separate_val_and_test(uneven_node):
    from src.fl.client import FedRGBDClient

    client = FedRGBDClient(uneven_node, batch_size=4, local_epochs=1, device="cpu", seed=1,
                           pretrained=False, img_size=IMG)
    _, _, ev = client.evaluate(client.get_parameters({}), {"server_round": 1})
    for k in ("eval_time_s", "val_eval_time_s", "test_eval_time_s", "eval_wall_s"):
        assert ev[k] > 0, k
    # eval_time_s (counted in the reported round time) = loading + validation, no test pass
    assert ev["val_eval_time_s"] <= ev["eval_time_s"]
    assert ev["eval_time_s"] + ev["test_eval_time_s"] <= ev["eval_wall_s"] + 1e-6
    assert ev["eval_wall_s"] - ev["eval_time_s"] >= ev["test_eval_time_s"] - 1e-6


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
    assert res["results_schema_version"] == 3
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
    assert res["results_schema_version"] == 3
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
