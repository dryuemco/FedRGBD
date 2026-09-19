"""Centralized / local-only baselines under the declared selection rule (schema 3).

Same rule as the FL runs: validation and test are evaluated every epoch; the
reported model is the one from the epoch with the lowest validation loss
(earliest on ties); test metrics are logged every epoch, never used for
selection, and the reported ones are those of the selected epoch.  Old
(final-epoch) group-level baselines must never share a protocol or a column
with the new ones.
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

from scripts.analyze_results import (  # noqa: E402
    HEADLINE_FINAL_EPOCH,
    HEADLINE_SELECTED,
    PROTOCOL_GROUP,
    PROTOCOL_GROUP_FINAL_EPOCH,
    collect_runs,
    load_run,
    pairwise_tests,
    summarize,
)
from src.evaluation.metrics import METRIC_KEYS  # noqa: E402

IMG = 16


def _make_node(root, seed):
    rng = np.random.RandomState(seed)
    for split, per_class in (("train", 6), ("val", 3), ("test", 4)):
        for cls in ("Fire", "No_Fire"):
            d = os.path.join(root, split, cls)
            os.makedirs(d, exist_ok=True)
            for i in range(per_class):
                arr = rng.randint(0, 255, (IMG, IMG, 3), dtype=np.uint8)
                if cls == "Fire":
                    arr[..., 0] = 255
                Image.fromarray(arr).save(os.path.join(d, f"{split}_{cls}_{i}.png"))
    return root


@pytest.fixture(scope="module")
def nodes(tmp_path_factory):
    base = tmp_path_factory.mktemp("sel_processed")
    return [_make_node(str(base / f"node_{c}"), seed=i) for i, c in enumerate("ab")]


COMMON = ["--epochs", "3", "--batch_size", "4", "--seed", "42", "--no_pretrained",
          "--img_size", str(IMG)]


# --------------------------------------------------------------------------- #
# the training scripts
# --------------------------------------------------------------------------- #
def test_centralized_logs_val_and_test_every_epoch_and_selects_on_val(nodes, tmp_path):
    from scripts import train_centralized

    torch.set_num_threads(1)
    out = tmp_path / "rev_iid_centralized_seed42"
    train_centralized.main(["--data_dirs", *nodes, "--output_dir", str(out)] + COMMON)
    res = json.loads((out / "results.json").read_text())

    assert res["results_schema_version"] == 3
    for h in res["history"]:
        for k in METRIC_KEYS:
            assert k in h["val_metrics"] and k in h["test_metrics"], k
        assert set(h["test_per_node"]) == {"node_a", "node_b"}
        for k in ("train_time_s", "val_eval_time_s", "test_eval_time_s", "epoch_wall_s",
                  "elapsed_excl_test_s"):
            assert h[k] > 0, k
        # reported epoch time = train + validation, the test pass is excluded
        assert h["epoch_time_s"] <= h["epoch_wall_s"] - h["test_eval_time_s"] + 0.1
        # pooled test = union of the per-node tests (one pass)
        assert h["test_metrics"]["n_examples"] == sum(
            v["test_metrics"]["n_examples"] for v in h["test_per_node"].values())

    sel = res["model_selection"]
    losses = {int(e): v for e, v in sel["val_loss_by_epoch"].items()}
    expected = min(losses, key=lambda e: (losses[e], e))
    assert sel["selected_epoch"] == res["selected_epoch"] == expected
    chosen = res["history"][expected - 1]
    assert res["selected_test_metrics"] == chosen["test_metrics"]
    assert res["selected_per_node_test"] == chosen["test_per_node"]
    # old keys keep their meaning: last epoch
    assert res["final_test_metrics"] == res["history"][-1]["test_metrics"]
    assert res["total_time_excl_test_s"] < res["total_time_s"]
    assert (out / "model_selected.pt").is_file() and (out / "model_final.pt").is_file()


def test_selected_checkpoint_is_the_selected_epochs_model(nodes, tmp_path):
    """Re-evaluating model_selected.pt reproduces the selected epoch's validation loss."""
    from scripts import train_centralized
    from src.data.dataset import FlameDataset
    from src.models.mobilenetv3_multimodal import create_model
    from torch.utils.data import ConcatDataset, DataLoader

    out = tmp_path / "ckpt"
    train_centralized.main(["--data_dirs", *nodes, "--output_dir", str(out)] + COMMON)
    res = json.loads((out / "results.json").read_text())
    model = create_model(num_classes=2, in_channels=3, pretrained=False)
    model.load_state_dict(torch.load(out / "model_selected.pt", map_location="cpu", weights_only=True))
    val = ConcatDataset([FlameDataset(n, split="val", img_size=IMG) for n in nodes])
    loss, _, _ = train_centralized.evaluate(model, DataLoader(val, batch_size=4), torch.nn.CrossEntropyLoss(),
                                            torch.device("cpu"))
    assert loss == pytest.approx(res["model_selection"]["selected_val_loss"], rel=1e-4, abs=1e-5)


def test_local_selects_per_node_and_cross_evaluates_the_selected_model(nodes, tmp_path):
    from scripts import train_local

    out = tmp_path / "rev_iid_local_seed42"
    train_local.main(["--batch", "--cross_eval", "--data_dirs", *nodes, "--output_dir", str(out)]
                     + COMMON)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["results_schema_version"] == 3
    for node in ("node_a", "node_b"):
        res = json.loads((out / node / "results.json").read_text())
        losses = {int(e): v for e, v in res["model_selection"]["val_loss_by_epoch"].items()}
        expected = min(losses, key=lambda e: (losses[e], e))
        assert res["selected_epoch"] == expected
        assert res["selected_test_metrics"] == res["history"][expected - 1]["test_metrics"]
        assert res["cross_eval_model"] == "selected_epoch"
        assert summary["nodes"][node]["selected_epoch"] == expected
        for h in res["history"]:
            assert "test_metrics" in h and h["test_eval_time_s"] > 0
    assert summary["mean_selected_test_accuracy"] == pytest.approx(np.mean(
        [summary["nodes"][n]["selected_test_accuracy"] for n in ("node_a", "node_b")]), abs=1e-6)


def test_test_pass_does_not_change_training(nodes, tmp_path, monkeypatch):
    """Evaluating test every epoch must not alter the trajectory: val losses are
    identical with the test pass replaced by a no-op."""
    from scripts import train_centralized

    out_a = tmp_path / "with_test"
    train_centralized.main(["--data_dirs", *nodes, "--output_dir", str(out_a)] + COMMON)

    real = train_centralized.evaluate_by_node

    def fake(model, loaders, device):
        loss, acc, m, per_node = real(model, {k: [] for k in loaders}, device)
        return loss, acc, m, per_node

    monkeypatch.setattr(train_centralized, "evaluate_by_node", fake)
    out_b = tmp_path / "without_test"
    train_centralized.main(["--data_dirs", *nodes, "--output_dir", str(out_b)] + COMMON)
    a = json.loads((out_a / "results.json").read_text())["model_selection"]["val_loss_by_epoch"]
    b = json.loads((out_b / "results.json").read_text())["model_selection"]["val_loss_by_epoch"]
    assert a == pytest.approx(b)


# --------------------------------------------------------------------------- #
# the analysis: new vs. old baselines
# --------------------------------------------------------------------------- #
def _poison_test(path):
    """Overwrite every logged test metric; the selected epoch must not move."""
    data = json.loads(open(path).read())
    rng = np.random.RandomState(0)
    for h in data["history"]:
        for key in list(h["test_metrics"]):
            if isinstance(h["test_metrics"][key], float):
                h["test_metrics"][key] = float(rng.rand())
    open(path, "w").write(json.dumps(data))
    return data


def test_analysis_reads_new_baselines_under_the_rule(nodes, tmp_path):
    from scripts import train_centralized, train_local

    root = tmp_path / "results"
    new_root = root / "rev_baselines_sel"
    train_centralized.main(["--data_dirs", *nodes, "--output_dir",
                            str(new_root / "rev_iid_centralized_seed42")] + COMMON)
    train_local.main(["--batch", "--data_dirs", *nodes, "--output_dir",
                      str(new_root / "rev_iid_local_seed42")] + COMMON)

    central = load_run(str(new_root / "rev_iid_centralized_seed42"), warn=False)
    assert central["protocol"] == PROTOCOL_GROUP
    assert central["headline_source"] == HEADLINE_SELECTED
    raw = json.loads((new_root / "rev_iid_centralized_seed42" / "results.json").read_text())
    assert central["selected_epoch"] == raw["selected_epoch"]
    assert central["selected_test_metrics"]["accuracy"] == pytest.approx(
        raw["selected_test_metrics"]["accuracy"])
    assert set(central["selected_test_clients"]) == {"node_a", "node_b"}

    local = load_run(str(new_root / "rev_iid_local_seed42"), warn=False)
    assert local["headline_source"] == HEADLINE_SELECTED
    node_accs = [json.loads((new_root / "rev_iid_local_seed42" / n / "results.json").read_text())
                 ["selected_test_accuracy"] for n in ("node_a", "node_b")]
    assert local["selected_test_metrics"]["accuracy"] == pytest.approx(np.mean(node_accs))
    assert local["selected_test_clients"]["node_a"]["selected_epoch"] >= 1

    # selection never depends on the logged test metrics
    before = central["selected_epoch"]
    _poison_test(new_root / "rev_iid_centralized_seed42" / "results.json")
    assert load_run(str(new_root / "rev_iid_centralized_seed42"), warn=False)["selected_epoch"] == before


def test_old_and_new_baselines_are_different_protocols_and_columns(nodes, tmp_path):
    from scripts import train_centralized, train_local

    root = tmp_path / "results"
    for seed in ("42", "123"):
        args = ["--epochs", "2", "--batch_size", "4", "--seed", seed, "--no_pretrained",
                "--img_size", str(IMG)]
        train_centralized.main(["--data_dirs", *nodes, "--output_dir",
                                str(root / "rev_baselines_sel" / f"rev_iid_centralized_seed{seed}")] + args)
        train_local.main(["--batch", "--data_dirs", *nodes, "--output_dir",
                          str(root / "rev_baselines_sel" / f"rev_iid_local_seed{seed}")] + args)
        # an "old" (schema 2, final-epoch) group-level baseline of the same config and seed
        for kind in ("centralized", "local"):
            src_dir = root / "rev_baselines_sel" / f"rev_iid_{kind}_seed{seed}"
            dst_dir = root / f"rev_iid_{kind}_seed{seed}"
            for path in src_dir.rglob("*.json"):
                data = json.loads(path.read_text())
                for key in ("model_selection", "selected_epoch", "selected_test_metrics",
                            "selected_test_accuracy", "mean_selected_test_accuracy"):
                    data.pop(key, None)
                data["results_schema_version"] = 2
                target = dst_dir / path.relative_to(src_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(data))

    runs = collect_runs(str(root), warn=False)
    assert len(runs) == 8
    by_protocol = {}
    for r in runs:
        by_protocol.setdefault(r["protocol"], set()).add(r["headline_source"])
    assert by_protocol == {PROTOCOL_GROUP: {HEADLINE_SELECTED},
                           PROTOCOL_GROUP_FINAL_EPOCH: {HEADLINE_FINAL_EPOCH}}

    tables = summarize(runs)
    summary = tables["summary"]
    new = summary[summary["protocol"] == PROTOCOL_GROUP]
    old = summary[summary["protocol"] == PROTOCOL_GROUP_FINAL_EPOCH]
    assert set(new["config_id"]).isdisjoint(set(old["config_id"]))
    assert "selected_test_accuracy" in set(new["metric"])
    assert not any(m.startswith("final_") and m != "final_cumulative_mb" for m in new["metric"])
    assert "final_accuracy" in set(old["metric"])
    assert not any(m.startswith("selected_") for m in old["metric"])
    assert all("{group_final_epoch}" in label for label in old["label"])

    pairs = pairwise_tests(runs, metric="accuracy")
    assert set(pairs["protocol"]) == {PROTOCOL_GROUP, PROTOCOL_GROUP_FINAL_EPOCH}
    assert (pairs.groupby("protocol").size() == 1).all()   # Centralized vs Local-only, each
