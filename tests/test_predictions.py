"""Per-image predictions: join key, server stripping, exact reproduction of the logged
metrics, round-time exclusion, the regeneration script, and the analysis that uses them
(pre-registered clean subset, sequence-level cluster bootstrap)."""

import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src", "fl"))

import server as fl_server  # noqa: E402
from src.evaluation.bootstrap import Unit, config_ci  # noqa: E402
from src.evaluation.metrics import METRIC_KEYS  # noqa: E402
from src.evaluation.predictions import (manifest_key, metrics_from_predictions,  # noqa: E402
                                        pack, unpack)

IMG = 16
COUNT_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "specificity",
                 "f1", "macro_f1", "mcc")


@pytest.fixture(scope="module")
def node(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("pred") / "node_a")
    rng = np.random.RandomState(3)
    for split, k in (("train", 6), ("val", 5), ("test", 7)):
        for cls in ("Fire", "No_Fire"):
            d = os.path.join(root, split, cls)
            os.makedirs(d, exist_ok=True)
            for i in range(k):
                arr = rng.randint(0, 255, (IMG, IMG, 3), dtype=np.uint8)
                if cls == "Fire":
                    arr[..., 0] = 255
                Image.fromarray(arr).save(os.path.join(d, "%s_%s_%d.png" % (split, cls, i)))
    return root


def _client(data_dir, **kw):
    from src.fl.client import FedRGBDClient
    torch.set_num_threads(1)
    return FedRGBDClient(data_dir, batch_size=4, local_epochs=1, device="cpu", seed=7,
                         pretrained=False, img_size=IMG, **kw)


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #
def test_saved_predictions_reproduce_the_logged_metrics(node):
    client = _client(node)
    params = client.get_parameters({})
    _, n, ev = client.evaluate(params, {"server_round": 1})
    for split, n_split in (("val", len(client.val_ds)), ("test", len(client.test_ds))):
        d = unpack(ev["pred_%s_npz" % split])
        assert len(d["path"]) == n_split == ev["%s_n_examples" % split]
        got = metrics_from_predictions(d["label"], d["logit_margin"])
        for k in COUNT_METRICS:                       # exact: same argmax, same counts
            assert got[k] == ev["%s_%s" % (split, k)], (split, k)
        assert got["roc_auc"] == pytest.approx(ev[split + "_roc_auc"], abs=1e-9)
        assert got["loss"] == pytest.approx(ev[split + "_loss"], rel=1e-5)
        assert got["confusion_matrix"] == [[ev["%s_cm_%d_%d" % (split, i, j)] for j in (0, 1)]
                                           for i in (0, 1)]
        assert np.allclose(d["p_fire"], 1 / (1 + np.exp(-d["logit_margin"].astype(float))),
                           atol=1e-6)
    assert ev["pred_pack_time_s"] >= 0
    assert ev["eval_wall_s"] >= ev["eval_time_s"] + ev["test_eval_time_s"] + ev["pred_pack_time_s"] - 1e-6


def test_join_key_survives_an_unsorted_listdir(node, monkeypatch):
    """FlameDataset uses os.listdir; a different order must give the same (path -> prediction)."""
    params = _client(node).get_parameters({})
    _, _, ev_a = _client(node).evaluate(params, {"server_round": 1})

    import src.data.dataset as dataset_mod
    real = os.listdir
    monkeypatch.setattr(dataset_mod.os, "listdir", lambda p: sorted(real(p), reverse=True))
    client_b = _client(node)
    _, _, ev_b = client_b.evaluate(params, {"server_round": 1})

    for split in ("val", "test"):
        a, b = unpack(ev_a["pred_%s_npz" % split]), unpack(ev_b["pred_%s_npz" % split])
        assert list(a["path"]) != list(b["path"])                 # order really differs
        ma = {p: (l, m) for p, l, m in zip(a["path"], a["label"], a["logit_margin"])}
        mb = {p: (l, m) for p, l, m in zip(b["path"], b["label"], b["logit_margin"])}
        assert ma.keys() == mb.keys()
        for p in ma:
            assert ma[p][0] == mb[p][0]
            assert ma[p][1] == pytest.approx(mb[p][1], abs=1e-5)
            # the key is <Class>/<file>, and the label agrees with the class directory
            assert p.split("/")[0] == ("Fire" if ma[p][0] == 1 else "No_Fire")
    assert manifest_key(os.path.join("x", "val", "Fire", "a.jpg")) == "Fire/a.jpg"


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #
def _eval_metrics(node_name, rnd, n=20, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    logits = rng.normal(size=(n, 2)).astype(np.float32)
    paths = ["%s/%s_%d.jpg" % ("Fire" if t else "No_Fire", node_name, i) for i, t in enumerate(y)]
    blob = pack(paths, y, logits)
    return n, {"accuracy": 0.5, "loss": 0.7, "val_loss": 0.7, "val_n_examples": n,
               "node_name": node_name, "server_round": rnd, "num_examples": n,
               "eval_wall_s": 10.0, "test_eval_time_s": 3.0, "pred_pack_time_s": 1.0,
               "pred_val_npz": blob, "pred_test_npz": blob}


def test_server_strips_byte_entries_and_writes_files(tmp_path):
    rec = fl_server.RoundRecorder(start_time=0.0)
    rec.pred_dir = str(tmp_path / "predictions")
    rec.evaluate_aggregation([_eval_metrics("node_a", 1), _eval_metrics("node_b", 1, seed=1)])
    hist = types.SimpleNamespace(losses_distributed=[], metrics_distributed={},
                                 metrics_distributed_fit={})
    args = types.SimpleNamespace(strategy="fedavg", rounds=1, min_clients=2, seed=42,
                                 address="0.0.0.0:8080", tag=["iid"])
    res = fl_server.build_results(args, hist, 30.0, rec, payload_bytes=6_000_000)
    json.dumps(res, allow_nan=False)                        # would raise on a bytes value

    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from keys(v)

    all_keys = set(keys(res))
    assert not any(str(k).endswith("_npz") for k in all_keys)
    # the packing timer is kept (and aggregated) -- only the payloads are stripped
    ev = res["rounds"][0]["evaluate"]
    assert ev["clients"]["node_a"]["pred_pack_time_s"] == 1.0
    assert "pred_pack_time_s_max" in ev["aggregate"]
    files = sorted(os.listdir(rec.pred_dir))
    assert files == ["README.md", "r001_node_a_test.npz", "r001_node_a_val.npz",
                     "r001_node_b_test.npz", "r001_node_b_val.npz"]
    assert res["rounds"][0]["predictions"]["node_a"]["test"]["file"] == "r001_node_a_test.npz"
    d = unpack(open(os.path.join(rec.pred_dir, "r001_node_b_val.npz"), "rb").read())
    assert len(d["path"]) == 20


def test_prediction_packing_and_writing_stay_out_of_the_round_time(tmp_path):
    clock = {"t": 0.0}
    rec = fl_server.RoundRecorder(start_time=0.0)
    rec.elapsed = lambda: round(clock["t"], 3)
    rec.pred_dir = str(tmp_path / "p")
    real_write = rec.write_predictions

    def slow_write(rnd, preds):                     # writing takes 5 s of server time
        clock["t"] += 5.0
        return real_write(rnd, preds)

    rec.write_predictions = slow_write
    for rnd in (1, 2):
        clock["t"] += 100.0
        rec.evaluate_aggregation([_eval_metrics("node_a", rnd), _eval_metrics("node_b", rnd, seed=1)])
    t1, t2 = rec.rounds[1]["timing"], rec.rounds[2]["timing"]
    # client side: test pass (3 s) + packing (1 s) removed from the critical path
    assert t1["test_eval_overhead_s"] == pytest.approx(4.0)
    assert t1["round_time_s"] == pytest.approx(96.0)
    # server side: the 5 s write after round 1 is not in round 2's wall-clock
    assert t2["round_wall_s"] == pytest.approx(100.0)
    assert t2["round_time_s"] == pytest.approx(96.0)


# --------------------------------------------------------------------------- #
# regeneration from a baseline checkpoint
# --------------------------------------------------------------------------- #
def test_predict_from_checkpoint_reproduces_the_selected_epoch(node, tmp_path):
    from scripts import predict_from_checkpoint, train_centralized

    run = tmp_path / "rev_iid_centralized_seed42"
    train_centralized.main(["--data_dirs", node, "--epochs", "2", "--batch_size", "4",
                            "--seed", "42", "--no_pretrained", "--img_size", str(IMG),
                            "--output_dir", str(run)])
    assert predict_from_checkpoint.main([str(run), "--img_size", str(IMG)]) == 0
    files = sorted(os.listdir(run / "predictions"))
    assert files == ["README.md", "selected_node_a_test.npz", "selected_node_a_val.npz"]
    res = json.loads((run / "results.json").read_text())
    d = unpack((run / "predictions" / "selected_node_a_test.npz").read_bytes())
    got = metrics_from_predictions(d["label"], d["logit_margin"])
    for k in ("accuracy", "balanced_accuracy", "mcc"):
        assert got[k] == pytest.approx(res["selected_test_metrics"][k], abs=1e-6)


# --------------------------------------------------------------------------- #
# analysis: clean subset and cluster bootstrap
# --------------------------------------------------------------------------- #
def test_cluster_bootstrap_widens_with_sequence_dominance():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    # correct on 60% of images, but correctness is decided per block of 400 images
    block = np.repeat(np.arange(5), 400)
    right = np.isin(block, [0, 1, 2])
    margin = np.where(right == (y == 1), 3.0, -3.0)
    few = config_ci([[Unit(y, margin, block)]], "pooled", "few", B=400)
    many = config_ci([[Unit(y, margin, np.arange(n))]], "pooled", "many", B=400)
    w_few = few["accuracy"]["ci_high"] - few["accuracy"]["ci_low"]
    w_many = many["accuracy"]["ci_high"] - many["accuracy"]["ci_low"]
    assert w_few > 5 * w_many                          # 5 clusters vs 2000 independent images
    again = config_ci([[Unit(y, margin, block)]], "pooled", "few", B=400)
    assert again == few                                # reproducible (seeded by config id)


def test_analysis_clean_subset_and_bootstrap_on_real_partition_paths(tmp_path, monkeypatch):
    """A synthetic FL run whose predictions use real node_c test paths of the iid partition."""
    from scripts import analyze_results as ar

    m = pd.read_csv(os.path.join(_REPO, "data", "splits", "iid.csv.gz"), dtype={"group_id": str})
    ex = set(pd.read_csv(os.path.join(_REPO, "analysis", "leakage", "clean_subset",
                                      "iid_excluded.csv.gz")).path)
    runs = []
    for seed in (42, 123):
        run_dir = tmp_path / ("rev_iid_fedavg_seed%d" % seed)
        (run_dir / "predictions").mkdir(parents=True)
        rng = np.random.default_rng(seed)
        per_client = {}
        for node in ("node_a", "node_b", "node_c"):
            t = m[(m.node == node) & (m.split == "test")]
            y = (t.label == "Fire").to_numpy(int)
            logits = np.stack([np.zeros(len(t)), rng.normal(1.5, 2, len(t)) * (2 * y - 1)], 1)
            (run_dir / "predictions" / ("r002_%s_test.npz" % node)).write_bytes(
                pack(list(t.path), y, logits))
            per_client[node] = (metrics_from_predictions(y, logits[:, 1] - logits[:, 0]), len(t))
        tot = sum(n for _, n in per_client.values())
        logged = {k: sum(mm[k] * n for mm, n in per_client.values()) / tot
                  for k in ("accuracy", "balanced_accuracy", "mcc")}
        runs.append({"run_dir": str(run_dir), "run_name": run_dir.name, "kind": "fl",
                     "headline_source": ar.HEADLINE_SELECTED, "selected_round": 2,
                     "distribution": "iid", "selected_test_metrics": logged,
                     "config_id": "cfg", "label": "FedAvg iid [3N] {group}", "protocol": "group",
                     "strategy": "fedavg", "seed_label": str(seed), "n_nodes": 3})
    ar.attach_prediction_metrics(runs, warn=False)
    n_test = int(((m.split == "test")).sum())
    n_ex = len(ex & set(m[m.split == "test"].path))
    for r in runs:
        assert r["pred_check_max_diff"] < 1e-12            # full set reproduces the logged values
        assert r["clean_excluded_n"] == n_ex and 0 < n_ex < n_test
        assert r["selected_test_clean_metrics"]["accuracy"] != r["selected_test_metrics"]["accuracy"]
    summary = ar.summary_table(runs, bootstrap_B=50)
    row = summary[summary.metric == "selected_test_accuracy"].iloc[0]
    assert row.ci_method == "cluster_bootstrap_B50" and row.ci_low < row["mean"] < row.ci_high
    clean = summary[summary.metric == "selected_test_clean_accuracy"].iloc[0]
    assert clean.ci_method == "cluster_bootstrap_B50"
    other = summary[summary.metric == "selected_round"].iloc[0]
    assert other.ci_method == "t_seeds"
