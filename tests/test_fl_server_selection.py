"""Server recorder, schema 3: validation / test namespaces, model selection and
test-free round timing (no network, no Flower run)."""

import json
import os
import sys
import types

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src", "fl"))

import server as fl_server  # noqa: E402  (src/fl/server.py)
from src.evaluation.metrics import compute_metrics, to_flower_metrics  # noqa: E402


def _fit(node, rnd, n):
    return n, {"train_loss": 0.5, "train_time": 10.0, "hostname": f"jetson-{node[-1]}",
               "strategy": "FedAvg", "node_name": node, "data_dir": f"data/processed/iid/{node}",
               "server_round": rnd, "local_epochs": 5, "lr": 0.001, "batch_size": 8,
               "proximal_mu": 0.0, "num_examples": n, "fit_time_s": 10.0, "fit_wall_s": 11.0,
               "payload_bytes_down": 6000, "payload_bytes_up": 6000}


def _eval(rng, node, rnd, n_val, n_test, val_loss, test_loss, eval_wall=10.0, test_time=4.0):
    """What src/fl/client.py sends from evaluate(): val_*, test_*, legacy keys = val."""
    def metrics(n, bias):
        y = rng.randint(0, 2, n)
        logits = rng.randn(n, 2) + np.stack([1 - y, y], 1) * (1.0 + bias)
        return compute_metrics(logits, y)

    val, test = metrics(n_val, 0.5), metrics(n_test, 0.0)
    flat = to_flower_metrics(val)
    flat.update(to_flower_metrics(val, prefix="val_"))
    flat.update(to_flower_metrics(test, prefix="test_"))
    flat.update({"loss": val_loss, "val_loss": val_loss, "test_loss": test_loss,
                 "hostname": f"jetson-{node[-1]}", "node_name": node, "server_round": rnd,
                 "eval_split": "val", "num_examples": n_val,
                 "eval_time_s": eval_wall - test_time,
                 "val_eval_time_s": eval_wall - test_time - 0.1,
                 "test_eval_time_s": test_time, "eval_wall_s": eval_wall,
                 "payload_bytes_down": 1000})
    return n_val, flat, val, test


def _recorder_with_clock():
    clock = {"t": 0.0}
    rec = fl_server.RoundRecorder(start_time=0.0)
    rec.elapsed = lambda: round(clock["t"], 3)
    return rec, clock


def _run_rounds(rec, rng, val_losses, test_losses, clock, test_time=4.0):
    """Drive the recorder through fit+evaluate rounds with a controlled clock."""
    for rnd, (vl, tl) in enumerate(zip(val_losses, test_losses), start=1):
        clock["t"] += 100.0                                   # fit phase
        rec.fit_aggregation([_fit("node_a", rnd, 100), _fit("node_b", rnd, 100)])
        clock["t"] += 10.0                                    # evaluate phase incl. test pass
        rec.evaluate_aggregation([
            _eval(rng, "node_a", rnd, 30, 30, vl, tl, test_time=test_time)[:2],
            _eval(rng, "node_b", rnd, 50, 50, vl, tl, test_time=test_time)[:2],
        ])


def _results(rec, total_time):
    hist = types.SimpleNamespace(losses_distributed=[], metrics_distributed={},
                                 metrics_distributed_fit={})
    args = types.SimpleNamespace(strategy="fedavg", rounds=len(rec.rounds), min_clients=2,
                                 seed=42, address="0.0.0.0:8080", tag=["iid"])
    return fl_server.build_results(args, hist, total_time, rec, payload_bytes=6_000_000)


# --------------------------------------------------------------------------- #
# namespacing
# --------------------------------------------------------------------------- #
def test_aggregate_keeps_val_and_test_apart_and_weights_test_by_test_size():
    rng = np.random.RandomState(5)
    rec = fl_server.RoundRecorder(start_time=0.0)
    na, ma, va, ta = _eval(rng, "node_a", 1, n_val=40, n_test=200, val_loss=0.3, test_loss=0.9)
    nb, mb, vb, tb = _eval(rng, "node_b", 1, n_val=60, n_test=20, val_loss=0.5, test_loss=0.1)
    agg = rec.evaluate_aggregation([(na, ma), (nb, mb)])

    # validation: weighted by validation-set size (= what Flower passes in)
    assert agg["val_loss"] == pytest.approx((40 * 0.3 + 60 * 0.5) / 100)
    assert agg["val_accuracy"] == pytest.approx(agg["accuracy"])      # legacy key = validation
    assert agg["loss"] == pytest.approx(agg["val_loss"])
    # test: weighted by *test*-set size, not by the validation count
    assert agg["test_loss"] == pytest.approx((200 * 0.9 + 20 * 0.1) / 220)
    assert agg["test_accuracy"] == pytest.approx(
        (200 * ta["accuracy"] + 20 * tb["accuracy"]) / 220)
    assert agg["val_n_examples_total"] == 100 and agg["test_n_examples_total"] == 220
    # pooled per namespace, from the summed confusion matrices
    cm_test = np.array(ta["confusion_matrix"]) + np.array(tb["confusion_matrix"])
    assert agg["pooled_test_accuracy"] == pytest.approx(np.trace(cm_test) / cm_test.sum())
    assert agg["test_cm_0_0"] == cm_test[0, 0]
    assert "pooled_val_mcc" in agg and "pooled_accuracy" in agg
    # the timers are aggregated (mean and max)
    for t in ("eval_time_s", "val_eval_time_s", "test_eval_time_s", "eval_wall_s"):
        assert t + "_mean" in agg and t + "_max" in agg
    assert all(isinstance(v, (int, float)) for v in agg.values())

    # per-client rows: both metric sets, nested confusion matrices, no flat cm_ keys
    row = rec.rounds[1]["evaluate"]["clients"]["node_a"]
    assert row["test_confusion_matrix"] == ta["confusion_matrix"]
    assert row["val_confusion_matrix"] == va["confusion_matrix"]
    assert row["confusion_matrix"] == va["confusion_matrix"]
    assert not any(k.startswith(("test_cm_", "val_cm_", "cm_")) or k.endswith("_json")
                   for k in row)
    assert row["test_accuracy"] == pytest.approx(ta["accuracy"])
    for k in ("eval_time_s", "val_eval_time_s", "test_eval_time_s", "eval_wall_s",
              "payload_bytes_down"):
        assert k in row


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def test_test_eval_overhead_is_the_critical_path_share():
    # a: wall 10 of which test 4 -> 6 without test; b: wall 8 of which test 1 -> 7
    # phase with test = max(10, 8) = 10; without = max(6, 7) = 7; overhead = 3
    metrics = [(10, {"eval_wall_s": 10.0, "test_eval_time_s": 4.0}),
               (10, {"eval_wall_s": 8.0, "test_eval_time_s": 1.0})]
    assert fl_server.RoundRecorder.test_eval_overhead(metrics) == pytest.approx(3.0)
    assert fl_server.RoundRecorder.test_eval_overhead([(10, {"accuracy": 0.9})]) == 0.0


def test_reported_round_time_excludes_the_test_pass():
    rng = np.random.RandomState(0)
    rec, clock = _recorder_with_clock()
    _run_rounds(rec, rng, [0.5, 0.4, 0.3], [0.5, 0.4, 0.3], clock, test_time=4.0)
    for rnd in (1, 2, 3):
        timing = rec.rounds[rnd]["timing"]
        assert timing["round_wall_s"] == pytest.approx(110.0)       # raw, incl. test
        assert timing["test_eval_overhead_s"] == pytest.approx(4.0)
        assert timing["round_time_s"] == pytest.approx(106.0)      # reported
        assert timing["fit_phase_s"] == pytest.approx(100.0)
        assert timing["eval_phase_s"] == pytest.approx(10.0)
        assert timing["elapsed_excl_test_s"] == pytest.approx(106.0 * rnd)
        assert rec.rounds[rnd]["evaluate"]["elapsed_excl_test_s"] == pytest.approx(106.0 * rnd)
        assert rec.rounds[rnd]["evaluate"]["elapsed_s"] == pytest.approx(110.0 * rnd)

    res = _results(rec, 335.0)
    assert res["results_schema_version"] == 3
    assert res["total_time_s"] == pytest.approx(335.0)                # raw, key unchanged
    assert res["total_test_eval_overhead_s"] == pytest.approx(12.0)
    assert res["total_time_excl_test_s"] == pytest.approx(323.0)
    assert "test_eval_overhead_s" in res["timing_definition"]


# --------------------------------------------------------------------------- #
# model selection
# --------------------------------------------------------------------------- #
def test_model_selection_uses_validation_only_with_earliest_tie():
    rng = np.random.RandomState(1)
    rec, clock = _recorder_with_clock()
    # validation is lowest at round 2 (tie with round 3 -> earlier); test is best at round 3
    _run_rounds(rec, rng, val_losses=[0.6, 0.2, 0.2], test_losses=[0.9, 0.8, 0.01], clock=clock)
    res = _results(rec, 330.0)
    sel = res["model_selection"]
    assert sel["rule"] == "min_weighted_val_loss_earliest_round"
    assert sel["selected_round"] == 2
    assert sel["selected_val_loss"] == pytest.approx(0.2)
    assert sel["val_loss_by_round"] == {"1": pytest.approx(0.6), "2": pytest.approx(0.2),
                                        "3": pytest.approx(0.2)}
    # the reported test metrics are those of round 2, not of the test-best round 3
    agg2 = res["rounds"][1]["evaluate"]["aggregate"]
    assert sel["selected_round_test"]["accuracy"] == pytest.approx(agg2["test_accuracy"])
    assert sel["selected_round_test"]["loss"] == pytest.approx(0.8)
    assert res["rounds"][1]["weighted_val_loss"] == pytest.approx(0.2)
    json.dumps(res, allow_nan=False)


def test_test_metrics_never_change_the_selected_round():
    """Randomising every test_* value leaves the selection untouched."""
    val_losses = [0.7, 0.35, 0.5, 0.35]
    selected = set()
    for seed in range(5):
        rng = np.random.RandomState(seed)
        rec, clock = _recorder_with_clock()
        test_losses = list(rng.rand(4) * 3)
        _run_rounds(rec, rng, val_losses, test_losses, clock)
        selected.add(fl_server.model_selection_block(rec)["selected_round"])
    assert selected == {2}


def test_round_val_loss_weights_by_validation_size_and_rejects_divergence():
    ok = [(100, {"val_loss": 0.2, "val_n_examples": 100, "test_loss": 9.0, "test_n_examples": 5}),
          (300, {"val_loss": 0.6, "val_n_examples": 300, "test_loss": 0.0,
                 "test_n_examples": 999})]
    assert fl_server.RoundRecorder.round_val_loss(ok) == pytest.approx(0.5)
    # v2 clients (no val_ keys): the legacy loss / num_examples are validation too
    v2 = [(100, {"loss": 0.2}), (300, {"loss": 0.6})]
    assert fl_server.RoundRecorder.round_val_loss(v2) == pytest.approx(0.5)
    diverged = [(100, {"val_loss": 0.2}), (300, {"val_loss": float("nan")})]
    assert fl_server.RoundRecorder.round_val_loss(diverged) is None


def test_a_diverged_round_cannot_be_selected():
    rng = np.random.RandomState(2)
    rec, clock = _recorder_with_clock()
    _run_rounds(rec, rng, [0.6, float("nan"), 0.4], [0.5, 0.5, 0.5], clock)
    sel = fl_server.model_selection_block(rec)
    assert sel["val_loss_by_round"]["2"] is None
    assert sel["selected_round"] == 3
