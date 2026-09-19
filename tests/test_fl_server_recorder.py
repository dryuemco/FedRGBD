"""Tests for the server-side per-round metric recorder (no network, no Flower run)."""

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


def _client_eval_metrics(rng, n, node, rnd, bias=0.0):
    y = rng.randint(0, 2, n)
    logits = rng.randn(n, 2) + np.stack([1 - y, y], 1) * (1.0 + bias)
    m = compute_metrics(logits, y)
    flat = to_flower_metrics(m)
    flat["loss"] = float(rng.rand())
    flat.update({"hostname": f"jetson-{node[-1]}", "node_name": node, "server_round": rnd,
                 "num_examples": n, "eval_time_s": 1.0 + rng.rand(), "payload_bytes_down": 1000})
    return n, flat, m


def _client_fit_metrics(node, rnd, n):
    return n, {"train_loss": 0.5, "train_time": 10.0, "hostname": f"jetson-{node[-1]}", "strategy": "FedAvg",
               "node_name": node, "data_dir": f"data/processed/iid/{node}", "server_round": rnd,
               "local_epochs": 5, "lr": 0.001, "batch_size": 8, "proximal_mu": 0.0,
               "num_examples": n, "fit_time_s": 10.0, "fit_wall_s": 11.0,
               "payload_bytes_down": 6000, "payload_bytes_up": 6000}


def test_evaluate_aggregation_weighted_and_pooled():
    rng = np.random.RandomState(0)
    rec = fl_server.RoundRecorder(start_time=0.0)
    na, ma, raw_a = _client_eval_metrics(rng, 40, "node_a", 1)
    nb, mb, raw_b = _client_eval_metrics(rng, 60, "node_b", 1, bias=1.0)

    agg = rec.evaluate_aggregation([(na, ma), (nb, mb)])

    # weighted-global accuracy (v2 semantics)
    expected = (na * ma["accuracy"] + nb * mb["accuracy"]) / (na + nb)
    assert agg["accuracy"] == pytest.approx(expected)
    assert agg["accuracy"] == pytest.approx(fl_server.weighted_average([(na, ma), (nb, mb)])["accuracy"])
    for key in ("balanced_accuracy", "precision", "recall", "specificity", "f1", "macro_f1", "mcc", "roc_auc", "loss"):
        assert key in agg and isinstance(agg[key], float)

    # pooled metrics from the summed confusion matrix == metrics of the union
    cm_sum = np.array(raw_a["confusion_matrix"]) + np.array(raw_b["confusion_matrix"])
    assert agg["cm_0_0"] == cm_sum[0, 0] and agg["cm_1_1"] == cm_sum[1, 1]
    assert agg["pooled_accuracy"] == pytest.approx(np.trace(cm_sum) / cm_sum.sum())
    assert agg["pooled_accuracy"] == pytest.approx(agg["accuracy"])  # identical for accuracy
    assert "pooled_mcc" in agg and "pooled_f1" in agg

    # timing: mean & max, payload: totals, examples: total
    assert agg["eval_time_s_mean"] == pytest.approx((ma["eval_time_s"] + mb["eval_time_s"]) / 2)
    assert agg["eval_time_s_max"] == pytest.approx(max(ma["eval_time_s"], mb["eval_time_s"]))
    assert agg["payload_bytes_down_total"] == 2000
    assert agg["num_examples_total"] == 100
    # identity keys are not aggregated
    assert "server_round" not in agg and "hostname" not in agg

    # per-client rows are keyed by node name with a nested confusion matrix
    rows = rec.rounds[1]["evaluate"]["clients"]
    assert set(rows) == {"node_a", "node_b"}
    assert rows["node_a"]["confusion_matrix"] == raw_a["confusion_matrix"]
    assert rows["node_a"]["num_examples"] == na
    assert "cm_0_0" not in rows["node_a"] and "confusion_matrix_json" not in rows["node_a"]
    assert rec.rounds[1]["cumulative_communication_bytes"] == 2000

    # every aggregated value must be a Flower scalar
    assert all(isinstance(v, (int, float)) for v in agg.values())


def test_fit_aggregation_tracks_bytes_config_and_rounds():
    rec = fl_server.RoundRecorder(start_time=0.0)
    for rnd in (1, 2):
        agg = rec.fit_aggregation([_client_fit_metrics("node_a", rnd, 100), _client_fit_metrics("node_b", rnd, 300)])
        assert agg["train_loss"] == pytest.approx(0.5)
        assert agg["fit_time_s_max"] == 10.0 and agg["train_time_mean"] == 10.0
        assert agg["payload_bytes_up_total"] == 12000 and agg["payload_bytes_down_total"] == 12000
    assert sorted(rec.rounds) == [1, 2]
    assert rec.rounds[2]["cumulative_communication_bytes"] == 2 * 24000
    assert rec.client_config["node_a"]["local_epochs"] == 5
    assert rec.client_config["node_b"]["data_dir"].endswith("node_b")
    assert rec.rounds[1]["fit"]["elapsed_s"] >= 0


def test_round_fallback_for_old_clients_and_missing_metrics():
    """Clients without server_round/node_name (v2) still get recorded per call."""
    rec = fl_server.RoundRecorder(start_time=0.0)
    old = [(10, {"accuracy": 0.9, "hostname": "h1"}), (30, {"accuracy": 0.7, "hostname": "h2"})]
    agg1 = rec.evaluate_aggregation(old)
    agg2 = rec.evaluate_aggregation(old)
    assert agg1["accuracy"] == pytest.approx((10 * 0.9 + 30 * 0.7) / 40)
    assert sorted(rec.rounds) == [1, 2]
    assert set(rec.rounds[1]["evaluate"]["clients"]) == {"h1", "h2"}
    assert "pooled_accuracy" not in agg2  # no confusion matrices available


def test_build_results_keeps_v2_keys_and_is_json_serialisable(tmp_path):
    rng = np.random.RandomState(1)
    rec = fl_server.RoundRecorder(start_time=0.0)
    hist = types.SimpleNamespace(losses_distributed=[], metrics_distributed={}, metrics_distributed_fit={})
    for rnd in (1, 2, 3):
        fit_agg = rec.fit_aggregation([_client_fit_metrics("node_a", rnd, 100), _client_fit_metrics("node_b", rnd, 100)])
        na, ma, _ = _client_eval_metrics(rng, 30, "node_a", rnd)
        nb, mb, _ = _client_eval_metrics(rng, 50, "node_b", rnd)
        ev_agg = rec.evaluate_aggregation([(na, ma), (nb, mb)])
        hist.losses_distributed.append((rnd, (na * ma["loss"] + nb * mb["loss"]) / (na + nb)))
        for k, v in ev_agg.items():
            hist.metrics_distributed.setdefault(k, []).append((rnd, v))
        for k, v in fit_agg.items():
            hist.metrics_distributed_fit.setdefault(k, []).append((rnd, v))

    args = types.SimpleNamespace(strategy="fedprox_0.01", rounds=3, min_clients=2, seed=42,
                                 address="0.0.0.0:8080", tag=["dirichlet_0.1"])
    res = fl_server.build_results(args, hist, 123.4, rec, payload_bytes=6_000_000)

    # v2 keys, unchanged structure
    for k in ("strategy", "num_rounds", "min_clients", "seed", "total_time_s", "timestamp",
              "losses_distributed", "metrics_distributed"):
        assert k in res
    assert res["metrics_distributed"]["accuracy"][0] == {"round": 1, "value": pytest.approx(res["rounds"][0]["evaluate"]["aggregate"]["accuracy"])}
    assert [d["round"] for d in res["losses_distributed"]] == [1, 2, 3]
    # v3 keys
    assert res["results_schema_version"] == 3
    assert res["proximal_mu"] == 0.01
    assert res["tags"] == ["dirichlet_0.1"]
    assert res["model_payload_bytes"] == 6_000_000
    assert len(res["rounds"]) == 3 and res["rounds"][2]["round"] == 3
    assert res["rounds"][2]["cumulative_communication_bytes"] == res["total_communication_bytes"]
    assert "balanced_accuracy" in res["rounds"][0]["evaluate"]["clients"]["node_a"]
    assert res["client_config"]["node_a"]["lr"] == 0.001
    assert "train_loss" in res["metrics_distributed_fit"]

    out = tmp_path / "results.json"
    out.write_text(json.dumps(res, indent=2))
    assert "NaN" not in out.read_text()
    assert json.loads(out.read_text())["rounds"][0]["evaluate"]["clients"]["node_b"]["confusion_matrix"]


def test_strategy_factory_wires_recorder_and_configs():
    rec = fl_server.RoundRecorder()
    for name, cls_name, mu in (("fedavg", "FedAvg", 0.0), ("fedprox_0.05", "FedProx", 0.05), ("fedbn", "FedBN", 0.0)):
        strat = fl_server.get_strategy(name, min_clients=2, recorder=rec)
        assert type(strat).__name__ == cls_name
        assert strat.evaluate_metrics_aggregation_fn == rec.evaluate_aggregation
        assert strat.fit_metrics_aggregation_fn == rec.fit_aggregation
        assert strat.on_fit_config_fn(4)["server_round"] == 4
        assert strat.on_evaluate_config_fn(4) == {"server_round": 4}
        assert fl_server.parse_mu(name) == mu
        if name == "fedbn":
            assert strat.on_fit_config_fn(1)["fedbn"] is True
        if name.startswith("fedprox"):
            assert strat.proximal_mu == mu


def test_model_payload_bytes_matches_state_dict():
    n = fl_server.model_payload_bytes()
    assert n is not None and n > 1_000_000  # MobileNetV3-Small ≈ 6 MB


def test_results_json_is_strict_json_when_a_client_diverges():
    """A NaN/inf client metric must not make results.json unreadable.

    ``json.dump`` writes the bare tokens ``NaN`` / ``Infinity``, which are not
    valid JSON (RFC 8259) and are rejected by every non-Python reader.  A
    diverging local round (large ``proximal_mu``, high lr) produces exactly
    that through ``train_loss``, and it would only surface after 1.5-3 h of
    testbed time, when the run's results are written.
    """
    rng = np.random.RandomState(3)
    rec = fl_server.RoundRecorder(start_time=0.0)

    n_a, fit_a = _client_fit_metrics("node_a", 1, 100)
    n_b, fit_b = _client_fit_metrics("node_b", 1, 100)
    fit_b["train_loss"] = float("nan")          # diverged
    fit_b["fit_time_s"] = float("inf")          # pathological timer
    rec.fit_aggregation([(n_a, fit_a), (n_b, fit_b)])

    na, ma, _ = _client_eval_metrics(rng, 20, "node_a", 1)
    nb, mb, _ = _client_eval_metrics(rng, 20, "node_b", 1)
    mb["loss"] = float("nan")
    rec.evaluate_aggregation([(na, ma), (nb, mb)])

    history = types.SimpleNamespace(
        losses_distributed=[(1, float("nan"))],
        metrics_distributed={"accuracy": [(1, 0.5)], "loss": [(1, float("nan"))]},
        metrics_distributed_fit={"train_loss": [(1, float("nan"))]},
    )
    args = types.SimpleNamespace(strategy="fedprox_0.5", rounds=3, min_clients=3, seed=42,
                                 tag=["noniid"], address="0.0.0.0:8080")
    res = fl_server.build_results(args, history, 42.0, rec, 6_000_000)

    # strict JSON: no NaN / Infinity tokens anywhere
    text = json.dumps(res, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["rounds"][0]["fit"]["clients"]["node_b"]["train_loss"] is None
    assert res["losses_distributed"][0]["loss"] is None
    # finite values are untouched
    assert res["rounds"][0]["fit"]["clients"]["node_a"]["train_loss"] == 0.5


def test_json_safe_leaves_finite_values_and_converts_numpy():
    out = fl_server.json_safe({
        "i": np.int64(7), "f": np.float32(0.5), "arr": np.arange(3),
        "nan": float("nan"), "inf": float("-inf"), "ok": 1.25,
        "nested": [{"x": np.float64(2.0)}], "s": "text", "b": True, "none": None,
    })
    assert out["i"] == 7 and isinstance(out["i"], int)
    assert out["f"] == pytest.approx(0.5) and isinstance(out["f"], float)
    assert out["arr"] == [0, 1, 2]
    assert out["nan"] is None and out["inf"] is None
    assert out["ok"] == 1.25 and out["nested"] == [{"x": 2.0}]
    assert out["s"] == "text" and out["b"] is True and out["none"] is None
    json.dumps(out, allow_nan=False)
