"""Aggregation must not depend on the order in which clients report.

Flower hands the strategies their results in arrival order, and both of its
aggregation helpers sum in list order, so on a real testbed the global model
depended on which Jetson finished first. This reproduced as two runs of the same
commit at the same seed diverging at ~1e-7 in round 1 and 6 % by round 2.

Every test here feeds the *same* client results in every permutation and demands
bitwise-identical output. `test_control_*` deliberately aggregates unsorted and
asserts the order does change the result, so that these tests cannot pass simply
because the inputs are too benign to expose the bug.
"""

import itertools
import os
import sys

import numpy as np
import pytest
from flwr.common import (Code, EvaluateRes, FitRes, Status, ndarrays_to_parameters,
                         parameters_to_ndarrays)
from flwr.server.strategy.aggregate import aggregate

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src", "fl"))

from src.fl.aggregation_order import (DeterministicClientOrder,  # noqa: E402
                                      sorted_metrics, sorted_results)

NODES = ("node_a", "node_b", "node_c")


class _Proxy:
    """Stands in for a Flower ClientProxy; cid deliberately unrelated to node name."""

    def __init__(self, cid):
        self.cid = cid


def _payload(seed, n_layers=3):
    """Values whose float sum is genuinely order-dependent."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(97).astype(np.float32) * 10.0 ** rng.integers(-6, 6)
            for _ in range(n_layers)]


def _fit_results(num_examples=(911, 41786, 2400)):
    out = []
    for i, node in enumerate(NODES):
        res = FitRes(status=Status(Code.OK, ""),
                     parameters=ndarrays_to_parameters(_payload(i)),
                     num_examples=num_examples[i],
                     metrics={"node_name": node, "server_round": 1,
                              "train_loss": 0.11324341938775163 + i})
        out.append((_Proxy("cid-%d" % (7 - i)), res))
    return out


def _evaluate_results(num_examples=(911, 41786, 2400)):
    out = []
    for i, node in enumerate(NODES):
        res = EvaluateRes(status=Status(Code.OK, ""),
                          loss=[4.48001877359, 1e-7, 3.7300890427681955][i],
                          num_examples=num_examples[i],
                          metrics={"node_name": node, "server_round": 1,
                                   "val_loss": [4.48001877359, 1e-7, 3.73008904][i],
                                   "val_n_examples": num_examples[i]})
        out.append((_Proxy("cid-%d" % (7 - i)), res))
    return out


def _permutations(results):
    return [list(p) for p in itertools.permutations(results)]


# --------------------------------------------------------------------------- #
# the control: without sorting, order really does change the answer
# --------------------------------------------------------------------------- #
def test_control_unsorted_aggregation_is_order_dependent():
    """If this ever fails, the fixtures stopped exercising the bug."""
    seen = set()
    for perm in _permutations(_fit_results()):
        weights = [(parameters_to_ndarrays(r.parameters), r.num_examples) for _, r in perm]
        out = aggregate(weights)
        seen.add(tuple(float(layer.sum()) for layer in out))
    assert len(seen) > 1, ("arrival order no longer changes Flower's aggregate(); "
                           "the ordering tests below would pass vacuously")


# --------------------------------------------------------------------------- #
# the fix
# --------------------------------------------------------------------------- #
def test_sorted_results_orders_by_node_name_not_cid():
    ordered = sorted_results(_fit_results())
    assert [r.metrics["node_name"] for _, r in ordered] == list(NODES)
    # the cids run the other way, so a cid-based sort would give the reverse
    assert [p.cid for p, _ in ordered] == ["cid-7", "cid-6", "cid-5"]


def test_sorted_results_is_a_fixed_point_over_every_permutation():
    expected = [r.metrics["node_name"] for _, r in sorted_results(_fit_results())]
    for perm in _permutations(_fit_results()):
        assert [r.metrics["node_name"] for _, r in sorted_results(perm)] == expected


@pytest.mark.parametrize("strategy_name", ["fedavg", "fedprox", "fedbn"])
def test_aggregate_fit_is_bitwise_identical_under_every_permutation(strategy_name):
    from src.fl.server import FedAvgOrdered, FedProxOrdered
    from src.fl.fedbn_strategy import FedBN

    def make():
        if strategy_name == "fedavg":
            return FedAvgOrdered()
        if strategy_name == "fedprox":
            return FedProxOrdered(proximal_mu=0.01)
        return FedBN()

    reference = None
    for perm in _permutations(_fit_results()):
        params, _metrics = make().aggregate_fit(1, perm, [])
        arrays = parameters_to_ndarrays(params)
        if reference is None:
            reference = arrays
            continue
        assert len(arrays) == len(reference)
        for got, want in zip(arrays, reference):
            # bitwise, not approximate: 1e-7 here became 6 % by round 2
            assert got.tobytes() == want.tobytes(), \
                "%s aggregate_fit depends on client arrival order" % strategy_name


@pytest.mark.parametrize("strategy_name", ["fedavg", "fedprox", "fedbn"])
def test_aggregate_evaluate_is_bitwise_identical_under_every_permutation(strategy_name):
    from src.fl.server import FedAvgOrdered, FedProxOrdered
    from src.fl.fedbn_strategy import FedBN

    def make():
        if strategy_name == "fedavg":
            return FedAvgOrdered()
        if strategy_name == "fedprox":
            return FedProxOrdered(proximal_mu=0.01)
        return FedBN()

    reference = None
    for perm in _permutations(_evaluate_results()):
        loss, _metrics = make().aggregate_evaluate(1, perm, [])
        if reference is None:
            reference = loss
            continue
        assert loss.hex() == reference.hex(), \
            "%s aggregate_evaluate depends on client arrival order" % strategy_name


# --------------------------------------------------------------------------- #
# the recorder: weighted means, model-selection input, results.json row order
# --------------------------------------------------------------------------- #
def _metric_rows():
    rows = []
    for i, node in enumerate(NODES):
        n = (911, 41786, 2400)[i]
        rows.append((n, {"node_name": node, "server_round": 1,
                         "val_loss": [4.48001877359, 1e-7, 3.73008904][i],
                         "val_n_examples": n,
                         "accuracy": [0.9339, 1e-8, 0.8685][i],
                         "test_n_examples": n,
                         "test_accuracy": [0.94, 1e-8, 0.86][i]}))
    return rows


def test_recorder_aggregate_and_val_loss_are_identical_under_every_permutation():
    from src.fl.server import RoundRecorder

    reference_agg, reference_loss = None, None
    for perm in _permutations(_metric_rows()):
        ordered = sorted_metrics(perm)
        agg = RoundRecorder.aggregate(ordered)
        loss = RoundRecorder.round_val_loss(ordered)
        if reference_agg is None:
            reference_agg, reference_loss = agg, loss
            continue
        assert agg.keys() == reference_agg.keys()
        for key in agg:
            assert repr(agg[key]) == repr(reference_agg[key]), \
                "aggregate()[%r] depends on arrival order" % key
        assert loss.hex() == reference_loss.hex(), \
            "the model-selection input depends on arrival order"


def test_results_json_client_rows_are_ordered_by_node_name():
    from src.fl.server import RoundRecorder

    for perm in _permutations(_metric_rows()):
        rec = RoundRecorder(start_time=0.0)
        rec.evaluate_aggregation([(n, dict(m)) for n, m in perm])
        rows = rec.rounds[1]["evaluate"]["clients"]
        assert list(rows.keys()) == list(NODES), \
            "per-client rows follow arrival order, not node name"


def test_fit_aggregation_rows_are_ordered_by_node_name():
    from src.fl.server import RoundRecorder

    for perm in _permutations(_metric_rows()):
        rec = RoundRecorder(start_time=0.0)
        rec.fit_aggregation([(n, dict(m)) for n, m in perm])
        assert list(rec.rounds[1]["fit"]["clients"].keys()) == list(NODES)


def test_missing_node_name_falls_back_to_arrival_order_and_warns(capsys):
    """A client without any identifier must not crash, but must be loud.

    `hostname` is an accepted identifier, so a client reporting only a hostname
    still sorts deterministically (by that hostname). Only a client reporting
    neither falls back to arrival order, and that is what has to warn.
    """
    import src.fl.aggregation_order as ao
    ao._warned.clear()
    rows = [(10, {"hostname": "h2"}), (20, {}), (30, {"node_name": "node_a"})]
    ordered = sorted_metrics(rows)

    # identified clients sort by their identifier; the unidentified one goes last
    assert [m.get("hostname") or m.get("node_name") for _, m in ordered[:2]] == ["h2", "node_a"]
    assert ordered[-1][1] == {}
    assert "not reproducible" in capsys.readouterr().out


def test_identified_clients_stay_ordered_even_when_one_is_unidentified(capsys):
    import src.fl.aggregation_order as ao

    reference = None
    for perm in itertools.permutations([(10, {"node_name": "node_b"}),
                                        (20, {}),
                                        (30, {"node_name": "node_a"})]):
        ao._warned.clear()
        ordered = sorted_metrics(list(perm))
        capsys.readouterr()
        names = [m.get("node_name") for _, m in ordered if m]
        if reference is None:
            reference = names
        assert names == reference == ["node_a", "node_b"]


def test_results_json_records_the_cli_strategy_name_not_the_class_name():
    """analyze_results.py groups runs by results.json["strategy"].

    get_strategy now returns FedAvgOrdered / FedProxOrdered, so this guards that
    the *recorded* string stayed "fedavg" / "fedprox_<mu>" / "fedbn"; otherwise
    every new run would land in its own configuration and be paired with nothing.
    """
    import types

    from src.fl.server import build_results, get_strategy, RoundRecorder

    history = types.SimpleNamespace(losses_distributed=[], metrics_distributed={},
                                    metrics_distributed_fit={}, losses_centralized=[],
                                    metrics_centralized={})
    for name in ("fedavg", "fedprox_0.05", "fedbn"):
        strategy = get_strategy(name, min_clients=3)
        args = types.SimpleNamespace(strategy=name, rounds=3, min_clients=3, seed=42,
                                     address="0.0.0.0:8080", tag=[])
        res = build_results(args, history, 1.0, RoundRecorder(start_time=0.0), None)
        assert res["strategy"] == name
        assert "Ordered" not in res["strategy"]
        # the class really is the ordered variant for the two flwr-provided ones
        if name != "fedbn":
            assert type(strategy).__name__.endswith("Ordered")


def test_strategy_names_recorded_are_the_ones_the_analysis_recognises():
    """The recorded strategy string must survive analyze_results' own parsing."""
    from scripts.analyze_results import parse_mu, strategy_display

    # analyze_results.parse_mu returns None where there is no proximal term
    # (server.parse_mu returns 0.0 there); only FedProx carries a mu.
    for name, mu, expected in (("fedavg", None, "FedAvg"),
                               ("fedprox_0.05", 0.05, "FedProx(mu=0.05)"),
                               ("fedbn", None, "FedBN")):
        got_mu = parse_mu(name, None)
        assert got_mu == mu, "%r -> mu %r, expected %r" % (name, got_mu, mu)
        display = strategy_display(name, got_mu, kind="fl")
        assert display == expected, \
            "%r displays as %r; the analysis no longer recognises it" % (name, display)
