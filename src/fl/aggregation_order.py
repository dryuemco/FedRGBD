"""FedRGBD — deterministic client order for aggregation.

Flower hands ``aggregate_fit`` / ``aggregate_evaluate`` the client results in
*arrival* order, which on a physical testbed is whichever node finished its
local epochs first. Floating-point addition is not associative, and both of
Flower's aggregation helpers sum in list order::

    aggregate():        reduce(np.add, layer_updates) / num_examples_total
    weighted_loss_avg(): sum(num_examples * loss for ...) / total

so the global model depends on the order in which three Jetsons happen to
report. Two runs of the same commit at the same seed (``iid_sub0.01``,
\\fedavg, 2 rounds, seed 42) produced bit-identical *local* training --
identical client ``train_loss``, identical per-client val/test accuracy -- and
still diverged at aggregation:

    round 1 aggregated val loss   4.4800187735908255  vs  4.480026002017449
    round 2 aggregated val loss   0.7300890427681955  vs  0.6840349276794742

The round-1 gap is ~7e-6 relative; by round 2 it is 6 %, because the perturbed
global model changes every client's next local optimum.

Sorting the results by a stable client identifier before aggregating removes
this, and it is the *only* nondeterminism worth removing here: it costs nothing
at runtime. Enabling ``cudnn.deterministic`` or
``torch.use_deterministic_algorithms`` would be the wrong fix, because per-round
wall-clock is a reported result of this paper and those flags change it.

The identifier is the **node name** the client echoes in its fit and evaluate
metrics (``src/fl/client.py``), never Flower's ``cid``, which is assigned per
connection and differs between runs.
"""

from typing import Any, Dict, List, Sequence, Tuple

#: metric keys carrying a stable client identity, most specific first
CLIENT_ID_KEYS = ("node_name", "hostname")

_warned = set()


def _warn_once(what: str) -> None:
    if what not in _warned:
        _warned.add(what)
        print("  [aggregation_order WARNING] %s: aggregation falls back to arrival "
              "order and the run is not reproducible at a fixed seed." % what)


def _key_from_metrics(metrics: Dict[str, Any], index: int) -> Tuple[int, str]:
    """Sort key from a client's metrics dict.

    Rank 0 means a real identifier was found; rank 1 is the arrival index, which
    preserves the previous behaviour but cannot make the run reproducible.
    """
    for key in CLIENT_ID_KEYS:
        value = (metrics or {}).get(key)
        if value not in (None, ""):
            return (0, str(value))
    _warn_once("a client reported none of %s" % (CLIENT_ID_KEYS,))
    return (1, "%08d" % index)


def sorted_results(results: Sequence[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
    """``[(ClientProxy, FitRes | EvaluateRes), ...]`` sorted by node name.

    Used before the parameter and loss aggregation of every strategy.
    """
    keyed = [(_key_from_metrics(getattr(res, "metrics", None), i), i, pair)
             for i, pair in enumerate(results)
             for _, res in (pair,)]
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [pair for _, _, pair in keyed]


def sorted_metrics(metrics: Sequence[Tuple[int, Dict[str, Any]]]
                   ) -> List[Tuple[int, Dict[str, Any]]]:
    """``[(num_examples, metrics), ...]`` sorted by node name.

    Used by :class:`~src.fl.server.RoundRecorder` so that the weighted means it
    computes, and the order of the per-client rows it writes into
    ``results.json``, do not depend on arrival order either.
    """
    keyed = [(_key_from_metrics(m, i), i, (n, m)) for i, (n, m) in enumerate(metrics)]
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [pair for _, _, pair in keyed]


class DeterministicClientOrder:
    """Mixin: sort client results by node name before Flower aggregates them.

    Mix in *before* the strategy class so that it wins the MRO::

        class FedAvgOrdered(DeterministicClientOrder, FedAvg):
            pass

    A subclass that overrides ``aggregate_fit`` without delegating to ``super()``
    (FedBN does) must call :func:`sorted_results` itself.
    """

    def aggregate_fit(self, server_round, results, failures):  # type: ignore[no-untyped-def]
        return super().aggregate_fit(server_round, sorted_results(results), failures)

    def aggregate_evaluate(self, server_round, results, failures):  # type: ignore[no-untyped-def]
        return super().aggregate_evaluate(server_round, sorted_results(results), failures)
