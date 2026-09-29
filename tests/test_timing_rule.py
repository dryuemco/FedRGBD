"""The declared timing reporting rule (CLAUDE.md rule 12, paper sec:timecomm).

Per run: round 1 is the cold start and is reported separately; the steady-state
per-round time is the median of the test-free round time over rounds 2..R -- never the
mean over all rounds, never total / R.  Across seeds both are summarised like every
other time.  One rule for every power configuration and strategy; the time table of each
configuration is built at that configuration's round count and never holds two.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.analyze_results import load_run, round_time_rule, summary_table  # noqa: E402
from scripts.export_latex_tables import time_tex  # noqa: E402
from tests.test_analyze_selection import schema3_payload, write_run  # noqa: E402


def _data(times, num_rounds=None):
    rounds = [{"round": i + 1, "timing": {"round_time_s": t, "round_wall_s": t + 40.0}}
              for i, t in enumerate(times)]
    return {"num_rounds": len(times) if num_rounds is None else num_rounds, "rounds": rounds}


def test_round_one_apart_and_median_of_the_rest():
    # a cold first round, then an odd count of steady rounds with one slow outlier
    times = [1080.0, 930.0, 940.0, 920.0, 925.0, 2000.0, 935.0, 931.0, 929.0, 933.0]
    r1, steady = round_time_rule(_data(times))
    assert r1 == 1080.0
    assert steady == pytest.approx(np.median(times[1:])) == pytest.approx(931.0)
    # neither the mean over all rounds nor total / R
    assert steady != pytest.approx(np.mean(times)) and steady != pytest.approx(sum(times) / 10)


def test_three_rounds_median_of_rounds_two_and_three():
    assert round_time_rule(_data([1400.0, 1300.0, 1310.0])) == (1400.0, pytest.approx(1305.0))


def test_one_round_has_no_steady_state():
    assert round_time_rule(_data([1000.0])) == (1000.0, None)


@pytest.mark.parametrize("data", [
    {"num_rounds": 3, "rounds": []},
    _data([1000.0, 900.0], num_rounds=3),                       # a round missing
    {"num_rounds": 2, "rounds": [{"round": 1, "timing": {"round_time_s": 1000.0}},
                                 {"round": 2, "timing": {}}]},   # a round unmeasured
    {"num_rounds": 2, "rounds": [{"round": 1, "timing": {"round_time_s": 1000.0}},
                                 {"round": 2, "timing": {"round_time_s": float("nan")}}]},
    {"num_rounds": 2, "rounds": [{"round": 1, "timing": {"round_time_s": 1000.0}},
                                 {"round": 3, "timing": {"round_time_s": 900.0}}]},
    {"total_time_s": 3000.0, "num_rounds": 3},                   # v1: no per-round timing
])
def test_incomplete_timing_gives_nothing_rather_than_an_estimate(data):
    assert round_time_rule(data) == (None, None)


def _payload(dist, seed, n_rounds, first, steady):
    p = schema3_payload("fedavg", dist, seed, [0.5 - 0.01 * r for r in range(n_rounds)],
                        [0.8] * n_rounds)
    for i, entry in enumerate(p["rounds"]):
        entry["timing"]["round_time_s"] = first if i == 0 else steady + i
    p["client_config"] = {n: dict(c) for n, c in p["client_config"].items()}
    return p


def test_run_record_and_summary_carry_the_rule(tmp_path):
    root = str(tmp_path)
    runs = [load_run(write_run(root, "rev_iid_fedavg_seed%d" % s,
                               _payload("iid", s, 3, 1400.0 + s, 1300.0)), warn=False)
            for s in (42, 123, 456)]
    for r in runs:
        assert r["round1_time_s"] == 1400.0 + r["seed"]
        assert r["steady_round_time_s"] == pytest.approx(1301.5)   # median(1301, 1302)
    summary = summary_table(runs, bootstrap_B=200)
    row = summary[summary.metric == "steady_round_time_s"].iloc[0]
    assert int(row.n_seeds) == 3 and row["mean"] == pytest.approx(1301.5)
    row = summary[summary.metric == "round1_time_s"].iloc[0]
    assert row["mean"] == pytest.approx(np.mean([1442.0, 1523.0, 1856.0]))


def _rows(n_rounds, power, first, steady, total):
    out = []
    for metric, mean in (("total_time_s", total), ("final_cumulative_mb", 165.5),
                         ("round1_time_s", first), ("steady_round_time_s", steady)):
        out.append({"config_id": "fedavg|iid|%d|%s" % (n_rounds, power),
                    "label": "FedAvg iid [3N] {group}", "kind": "fl", "strategy": "fedavg",
                    "mu": None, "distribution": "iid", "n_nodes": 3, "num_rounds": n_rounds,
                    "local_epochs": 5, "lr": 0.001, "protocol": "group", "power_config": power,
                    "metric": metric, "n_seeds": 5, "seeds": "42,123,456,789,1011",
                    "mean": mean, "std": 2.0, "ci_low": mean - 3.0, "ci_high": mean + 3.0})
    return out


def test_time_table_prints_both_rule_columns_per_configuration():
    hetero = pd.DataFrame(_rows(3, "heterogeneous", 1432.0, 1371.0, 4128.0)
                          + _rows(10, "heterogeneous", 1500.0, 1400.0, 14000.0))
    text = time_tex(hetero)
    assert "Round 1 (s)" in text and "Rounds 2--3, median (s)" in text
    assert "(3 Rounds)" in text and "MAXN" not in text
    line = next(l for l in text.splitlines() if l.startswith("3N IID FedAvg"))
    assert "1432.0 $\\pm$ 2.0 [1429.0, 1435.0] (5)" in line and "1371.0 $\\pm$ 2.0" in line
    assert "1500" not in text                        # the 10-round runs are not the 3-round point

    maxn = pd.DataFrame(_rows(10, "maxn", 1062.0, 925.0, 9400.0))
    text = time_tex(maxn, power_config="maxn")
    assert "Rounds 2--10, median (s)" in text and "All Nodes at MAXN" in text
    assert "1062.0 $\\pm$ 2.0" in text and "925.0 $\\pm$ 2.0" in text
    assert time_tex(maxn) is None                    # never at the heterogeneous point
