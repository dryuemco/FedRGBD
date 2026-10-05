#!/usr/bin/env python3
"""FedRGBD -- declared timing comparison: MAXN_SUPER vs the heterogeneous matrix.

Declared in ``docs/CROSS_CONFIG_COMPARISON.md`` (c) and its amendment (2026-09-29, while
5b was still running, before any ratio existed); read that section first.  Descriptive
only: no verdicts, no tests, no Holm, and no ratio is ever called "equal", "equivalent"
or "no difference".

Per run, following the timing reporting rule (CLAUDE.md rule 12): T_round is the
test-free round time ``rounds[].timing.round_time_s``; round 1 is the cold start and is
its own quantity; a steady-state value is the median of T_round over the stated rounds.
Never a mean over all rounds, never total / R.

Per cell and quantity, per paired seed s: r_s = T_MAXN,s / T_het,s.  The reported ratio
is the geometric mean exp(mean_s log r_s), always printed with every r_s.  Its interval
is a seed bootstrap (paired seeds resampled with replacement, pairs kept together,
B = 10,000, 95 % percentile), generator ``BASE_SEED + crc32("timing|" + cell + "|" +
quantity)`` with cell = ``<family>|<partition>|<strategy>``.  With n paired seeds the
bootstrap has at most C(2n-1, n) distinct resamples (10 for 3 seeds, 126 for 5), so the
interval is labelled indicative wherever it is printed.

Families (cells and seeds as in (b)):

``rounds_1_3``  IID and label skew x FedAvg and FedProx(0.01), seeds 42/123/456/789/1011;
                MAXN ten-round 5b run vs the heterogeneous three-round run.
                primary   = median T_round of rounds 2-3 in both;
                secondary = MAXN rounds 2-10 vs heterogeneous rounds 2-3;
                round1    = T_round of round 1.
``ten_rounds``  label skew x FedAvg and FedBN, seeds 42/123/456; MAXN vs the
                heterogeneous ``long_horizon_fedbn`` ten-round run.
                steady    = rounds 2-10 in both;  round1 = T_round of round 1.

Straggler: in each round the client with the largest ``fit_wall_s``.  Per configuration
and cell, how often each node is the straggler over the rounds entering the steady
state (rounds 2-3 for the primary of ``rounds_1_3``, rounds 2-10 otherwise) and,
separately, in round 1, with each node's median ``fit_wall_s`` over the same rounds
(pooled over the paired seeds).  Reported, never tested.

    python scripts/cross_config_timing.py --results_dir results --output_dir analysis/cross_config_timing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zlib
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from scripts.cross_config_comparison import heterogeneous_dir, maxn_dir  # noqa: E402
from src.evaluation.bootstrap import BASE_SEED  # noqa: E402

NODES = ("node_a", "node_b", "node_c")
B = 10000
LEVEL = 0.95
R2_3 = (2, 3)
R2_10 = tuple(range(2, 11))

#: quantity -> (MAXN rounds, heterogeneous rounds); round 1 is a single round, not a median
FAMILIES = (
    {"family": "rounds_1_3",
     "partitions": (("iid", "iid"), ("noniid", "non_iid_label")),
     "strategies": (("fedavg", "FedAvg"), ("fedprox_0.01", "FedProx(0.01)")),
     "seeds": (42, 123, 456, 789, 1011), "reference_rounds": 3,
     "quantities": (("primary", R2_3, R2_3), ("secondary", R2_10, R2_3),
                    ("round1", (1,), (1,)))},
    {"family": "ten_rounds",
     "partitions": (("noniid", "non_iid_label"),),
     "strategies": (("fedavg", "FedAvg"), ("fedbn", "FedBN")),
     "seeds": (42, 123, 456), "reference_rounds": 10,
     "quantities": (("steady", R2_10, R2_10), ("round1", (1,), (1,)))},
)


def span(rounds: Sequence[int]) -> str:
    """(2, 3, ..., 10) -> "2-10"; (1,) -> "1"."""
    return str(rounds[0]) if len(rounds) == 1 else "%d-%d" % (rounds[0], rounds[-1])


def load_rounds(run_dir: str) -> Dict[int, dict]:
    """{round: round record} of a finished run."""
    with open(os.path.join(run_dir, "results.json")) as f:
        data = json.load(f)
    return {int(r["round"]): r for r in data["rounds"]}


def round_time(rounds: Dict[int, dict], which: Sequence[int]) -> float:
    """Round 1: its own T_round.  Otherwise the median T_round over ``which``."""
    values = []
    for r in which:
        v = rounds[r]["timing"]["round_time_s"]
        if v is None or not math.isfinite(float(v)):
            raise ValueError("round %d has no measured round_time_s" % r)
        values.append(float(v))
    return values[0] if len(values) == 1 else float(np.median(values))


def straggler(record: dict) -> str:
    """The client with the largest fit_wall_s in this round."""
    clients = record["fit"]["clients"]
    return max(NODES, key=lambda n: float(clients[n]["fit_wall_s"]))


def n_distinct_resamples(n: int) -> int:
    """Multisets of size n from n seeds: C(2n-1, n)."""
    return math.comb(2 * n - 1, n)


def geometric_ratio(ratios: Sequence[float], key: str, B: int = B) -> Dict[str, float]:
    """Geometric mean of the per-seed ratios with its seed-bootstrap percentile interval."""
    logs = np.log(np.asarray(ratios, dtype=float))
    n = len(logs)
    rng = np.random.default_rng(BASE_SEED + zlib.crc32(("timing|" + key).encode("utf-8")))
    pick = rng.integers(0, n, size=(B, n))
    rep = np.exp(logs[pick].mean(axis=1))
    lo_q, hi_q = 100 * (1 - LEVEL) / 2, 100 * (1 + LEVEL) / 2
    return {"ratio": float(np.exp(logs.mean())),
            "ci_low": float(np.percentile(rep, lo_q)),
            "ci_high": float(np.percentile(rep, hi_q)),
            "n_pairs": n, "max_distinct_resamples": n_distinct_resamples(n), "B": B}


def compare(results_dir: str, B: int = B):
    """-> (per-run values, ratios, straggler counts) as DataFrames."""
    run_rows, ratio_rows, strag_rows = [], [], []
    for fam in FAMILIES:
        for tag, partition in fam["partitions"]:
            for strategy, label in fam["strategies"]:
                cell = "%s|%s|%s" % (fam["family"], partition, strategy)
                pairs = {}
                for seed in fam["seeds"]:
                    dirs = {"maxn": maxn_dir(results_dir, tag, strategy, seed),
                            "heterogeneous": heterogeneous_dir(results_dir, tag, strategy, seed,
                                                               fam["reference_rounds"])}
                    if all(os.path.isfile(os.path.join(d, "results.json"))
                           for d in dirs.values()):
                        pairs[seed] = {side: (d, load_rounds(d)) for side, d in dirs.items()}
                base = {"family": fam["family"], "partition": partition, "strategy": label}
                seeds = sorted(pairs)
                for quantity, m_rounds, h_rounds in fam["quantities"]:
                    rounds_of = {"maxn": m_rounds, "heterogeneous": h_rounds}
                    values = {}
                    for seed in seeds:
                        for side in ("maxn", "heterogeneous"):
                            d, rounds = pairs[seed][side]
                            v = round_time(rounds, rounds_of[side])
                            values[(seed, side)] = v
                            run_rows.append(dict(base, quantity=quantity, seed=seed,
                                                 power_config=side, run_dir=d,
                                                 rounds=span(rounds_of[side]),
                                                 value_s=v))
                    if not seeds:
                        ratio_rows.append(dict(base, quantity=quantity, n_pairs=0,
                                               paired_seeds="", note="no seed pair"))
                        continue
                    per_seed = [values[(s, "maxn")] / values[(s, "heterogeneous")] for s in seeds]
                    res = geometric_ratio(per_seed, cell + "|" + quantity, B=B)
                    ratio_rows.append(dict(
                        base, quantity=quantity,
                        maxn_rounds=span(m_rounds),
                        heterogeneous_rounds=span(h_rounds),
                        paired_seeds=" ".join(map(str, seeds)),
                        per_seed_ratios=" ".join("%.4f" % r for r in per_seed),
                        note="", **res))
                # straggler counts: the rounds of each steady-state set, then round 1
                strag_sets = {("maxn", m) for _, m, _ in fam["quantities"]} | \
                             {("heterogeneous", h) for _, _, h in fam["quantities"]}
                for side, which in sorted(strag_sets, key=lambda x: (x[0], len(x[1]), x[1])):
                    counts = {n: 0 for n in NODES}
                    walls = {n: [] for n in NODES}
                    for seed in seeds:
                        rounds = pairs[seed][side][1]
                        for r in which:
                            counts[straggler(rounds[r])] += 1
                            for n in NODES:
                                walls[n].append(float(rounds[r]["fit"]["clients"][n]["fit_wall_s"]))
                    row = dict(base, power_config=side, rounds=span(which),
                               n_seeds=len(seeds), n_rounds_counted=len(seeds) * len(which))
                    for n in NODES:
                        row["straggler_%s" % n] = counts[n]
                    for n in NODES:
                        row["median_fit_wall_s_%s" % n] = (float(np.median(walls[n]))
                                                           if walls[n] else np.nan)
                    strag_rows.append(row)
    return pd.DataFrame(run_rows), pd.DataFrame(ratio_rows), pd.DataFrame(strag_rows)


def to_markdown(ratios: pd.DataFrame, strag: pd.DataFrame) -> str:
    lines = ["# Timing comparison: MAXN_SUPER / heterogeneous", "",
             "Declared in docs/CROSS_CONFIG_COMPARISON.md (c) and its amendment. Descriptive "
             "only: no verdicts, no tests. T_round = test-free round time "
             "(`rounds[].timing.round_time_s`); round 1 is the T_round of round 1, a steady-state "
             "value is the median T_round over the stated rounds. Ratio = geometric mean over the "
             "paired seeds of T_MAXN / T_het, shown with every per-seed ratio; interval = seed "
             "bootstrap, B = %d, 95 %% percentile. **The interval is indicative**: with n paired "
             "seeds the bootstrap has at most C(2n-1, n) distinct resamples (column 'distinct')."
             % B, ""]
    for fam, g in ratios.groupby("family", sort=False):
        lines += ["## %s" % fam, "",
                  "| partition | strategy | quantity | MAXN rounds | het rounds | seeds | ratio | "
                  "95 % interval (indicative) | distinct | per-seed ratios |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            if not r.n_pairs:
                lines.append("| %s | %s | %s | | | none | | | | %s |"
                             % (r.partition, r.strategy, r.quantity, r.note))
                continue
            lines.append("| %s | %s | %s | %s | %s | %s | %.4f | [%.4f, %.4f] | %d | %s |"
                         % (r.partition, r.strategy, r.quantity, r.maxn_rounds,
                            r.heterogeneous_rounds, r.paired_seeds, r.ratio, r.ci_low,
                            r.ci_high, r.max_distinct_resamples, r.per_seed_ratios))
        lines.append("")
    lines += ["## Straggler (client with the largest fit_wall_s per round)", "",
              "Counts over the paired seeds x the listed rounds; median fit_wall_s (s) of each "
              "node over the same rounds. Reported, never tested.", "",
              "| family | partition | strategy | configuration | rounds | n | node_a | node_b | "
              "node_c | median a | median b | median c |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in strag.itertuples():
        lines.append("| %s | %s | %s | %s | %s | %d | %d | %d | %d | %.1f | %.1f | %.1f |"
                     % (r.family, r.partition, r.strategy, r.power_config, r.rounds,
                        r.n_rounds_counted, r.straggler_node_a, r.straggler_node_b,
                        r.straggler_node_c, r.median_fit_wall_s_node_a,
                        r.median_fit_wall_s_node_b, r.median_fit_wall_s_node_c))
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--output_dir", default=os.path.join("analysis", "cross_config_timing"))
    ap.add_argument("--B", type=int, default=B)
    args = ap.parse_args(argv)
    runs, ratios, strag = compare(args.results_dir, B=args.B)
    os.makedirs(args.output_dir, exist_ok=True)
    runs.to_csv(os.path.join(args.output_dir, "timing_runs.csv"), index=False)
    ratios.to_csv(os.path.join(args.output_dir, "timing_ratios.csv"), index=False)
    strag.to_csv(os.path.join(args.output_dir, "timing_stragglers.csv"), index=False)
    with open(os.path.join(args.output_dir, "timing_comparison.md"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(to_markdown(ratios, strag))
    print("wrote %d run values, %d ratios, %d straggler rows to %s"
          % (len(runs), len(ratios), len(strag), args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
