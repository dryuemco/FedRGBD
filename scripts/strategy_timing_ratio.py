#!/usr/bin/env python3
"""FedRGBD -- declared wall-clock ratio FedProx(0.01) / FedAvg.

Declared in ``docs/STRATEGY_TIMING_RATIO.md`` (2026-10-13, after the per-strategy times
of tab:time had been seen, before any paired ratio existed); read it first.  Descriptive
only: no verdicts, no tests, no Holm, and no ratio is ever called "equal",
"equivalent" or "no difference".

Per cell and quantity, per seed s: r_s = T_FedProx,s / T_FedAvg,s (same power
configuration, partition and seed).  The ratio is the geometric mean over the paired
seeds with a seed-bootstrap interval, the statistic of ``cross_config_timing.py``
(generator key ``"timing|strategy|<power_config>|<partition>|<quantity>"``), always
printed with every r_s; the interval is indicative (at most C(2n-1, n) distinct
resamples).

Quantities (timing reporting rule, CLAUDE.md rule 12; T_round = test-free round time):

``steady``                     median T_round over rounds 2..R (the overhead factor)
``round1``                     T_round of round 1 (the cold start)
``total``                      total_time_excl_test_s (R rounds)
``prox_round1_vs_avg_steady``  FedProx round-1 T_round / FedAvg steady T_round

Cells: heterogeneous main matrix (three rounds, ``results/rev_*``) and the MAXN_SUPER
block 5b (ten rounds, ``results/pc_maxn/rev_*_r10``), each IID and label skew, seeds
42/123/456/789/1011; never pooled.  Derived: FedAvg rounds bought by FedProx's R-round
budget = R x the ``total`` ratio (interval R x its interval).

    python scripts/strategy_timing_ratio.py --results_dir results --output_dir analysis/strategy_timing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from scripts.cross_config_comparison import heterogeneous_dir, maxn_dir  # noqa: E402
from scripts.cross_config_timing import B, geometric_ratio, load_rounds, round_time  # noqa: E402

SEEDS = (42, 123, 456, 789, 1011)
PARTITIONS = (("iid", "iid"), ("noniid", "non_iid_label"))
#: power configuration -> (rounds R, run directory of (results_dir, tag, strategy, seed))
CONFIGS = (
    ("heterogeneous", 3, lambda rd, tag, strategy, seed: heterogeneous_dir(rd, tag, strategy, seed)),
    ("maxn", 10, lambda rd, tag, strategy, seed: maxn_dir(rd, tag, strategy, seed)),
)
QUANTITIES = ("steady", "round1", "total", "prox_round1_vs_avg_steady")


def run_times(run_dir: str, R: int) -> Dict[str, float]:
    """round1, steady (median T_round of rounds 2..R) and total of one finished run."""
    rounds = load_rounds(run_dir)
    if sorted(rounds) != list(range(1, R + 1)):
        raise ValueError("%s: rounds %s, expected 1..%d" % (run_dir, sorted(rounds), R))
    with open(os.path.join(run_dir, "results.json")) as f:
        total = json.load(f).get("total_time_excl_test_s")
    if total is None:
        raise ValueError("%s: no total_time_excl_test_s" % run_dir)
    return {"round1": round_time(rounds, (1,)),
            "steady": round_time(rounds, tuple(range(2, R + 1))),
            "total": float(total)}


def per_seed_ratio(quantity: str, prox: Dict[str, float], avg: Dict[str, float]) -> float:
    if quantity == "prox_round1_vs_avg_steady":
        return prox["round1"] / avg["steady"]
    return prox[quantity] / avg[quantity]


def compare(results_dir: str, B: int = B) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """-> (per-run values, ratios)."""
    run_rows, ratio_rows = [], []
    for config, R, dir_of in CONFIGS:
        for tag, partition in PARTITIONS:
            times = {}
            for seed in SEEDS:
                dirs = {s: dir_of(results_dir, tag, s, seed) for s in ("fedprox_0.01", "fedavg")}
                if not all(os.path.isfile(os.path.join(d, "results.json")) for d in dirs.values()):
                    continue
                times[seed] = {s: run_times(d, R) for s, d in dirs.items()}
                for s, d in dirs.items():
                    run_rows.append(dict(power_config=config, partition=partition, strategy=s,
                                         seed=seed, rounds=R, run_dir=d.replace(os.sep, "/"),
                                         **times[seed][s]))
            seeds = sorted(times)
            for quantity in QUANTITIES:
                base = dict(power_config=config, partition=partition, rounds=R, quantity=quantity)
                if not seeds:
                    ratio_rows.append(dict(base, n_pairs=0, paired_seeds="", note="no seed pair"))
                    continue
                per_seed = [per_seed_ratio(quantity, times[s]["fedprox_0.01"], times[s]["fedavg"])
                            for s in seeds]
                res = geometric_ratio(per_seed, "strategy|%s|%s|%s" % (config, partition, quantity),
                                      B=B)
                ratio_rows.append(dict(base, paired_seeds=" ".join(map(str, seeds)),
                                       per_seed_ratios=" ".join("%.4f" % r for r in per_seed),
                                       note="", **res))
    ratios = pd.DataFrame(ratio_rows)
    tot = ratios["quantity"] == "total"
    budget = ratios[tot].copy()
    budget["quantity"] = "fedavg_rounds_in_fedprox_budget"
    for col in ("ratio", "ci_low", "ci_high"):
        budget[col] = budget[col] * budget["rounds"]
    budget["per_seed_ratios"] = ""
    budget["note"] = "derived: R x the total ratio and its interval"
    return pd.DataFrame(run_rows), pd.concat([ratios, budget], ignore_index=True)


def to_markdown(ratios: pd.DataFrame) -> str:
    lines = ["# Wall-clock ratio FedProx(0.01) / FedAvg", "",
             "Declared in docs/STRATEGY_TIMING_RATIO.md (2026-10-13, after the per-strategy "
             "times had been seen). Descriptive only: no verdicts, no tests. Ratio = geometric "
             "mean over the paired seeds of r_s = T_FedProx,s / T_FedAvg,s; interval = seed "
             "bootstrap, 95 % percentile, **indicative** (n paired seeds give at most "
             "C(2n-1, n) distinct resamples: 126 for five). Per-seed ratios in seed order. "
             "The last quantity is derived (R x the total ratio).", ""]
    for config in ratios["power_config"].unique():
        lines += ["## %s" % config, "",
                  "| partition | R | quantity | ratio | 95 % CI (indicative) | seeds | per-seed ratios |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in ratios[ratios["power_config"] == config].iterrows():
            if not r.get("n_pairs"):
                continue
            lines.append("| %s | %d | %s | %.3f | [%.3f, %.3f] | %s | %s |" % (
                r["partition"], r["rounds"], r["quantity"], r["ratio"], r["ci_low"],
                r["ci_high"], r["paired_seeds"], r["per_seed_ratios"] or "--"))
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--output_dir", default=os.path.join("analysis", "strategy_timing"))
    ap.add_argument("--B", type=int, default=B)
    args = ap.parse_args(argv)
    runs, ratios = compare(args.results_dir, B=args.B)
    os.makedirs(args.output_dir, exist_ok=True)
    runs.to_csv(os.path.join(args.output_dir, "runs.csv"), index=False, lineterminator="\n")
    ratios.to_csv(os.path.join(args.output_dir, "ratios.csv"), index=False,
                  lineterminator="\n", float_format="%.6g")
    md = to_markdown(ratios)
    with open(os.path.join(args.output_dir, "ratios.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
