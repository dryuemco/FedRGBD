#!/usr/bin/env python3
"""FedRGBD -- declared cross-configuration comparison: MAXN_SUPER vs the heterogeneous matrix.

Declared in ``docs/CROSS_CONFIG_COMPARISON.md`` before block 5b (``maxn_long_horizon``)
launched; read that file first.  This is an analysis, never a gate: it is reported
whatever it shows.

For every cell (partition x strategy) with a heterogeneous 3-round run and seed pairs:
the MAXN run's rounds 1-3 against the heterogeneous 3-round run of the same cell and
seed.  Each run's model is the one of the round in {1, 2, 3} with the lowest aggregated
validation loss (the declared selection rule, restricted to rounds 1-3; earlier round on
ties).  Its value is balanced accuracy pooled over the union of the three clients' test
splits (CLAUDE.md rule 8, primary aggregation).  The seed-paired difference
D = MAXN - heterogeneous gets the stratified cluster bootstrap of
``src.evaluation.bootstrap.seed_paired_diff_ci``, a three-way interval verdict and a
Holm-adjusted verdict across the cells.

    python scripts/cross_config_comparison.py --results_dir results --output_dir analysis/cross_config
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.evaluation.bootstrap import Unit, seed_paired_diff_ci  # noqa: E402
from src.evaluation.model_selection import select_round  # noqa: E402
from src.evaluation.predictions import load_npz  # noqa: E402

NODES = ("node_a", "node_b", "node_c")
METRIC = "balanced_accuracy"
B = 10000
ALPHA = 0.05
#: the rounds a MAXN run is compared on -- those the heterogeneous matrix has
ROUNDS = (1, 2, 3)
#: (directory tag, partition name in data/splits)
PARTITIONS = (("iid", "iid"), ("noniid", "non_iid_label"))
#: strategies with a heterogeneous 3-round run of every seed; FedBN has none (see the doc)
STRATEGIES = (("fedavg", "FedAvg"), ("fedprox_0.01", "FedProx(0.01)"))
SEEDS = (42, 123, 456, 789, 1011)
MAXN_ROOT = os.path.join("pc_maxn")

#: verdict phrases, fixed in docs/CROSS_CONFIG_COMPARISON.md
HIGHER = "higher at MAXN_SUPER"
NO_DIFFERENCE = "no detectable difference"
LOWER = "lower at MAXN_SUPER"


def maxn_dir(results_dir: str, tag: str, strategy: str, seed: int) -> str:
    return os.path.join(results_dir, MAXN_ROOT, "rev_%s_%s_r10_seed%d" % (tag, strategy, seed))


def heterogeneous_dir(results_dir: str, tag: str, strategy: str, seed: int) -> str:
    return os.path.join(results_dir, "rev_%s_%s_seed%d" % (tag, strategy, seed))


def selected_round_1_3(run_dir: str) -> Optional[int]:
    """The declared selection rule restricted to rounds 1-3."""
    with open(os.path.join(run_dir, "results.json")) as f:
        vl = (json.load(f).get("model_selection") or {}).get("val_loss_by_round") or {}
    return select_round({int(r): v for r, v in vl.items() if int(r) in ROUNDS})


def pooled_unit(run_dir: str, rnd: int, groups: Dict[str, str]) -> Unit:
    """The run's round-``rnd`` global model on the union of the clients' test splits."""
    labels, margins, gids = [], [], []
    for node in NODES:
        path = os.path.join(run_dir, "predictions", "r%03d_%s_test.npz" % (rnd, node))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        d = load_npz(path)
        labels.append(d["label"])
        margins.append(d["logit_margin"])
        gids.append(np.array([groups[str(p)] for p in d["path"]]))
    return Unit(np.concatenate(labels), np.concatenate(margins), np.concatenate(gids))


def _ba(u: Unit) -> float:
    tp, fp, fn, tn = u.counts.sum(axis=0)
    return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def verdict_from_ci(ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return HIGHER
    if ci_high < 0:
        return LOWER
    return NO_DIFFERENCE


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1), in input order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="mergesort")):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def verdict_from_holm(diff: float, p_holm: float) -> str:
    if p_holm < ALPHA:
        return HIGHER if diff > 0 else LOWER
    return NO_DIFFERENCE


def compare(results_dir: str, groups_of, B: int = B):
    """-> (per-run rows, per-cell comparison rows) as DataFrames.

    ``groups_of(partition)`` -> {manifest path: group id}.  A seed enters a cell only
    when both runs exist; a cell with no pair is left out and listed as missing.
    """
    run_rows, comp_rows = [], []
    for tag, partition in PARTITIONS:
        groups = None
        for strategy, label in STRATEGIES:
            pairs = {}
            for seed in SEEDS:
                m_dir = maxn_dir(results_dir, tag, strategy, seed)
                h_dir = heterogeneous_dir(results_dir, tag, strategy, seed)
                if not (os.path.isfile(os.path.join(m_dir, "results.json"))
                        and os.path.isfile(os.path.join(h_dir, "results.json"))):
                    continue
                groups = groups if groups is not None else groups_of(partition)
                units = {}
                for side, d in (("maxn", m_dir), ("heterogeneous", h_dir)):
                    rnd = selected_round_1_3(d)
                    if rnd is None:
                        raise ValueError("%s: no finite validation loss in rounds 1-3" % d)
                    units[side] = pooled_unit(d, rnd, groups)
                    run_rows.append({"partition": partition, "strategy": label, "seed": seed,
                                     "power_config": side, "run_dir": d, "selected_round": rnd,
                                     "n_images": int(units[side].n.sum()),
                                     METRIC: _ba(units[side])})
                pairs[seed] = units
            if not pairs:
                comp_rows.append({"partition": partition, "strategy": label, "n_pairs": 0,
                                  "paired_seeds": "", "note": "no seed pair yet"})
                continue
            seeds = sorted(pairs)
            res = seed_paired_diff_ci([[pairs[s]["maxn"]] for s in seeds],
                                      [[pairs[s]["heterogeneous"]] for s in seeds],
                                      "crossconfig|%s|%s" % (partition, strategy), B=B)
            comp_rows.append({"partition": partition, "strategy": label,
                              "n_pairs": res["n_pairs"], "paired_seeds": " ".join(map(str, seeds)),
                              "diff": res["diff"], "ci_low": res["ci_low"],
                              "ci_high": res["ci_high"], "p_boot": res["p_boot"], "B": res["B"],
                              "note": ""})
    runs = pd.DataFrame(run_rows)
    comps = pd.DataFrame(comp_rows)
    return runs, apply_rule(comps)


def apply_rule(comps: pd.DataFrame) -> pd.DataFrame:
    """Interval verdict per cell; Holm across all compared cells (one family)."""
    comps = comps.copy()
    if comps.empty or "diff" not in comps:
        return comps
    done = comps["n_pairs"] > 0
    comps["verdict"] = None
    comps["p_holm"] = np.nan
    comps["holm_m"] = int(done.sum())
    comps["verdict_holm"] = None
    if done.any():
        comps.loc[done, "verdict"] = [verdict_from_ci(lo, hi) for lo, hi in
                                      zip(comps.loc[done, "ci_low"], comps.loc[done, "ci_high"])]
        comps.loc[done, "p_holm"] = holm(comps.loc[done, "p_boot"].to_numpy())
        comps.loc[done, "verdict_holm"] = [verdict_from_holm(d, p) for d, p in
                                           zip(comps.loc[done, "diff"], comps.loc[done, "p_holm"])]
    return comps


def comparisons_markdown(comps: pd.DataFrame) -> str:
    lines = ["# Cross-configuration comparison: MAXN_SUPER - heterogeneous, rounds 1-3", "",
             "Declared in docs/CROSS_CONFIG_COMPARISON.md. Balanced accuracy pooled over the "
             "union of the clients' test splits at the round of rounds 1-3 with the lowest "
             "aggregated validation loss; seed-paired difference in percentage points, 95 %% "
             "stratified cluster-bootstrap interval (B = %d); Holm across the compared cells."
             % B, "",
             "| partition | strategy | seeds | diff | 95 % CI | p | p Holm | verdict | verdict (Holm) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in comps.itertuples():
        if not r.n_pairs:
            lines.append("| %s | %s | none | | | | | %s | |" % (r.partition, r.strategy, r.note))
            continue
        lines.append("| %s | %s | %s | %+.1f | [%+.1f, %+.1f] | %.4f | %.4f | %s | %s |"
                     % (r.partition, r.strategy, r.paired_seeds, 100 * r.diff, 100 * r.ci_low,
                        100 * r.ci_high, r.p_boot, r.p_holm, r.verdict, r.verdict_holm))
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--output_dir", default=os.path.join("analysis", "cross_config"))
    ap.add_argument("--B", type=int, default=B)
    args = ap.parse_args(argv)
    from scripts.analyze_results import partition_tables

    def groups_of(partition):
        tables = partition_tables(partition)
        if tables is None:
            raise SystemExit("partition %s not in data/splits" % partition)
        return tables[0]

    runs, comps = compare(args.results_dir, groups_of, B=args.B)
    os.makedirs(args.output_dir, exist_ok=True)
    runs.to_csv(os.path.join(args.output_dir, "cross_config_runs.csv"), index=False)
    comps.to_csv(os.path.join(args.output_dir, "cross_config_comparisons.csv"), index=False)
    with open(os.path.join(args.output_dir, "cross_config_comparisons.md"), "w",
              encoding="utf-8") as f:
        f.write(comparisons_markdown(comps))
    print("wrote %d run rows and %d cells to %s" % (len(runs), len(comps), args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
