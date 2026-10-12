#!/usr/bin/env python3
"""FedRGBD -- A5: the additional Dirichlet partition draws, descriptive only.

Declared in ``docs/A5_DIRICHLET_DRAWS_PREREG.md`` (sec. 3a, 3b, 4, Amendments A5-1 and
A5-2).  This script reads exactly the declared runs and nothing else:

* **testbed** (sec. 3a; primary for any A5 statement): alpha = 0.1, the draws ps123 and
  ps456, FedAvg and FedProx(0.01), MAXN_SUPER, 10 rounds, training seed 42 --
  ``results/pc_maxn/rev_dirichlet0.1_ps<draw>_<strategy>_r10_seed42``;
* **desktop simulation sensitivity analysis** (sec. 3b, restored by A5-2): alpha in
  {0.1, 0.5, 1}, the draws ps42 (= the main study's ``dirichlet_<alpha>``), ps123 and
  ps456, the same two strategies -- ``results/desktop_sim/sim_dirichlet<alpha>_ps<draw>_
  <strategy>_r10_seed42``.

Per run (sec. 4): the balanced accuracy of the round chosen by the declared model
selection rule, pooled over all held-out images (primary) and as the unweighted mean over
the clients (secondary), each with the paper's sequence-level stratified bootstrap
interval (``config_ci``, B = ``DEFAULT_B`` as in tab:dirichlet).  Across the draws of one
(setting, alpha, strategy): the range and the SD -- testbed over ps123 and ps456 only
(n = 2, A5-1 (c)), desktop over ps42, ps123 and ps456 (n = 3, A5-2).  The measured skew
of every draw (``scripts/partition_skew.describe``), and the testbed runs' times under
the timing rule (CLAUDE.md rule 12: round 1 and the median of rounds 2..R).

Never computed here: a test, a verdict, a Holm family, a clean subset (none is declared
for A5), a figure pooled across testbed and desktop, or anything compared with or
differenced against the main study's Dirichlet results.  ``scripts/analyze_results.py``
skips these runs (``is_a5_draw_entry``, ``DESKTOP_SIM_TREE``).

Writes ``analysis/a5/a5_runs.csv``, ``a5_spread.csv`` and ``a5_skew.csv``, which
``scripts/export_latex_tables.py`` turns into ``tab:a5_testbed``, ``tab:a5_desktop_sim``
and ``tab:a5_skew``.

    python scripts/a5_dirichlet_draws.py --results_dir results --output_dir analysis/a5
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import analyze_results as ar  # noqa: E402
import partition_skew  # noqa: E402
from src.evaluation.bootstrap import DEFAULT_B, config_ci  # noqa: E402
from src.evaluation.predictions import load_npz  # noqa: E402

NODES = ("node_a", "node_b", "node_c")
STRATEGIES = ("fedavg", "fedprox_0.01")
TESTBED = "testbed"
DESKTOP = "desktop_sim"
#: (setting, alphas, draws), sec. 3a / 3b
DECLARED = ((TESTBED, ("0.1",), ("ps123", "ps456")),
            (DESKTOP, ("0.1", "0.5", "1"), ("ps42", "ps123", "ps456")))
SETTING_LABEL = {TESTBED: "testbed (MAXN_SUPER, 10 rounds, seed 42)",
                 DESKTOP: "desktop simulation sensitivity analysis"}
ROUNDS, SEED = 10, 42
METRIC = "balanced_accuracy"
#: a run's metrics recomputed from its prediction files must match its own log
LOGGED_TOLERANCE = 1e-5


def partition_of(alpha: str, draw: str) -> str:
    """The data/splits manifest of a draw: ps42 is the main study's ``dirichlet_<alpha>``."""
    return "dirichlet_%s" % alpha if draw == "ps42" else "dirichlet_%s_%s" % (alpha, draw)


def run_dir_of(results_dir: str, setting: str, alpha: str, draw: str, strategy: str) -> str:
    if setting == TESTBED:
        return os.path.join(results_dir, "pc_maxn", "rev_dirichlet%s_%s_%s_r%d_seed%d"
                            % (alpha, draw, strategy, ROUNDS, SEED))
    return os.path.join(results_dir, "desktop_sim", "sim_dirichlet%s_%s_%s_r%d_seed%d"
                        % (alpha, draw, strategy, ROUNDS, SEED))


def declared_runs(results_dir: str) -> List[Dict[str, str]]:
    """Every declared A5 run, in table order (setting, alpha, strategy, draw)."""
    out = []
    for setting, alphas, draws in DECLARED:
        for alpha in alphas:
            for strategy in STRATEGIES:
                for draw in draws:
                    out.append({"setting": setting, "alpha": alpha, "draw": draw,
                                "strategy": strategy, "partition": partition_of(alpha, draw),
                                "run_dir": run_dir_of(results_dir, setting, alpha, draw,
                                                      strategy)})
    return out


def check_run(spec: Dict[str, str], data: Dict[str, Any]) -> None:
    """The run is the declared one and finished cleanly; raises ValueError otherwise."""
    name = os.path.basename(spec["run_dir"])
    if int(data.get("num_rounds", -1)) != ROUNDS or int(data.get("seed", -1)) != SEED:
        raise ValueError("%s: not %d rounds / seed %d" % (name, ROUNDS, SEED))
    if spec["setting"] == TESTBED:
        power = data.get("power") or {}
        modes = [m for k in ("measured_before", "measured_after")
                 for m in (power.get(k) or {}).values()]
        if power.get("power_config") != "maxn" or len(modes) != 6 \
                or any(m != "MAXN_SUPER" for m in modes):
            raise ValueError("%s: not run at MAXN_SUPER on all three nodes" % name)
    else:
        path = os.path.join(spec["run_dir"], "sim_info.json")
        with open(path, encoding="utf-8") as fh:
            info = json.load(fh)
        rcs = [info.get("server_rc")] + list((info.get("client_rc") or {}).values())
        if info.get("ok") is not True or len(rcs) != 4 or any(rc != 0 for rc in rcs):
            raise ValueError("%s: the simulation did not finish cleanly" % name)
        if info.get("partition") != spec["partition"] or info.get("draw") != spec["draw"]:
            raise ValueError("%s: sim_info names another partition" % name)


def prediction_units(record: Dict[str, Any], groups: Dict[str, str]) -> List[Dict[str, Any]]:
    """Per node: labels, logit margins and sequence ids of the selected round's test
    predictions (the declared model selection rule chose the round)."""
    sel = record.get("selected_round")
    if sel is None:
        raise ValueError("%s: no selected round" % record["run_name"])
    pred_dir = os.path.join(record["run_dir"], "predictions")
    units = []
    for node in NODES:
        path = os.path.join(pred_dir, "r%03d_%s_test.npz" % (int(sel), node))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        d = load_npz(path)
        missing = [p for p in d["path"] if p not in groups]
        if missing:
            raise ValueError("%s: %d prediction paths are not in the partition manifest"
                             % (path, len(missing)))
        units.append({"label": d["label"], "margin": d["logit_margin"],
                      "group": np.array([groups[p] for p in d["path"]])})
    return units


def load_groups(splits_dir: str, partition: str) -> Dict[str, str]:
    m = pd.read_csv(os.path.join(splits_dir, "%s.csv.gz" % partition), dtype={"group_id": str})
    return dict(zip(m.path, m.group_id))


def analyse_run(spec: Dict[str, str], splits_dir: str, B: int) -> Dict[str, Any]:
    with open(os.path.join(spec["run_dir"], "results.json"), encoding="utf-8") as fh:
        check_run(spec, json.load(fh))
    record = ar.load_run(spec["run_dir"], warn=False)
    if record.get("strategy") != spec["strategy"]:
        raise ValueError("%s: strategy %s" % (record["run_name"], record.get("strategy")))
    units = prediction_units(record, load_groups(splits_dir, spec["partition"]))
    # the run logged its own (test-size-weighted) metrics: the predictions must give them back
    native = ar._aggregate_point(units, "weighted")
    logged = record.get("selected_test_metrics") or {}
    diffs = [abs(native[m] - float(logged[m])) for m in ("accuracy", METRIC, "mcc")
             if m in native and logged.get(m) is not None]
    if not diffs or max(diffs) > LOGGED_TOLERANCE:
        raise ValueError("%s: predictions do not reproduce the logged metrics (%s)"
                         % (record["run_name"], diffs))
    key = "a5|%s|%s|%s" % (spec["setting"], spec["partition"], spec["strategy"])
    pooled_ci = config_ci([ar._units_for(units, "pooled")], "pooled", key, B=B)[METRIC]
    mean_ci = config_ci([ar._units_for(units, "mean")], "mean", key + "|clientmean", B=B)[METRIC]
    row = dict(spec)
    row.pop("run_dir")
    row.update({
        "label": SETTING_LABEL[spec["setting"]], "run_name": record["run_name"],
        "selected_round": int(record["selected_round"]),
        "n_test": int(sum(len(u["label"]) for u in units)),
        "ba_pooled": ar._aggregate_point(units, "pooled")[METRIC],
        "ba_pooled_ci_low": pooled_ci["ci_low"], "ba_pooled_ci_high": pooled_ci["ci_high"],
        "ba_clientmean": ar._aggregate_point(units, "mean")[METRIC],
        "ba_clientmean_ci_low": mean_ci["ci_low"], "ba_clientmean_ci_high": mean_ci["ci_high"],
        "B": B, "max_diff_vs_logged": max(diffs),
    })
    # timing rule (CLAUDE.md rule 12), testbed only: the desktop simulation ran several
    # cells in parallel on one GPU, so its wall-clock is not a testbed time
    for k in ("round1_time_s", "steady_round_time_s", "total_time_s"):
        row[k] = record.get(k) if spec["setting"] == TESTBED else None
    return row


def spread(runs: pd.DataFrame) -> pd.DataFrame:
    """Range and SD over the draws of each (setting, alpha, strategy); nothing else."""
    rows = []
    for (setting, alpha, strategy), g in runs.groupby(["setting", "alpha", "strategy"], sort=False):
        row = {"setting": setting, "label": SETTING_LABEL[setting], "alpha": alpha,
               "strategy": strategy, "n_draws": len(g), "draws": " ".join(g["draw"])}
        for col in ("ba_pooled", "ba_clientmean"):
            v = g[col].astype(float)
            row.update({col + "_min": v.min(), col + "_max": v.max(),
                        col + "_range": v.max() - v.min(), col + "_sd": v.std(ddof=1)})
        rows.append(row)
    return pd.DataFrame(rows)


def skew(stats_path: str, partitions: List[str]) -> pd.DataFrame:
    """The measured skew of every A5 draw (partitions, not results, A5-1 (c))."""
    with open(stats_path, encoding="utf-8") as fh:
        stats = json.load(fh)
    missing = [p for p in partitions if p not in stats]
    if missing:
        raise ValueError("no split statistics for %s" % missing)
    rows = partition_skew.describe({p: stats[p] for p in partitions})
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results_dir", default=os.path.join(REPO, "results"))
    ap.add_argument("--splits_dir", default=os.path.join(REPO, "data", "splits"))
    ap.add_argument("--output_dir", default=os.path.join(REPO, "analysis", "a5"))
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    args = ap.parse_args(argv)

    specs = declared_runs(args.results_dir)
    missing = [s["run_dir"] for s in specs if not os.path.isdir(s["run_dir"])]
    if missing:
        raise SystemExit("declared A5 runs missing: %s" % missing)
    runs = pd.DataFrame([analyse_run(s, args.splits_dir, args.B) for s in specs])
    parts = list(dict.fromkeys(s["partition"] for s in specs))
    os.makedirs(args.output_dir, exist_ok=True)
    runs.to_csv(os.path.join(args.output_dir, "a5_runs.csv"), index=False, lineterminator="\n")
    spread(runs).to_csv(os.path.join(args.output_dir, "a5_spread.csv"), index=False,
                        lineterminator="\n")
    skew(os.path.join(args.splits_dir, "split_stats.json"), parts).to_csv(
        os.path.join(args.output_dir, "a5_skew.csv"), index=False, lineterminator="\n")
    print("A5: %d runs (%d testbed, %d desktop simulation), %d partitions -> %s"
          % (len(runs), int((runs.setting == TESTBED).sum()), int((runs.setting == DESKTOP).sum()),
             len(parts), args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
