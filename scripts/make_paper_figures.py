#!/usr/bin/env python3
"""FedRGBD -- the paper's revision figures, generated from analysis/ and the committed
prediction files (CLAUDE.md rule 4: no number is typed by hand).

    python scripts/make_paper_figures.py --analysis_dir analysis --results_dir results \
        --output_dir paper/figures

Writes (PDF):

* ``fig_confmat.pdf`` (fig:confmat) -- manual label skew, main matrix (three rounds),
  FedAvg and FedProx(mu=0.01): each node's test confusion matrix at the selected round,
  summed over the seeds and normalised by the true class
  (``analysis/per_client_selected.csv``, ``selected_test_tp/fn/fp/tn``).
* ``fig_sensitivity.pdf`` (fig:sensitivity) -- label skew, three rounds: selected-round
  test balanced accuracy (pooled, sequence-bootstrap 95 % CI) against (a) mu (FedProx;
  FedAvg = mu 0 drawn as a reference line), (b) local epochs E and (c) learning rate eta,
  for FedAvg and FedProx(0.01).
* ``fig_mu_tradeoff.pdf`` (fig:mu_tradeoff) -- label skew: round-1 aggregated validation
  accuracy and selected-round test balanced accuracy against mu (revision runs).
* ``fig_dirichlet.pdf`` (fig:dirichlet) -- the three Dirichlet partitions as categories in
  order of measured skew (``analysis/partition_skew.csv``, size-weighted JSD), balanced
  accuracy pooled with its interval per method; no line joins partitions (one draw per
  alpha, so there is no alpha axis).
* ``fig_timecomm.pdf`` (fig:timecomm) -- label skew, main matrix, FedAvg and FedProx(0.01):
  pooled test balanced accuracy per round (from the per-round prediction files; test
  metrics are computed every round for logging and never select a model) against round,
  elapsed wall-clock time and cumulative megabytes (``analysis/per_round_table.csv``);
  mean over seeds, every seed drawn thin.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42})

BA = "selected_test_balanced_accuracy"
STRATEGY_STYLE = {"fedavg": ("FedAvg", "o", "C0"), "fedprox_0.01": ("FedProx ($\\mu$=0.01)", "s", "C1"),
                  "centralized": ("Centralized", "^", "C2"), "local_only": ("Local-only", "v", "C3")}
MU_GRID = (0.001, 0.01, 0.05, 0.1, 0.5)
NODES = ("node_a", "node_b", "node_c")
NODE_TITLE = {"node_a": "Node A (80% fire)", "node_b": "Node B (88.5% fire)",
              "node_c": "Node C (20% fire)"}


def cid_fl(strategy: str, dist: str, rounds: int = 3, epochs: int = 5, lr: str = "0.001") -> str:
    return "group|fl|%s|%s|%d|%d|%s|3" % (strategy, dist, rounds, epochs, lr)


def ref_cid(kind: str, dist: str) -> str:
    return {"centralized": "group|centralized|centralized|%s|3|15|0.001|3",
            "local_only": "group|local|local_only|%s|3|15||3"}[kind] % dist


def row(summary: pd.DataFrame, config_id: str, metric: str) -> Optional[pd.Series]:
    sel = summary[(summary["config_id"] == config_id) & (summary["metric"] == metric)]
    if len(sel) > 1:
        raise ValueError("%d rows for %s/%s" % (len(sel), config_id, metric))
    return None if sel.empty else sel.iloc[0]


def _err(r: pd.Series) -> Tuple[float, List[List[float]]]:
    m = float(r["mean"])
    return m, [[m - float(r["ci_low"])], [float(r["ci_high"]) - m]]


# --------------------------------------------------------------------------- figures
def fig_confmat(per_client: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
    for i, strategy in enumerate(("fedavg", "fedprox_0.01")):
        cid = cid_fl(strategy, "non_iid_label")
        for j, node in enumerate(NODES):
            sub = per_client[(per_client["config_id"] == cid) & (per_client["node"] == node)]
            # mean over seeds x n_seeds = the sum of the per-seed counts
            c = {k: float(sub[sub["metric"] == "selected_test_" + k]["mean"].iloc[0])
                 * int(sub[sub["metric"] == "selected_test_" + k]["n_seeds"].iloc[0])
                 for k in ("tp", "fn", "fp", "tn")}
            m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
            norm = m / m.sum(axis=1, keepdims=True)
            ax = axes[i, j]
            ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, "%.2f" % norm[a, b], ha="center", va="center",
                            color="white" if norm[a, b] > 0.6 else "black", fontsize=8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["no fire", "fire"], fontsize=7)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["no fire", "fire"], fontsize=7)
            if i == 1:
                ax.set_xlabel("predicted", fontsize=8)
            if j == 0:
                ax.set_ylabel("%s\ntrue" % STRATEGY_STYLE[strategy][0], fontsize=8)
            if i == 0:
                ax.set_title(NODE_TITLE[node], fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_sensitivity(summary: pd.DataFrame, path: str) -> None:
    d = "non_iid_label"
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    ax = axes[0]
    mus, means, errs = [], [], [[], []]
    for mu in MU_GRID:
        r = row(summary, cid_fl("fedprox_%g" % mu, d), BA)
        if r is not None:
            m, e = _err(r)
            mus.append(mu), means.append(m), errs[0].append(e[0][0]), errs[1].append(e[1][0])
    ax.errorbar(mus, means, yerr=errs, fmt="s", color="C1", capsize=2, label="FedProx")
    r = row(summary, cid_fl("fedavg", d), BA)
    if r is not None:
        ax.axhline(float(r["mean"]), color="C0", lw=1, label="FedAvg ($\\mu$=0)")
        ax.axhspan(float(r["ci_low"]), float(r["ci_high"]), color="C0", alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("$\\mu$")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("(a) proximal strength", fontsize=8)
    ax.legend(fontsize=6, loc="lower left")
    for k, (title, xs, cid_of, xlabel, log) in enumerate((
            ("(b) local epochs", (1, 2, 5), lambda s, x: cid_fl(s, d, epochs=x), "$E$", False),
            ("(c) learning rate", (0.0001, 0.001), lambda s, x: cid_fl(s, d, lr="%g" % x),
             "$\\eta$", True))):
        ax = axes[k + 1]
        for off, strategy in ((-0.04, "fedavg"), (0.04, "fedprox_0.01")):
            name, marker, color = STRATEGY_STYLE[strategy]
            px, py, pe = [], [], [[], []]
            for x in xs:
                r = row(summary, cid_of(strategy, x), BA)
                if r is None:
                    continue
                m, e = _err(r)
                px.append(x * (10 ** off) if log else x + off * 5)
                py.append(m), pe[0].append(e[0][0]), pe[1].append(e[1][0])
            ax.errorbar(px, py, yerr=pe, fmt=marker, color=color, capsize=2, label=name)
        if log:
            ax.set_xscale("log")
        else:
            ax.set_xticks(xs)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=8)
        if k == 0:
            ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_mu_tradeoff(summary: pd.DataFrame, path: str) -> None:
    d = "non_iid_label"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    xs = (0.0,) + MU_GRID
    for ax, metric, ylabel in ((axes[0], "round1_accuracy", "round-1 validation accuracy"),
                               (axes[1], BA, "selected-round test BA")):
        px, py, pe = [], [], [[], []]
        for mu in xs:
            strategy = "fedavg" if mu == 0 else "fedprox_%g" % mu
            r = row(summary, cid_fl(strategy, d), metric)
            if r is None:
                continue
            m, e = _err(r)
            px.append(mu if mu > 0 else MU_GRID[0] / 10)
            py.append(m), pe[0].append(e[0][0]), pe[1].append(e[1][0])
        ax.errorbar(px, py, yerr=pe, fmt="o", color="C1", capsize=2)
        ax.set_xscale("log")
        ax.set_xticks([MU_GRID[0] / 10] + list(MU_GRID))
        ax.set_xticklabels(["0 (FedAvg)"] + ["%g" % m for m in MU_GRID], fontsize=6, rotation=30)
        ax.set_xlabel("$\\mu$")
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_dirichlet(summary: pd.DataFrame, skew: pd.DataFrame, path: str) -> None:
    allp = skew[skew["node"] == "ALL"].set_index("partition")["jsd"]
    parts = sorted((p for p in ("dirichlet_0.1", "dirichlet_0.5", "dirichlet_1") if p in allp),
                   key=lambda p: allp[p])
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    methods = (("centralized", lambda p: ref_cid("centralized", p)),
               ("local_only", lambda p: ref_cid("local_only", p)),
               ("fedavg", lambda p: cid_fl("fedavg", p)),
               ("fedprox_0.01", lambda p: cid_fl("fedprox_0.01", p)))
    for k, (method, cid_of) in enumerate(methods):
        name, marker, color = STRATEGY_STYLE[method]
        px, py, pe = [], [], [[], []]
        for i, p in enumerate(parts):
            r = row(summary, cid_of(p), BA)
            if r is None:
                continue
            m, e = _err(r)
            px.append(i + (k - 1.5) * 0.12)
            py.append(m), pe[0].append(e[0][0]), pe[1].append(e[1][0])
        ax.errorbar(px, py, yerr=pe, fmt=marker, color=color, capsize=2, label=name, ls="none")
    ax.set_xticks(range(len(parts)))
    ax.set_xticklabels(["$\\alpha$=%s\nJSD %.3f" % (p.split("_")[1], allp[p]) for p in parts],
                       fontsize=7)
    ax.set_xlabel("partition, in order of measured skew")
    ax.set_ylabel("balanced accuracy")
    ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _pooled_ba(files: Sequence[str]) -> float:
    """Balanced accuracy pooled over the clients' prediction files, with the repository's
    own metric code (``predictions.metrics_from_predictions``)."""
    from src.evaluation.predictions import load_npz, metrics_from_predictions
    ys, ms = [], []
    for f in files:
        d = load_npz(f)
        ys.append(np.asarray(d["label"]))
        ms.append(np.asarray(d["logit_margin"]))
    return float(metrics_from_predictions(np.concatenate(ys), np.concatenate(ms))["balanced_accuracy"])


def fig_timecomm(per_round: pd.DataFrame, runs: pd.DataFrame, results_dir: str, path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for strategy in ("fedavg", "fedprox_0.01"):
        name, marker, color = STRATEGY_STYLE[strategy]
        cid = cid_fl(strategy, "non_iid_label")
        pr = per_round[per_round["config_id"] == cid].sort_values("round")
        dirs = sorted(runs[runs["config_id"] == cid]["run_dir"].astype(str))
        per_seed = []
        for d in dirs:
            rd = d if os.path.isabs(d) else os.path.join(REPO, d)
            vals = []
            for r in pr["round"].astype(int):
                files = sorted(glob.glob(os.path.join(rd, "predictions", "r%03d_node_*_test.npz" % r)))
                vals.append(_pooled_ba(files) if len(files) == 3 else np.nan)
            per_seed.append(vals)
        per_seed = np.array(per_seed, dtype=float)
        mean = np.nanmean(per_seed, axis=0)
        for ax, x in zip(axes, (pr["round"].values, pr["elapsed_s_mean"].values / 60.0,
                                pr["cumulative_mb_mean"].values)):
            for s in per_seed:
                ax.plot(x, s, color=color, lw=0.5, alpha=0.35)
            ax.plot(x, mean, marker=marker, color=color, lw=1.5, label=name)
    axes[0].set_xlabel("round")
    axes[1].set_xlabel("elapsed wall-clock time (min)")
    axes[2].set_xlabel("cumulative transmitted MB")
    axes[0].set_ylabel("pooled test balanced accuracy")
    axes[0].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--analysis_dir", default="analysis")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--output_dir", default=os.path.join("paper", "figures"))
    args = ap.parse_args(argv)
    a = args.analysis_dir
    summary = pd.read_csv(os.path.join(a, "summary_table.csv"))
    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda n: os.path.join(args.output_dir, n)  # noqa: E731
    fig_confmat(pd.read_csv(os.path.join(a, "per_client_selected.csv")), out("fig_confmat.pdf"))
    fig_sensitivity(summary, out("fig_sensitivity.pdf"))
    fig_mu_tradeoff(summary, out("fig_mu_tradeoff.pdf"))
    fig_dirichlet(summary, pd.read_csv(os.path.join(a, "partition_skew.csv")), out("fig_dirichlet.pdf"))
    fig_timecomm(pd.read_csv(os.path.join(a, "per_round_table.csv")),
                 pd.read_csv(os.path.join(a, "runs.csv")), args.results_dir, out("fig_timecomm.pdf"))
    for n in ("fig_confmat", "fig_sensitivity", "fig_mu_tradeoff", "fig_dirichlet", "fig_timecomm"):
        print("wrote %s" % out(n + ".pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
