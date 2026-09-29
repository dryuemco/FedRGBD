#!/usr/bin/env python3
"""FedRGBD -- camera experiment, question (b): federated training with sensor-skewed clients.

Implements ``docs/CAMERA_EXPERIMENT_PREREG.md`` section 6 (with the machinery of
section 4) and nothing else.  Read that file first.

Per (method, seed) the test predictions of the model selected by the declared rule are
concatenated over the five scene folds, so every scene counts once:

* FedAvg, FedProx(0.01): the global model on every client's test split, FedBN: each
  client's own model on its own split -- in both cases exactly the FL run's
  ``predictions/r<selected round>_<node>_test.npz``;
* local-only: each node's own model on its own split, centralized: the pooled model on
  all three clients' test splits -- ``predictions/selected_<node>_test.npz``
  (``scripts/predict_from_checkpoint.py``).

The selected round / epoch and the prediction files are located with the loaders of
``scripts/analyze_results.py`` (``load_run`` + ``_prediction_files``), the same code the
FLAME analysis uses.  Every method must cover the identical image set -- every valid frame
of ``labels.csv``, each in the fold whose test scenes contain its scene -- which is checked
image by image and with ``bootstrap.check_same_set``.

Statistic: balanced accuracy, pooled over the concatenated images (primary), and the
unweighted mean over the three clients (secondary).  Cluster bootstrap with the **scene**
as the cluster, B = 10,000, seeds resampled jointly with the scenes
(``bootstrap.seed_paired_diff_ci``; for the client mean the same procedure with one
shared scene resample applied to each client's own images).

Families (Holm m = 3 each, never pooled with each other):

* B-local:   D = FL_f - local-only    "federation improves on training each sensor alone" /
             "federation worse than training each sensor alone" / "no detectable difference"
* B-central: D = FL_f - centralized   "higher than centralized" / "lower than centralized" /
             "no detectable difference"

Verdicts: the interval verdict (L > 0 positive, U < 0 negative, otherwise "no detectable
difference") and the Holm verdict (Holm-adjusted bootstrap p < 0.05, direction of D); the
Holm verdict is the headline.  "No detectable difference" is never "equivalent".

    python scripts/camera_analysis_b.py                      # -> analysis/camera/b/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.evaluation.bootstrap import (BASE_SEED, Unit, check_same_set,  # noqa: E402
                                      metrics_from_counts, seed_paired_diff_ci, shared_set_ci)
from src.evaluation.predictions import load_npz, metrics_from_predictions  # noqa: E402

NODES = ("node_a", "node_b", "node_c")
METRIC = "balanced_accuracy"
B = 10000
ALPHA = 0.05
N_FOLDS = 5

#: (label, --strategy) of the federated methods, in the prereg's order
FL_METHODS = (("FedAvg", "fedavg"), ("FedProx(0.01)", "fedprox_0.01"), ("FedBN", "fedbn"))
LOCAL = "Local-only"
CENTRAL = "Centralized"
BASELINE_LABELS = {"local_only": LOCAL, "centralized": CENTRAL}
METHOD_ORDER = [m for m, _ in FL_METHODS] + [LOCAL, CENTRAL]
AGGREGATIONS = ("pooled", "clientmean")          # primary, secondary

NO_DIFFERENCE = "no detectable difference"
#: family -> (reference method, positive phrase, negative phrase); fixed by the prereg
FAMILIES = {
    "B-local": (LOCAL, "federation improves on training each sensor alone",
                "federation worse than training each sensor alone"),
    "B-central": (CENTRAL, "higher than centralized", "lower than centralized"),
}
HOLM_M = len(FL_METHODS)


# --------------------------------------------------------------------------- verdicts
def verdict_from_ci(family: str, ci_low: float, ci_high: float) -> str:
    _, pos, neg = FAMILIES[family]
    if ci_low > 0:
        return pos
    if ci_high < 0:
        return neg
    return NO_DIFFERENCE


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1), in input order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def verdict_from_holm(family: str, diff: float, p_holm: float) -> str:
    _, pos, neg = FAMILIES[family]
    if p_holm < ALPHA:
        return pos if diff > 0 else neg
    return NO_DIFFERENCE


def apply_rule(comps: pd.DataFrame) -> pd.DataFrame:
    """Interval and Holm verdicts; Holm within each (family, aggregation), m = 3."""
    comps = comps.copy()
    if comps.empty:
        return comps
    comps["verdict"] = [verdict_from_ci(f, lo, hi)
                        for f, lo, hi in zip(comps.family, comps.ci_low, comps.ci_high)]
    comps["holm_m"] = 0
    comps["p_holm"] = np.nan
    for _, idx in comps.groupby(["family", "aggregation"]).groups.items():
        if len(idx) != HOLM_M:
            raise ValueError("a family needs exactly %d comparisons (one per strategy), got %d"
                             % (HOLM_M, len(idx)))
        comps.loc[idx, "p_holm"] = holm(comps.loc[idx, "p_boot"].to_numpy())
        comps.loc[idx, "holm_m"] = len(idx)
    comps["verdict_holm"] = [verdict_from_holm(f, d, p) for f, d, p in
                             zip(comps.family, comps["diff"], comps.p_holm)]
    return comps


# --------------------------------------------------------------------------- inputs
def read_scene_table(labels_csv: str, folds_csv: str) -> Tuple[Dict[Tuple[str, str], Tuple[str, str]],
                                                              Dict[str, int]]:
    """({(node, id): (scene, class dir)} of every valid frame, {scene: fold})."""
    from scripts.camera_fl_prepare import CLASS_DIRS, read_folds, read_labels

    folds = read_folds(folds_csv)
    frames = {}
    for r in read_labels(labels_csv):
        if int(r["valid"]):
            frames[(str(r["node"]), str(r["id"]))] = (str(r["scene"]), CLASS_DIRS[str(r["label"])])
    return frames, folds


def expected_runs(config_path: str) -> Dict[str, Dict[Tuple[int, int], str]]:
    """{method: {(fold, seed): run dir relative to the repo}} of the camera block and its
    desktop baselines, exactly as print_revision_commands.py emits them."""
    from scripts import print_revision_commands as prc

    revision = prc.load_config(config_path)["revision"]
    out: Dict[str, Dict[Tuple[int, int], str]] = {m: {} for m in METHOD_ORDER}
    labels = dict((s, m) for m, s in FL_METHODS)
    for run in prc.expand_all(revision, prc.CAMERA_BLOCK)[prc.CAMERA_BLOCK]:
        out[labels[run.strategy]][(_fold_of(run.split), int(run.seed))] = run.output_dir
    for run in prc.expand_all(revision, prc.CAMERA_BASELINE_BLOCK)[prc.CAMERA_BASELINE_BLOCK]:
        out[BASELINE_LABELS[run.baseline_type]][(_fold_of(run.split), int(run.seed))] = run.output_dir
    return out


def _fold_of(split: str) -> int:
    if not split.startswith("camera_fold"):
        raise ValueError(split)
    return int(split[len("camera_fold"):])


def selected_test_files(run_dir: str) -> Dict[str, str]:
    """{node: test-prediction file of the model the declared rule selects} of one run,
    located by analyze_results (selected round for FL, selected epoch for baselines)."""
    from scripts.analyze_results import HEADLINE_SELECTED, _prediction_files, load_run

    record = load_run(run_dir, warn=False)
    if record.get("headline_source") != HEADLINE_SELECTED:
        raise ValueError("%s: no model selected by the declared rule (headline %s)"
                         % (run_dir, record.get("headline_source")))
    _, files = _prediction_files(record)
    out = {}
    for path in files:
        name = os.path.basename(path)
        node = name.split("_", 1)[1][:-len("_test.npz")]      # r010_node_a_test / selected_node_a_test
        out[node] = path
    if sorted(out) != sorted(NODES):
        hint = "" if record["kind"] == "fl" else \
            " -- run scripts/predict_from_checkpoint.py %s" % run_dir
        raise ValueError("%s: test predictions for %s, need %s%s"
                         % (run_dir, sorted(out) or "no node", list(NODES), hint))
    return out


def load_method_seed(files_by_fold: Dict[int, Dict[str, str]],
                     frames: Dict[Tuple[str, str], Tuple[str, str]],
                     folds: Dict[str, int]) -> Dict[str, Dict[str, np.ndarray]]:
    """Concatenate one (method, seed)'s five folds -> {node: {key, label, margin, scene}}.

    Every image must be a valid frame of its node whose scene is a test scene of the fold
    it came from, with the declared label; no image may occur twice.
    """
    if sorted(files_by_fold) != list(range(N_FOLDS)):
        raise ValueError("need the five folds, got %s" % sorted(files_by_fold))
    out = {}
    for node in NODES:
        keys, labels, margins, scenes = [], [], [], []
        for fold in range(N_FOLDS):
            path = files_by_fold[fold][node]
            d = load_npz(path)
            for p, y in zip(d["path"], d["label"]):
                cls, fname = str(p).split("/", 1)
                fid = fname[:-len(".png")] if fname.endswith(".png") else fname
                info = frames.get((node, fid))
                if info is None:
                    raise ValueError("%s: %s is not a valid %s frame of labels.csv" % (path, p, node))
                scene, want_cls = info
                if folds[scene] != fold:
                    raise ValueError("%s: %s (scene %s, fold %d) is not a test image of fold %d"
                                     % (path, p, scene, folds[scene], fold))
                if cls != want_cls or int(y) != (1 if want_cls == "Fire" else 0):
                    raise ValueError("%s: %s label disagrees with labels.csv" % (path, p))
                keys.append(fid)
                scenes.append(scene)
            labels.append(d["label"].astype(np.int64))
            margins.append(d["logit_margin"].astype(np.float64))
        keys = np.asarray(keys, dtype=str)
        if len(set(keys.tolist())) != len(keys):
            raise ValueError("%s: an image occurs in more than one fold's test predictions" % node)
        order = np.argsort(keys, kind="mergesort")
        out[node] = {"key": keys[order], "label": np.concatenate(labels)[order],
                     "margin": np.concatenate(margins)[order],
                     "scene": np.asarray(scenes, dtype=str)[order]}
    return out


def check_image_set(preds: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]],
                    frames: Dict[Tuple[str, str], Tuple[str, str]]) -> Dict[str, int]:
    """Every (method, seed) must cover exactly the valid frames of labels.csv, node by node.
    -> {node: images}."""
    want = {n: sorted(fid for (node, fid) in frames if node == n) for n in NODES}
    for method, by_seed in preds.items():
        for seed, nodes in by_seed.items():
            for n in NODES:
                if nodes[n]["key"].tolist() != want[n]:
                    raise ValueError("%s seed %d, %s: test predictions cover %d images, labels.csv "
                                     "has %d valid frames -- not the same image set"
                                     % (method, seed, n, len(nodes[n]["key"]), len(want[n])))
    return {n: len(want[n]) for n in NODES}


# --------------------------------------------------------------------------- units
def pooled_unit(nodes: Dict[str, Dict[str, np.ndarray]]) -> Unit:
    """All three clients' images, clustered by scene (a scene resample takes the scene on
    every camera at once)."""
    return Unit(np.concatenate([nodes[n]["label"] for n in NODES]),
                np.concatenate([nodes[n]["margin"] for n in NODES]),
                np.concatenate([nodes[n]["scene"] for n in NODES]))


def client_units(nodes: Dict[str, Dict[str, np.ndarray]]) -> List[Unit]:
    return [Unit(nodes[n]["label"], nodes[n]["margin"], nodes[n]["scene"]) for n in NODES]


def _check_client_runs(runs: Sequence[Sequence[Unit]]) -> None:
    """Client-mean runs: every client unit has the same scenes, and client j covers the
    same images (per scene and class) in every run."""
    gid = runs[0][0].gid
    for run in runs:
        if len(run) != len(NODES) or any(not np.array_equal(u.gid, gid) for u in run):
            raise ValueError("every client must hold every scene")
    for j in range(len(NODES)):
        check_same_set([run[j] for run in runs])


def _clientmean_value(run: Sequence[Unit], W: np.ndarray, metric: str) -> np.ndarray:
    """Unweighted mean over the clients of the metric under scene weights W (B, G)."""
    vals = []
    for u in run:
        tp, fp, fn, tn = (W @ u.counts).T
        vals.append(metrics_from_counts(tp, fp, fn, tn)[metric])
    return np.mean(vals, axis=0)


def _pct(values: np.ndarray, level: float = 0.95) -> Dict[str, float]:
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return {"ci_low": float(np.percentile(values, lo)), "ci_high": float(np.percentile(values, hi))}


def clientmean_paired_diff_ci(runs_a: List[List[Unit]], runs_b: List[List[Unit]], key: str,
                              metric: str = METRIC, B: int = B) -> Dict[str, float]:
    """``seed_paired_diff_ci`` for the unweighted client mean: one stratified scene resample
    per replicate, applied to every client of both sides, one seed draw applied to both
    sides; same interval and p-value definitions."""
    if not runs_a or len(runs_a) != len(runs_b):
        raise ValueError("need the same, non-zero number of runs on both sides (one per seed)")
    _check_client_runs(list(runs_a) + list(runs_b))
    rng = np.random.default_rng(BASE_SEED + zlib.crc32(("seedpaired|clientmean|" + key)
                                                       .encode("utf-8")))
    ref = runs_a[0][0]
    ones = np.ones((1, ref.G))
    W = ref.weights(B, rng)
    point = float(np.mean([_clientmean_value(a, ones, metric)[0] - _clientmean_value(b, ones, metric)[0]
                           for a, b in zip(runs_a, runs_b)]))
    d = np.stack([_clientmean_value(a, W, metric) - _clientmean_value(b, W, metric)
                  for a, b in zip(runs_a, runs_b)])
    S = len(runs_a)
    pick = rng.integers(0, S, size=(B, S))
    rep = d[pick, np.arange(B)[:, None]].mean(axis=1)
    p = min(1.0, 2 * min(int((rep <= 0).sum()) + 1, int((rep >= 0).sum()) + 1) / (B + 1))
    out = {"diff": point, "p_boot": p, "n_pairs": S, "B": B}
    out.update(_pct(rep))
    return out


def clientmean_ci(runs: List[List[Unit]], key: str, metric: str = METRIC,
                  B: int = B) -> Dict[str, float]:
    """``shared_set_ci`` for the unweighted client mean (seed x scene bootstrap)."""
    _check_client_runs(runs)
    rng = np.random.default_rng(BASE_SEED + zlib.crc32(("clientmean|" + key).encode("utf-8")))
    ones = np.ones((1, runs[0][0].G))
    point = float(np.mean([_clientmean_value(r, ones, metric)[0] for r in runs]))
    per_run = np.stack([_clientmean_value(r, r[0].weights(B, rng), metric) for r in runs])
    S = len(runs)
    pick = rng.integers(0, S, size=(B, S))
    col = (np.arange(B)[:, None] * S + np.arange(S)[None, :]) % B
    rep = per_run[pick, col].mean(axis=1)
    out = {"mean": point, "se": float(np.std(rep, ddof=1)), "n_runs": S, "B": B}
    out.update(_pct(rep))
    return out


# --------------------------------------------------------------------------- analysis
def load_all(root: str, config_path: str, labels_csv: str, folds_csv: str):
    """-> (preds {method: {seed: {node: arrays}}}, frames, folds, run dirs)."""
    frames, folds = read_scene_table(labels_csv, folds_csv)
    runs = expected_runs(config_path)
    outside = [d for by in runs.values() for d in by.values()
               if not d.replace("\\", "/").startswith("results/camera/")]
    if outside:
        raise SystemExit("camera analyses read only results/camera/, but the config places "
                         "%d run(s) elsewhere (first: %s)" % (len(outside), outside[0]))
    missing = [d for by in runs.values() for d in by.values()
               if not os.path.isdir(os.path.join(root, d))]
    if missing:
        raise SystemExit("%d of the camera runs are missing (first: %s); question (b) is analysed "
                         "only when all 45 federated runs and 30 baselines exist"
                         % (len(missing), missing[0]))
    preds: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]] = {}
    for method in METHOD_ORDER:
        seeds = sorted({s for (_, s) in runs[method]})
        preds[method] = {}
        for seed in seeds:
            files = {f: selected_test_files(os.path.join(root, runs[method][(f, seed)]))
                     for f in range(N_FOLDS)}
            preds[method][seed] = load_method_seed(files, frames, folds)
    return preds, frames, folds, runs


def _ba(unit: Unit) -> float:
    ones = np.ones((1, unit.G))
    tp, fp, fn, tn = (ones @ unit.counts)[0]
    return float(metrics_from_counts(tp, fp, fn, tn)[METRIC])


def analyze(preds: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]], B: int = B):
    """-> (per-run rows, per-method rows, comparison rows) as DataFrames (no verdicts yet)."""
    pooled = {m: {s: [pooled_unit(n)] for s, n in by.items()} for m, by in preds.items()}
    clients = {m: {s: client_units(n) for s, n in by.items()} for m, by in preds.items()}
    check_same_set([u for by in pooled.values() for run in by.values() for u in run])
    _check_client_runs([run for by in clients.values() for run in by.values()])

    run_rows, method_rows, comp_rows = [], [], []
    for method in METHOD_ORDER:
        for seed, nodes in sorted(preds[method].items()):
            label = np.concatenate([nodes[n]["label"] for n in NODES])
            margin = np.concatenate([nodes[n]["margin"] for n in NODES])
            m = metrics_from_predictions(label, margin)
            row = {"method": method, "seed": seed, "n_images": int(len(label)),
                   "pooled_balanced_accuracy": float(m["balanced_accuracy"]),
                   "pooled_mcc": float(m["mcc"]), "pooled_accuracy": float(m["accuracy"])}
            per = {n: metrics_from_predictions(nodes[n]["label"], nodes[n]["margin"]) for n in NODES}
            row["clientmean_balanced_accuracy"] = float(np.mean([per[n][METRIC] for n in NODES]))
            for n in NODES:
                row["%s_balanced_accuracy" % n] = float(per[n][METRIC])
            run_rows.append(row)
        seeds = sorted(preds[method])
        ci = shared_set_ci([pooled[method][s] for s in seeds], "camera_b|%s|pooled" % method, B=B)
        method_rows.append({"method": method, "aggregation": "pooled", "seeds": " ".join(map(str, seeds)),
                            "n_seeds": ci["n_runs"], "mean": ci["mean"], "ci_low": ci["ci_low"],
                            "ci_high": ci["ci_high"], "se": ci["se"], "B": ci["B"]})
        ci = clientmean_ci([clients[method][s] for s in seeds], "camera_b|%s" % method, B=B)
        method_rows.append({"method": method, "aggregation": "clientmean",
                            "seeds": " ".join(map(str, seeds)), "n_seeds": ci["n_runs"],
                            "mean": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                            "se": ci["se"], "B": ci["B"]})

    for family, (ref, _, _) in FAMILIES.items():
        for aggregation in AGGREGATIONS:
            for fl, _ in FL_METHODS:
                seeds = sorted(set(preds[fl]) & set(preds[ref]))
                if not seeds:
                    raise ValueError("%s and %s share no seed" % (fl, ref))
                key = "camera_b|%s|%s-%s|%s" % (family, fl, ref, aggregation)
                if aggregation == "pooled":
                    res = seed_paired_diff_ci([pooled[fl][s] for s in seeds],
                                              [pooled[ref][s] for s in seeds], key, B=B)
                else:
                    res = clientmean_paired_diff_ci([clients[fl][s] for s in seeds],
                                                    [clients[ref][s] for s in seeds], key, B=B)
                comp_rows.append({"family": family, "aggregation": aggregation,
                                  "strategy": fl, "reference": ref,
                                  "paired_seeds": " ".join(map(str, seeds)),
                                  "n_pairs": res["n_pairs"], "diff": res["diff"],
                                  "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                                  "p_boot": res["p_boot"], "B": res["B"]})
    return pd.DataFrame(run_rows), pd.DataFrame(method_rows), pd.DataFrame(comp_rows)


def comparisons_markdown(comps: pd.DataFrame, methods: pd.DataFrame, n_images: Dict[str, int],
                         n_scenes: int) -> str:
    lines = ["# Camera experiment, question (b): sensor-skewed clients", "",
             "docs/CAMERA_EXPERIMENT_PREREG.md section 6. Balanced accuracy of the model "
             "selected by the declared rule, test predictions of the five scene folds "
             "concatenated (%d scenes; %s). Pooled over all images = primary; unweighted mean "
             "over the three clients = secondary. 95 %% scene-cluster bootstrap, seeds "
             "resampled jointly, B = %d. Verdict: interval rule; Holm: adjusted p < %.2f within "
             "the family (m = %d); the Holm verdict is the headline. \"No detectable "
             "difference\" is absence of evidence, never equivalence."
             % (n_scenes, ", ".join("%s %d images" % kv for kv in n_images.items()),
                int(comps["B"].iloc[0]) if len(comps) else B, ALPHA, HOLM_M), "",
             "## Methods", "",
             "| method | aggregation | seeds | mean | 95 % CI |", "|---|---|---|---|---|"]
    for r in methods.itertuples():
        lines.append("| %s | %s | %s | %.1f | [%.1f, %.1f] |" % (
            r.method, r.aggregation, r.seeds, 100 * r.mean, 100 * r.ci_low, 100 * r.ci_high))
    lines.append("")
    for (family, aggregation), g in comps.groupby(["family", "aggregation"], sort=False):
        lines += ["## %s, %s (%s; Holm m = %d)"
                  % (family, aggregation, "primary" if aggregation == "pooled" else "secondary",
                     len(g)), "",
                  "| strategy | reference | seeds | diff | 95 % CI | p | p Holm | verdict | verdict (Holm) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            lines.append("| %s | %s | %s | %+.1f | [%+.1f, %+.1f] | %.4f | %.4f | %s | **%s** |"
                         % (r.strategy, r.reference, r.paired_seeds, 100 * r.diff,
                            100 * r.ci_low, 100 * r.ci_high, r.p_boot, r.p_holm, r.verdict,
                            r.verdict_holm))
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default=REPO, help="repository root the run directories are relative to")
    ap.add_argument("--config", default=None, help="default: <root>/configs/experiment_matrix.yaml")
    ap.add_argument("--labels_csv", default=None, help="default: <root>/data/raw/camera/labels.csv")
    ap.add_argument("--folds_csv", default=None, help="default: <root>/data/splits_camera/fl_folds.csv")
    ap.add_argument("--output_dir", default=None, help="default: <root>/analysis/camera/b")
    ap.add_argument("--B", type=int, default=B)
    args = ap.parse_args(argv)
    from scripts.camera_labels import refuse_pilot
    refuse_pilot(args.labels_csv, args.folds_csv)
    root = os.path.abspath(args.root)
    config = args.config or os.path.join(root, "configs", "experiment_matrix.yaml")
    labels_csv = args.labels_csv or os.path.join(root, "data", "raw", "camera", "labels.csv")
    folds_csv = args.folds_csv or os.path.join(root, "data", "splits_camera", "fl_folds.csv")
    out_dir = args.output_dir or os.path.join(root, "analysis", "camera", "b")

    preds, frames, folds, _ = load_all(root, config, labels_csv, folds_csv)
    n_images = check_image_set(preds, frames)
    runs, methods, comps = analyze(preds, B=args.B)
    comps = apply_rule(comps)
    scenes = sorted({s for (s, _) in frames.values()})
    os.makedirs(out_dir, exist_ok=True)
    runs.to_csv(os.path.join(out_dir, "camera_b_runs.csv"), index=False)
    methods.to_csv(os.path.join(out_dir, "camera_b_methods.csv"), index=False)
    comps.to_csv(os.path.join(out_dir, "camera_b_comparisons.csv"), index=False)
    with open(os.path.join(out_dir, "camera_b_comparisons.md"), "w", encoding="utf-8") as f:
        f.write(comparisons_markdown(comps, methods, n_images, len(scenes)))
    keys = "\n".join("%s/%s" % k for k in sorted(frames)).encode("utf-8")
    with open(os.path.join(out_dir, "camera_b_image_set.json"), "w", encoding="utf-8") as f:
        json.dump({"n_images": n_images, "n_scenes": len(scenes), "scenes": scenes,
                   "image_set_sha256": hashlib.sha256(keys).hexdigest(), "B": args.B}, f, indent=2)
    print("question (b): %d runs, %d comparisons -> %s" % (len(runs), len(comps), out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
