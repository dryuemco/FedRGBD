#!/usr/bin/env python3
"""FedRGBD -- camera experiment, question (a): the declared analysis.

Implements ``docs/CAMERA_EXPERIMENT_PREREG.md`` sections 4 and 5 on the out-of-fold
predictions of ``scripts/camera_loso.py``; nothing else.

For an ordered pair X -> Y (X != Y) and seed s, with every balanced accuracy pooled over
all scenes' out-of-fold predictions:

    D(X -> Y) = mean_s [ BA(model X, sensor-Y frames) - BA(model X, sensor-X frames) ]

Both sides cover the same scenes, so one scene resample (cluster = scene) is applied to
both, jointly with one draw of the seeds (``bootstrap.scene_paired_contrast_ci``),
B = 10,000, 95 % percentile interval, p = min(1, 2 min(#{D* <= 0} + 1, #{D* >= 0} + 1) /
(B + 1)).

* Family A-RGB (primary): the six ordered pairs, Holm m = 6.
* Contrast A-hierarchy (primary, single test, no Holm):
  K = mean(D over the four pairs involving the ZED 2i) - mean(D over D435if <-> D435i).
* Family A-RGBD (secondary): the same six pairs with RGB-D input, Holm m = 6.
* IR (descriptive): D435if <-> D435i, D and interval, no verdict.
* Descriptive: the matrix of pooled balanced accuracy with intervals per modality, for the
  leave-one-scene-out predictions and for the random frame-level split.

Verdicts: interval rule (L > 0 positive phrase, U < 0 negative phrase, otherwise
"no detectable difference"); within a family the Holm verdict (Holm-adjusted p < 0.05, in
the direction of the point estimate) is the headline, the interval verdict is reported
alongside.  A "no detectable difference" is absence of evidence, never sensor invariance.

    python scripts/camera_analysis_a.py --results_root results/camera_a \
        --labels_csv data/raw/camera/labels.csv --output_dir analysis/camera/a
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.camera_labels import NODES, SENSORS, read_labels  # noqa: E402
from scripts.camera_loso import EXPERIMENT, IR_NODES, oof_name, target_nodes  # noqa: E402
from scripts.global_evaluation import holm  # noqa: E402
from src.evaluation.bootstrap import (Unit, metrics_from_counts,  # noqa: E402
                                      scene_paired_contrast_ci, shared_set_ci)
from src.evaluation.predictions import load_npz  # noqa: E402

METRIC = "balanced_accuracy"
B = 10000
ALPHA = 0.05
SEEDS = (42, 123, 456)

NO_DIFFERENCE = "no detectable difference"
PAIR_POS = "higher on the target sensor"
PAIR_NEG = "lower on the target sensor"
K_NEG = "the cross-technology shift costs more than the within-family shift"
K_POS = "the within-family shift costs more than the cross-technology shift"

ZED = "node_c"
WITHIN = (("node_a", "node_b"), ("node_b", "node_a"))
CROSS = (("node_a", "node_c"), ("node_b", "node_c"), ("node_c", "node_a"), ("node_c", "node_b"))
PAIRS = WITHIN + CROSS
FAMILIES = (("A-RGB", "rgb", "primary"), ("A-RGBD", "rgb_d", "secondary"))


# --------------------------------------------------------------------------- verdicts
def verdict_from_ci(ci_low: float, ci_high: float, pos: str, neg: str) -> str:
    if ci_low > 0:
        return pos
    if ci_high < 0:
        return neg
    return NO_DIFFERENCE


def verdict_from_holm(diff: float, p_holm: float, pos: str, neg: str) -> str:
    if p_holm < ALPHA:
        return pos if diff > 0 else neg
    return NO_DIFFERENCE


# --------------------------------------------------------------------------- loading
def scene_map(labels_csv: str) -> Dict[Tuple[str, str], str]:
    return {(str(r["node"]), str(r["id"])): str(r["scene"]) for r in read_labels(labels_csv)}


def kept_ids(labels_csv: str) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for r in read_labels(labels_csv):
        if int(r["valid"]):
            out.setdefault(str(r["node"]), set()).add(str(r["id"]))
    return out


def collect(results_root: str) -> Dict[Tuple[str, str, int], str]:
    """(modality, source, seed) -> run directory, from the runner's results.json."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_root, "**", "results.json"),
                                 recursive=True)):
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
        if res.get("experiment") != EXPERIMENT:
            continue
        c = res["config"]
        key = (c["modality"], c["source"], int(c["seed"]))
        if key in runs:
            raise ValueError("two runs for %s: %s and %s" % (key, runs[key], path))
        runs[key] = os.path.dirname(path)
    return runs


def load_units(runs, labels_csv: str, protocol: str = "loso"
               ) -> Dict[Tuple[str, str, str, int], Unit]:
    """(modality, source, target, seed) -> Unit (cluster = scene) of the oof predictions.

    Checks every file covers the kept frames of its target exactly once."""
    scenes = scene_map(labels_csv)
    kept = kept_ids(labels_csv)
    units = {}
    for (modality, source, seed), run_dir in runs.items():
        for target in target_nodes(modality):
            path = os.path.join(run_dir, oof_name(source, target, protocol))
            if not os.path.isfile(path):
                continue
            d = load_npz(path)
            ids = [str(p) for p in d["path"]]
            if len(ids) != len(set(ids)) or not set(ids) <= kept.get(target, set()):
                raise ValueError("%s: predictions are not a subset of the kept frames of %s, "
                                 "each once" % (path, target))
            units[(modality, source, target, seed)] = Unit(
                d["label"].astype(np.int64), d["logit_margin"],
                np.array([scenes[(target, i)] for i in ids]))
    return units


def _seeds(units, modality: str, cells: Sequence[Tuple[str, str]]) -> List[int]:
    have = [set(s for (m, x, y, s) in units if m == modality and (x, y) == cell)
            for cell in cells]
    return sorted(set.intersection(*have)) if have else []


def point_ba(u: Unit) -> Dict[str, float]:
    tp, fp, fn, tn = u.counts.sum(axis=0)
    m = metrics_from_counts(tp, fp, fn, tn)
    return {k: float(m[k]) for k in ("balanced_accuracy", "mcc", "accuracy")}


# --------------------------------------------------------------------------- statistics
def pair_terms(units, modality: str, x: str, y: str, seeds: Sequence[int], coef: float = 1.0):
    return [(coef, [[units[(modality, x, y, s)]] for s in seeds]),
            (-coef, [[units[(modality, x, x, s)]] for s in seeds])]


def pair_rows(units, modality: str, pairs, family: str, B: int) -> List[Dict[str, object]]:
    rows = []
    for x, y in pairs:
        seeds = _seeds(units, modality, [(x, y), (x, x)])
        if not seeds:
            continue
        res = scene_paired_contrast_ci(pair_terms(units, modality, x, y, seeds),
                                       "camera_a|%s|%s|%s->%s" % (family, modality, x, y), B=B)
        rows.append({"family": family, "modality": modality, "source": x, "target": y,
                     "pair": "%s -> %s" % (SENSORS[x], SENSORS[y]),
                     "seeds": " ".join(map(str, seeds)), "n_pairs": res["n_pairs"],
                     "n_scenes": res["n_clusters"], "diff": res["diff"],
                     "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                     "p_boot": res["p_boot"], "B": res["B"]})
    return rows


def apply_family_rule(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Interval verdicts, and Holm over the rows of one family (m = number of rows)."""
    if not rows:
        return rows
    adj = holm([r["p_boot"] for r in rows])
    for r, p in zip(rows, adj):
        r["verdict"] = verdict_from_ci(r["ci_low"], r["ci_high"], PAIR_POS, PAIR_NEG)
        r["holm_m"] = len(rows)
        r["p_holm"] = float(p)
        r["verdict_holm"] = verdict_from_holm(r["diff"], float(p), PAIR_POS, PAIR_NEG)
    return rows


def hierarchy_contrast(units, modality: str = "rgb", B: int = B) -> Optional[Dict[str, object]]:
    """K = mean(D over the 4 pairs involving the ZED 2i) - mean(D over the 2 within-family)."""
    seeds = _seeds(units, modality, list(PAIRS) + [(n, n) for n in NODES])
    if not seeds:
        return None
    terms = []
    for x, y in CROSS:
        terms += pair_terms(units, modality, x, y, seeds, 1.0 / len(CROSS))
    for x, y in WITHIN:
        terms += pair_terms(units, modality, x, y, seeds, -1.0 / len(WITHIN))
    res = scene_paired_contrast_ci(terms, "camera_a|A-hierarchy|%s" % modality, B=B)
    return {"contrast": "A-hierarchy", "modality": modality,
            "definition": "mean D over the four pairs involving the ZED 2i - "
                          "mean D over D435if <-> D435i",
            "seeds": " ".join(map(str, seeds)), "n_pairs": res["n_pairs"],
            "n_scenes": res["n_clusters"], "K": res["diff"], "ci_low": res["ci_low"],
            "ci_high": res["ci_high"], "p_boot": res["p_boot"], "B": res["B"],
            "verdict": verdict_from_ci(res["ci_low"], res["ci_high"], K_POS, K_NEG)}


def matrix_rows(units, protocol: str, B: int) -> List[Dict[str, object]]:
    rows = []
    modalities = sorted({m for (m, _, _, _) in units})
    for modality in modalities:
        for x in NODES:
            for y in NODES:
                seeds = sorted(s for (m, a, b, s) in units if (m, a, b) == (modality, x, y))
                if not seeds:
                    continue
                runs = [[units[(modality, x, y, s)]] for s in seeds]
                ci = shared_set_ci(runs, "camera_a|matrix|%s|%s|%s->%s"
                                   % (protocol, modality, x, y), B=B)
                pts = [point_ba(r[0]) for r in runs]
                rows.append({"protocol": protocol, "modality": modality, "source": x,
                             "target": y, "source_sensor": SENSORS[x],
                             "target_sensor": SENSORS[y],
                             "seeds": " ".join(map(str, seeds)),
                             "n_frames": int(runs[0][0].n.sum()),
                             "n_scenes": int(runs[0][0].G),
                             "balanced_accuracy": ci["mean"], "ci_low": ci["ci_low"],
                             "ci_high": ci["ci_high"],
                             "mcc": float(np.mean([p["mcc"] for p in pts])),
                             "accuracy": float(np.mean([p["accuracy"] for p in pts])),
                             "B": ci["B"]})
    return rows


def analyze(results_root: str, labels_csv: str, B: int = B):
    runs = collect(results_root)
    units = load_units(runs, labels_csv, "loso")
    random_units = load_units(runs, labels_csv, "random")
    pairs = []
    for family, modality, _ in FAMILIES:
        pairs += apply_family_rule(pair_rows(units, modality, PAIRS, family, B))
    ir = pair_rows(units, "ir", (("node_a", "node_b"), ("node_b", "node_a")), "IR", B)
    for r in ir:
        r["verdict"] = "descriptive (no verdict)"
    k = hierarchy_contrast(units, "rgb", B)
    matrix = matrix_rows(units, "loso", B) + matrix_rows(random_units, "random", B)
    return pd.DataFrame(pairs + ir), pd.DataFrame([k] if k else []), pd.DataFrame(matrix)


# --------------------------------------------------------------------------- report
def _pp(v: float) -> str:
    return "%+.1f" % (100 * v)


def markdown(pairs: pd.DataFrame, contrast: pd.DataFrame, matrix: pd.DataFrame, B: int) -> str:
    L = ["# Camera experiment, question (a): leave-one-scene-out", "",
         "Declared in docs/CAMERA_EXPERIMENT_PREREG.md, section 5. D(X -> Y) = mean over seeds "
         "of [BA(model X, sensor-Y frames) - BA(model X, sensor-X frames)], balanced accuracy "
         "pooled over all scenes' out-of-fold predictions, in percentage points; scene "
         "cluster bootstrap applied to both sides, seed-paired, B = %d, 95 %% percentile "
         "interval. Verdict (Holm) is the headline within a family; the interval verdict is "
         "reported alongside. A \"no detectable difference\" is absence of evidence, never "
         "sensor invariance." % B, ""]
    for family, modality, role in FAMILIES:
        g = pairs[pairs.family == family] if len(pairs) else pairs
        L += ["## Family %s (%s, %s input, Holm m = %d)" % (family, role, modality, len(g)), ""]
        if not len(g):
            L += ["No predictions.", ""]
            continue
        L += ["| pair | seeds | D | 95 % CI | p | p Holm | verdict (Holm) | verdict (interval) |",
              "|---|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            L.append("| %s | %s | %s | [%s, %s] | %.4f | %.4f | %s | %s |"
                     % (r.pair, r.seeds, _pp(r.diff), _pp(r.ci_low), _pp(r.ci_high), r.p_boot,
                        r.p_holm, r.verdict_holm, r.verdict))
        counts = g.verdict_holm.value_counts()
        L += ["", "Pairs per Holm verdict: " + "; ".join(
            "%s: %d" % (v, int(counts.get(v, 0))) for v in (PAIR_NEG, NO_DIFFERENCE, PAIR_POS)), ""]
    L += ["## Contrast A-hierarchy (primary, single test, no Holm)", ""]
    if len(contrast):
        r = contrast.iloc[0]
        L += ["K = %s. K = %s pp, 95 %% CI [%s, %s], p = %.4f, seeds %s: **%s**."
              % (r.definition, _pp(r.K), _pp(r.ci_low), _pp(r.ci_high), r.p_boot, r.seeds,
                 r.verdict), ""]
    else:
        L += ["No predictions.", ""]
    L += ["## IR (descriptive, D435if <-> D435i, no verdict)", ""]
    g = pairs[pairs.family == "IR"] if len(pairs) else pairs
    for r in g.itertuples():
        L.append("* %s: D = %s pp, 95 %% CI [%s, %s], seeds %s"
                 % (r.pair, _pp(r.diff), _pp(r.ci_low), _pp(r.ci_high), r.seeds))
    L.append("")
    for protocol, title in (("loso", "leave-one-scene-out"),
                            ("random", "random frame-level split (v1 protocol, scene overlap)")):
        L += ["## Pooled balanced accuracy, %s (descriptive)" % title, ""]
        g = matrix[matrix.protocol == protocol] if len(matrix) else matrix
        if not len(g):
            L += ["No predictions.", ""]
            continue
        for modality in sorted(g.modality.unique()):
            gm = g[g.modality == modality]
            targets = [n for n in NODES if n in set(gm.target)]
            L += ["### %s (rows: model trained on; columns: frames of)" % modality, "",
                  "| source \\ target | " + " | ".join(SENSORS[t] for t in targets) + " |",
                  "|---|" + "---|" * len(targets)]
            for x in [n for n in NODES if n in set(gm.source)]:
                cells = []
                for y in targets:
                    c = gm[(gm.source == x) & (gm.target == y)]
                    cells.append("-" if not len(c) else "%.1f [%.1f, %.1f]"
                                 % (100 * c.balanced_accuracy.iloc[0], 100 * c.ci_low.iloc[0],
                                    100 * c.ci_high.iloc[0]))
                L.append("| %s | %s |" % (SENSORS[x], " | ".join(cells)))
            L.append("")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results_root", default=os.path.join("results", "camera_a"))
    ap.add_argument("--labels_csv", default=os.path.join("data", "raw", "camera", "labels.csv"))
    ap.add_argument("--output_dir", default=os.path.join("analysis", "camera", "a"))
    ap.add_argument("--B", type=int, default=B)
    args = ap.parse_args(argv)
    pairs, contrast, matrix = analyze(args.results_root, args.labels_csv, B=args.B)
    os.makedirs(args.output_dir, exist_ok=True)
    pairs.to_csv(os.path.join(args.output_dir, "pairs.csv"), index=False)
    contrast.to_csv(os.path.join(args.output_dir, "hierarchy_contrast.csv"), index=False)
    matrix.to_csv(os.path.join(args.output_dir, "ba_matrix.csv"), index=False)
    with open(os.path.join(args.output_dir, "camera_a.md"), "w", encoding="utf-8") as f:
        f.write(markdown(pairs, contrast, matrix, args.B))
    print("wrote %d pair rows, %d contrast, %d matrix cells to %s"
          % (len(pairs), len(contrast), len(matrix), args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
