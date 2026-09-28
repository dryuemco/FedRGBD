"""FedRGBD — sequence-level (cluster) bootstrap confidence intervals for held-out metrics.

Held-out images are not independent: a few sequences (``group_id``) can make up
most of a node's validation or test set.  Resampling images would ignore that and
give intervals that are far too narrow, so the unit of resampling is the
sequence.  For a configuration with S seeds, one bootstrap replicate

    1. draws S runs with replacement (seed-level uncertainty), and
    2. for each drawn run, draws its held-out sequences with replacement, per
       evaluation unit (a client's split for FL / local-only, the pooled split for
       the centralized model) — groups never span nodes, so this is exact, and
    3. recomputes the run's headline metric and averages over the drawn runs.

The interval is the 2.5 / 97.5 percentile of B replicates.  Everything is
vectorised over replicates: threshold metrics come from resampled confusion
counts (weights @ per-group counts), ROC-AUC from a weighted Mann-Whitney
statistic with exact ties, the loss from per-group loss sums.

**The sequence resampling is stratified by class composition.**  Drawing
sequences uniformly lets a resample contain no sequence carrying one of the
classes, and ``metrics_from_counts`` then silently reports balanced accuracy as
the recall of the surviving class alone.  That is not a rare edge case here:
under Dirichlet 0.1 one node holds all 211 of its no-fire test images in a
*single* sequence, so 37.8 % of uniform resamples dropped the class entirely and
biased the interval upward, away from its own point estimate.  Even the IID
partition has a node with 1011 no-fire images in two sequences (13.5 %).

Sequences are therefore split into three strata -- those carrying only class 0,
only class 1, and both -- and each stratum is resampled with replacement to its
own size.  Stratifying by a sequence's *majority* class would not work: FLAME
sequences are videos in which the fire appears and disappears, so a node's whole
minority class can live inside sequences that are majority the other class (two
of the three nodes above have no no-fire-dominant sequence at all).  Splitting on
composition avoids this, because a mixed sequence carries both classes by
definition: if a class occurs anywhere in a unit, it occurs in every resample.
"""

from __future__ import annotations

import zlib
from typing import Dict, List, Sequence

import numpy as np

from src.evaluation.predictions import per_image_loss

BOOT_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1",
                "macro_f1", "mcc", "roc_auc", "loss")
DEFAULT_B = 1000
BASE_SEED = 20260919


class Unit:
    """One evaluation unit: labels, logit margins and sequence ids of its images."""

    def __init__(self, label: Sequence[int], margin: Sequence[float], group: Sequence):
        self.label = np.asarray(label, dtype=np.int64)
        self.margin = np.asarray(margin, dtype=np.float64)
        self.gid, self.g = np.unique(np.asarray(group).astype(str), return_inverse=True)
        self.G = len(self.gid)
        pred = (self.margin > 0).astype(np.int64)
        y = self.label
        self.counts = np.zeros((self.G, 4))              # tp, fp, fn, tn (positive = Fire)
        for col, mask in enumerate(((pred == 1) & (y == 1), (pred == 1) & (y == 0),
                                    (pred == 0) & (y == 1), (pred == 0) & (y == 0))):
            np.add.at(self.counts[:, col], self.g[mask], 1)
        self.loss_sum = np.zeros(self.G)
        np.add.at(self.loss_sum, self.g, per_image_loss(y, self.margin))
        self.n = np.bincount(self.g, minlength=self.G).astype(float)
        # class-composition strata of the sequences: only class 0, only class 1, both.
        # Resampling within these keeps every class that occurs in the unit present in
        # every resample (a mixed sequence carries both classes by construction).
        n1 = np.bincount(self.g[y == 1], minlength=self.G)
        n0 = np.bincount(self.g[y == 0], minlength=self.G)
        self.strata = [np.flatnonzero(s) for s in ((n0 > 0) & (n1 == 0),
                                                   (n1 > 0) & (n0 == 0),
                                                   (n0 > 0) & (n1 > 0))]
        self.strata = [s for s in self.strata if len(s)]
        order = np.argsort(self.margin, kind="mergesort")
        self._y = y[order]
        self._g = self.g[order]
        s = self.margin[order]
        self._starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])

    def weights(self, B: int, rng: np.random.Generator,
                stratified: bool = True) -> np.ndarray:
        """(B, G) multiplicities of each sequence in B cluster resamples.

        Stratified by class composition by default, so no resample can drop a
        class that the unit actually contains (see the module docstring).
        ``stratified=False`` is the plain uniform resampling, kept so the effect
        of the stratification can be measured; it must not be used for reported
        intervals.
        """
        offset = np.arange(B)[:, None] * self.G
        if not stratified:
            draw = rng.integers(0, self.G, size=(B, self.G))
            flat = (draw + offset).ravel()
            return np.bincount(flat, minlength=B * self.G).reshape(B, self.G).astype(float)

        W = np.zeros(B * self.G)
        for idx in self.strata:                      # disjoint, so the counts just add
            k = len(idx)
            draw = idx[rng.integers(0, k, size=(B, k))]
            flat = (draw + offset).ravel()
            W += np.bincount(flat, minlength=B * self.G)
        return W.reshape(B, self.G).astype(float)

    def auc(self, W: np.ndarray) -> np.ndarray:
        wi = W[:, self._g]                               # (B, N) image weights, sorted by score
        pos = wi * (self._y == 1)
        neg = wi * (self._y == 0)
        bpos = np.add.reduceat(pos, self._starts, axis=1)
        bneg = np.add.reduceat(neg, self._starts, axis=1)
        below = np.cumsum(bneg, axis=1) - bneg
        num = (bpos * (below + 0.5 * bneg)).sum(axis=1)
        den = pos.sum(axis=1) * neg.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    def replicate(self, W: np.ndarray) -> Dict[str, np.ndarray]:
        tp, fp, fn, tn = (W @ self.counts).T
        out = metrics_from_counts(tp, fp, fn, tn)
        n = W @ self.n
        with np.errstate(invalid="ignore", divide="ignore"):
            out["loss"] = (W @ self.loss_sum) / n
        out["roc_auc"] = self.auc(W)
        out["_n"] = n
        return out


def _div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), 0.0)


def metrics_from_counts(tp, fp, fn, tn) -> Dict[str, np.ndarray]:
    """Binary metrics from (vectors of) confusion counts, as in metrics.metrics_from_confusion_matrix."""
    tp, fp, fn, tn = (np.asarray(v, float) for v in (tp, fp, fn, tn))
    n = tp + fp + fn + tn
    rec1, rec0 = _div(tp, tp + fn), _div(tn, tn + fp)
    prec1, prec0 = _div(tp, tp + fp), _div(tn, tn + fn)
    f1_1, f1_0 = _div(2 * prec1 * rec1, prec1 + rec1), _div(2 * prec0 * rec0, prec0 + rec0)
    has1, has0 = (tp + fn) > 0, (tn + fp) > 0
    bal = _div(rec1 * has1 + rec0 * has0, has1.astype(float) + has0.astype(float))
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"accuracy": _div(tp + tn, n), "balanced_accuracy": bal, "precision": prec1,
            "recall": rec1, "specificity": rec0, "f1": f1_1, "macro_f1": (f1_1 + f1_0) / 2,
            "mcc": _div(num, den)}


def run_replicates(units: List[Unit], aggregation: str, B: int,
                   rng: np.random.Generator, stratified: bool = True) -> Dict[str, np.ndarray]:
    """B cluster-bootstrap values of one run's headline metrics.

    ``aggregation``: "weighted" (FL: client metrics weighted by resampled test size),
    "mean" (local-only: mean over nodes) or "pooled" (centralized: **one** unit --
    the caller pools the per-node predictions before constructing it).
    """
    reps = [u.replicate(u.weights(B, rng, stratified)) for u in units]
    return _aggregate(reps, aggregation)


def _aggregate(reps: List[Dict[str, np.ndarray]], aggregation: str) -> Dict[str, np.ndarray]:
    """Combine per-unit replicate metrics into the run's headline metrics."""
    if aggregation == "pooled" or len(reps) == 1:
        return {m: reps[0][m] for m in BOOT_METRICS}
    if aggregation == "weighted":
        w = np.stack([r["_n"] for r in reps])
        return {m: np.nansum(np.stack([r[m] for r in reps]) * w, axis=0) / w.sum(axis=0)
                for m in BOOT_METRICS}
    if aggregation == "mean":
        return {m: np.nanmean(np.stack([r[m] for r in reps]), axis=0) for m in BOOT_METRICS}
    raise ValueError(aggregation)


def paired_diff_ci(runs_a: List[List[Unit]], runs_b: List[List[Unit]], aggregation: str,
                   key: str, B: int = DEFAULT_B, level: float = 0.95,
                   stratified: bool = True) -> Dict[str, Dict[str, float]]:
    """CI of mean(A) - mean(B) for two configurations evaluated on the SAME held-out set.

    Every run of both configurations must carry the same evaluation units with the
    same sequences (unit j of every run covers the same images; for ``"pooled"`` the
    caller passes one pooled unit per run).  One replicate draws **one** stratified
    sequence resample per unit and applies it to every run of both sides -- the test
    set is shared, so resampling it independently per side would double-count its
    variance -- and draws the seeds of each side independently with replacement.
    The point estimate is the difference of the seed means on the full held-out set.

    ``key`` seeds the generator.  -> {metric: {"diff", "ci_low", "ci_high",
    "p_gt0", "B"}} where ``p_gt0`` is the fraction of replicates with A > B.
    """
    if not runs_a or not runs_b:
        raise ValueError("both sides need at least one run")
    ref = runs_a[0]
    for units in list(runs_a) + list(runs_b):
        if len(units) != len(ref) or not all(np.array_equal(u.gid, v.gid) and u.G == v.G
                                              for u, v in zip(units, ref)):
            raise ValueError("the two configurations are not evaluated on the same sequences")
    rng = np.random.default_rng(BASE_SEED + zlib.crc32(("diff|" + key).encode("utf-8")))
    W = [u.weights(B, rng, stratified) for u in ref]
    ones = [np.ones((1, u.G)) for u in ref]

    def side(runs):
        full = [_aggregate([u.replicate(w) for u, w in zip(units, ones)], aggregation) for units in runs]
        reps = [_aggregate([u.replicate(w) for u, w in zip(units, W)], aggregation) for units in runs]
        S = len(runs)
        pick = rng.integers(0, S, size=(B, S))
        col = np.broadcast_to(np.arange(B)[:, None], pick.shape)
        point = {m: float(np.nanmean([f[m][0] for f in full])) for m in BOOT_METRICS}
        boot = {m: np.nanmean(np.stack([r[m] for r in reps])[pick, col], axis=1) for m in BOOT_METRICS}
        return point, boot

    pa, ba = side(runs_a)
    pb, bb = side(runs_b)
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    out = {}
    for m in BOOT_METRICS:
        d = ba[m] - bb[m]
        out[m] = {"diff": pa[m] - pb[m], "ci_low": float(np.nanpercentile(d, lo_q)),
                  "ci_high": float(np.nanpercentile(d, hi_q)),
                  "p_gt0": float(np.nanmean(d > 0)), "B": B}
    return out


def config_ci(runs: List[List[Unit]], aggregation: str, key: str, B: int = DEFAULT_B,
              level: float = 0.95, stratified: bool = True) -> Dict[str, Dict[str, float]]:
    """Hierarchical (seed x sequence) bootstrap CI of each metric for one configuration.

    ``key`` (the configuration id) seeds the generator, so results are reproducible.
    -> {metric: {"ci_low", "ci_high", "se", "B"}}.
    """
    rng = np.random.default_rng(BASE_SEED + zlib.crc32(key.encode("utf-8")))
    per_run = [run_replicates(units, aggregation, B, rng, stratified) for units in runs]
    S = len(per_run)
    pick = rng.integers(0, S, size=(B, S))
    col = (np.arange(B)[:, None] * S + np.arange(S)[None, :]) % B   # distinct draw per slot
    out = {}
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    for m in BOOT_METRICS:
        vals = np.stack([per_run[k][m] for k in range(S)])            # (S, B)
        rep = np.nanmean(vals[pick, col], axis=1)                     # (B,)
        out[m] = {"ci_low": float(np.nanpercentile(rep, lo_q)),
                  "ci_high": float(np.nanpercentile(rep, hi_q)),
                  "se": float(np.nanstd(rep, ddof=1)), "B": B}
    return out


# ---------------------------------------------------------------------------------------
# Several models on ONE shared held-out set (docs/GLOBAL_EVALUATION.md)
#
# In the global evaluation every model of a partition -- the federated global model, the
# centralized model and each of the three local-only node models -- is evaluated on the
# same images, the union of the three nodes' test splits.  A run may therefore consist of
# several models (local-only: one per node) whose value is their mean, and all of them
# must see the same resample of the shared set.  Threshold metrics only (from resampled
# confusion counts): the ROC-AUC path of ``Unit.replicate`` holds B x N image weights,
# which does not fit in memory at B = 10,000 on a union of several thousand images.
# ---------------------------------------------------------------------------------------

THRESHOLD_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "specificity",
                     "f1", "macro_f1", "mcc")


def check_same_set(units: Sequence[Unit]) -> None:
    """Raise unless every unit covers the same sequences with the same images per class."""
    ref = units[0]
    pos = ref.counts[:, 0] + ref.counts[:, 2]                  # tp + fn = class-1 images
    for u in units[1:]:
        if not (u.G == ref.G and np.array_equal(u.gid, ref.gid) and np.array_equal(u.n, ref.n)
                and np.array_equal(u.counts[:, 0] + u.counts[:, 2], pos)):
            raise ValueError("the models are not evaluated on the same held-out images")


def _threshold_metric(unit: Unit, W: np.ndarray, metric: str) -> np.ndarray:
    tp, fp, fn, tn = (W @ unit.counts).T
    return metrics_from_counts(tp, fp, fn, tn)[metric]


def _run_value(models: Sequence[Unit], W: np.ndarray, metric: str) -> np.ndarray:
    """Value of one run under the sequence weights W: the mean over its models."""
    return np.mean([_threshold_metric(u, W, metric) for u in models], axis=0)


def _percentile_ci(values: np.ndarray, level: float) -> Dict[str, float]:
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return {"ci_low": float(np.percentile(values, lo_q)),
            "ci_high": float(np.percentile(values, hi_q))}


def shared_set_ci(runs: List[List[Unit]], key: str, metric: str = "balanced_accuracy",
                  B: int = DEFAULT_B, level: float = 0.95) -> Dict[str, float]:
    """Seed x sequence bootstrap CI of a configuration whose runs are lists of models
    evaluated on one shared set; a run's value is the mean over its models.

    Each run draws its own stratified sequence resample, shared by all of that run's
    models, then the runs are resampled with replacement -- ``config_ci`` in every other
    respect, with which it coincides (same key, same draws) for one model per run.
    -> {"mean", "ci_low", "ci_high", "se", "n_runs", "B"}.
    """
    if not runs:
        raise ValueError("no runs")
    check_same_set([u for run in runs for u in run])
    rng = np.random.default_rng(BASE_SEED + zlib.crc32(key.encode("utf-8")))
    ones = np.ones((1, runs[0][0].G))
    point = float(np.mean([_run_value(run, ones, metric)[0] for run in runs]))
    per_run = np.stack([_run_value(run, run[0].weights(B, rng), metric) for run in runs])
    S = len(runs)
    pick = rng.integers(0, S, size=(B, S))
    col = (np.arange(B)[:, None] * S + np.arange(S)[None, :]) % B
    rep = per_run[pick, col].mean(axis=1)
    out = {"mean": point, "se": float(np.std(rep, ddof=1)), "n_runs": S, "B": B}
    out.update(_percentile_ci(rep, level))
    return out


def seed_paired_diff_ci(runs_a: List[List[Unit]], runs_b: List[List[Unit]], key: str,
                        metric: str = "balanced_accuracy", B: int = DEFAULT_B,
                        level: float = 0.95) -> Dict[str, float]:
    """CI and bootstrap p-value of the seed-paired difference mean_s [A_s - B_s].

    ``runs_a[s]`` and ``runs_b[s]`` are the runs of the same seed s (lists of models, a
    run's value being the mean over its models), all evaluated on one shared set.  One
    replicate draws ONE stratified sequence resample of the shared set, applied to every
    model of both sides, and ONE draw of S seed indices with replacement, applied to both
    sides, so the pairing is kept.  The interval is the percentile interval of the B
    replicate differences; the two-sided p-value is
    ``min(1, 2 * min(#{D* <= 0} + 1, #{D* >= 0} + 1) / (B + 1))``.
    -> {"diff", "ci_low", "ci_high", "p_boot", "n_pairs", "B"}.
    """
    if not runs_a or len(runs_a) != len(runs_b):
        raise ValueError("need the same, non-zero number of runs on both sides (one per seed)")
    check_same_set([u for run in list(runs_a) + list(runs_b) for u in run])
    rng = np.random.default_rng(BASE_SEED + zlib.crc32(("seedpaired|" + key).encode("utf-8")))
    ref = runs_a[0][0]
    ones = np.ones((1, ref.G))
    W = ref.weights(B, rng)
    point = float(np.mean([_run_value(a, ones, metric)[0] - _run_value(b, ones, metric)[0]
                           for a, b in zip(runs_a, runs_b)]))
    d = np.stack([_run_value(a, W, metric) - _run_value(b, W, metric)
                  for a, b in zip(runs_a, runs_b)])                       # (S, B)
    S = len(runs_a)
    pick = rng.integers(0, S, size=(B, S))
    rep = d[pick, np.arange(B)[:, None]].mean(axis=1)
    p = min(1.0, 2 * min(int((rep <= 0).sum()) + 1, int((rep >= 0).sum()) + 1) / (B + 1))
    out = {"diff": point, "p_boot": p, "n_pairs": S, "B": B}
    out.update(_percentile_ci(rep, level))
    return out


# ---------------------------------------------------------------------------------------
# Different images of the SAME clusters (docs/CAMERA_EXPERIMENT_PREREG.md, section 5)
#
# In the camera experiment every scene is recorded by every sensor, so "model X on sensor-Y
# frames" and "model X on sensor-X frames" cover different images of one common scene set.
# ``check_same_set`` does not hold, but the scenes are shared: one scene resample is applied
# to every side, so the scene variance common to both sides cancels in the difference.
# ---------------------------------------------------------------------------------------

def scene_paired_contrast_ci(terms: Sequence, key: str, metric: str = "balanced_accuracy",
                             B: int = DEFAULT_B, level: float = 0.95) -> Dict[str, float]:
    """CI and bootstrap p-value of a seed-paired linear contrast of sides evaluated on
    different images of the same clusters (scenes).

    ``terms`` is a sequence of ``(coefficient, runs)``; ``runs[s]`` is the run of seed s
    (a list of models, the run's value being the mean over its models, as in
    ``seed_paired_diff_ci``).  Every term has the same number S of runs, index s being the
    same seed in every term.  The contrast is ``C = mean_s sum_k coef_k * value_k,s``; a
    plain difference D(X -> Y) is ``[(+1, runs on Y frames), (-1, runs on X frames)]``.

    Every unit of every term must cover the same clusters (``gid``), not the same images.
    One replicate draws ONE cluster resample over that common set -- stratified by the
    classes a cluster carries across all units (with both classes in every scene, one
    stratum) -- applied to every model of every term, and ONE draw of S seed indices with
    replacement, applied to every term, so both the scene and the seed pairing are kept.
    Percentile interval; two-sided p-value
    ``min(1, 2 * min(#{C* <= 0} + 1, #{C* >= 0} + 1) / (B + 1))``; the generator is keyed
    by ``BASE_SEED + crc32("scenepaired|" + key)``.  Threshold metrics only.
    -> {"diff", "ci_low", "ci_high", "p_boot", "n_pairs", "n_clusters", "B"}.
    """
    terms = [(float(c), list(runs)) for c, runs in terms]
    if not terms or not terms[0][1]:
        raise ValueError("need at least one term with at least one run")
    S = len(terms[0][1])
    if any(len(runs) != S for _, runs in terms):
        raise ValueError("every term needs the same number of runs (one per seed)")
    units = [u for _, runs in terms for run in runs for u in run]
    ref = units[0]
    if any(u.G != ref.G or not np.array_equal(u.gid, ref.gid) for u in units):
        raise ValueError("the sides do not cover the same clusters")
    joint = Unit(np.concatenate([u.label for u in units]),
                 np.zeros(sum(len(u.label) for u in units)),
                 np.concatenate([u.gid[u.g] for u in units]))
    rng = np.random.default_rng(BASE_SEED + zlib.crc32(("scenepaired|" + key).encode("utf-8")))
    W = joint.weights(B, rng)
    ones = np.ones((1, ref.G))
    point = float(np.mean([sum(c * _run_value(runs[s], ones, metric)[0] for c, runs in terms)
                           for s in range(S)]))
    d = np.stack([sum(c * _run_value(runs[s], W, metric) for c, runs in terms)
                  for s in range(S)])                                     # (S, B)
    pick = rng.integers(0, S, size=(B, S))
    rep = d[pick, np.arange(B)[:, None]].mean(axis=1)
    p = min(1.0, 2 * min(int((rep <= 0).sum()) + 1, int((rep >= 0).sum()) + 1) / (B + 1))
    out = {"diff": point, "p_boot": p, "n_pairs": S, "n_clusters": int(ref.G), "B": B}
    out.update(_percentile_ci(rep, level))
    return out


__all__ = ["BOOT_METRICS", "DEFAULT_B", "THRESHOLD_METRICS", "Unit", "check_same_set",
           "config_ci", "metrics_from_counts", "paired_diff_ci", "run_replicates",
           "scene_paired_contrast_ci", "seed_paired_diff_ci", "shared_set_ci"]
