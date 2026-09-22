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
    if aggregation == "pooled" or len(reps) == 1:
        return {m: reps[0][m] for m in BOOT_METRICS}
    if aggregation == "weighted":
        w = np.stack([r["_n"] for r in reps])
        return {m: np.nansum(np.stack([r[m] for r in reps]) * w, axis=0) / w.sum(axis=0)
                for m in BOOT_METRICS}
    if aggregation == "mean":
        return {m: np.nanmean(np.stack([r[m] for r in reps]), axis=0) for m in BOOT_METRICS}
    raise ValueError(aggregation)


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


__all__ = ["BOOT_METRICS", "DEFAULT_B", "Unit", "config_ci", "metrics_from_counts",
           "run_replicates"]
