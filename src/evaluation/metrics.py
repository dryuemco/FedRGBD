"""FedRGBD — Classification metrics computed from logits and labels.

Pure NumPy implementation (no scikit-learn dependency at run time) so it runs
identically on the Jetson clients, the FL server and the analysis desktop.
Every function accepts either ``torch.Tensor`` or ``numpy.ndarray`` inputs.

Metrics returned by :func:`compute_metrics` (binary task, positive class = 1 = Fire):

    accuracy, balanced_accuracy, precision, recall (sensitivity), specificity,
    f1, macro_f1, macro_precision, macro_recall, macro_specificity, mcc, roc_auc,
    confusion_matrix (rows = true class, cols = predicted class), n_examples,
    per-class support.

For a multi-class problem the "precision/recall/specificity/f1" entries are
the macro averages and ``roc_auc`` is the macro one-vs-rest AUC.

Metrics that are undefined for a batch (e.g. ROC-AUC when only one class is
present) are returned as ``None`` and *dropped* by :func:`to_flower_metrics`
so that they never turn into NaN in results.json or Flower messages.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

try:  # torch is optional for the analysis desktop
    import torch
except ImportError:  # pragma: no cover
    torch = None

ArrayLike = Union["np.ndarray", "torch.Tensor", Sequence]

METRIC_KEYS: List[str] = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "macro_specificity",
    "mcc",
    "roc_auc",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_numpy(x: ArrayLike) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    logits = np.asarray(logits, dtype=np.float64)
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, num_classes: int) -> np.ndarray:
    """Confusion matrix with rows = true label, cols = predicted label."""
    y_true = _to_numpy(y_true).astype(np.int64).ravel()
    y_pred = _to_numpy(y_pred).astype(np.int64).ravel()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with tie handling, like scipy.stats.rankdata."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def binary_roc_auc(y_true: ArrayLike, scores: ArrayLike) -> Optional[float]:
    """ROC-AUC via the Mann-Whitney U statistic (ties get half credit).

    Returns ``None`` when only one class is present.
    """
    y_true = _to_numpy(y_true).astype(np.int64).ravel()
    scores = _to_numpy(scores).astype(np.float64).ravel()
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rankdata_average(scores)
    sum_pos = ranks[y_true == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def matthews_corrcoef_from_cm(cm: np.ndarray) -> float:
    """Multi-class MCC (Gorodkin 2004), identical to scikit-learn's formula."""
    cm = cm.astype(np.float64)
    t = cm.sum(axis=1)  # true counts per class
    p = cm.sum(axis=0)  # predicted counts per class
    c = np.trace(cm)
    s = cm.sum()
    cov_ytyp = c * s - (t * p).sum()
    cov_ypyp = s * s - (p * p).sum()
    cov_ytyt = s * s - (t * t).sum()
    den = np.sqrt(cov_ypyp * cov_ytyt)
    if den == 0:
        return 0.0
    return float(cov_ytyp / den)


# --------------------------------------------------------------------------- #
# main entry points
# --------------------------------------------------------------------------- #
def metrics_from_confusion_matrix(cm: np.ndarray, positive_class: int = 1) -> Dict[str, object]:
    """All threshold-based metrics derived from a confusion matrix.

    Used both on a single client and on the server (from the *summed*
    confusion matrix of all clients, giving true pooled global metrics).
    """
    cm = np.asarray(cm, dtype=np.int64)
    k = cm.shape[0]
    n = int(cm.sum())
    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = n - tp - fn - fp
    support = cm.sum(axis=1)

    recall_c = np.array([_safe_div(tp[i], tp[i] + fn[i]) for i in range(k)])
    precision_c = np.array([_safe_div(tp[i], tp[i] + fp[i]) for i in range(k)])
    specificity_c = np.array([_safe_div(tn[i], tn[i] + fp[i]) for i in range(k)])
    f1_c = np.array([
        _safe_div(2 * precision_c[i] * recall_c[i], precision_c[i] + recall_c[i]) for i in range(k)
    ])

    present = support > 0  # classes that occur in y_true
    balanced_acc = float(recall_c[present].mean()) if present.any() else 0.0

    out: Dict[str, object] = {
        "n_examples": n,
        "accuracy": _safe_div(tp.sum(), n),
        "balanced_accuracy": balanced_acc,
        "macro_precision": float(precision_c.mean()),
        "macro_recall": float(recall_c.mean()),
        "macro_specificity": float(specificity_c.mean()),
        "macro_f1": float(f1_c.mean()),
        "mcc": matthews_corrcoef_from_cm(cm),
        "confusion_matrix": cm.tolist(),
        "support": support.tolist(),
    }
    if k == 2:
        pc = positive_class
        out.update({
            "precision": float(precision_c[pc]),
            "recall": float(recall_c[pc]),
            "specificity": float(specificity_c[pc]),
            "f1": float(f1_c[pc]),
            "tp": int(tp[pc]),
            "fp": int(fp[pc]),
            "fn": int(fn[pc]),
            "tn": int(tn[pc]),
        })
    else:
        out.update({
            "precision": float(precision_c.mean()),
            "recall": float(recall_c.mean()),
            "specificity": float(specificity_c.mean()),
            "f1": float(f1_c.mean()),
        })
    out["per_class"] = {
        str(i): {
            "precision": float(precision_c[i]),
            "recall": float(recall_c[i]),
            "specificity": float(specificity_c[i]),
            "f1": float(f1_c[i]),
            "support": int(support[i]),
        }
        for i in range(k)
    }
    return out


def compute_metrics(
    logits: ArrayLike,
    labels: ArrayLike,
    num_classes: Optional[int] = None,
    positive_class: int = 1,
) -> Dict[str, object]:
    """Compute the full metric set from raw model outputs.

    Args:
        logits: ``(N, C)`` raw scores (pre-softmax).  ``(N,)`` is accepted for a
            binary task and is interpreted as the logit of the positive class.
        labels: ``(N,)`` integer class labels.
        num_classes: defaults to ``C`` (or 2 for 1-D logits).
        positive_class: index of the positive class (Fire = 1).
    """
    logits = _to_numpy(logits).astype(np.float64)
    labels = _to_numpy(labels).astype(np.int64).ravel()

    if logits.ndim == 1:  # single logit for positive class
        logits = np.stack([-logits, logits], axis=1)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (N, C); got shape {logits.shape}")
    n, c = logits.shape
    if n != len(labels):
        raise ValueError(f"{n} logits but {len(labels)} labels")
    if num_classes is None:
        num_classes = c
    num_classes = max(int(num_classes), c, int(labels.max()) + 1 if n else 0)

    if n == 0:
        out: Dict[str, object] = {k: None for k in METRIC_KEYS}
        out.update({"n_examples": 0, "confusion_matrix": np.zeros((num_classes, num_classes), int).tolist()})
        return out

    probs = softmax(logits)
    preds = logits.argmax(axis=1)
    cm = confusion_matrix(labels, preds, num_classes)
    out = metrics_from_confusion_matrix(cm, positive_class=positive_class)

    # ROC-AUC
    if num_classes == 2:
        out["roc_auc"] = binary_roc_auc(labels == positive_class, probs[:, positive_class])
    else:
        aucs = []
        for k in range(num_classes):
            a = binary_roc_auc(labels == k, probs[:, k])
            if a is not None:
                aucs.append(a)
        out["roc_auc"] = float(np.mean(aucs)) if aucs else None
    return out


class MetricAccumulator:
    """Collects logits/labels over mini-batches and computes metrics at the end.

    Memory footprint is ``N × C`` float32 — for the FLAME validation split
    (≈7 k images × 2 classes) that is a few tens of kilobytes.
    """

    def __init__(self, num_classes: int = 2, positive_class: int = 1):
        self.num_classes = num_classes
        self.positive_class = positive_class
        self._logits: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []
        self.total_loss = 0.0

    def update(self, logits: ArrayLike, labels: ArrayLike, loss_sum: float = 0.0) -> None:
        self._logits.append(_to_numpy(logits).astype(np.float32))
        self._labels.append(_to_numpy(labels).astype(np.int64))
        self.total_loss += float(loss_sum)

    def __len__(self) -> int:
        return int(sum(len(l) for l in self._labels))

    def outputs(self):
        """``(logits (N, C) float32, labels (N,) int64)`` in the order they were added."""
        if not self._labels:
            return np.zeros((0, self.num_classes), np.float32), np.zeros(0, np.int64)
        return np.concatenate(self._logits, axis=0), np.concatenate(self._labels, axis=0)

    def compute(self) -> Dict[str, object]:
        if not self._labels:
            m = compute_metrics(np.zeros((0, self.num_classes)), np.zeros(0), self.num_classes)
            m["loss"] = None
            return m
        logits = np.concatenate(self._logits, axis=0)
        labels = np.concatenate(self._labels, axis=0)
        m = compute_metrics(logits, labels, self.num_classes, self.positive_class)
        m["loss"] = self.total_loss / max(len(labels), 1)
        return m


# --------------------------------------------------------------------------- #
# serialisation helpers
# --------------------------------------------------------------------------- #
def to_flower_metrics(metrics: Dict[str, object], prefix: str = "") -> Dict[str, Union[int, float, str, bool]]:
    """Flatten a metrics dict into Flower-compatible scalars.

    * ``None`` values are dropped (undefined metric for this client/round)
    * the confusion matrix is flattened to ``cm_<i>_<j>`` ints **and** kept as a
      JSON string under ``confusion_matrix_json``
    * nested ``per_class`` dicts are flattened to ``cls<k>_<metric>``
    """
    flat: Dict[str, Union[int, float, str, bool]] = {}
    for key, val in metrics.items():
        name = f"{prefix}{key}"
        if val is None:
            continue
        if key == "confusion_matrix":
            cm = np.asarray(val)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    flat[f"{prefix}cm_{i}_{j}"] = int(cm[i, j])
            flat[f"{prefix}confusion_matrix_json"] = json.dumps(cm.tolist())
        elif key == "per_class" and isinstance(val, dict):
            for cls, sub in val.items():
                for mk, mv in sub.items():
                    if mv is not None:
                        flat[f"{prefix}cls{cls}_{mk}"] = float(mv) if mk != "support" else int(mv)
        elif key == "support" and isinstance(val, (list, tuple)):
            for i, s in enumerate(val):
                flat[f"{prefix}support_{i}"] = int(s)
        elif isinstance(val, (bool, int, float, str)):
            flat[name] = val
        elif isinstance(val, np.generic):
            flat[name] = val.item()
        # anything else (lists/dicts) is not a Flower scalar → skipped
    return flat


def confusion_matrix_from_flat(flat: Dict[str, object], prefix: str = "") -> Optional[np.ndarray]:
    """Inverse of the ``cm_i_j`` flattening produced by :func:`to_flower_metrics`."""
    js = flat.get(f"{prefix}confusion_matrix_json")
    if isinstance(js, str):
        try:
            return np.asarray(json.loads(js), dtype=np.int64)
        except json.JSONDecodeError:
            pass
    cells = {}
    for key, val in flat.items():
        if key.startswith(f"{prefix}cm_"):
            parts = key[len(prefix) + 3 :].split("_")
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                cells[(int(parts[0]), int(parts[1]))] = int(val)
    if not cells:
        return None
    k = max(max(i, j) for i, j in cells) + 1
    cm = np.zeros((k, k), dtype=np.int64)
    for (i, j), v in cells.items():
        cm[i, j] = v
    return cm


def format_metrics(metrics: Dict[str, object], keys: Iterable[str] = METRIC_KEYS, digits: int = 4) -> str:
    """One-line human readable summary, e.g. for log output."""
    parts = []
    for k in keys:
        v = metrics.get(k)
        if v is None:
            continue
        parts.append(f"{k}={v:.{digits}f}")
    return ", ".join(parts)


__all__ = [
    "METRIC_KEYS",
    "MetricAccumulator",
    "binary_roc_auc",
    "compute_metrics",
    "confusion_matrix",
    "confusion_matrix_from_flat",
    "format_metrics",
    "matthews_corrcoef_from_cm",
    "metrics_from_confusion_matrix",
    "softmax",
    "to_flower_metrics",
]
