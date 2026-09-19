"""FedRGBD — per-image predictions: format, join key, metric recomputation.

One ``.npz`` per (round, client, split) for the FL runs and per (node, split) for
the centralized / local-only baselines, with four arrays of equal length:

    path          str      manifest key "<Class>/<file>", e.g. "Fire/resized_frame21126.jpg"
                           (joins data/splits/*.csv.gz, analysis/leakage/groups.csv and the
                           clean-subset lists; never a position: FlameDataset lists files with
                           an unsorted os.listdir, so the order differs between machines)
    label         uint8    true class, 1 = Fire
    logit_margin  float32  z_fire - z_nofire of the evaluated model
    p_fire        float32  sigmoid(logit_margin), for convenience

Every metric of ``src/evaluation/metrics.py`` and the per-image cross-entropy can
be recomputed from ``label`` and ``logit_margin`` (softmax over [0, margin] equals
softmax over [z_nofire, z_fire]).
"""

from __future__ import annotations

import io
import os
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.evaluation.metrics import compute_metrics

FORMAT_VERSION = 1
PRED_PREFIX = "pred_"                       # Flower metric keys: pred_val_npz, pred_test_npz
FIELDS = ("path", "label", "logit_margin", "p_fire")

README_TEXT = """# Per-image predictions (format v{version})

One file per round, client and split: `r<round>_<node>_<split>.npz` (FL runs, written by
`src/fl/server.py`) or `selected_<node>_<split>.npz` (centralized / local-only baselines,
written by `scripts/predict_from_checkpoint.py` from `model_selected.pt`).

| array | dtype | meaning |
|---|---|---|
| `path` | str | manifest key `<Class>/<file>`; joins `data/splits/*.csv.gz`, `analysis/leakage/groups.csv` and `analysis/leakage/clean_subset/` |
| `label` | uint8 | true class, 1 = Fire |
| `logit_margin` | float32 | `z_fire - z_nofire` of the evaluated model |
| `p_fire` | float32 | `sigmoid(logit_margin)` |

Load with `src.evaluation.predictions.load_npz(path)`; recompute metrics with
`metrics_from_predictions(label, logit_margin)`.

The `.npz` files are gitignored (the whole matrix is ~54 MB); archive them with the run.
FL predictions exist only as written by the run itself: they cannot be regenerated without
re-running it. Baseline predictions can be regenerated at any time from the saved
checkpoints: `python scripts/predict_from_checkpoint.py <run_dir>`.
""".format(version=FORMAT_VERSION)


def manifest_key(sample_path: str) -> str:
    """``.../<split>/<Class>/<file>`` -> ``<Class>/<file>`` (forward slash)."""
    parent, name = os.path.split(os.path.normpath(sample_path))
    return "%s/%s" % (os.path.basename(parent), name)


def margins(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    return (logits[:, 1] - logits[:, 0]).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def pack(paths: Sequence[str], labels: Iterable[int], logits: np.ndarray) -> bytes:
    """Compressed ``.npz`` bytes (send as a Flower metric, or write to disk)."""
    m = margins(logits)
    paths = np.asarray([manifest_key(p) for p in paths], dtype=str)
    labels = np.asarray(list(labels), dtype=np.uint8)
    if not (len(paths) == len(labels) == len(m)):
        raise ValueError("paths/labels/logits lengths differ: %d/%d/%d"
                         % (len(paths), len(labels), len(m)))
    buf = io.BytesIO()
    np.savez_compressed(buf, path=paths, label=labels, logit_margin=m,
                        p_fire=_sigmoid(m).astype(np.float32),
                        format_version=np.array(FORMAT_VERSION))
    return buf.getvalue()


def unpack(blob: bytes) -> Dict[str, np.ndarray]:
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        return {k: z[k] for k in FIELDS}


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        return unpack(f.read())


def per_image_loss(label: np.ndarray, margin: np.ndarray) -> np.ndarray:
    """Cross-entropy of each image: softplus(-m) for Fire, softplus(m) for No_Fire."""
    m = np.asarray(margin, dtype=np.float64)
    signed = np.where(np.asarray(label) == 1, -m, m)
    return np.logaddexp(0.0, signed)


def metrics_from_predictions(label: np.ndarray, margin: np.ndarray) -> Dict[str, object]:
    """The full metric set of ``compute_metrics`` plus the mean cross-entropy ``loss``."""
    label = np.asarray(label, dtype=np.int64)
    m = np.asarray(margin, dtype=np.float32)
    logits = np.stack([np.zeros_like(m), m], axis=1)
    out = compute_metrics(logits, label, num_classes=2)
    out["loss"] = float(per_image_loss(label, m).mean()) if len(label) else None
    return out


__all__ = ["FIELDS", "FORMAT_VERSION", "PRED_PREFIX", "README_TEXT", "load_npz", "manifest_key",
           "margins", "metrics_from_predictions", "pack", "per_image_loss", "unpack"]
