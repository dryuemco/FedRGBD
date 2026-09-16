"""Tests for src/evaluation/metrics.py.

Cross-checks the pure-NumPy metric implementation against scikit-learn on
synthetic random logits, and exercises edge cases (empty input, single-class
labels, 1-D logits, torch tensor inputs) plus the serialization helpers
(``to_flower_metrics`` / ``confusion_matrix_from_flat``) and
``MetricAccumulator``.
"""

import json

import numpy as np
import pytest
import sklearn.metrics as skm

import torch

from src.evaluation import metrics as M


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_binary(seed, n=200, n_classes=2):
    rs = np.random.RandomState(seed)
    logits = rs.randn(n, n_classes) * 2.0
    labels = rs.randint(0, n_classes, size=n)
    return logits, labels


def make_multiclass(seed, n=300, n_classes=4):
    rs = np.random.RandomState(seed)
    logits = rs.randn(n, n_classes) * 2.0
    # ensure every class appears at least a handful of times
    labels = rs.randint(0, n_classes, size=n)
    labels[:n_classes] = np.arange(n_classes)
    return logits, labels


def softmax_np(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# binary
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_binary_matches_sklearn(seed):
    logits, labels = make_binary(seed)
    preds = logits.argmax(axis=1)
    probs = softmax_np(logits)

    out = M.compute_metrics(logits, labels, positive_class=1)

    assert out["accuracy"] == pytest.approx(skm.accuracy_score(labels, preds))
    assert out["balanced_accuracy"] == pytest.approx(skm.balanced_accuracy_score(labels, preds))
    assert out["precision"] == pytest.approx(skm.precision_score(labels, preds, pos_label=1, zero_division=0))
    assert out["recall"] == pytest.approx(skm.recall_score(labels, preds, pos_label=1, zero_division=0))
    assert out["f1"] == pytest.approx(skm.f1_score(labels, preds, pos_label=1, zero_division=0))
    assert out["macro_f1"] == pytest.approx(skm.f1_score(labels, preds, average="macro", zero_division=0))
    assert out["mcc"] == pytest.approx(skm.matthews_corrcoef(labels, preds))
    assert out["roc_auc"] == pytest.approx(skm.roc_auc_score(labels, probs[:, 1]))

    cm_sklearn = skm.confusion_matrix(labels, preds, labels=[0, 1])
    np.testing.assert_array_equal(np.asarray(out["confusion_matrix"]), cm_sklearn)

    tn, fp, fn, tp = cm_sklearn.ravel()
    expected_specificity = tn / (tn + fp) if (tn + fp) else 0.0
    assert out["specificity"] == pytest.approx(expected_specificity)


def test_binary_n_examples_and_support():
    logits, labels = make_binary(7, n=137)
    out = M.compute_metrics(logits, labels)
    assert out["n_examples"] == 137
    assert sum(out["support"]) == 137


# --------------------------------------------------------------------------- #
# multiclass
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [10, 11, 12])
def test_multiclass_matches_sklearn(seed):
    logits, labels = make_multiclass(seed, n_classes=4)
    preds = logits.argmax(axis=1)
    probs = softmax_np(logits)

    out = M.compute_metrics(logits, labels, num_classes=4)

    assert out["macro_f1"] == pytest.approx(skm.f1_score(labels, preds, average="macro", zero_division=0))
    assert out["mcc"] == pytest.approx(skm.matthews_corrcoef(labels, preds))

    cm_sklearn = skm.confusion_matrix(labels, preds, labels=list(range(4)))
    np.testing.assert_array_equal(np.asarray(out["confusion_matrix"]), cm_sklearn)

    sk_auc = skm.roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=list(range(4)))
    assert out["roc_auc"] == pytest.approx(sk_auc, rel=1e-6)


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_all_one_class_binary_no_crash():
    rs = np.random.RandomState(5)
    logits = rs.randn(20, 2)
    labels = np.zeros(20, dtype=int)
    out = M.compute_metrics(logits, labels)
    assert out["roc_auc"] is None
    assert out["n_examples"] == 20


def test_all_one_class_multiclass_no_crash():
    rs = np.random.RandomState(6)
    logits = rs.randn(15, 3)
    labels = np.ones(15, dtype=int)  # only class 1 present
    out = M.compute_metrics(logits, labels, num_classes=3)
    assert out["roc_auc"] is None
    assert out["n_examples"] == 15


def test_empty_input():
    out = M.compute_metrics(np.zeros((0, 2)), np.zeros(0))
    assert out["n_examples"] == 0
    for k in M.METRIC_KEYS:
        assert out[k] is None
    cm = np.asarray(out["confusion_matrix"])
    assert cm.shape == (2, 2)
    assert cm.sum() == 0


def test_1d_logits_equivalent_to_stacked_2d():
    rs = np.random.RandomState(21)
    logits_1d = rs.randn(50)
    labels = rs.randint(0, 2, size=50)

    out_1d = M.compute_metrics(logits_1d, labels)
    out_2d = M.compute_metrics(np.stack([-logits_1d, logits_1d], axis=1), labels)

    for key in M.METRIC_KEYS:
        if out_1d[key] is None or out_2d[key] is None:
            assert out_1d[key] == out_2d[key]
        else:
            assert out_1d[key] == pytest.approx(out_2d[key])
    np.testing.assert_array_equal(out_1d["confusion_matrix"], out_2d["confusion_matrix"])


def test_torch_tensor_inputs_match_numpy():
    logits, labels = make_binary(33)
    out_np = M.compute_metrics(logits, labels)

    t_logits = torch.from_numpy(logits.astype(np.float32))
    t_labels = torch.from_numpy(labels.astype(np.int64))
    out_t = M.compute_metrics(t_logits, t_labels)

    for key in M.METRIC_KEYS:
        if out_np[key] is None:
            assert out_t[key] is None
        else:
            assert out_t[key] == pytest.approx(out_np[key], rel=1e-4, abs=1e-4)
    np.testing.assert_array_equal(out_np["confusion_matrix"], out_t["confusion_matrix"])


# --------------------------------------------------------------------------- #
# MetricAccumulator
# --------------------------------------------------------------------------- #
def test_metric_accumulator_matches_one_shot_binary():
    rs = np.random.RandomState(99)
    batches = [(rs.randn(17, 2), rs.randint(0, 2, size=17)) for _ in range(5)]

    acc = M.MetricAccumulator(num_classes=2, positive_class=1)
    for lg, lb in batches:
        acc.update(lg, lb, loss_sum=float(lg.shape[0]) * 0.5)

    all_logits = np.concatenate([b[0] for b in batches], axis=0)
    all_labels = np.concatenate([b[1] for b in batches], axis=0)
    expected = M.compute_metrics(all_logits, all_labels, num_classes=2, positive_class=1)

    got = acc.compute()
    assert len(acc) == len(all_labels)
    for key in M.METRIC_KEYS + ["n_examples"]:
        if expected[key] is None:
            assert got[key] is None
        else:
            assert got[key] == pytest.approx(expected[key], rel=1e-4, abs=1e-4)
    np.testing.assert_array_equal(got["confusion_matrix"], expected["confusion_matrix"])

    total_examples = sum(len(b[1]) for b in batches)
    expected_loss = (sum(b[0].shape[0] for b in batches) * 0.5) / total_examples
    assert got["loss"] == pytest.approx(expected_loss)


def test_metric_accumulator_empty():
    acc = M.MetricAccumulator(num_classes=2)
    out = acc.compute()
    assert out["n_examples"] == 0
    assert out["loss"] is None
    assert len(acc) == 0


# --------------------------------------------------------------------------- #
# to_flower_metrics / confusion_matrix_from_flat
# --------------------------------------------------------------------------- #
def test_to_flower_metrics_drops_none_and_types():
    logits, labels = np.zeros((20, 2)), np.zeros(20, dtype=int)  # single class -> roc_auc None
    metrics = M.compute_metrics(logits, labels)
    assert metrics["roc_auc"] is None

    flat = M.to_flower_metrics(metrics)
    assert "roc_auc" not in flat
    for k, v in flat.items():
        assert isinstance(v, (int, float, str, bool)), f"{k} has type {type(v)}"
        assert v is not None


def test_to_flower_metrics_flattens_cm_and_per_class():
    logits, labels = make_binary(3)
    metrics = M.compute_metrics(logits, labels)
    flat = M.to_flower_metrics(metrics)

    cm = np.asarray(metrics["confusion_matrix"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            assert flat[f"cm_{i}_{j}"] == int(cm[i, j])
    assert "confusion_matrix_json" in flat
    assert json.loads(flat["confusion_matrix_json"]) == cm.tolist()

    for cls, sub in metrics["per_class"].items():
        for mk, mv in sub.items():
            if mv is None:
                continue
            key = f"cls{cls}_{mk}"
            assert key in flat

    for i, s in enumerate(metrics["support"]):
        assert flat[f"support_{i}"] == int(s)


def test_confusion_matrix_from_flat_roundtrip():
    logits, labels = make_multiclass(44, n_classes=3)
    metrics = M.compute_metrics(logits, labels, num_classes=3)
    cm = np.asarray(metrics["confusion_matrix"])

    flat = M.to_flower_metrics(metrics)
    recovered = M.confusion_matrix_from_flat(flat)
    np.testing.assert_array_equal(recovered, cm)

    # fallback path: drop the JSON blob, keep only cm_i_j scalars
    flat_no_json = {k: v for k, v in flat.items() if k != "confusion_matrix_json"}
    recovered2 = M.confusion_matrix_from_flat(flat_no_json)
    np.testing.assert_array_equal(recovered2, cm)


def test_confusion_matrix_from_flat_missing_returns_none():
    assert M.confusion_matrix_from_flat({}) is None


def test_to_flower_metrics_prefix():
    logits, labels = make_binary(55)
    metrics = M.compute_metrics(logits, labels)
    flat = M.to_flower_metrics(metrics, prefix="val_")
    assert "val_accuracy" in flat
    assert "accuracy" not in flat


# --------------------------------------------------------------------------- #
# metrics_from_confusion_matrix on pooled / summed confusion matrices
# --------------------------------------------------------------------------- #
def test_metrics_from_summed_confusion_matrix_equals_pooled_compute():
    logits_a, labels_a = make_multiclass(101, n=120, n_classes=3)
    logits_b, labels_b = make_multiclass(202, n=90, n_classes=3)

    preds_a = logits_a.argmax(axis=1)
    preds_b = logits_b.argmax(axis=1)
    cm_a = M.confusion_matrix(labels_a, preds_a, 3)
    cm_b = M.confusion_matrix(labels_b, preds_b, 3)
    summed = cm_a + cm_b

    from_summed = M.metrics_from_confusion_matrix(summed)

    all_logits = np.concatenate([logits_a, logits_b], axis=0)
    all_labels = np.concatenate([labels_a, labels_b], axis=0)
    pooled = M.compute_metrics(all_logits, all_labels, num_classes=3)

    for key in ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision",
                "macro_recall", "macro_specificity", "mcc"):
        assert from_summed[key] == pytest.approx(pooled[key])
    np.testing.assert_array_equal(from_summed["confusion_matrix"], pooled["confusion_matrix"])
    assert from_summed["n_examples"] == pooled["n_examples"]


# --------------------------------------------------------------------------- #
# format_metrics smoke test
# --------------------------------------------------------------------------- #
def test_format_metrics_smoke():
    logits, labels = make_binary(1)
    metrics = M.compute_metrics(logits, labels)
    s = M.format_metrics(metrics)
    assert "accuracy=" in s
    assert isinstance(s, str)
