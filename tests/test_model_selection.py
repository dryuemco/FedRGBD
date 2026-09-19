"""Tests for the declared model-selection rule (src/evaluation/model_selection.py).

Rule: the reported model is the one from the round with the lowest validation
loss, aggregated across clients weighted by client validation-set size; ties go
to the earlier round; test data never influences selection.
"""

import math
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from src.evaluation.model_selection import (  # noqa: E402
    SELECTION_RULE,
    select_round,
    weighted_val_loss,
)


def test_selects_the_round_with_the_lowest_validation_loss():
    assert select_round({1: 0.50, 2: 0.21, 3: 0.35}) == 2
    assert select_round({1: 0.10, 2: 0.21, 3: 0.35}) == 1
    assert select_round({3: 0.05, 1: 0.50, 2: 0.21}) == 3   # dict order is irrelevant


def test_ties_are_broken_toward_the_earlier_round():
    assert select_round({1: 0.40, 2: 0.20, 3: 0.20}) == 2
    assert select_round({3: 0.20, 2: 0.20, 1: 0.20}) == 1
    assert select_round({5: 0.3, 10: 0.3}) == 5


def test_non_finite_rounds_are_not_eligible():
    assert select_round({1: float("nan"), 2: 0.4, 3: float("inf")}) == 2
    assert select_round({1: None, 2: 0.4}) == 2
    assert select_round({1: float("nan"), 2: None}) is None
    assert select_round({}) is None


def test_weighted_val_loss_uses_validation_set_sizes():
    # (n_val, val_loss): 100 * 0.2 + 300 * 0.6 = 200 / 400 = 0.5 (plain mean would be 0.4)
    assert weighted_val_loss([(100, 0.2), (300, 0.6)]) == pytest.approx(0.5)
    # a client without validation examples carries no weight
    assert weighted_val_loss([(0, 9.9), (50, 0.3)]) == pytest.approx(0.3)
    assert weighted_val_loss([]) is None


def test_a_diverged_client_invalidates_the_round():
    """Averaging over the surviving clients would make a broken round look good."""
    assert weighted_val_loss([(100, 0.2), (300, float("nan"))]) is None
    assert weighted_val_loss([(100, 0.2), (300, None)]) is None


def test_weighting_can_change_the_selected_round():
    """Unweighted means would pick round 1; the size-weighted rule picks round 2."""
    rounds = {
        1: [(100, 0.10), (900, 0.50)],   # weighted 0.46, unweighted mean 0.30
        2: [(100, 0.70), (900, 0.20)],   # weighted 0.25, unweighted mean 0.45
    }
    weighted = {r: weighted_val_loss(c) for r, c in rounds.items()}
    unweighted = {r: sum(l for _, l in c) / len(c) for r, c in rounds.items()}
    assert select_round(unweighted) == 1
    assert select_round(weighted) == 2


def test_rule_name_is_stable():
    # written into results.json and runs.csv; changing it silently would break provenance
    assert SELECTION_RULE == "min_weighted_val_loss_earliest_round"
    assert not math.isnan(weighted_val_loss([(1, 0.0)]))
