"""FedRGBD — the declared model-selection rule for federated runs.

Rule (paper, Section III, "Model selection"):

    The reported model is the one from the round with the lowest validation
    loss, aggregated across clients weighted by client validation-set size.
    Ties are broken toward the earlier round.  Test data is used exactly once
    per run, to report the metrics of that selected round, and never
    influences selection.

This module is the single implementation of that rule.  ``src/fl/server.py``
uses it to record ``model_selection`` in ``results.json`` and
``scripts/analyze_results.py`` uses it to pick the headline round.  Its inputs
are validation quantities only; no function here accepts test metrics.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Tuple

SELECTION_RULE = "min_weighted_val_loss_earliest_round"
SELECTION_RULE_TEXT = (
    "round with the lowest validation loss, aggregated across clients weighted by "
    "client validation-set size; ties broken toward the earlier round; test metrics "
    "are reported for that round only and never influence selection"
)


def _finite(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def weighted_val_loss(clients: Iterable[Tuple[float, float]]) -> Optional[float]:
    """Validation loss of one round: ``sum(n_k * loss_k) / sum(n_k)``.

    ``clients`` holds ``(n_val_examples, val_loss)`` per client.  Clients without
    validation examples carry no weight and are skipped.  If any client with
    examples has a missing or non-finite loss (it diverged), the round has no
    valid loss and ``None`` is returned: averaging over the surviving clients
    would make a broken round look good.
    """
    num = 0.0
    den = 0.0
    for n, loss in clients:
        n_f = _finite(n)
        if n_f is None or n_f <= 0:
            continue
        loss_f = _finite(loss)
        if loss_f is None:
            return None
        num += n_f * loss_f
        den += n_f
    return num / den if den > 0 else None


def select_round(val_loss_by_round: Mapping[int, float]) -> Optional[int]:
    """The round with the lowest weighted validation loss (earliest on ties).

    ``val_loss_by_round`` maps round -> weighted validation loss (see
    :func:`weighted_val_loss`).  Rounds whose loss is missing or non-finite are
    not eligible.  Returns ``None`` when no round is eligible.
    """
    best: Optional[Tuple[float, int]] = None
    for rnd, loss in val_loss_by_round.items():
        loss_f = _finite(loss)
        if loss_f is None:
            continue
        key = (loss_f, int(rnd))
        if best is None or key < best:
            best = key
    return None if best is None else best[1]


__all__ = [
    "SELECTION_RULE",
    "SELECTION_RULE_TEXT",
    "select_round",
    "weighted_val_loss",
]
