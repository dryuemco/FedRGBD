"""The baseline numbers typed into paper/main.tex must match analysis/summary_table.csv.

CLAUDE.md rule 4 says no number is typed into the paper by hand.  Two tables --
``tab:protocol_effect`` and ``tab:fullmetrics`` -- still carry their values
inline, because they also hold ``\\PH`` placeholder rows for the federated
strategies that have not run yet and so cannot simply ``\\input`` a generated
table.  This test is what makes that safe: it re-derives every one of those
cells from ``analysis/summary_table.csv`` and fails if the paper and the
analysis ever disagree.

When the Jetson runs land and the tables are regenerated, this test is the
thing that tells you which hand-typed cells are now stale.
"""

import os
import re

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_TEX = os.path.join(ROOT, "paper", "main.tex")
SUMMARY = os.path.join(ROOT, "analysis", "summary_table.csv")

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(MAIN_TEX) and os.path.isfile(SUMMARY)),
    reason="paper/main.tex and analysis/summary_table.csv are both needed")

# metric column of tab:fullmetrics -> metric name in summary_table.csv, in the table's
# column order.  The table prints percentages for everything except MCC.
# Balanced accuracy and MCC lead: they are the declared primary metric and the second
# summary statistic (CLAUDE.md hard rule 8).  Keep this list in the table's order.
FULLMETRICS_COLUMNS = [
    ("selected_test_balanced_accuracy", 100.0),
    ("selected_test_mcc", 1.0),
    ("selected_test_accuracy", 100.0),
    ("selected_test_recall", 100.0),
    ("selected_test_specificity", 100.0),
    ("selected_test_macro_f1", 100.0),
    ("selected_test_roc_auc", 100.0),
]


def _summary():
    df = pd.read_csv(SUMMARY)
    return df[df.protocol == "group"]


def _cells(line):
    """The ``$a \\pm b$`` pairs of one LaTeX table row, in order."""
    return [(float(a), float(b)) for a, b in
            re.findall(r"\$([0-9.]+) \\pm ([0-9.]+)\$", line)]


def _row(tex, table_label, method, distribution_block):
    """Find one ``& <method> &`` row inside the named table environment."""
    start = tex.index(r"\label{%s}" % table_label)
    end = tex.index(r"\end{table", start)
    block = tex[start:end]
    # the non-IID rows come after the \midrule that follows the IID ones
    halves = block.split(r"\multirow")
    half = halves[1] if distribution_block == "iid" else halves[2]
    for line in half.splitlines():
        if re.search(r"&\s*%s\s*&" % re.escape(method), line):
            return line
    raise AssertionError("no %r row in %s (%s)" % (method, table_label, distribution_block))


def _lookup(summary, distribution, kind, metric):
    sel = summary[(summary.distribution == distribution) & (summary.kind == kind)
                  & (summary.metric == metric)]
    assert len(sel) == 1, "expected one %s/%s/%s row, got %d" % (
        distribution, kind, metric, len(sel))
    return float(sel.iloc[0]["mean"]), float(sel.iloc[0]["std"])


@pytest.mark.parametrize("distribution,kind,method", [
    ("iid", "centralized", "Centralized"),
    ("iid", "local", "Local-only"),
    ("non_iid_label", "centralized", "Centralized"),
    ("non_iid_label", "local", "Local-only"),
])
def test_fullmetrics_row_matches_summary(distribution, kind, method):
    tex = open(MAIN_TEX, encoding="utf-8").read()
    summary = _summary()
    line = _row(tex, "tab:fullmetrics", method, distribution)
    cells = _cells(line)
    assert len(cells) == len(FULLMETRICS_COLUMNS), (
        "tab:fullmetrics %s/%s has %d numeric cells, expected %d"
        % (distribution, method, len(cells), len(FULLMETRICS_COLUMNS)))

    for (printed_mean, printed_std), (metric, scale) in zip(cells, FULLMETRICS_COLUMNS):
        mean, std = _lookup(summary, distribution, kind, metric)
        digits = 2 if scale == 1.0 else 1
        assert printed_mean == pytest.approx(round(mean * scale, digits), abs=0.05), (
            "%s %s %s: paper says %.2f, analysis says %.4f"
            % (distribution, method, metric, printed_mean, mean * scale))
        assert printed_std == pytest.approx(round(std * scale, digits), abs=0.05), (
            "%s %s %s std: paper says %.2f, analysis says %.4f"
            % (distribution, method, metric, printed_std, std * scale))


@pytest.mark.parametrize("distribution,kind,method", [
    ("iid", "centralized", "Centralized"),
    ("iid", "local", "Local-only"),
    ("non_iid_label", "centralized", "Centralized"),
    ("non_iid_label", "local", "Local-only"),
])
def test_protocol_effect_group_column_matches_summary(distribution, kind, method):
    """The group-level column of tab:protocol_effect is selected-round accuracy."""
    tex = open(MAIN_TEX, encoding="utf-8").read()
    summary = _summary()
    line = _row(tex, "tab:protocol_effect", method, distribution)
    cells = _cells(line)
    assert len(cells) == 2, "expected an image-level and a group-level cell"

    mean, std = _lookup(summary, distribution, kind, "selected_test_accuracy")
    printed_mean, printed_std = cells[1]
    assert printed_mean == pytest.approx(round(mean * 100, 1), abs=0.05), (
        "%s %s: paper says %.2f, analysis says %.4f"
        % (distribution, method, printed_mean, mean * 100))
    assert printed_std == pytest.approx(round(std * 100, 1), abs=0.05)


def test_paper_does_not_report_the_superseded_final_epoch_references():
    """78.3 % / 90.9 % may appear only where they are named as the old protocol.

    They were the headline references before the selection rule was declared.
    Any *unlabelled* occurrence is a cell that was missed when the tables were
    regenerated.
    """
    tex = open(MAIN_TEX, encoding="utf-8").read()
    for value in ("78.3", "90.9"):
        # a whole value, so that unrelated v1 numbers such as 78.36 do not match
        for match in re.finditer(re.escape(value) + r"(?!\d)", tex):
            window = tex[max(0, match.start() - 400):match.start() + 200]
            assert "final-epoch" in window or "final epoch" in window, (
                "%r appears at offset %d without being labelled as the superseded "
                "final-epoch protocol" % (value, match.start()))
