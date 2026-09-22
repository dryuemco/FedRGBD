#!/usr/bin/env python3
r"""FedRGBD -- turn the ``analyze_results.py`` CSVs into booktabs LaTeX tables.

``scripts/analyze_results.py`` writes ``runs.csv``, ``summary_table.csv``,
``per_round_table.csv``, ``pairwise_tests.csv`` and ``friedman.csv`` into its
``--output_dir``.  This script reads those files and renders paper-ready
tables, so no number in the manuscript is ever re-typed by hand:

    summary_<metric>.tex      mean +/- std [95% CI] (n) per config x distribution
    per_round_accuracy.tex    accuracy per communication round, grouped by dist.
    pairwise_tests.tex        paired strategy comparisons (d, Wilcoxon p, t p)
    friedman.tex              Friedman omnibus test per distribution
    full_metrics_<dist>.tex   selected-round / selected-epoch test metrics (rule-following runs)
    time.tex                  tab:time -- revision FL runs only (v1 timings omitted)

Every table uses ``booktabs`` (``\toprule`` / ``\midrule`` / ``\bottomrule``)
and carries a ``\caption`` and a ``\label``; ``\multicolumn`` groups the
per-distribution column pairs.  Text coming from the CSVs is LaTeX-escaped, so
``non_iid_label`` or ``95%`` cannot break the build.  With ``--siunitx`` the
numbers are wrapped in ``\num{...}`` (requires ``\usepackage{siunitx}``).

Usage
-----
    python3 scripts/analyze_results.py --results_dir results --output_dir analysis
    python3 scripts/export_latex_tables.py --analysis_dir analysis \
        --output_dir paper/tables

    # then, in the manuscript
    \input{tables/summary_final_accuracy.tex}
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

#: CSV files written by analyze_results.py (name -> file)
INPUT_FILES = {
    "runs": "runs.csv",
    "summary": "summary_table.csv",
    "per_round": "per_round_table.csv",
    "pairwise": "pairwise_tests.csv",
    "friedman": "friedman.csv",
}

#: summary metrics exported by default (those actually present are used).  The declared
#: primary metric (balanced accuracy at the selected round) comes first so that its table
#: is the one the paper leads with; accuracy stays, as the secondary metric.
DEFAULT_METRICS = ("selected_test_balanced_accuracy", "selected_test_mcc",
                   "selected_test_accuracy", "final_accuracy", "final_loss", "total_time_s",
                   "v1_final_round_accuracy")

#: column order of full_metrics_<dist>.tex.  Balanced accuracy leads because it is the
#: declared primary metric (Section "Primary Metric" of the paper, CLAUDE.md hard rule 8);
#: plain accuracy follows it as a secondary metric.  Do not reorder these two back.
FULL_METRIC_ORDER = (
    "final_balanced_accuracy", "final_mcc", "final_accuracy", "final_precision", "final_recall",
    "final_specificity", "final_f1", "final_macro_f1", "final_macro_precision",
    "final_macro_recall", "final_macro_specificity", "final_roc_auc",
    "final_loss",
)

METRIC_LABELS = {
    "final_accuracy": "Accuracy",
    "selected_test_accuracy": "Test accuracy (selected round)",
    "selected_test_balanced_accuracy": "Test balanced accuracy (selected round)",
    "selected_test_mcc": "Test MCC (selected round)",
    "selected_round": "Selected round",
    "selected_val_loss": "Validation loss (selected round)",
    "v1_final_round_accuracy": "v1 final-round accuracy (validation)",
    "v1_final_round_loss": "v1 final-round loss (validation)",
    "round1_accuracy": "Round-1 accuracy (validation)",
    "final_loss": "Loss",
    "final_f1": "F1",
    "final_macro_f1": "Macro F1",
    "final_mcc": "MCC",
    "final_roc_auc": "ROC AUC",
    "final_balanced_accuracy": "Balanced accuracy",
    "final_precision": "Precision",
    "final_recall": "Recall",
    "final_specificity": "Specificity",
    "total_time_s": "Wall-clock time (s)",
    "final_elapsed_s": "Elapsed time (s)",
    "final_cumulative_mb": "Communication (MB)",
}

DIST_LABELS = {
    "iid": "IID",
    "non_iid_label": "Non-IID (label skew)",
    "unknown": "unknown",
}

#: metrics measured in seconds -> --seconds_digits instead of --digits
_SECONDS_RE = re.compile(r"(_s|_time_s|seconds)$")

_KIND_ORDER = {"fl": 0, "centralized": 1, "local": 2}

_LATEX_ESCAPES = (
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
)

MISSING = "--"


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def escape_latex(text: Any) -> str:
    """Escape the LaTeX specials in a CSV string (``_``, ``%``, ``&``, ...)."""
    out = "" if text is None else str(text)
    for char, replacement in _LATEX_ESCAPES:
        out = out.replace(char, replacement)
    return out


def pretty_name(text: Any) -> str:
    r"""Escape a configuration / strategy name and typeset ``mu=`` as ``$\mu$=``."""
    out = escape_latex(text)
    return out.replace("(mu=", r"($\mu$=").replace("mu=", r"$\mu$=")


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def digits_for(metric: str, digits: int, seconds_digits: int) -> int:
    """Seconds-valued metrics get ``--seconds_digits``, everything else ``--digits``."""
    name = str(metric or "")
    if _SECONDS_RE.search(name) or "time" in name:
        return seconds_digits
    return digits


def fmt_number(value: Any, digits: int = 4, siunitx: bool = False) -> str:
    """One number, ``--`` when missing; large values never explode the column."""
    number = _as_float(value)
    if number is None:
        return MISSING
    if abs(number) >= 1e5:
        text = "{:.2e}".format(number)
    elif abs(number) >= 1000:
        text = "{:.1f}".format(number)
    else:
        text = "{:.{d}f}".format(number, d=max(int(digits), 0))
    return r"\num{" + text + "}" if siunitx else text


def fmt_mean_std(mean: Any, std: Any, digits: int = 4, siunitx: bool = False) -> str:
    """``mean $\\pm$ std`` (the ``$\\pm$`` part is dropped for a single seed)."""
    mean_text = fmt_number(mean, digits, siunitx)
    if mean_text == MISSING:
        return MISSING
    std_text = fmt_number(std, digits, siunitx)
    if std_text == MISSING:
        return mean_text
    return "{} $\\pm$ {}".format(mean_text, std_text)


def fmt_ci(low: Any, high: Any, digits: int = 4, siunitx: bool = False) -> str:
    low_text = fmt_number(low, digits, siunitx)
    high_text = fmt_number(high, digits, siunitx)
    if low_text == MISSING or high_text == MISSING:
        return MISSING
    return "[{}, {}]".format(low_text, high_text)


def stars(p_value: Any) -> str:
    """``***`` p<0.001, ``**`` p<0.01, ``*`` p<0.05, empty otherwise."""
    p = _as_float(p_value)
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_p(p_value: Any, digits: int = 4, siunitx: bool = False, with_stars: bool = True) -> str:
    """p-value with significance stars; tiny values become ``$<$0.0001``."""
    p = _as_float(p_value)
    if p is None:
        return MISSING
    floor = 10.0 ** (-max(int(digits), 1))
    if 0 <= p < floor:
        text = "$<${}".format(("{:.%df}" % max(int(digits), 1)).format(floor))
    else:
        text = fmt_number(p, digits, siunitx)
    if with_stars:
        marks = stars(p)
        if marks:
            text += "$^{" + marks + "}$"
    return text


def metric_label(metric: str) -> str:
    """Human-readable, LaTeX-escaped metric name."""
    if metric in METRIC_LABELS:
        return METRIC_LABELS[metric]
    core = metric[len("final_"):] if str(metric).startswith("final_") else str(metric)
    return escape_latex(core.replace("_", " ").capitalize())


def dist_label(distribution: str) -> str:
    if distribution in DIST_LABELS:
        return DIST_LABELS[distribution]
    if str(distribution).startswith("dirichlet_"):
        return "Dirichlet($\\alpha$={})".format(escape_latex(str(distribution).split("_", 1)[1]))
    return escape_latex(distribution)


def config_name(label: Any, distribution: Any) -> str:
    """``analyze_results`` labels embed the distribution -- strip it for the rows.

    The ``{group}`` / ``{image}`` protocol suffix is rendered as a short marker
    so that v1 (image-level) and revision (group-level) rows stay distinct."""
    text = str(label or "")
    text = text.replace("{group_final_epoch}", "(group-level, final epoch)")
    text = text.replace("{group}", "(group-level)").replace("{image}", "(image-level)")
    dist = str(distribution or "")
    if dist and dist in text:
        text = text.replace(dist, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return pretty_name(text or str(label or "config"))


def _config_sort_key(kind: Any, name: str) -> Tuple[int, str]:
    return (_KIND_ORDER.get(str(kind), 9), name)


# --------------------------------------------------------------------------- #
# table skeleton
# --------------------------------------------------------------------------- #
def latex_table(column_spec: str, header_lines: Sequence[str], body_lines: Sequence[str],
                caption: str, label: str, note: str = "",
                position: str = "t", small: bool = False) -> str:
    """A complete ``table`` float with ``booktabs`` rules."""
    out: List[str] = []
    out.append("% generated by scripts/export_latex_tables.py -- do not edit by hand")
    out.append("\\begin{table}[" + position + "]")
    out.append("\\centering")
    out.append("\\caption{" + caption + "}")
    out.append("\\label{" + label + "}")
    if small:
        out.append("\\small")
    out.append("\\begin{tabular}{" + column_spec + "}")
    out.append("\\toprule")
    out.extend(header_lines)
    out.append("\\midrule")
    out.extend(body_lines)
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    if note:
        out.append("\\par\\smallskip")
        out.append("\\footnotesize " + note)
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def _row(cells: Sequence[str]) -> str:
    return " & ".join(cells) + " \\\\"


def _group_row(n_columns: int, text: str) -> str:
    return "\\multicolumn{" + str(n_columns) + "}{l}{\\textit{" + text + "}} \\\\"


# --------------------------------------------------------------------------- #
# summary_<metric>.tex
# --------------------------------------------------------------------------- #
def summary_tex(df: pd.DataFrame, metric: str, digits: int = 4, seconds_digits: int = 1,
                siunitx: bool = False) -> Optional[str]:
    """Rows = configuration, column pair = one data distribution."""
    sub = df[df["metric"] == metric]
    if sub.empty:
        return None
    ndigits = digits_for(metric, digits, seconds_digits)
    dists = sorted(sub["distribution"].astype(str).unique())

    cells: Dict[Tuple[str, str], Dict[str, Any]] = {}
    order: Dict[str, Tuple[int, str]] = {}
    for _, row in sub.iterrows():
        name = config_name(row.get("label"), row.get("distribution"))
        cells[(name, str(row["distribution"]))] = row
        order.setdefault(name, _config_sort_key(row.get("kind"), name))
    names = sorted(order, key=lambda n: order[n])

    header = [
        _row([""] + ["\\multicolumn{2}{c}{" + dist_label(d) + "}" for d in dists]),
        " ".join("\\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(len(dists))),
        _row(["Configuration"] + ["mean $\\pm$ std", "95\\% CI ($n$)"] * len(dists)),
    ]

    body: List[str] = []
    for name in names:
        row_cells = [name]
        for dist in dists:
            row = cells.get((name, dist))
            if row is None:
                row_cells.extend([MISSING, MISSING])
                continue
            row_cells.append(fmt_mean_std(row.get("mean"), row.get("std"), ndigits, siunitx))
            ci = fmt_ci(row.get("ci_low"), row.get("ci_high"), ndigits, siunitx)
            if str(row.get("ci_method", "")).startswith("cluster_bootstrap") and ci != MISSING:
                ci += "$^{b}$"
            n_seeds = _as_float(row.get("n_seeds"))
            row_cells.append("{} ({})".format(ci, int(n_seeds) if n_seeds else 0))
        body.append(_row(row_cells))

    return latex_table(
        column_spec="l" + " cc" * len(dists),
        header_lines=header,
        body_lines=body,
        caption="{} per configuration and data distribution: mean $\\pm$ standard deviation "
                "over seeds, with the 95\\% confidence interval of the mean and the number "
                "of seeds $n$.".format(metric_label(metric)),
        label="tab:summary_{}".format(_safe_label(metric)),
        note="Produced from \\texttt{summary\\_table.csv}; $\\pm$ and CI are omitted when "
             "only one seed is available. CI: 95\\% $t$-interval of the mean over seeds, or, "
             "marked $^{b}$, the sequence-level (cluster) bootstrap over seeds and held-out "
             "sequences.",
        small=len(dists) > 1,
    )


# --------------------------------------------------------------------------- #
# per_round_accuracy.tex
# --------------------------------------------------------------------------- #
def per_round_tex(df: pd.DataFrame, metric: str = "accuracy", digits: int = 4,
                  max_rounds: int = 10, siunitx: bool = False) -> Optional[str]:
    """Rows = configuration (grouped by distribution), columns = rounds."""
    mean_col = "{}_mean".format(metric)
    std_col = "{}_std".format(metric)
    if df.empty or mean_col not in df.columns:
        return None
    sub = df[df[mean_col].notna()]
    if sub.empty:
        return None

    rounds = sorted({int(r) for r in sub["round"]})
    if len(rounds) > max_rounds:
        step = max(1, int(math.ceil(len(rounds) / float(max_rounds))))
        kept = rounds[::step]
        if rounds[-1] not in kept:
            kept.append(rounds[-1])
        rounds = kept

    header = [
        _row(["", "\\multicolumn{%d}{c}{Communication round}" % len(rounds)]),
        "\\cmidrule(lr){2-%d}" % (len(rounds) + 1),
        _row(["Configuration"] + [str(r) for r in rounds]),
    ]

    body: List[str] = []
    n_columns = len(rounds) + 1
    for dist in sorted(sub["distribution"].astype(str).unique()):
        block = sub[sub["distribution"].astype(str) == dist]
        if body:
            body.append("\\midrule")
        body.append(_group_row(n_columns, dist_label(dist)))
        entries: Dict[str, Dict[int, Any]] = {}
        order: Dict[str, Tuple[int, str]] = {}
        for _, row in block.iterrows():
            name = config_name(row.get("label"), dist)
            entries.setdefault(name, {})[int(row["round"])] = row
            order.setdefault(name, _config_sort_key(row.get("kind"), name))
        for name in sorted(order, key=lambda n: order[n]):
            cells = [name]
            for rnd in rounds:
                row = entries[name].get(rnd)
                if row is None:
                    cells.append(MISSING)
                else:
                    cells.append(fmt_mean_std(row.get(mean_col), row.get(std_col),
                                              digits, siunitx))
            body.append(_row(cells))

    return latex_table(
        column_spec="l" + "c" * len(rounds),
        header_lines=header,
        body_lines=body,
        caption="Global {} per communication round (mean $\\pm$ standard deviation over "
                "seeds).".format(escape_latex(metric.replace("_", " "))),
        label="tab:per_round_{}".format(_safe_label(metric)),
        note="Produced from \\texttt{per\\_round\\_table.csv}. Centralized and local-only "
             "baselines are mapped onto rounds via their local-epoch equivalent.",
        small=True,
    )


# --------------------------------------------------------------------------- #
# pairwise_tests.tex
# --------------------------------------------------------------------------- #
_PROTOCOL_TEXT = {"group": "group-level", "image": "image-level, v1",
                  "group_final_epoch": "group-level, final epoch"}


def _protocol_column(df: pd.DataFrame) -> pd.Series:
    """The ``protocol`` column as strings ('' for CSVs written before it existed)."""
    if "protocol" in df.columns:
        return df["protocol"].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)


def _protocol_suffix(protocol: str) -> str:
    return " ({})".format(_PROTOCOL_TEXT.get(protocol, escape_latex(protocol))) if protocol else ""


def pairwise_tex(df: pd.DataFrame, digits: int = 4, siunitx: bool = False) -> Optional[str]:
    """Paired strategy comparisons, one block per data distribution."""
    if df.empty:
        return None
    header = [
        _row(["Comparison", "$n$", "mean A", "mean B", "$\\Delta$", "$d_z$",
              "Wilcoxon $p$", "$t$-test $p$"]),
    ]
    body: List[str] = []
    n_columns = 8
    protocols = _protocol_column(df)
    for protocol, dist in sorted(set(zip(protocols, df["distribution"].astype(str)))):
        block = df[(protocols == protocol) & (df["distribution"].astype(str) == dist)]
        if body:
            body.append("\\midrule")
        body.append(_group_row(n_columns, dist_label(dist) + _protocol_suffix(protocol)))
        for _, row in block.iterrows():
            n_seeds = _as_float(row.get("n_seeds"))
            body.append(_row([
                "{} vs.\\ {}".format(pretty_name(row.get("strategy_a")),
                                     pretty_name(row.get("strategy_b"))),
                str(int(n_seeds) if n_seeds else 0),
                fmt_number(row.get("mean_a"), digits, siunitx),
                fmt_number(row.get("mean_b"), digits, siunitx),
                fmt_number(row.get("mean_diff"), digits, siunitx),
                fmt_number(row.get("cohen_d_paired"), 3, siunitx),
                fmt_p(row.get("wilcoxon_p"), digits, siunitx),
                fmt_p(row.get("ttest_p"), digits, siunitx),
            ]))

    metric = str(df["metric"].iloc[0]) if "metric" in df.columns and not df.empty else "accuracy"
    return latex_table(
        column_spec="l" + "c" * 7,
        header_lines=header,
        body_lines=body,
        caption="Pairwise strategy comparisons on the headline {}, paired by seed within each "
                "partitioning protocol and data distribution.".format(
                    escape_latex(metric.replace("_", " "))),
        label="tab:pairwise_tests",
        note="Headline value per run: test metric of the selected round (revision "
             "federated runs), final-epoch test metric (centralized, local-only), final-round "
             "validation metric (v1 federated runs). "
             "$\\Delta$ = mean(A) $-$ mean(B); $d_z$ = mean(diff) / std(diff). "
             "Significance: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ (uncorrected). "
             "Produced from \\texttt{pairwise\\_tests.csv}.",
        small=True,
    )


# --------------------------------------------------------------------------- #
# friedman.tex
# --------------------------------------------------------------------------- #
def friedman_tex(df: pd.DataFrame, digits: int = 4, siunitx: bool = False) -> Optional[str]:
    if df.empty:
        return None
    header = [_row(["Distribution", "$k$ strategies", "$n$ seeds", "$\\chi^2$", "$p$"])]
    body: List[str] = []
    protocols = _protocol_column(df)
    for idx, row in df.iterrows():
        n_strategies = _as_float(row.get("n_strategies"))
        n_seeds = _as_float(row.get("n_seeds"))
        body.append(_row([
            dist_label(str(row.get("distribution"))) + _protocol_suffix(protocols[idx]),
            str(int(n_strategies) if n_strategies else 0),
            str(int(n_seeds) if n_seeds else 0),
            fmt_number(row.get("chi_square"), 3, siunitx),
            fmt_p(row.get("p_value"), digits, siunitx),
        ]))
    return latex_table(
        column_spec="lcccc",
        header_lines=header,
        body_lines=body,
        caption="Friedman omnibus test over the strategies sharing the same seeds, per data "
                "distribution.",
        label="tab:friedman",
        note="Significance: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. "
             "Produced from \\texttt{friedman.csv}.",
    )


# --------------------------------------------------------------------------- #
# full_metrics_<dist>.tex
# --------------------------------------------------------------------------- #
def selected_counterpart(metric: str) -> str:
    """``final_<m>`` (baselines) -> ``selected_test_<m>`` (revision FL runs)."""
    return "selected_test_" + metric[len("final_"):] if metric.startswith("final_") else metric


def available_full_metrics(df: pd.DataFrame) -> List[str]:
    present = set(df["metric"].astype(str).unique()) if not df.empty else set()
    return [m for m in FULL_METRIC_ORDER if selected_counterpart(m) in present]


def full_metrics_tex(df: pd.DataFrame, distribution: str, metrics: Sequence[str],
                     digits: int = 4, siunitx: bool = False) -> Optional[str]:
    """Rows = configuration, columns = every headline test metric of one distribution.

    Only runs that follow the declared selection rule appear: federated rows show
    the test metrics of the selected round, centralized / local-only rows (schema
    3) those of the selected epoch -- all ``selected_test_<m>``.  Final-epoch
    baselines (``final_<m>``) and v1 FL rows follow other rules and are left out,
    so no column mixes selection rules.
    """
    wanted = {selected_counterpart(m) for m in metrics}
    sub = df[(df["distribution"].astype(str) == str(distribution))
             & (df["metric"].astype(str).isin(list(wanted)))]
    if sub.empty:
        return None

    entries: Dict[str, Dict[str, Any]] = {}
    order: Dict[str, Tuple[int, str]] = {}
    for _, row in sub.iterrows():
        name = config_name(row.get("label"), distribution)
        entries.setdefault(name, {})[str(row["metric"])] = row
        order.setdefault(name, _config_sort_key(row.get("kind"), name))

    header = [_row(["Configuration"] + [metric_label(m) for m in metrics])]
    body: List[str] = []
    for name in sorted(order, key=lambda n: order[n]):
        cells = [name]
        for metric in metrics:
            row = entries[name].get(selected_counterpart(metric))
            cells.append(MISSING if row is None
                         else fmt_mean_std(row.get("mean"), row.get("std"), digits, siunitx))
        body.append(_row(cells))

    return latex_table(
        column_spec="l" + "c" * len(metrics),
        header_lines=header,
        body_lines=body,
        caption="Test-set metrics for the {} partition (mean $\\pm$ standard deviation "
                "over seeds).".format(dist_label(distribution)),
        label="tab:full_metrics_{}".format(_safe_label(distribution)),
        note="Test metrics of the round (federated) or epoch (centralized, local-only) with "
             "the lowest validation loss (weighted by client validation-set size, earlier "
             "on ties). Produced from \\texttt{summary\\_table.csv}.",
        small=len(metrics) > 3,
    )


# --------------------------------------------------------------------------- #
# time.tex  (tab:time -- revision runs only)
# --------------------------------------------------------------------------- #
#: the default operating point of the main comparison (Section III-J)
TIME_TABLE_POINT = {"num_rounds": 3, "local_epochs": 5, "lr": 0.001}
TIME_TABLE_DISTS = ("iid", "non_iid_label")


def _protocol_of(row: Any) -> str:
    """``protocol`` column, else the ``{group}`` / ``{image}`` marker of the label."""
    value = row.get("protocol") if hasattr(row, "get") else None
    if isinstance(value, str) and value:
        return value
    label = str(row.get("label") or "")
    if "{group_final_epoch}" in label:
        return "group_final_epoch"
    if "{group}" in label:
        return "group"
    if "{image}" in label:
        return "image"
    return ""


def _at_point(row: Any, field: str, target: float) -> bool:
    """True when ``field`` equals ``target`` or is missing (older summary CSVs)."""
    value = _as_float(row.get(field))
    return value is None or math.isclose(value, target, rel_tol=1e-9, abs_tol=1e-12)


def time_tex(df: pd.DataFrame, siunitx: bool = False) -> Optional[str]:
    """``tab:time``: wall-clock time and communication of the revision FL runs.

    Only group-level-split (revision) federated runs at the default operating
    point are included.  v1 runs are excluded on purpose: they were timed over
    WiFi and under the image-level split, so their wall-clock times are not
    comparable with the revision runs (wired Gigabit Ethernet).
    """
    if df.empty or "metric" not in df.columns:
        return None
    rows: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for _, row in df.iterrows():
        if str(row.get("kind")) != "fl" or _protocol_of(row) != "group":
            continue
        dist = str(row.get("distribution"))
        if dist not in TIME_TABLE_DISTS:
            continue
        if not all(_at_point(row, f, v) for f, v in TIME_TABLE_POINT.items()):
            continue
        n_nodes = _as_float(row.get("n_nodes"))
        name = config_name(row.get("label"), dist).replace(" (group-level)", "")
        name = re.sub(r"\s*\[\d+N\]", "", name).strip()
        key = (int(n_nodes) if n_nodes else 0, dist, name)
        rows.setdefault(key, {})[str(row["metric"])] = row
    rows = {k: v for k, v in rows.items() if "total_time_s" in v}
    if not rows:
        return None

    header = [_row(["Configuration", "Time (min)", "Comm.\\ (MB)", "Test acc.\\ (\\%)"])]
    body: List[str] = []
    for (n_nodes, dist, name) in sorted(rows, key=lambda k: (k[0], TIME_TABLE_DISTS.index(k[1]),
                                                              k[2])):
        metrics = rows[(n_nodes, dist, name)]
        t = metrics["total_time_s"]
        minutes = {k: (_as_float(t.get(k)) / 60.0 if _as_float(t.get(k)) is not None else None)
                   for k in ("mean", "std", "ci_low", "ci_high")}
        n_seeds = _as_float(t.get("n_seeds"))
        time_cell = "{} {} ({})".format(
            fmt_mean_std(minutes["mean"], minutes["std"], 1, siunitx),
            fmt_ci(minutes["ci_low"], minutes["ci_high"], 1, siunitx),
            int(n_seeds) if n_seeds else 0).replace(" -- (", " (")
        comm = metrics.get("final_cumulative_mb")
        comm_cell = fmt_number(comm.get("mean"), 1, siunitx) if comm is not None else MISSING
        acc = metrics.get("selected_test_accuracy")
        acc_cell = MISSING
        if acc is not None and _as_float(acc.get("mean")) is not None:
            std = _as_float(acc.get("std"))
            acc_cell = fmt_mean_std(100.0 * _as_float(acc.get("mean")),
                                    100.0 * std if std is not None else None, 2, siunitx)
        body.append(_row(["{}N {} {}".format(n_nodes, dist_label(dist), name),
                          time_cell, comm_cell, acc_cell]))

    return latex_table(
        column_spec="lccc",
        header_lines=header,
        body_lines=body,
        caption="Measured Total Training Time, Group-Level Split, Wired Gigabit Ethernet "
                "(3 Rounds)",
        label="tab:time",
        small=True,
        note="Time: mean $\\pm$ std [95\\% CI] ($n$ seeds) of the server wall-clock time "
             "excluding the report-only test evaluation. "
             "Communication: measured cumulative payload over the run. Test accuracy: at the "
             "round selected by the lowest weighted validation loss. Only revision runs "
             "(group-level split, wired Gigabit Ethernet) are included; v1 timings (WiFi, "
             "image-level split) are not comparable and are omitted. Produced from "
             "\\texttt{summary\\_table.csv}.",
    )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _safe_label(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower() or "table"


def load_tables(analysis_dir: str, warn: bool = True) -> Dict[str, pd.DataFrame]:
    """Read the analyze_results CSVs; a missing file becomes an empty frame."""
    out: Dict[str, pd.DataFrame] = {}
    for key, name in INPUT_FILES.items():
        path = os.path.join(analysis_dir, name)
        if os.path.isfile(path):
            try:
                out[key] = pd.read_csv(path)
                continue
            except Exception as exc:  # pragma: no cover - corrupted CSV
                if warn:
                    print("[warn] could not read {}: {}".format(path, exc))
        elif warn:
            print("[warn] {} not found in {}".format(name, analysis_dir))
        out[key] = pd.DataFrame()
    return out


def select_metrics(summary: pd.DataFrame, requested: Optional[Sequence[str]]) -> List[str]:
    """``--metrics`` (or ``all``) intersected with what the summary really holds."""
    present = list(dict.fromkeys(summary["metric"].astype(str))) if not summary.empty else []
    if requested and len(requested) == 1 and str(requested[0]).lower() == "all":
        return present
    wanted = list(requested) if requested else list(DEFAULT_METRICS)
    return [m for m in wanted if m in present]


def export(analysis_dir: str, output_dir: str, metrics: Optional[Sequence[str]] = None,
           digits: int = 4, seconds_digits: int = 1, max_rounds: int = 10,
           siunitx: bool = False, per_round_metric: str = "accuracy",
           warn: bool = True) -> List[str]:
    """Write every table; returns the paths that were written."""
    tables = load_tables(analysis_dir, warn=warn)
    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []

    def _write(text: Optional[str], name: str) -> None:
        if not text:
            return
        path = os.path.join(output_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        written.append(path)

    summary = tables["summary"]
    for metric in select_metrics(summary, metrics):
        _write(summary_tex(summary, metric, digits, seconds_digits, siunitx),
               "summary_{}.tex".format(_safe_label(metric)))

    _write(per_round_tex(tables["per_round"], per_round_metric, digits, max_rounds, siunitx),
           "per_round_{}.tex".format(_safe_label(per_round_metric)))
    _write(pairwise_tex(tables["pairwise"], digits, siunitx), "pairwise_tests.tex")
    _write(friedman_tex(tables["friedman"], digits, siunitx), "friedman.tex")
    _write(time_tex(summary, siunitx), "time.tex")

    # the leakage-definition sensitivity table, generated by scripts/clean_subset.py
    nn_table = os.path.join(analysis_dir, "leakage", "clean_subset", "nearest_train_distance.tex")
    if os.path.isfile(nn_table):
        with open(nn_table, encoding="utf-8") as fh:
            _write(fh.read(), "nn_distance.tex")

    # the grouping trade-off table, generated by scripts/grouping_tradeoff.py
    gt_table = os.path.join(analysis_dir, "leakage", "grouping_tradeoff.tex")
    if os.path.isfile(gt_table):
        with open(gt_table, encoding="utf-8") as fh:
            _write(fh.read(), "grouping_tradeoff.tex")

    full_metrics = available_full_metrics(summary)
    if len(full_metrics) > 1:
        for dist in sorted(summary["distribution"].astype(str).unique()):
            _write(full_metrics_tex(summary, dist, full_metrics, digits, siunitx),
                   "full_metrics_{}.tex".format(_safe_label(dist)))
    elif warn:
        print("[info] only {} final metric(s) in the summary; full_metrics_<dist>.tex is "
              "written for schema-2 runs only".format(len(full_metrics)))

    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the analyze_results.py CSVs as booktabs LaTeX tables.")
    parser.add_argument("--analysis_dir", default="analysis",
                        help="directory holding runs.csv / summary_table.csv / ... "
                             "(default: analysis)")
    parser.add_argument("--output_dir", default="paper/tables",
                        help="where the .tex files are written (default: paper/tables)")
    parser.add_argument("--metrics", nargs="+", default=None, metavar="NAME",
                        help="summary metrics to export, or 'all' (default: {})".format(
                            ", ".join(DEFAULT_METRICS)))
    parser.add_argument("--per_round_metric", default="accuracy",
                        help="metric of per_round_<metric>.tex (default: accuracy)")
    parser.add_argument("--digits", type=int, default=4,
                        help="decimals for metric values (default: 4)")
    parser.add_argument("--seconds_digits", type=int, default=1,
                        help="decimals for second-valued metrics (default: 1)")
    parser.add_argument("--max_rounds", type=int, default=10,
                        help="maximum number of round columns before thinning (default: 10)")
    parser.add_argument("--siunitx", action="store_true",
                        help="wrap numbers in \\num{} (requires \\usepackage{siunitx})")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    written = export(
        args.analysis_dir, args.output_dir, metrics=args.metrics, digits=args.digits,
        seconds_digits=args.seconds_digits, max_rounds=args.max_rounds,
        siunitx=args.siunitx, per_round_metric=args.per_round_metric)
    if not written:
        print("[warn] no table written -- is {} the output_dir of "
              "analyze_results.py?".format(args.analysis_dir))
        return 1
    print("wrote {} LaTeX table(s) to {}".format(len(written), os.path.abspath(args.output_dir)))
    for path in written:
        print("    {}".format(os.path.basename(path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
