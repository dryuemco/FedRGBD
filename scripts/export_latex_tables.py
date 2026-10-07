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

#: Jetson power configuration written into the tables (analyze_results.py
#: ``power_config``).  One export never mixes two: rows of other configurations are
#: dropped before any table is built.  The desktop baselines (``desktop_gpu``) and the
#: v1 runs (``unrecorded``) carry no Jetson configuration and appear in every export.
DEFAULT_POWER_CONFIG = "heterogeneous"
POWER_SHARED = ("desktop_gpu", "unrecorded")


def select_power_config(df: pd.DataFrame, power_config: str) -> pd.DataFrame:
    """Rows of ``power_config`` plus the rows that have none (see ``POWER_SHARED``)."""
    if df.empty or "power_config" not in df.columns:
        return df
    power = df["power_config"].fillna("").astype(str)
    return df[(power == power_config) | power.isin(POWER_SHARED) | (power == "")]


def claim_cell(seen: Dict[Any, str], key: Any, row: Any) -> None:
    """Refuse to let two configurations render into the same table cell.

    Tables are keyed by display name; if two configurations ever share one (a sweep
    variant without its suffix, two power configurations in one export), the later
    row would silently overwrite the earlier one.
    """
    config_id = str(row.get("config_id")) if hasattr(row, "get") else str(row)
    previous = seen.setdefault(key, config_id)
    if previous != config_id:
        raise ValueError("two configurations render into the same table cell {!r}: {} and {}"
                         .format(key, previous, config_id))


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
    text = re.sub(r"\s*\[pc:[^\]]+\]", "", text)      # one power configuration per export
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
def load_scarce_minority(analysis_dir: str) -> set:
    """{(distribution, kind)} whose held-out sets rest on <= 2 minority-class sequences.

    Written by ``scripts/heldout_dominance.py``. Cells of those configurations get a
    dagger: the sequence-level bootstrap has no between-sequence variance to estimate
    for that class, so the interval is conditional on one or two particular videos.
    """
    path = os.path.join(analysis_dir, "leakage", "scarce_minority.csv")
    if not os.path.isfile(path):
        return set()
    flags = pd.read_csv(path)
    return {(str(r.partition), str(r.kind))
            for _, r in flags[flags["flagged"] == 1].iterrows()}


def summary_tex(df: pd.DataFrame, metric: str, digits: int = 4, seconds_digits: int = 1,
                siunitx: bool = False, scarce: Optional[set] = None) -> Optional[str]:
    """Rows = configuration, column pair = one data distribution."""
    sub = df[df["metric"] == metric]
    if sub.empty:
        return None
    ndigits = digits_for(metric, digits, seconds_digits)
    dists = sorted(sub["distribution"].astype(str).unique())

    cells: Dict[Tuple[str, str], Dict[str, Any]] = {}
    order: Dict[str, Tuple[int, str]] = {}
    seen: Dict[Any, str] = {}
    for _, row in sub.iterrows():
        name = config_name(row.get("label"), row.get("distribution"))
        claim_cell(seen, (name, str(row["distribution"])), row)
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
            if scarce and ci != MISSING:
                pooled_flag, mean_flag = scarce_cells(scarce, str(row.get("distribution")))
                per_client = ("clientmean_" in metric) or not metric.startswith("selected_test_")
                # pooled selected-test metrics rest on every held-out sequence of the
                # partition; client means and the older final-epoch/v1 metrics are per client
                flag = (mean_flag if per_client else pooled_flag) if metric.startswith(
                    "selected_test_") else (str(row.get("distribution")), str(row.get("kind"))) in scarce
                if flag:
                    ci += "$^{\\dagger}$"
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
             "sequences, stratified by class composition. $^{\\dagger}$ marks an interval "
             "that rests on a minority class carried by at most two held-out sequences -- for a "
             "pooled metric, of the pooled held-out set; for a client mean or a per-client "
             "metric, of some client's own split (Table~\\ref{tab:heldout_sequences}): the "
             "bootstrap cannot "
             "estimate between-sequence variance for that class, so the interval is "
             "conditional on those particular videos and understates uncertainty about new "
             "footage.",
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
        seen: Dict[Any, str] = {}
        for _, row in block.iterrows():
            name = config_name(row.get("label"), dist)
            claim_cell(seen, (name, int(row["round"])), row)
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
    metrics = df["metric"].astype(str) if "metric" in df.columns else pd.Series(
        ["accuracy"] * len(df), index=df.index)
    for metric, protocol, dist in sorted(set(zip(metrics, protocols,
                                                 df["distribution"].astype(str)))):
        block = df[(metrics == metric) & (protocols == protocol)
                   & (df["distribution"].astype(str) == dist)]
        if body:
            body.append("\\midrule")
        body.append(_group_row(n_columns, dist_label(dist) + _protocol_suffix(protocol)
                               + " -- " + escape_latex(metric.replace("_", " "))))
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

    return latex_table(
        column_spec="l" + "c" * 7,
        header_lines=header,
        body_lines=body,
        caption="Pairwise strategy comparisons, paired by seed within each metric, "
                "partitioning protocol and data distribution (balanced accuracy pooled and "
                "as the client mean for the revision runs; accuracy for the v1 runs).",
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
            dist_label(str(row.get("distribution"))) + _protocol_suffix(protocols[idx])
            + " -- " + escape_latex(str(row.get("metric", "")).replace("_", " ")),
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
    seen: Dict[Any, str] = {}
    for _, row in sub.iterrows():
        name = config_name(row.get("label"), distribution)
        claim_cell(seen, (name, str(row["metric"])), row)
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
#: rounds of the time table per power configuration: the main matrix at its default
#: point (3), the MAXN_SUPER block (10).  One table never holds two configurations.
TIME_TABLE_ROUNDS = {"heterogeneous": 3, "maxn": 10}
#: the timing reporting rule (CLAUDE.md rule 12): round 1 = cold start, reported
#: separately; steady state = median test-free round time over rounds 2..R
ROUND1_TIME = "round1_time_s"
STEADY_ROUND_TIME = "steady_round_time_s"


def _seconds_cell(row: Any, siunitx: bool = False) -> str:
    """``mean $\\pm$ std [95% CI] (n)`` in seconds, one decimal, ``--`` without the metric."""
    if row is None:
        return MISSING
    n_seeds = _as_float(row.get("n_seeds"))
    return "{} {} ({})".format(
        fmt_mean_std(row.get("mean"), row.get("std"), 1, siunitx),
        fmt_ci(row.get("ci_low"), row.get("ci_high"), 1, siunitx),
        int(n_seeds) if n_seeds else 0).replace(" -- (", " (")


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


def time_tex(df: pd.DataFrame, siunitx: bool = False,
             power_config: str = DEFAULT_POWER_CONFIG) -> Optional[str]:
    """``tab:time``: wall-clock time and communication of the revision FL runs.

    Only group-level-split (revision) federated runs at the default operating
    point of ``power_config`` (``TIME_TABLE_ROUNDS``) are included; the caller passes
    one configuration's rows only.  Per-round time follows the timing reporting rule:
    round 1 (cold start) and the median over rounds 2..R, never total / R.  v1 runs are
    excluded on purpose: they were timed over WiFi and under the image-level split, so
    their wall-clock times are not comparable with the revision runs (wired Gigabit
    Ethernet).
    """
    if df.empty or "metric" not in df.columns:
        return None
    n_rounds = TIME_TABLE_ROUNDS.get(power_config, TIME_TABLE_POINT["num_rounds"])
    point = dict(TIME_TABLE_POINT, num_rounds=n_rounds)
    rows: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    seen: Dict[Any, str] = {}
    for _, row in df.iterrows():
        if str(row.get("kind")) != "fl" or _protocol_of(row) != "group":
            continue
        dist = str(row.get("distribution"))
        if dist not in TIME_TABLE_DISTS:
            continue
        if not all(_at_point(row, f, v) for f, v in point.items()):
            continue
        n_nodes = _as_float(row.get("n_nodes"))
        name = config_name(row.get("label"), dist).replace(" (group-level)", "")
        name = re.sub(r"\s*\[\d+N\]", "", name).strip()
        key = (int(n_nodes) if n_nodes else 0, dist, name)
        claim_cell(seen, key + (str(row["metric"]),), row)
        rows.setdefault(key, {})[str(row["metric"])] = row
    rows = {k: v for k, v in rows.items() if "total_time_s" in v}
    if not rows:
        return None

    header = [_row(["Configuration", "Time (min)", "Round 1 (s)",
                    "Rounds 2--{}, median (s)".format(n_rounds), "Comm.\\ (MB)",
                    "Bal.\\ acc.\\ (\\%)", "Bal.\\ acc., client mean (\\%)"])]
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
        # the declared primary metric under both aggregations, with the cluster-bootstrap
        # interval (CLAUDE.md rule 8)
        ba_cells = []
        for metric in (PRIMARY_BA, SECONDARY_BA):
            row = metrics.get(metric)
            # a configuration without prediction files has no figure under the declared
            # aggregation: print "--" rather than its own logged (weighted) number
            if row is not None and str(row.get("aggregation", "pooled")) != "pooled":
                row = None
            ba_cells.append(_pct_ci(row))
        short = {"iid": "IID", "non_iid_label": "Label skew"}.get(dist, dist_label(dist))
        rule_cells = [_seconds_cell(metrics.get(m), siunitx)
                      for m in (ROUND1_TIME, STEADY_ROUND_TIME)]
        body.append(_row(["{}N {} {}".format(n_nodes, short, name),
                          time_cell] + rule_cells + [comm_cell] + ba_cells))

    text = latex_table(
        column_spec="lcccccc",
        header_lines=header,
        body_lines=body,
        caption="Measured Training Time, Group-Level Split, Wired Gigabit Ethernet "
                "({} Rounds{})".format(n_rounds, ", All Nodes at MAXN\\_SUPER"
                                       if power_config == "maxn" else ""),
        label="tab:time",
        small=True,
        note="Time: mean $\\pm$ std [95\\% CI] ($n$ seeds) of the server wall-clock time "
             "excluding the report-only test evaluation. Round~1 is the cold start (process "
             "and CUDA initialisation, first read of the partition) and is reported "
             "separately; the steady-state per-round time is the median test-free round "
             "time over rounds 2 to {}, per run, summarised over seeds in the same way "
             "(timing reporting rule, Section~\\ref{{sec:timecomm}}). ".format(n_rounds) +
             "Communication: measured cumulative payload over the run. Balanced accuracy: "
             "test set at the round selected by the lowest weighted validation loss, pooled "
             "over all held-out images and as the unweighted mean over the clients, mean "
             "[95\\% cluster-bootstrap CI]. Only revision runs "
             "(group-level split, wired Gigabit Ethernet) are included; v1 timings (WiFi, "
             "image-level split) are not comparable and are omitted. Produced from "
             "\\texttt{summary\\_table.csv}.",
    )
    # five columns do not fit one column of the two-column layout
    return text.replace("\\begin{table}[t]", "\\begin{table*}[t]").replace(
        "\\end{table}", "\\end{table*}")


# --------------------------------------------------------------------------- #
# paper tabulars: tab:fullmetrics, tab:dirichlet, tab:lowdata
# --------------------------------------------------------------------------- #
# Each file is ONLY the tabular environment: main.tex keeps the float, caption,
# label and hand-written footnote (with its \todo markers) around an \input, so the
# reference cells come from analysis/ while the federated rows stay \PHs
# placeholders until the narrative is decided.  Balanced accuracy appears under
# both aggregations (CLAUDE.md hard rule 8): pooled over all held-out images of
# the three nodes (primary) and the unweighted mean over the clients (secondary).
PRIMARY_BA = "selected_test_balanced_accuracy"
SECONDARY_BA = "selected_test_clientmean_balanced_accuracy"
PAPER_PH = r"\PHs"
#: (summary kind, row label) of the reference rows, in table order
PAPER_REFERENCES = (("centralized", "Centralized"), ("local", "Local-only"))
#: federated rows kept as placeholders, in table order.  No FedBN row: these tables
#: report the three-round main matrix, and under the group-level split FedBN ran at
#: ten rounds only (decision 2026-10-07: FedBN is reported for ten rounds only)
PAPER_FL_ROWS = (r"\fedavg{}", r"\fedprox{} 0.01")
DAGGER = r"$^{\dagger}$"
_GENERATED = "% generated by scripts/export_latex_tables.py -- do not edit by hand"


def reference_row(summary: pd.DataFrame, distribution: str, kind: str, metric: str):
    """The one selection-rule summary row of a reference, or None."""
    if summary.empty:
        return None
    sel = summary[(_protocol_column(summary) == "group")
                  & (summary["distribution"].astype(str) == distribution)
                  & (summary["kind"].astype(str) == kind)
                  & (summary["metric"].astype(str) == metric)]
    if len(sel) > 1:
        raise ValueError("%d summary rows for %s/%s/%s" % (len(sel), distribution, kind, metric))
    if sel.empty:
        return None
    row = sel.iloc[0]
    check_aggregation(row)
    return row


def check_aggregation(row: Any) -> None:
    """A selected-test cell printed under the declared aggregation must have been computed
    under it: a configuration without prediction files carries its own logged figures
    (``aggregation`` = logged_native), which are neither pooled nor a client mean."""
    metric = str(row.get("metric", ""))
    if metric.startswith("selected_test_") and "aggregation" in row.index:
        agg = str(row.get("aggregation"))
        if agg != "pooled":
            raise ValueError("%s of %s is aggregated as %r, not pooled/client-mean from "
                             "prediction files" % (metric, row.get("config_id"), agg))


def _pct_ci(row: Any) -> str:
    if row is None or _as_float(row.get("mean")) is None:
        return MISSING
    lo, hi = _as_float(row.get("ci_low")), _as_float(row.get("ci_high"))
    text = "%.1f" % (100 * float(row["mean"]))
    if lo is not None and hi is not None:
        text += " [%.1f, %.1f]" % (100 * lo, 100 * hi)
    return text


def _pm(row: Any, scale: float, digits: int) -> str:
    if row is None or _as_float(row.get("mean")) is None:
        return MISSING
    std = _as_float(row.get("std"))
    fmt = "%%.%df" % digits
    if std is None or not math.isfinite(std):
        return "$" + fmt % (scale * float(row["mean"])) + "$"
    return "$" + (fmt + " \\pm " + fmt) % (scale * float(row["mean"]), scale * std) + "$"


def scarce_cells(scarce: set, distribution: str) -> Tuple[bool, bool]:
    """(pooled flagged, client-mean flagged) for one partition.

    The flags of scripts/heldout_dominance.py are per evaluation unit: the pooled
    held-out set is the centralized unit, the per-client units are those of the
    federated / local-only rows -- the same splits whichever model is evaluated.
    """
    return ((distribution, "centralized") in scarce,
            (distribution, "fl") in scarce or (distribution, "local") in scarce)


FULLMETRICS_PAPER_COLUMNS = (
    # (summary metric, header, formatter)
    (PRIMARY_BA, r"\textbf{Bal.\ acc.}", "ci"),
    (SECONDARY_BA, r"\textbf{Bal.\ acc.,} \textbf{client mean}", "ci"),
    ("selected_test_mcc", r"\textbf{MCC}", "mcc"),
    ("selected_test_accuracy", r"\textbf{Acc.}", "pct"),
    ("selected_test_recall", r"\textbf{Sens.}", "pct"),
    ("selected_test_specificity", r"\textbf{Spec.}", "pct"),
    ("selected_test_macro_f1", r"\textbf{Macro-}$F_1$", "pct"),
    ("selected_test_roc_auc", r"\textbf{ROC-AUC}", "pct"),
)


def fullmetrics_paper_tabular(summary: pd.DataFrame, scarce: set) -> Optional[str]:
    """tab:fullmetrics: IID and label skew, every headline metric, both BA aggregations."""
    dists = (("iid", "IID"), ("non_iid_label", "Non-IID"))
    if all(reference_row(summary, d, "centralized", PRIMARY_BA) is None for d, _ in dists):
        return None
    ncol = 2 + len(FULLMETRICS_PAPER_COLUMNS)
    out = [_GENERATED, r"\begin{tabular}{@{}ll" + "c" * len(FULLMETRICS_PAPER_COLUMNS) + "@{}}",
           r"\toprule",
           _row([r"\textbf{Dist.}", r"\textbf{Method}"] + [h for _, h, _ in FULLMETRICS_PAPER_COLUMNS]),
           r"\midrule"]
    for i, (dist, label) in enumerate(dists):
        if i:
            out.append(r"\midrule")
        pooled_flag, mean_flag = scarce_cells(scarce, dist)
        nrows = len(PAPER_REFERENCES) + len(PAPER_FL_ROWS)
        first = True
        for kind, name in PAPER_REFERENCES:
            cells = []
            for metric, _, fmt in FULLMETRICS_PAPER_COLUMNS:
                row = reference_row(summary, dist, kind, metric)
                if fmt == "ci":
                    text = _pct_ci(row)
                    flag = mean_flag if metric == SECONDARY_BA else pooled_flag
                    cells.append(text + (DAGGER if flag and text != MISSING else ""))
                elif fmt == "mcc":
                    cells.append(_pm(row, 1.0, 2))
                else:
                    cells.append(_pm(row, 100.0, 1))
            lead = (r"\multirow{%d}{*}{%s}" % (nrows, label)) if first else ""
            first = False
            out.append(_row([lead, name] + cells))
        for name in PAPER_FL_ROWS:
            out.append(_row(["", name] + [PAPER_PH] * len(FULLMETRICS_PAPER_COLUMNS)))
    out += [r"\bottomrule", r"\end{tabular}"]
    del ncol
    return "\n".join(out) + "\n"


#: tab:protocol_effect: the configurations run under BOTH protocols, (label, image-level
#: config_id + metric, group-level config_id); {d} is the distribution
PROTOCOL_EFFECT_ROWS = (
    ("Centralized", "image|centralized|centralized|{d}|3|15|0.001|3", "final_accuracy",
     "group|centralized|centralized|{d}|3|15|0.001|3"),
    ("Local-only", "image|local|local_only|{d}|3|15||3", "final_accuracy",
     "group|local|local_only|{d}|3|15||3"),
    (r"\fedavg{}", "image|fl|fedavg|{d}|3|||3", "v1_final_round_accuracy",
     "group|fl|fedavg|{d}|3|5|0.001|3"),
    (r"\fedprox{} 0.01", "image|fl|fedprox_0.01|{d}|3|||3", "v1_final_round_accuracy",
     "group|fl|fedprox_0.01|{d}|3|5|0.001|3"),
)


def _pm_n(summary: pd.DataFrame, config_id: str, metric: str) -> str:
    r"""``$mean \pm std$ ($n{=}k$)`` in %, from one summary row (exactly one required)."""
    sel = summary[(summary["config_id"].astype(str) == config_id)
                  & (summary["metric"].astype(str) == metric)]
    if sel.empty:
        return MISSING
    if len(sel) > 1:
        raise ValueError("%d summary rows for %s / %s" % (len(sel), config_id, metric))
    r = sel.iloc[0]
    return r"$%.1f \pm %.1f$ ($n{=}%d$)" % (100 * float(r["mean"]), 100 * float(r["std"]),
                                            int(r["n_seeds"]))


def protocol_effect_tabular(summary: pd.DataFrame) -> Optional[str]:
    """tab:protocol_effect: accuracy under the image-level (v1) and the group-level split,
    for the configurations that exist under both.  Image-level: v1 final-round validation
    accuracy (federated) / final-epoch test accuracy (references); group-level: test
    accuracy at the selected round / epoch."""
    wanted = {cid.format(d=d) for _, img, _, grp in PROTOCOL_EFFECT_ROWS for cid in (img, grp)
              for d in ("iid", "non_iid_label")}
    if summary.empty or not set(summary["config_id"].astype(str)) & wanted:
        return None
    dists = (("iid", "IID"), ("non_iid_label", "Non-IID"))
    out = [_GENERATED, r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
           _row([r"\textbf{Dist.}", r"\textbf{Method}", r"\textbf{Image-level (\%)}",
                 r"\textbf{Group-level (\%)}"]), r"\midrule"]
    for i, (d, label) in enumerate(dists):
        if i:
            out.append(r"\midrule")
        for j, (name, image_id, image_metric, group_id) in enumerate(PROTOCOL_EFFECT_ROWS):
            lead = (r"\multirow{%d}{*}{%s}" % (len(PROTOCOL_EFFECT_ROWS), label)) if j == 0 else ""
            out.append(_row([lead, name, _pm_n(summary, image_id.format(d=d), image_metric),
                             _pm_n(summary, group_id.format(d=d), "selected_test_accuracy")]))
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


#: tab:fedbn10: the MAXN_SUPER ten-round block (5b, ``maxn_long_horizon``) only -- one
#: family, one power configuration, five seeds; {d} is the distribution
FEDBN10_ROWS = ((r"\fedavg{}", "group|fl|fedavg|{d}|10|5|0.001|3|pc=maxn"),
                (r"\fedbn{}", "group|fl|fedbn|{d}|10|5|0.001|3|pc=maxn"))


def _ci_cell(summary: pd.DataFrame, config_id: str, metric: str, flagged: bool) -> str:
    sel = summary[(summary["config_id"].astype(str) == config_id)
                  & (summary["metric"].astype(str) == metric)]
    if len(sel) > 1:
        raise ValueError("%d summary rows for %s / %s" % (len(sel), config_id, metric))
    if sel.empty:
        return MISSING
    row = sel.iloc[0]
    check_aggregation(row)
    text = _pct_ci(row)
    return text + (DAGGER if flagged and text != MISSING else "")


def fedbn10_tabular(summary: pd.DataFrame, scarce: set) -> Optional[str]:
    """tab:fedbn10: balanced accuracy pooled (primary) and client mean (secondary), mean
    [95% CI], of FedAvg and FedBN in the 5b family (MAXN_SUPER, 10 rounds, 5 seeds)."""
    wanted = {cid.format(d=d) for _, cid in FEDBN10_ROWS for d in ("iid", "non_iid_label")}
    if summary.empty or not set(summary["config_id"].astype(str)) & wanted:
        return None
    out = [_GENERATED, r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
           _row([r"\textbf{Dist.}", r"\textbf{Method}", r"\textbf{Bal.\ acc.}",
                 r"\textbf{Bal.\ acc., client mean}"]), r"\midrule"]
    for i, (d, label) in enumerate((("iid", "IID"), ("non_iid_label", "Non-IID"))):
        if i:
            out.append(r"\midrule")
        pooled_flag, mean_flag = scarce_cells(scarce, d)
        for j, (name, cid) in enumerate(FEDBN10_ROWS):
            lead = (r"\multirow{%d}{*}{%s}" % (len(FEDBN10_ROWS), label)) if j == 0 else ""
            out.append(_row([lead, name,
                             _ci_cell(summary, cid.format(d=d), PRIMARY_BA, pooled_flag),
                             _ci_cell(summary, cid.format(d=d), SECONDARY_BA, mean_flag)]))
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def load_partition_skew(analysis_dir: str) -> Dict[str, float]:
    """{partition: size-weighted mean JSD} from scripts/partition_skew.py."""
    path = os.path.join(analysis_dir, "partition_skew.csv")
    if not os.path.isfile(path):
        return {}
    df = pd.read_csv(path)
    df = df[df["node"] == "ALL"]
    return {str(r.partition): float(r.jsd) for r in df.itertuples()}


DIRICHLET_PARTITIONS = (("dirichlet_0.1", r"$\alpha{=}0.1$"), ("dirichlet_0.5", r"$\alpha{=}0.5$"),
                        ("dirichlet_1", r"$\alpha{=}1.0$"))


def _two_aggregation_blocks(summary, columns, scarce, fl_rows, lead_label=None):
    """Rows for the pooled block then the client-mean block of a partition-column table."""
    out = []
    ncol = len(columns) + 1 + (1 if lead_label else 0)
    for metric, title in ((PRIMARY_BA, "Balanced accuracy pooled over all held-out images (primary)"),
                          (SECONDARY_BA, "Balanced accuracy, unweighted mean over the three clients (secondary)")):
        out.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (ncol, title))
        for kind, name in PAPER_REFERENCES:
            cells = []
            for dist in columns:
                pooled_flag, mean_flag = scarce_cells(scarce, dist)
                flag = mean_flag if metric == SECONDARY_BA else pooled_flag
                text = _pct_ci(reference_row(summary, dist, kind, metric))
                cells.append(text + (DAGGER if flag and text != MISSING else ""))
            out.append(_row(([lead_label] if lead_label else []) + [name] + cells))
        for name in fl_rows:
            out.append(_row(([""] if lead_label else []) + [name] + [PAPER_PH] * len(columns)))
    return out


def dirichlet_paper_tabular(summary: pd.DataFrame, scarce: set, skew: Dict[str, float]) -> Optional[str]:
    """tab:dirichlet: the three Dirichlet draws as distinct partitions, ordered by the
    measured skew of scripts/partition_skew.py (never by alpha)."""
    parts = [p for p in DIRICHLET_PARTITIONS if p[0] in skew]
    if len(parts) != len(DIRICHLET_PARTITIONS):
        return None
    if all(reference_row(summary, p, "centralized", PRIMARY_BA) is None for p, _ in parts):
        return None
    parts.sort(key=lambda p: skew[p[0]])
    cols = [p for p, _ in parts]
    out = [_GENERATED, r"\begin{tabular}{@{}l" + "c" * len(parts) + "@{}}", r"\toprule",
           r" & \multicolumn{%d}{c}{\textbf{Partition, in order of measured skew}} \\" % len(parts),
           r"\cmidrule(lr){2-%d}" % (len(parts) + 1),
           _row([r"\textbf{Method}"] + [label for _, label in parts]),
           _row([r"{\footnotesize measured skew (JSD)}"]
                + [r"{\footnotesize %.3f}" % skew[p] for p in cols]),
           r"\midrule"]
    # the dirichlet_skew block runs FedAvg and FedProx only
    fl_rows = PAPER_FL_ROWS
    body = _two_aggregation_blocks(summary, cols, scarce, fl_rows)
    # a \midrule between the two aggregation blocks
    split = 1 + len(PAPER_REFERENCES) + len(fl_rows)
    out += body[:split] + [r"\midrule"] + body[split:]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


LOWDATA_COLUMNS = ((None, r"$\rho{=}1.00$"), ("0.05", r"$\rho{=}0.05$"), ("0.01", r"$\rho{=}0.01$"))


def lowdata_paper_tabular(summary: pd.DataFrame, scarce: set) -> Optional[str]:
    """tab:lowdata: IID and label skew at rho = 1, 0.05, 0.01, both aggregations."""
    dists = (("iid", "IID"), ("non_iid_label", "Non-IID"))
    fl_rows = PAPER_FL_ROWS                      # the low-data block runs FedAvg and FedProx only
    if reference_row(summary, "iid_sub0.01", "centralized", PRIMARY_BA) is None:
        return None
    out = [_GENERATED, r"\begin{tabular}{@{}ll" + "c" * len(LOWDATA_COLUMNS) + "@{}}", r"\toprule",
           _row([r"\textbf{Dist.}", r"\textbf{Method}"] + [label for _, label in LOWDATA_COLUMNS]),
           r"\midrule"]
    for metric, title in ((PRIMARY_BA, "Balanced accuracy pooled over all held-out images (primary)"),
                          (SECONDARY_BA, "Balanced accuracy, unweighted mean over the three clients (secondary)")):
        if metric == SECONDARY_BA:
            out.append(r"\midrule")
        out.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (2 + len(LOWDATA_COLUMNS), title))
        for i, (dist, label) in enumerate(dists):
            if i:
                out.append(r"\cmidrule(lr){1-%d}" % (2 + len(LOWDATA_COLUMNS)))
            cols = [dist if f is None else "%s_sub%s" % (dist, f) for f, _ in LOWDATA_COLUMNS]
            nrows = len(PAPER_REFERENCES) + len(fl_rows)
            first = True
            for kind, name in PAPER_REFERENCES:
                cells = []
                for col in cols:
                    pooled_flag, mean_flag = scarce_cells(scarce, col)
                    flag = mean_flag if metric == SECONDARY_BA else pooled_flag
                    text = _pct_ci(reference_row(summary, col, kind, metric))
                    cells.append(text + (DAGGER if flag and text != MISSING else ""))
                lead = (r"\multirow{%d}{*}{%s}" % (nrows, label)) if first else ""
                first = False
                out.append(_row([lead, name] + cells))
            for name in fl_rows:
                out.append(_row(["", name] + [PAPER_PH] * len(cols)))
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


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
           warn: bool = True, power_config: str = DEFAULT_POWER_CONFIG) -> List[str]:
    """Write every table; returns the paths that were written.

    Only runs of ``power_config`` (plus the desktop baselines and v1 runs, which have
    none) enter the tables; see :func:`select_power_config`.
    """
    tables = {key: select_power_config(df, power_config)
              for key, df in load_tables(analysis_dir, warn=warn).items()}
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
    scarce = load_scarce_minority(analysis_dir)
    for metric in select_metrics(summary, metrics):
        _write(summary_tex(summary, metric, digits, seconds_digits, siunitx, scarce),
               "summary_{}.tex".format(_safe_label(metric)))

    _write(per_round_tex(tables["per_round"], per_round_metric, digits, max_rounds, siunitx),
           "per_round_{}.tex".format(_safe_label(per_round_metric)))
    _write(pairwise_tex(tables["pairwise"], digits, siunitx), "pairwise_tests.tex")
    _write(friedman_tex(tables["friedman"], digits, siunitx), "friedman.tex")
    _write(time_tex(summary, siunitx, power_config), "time.tex")

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

    # measured partition skew, generated by scripts/partition_skew.py
    skew_table = os.path.join(analysis_dir, "partition_skew.tex")
    if os.path.isfile(skew_table):
        with open(skew_table, encoding="utf-8") as fh:
            _write(fh.read(), "partition_skew.tex")

    # the paper's reference tabulars (tab:fullmetrics, tab:dirichlet, tab:lowdata)
    _write(fullmetrics_paper_tabular(summary, scarce), "fullmetrics_tabular.tex")
    _write(dirichlet_paper_tabular(summary, scarce, load_partition_skew(analysis_dir)),
           "dirichlet_tabular.tex")
    _write(lowdata_paper_tabular(summary, scarce), "lowdata_tabular.tex")
    _write(protocol_effect_tabular(summary), "protocol_effect_tabular.tex")
    _write(fedbn10_tabular(summary, scarce), "fedbn10_tabular.tex")

    # held-out sequences per class, generated by scripts/heldout_dominance.py
    hs_table = os.path.join(analysis_dir, "leakage", "heldout_sequences.tex")
    if os.path.isfile(hs_table):
        with open(hs_table, encoding="utf-8") as fh:
            _write(fh.read(), "heldout_sequences.tex")

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
    parser.add_argument("--power_config", default=DEFAULT_POWER_CONFIG,
                        help="Jetson power configuration whose federated runs are exported "
                             "(default: %(default)s, the main matrix); never mixed with another. "
                             "Write other configurations to another --output_dir")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    written = export(
        args.analysis_dir, args.output_dir, metrics=args.metrics, digits=args.digits,
        seconds_digits=args.seconds_digits, max_rounds=args.max_rounds,
        siunitx=args.siunitx, per_round_metric=args.per_round_metric,
        power_config=args.power_config)
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
