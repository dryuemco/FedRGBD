"""Tests for ``scripts/export_latex_tables.py``.

Most tests build the ``analyze_results.py`` CSVs by hand (with the exact
columns that script writes) so the expected LaTeX is known cell by cell; the
last test runs the real pipeline over the repository's ``results/`` directory
read-only, writing only into ``tmp_path``.
"""

from __future__ import annotations

import os
import re
import sys

import pandas as pd
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.export_latex_tables import (  # noqa: E402
    escape_latex,
    export,
    fmt_mean_std,
    fmt_number,
    fmt_p,
    main,
    select_metrics,
    stars,
    summary_tex,
)

SUMMARY_COLUMNS = [
    "config_id", "label", "kind", "strategy", "mu", "distribution", "n_nodes",
    "num_rounds", "metric", "n_seeds", "seeds", "mean", "std", "ci95",
    "ci_low", "ci_high", "min", "max",
]
PER_ROUND_COLUMNS = [
    "config_id", "label", "kind", "strategy", "distribution", "round", "n_seeds",
    "accuracy_mean", "accuracy_std", "accuracy_ci95", "accuracy_ci_low", "accuracy_ci_high",
    "loss_mean", "loss_std", "loss_ci95", "loss_ci_low", "loss_ci_high",
    "elapsed_s_mean", "cumulative_mb_mean",
]
PAIRWISE_COLUMNS = [
    "distribution", "metric", "strategy_a", "strategy_b", "n_seeds", "seeds",
    "mean_a", "mean_b", "mean_diff", "cohen_d_paired", "cohen_d_unpaired",
    "wilcoxon_stat", "wilcoxon_p", "ttest_t", "ttest_p", "pg_cohen_d_av",
    "pg_wilcoxon_p", "note",
]
FRIEDMAN_COLUMNS = [
    "distribution", "metric", "n_strategies", "strategies", "n_seeds", "seeds",
    "chi_square", "p_value", "pg_chi_square", "pg_p_value", "note",
]

#: a label with every LaTeX special the CSVs can realistically contain
NASTY_LABEL = "Weird_name 100% & co non_iid_label"


# --------------------------------------------------------------------------- #
# synthetic analysis directory
# --------------------------------------------------------------------------- #
def _summary_row(label, kind, strategy, dist, metric, mean, std, n_seeds=3):
    half = 0.5 * std
    return {
        "config_id": "{}|{}".format(strategy, dist), "label": label, "kind": kind,
        "strategy": strategy, "mu": None, "distribution": dist, "n_nodes": 3,
        "num_rounds": 3, "metric": metric, "n_seeds": n_seeds, "seeds": "42,123,456",
        "mean": mean, "std": std, "ci95": half, "ci_low": mean - half,
        "ci_high": mean + half, "min": mean - std, "max": mean + std,
    }


@pytest.fixture
def analysis_dir(tmp_path):
    """A minimal but column-exact copy of what analyze_results.py writes."""
    directory = tmp_path / "analysis"
    directory.mkdir()

    summary = []
    for dist in ("iid", "non_iid_label"):
        summary.append(_summary_row("FedAvg {} [3N]".format(dist), "fl", "fedavg", dist,
                                    "final_accuracy", 0.99, 0.01))
        summary.append(_summary_row("FedAvg {} [3N]".format(dist), "fl", "fedavg", dist,
                                    "final_loss", 0.0512, 0.004))
        summary.append(_summary_row("FedAvg {} [3N]".format(dist), "fl", "fedavg", dist,
                                    "total_time_s", 6107.25, 203.6))
        summary.append(_summary_row("FedProx(mu=0.01) {} [3N]".format(dist), "fl",
                                    "fedprox", dist, "final_accuracy", 0.9812, 0.002))
        summary.append(_summary_row("Centralized {}".format(dist), "centralized",
                                    "centralized", dist, "final_accuracy", 0.9977, 0.0005))
    # a single-seed row (no std / CI) and a label full of LaTeX specials
    summary.append(_summary_row(NASTY_LABEL, "fl", "fedbn", "non_iid_label",
                                "final_accuracy", 0.75, float("nan"), n_seeds=1))
    pd.DataFrame(summary, columns=SUMMARY_COLUMNS).to_csv(
        directory / "summary_table.csv", index=False)

    per_round = []
    for dist in ("iid", "non_iid_label"):
        for rnd, acc in ((1, 0.71), (2, 0.95), (3, 0.99)):
            per_round.append({
                "config_id": "fedavg|" + dist, "label": "FedAvg {} [3N]".format(dist),
                "kind": "fl", "strategy": "fedavg", "distribution": dist, "round": rnd,
                "n_seeds": 3, "accuracy_mean": acc, "accuracy_std": 0.01,
                "accuracy_ci95": 0.005, "accuracy_ci_low": acc - 0.005,
                "accuracy_ci_high": acc + 0.005, "loss_mean": 1 - acc, "loss_std": 0.01,
                "loss_ci95": 0.005, "loss_ci_low": 0.0, "loss_ci_high": 0.1,
                "elapsed_s_mean": 100.0 * rnd, "cumulative_mb_mean": 36.6 * rnd,
            })
    pd.DataFrame(per_round, columns=PER_ROUND_COLUMNS).to_csv(
        directory / "per_round_table.csv", index=False)

    pairwise = [{
        "distribution": "iid", "metric": "accuracy", "strategy_a": "FedAvg",
        "strategy_b": "FedProx(mu=0.01)", "n_seeds": 3, "seeds": "42,123,456",
        "mean_a": 0.99, "mean_b": 0.9812, "mean_diff": 0.0088,
        "cohen_d_paired": 2.345, "cohen_d_unpaired": 1.2, "wilcoxon_stat": 0.0,
        "wilcoxon_p": 0.25, "ttest_t": 4.2, "ttest_p": 0.0062,
        "pg_cohen_d_av": 1.1, "pg_wilcoxon_p": float("nan"), "note": "",
    }, {
        "distribution": "non_iid_label", "metric": "accuracy", "strategy_a": "FedAvg",
        "strategy_b": "FedBN", "n_seeds": 3, "seeds": "42,123,456",
        "mean_a": 0.99, "mean_b": 0.75, "mean_diff": 0.24,
        "cohen_d_paired": 3.0, "cohen_d_unpaired": 2.0, "wilcoxon_stat": 0.0,
        "wilcoxon_p": 0.5, "ttest_t": 2.9, "ttest_p": 0.0000004,
        "pg_cohen_d_av": 2.1, "pg_wilcoxon_p": float("nan"), "note": "",
    }]
    pd.DataFrame(pairwise, columns=PAIRWISE_COLUMNS).to_csv(
        directory / "pairwise_tests.csv", index=False)

    friedman = [{
        "distribution": dist, "metric": "accuracy", "n_strategies": 3,
        "strategies": "Centralized,FedAvg,FedProx(mu=0.01)", "n_seeds": 3,
        "seeds": "42,123,456", "chi_square": 8.2857, "p_value": p,
        "pg_chi_square": float("nan"), "pg_p_value": float("nan"), "note": "",
    } for dist, p in (("iid", 0.0752), ("non_iid_label", 0.012))]
    pd.DataFrame(friedman, columns=FRIEDMAN_COLUMNS).to_csv(
        directory / "friedman.csv", index=False)

    pd.DataFrame([{"run_name": "r1", "kind": "fl"}]).to_csv(
        directory / "runs.csv", index=False)
    return str(directory)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def assert_valid_latex(text):
    """Balanced braces and matching begin/end for the environments we emit."""
    assert text.count("{") == text.count("}"), "unbalanced braces"
    for env in ("table", "tabular"):
        assert text.count("\\begin{" + env + "}") == text.count("\\end{" + env + "}")
    assert text.count("\\toprule") == 1
    assert text.count("\\bottomrule") == 1
    assert "\\caption{" in text and "\\label{tab:" in text
    # no stray unescaped special outside comments, cross-reference keys and math mode.
    # \label{} and \ref{} arguments are keys, not typeset text, so underscores in them
    # are legitimate and must not be escaped.
    typeset = "\n".join(l for l in text.splitlines() if not l.startswith("%"))
    typeset = re.sub(r"\\(label|ref|eqref|autoref)\{[^}]*\}", "", typeset)
    typeset = re.sub(r"\$[^$]*\$", "", typeset)
    for escaped in ("\\_", "\\%", "\\&", "\\#"):
        typeset = typeset.replace(escaped, "")
    for special in ("_", "%", "#"):  # '&' is the legitimate column separator
        assert special not in typeset, "unescaped {!r}".format(special)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# 1. formatting primitives
# --------------------------------------------------------------------------- #
def test_escape_latex_handles_every_special():
    assert escape_latex("non_iid_label") == "non\\_iid\\_label"
    assert escape_latex("95% & more") == "95\\% \\& more"
    assert escape_latex("a#b$c") == "a\\#b\\$c"
    assert escape_latex(None) == ""


def test_fmt_number_and_mean_std():
    assert fmt_number(0.123456, 4) == "0.1235"
    assert fmt_number(0.123456, 2) == "0.12"
    assert fmt_number(6107.25, 1) == "6107.2"       # >= 1000 -> one decimal
    assert fmt_number(None) == "--"
    assert fmt_number(float("nan")) == "--"
    assert fmt_number(0.5, 4, siunitx=True) == "\\num{0.5000}"
    assert fmt_mean_std(0.99, 0.01, 4) == "0.9900 $\\pm$ 0.0100"
    assert fmt_mean_std(0.99, float("nan"), 4) == "0.9900"   # single seed
    assert fmt_mean_std(None, 0.01) == "--"


def test_stars_and_p_values():
    assert stars(0.0004) == "***"
    assert stars(0.004) == "**"
    assert stars(0.04) == "*"
    assert stars(0.4) == ""
    assert stars(None) == ""
    assert fmt_p(0.0062, 4) == "0.0062$^{**}$"
    assert fmt_p(0.0000001, 4).startswith("$<$0.0001")
    assert fmt_p(float("nan")) == "--"


def test_select_metrics(analysis_dir):
    summary = pd.read_csv(os.path.join(analysis_dir, "summary_table.csv"))
    assert select_metrics(summary, None) == ["final_accuracy", "final_loss", "total_time_s"]
    assert select_metrics(summary, ["final_accuracy"]) == ["final_accuracy"]
    assert select_metrics(summary, ["nope"]) == []
    assert set(select_metrics(summary, ["all"])) == set(summary["metric"].unique())


# --------------------------------------------------------------------------- #
# 2. the generated tables
# --------------------------------------------------------------------------- #
def test_export_writes_every_table(analysis_dir, tmp_path):
    out = str(tmp_path / "tables")
    written = export(analysis_dir, out, warn=False)
    names = sorted(os.path.basename(p) for p in written)
    assert names == sorted([
        "summary_final_accuracy.tex", "summary_final_loss.tex", "summary_total_time_s.tex",
        "per_round_accuracy.tex", "pairwise_tests.tex", "friedman.tex",
    ])   # no full_metrics_*: the fixture has no rule-following (selected_test_*) rows
    for path in written:
        assert_valid_latex(read(path))


def test_summary_table_content(analysis_dir, tmp_path):
    out = str(tmp_path / "tables")
    export(analysis_dir, out, metrics=["final_accuracy"], warn=False)
    text = read(os.path.join(out, "summary_final_accuracy.tex"))

    assert "\\label{tab:summary_final_accuracy}" in text
    assert "\\multicolumn{2}{c}{IID}" in text
    assert "\\multicolumn{2}{c}{Non-IID (label skew)}" in text
    assert "\\cmidrule(lr){2-3}" in text and "\\cmidrule(lr){4-5}" in text
    assert "\\begin{tabular}{l cc cc}" in text
    # the distribution is stripped from the row name, mu is typeset
    assert "FedAvg [3N]" in text
    assert "FedProx($\\mu$=0.01) [3N]" in text
    assert "0.9900 $\\pm$ 0.0100 & [0.9850, 0.9950] (3)" in text
    # baselines sort after the federated strategies
    assert text.index("FedAvg [3N]") < text.index("Centralized")
    # a config missing from one distribution gets the placeholder
    assert " & -- & --" in text


def test_latex_specials_are_escaped(analysis_dir, tmp_path):
    out = str(tmp_path / "tables")
    export(analysis_dir, out, metrics=["final_accuracy"], warn=False)
    text = read(os.path.join(out, "summary_final_accuracy.tex"))
    assert "Weird\\_name 100\\% \\& co" in text
    assert "Weird_name" not in text
    assert "100% " not in text


def test_digits_are_respected(analysis_dir, tmp_path):
    out = str(tmp_path / "tables")
    export(analysis_dir, out, metrics=["final_accuracy", "total_time_s"], digits=2,
           seconds_digits=1, warn=False)
    accuracy = read(os.path.join(out, "summary_final_accuracy.tex"))
    seconds = read(os.path.join(out, "summary_total_time_s.tex"))
    assert "0.99 $\\pm$ 0.01" in accuracy
    assert "0.9900" not in accuracy
    assert "6107.2 $\\pm$ 203.6" in seconds  # seconds keep one decimal


def test_siunitx_wraps_numbers(analysis_dir, tmp_path):
    out = str(tmp_path / "tables")
    export(analysis_dir, out, metrics=["final_accuracy"], siunitx=True, warn=False)
    text = read(os.path.join(out, "summary_final_accuracy.tex"))
    assert "\\num{0.9900} $\\pm$ \\num{0.0100}" in text
    assert_valid_latex(text)


def test_per_round_table_content(analysis_dir, tmp_path):
    out = str(tmp_path / "tables")
    export(analysis_dir, out, warn=False)
    text = read(os.path.join(out, "per_round_accuracy.tex"))
    assert "\\multicolumn{3}{c}{Communication round}" in text
    assert "Configuration & 1 & 2 & 3 \\\\" in text
    assert "\\textit{IID}" in text and "\\textit{Non-IID (label skew)}" in text
    assert "0.7100 $\\pm$ 0.0100" in text
    assert_valid_latex(text)


def test_per_round_thins_many_rounds(analysis_dir, tmp_path):
    """More rounds than --max_rounds are sub-sampled but the last one is kept."""
    df = pd.read_csv(os.path.join(analysis_dir, "per_round_table.csv"))
    rows = []
    for rnd in range(1, 21):
        row = df.iloc[0].to_dict()
        row["round"] = rnd
        row["accuracy_mean"] = 0.5 + rnd / 100.0
        rows.append(row)
    pd.DataFrame(rows, columns=PER_ROUND_COLUMNS).to_csv(
        os.path.join(analysis_dir, "per_round_table.csv"), index=False)

    out = str(tmp_path / "tables")
    export(analysis_dir, out, max_rounds=5, warn=False)
    text = read(os.path.join(out, "per_round_accuracy.tex"))
    header = [line for line in text.splitlines() if line.startswith("Configuration &")][0]
    assert header.endswith("20 \\\\")
    assert len(header.split("&")) - 1 <= 6
    assert_valid_latex(text)


def test_pairwise_table_content(analysis_dir, tmp_path):
    out = str(tmp_path / "tables")
    export(analysis_dir, out, warn=False)
    text = read(os.path.join(out, "pairwise_tests.tex"))
    assert "FedAvg vs.\\ FedProx($\\mu$=0.01)" in text
    assert "0.0062$^{**}$" in text          # t-test p with two stars
    assert "$<$0.0001$^{***}$" in text      # tiny p-value
    assert "$d_z$" in text
    assert "\\label{tab:pairwise_tests}" in text
    assert_valid_latex(text)


def test_friedman_table_content(analysis_dir, tmp_path):
    out = str(tmp_path / "tables")
    export(analysis_dir, out, warn=False)
    text = read(os.path.join(out, "friedman.tex"))
    assert "$\\chi^2$" in text
    assert "8.286" in text
    assert "0.0752" in text
    assert "0.0120$^{*}$" in text
    assert_valid_latex(text)


def test_full_metrics_table_content(analysis_dir, tmp_path):
    """Only rule-following rows (selected_test_*) appear; final-epoch rows never do."""
    path = os.path.join(analysis_dir, "summary_table.csv")
    df = pd.read_csv(path)
    extra = [_summary_row("FedAvg iid [3N] {group}", "fl", "fedavg", "iid", m, v, 0.01, 5)
             for m, v in (("selected_test_accuracy", 0.912), ("selected_test_loss", 0.25))]
    extra += [_summary_row("Centralized iid {group}", "centralized", "centralized", "iid", m, v,
                           0.01, 5)
              for m, v in (("selected_test_accuracy", 0.934), ("selected_test_loss", 0.2))]
    pd.concat([df, pd.DataFrame(extra)]).to_csv(path, index=False)

    out = str(tmp_path / "tables")
    written = [os.path.basename(p) for p in export(analysis_dir, out, warn=False)]
    assert "full_metrics_iid.tex" in written
    assert "full_metrics_non_iid_label.tex" not in written   # no rule-following rows there
    text = read(os.path.join(out, "full_metrics_iid.tex"))
    assert "Configuration & Accuracy & Loss \\\\" in text
    assert "\\label{tab:full_metrics_iid}" in text
    assert "0.9120" in text and "0.9340" in text
    assert "0.9900" not in text          # the fixture's final_accuracy rows are left out
    assert_valid_latex(text)


def test_full_metrics_skipped_with_a_single_metric(analysis_dir, tmp_path):
    df = pd.read_csv(os.path.join(analysis_dir, "summary_table.csv"))
    df = df[df["metric"] == "final_accuracy"]
    df.to_csv(os.path.join(analysis_dir, "summary_table.csv"), index=False)
    out = str(tmp_path / "tables")
    written = [os.path.basename(p) for p in export(analysis_dir, out, warn=False)]
    assert not any(name.startswith("full_metrics") for name in written)


# --------------------------------------------------------------------------- #
# 3. CLI and edge cases
# --------------------------------------------------------------------------- #
def test_cli_writes_tables(analysis_dir, tmp_path, capsys):
    out = str(tmp_path / "cli_tables")
    assert main(["--analysis_dir", analysis_dir, "--output_dir", out]) == 0
    captured = capsys.readouterr().out
    assert "summary_final_accuracy.tex" in captured
    assert os.path.isfile(os.path.join(out, "summary_final_accuracy.tex"))


def test_cli_on_an_empty_analysis_dir(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--analysis_dir", str(empty), "--output_dir", str(tmp_path / "t")]) == 1
    assert "no table written" in capsys.readouterr().out


def test_empty_summary_returns_none():
    empty = pd.DataFrame(columns=SUMMARY_COLUMNS)
    assert summary_tex(empty, "final_accuracy") is None


# --------------------------------------------------------------------------- #
# 4. end to end on the real repository results (read-only)
# --------------------------------------------------------------------------- #
def test_end_to_end_on_the_real_results(tmp_path):
    results_dir = os.path.join(_REPO_ROOT, "results")
    if not os.path.isdir(results_dir):
        pytest.skip("no results/ directory in this checkout")

    from scripts.analyze_results import main as analyze_main

    analysis = tmp_path / "analysis"
    assert analyze_main(["--results_dir", results_dir, "--output_dir", str(analysis),
                         "--no_plots"]) == 0

    out = str(tmp_path / "tables")
    written = export(str(analysis), out, warn=False)
    assert written, "the real results produced no table"
    names = [os.path.basename(p) for p in written]
    assert "summary_final_accuracy.tex" in names
    assert "pairwise_tests.tex" in names

    for path in written:
        text = read(path)
        assert_valid_latex(text)
        # every body row must have the same number of cells as the column spec
        spec = re.search(r"\\begin\{tabular\}\{([^}]*)\}", text).group(1)
        n_columns = sum(1 for c in spec if c in "lcrS")
        for line in text.splitlines():
            if not line.endswith("\\\\") or "\\multicolumn" in line:
                continue
            assert line.count("&") + 1 == n_columns, "{}: {}".format(path, line)


# --------------------------------------------------------------------------- #
# 5. tab:time -- revision runs only
# --------------------------------------------------------------------------- #
def _time_rows(label, protocol, dist, time_s, mb, n_nodes=3, **point):
    rows = []
    for metric, mean, std in (("total_time_s", time_s, 60.0), ("final_cumulative_mb", mb, 0.0)):
        row = _summary_row(label, "fl", "fedavg", dist, metric, mean, std, n_seeds=5)
        row.update({"protocol": protocol, "n_nodes": n_nodes, "local_epochs": 5, "lr": 0.001})
        row.update(point)
        rows.append(row)
    return rows


def test_time_table_uses_revision_runs_only(tmp_path):
    from scripts.export_latex_tables import time_tex

    rows = []
    rows += _time_rows("FedAvg non_iid_label [3N] {group}", "group", "non_iid_label", 3000.0, 110.3)
    rows += _time_rows("FedAvg non_iid_label [2N] {group}", "group", "non_iid_label", 2400.0, 73.5,
                       n_nodes=2)
    # v1 (WiFi, image-level) run: must be excluded
    rows += _time_rows("FedAvg non_iid_label [3N] {image}", "image", "non_iid_label", 6146.9, 110.3)
    # revision run off the default operating point: excluded
    rows += _time_rows("FedAvg non_iid_label [3N] {group}", "group", "non_iid_label", 9999.0, 367.7,
                       num_rounds=10)
    # revision run on a distribution outside the table: excluded
    rows += _time_rows("FedAvg dirichlet_0.1 [3N] {group}", "group", "dirichlet_0.1", 8888.0, 110.3)
    df = pd.DataFrame(rows)

    text = time_tex(df)
    assert text is not None
    assert "\label{tab:time}" in text
    assert "3N Non-IID (label skew) FedAvg" in text and "2N Non-IID (label skew) FedAvg" in text
    assert "50.0 $\pm$ 1.0" in text          # 3000 s -> 50.0 min, std 60 s -> 1.0 min
    assert "110.3" in text and "73.5" in text
    assert "102.4" not in text                # 6146.9 s (v1) -> 102.4 min must not appear
    assert "166.6" not in text and "148.1" not in text
    assert text.index("2N") < text.index("3N")
    assert_valid_latex(text)


def test_time_table_absent_without_revision_runs(analysis_dir, tmp_path):
    """The fixture holds only unlabelled/v1-style rows: no tab:time is written."""
    out = str(tmp_path / "tables")
    written = [os.path.basename(p) for p in export(analysis_dir, out, warn=False)]
    assert "time.tex" not in written


def test_scarce_minority_configurations_are_daggered(tmp_path):
    """Intervals resting on <= 2 minority-class sequences must be marked.

    A reader cannot otherwise tell that a per-client interval is conditional on one
    or two videos rather than on the dataset.
    """
    leakage = tmp_path / "leakage"
    leakage.mkdir()
    (leakage / "scarce_minority.csv").write_text(
        "partition,kind,min_sequences,flagged,detail\n"
        "iid,local,2,1,node_a No_Fire (1011 images)\n"
        "iid,centralized,21,0,pooled No_Fire\n", encoding="utf-8")

    from scripts.export_latex_tables import load_scarce_minority

    scarce = load_scarce_minority(str(tmp_path))
    assert scarce == {("iid", "local")}

    rows = []
    for kind, label in (("local", "Local-only (group-level)"),
                        ("centralized", "Centralized (group-level)")):
        rows.append({"metric": "selected_test_balanced_accuracy", "distribution": "iid",
                     "kind": kind, "label": label, "mean": 0.84, "std": 0.01,
                     "ci_low": 0.78, "ci_high": 0.88, "n_seeds": 5,
                     "ci_method": "cluster_bootstrap_B1000"})
    tex = summary_tex(pd.DataFrame(rows), "selected_test_balanced_accuracy",
                      scarce=scarce)

    local_line = next(l for l in tex.splitlines() if "Local-only" in l)
    central_line = next(l for l in tex.splitlines() if "Centralized" in l)
    assert r"$^{\dagger}$" in local_line, "the scarce-minority row is not marked"
    assert r"$^{\dagger}$" not in central_line, "the pooled row must not be marked"
    assert r"$^{\dagger}$" in tex.split(r"\bottomrule")[-1], "the footnote must explain it"


def test_no_dagger_when_no_configuration_is_scarce():
    rows = [{"metric": "selected_test_balanced_accuracy", "distribution": "iid",
             "kind": "local", "label": "Local-only (group-level)", "mean": 0.84, "std": 0.01,
             "ci_low": 0.78, "ci_high": 0.88, "n_seeds": 5,
             "ci_method": "cluster_bootstrap_B1000"}]
    tex = summary_tex(pd.DataFrame(rows), "selected_test_balanced_accuracy", scarce=set())
    body = tex.split(r"\midrule")[-1].split(r"\bottomrule")[0]
    assert r"\dagger" not in body
