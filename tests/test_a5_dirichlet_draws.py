"""A5 -- the additional Dirichlet draws (docs/A5_DIRICHLET_DRAWS_PREREG.md), descriptive only.

The analysis reads exactly the declared runs, refuses a run that is not the declared one,
summarises the draws by range and SD only, and stays out of the main FLAME analysis; the
paper tables print its numbers verbatim.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import a5_dirichlet_draws as a5  # noqa: E402
import analyze_results as ar  # noqa: E402
from scripts.export_latex_tables import (  # noqa: E402
    a5_desktop_sim_tabular, a5_skew_tabular, a5_testbed_tabular,
)
from tests.test_export_latex_tables import assert_valid_latex  # noqa: E402


def test_declared_runs_are_exactly_those_of_the_prereg():
    specs = a5.declared_runs("results")
    assert len(specs) == 22
    testbed = [s for s in specs if s["setting"] == a5.TESTBED]
    desktop = [s for s in specs if s["setting"] == a5.DESKTOP]
    # sec. 3a / A5-1 (c): alpha 0.1 only, the two new draws; ps42 is not re-run on the testbed
    assert len(testbed) == 4 and {s["alpha"] for s in testbed} == {"0.1"}
    assert {s["draw"] for s in testbed} == {"ps123", "ps456"}
    assert {os.path.basename(s["run_dir"]) for s in testbed} == {
        "rev_dirichlet0.1_ps%s_%s_r10_seed42" % (d, st)
        for d in ("123", "456") for st in ("fedavg", "fedprox_0.01")}
    assert all(os.path.basename(os.path.dirname(s["run_dir"])) == "pc_maxn" for s in testbed)
    # sec. 3b / A5-2: 3 alpha x 3 draws x 2 strategies
    assert len(desktop) == 18
    assert {(s["alpha"], s["draw"]) for s in desktop} == {
        (a, d) for a in ("0.1", "0.5", "1") for d in ("ps42", "ps123", "ps456")}
    # ps42 is the main study's partition, the others carry their suffix
    assert a5.partition_of("0.5", "ps42") == "dirichlet_0.5"
    assert a5.partition_of("1", "ps456") == "dirichlet_1_ps456"


def test_every_declared_partition_has_a_manifest_and_split_statistics():
    stats = json.load(open(os.path.join(ROOT, "data", "splits", "split_stats.json"),
                           encoding="utf-8"))
    for part in {s["partition"] for s in a5.declared_runs("results")}:
        assert os.path.isfile(os.path.join(ROOT, "data", "splits", part + ".csv.gz")), part
        assert part in stats, part


def _testbed_spec(tmp_path):
    return {"setting": a5.TESTBED, "alpha": "0.1", "draw": "ps123", "strategy": "fedavg",
            "partition": "dirichlet_0.1_ps123",
            "run_dir": str(tmp_path / "rev_dirichlet0.1_ps123_fedavg_r10_seed42")}


def _maxn(before="MAXN_SUPER", after="MAXN_SUPER"):
    nodes = ("node_a", "node_b", "node_c")
    return {"power_config": "maxn", "measured_before": {n: before for n in nodes},
            "measured_after": {n: after for n in nodes}}


def test_check_run_refuses_runs_that_are_not_the_declared_ones(tmp_path):
    spec = _testbed_spec(tmp_path)
    a5.check_run(spec, {"num_rounds": 10, "seed": 42, "power": _maxn()})
    with pytest.raises(ValueError, match="rounds"):
        a5.check_run(spec, {"num_rounds": 3, "seed": 42, "power": _maxn()})
    with pytest.raises(ValueError, match="MAXN_SUPER"):
        a5.check_run(spec, {"num_rounds": 10, "seed": 42, "power": _maxn(after="7W")})
    with pytest.raises(ValueError, match="MAXN_SUPER"):
        a5.check_run(spec, {"num_rounds": 10, "seed": 42,
                            "power": dict(_maxn(), power_config="heterogeneous")})

    run_dir = tmp_path / "sim_dirichlet0.5_ps42_fedavg_r10_seed42"
    run_dir.mkdir()
    desk = {"setting": a5.DESKTOP, "alpha": "0.5", "draw": "ps42", "strategy": "fedavg",
            "partition": "dirichlet_0.5", "run_dir": str(run_dir)}
    info = {"ok": True, "server_rc": 0, "client_rc": {"node_a": 0, "node_b": 0, "node_c": 0},
            "partition": "dirichlet_0.5", "draw": "ps42"}
    (run_dir / "sim_info.json").write_text(json.dumps(info))
    a5.check_run(desk, {"num_rounds": 10, "seed": 42})
    for bad in ({"ok": False}, {"client_rc": {"node_a": 0, "node_b": 1, "node_c": 0}},
                {"partition": "dirichlet_0.5_ps123"}):
        (run_dir / "sim_info.json").write_text(json.dumps(dict(info, **bad)))
        with pytest.raises(ValueError):
            a5.check_run(desk, {"num_rounds": 10, "seed": 42})


def test_spread_is_range_and_sd_over_the_draws_only():
    runs = pd.DataFrame([
        {"setting": a5.TESTBED, "alpha": "0.1", "strategy": "fedavg", "draw": d,
         "ba_pooled": v, "ba_clientmean": v - 0.1} for d, v in (("ps123", 0.90), ("ps456", 0.94))]
        + [{"setting": a5.DESKTOP, "alpha": "1", "strategy": "fedavg", "draw": d,
            "ba_pooled": v, "ba_clientmean": v} for d, v in (("ps42", 0.8), ("ps123", 0.9),
                                                             ("ps456", 1.0))])
    s = a5.spread(runs).set_index("setting")
    t = s.loc[a5.TESTBED]
    assert t["n_draws"] == 2 and t["draws"] == "ps123 ps456"
    assert t["ba_pooled_range"] == pytest.approx(0.04)
    assert t["ba_pooled_sd"] == pytest.approx(np.std([0.90, 0.94], ddof=1))
    d = s.loc[a5.DESKTOP]
    assert d["n_draws"] == 3 and d["ba_clientmean_range"] == pytest.approx(0.2)
    assert d["ba_clientmean_sd"] == pytest.approx(0.1)
    # nothing but per-group range and SD: no test, no pooled-over-settings figure
    assert not [c for c in s.columns if "p_" in c or "verdict" in c or "pooled_over" in c]


def test_the_main_analysis_never_reads_the_a5_draws(tmp_path):
    for name in ("rev_dirichlet0.1_ps123_fedavg_r10_seed42", "rev_iid_fedavg_r10_seed42"):
        d = tmp_path / "pc_maxn" / name
        d.mkdir(parents=True)
        (d / "results.json").write_text("{}")
    sim = tmp_path / "desktop_sim" / "sim_dirichlet0.1_ps42_fedavg_r10_seed42"
    sim.mkdir(parents=True)
    (sim / "results.json").write_text("{}")
    found = [os.path.basename(p) for p in ar.iter_run_dirs(str(tmp_path))]
    assert found == ["rev_iid_fedavg_r10_seed42"]
    assert ar.is_a5_draw_entry("rev_dirichlet1_ps456_fedprox_0.01_r10_seed42")
    assert not ar.is_a5_draw_entry("rev_dirichlet1_fedprox_0.01_r10_seed42")


def _write_a5(tmp_path):
    d = tmp_path / "a5"
    d.mkdir()
    rows = []
    for setting, alphas, draws in a5.DECLARED:
        for alpha in alphas:
            for strategy in a5.STRATEGIES:
                for k, draw in enumerate(draws):
                    v = 0.80 + 0.05 * k
                    rows.append({"setting": setting, "alpha": alpha, "draw": draw,
                                 "strategy": strategy, "partition": a5.partition_of(alpha, draw),
                                 "selected_round": 3 + k, "ba_pooled": v,
                                 "ba_pooled_ci_low": v - 0.1, "ba_pooled_ci_high": v + 0.05,
                                 "ba_clientmean": v - 0.02, "ba_clientmean_ci_low": v - 0.12,
                                 "ba_clientmean_ci_high": v + 0.03,
                                 "round1_time_s": 1500.0 + k if setting == a5.TESTBED else None,
                                 "steady_round_time_s": 1400.0 + k if setting == a5.TESTBED else None})
    runs = pd.DataFrame(rows)
    runs.to_csv(d / "a5_runs.csv", index=False)
    a5.spread(runs).to_csv(d / "a5_spread.csv", index=False)
    stats = json.load(open(os.path.join(ROOT, "data", "splits", "split_stats.json"),
                           encoding="utf-8"))
    import partition_skew
    pd.DataFrame(partition_skew.describe({p: stats[p] for p in runs.partition.unique()})).to_csv(
        d / "a5_skew.csv", index=False)
    return runs


def test_a5_tables_print_the_analysis(tmp_path):
    _write_a5(tmp_path)
    tb = a5_testbed_tabular(str(tmp_path))
    assert_valid_latex(tb, tabular_only=True)
    assert "80.0 [70.0, 85.0]" in tb and "85.0 [75.0, 90.0]" in tb      # ps123, ps456
    assert "range / SD, $n{=}2$" in tb and "5.0 / 3.5" in tb
    assert "1500.0" in tb and "1401.0" in tb and " 42 " not in tb       # no seed-42 row
    ds = a5_desktop_sim_tabular(str(tmp_path))
    assert_valid_latex(ds, tabular_only=True)
    assert ds.count("range / SD, $n{=}3$") == 6 and "Dirichlet $\\alpha{=}1.0$" in ds
    assert "1500.0" not in ds                                            # no desktop times
    sk = a5_skew_tabular(str(tmp_path))
    assert_valid_latex(sk, tabular_only=True)
    rows = [l for l in sk.splitlines() if l.endswith(" \\\\")]
    assert len(rows) == 10                                               # header + 9 draws


def test_no_a5_table_without_its_analysis(tmp_path):
    assert a5_testbed_tabular(str(tmp_path)) is None
    assert a5_desktop_sim_tabular(str(tmp_path)) is None
    assert a5_skew_tabular(str(tmp_path)) is None


def test_paper_a5_tables_are_the_committed_analysis_verbatim():
    """paper/tables/a5_*_tabular.tex are analysis/a5/ rendered, number for number."""
    analysis = os.path.join(ROOT, "analysis")
    if not os.path.isfile(os.path.join(analysis, "a5", "a5_runs.csv")):
        pytest.skip("analysis/a5 not generated")
    for name, fn in (("a5_testbed_tabular.tex", a5_testbed_tabular),
                     ("a5_desktop_sim_tabular.tex", a5_desktop_sim_tabular),
                     ("a5_skew_tabular.tex", a5_skew_tabular)):
        committed = open(os.path.join(ROOT, "paper", "tables", name), encoding="utf-8").read()
        assert committed.replace("\r\n", "\n") == fn(analysis), name
    runs = pd.read_csv(os.path.join(analysis, "a5", "a5_runs.csv"))
    assert len(runs) == 22 and (runs.max_diff_vs_logged <= a5.LOGGED_TOLERANCE).all()
    assert runs[runs.setting == a5.DESKTOP][["round1_time_s", "steady_round_time_s"]].isna().all().all()
