"""Power configurations: namespace, pre-flight lock, recording, and no pooling.

The main matrix (98 runs, results/rev_*) ran with Jetson power modes nobody had
harmonised (node_a 15W, node_b MAXN_SUPER, node_c 7W).  Reruns under another
configuration (all MAXN) must

* live in their own namespace, results/pc_<name>/rev_*, so the generator and the
  runner never skip them because the heterogeneous run of the same cell exists;
* be refused by run_matrix.py unless every node reports the declared mode;
* carry the measured modes in results.json;
* never be pooled, paired or tabulated with the main matrix by analyze_results.py
  and export_latex_tables.py.
"""

import copy
import json
import os
import sys

import pandas as pd
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from scripts import print_revision_commands as prc  # noqa: E402
from scripts import run_matrix  # noqa: E402
from scripts.analyze_results import (  # noqa: E402
    POWER_HETEROGENEOUS,
    POWER_NOT_APPLICABLE,
    PowerConfigError,
    collect_runs,
    pairwise_tests,
    summarize,
)
from scripts.export_latex_tables import claim_cell, select_power_config  # noqa: E402
from tests.test_analyze_selection import schema3_payload, write_run  # noqa: E402

EXAMPLE = os.path.join(_REPO, "configs", "testbed.example.yaml")
MAXN = {"node_a": "MAXN_SUPER", "node_b": "MAXN_SUPER", "node_c": "MAXN_SUPER"}


# --------------------------------------------------------------------------- #
# namespace: generator and runner agree
# --------------------------------------------------------------------------- #
def test_runner_and_generator_share_the_power_configurations():
    assert run_matrix.POWER_CONFIGS == prc.POWER_CONFIGS
    assert run_matrix.DEFAULT_POWER_CONFIG == prc.DEFAULT_POWER_CONFIG == "heterogeneous"
    for pc in prc.POWER_CONFIGS:
        assert run_matrix.power_root(pc) == prc.power_config_root(pc)
    assert prc.power_config_root("heterogeneous") == "results"
    assert prc.power_config_root("maxn") == "results/pc_maxn"


def _generated(block, power_config, tmp_path, capsys):
    assert prc.main(["--all_seeds", "--block", block, "--format", "bash",
                     "--no_skip_existing", "--power_config", power_config]) == 0
    path = tmp_path / "block_{}_{}.sh".format(block, power_config)
    path.write_text(capsys.readouterr().out, encoding="utf-8")
    return str(path)


def test_maxn_seed_extension_is_the_same_20_cells_in_its_own_namespace(tmp_path, capsys):
    main = run_matrix.parse_block(_generated("seed_extension", "heterogeneous", tmp_path, capsys))
    maxn = run_matrix.parse_block(_generated("seed_extension", "maxn", tmp_path, capsys))
    assert len(main) == len(maxn) == 20
    for a, b in zip(main, maxn):
        assert a["out_dir"].startswith("results/rev_")
        assert b["out_dir"] == "results/pc_maxn/" + a["out_dir"][len("results/"):]
        # identical experiment: only the output directory differs
        assert b["server"] == a["server"].replace(a["out_dir"], b["out_dir"])
        assert b["clients"] == a["clients"]
    run_matrix.check_namespace(maxn, "maxn")
    run_matrix.check_namespace(main, "heterogeneous")
    with pytest.raises(SystemExit, match="outside results/pc_maxn/"):
        run_matrix.check_namespace(main, "maxn")
    with pytest.raises(SystemExit, match="namespace of power_config heterogeneous"):
        run_matrix.check_namespace(maxn, "heterogeneous")


def test_an_existing_heterogeneous_run_does_not_skip_the_maxn_run(tmp_path, monkeypatch):
    monkeypatch.setattr(prc, "REPO_ROOT", str(tmp_path))
    cfg = prc.apply_all_seeds(prc.load_config(prc.DEFAULT_CONFIG)["revision"])
    runs = prc.expand_all(cfg, "seed_extension")
    done = runs["seed_extension"][0]
    os.makedirs(tmp_path / done.output_dir)
    (tmp_path / done.output_dir / "results.json").write_text("{}")
    assert done.exists()

    maxn = prc.expand_all(cfg, "seed_extension")
    prc.rebase_power_config(maxn, "maxn")
    twin = maxn["seed_extension"][0]
    assert twin.output_dir == "results/pc_maxn/" + done.output_dir[len("results/"):]
    assert not twin.exists()
    assert prc.emit_status(twin, True, set()) == "run"


def test_baselines_are_not_moved_into_a_power_namespace():
    cfg = prc.apply_all_seeds(prc.load_config(prc.DEFAULT_CONFIG)["revision"])
    runs = prc.expand_all(cfg, "baselines_extension")
    before = [r.output_dir for r in runs["baselines_extension"]]
    prc.rebase_power_config(runs, "maxn")
    assert [r.output_dir for r in runs["baselines_extension"]] == before


# --------------------------------------------------------------------------- #
# testbed.local.yaml: expected modes
# --------------------------------------------------------------------------- #
def _testbed(tmp_path, power_modes):
    text = open(EXAMPLE, encoding="utf-8").read()
    for node in ("a", "b", "c"):
        text = text.replace("<user-on-node-%s>" % node, "op%s" % node)
    text = text[:text.index("power_modes:")] if "power_modes:" in text else text
    if power_modes is not None:
        text += "power_modes:\n"
        for pc, modes in power_modes.items():
            text += "  %s:\n" % pc
            for node, mode in modes.items():
                text += "    %s: %s\n" % (node, mode)
    path = tmp_path / "testbed.local.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_example_declares_every_configuration_with_placeholders_only():
    import yaml
    cfg = yaml.safe_load(open(EXAMPLE, encoding="utf-8"))
    assert set(cfg["power_modes"]) == set(prc.POWER_CONFIGS)
    for pc, modes in cfg["power_modes"].items():
        assert set(modes) == {"node_a", "node_b", "node_c"}
        assert all(str(v).startswith("<") and str(v).endswith(">") for v in modes.values())
        with pytest.raises(SystemExit, match="placeholder"):
            run_matrix.load_power_modes(EXAMPLE, pc)


def test_filled_modes_load(tmp_path):
    path = _testbed(tmp_path, {"maxn": MAXN,
                               "heterogeneous": {"node_a": "15W", "node_b": "MAXN_SUPER",
                                                 "node_c": "7W"}})
    assert run_matrix.load_power_modes(path, "maxn") == MAXN
    assert run_matrix.load_power_modes(path, "heterogeneous")["node_c"] == "7W"
    # the host/user part of the same file still loads
    nodes, _ = run_matrix.load_testbed(path)
    assert nodes["node_b"]["user"] == "opb"


def test_undeclared_configuration_is_refused(tmp_path):
    path = _testbed(tmp_path, {"maxn": MAXN})
    with pytest.raises(SystemExit, match="power_modes.heterogeneous"):
        run_matrix.load_power_modes(path, "heterogeneous")
    with pytest.raises(SystemExit, match="power_modes.maxn"):
        run_matrix.load_power_modes(_testbed(tmp_path, None), "maxn")


def test_maxn_must_declare_maxn_modes_on_all_three_nodes(tmp_path):
    path = _testbed(tmp_path, {"maxn": dict(MAXN, node_c="7W")})
    with pytest.raises(SystemExit, match="node_c is '7W', which is not a MAXN mode"):
        run_matrix.load_power_modes(path, "maxn")
    path = _testbed(tmp_path, {"maxn": {"node_a": "MAXN_SUPER", "node_b": "MAXN_SUPER"}})
    with pytest.raises(SystemExit, match="node_c: missing"):
        run_matrix.load_power_modes(path, "maxn")


# --------------------------------------------------------------------------- #
# pre-flight lock
# --------------------------------------------------------------------------- #
def test_nvpmodel_output_is_parsed():
    out = "NV Power Mode: MAXN_SUPER\n2\n"
    assert run_matrix.RE_POWER_MODE.search(out).group(1) == "MAXN_SUPER"
    assert run_matrix.RE_POWER_MODE.search("NV Power Mode: 15W\n0\n").group(1) == "15W"


def test_check_power_flags_every_mismatch_and_unreadable_node(monkeypatch):
    measured = {"node_a": "15W", "node_b": "MAXN_SUPER", "node_c": None}
    monkeypatch.setattr(run_matrix, "NODES", {n: {} for n in measured})
    monkeypatch.setattr(run_matrix, "node_power_mode", lambda node: measured[node])
    problems, got = run_matrix.check_power(MAXN)
    assert got == measured
    assert len(problems) == 2
    assert any("node_a: power mode 15W, expected MAXN_SUPER" in p for p in problems)
    assert any("node_c: could not read the power mode" in p for p in problems)

    measured.update(node_a="MAXN_SUPER", node_c="MAXN_SUPER")
    problems, _ = run_matrix.check_power(MAXN)
    assert problems == []


def test_preflight_reports_power_problems_and_the_measurement(monkeypatch):
    monkeypatch.setattr(run_matrix, "NODES", {})          # no node: skip RAM/disk/commit/ssh
    monkeypatch.setattr(run_matrix, "check_power",
                        lambda expected: (["node_c: power mode 7W, expected MAXN_SUPER"],
                                          {"node_c": "7W"}))
    monkeypatch.setattr(run_matrix, "SERVER_PORT", 0)      # any free port binds
    measured = {"stale": "x"}
    problems = run_matrix.preflight(strict_commit=False, power=MAXN, measured=measured)
    assert problems == ["node_c: power mode 7W, expected MAXN_SUPER"]
    assert measured == {"node_c": "7W"}


# --------------------------------------------------------------------------- #
# recording after the run
# --------------------------------------------------------------------------- #
def _finished_run(tmp_path):
    out_dir = tmp_path / "results" / "pc_maxn" / "rev_iid_fedavg_seed42"
    out_dir.mkdir(parents=True)
    (out_dir / "results.json").write_text(json.dumps({"strategy": "fedavg", "seed": 42,
                                                      "model_selection": {"selected_round": 2}}))
    return str(out_dir)


def test_power_modes_are_recorded_in_results_json(tmp_path):
    out_dir = _finished_run(tmp_path)
    ok, why = run_matrix.record_power(out_dir, "maxn", MAXN, dict(MAXN), dict(MAXN))
    assert ok, why
    data = json.load(open(os.path.join(out_dir, "results.json")))
    assert data["strategy"] == "fedavg" and data["model_selection"]["selected_round"] == 2
    assert data["power"]["power_config"] == "maxn"
    assert data["power"]["expected"] == MAXN
    assert data["power"]["measured_before"] == MAXN == data["power"]["measured_after"]
    assert not os.path.exists(os.path.join(out_dir, "results.json.tmp"))


def test_a_mode_change_during_the_run_moves_the_run_out_of_results(tmp_path, monkeypatch):
    monkeypatch.setattr(run_matrix, "INVALID_DIR", str(tmp_path / "logs" / "invalid_runs"))
    out_dir = _finished_run(tmp_path)
    after = dict(MAXN, node_c="7W")
    ok, why = run_matrix.record_power(out_dir, "maxn", MAXN, dict(MAXN), after)
    assert not ok
    assert "node_c MAXN_SUPER -> 7W" in why and "moved to" in why
    assert not os.path.exists(out_dir)              # not skipped as done, not analysed
    moved = os.listdir(tmp_path / "logs" / "invalid_runs")
    assert len(moved) == 1 and "rev_iid_fedavg_seed42" in moved[0]


# --------------------------------------------------------------------------- #
# analysis: never pooled, never paired across configurations
# --------------------------------------------------------------------------- #
def _power_block(pc, modes, before=None, after=None):
    return {"power_config": pc, "expected": dict(modes),
            "measured_before": dict(before or modes), "measured_after": dict(after or modes)}


@pytest.fixture
def two_configs(tmp_path):
    root = str(tmp_path / "results")
    for seed, shift in ((42, 0.0), (123, 0.01), (456, 0.02)):
        for strategy in ("fedavg", "fedprox_0.01"):
            name = "rev_iid_{}_seed{}".format(strategy, seed)
            mu = 0.01 if strategy.startswith("fedprox") else 0.0
            main = schema3_payload(strategy, "iid", seed, [0.6, 0.2, 0.3],
                                   [0.80 + shift, 0.85 + shift, 0.9])
            main["proximal_mu"] = mu
            write_run(root, name, main)
            maxn = schema3_payload(strategy, "iid", seed, [0.6, 0.2, 0.3],
                                   [0.60 + shift, 0.65 + shift, 0.7], round_wall=600.0)
            maxn["proximal_mu"] = mu
            maxn["power"] = _power_block("maxn", MAXN)
            write_run(os.path.join(root, "pc_maxn"), name, maxn)
    return root


def test_configurations_are_separate_rows_and_never_averaged(two_configs):
    runs = collect_runs(two_configs, warn=False)
    assert len(runs) == 12
    assert {r["power_config"] for r in runs} == {POWER_HETEROGENEOUS, "maxn"}
    summary = summarize(runs)["summary"]
    acc = summary[summary["metric"] == "selected_test_accuracy"]
    assert len(acc) == 4                                   # 2 strategies x 2 configurations
    by = {(r.power_config, r.strategy): r["mean"] for _, r in acc.iterrows()}
    assert by[("heterogeneous", "fedavg")] == pytest.approx(0.86)
    assert by[("maxn", "fedavg")] == pytest.approx(0.66)
    assert acc[acc.power_config == "maxn"]["label"].str.contains(r"\[pc:maxn\]").all()
    assert not acc[acc.power_config == "heterogeneous"]["label"].str.contains("pc:").any()
    # the main matrix keeps its config ids (they seed the cluster bootstrap)
    assert not acc[acc.power_config == "heterogeneous"]["config_id"].str.contains("pc=").any()
    assert acc[acc.power_config == "maxn"]["config_id"].str.endswith("|pc=maxn").all()


def test_pairwise_tests_pair_within_a_configuration_only(two_configs):
    runs = collect_runs(two_configs, warn=False)
    pw = pairwise_tests(runs, metric="accuracy")
    assert set(pw["power_config"]) == {"heterogeneous", "maxn"}
    for _, row in pw.iterrows():
        assert {row.strategy_a, row.strategy_b} == {"FedAvg", "FedProx(mu=0.01)"}
        assert row.n_seeds == 3
    main = pw[pw.power_config == "heterogeneous"].iloc[0]
    assert main.mean_a == pytest.approx(0.86) and main.mean_b == pytest.approx(0.86)
    maxn = pw[pw.power_config == "maxn"].iloc[0]
    assert maxn.mean_a == pytest.approx(0.66) and maxn.mean_b == pytest.approx(0.66)


def test_baselines_join_every_configuration_as_references(two_configs):
    import shutil
    root = two_configs
    for seed in (42, 123, 456):                       # the committed desktop baselines
        name = "rev_iid_centralized_seed{}".format(seed)
        dest = os.path.join(root, "rev_baselines_sel", name)
        os.makedirs(dest)
        shutil.copy(os.path.join(_REPO, "results", "rev_baselines_sel", name, "results.json"),
                    dest)
    runs = collect_runs(root, warn=False)
    central = [r for r in runs if r["kind"] == "centralized"]
    assert central and all(r["power_config"] == POWER_NOT_APPLICABLE for r in central)
    pw = pairwise_tests(runs, metric="accuracy")
    with_central = pw[(pw.strategy_a == "Centralized") | (pw.strategy_b == "Centralized")]
    assert set(with_central["power_config"]) == {"heterogeneous", "maxn"}


def test_recorded_configuration_must_match_the_directory(tmp_path):
    root = str(tmp_path / "results")
    payload = schema3_payload("fedavg", "iid", 42, [0.6, 0.2, 0.3], [0.8, 0.85, 0.9])
    payload["power"] = _power_block("maxn", MAXN)
    write_run(root, "rev_iid_fedavg_seed42", payload)          # a maxn run in the main matrix
    with pytest.raises(PowerConfigError, match="namespace of 'heterogeneous'"):
        collect_runs(root, warn=False)


def test_a_run_whose_modes_drifted_is_refused(tmp_path):
    root = str(tmp_path / "results")
    payload = schema3_payload("fedavg", "iid", 42, [0.6, 0.2, 0.3], [0.8, 0.85, 0.9])
    payload["power"] = _power_block("maxn", MAXN, after=dict(MAXN, node_c="7W"))
    write_run(os.path.join(root, "pc_maxn"), "rev_iid_fedavg_seed42", payload)
    with pytest.raises(PowerConfigError, match="measured_after"):
        collect_runs(root, warn=False)


def test_sweep_variants_are_never_averaged_into_the_default_point(tmp_path):
    """Before the fix, FedAvg E=1 and E=5 of the same seed were averaged into one
    value per seed in pairwise_tests/friedman."""
    root = str(tmp_path / "results")
    for seed, shift in ((42, 0.0), (123, 0.01), (456, 0.02)):
        write_run(root, "rev_iid_fedavg_seed{}".format(seed), schema3_payload(
            "fedavg", "iid", seed, [0.6, 0.2, 0.3], [0.8 + shift, 0.85 + shift, 0.9]))
        e1 = schema3_payload("fedavg", "iid", seed, [0.6, 0.2, 0.3], [0.5, 0.55 + shift, 0.6])
        for cfg in e1["client_config"].values():
            cfg["local_epochs"] = 1
        write_run(root, "rev_iid_fedavg_ep1_seed{}".format(seed), e1)
    runs = collect_runs(root, warn=False)
    assert sorted(r["variant"] for r in runs) == ["FedAvg"] * 3 + ["FedAvg (E=1)"] * 3
    pw = pairwise_tests(runs, metric="accuracy")
    assert len(pw) == 1
    row = pw.iloc[0]
    means = {row.strategy_a: row.mean_a, row.strategy_b: row.mean_b}
    assert means["FedAvg"] == pytest.approx(0.86)
    assert means["FedAvg (E=1)"] == pytest.approx(0.56)


# --------------------------------------------------------------------------- #
# LaTeX export: one configuration per export, no silent overwrite
# --------------------------------------------------------------------------- #
def test_export_selects_one_configuration_plus_the_shared_rows():
    df = pd.DataFrame({"power_config": ["heterogeneous", "maxn", "desktop_gpu", "unrecorded"],
                       "x": [1, 2, 3, 4]})
    assert list(select_power_config(df, "heterogeneous")["x"]) == [1, 3, 4]
    assert list(select_power_config(df, "maxn")["x"]) == [2, 3, 4]
    old = pd.DataFrame({"x": [1, 2]})                    # CSV written before the column existed
    assert list(select_power_config(old, "maxn")["x"]) == [1, 2]


def test_two_configurations_in_one_cell_raise():
    seen = {}
    claim_cell(seen, ("FedAvg", "iid"), {"config_id": "a"})
    claim_cell(seen, ("FedAvg", "iid"), {"config_id": "a"})      # same config: fine
    with pytest.raises(ValueError, match="same table cell"):
        claim_cell(seen, ("FedAvg", "iid"), {"config_id": "b"})
