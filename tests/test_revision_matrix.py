"""Tests for the `revision` block of configs/experiment_matrix.yaml and for
scripts/print_revision_commands.py, which expands it into concrete commands.
"""

import re
import subprocess
import sys

import pytest
import yaml

from scripts import print_revision_commands as prc

CONFIG_PATH = prc.DEFAULT_CONFIG
PYTHON = sys.executable


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def revision(cfg):
    assert "revision" in cfg
    return cfg["revision"]


# --------------------------------------------------------------------------- #
# YAML structure / spec checks
# --------------------------------------------------------------------------- #
def test_block_names_present(revision):
    expected_blocks = {
        "seed_extension", "dirichlet_skew", "low_data", "long_horizon_fedbn",
        "mu_grid", "local_epochs", "learning_rate", "baselines_extension",
    }
    assert expected_blocks <= set(revision)
    assert "analysis" in revision
    assert "data_preparation" in revision


def test_seed_extension_spec(revision):
    b = revision["seed_extension"]
    assert set(b["strategies"]) == {"fedavg", "fedprox_0.01"}
    assert set(b["data_distributions"]) == {"iid", "non_iid_label"}
    assert b["seeds"] == [42, 123, 456, 789, 1011]
    assert set(b["new_seeds"]) == {789, 1011}
    assert b["total_runs"] == 20
    assert b["new_runs"] == 8


def test_dirichlet_skew_spec(revision):
    b = revision["dirichlet_skew"]
    assert b["alphas"] == [0.1, 0.5, 1.0]
    assert set(b["strategies"]) == {"fedavg", "fedprox_0.01"}
    assert b["seeds"] == [42, 123, 456]
    assert b["total_runs"] == 18


def test_low_data_spec(revision):
    b = revision["low_data"]
    assert b["fractions"] == [0.05, 0.01]
    assert set(b["data_distributions"]) == {"iid", "non_iid_label"}
    assert set(b["strategies"]) == {"fedavg", "fedprox_0.01"}
    assert b["seeds"] == [42, 123, 456]
    assert b["total_runs"] == 24


def test_long_horizon_fedbn_spec(revision):
    b = revision["long_horizon_fedbn"]
    assert set(b["strategies"]) == {"fedbn", "fedavg"}
    assert b["data_distributions"] == ["non_iid_label"]
    assert b["rounds"] == 10
    assert b["seeds"] == [42, 123, 456]
    assert b["total_runs"] == 6


def test_mu_grid_spec(revision):
    b = revision["mu_grid"]
    assert b["mus"] == [0.001, 0.01, 0.05, 0.1, 0.5]
    assert b["data_distributions"] == ["non_iid_label"]
    assert b["seeds"] == [42, 123, 456]
    assert b["total_runs"] == 15


def test_local_epochs_spec(revision):
    b = revision["local_epochs"]
    assert b["local_epochs_values"] == [1, 2, 5]
    assert set(b["strategies"]) == {"fedavg", "fedprox_0.01"}
    assert b["data_distributions"] == ["non_iid_label"]
    assert b["seeds"] == [42, 123, 456]
    assert b["total_runs"] == 18


def test_learning_rate_spec(revision):
    b = revision["learning_rate"]
    assert b["lrs"] == [0.0001, 0.001]
    assert set(b["strategies"]) == {"fedavg", "fedprox_0.01"}
    assert b["data_distributions"] == ["non_iid_label"]
    assert b["seeds"] == [42, 123, 456]
    assert b["total_runs"] == 12


def test_baselines_extension_spec(revision):
    b = revision["baselines_extension"]
    assert set(b["baseline_types"]) == {"centralized", "local_only"}
    parts = b["parts"]
    assert set(parts["new_seeds"]["seeds"]) == {789, 1011}
    assert set(parts["new_seeds"]["data_distributions"]) == {"iid", "non_iid_label"}
    assert parts["new_seeds"]["total_runs"] == 8
    assert parts["dirichlet"]["alphas"] == [0.1, 0.5, 1.0]
    assert parts["dirichlet"]["seeds"] == [42, 123, 456]
    assert parts["dirichlet"]["total_runs"] == 18
    assert set(parts["subsample"]["data_distributions"]) == {"iid", "non_iid_label"}
    assert parts["subsample"]["fractions"] == [0.05, 0.01]
    assert parts["subsample"]["seeds"] == [42, 123, 456]
    assert parts["subsample"]["total_runs"] == 24
    assert b["total_runs"] == 50
    assert b["rounds"] * b["local_epochs"] == 15  # epochs = rounds * local_epochs


def test_grand_totals(revision):
    per_block_total = sum(
        revision[name]["total_runs"]
        for name in ("seed_extension", "dirichlet_skew", "low_data", "long_horizon_fedbn",
                     "mu_grid", "local_epochs", "learning_rate", "baselines_extension")
    )
    assert revision["grand_total_runs"] == per_block_total == 163


def test_data_preparation_command(revision):
    cmd = revision["data_preparation"]["commands"][0]
    assert "src/data/data_splitter.py" in cmd
    assert "--group_file analysis/leakage/groups.json" in cmd
    assert "--dirichlet_alpha 0.1 0.5 1.0" in cmd
    assert "--subsample_frac 0.05 0.01" in cmd
    # re-splitting an existing data/processed REQUIRES --clean, otherwise the
    # previous partition is left behind and leaks train images into val/test
    assert "--clean" in cmd
    assert "--verify" in cmd


def test_analysis_commands(revision):
    cmds = revision["analysis"]["commands"]
    assert any("analyze_results.py" in c for c in cmds)
    assert any("analyze_flame_leakage.py" in c for c in cmds)


# --------------------------------------------------------------------------- #
# print_revision_commands.py expansion checks (in-process, via main())
# --------------------------------------------------------------------------- #
EXPECTED_BLOCK_RUN_COUNTS = {
    "seed_extension": 8,       # new_runs, not the full 20-cell grid
    "dirichlet_skew": 18,
    "low_data": 24,
    "long_horizon_fedbn": 6,
    "mu_grid": 15,
    "local_epochs": 18,
    "learning_rate": 12,
    "baselines_extension": 50,
}


@pytest.mark.parametrize("block,expected_n", EXPECTED_BLOCK_RUN_COUNTS.items())
def test_expand_block_run_counts(revision, block, expected_n):
    runs = prc.BLOCK_EXPANDERS[block](revision[block])
    assert len(runs) == expected_n


def test_expand_all_matches_sum(revision):
    runs_by_block = prc.expand_all(revision)
    total = sum(len(v) for v in runs_by_block.values())
    assert total == sum(EXPECTED_BLOCK_RUN_COUNTS.values()) == 151


def test_output_dir_naming_convention(revision):
    pattern = re.compile(
        r"^results/rev_(iid|noniid|dirichlet0\.1|dirichlet0\.5|dirichlet1|"
        r"iid_sub0\.05|iid_sub0\.01|noniid_sub0\.05|noniid_sub0\.01)_"
        r"[a-zA-Z0-9_.]+(_ep\d+)?(_lr[0-9.e-]+)?(_r\d+)?_seed\d+$"
    )
    runs_by_block = prc.expand_all(revision)
    for block, runs in runs_by_block.items():
        for run in runs:
            assert pattern.match(run.output_dir), f"{block}: bad dir name {run.output_dir!r}"


def test_default_valued_cells_share_directory_across_blocks(revision):
    """mu=0.01/E=5/lr=0.001 default cells should collapse onto the same dir
    used by other blocks (documented dedup behavior)."""
    mu_runs = {r.output_dir for r in prc.BLOCK_EXPANDERS["mu_grid"](revision["mu_grid"])
               if r.strategy == "fedprox_0.01"}
    epoch_runs = {r.output_dir for r in prc.BLOCK_EXPANDERS["local_epochs"](revision["local_epochs"])
                  if r.strategy == "fedprox_0.01" and r.local_epochs == 5}
    lr_runs = {r.output_dir for r in prc.BLOCK_EXPANDERS["learning_rate"](revision["learning_rate"])
               if r.strategy == "fedprox_0.01" and r.lr == 0.001}
    assert mu_runs == epoch_runs == lr_runs == {
        "results/rev_noniid_fedprox_0.01_seed42",
        "results/rev_noniid_fedprox_0.01_seed123",
        "results/rev_noniid_fedprox_0.01_seed456",
    }


def test_fl_run_has_server_and_three_client_commands(revision):
    runs = prc.BLOCK_EXPANDERS["dirichlet_skew"](revision["dirichlet_skew"])
    run = runs[0]
    assert run.kind == "fl"
    server_cmd = run.server_command()
    assert "src/fl/server.py" in server_cmd
    assert f"--output_dir {run.output_dir}" in server_cmd
    assert f"--tag {run.dist}" in server_cmd
    client_cmds = run.client_commands()
    assert len(client_cmds) == 3
    for cmd in client_cmds:
        assert "src/fl/client.py" in cmd
        assert "--server 192.168.1.4:8080" in cmd


def test_baseline_run_epochs_and_command(revision):
    runs = prc.BLOCK_EXPANDERS["baselines_extension"](revision["baselines_extension"])
    centralized = next(r for r in runs if r.baseline_type == "centralized")
    local_only = next(r for r in runs if r.baseline_type == "local_only")
    assert centralized.epochs == 15
    assert "train_centralized.py" in centralized.baseline_command()
    assert "--epochs 15" in centralized.baseline_command()
    assert "train_local.py --batch --cross_eval" in local_only.baseline_command()


def test_long_horizon_dir_has_round_suffix(revision):
    runs = prc.BLOCK_EXPANDERS["long_horizon_fedbn"](revision["long_horizon_fedbn"])
    for run in runs:
        assert run.output_dir.endswith(f"_r10_seed{run.seed}")


# --------------------------------------------------------------------------- #
# CLI smoke tests (subprocess, using the same interpreter running pytest)
# --------------------------------------------------------------------------- #
def test_cli_text_format_run_count():
    out = subprocess.run(
        [PYTHON, "scripts/print_revision_commands.py", "--format", "text"],
        capture_output=True, text=True, check=True,
    )
    dirs = re.findall(r"^--- (results/rev_\S+) ---$", out.stdout, flags=re.MULTILINE)
    # 151 logical cells, 9 of which share a results directory with an earlier
    # block (the default-valued cells of the mu / epoch / lr sweeps) and are
    # emitted once, as [DUP], instead of being launched again.
    assert len(dirs) == len(set(dirs)) == 142
    assert out.stdout.count("[DUP]") == 9


def test_cli_block_filter():
    out = subprocess.run(
        [PYTHON, "scripts/print_revision_commands.py", "--block", "mu_grid", "--format", "text"],
        capture_output=True, text=True, check=True,
    )
    n_dirs = len(re.findall(r"^--- (results/rev_\S+) ---$", out.stdout, flags=re.MULTILINE))
    assert n_dirs == 15
    assert "mu_grid" in out.stdout


def test_cli_bash_format_smoke():
    out = subprocess.run(
        [PYTHON, "scripts/print_revision_commands.py", "--block", "long_horizon_fedbn", "--format", "bash"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.startswith("#!/bin/bash")
    assert "sleep 10" in out.stdout
    assert "sleep 30" in out.stdout
    assert "src/fl/server.py" in out.stdout


def test_main_returns_zero(revision, capsys):
    rc = prc.main(["--block", "learning_rate"])
    assert rc == 0


def test_no_results_directory_is_launched_twice(revision):
    """Every emitted command must target a distinct results directory.

    The mu / local-epoch / learning-rate sweeps intentionally reuse the
    seed-extension directory at their default value.  ``--skip_existing`` only
    inspects the filesystem while the list is generated, so a script generated
    on an empty ``results/`` would otherwise run those configurations up to four
    times -- 1.5-3 h of testbed time each, with every repeat overwriting the
    previous results.json.
    """
    for all_seeds in (False, True):
        cfg = dict(revision)
        cfg["seed_extension"] = dict(cfg["seed_extension"], all_seeds=all_seeds)
        runs = [r for block in prc.expand_all(cfg).values() for r in block]

        seen = set()
        emitted = [r for r in runs if prc.emit_status(r, False, seen) == "run"]
        emitted_dirs = [r.output_dir for r in emitted]
        assert len(emitted_dirs) == len(set(emitted_dirs))
        assert set(emitted_dirs) == {r.output_dir for r in runs}  # nothing is lost
        assert len(emitted_dirs) == len({r.output_dir for r in runs})


def test_cli_bash_never_repeats_a_server_command():
    out = subprocess.run(
        [PYTHON, "scripts/print_revision_commands.py", "--format", "bash"],
        capture_output=True, text=True, check=True,
    )
    output_dirs = re.findall(r"--output_dir (results/rev_\S+)", out.stdout)
    assert output_dirs
    assert len(output_dirs) == len(set(output_dirs))
