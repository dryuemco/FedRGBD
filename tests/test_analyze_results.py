"""Tests for scripts/analyze_results.py.

Builds synthetic ``results/`` trees in ``tmp_path`` covering every input format
(old FL, new schema-2 FL, centralized, local-only), then checks label parsing,
estimation flags, the summary statistics, the pairwise tests, the figures and
the end-to-end CLI.  The last test runs the CLI over the *real* repository
results read-only, writing only into ``tmp_path``.
"""

import json
import math
import os
import sys

import numpy as np
import pytest
from scipy import stats

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.analyze_results import (  # noqa: E402
    IEEE_STYLE,
    collect_runs,
    friedman_tests,
    load_run,
    main,
    make_plots,
    pairwise_tests,
    parse_distribution,
    parse_mu,
    summarize,
)

NEW_METRICS = [
    "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
    "f1", "macro_f1", "mcc", "roc_auc",
]


# --------------------------------------------------------------------------- #
# synthetic result builders
# --------------------------------------------------------------------------- #
def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def write_old_fl_run(root, name, strategy, accs, losses, seed=None,
                     total_time_s=6000.0, min_clients=3):
    """Old-format FL results.json (accuracy only, no per-round times/comm)."""
    payload = {
        "strategy": strategy,
        "num_rounds": len(accs),
        "total_time_s": total_time_s,
        "timestamp": "2026-03-28T14:50:10.031418",
        "losses_distributed": [
            {"round": i + 1, "loss": v} for i, v in enumerate(losses)
        ],
        "metrics_distributed": {
            "accuracy": [{"round": i + 1, "value": v} for i, v in enumerate(accs)]
        },
    }
    if min_clients is not None:
        payload["min_clients"] = min_clients
    if seed is not None:
        payload["seed"] = seed
    _write_json(os.path.join(root, name, "results.json"), payload)
    return os.path.join(root, name)


def write_new_fl_run(root, name, strategy, accs, losses, seed, tags,
                     proximal_mu=None, nodes=("node_a", "node_b", "node_c"),
                     payload_bytes=6_128_344):
    """New schema-2 FL results.json with rounds/client_config/tags."""
    n_rounds = len(accs)
    metrics_distributed = {}
    for k, metric in enumerate(NEW_METRICS):
        metrics_distributed[metric] = [
            {"round": i + 1, "value": min(0.999, accs[i] - 0.001 * k)}
            for i in range(n_rounds)
        ]
    metrics_distributed["accuracy"] = [
        {"round": i + 1, "value": accs[i]} for i in range(n_rounds)
    ]
    metrics_distributed["loss"] = [
        {"round": i + 1, "value": losses[i]} for i in range(n_rounds)
    ]

    rounds = []
    elapsed = 0.0
    cumulative = 0
    for i in range(n_rounds):
        elapsed += 1200.0
        cumulative += payload_bytes * 2 * len(nodes)
        fit_clients = {
            node: {
                "train_loss": losses[i] * 1.1,
                "fit_time_s": 900.0,
                "payload_bytes_up": payload_bytes,
                "payload_bytes_down": payload_bytes,
                "num_examples": 1000,
            }
            for node in nodes
        }
        eval_clients = {
            node: {
                "loss": losses[i],
                "accuracy": accs[i],
                "balanced_accuracy": accs[i] - 0.001,
                "confusion_matrix": [[480, 20], [10, 490]],
                "eval_time_s": 120.0,
                "payload_bytes_down": payload_bytes,
                "num_examples": 1000,
            }
            for node in nodes
        }
        aggregate = {m: metrics_distributed[m][i]["value"] for m in NEW_METRICS}
        aggregate["loss"] = losses[i]
        aggregate["pooled"] = dict(aggregate)
        rounds.append({
            "round": i + 1,
            "fit": {"clients": fit_clients, "aggregate": {"train_loss": losses[i] * 1.1},
                    "elapsed_s": elapsed - 200.0},
            "evaluate": {"clients": eval_clients, "aggregate": aggregate,
                         "elapsed_s": elapsed},
            "cumulative_communication_bytes": cumulative,
        })

    payload = {
        "results_schema_version": 2,
        "strategy": strategy,
        "num_rounds": n_rounds,
        "min_clients": len(nodes),
        "seed": seed,
        "total_time_s": elapsed,
        "timestamp": "2026-09-01T10:00:00.000000",
        "tags": list(tags),
        "model_payload_bytes": payload_bytes,
        "client_config": {
            node: {
                "local_epochs": 5,
                "lr": 0.001,
                "batch_size": 8,
                "data_dir": "data/processed/{}/{}".format(tags[0], node),
                "hostname": "fedrgbd-{}".format(node[-1]),
            }
            for node in nodes
        },
        "losses_distributed": [
            {"round": i + 1, "loss": v} for i, v in enumerate(losses)
        ],
        "metrics_distributed": metrics_distributed,
        "rounds": rounds,
    }
    if proximal_mu is not None:
        payload["proximal_mu"] = proximal_mu
    _write_json(os.path.join(root, name, "results.json"), payload)
    return os.path.join(root, name)


def _history(n_epochs, base_acc, base_loss, epoch_time=300.0):
    return [
        {
            "epoch": e + 1,
            "train_loss": base_loss * (1.0 - 0.02 * e),
            "val_loss": base_loss * (1.0 - 0.03 * e),
            "val_accuracy": min(0.999, base_acc + 0.0005 * e),
            "epoch_time_s": epoch_time,
        }
        for e in range(n_epochs)
    ]


def write_centralized_run(root, name, seed, final_acc=0.9952, epochs=15):
    history = _history(epochs, 0.988, 0.02, 1000.0)
    payload = {
        "experiment": "centralized",
        "hostname": "fedrgbd-a",
        "data_dirs": ["data/processed/noniid/node_{}".format(c) for c in "abc"],
        "epochs": epochs,
        "batch_size": 8,
        "lr": 0.001,
        "seed": seed,
        "total_time_s": epochs * 1000.0,
        "final_test_loss": 0.0161,
        "final_test_accuracy": final_acc,
        "per_node_test": {
            "node_{}".format(c): {"test_loss": 0.02, "test_accuracy": final_acc}
            for c in "abc"
        },
        "history": history,
        "fl_round_equivalents": {
            "round_{}".format(r): history[r * 5 - 1] for r in (1, 2, 3)
        },
        "timestamp": "2026-04-01T22:33:22.988971",
    }
    _write_json(os.path.join(root, name, "results.json"), payload)
    return os.path.join(root, name)


def write_local_run(root, name, seed, epochs=15, nodes=("node_a", "node_b", "node_c")):
    run_dir = os.path.join(root, name)
    node_acc = {}
    for i, node in enumerate(nodes):
        history = _history(epochs, 0.985 + 0.001 * i, 0.03, 350.0)
        final_acc = 0.9943 + 0.0005 * i
        node_acc[node] = final_acc
        _write_json(os.path.join(run_dir, node, "results.json"), {
            "experiment": "local_only",
            "node_name": node,
            "hostname": "fedrgbd-{}".format(node[-1]),
            "data_dir": "data/processed/noniid/{}".format(node),
            "epochs": epochs,
            "batch_size": 8,
            "lr": 0.001,
            "seed": seed,
            "total_time_s": epochs * 350.0,
            "final_test_loss": 0.016,
            "final_test_accuracy": final_acc,
            "cross_eval": {
                other: {"test_loss": 0.02, "test_accuracy": 0.99}
                for other in nodes if other != node
            },
            "history": history,
        })
    _write_json(os.path.join(run_dir, "summary.json"), {
        "experiment": "local_only_batch",
        "seed": seed,
        "epochs": epochs,
        "total_time_s": epochs * 350.0 * len(nodes),
        "nodes": {
            node: {"final_test_accuracy": acc, "final_test_loss": 0.016, "cross_eval": {}}
            for node, acc in node_acc.items()
        },
        "mean_test_accuracy": float(np.mean(list(node_acc.values()))),
        "timestamp": "2026-04-01T23:54:25.814683",
    })
    return run_dir


#: final-round accuracies of the synthetic old-format FL runs, per seed
FEDAVG_ACCS = {42: 0.9910, 123: 0.9902, 456: 0.9921}
FEDPROX_ACCS = {42: 0.9949, 123: 0.9938, 456: 0.9955}
FEDBN_ACCS = {42: 0.9925, 123: 0.9917, 456: 0.9931}


@pytest.fixture()
def synthetic_results(tmp_path):
    """A complete synthetic ``results/`` tree covering every supported format."""
    root = str(tmp_path / "results")
    os.makedirs(root, exist_ok=True)

    # (a) old-format FL: fedavg and fedprox_0.01, three seeds each
    for seed, acc in FEDAVG_ACCS.items():
        payload_seed = None if seed == 456 else seed  # seed only in the dir name
        write_old_fl_run(
            root, "3node_noniid_fedavg_seed{}".format(seed), "fedavg",
            [0.54 + 0.001 * (seed % 7), 0.98, acc], [0.78, 0.12, 0.042],
            seed=payload_seed, total_time_s=6000.0 + seed,
        )
    for seed, acc in FEDPROX_ACCS.items():
        write_old_fl_run(
            root, "3node_noniid_fedprox_0.01_seed{}".format(seed), "fedprox_0.01",
            [0.94 + 0.001 * (seed % 5), 0.99, acc], [0.15, 0.03, 0.021],
            seed=seed, total_time_s=10000.0 + seed,
        )
    for seed, acc in FEDBN_ACCS.items():
        write_old_fl_run(
            root, "3n_noniid_fedbn_seed{}".format(seed), "fedbn",
            [0.90, 0.985, acc], [0.30, 0.05, 0.03], seed=seed, total_time_s=6400.0 + seed,
        )
    # an old file with no ``seed`` key and no seed in the directory name
    write_old_fl_run(
        root, "3node_noniid_fedavg", "fedavg",
        [0.5366, 0.9915, 0.9910], [0.7849, 0.1156, 0.0420], seed=None,
    )

    # (b) new-format (schema 2) runs
    write_new_fl_run(root, "3node_dirichlet01_fedavg_seed42", "fedavg",
                     [0.90, 0.97, 0.9880], [0.30, 0.09, 0.03], 42, ["dirichlet_0.1"])
    write_new_fl_run(root, "3node_dirichlet01_fedavg_seed123", "fedavg",
                     [0.89, 0.96, 0.9865], [0.31, 0.10, 0.035], 123, ["dirichlet_0.1"])

    # (c) centralized and local-only baselines
    write_centralized_run(root, "centralized_noniid_seed42", 42)
    write_local_run(root, "local_noniid_seed42", 42)

    # a test_* directory that must be skipped by default
    write_old_fl_run(root, "test_run_noniid", "fedavg", [0.5, 0.9, 0.95],
                     [0.9, 0.2, 0.1], seed=7)
    return root


@pytest.fixture()
def synthetic_runs(synthetic_results):
    return collect_runs(synthetic_results, warn=False)


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #
def test_parse_mu_variants():
    assert parse_mu("fedprox_0.01") == pytest.approx(0.01)
    assert parse_mu("fedprox_0.1") == pytest.approx(0.1)
    assert parse_mu("fedprox001") == pytest.approx(0.01)
    assert parse_mu("fedavg", proximal_mu=0.5) == pytest.approx(0.5)
    assert parse_mu("fedavg") is None


def test_parse_distribution_sources():
    assert parse_distribution({"tags": ["dirichlet_0.1"]}, "whatever") == "dirichlet_0.1"
    assert parse_distribution({}, "3node_noniid_fedavg_seed42") == "non_iid_label"
    assert parse_distribution({}, "3node_iid_fedavg_seed42") == "iid"
    assert parse_distribution({}, "3node_dirichlet_0.5_seed1") == "dirichlet_0.5"
    assert parse_distribution({}, "3node_iid_sub0.25_seed1") == "iid_sub0.25"
    assert parse_distribution({}, "some_unlabelled_run") == "unknown"
    cfg = {"node_a": {"data_dir": "data/processed/noniid/node_a"}}
    assert parse_distribution({}, "run", cfg) == "non_iid_label"


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #
def test_collect_runs_strategy_distribution_seed(synthetic_runs):
    by_name = {r["run_name"]: r for r in synthetic_runs}
    # test_* skipped by default
    assert "test_run_noniid" not in by_name
    # 3 fedavg + 3 fedprox + 3 fedbn + 1 unseeded + 2 new + centralized + local
    assert len(synthetic_runs) == 14

    for seed in FEDAVG_ACCS:
        rec = by_name["3node_noniid_fedavg_seed{}".format(seed)]
        assert rec["kind"] == "fl"
        assert rec["strategy"] == "fedavg"
        assert rec["distribution"] == "non_iid_label"
        assert rec["seed"] == seed          # seed 456 comes from the directory name
        assert rec["n_nodes"] == 3
        assert rec["mu"] is None

    prox = by_name["3node_noniid_fedprox_0.01_seed42"]
    assert prox["strategy"] == "fedprox_0.01"
    assert prox["mu"] == pytest.approx(0.01)
    assert prox["strategy_display"] == "FedProx(mu=0.01)"
    assert prox["label"].startswith("FedProx(mu=0.01) non_iid_label")

    unseeded = by_name["3node_noniid_fedavg"]
    assert unseeded["seed"] is None
    assert unseeded["seed_label"] == "NA"

    new_run = by_name["3node_dirichlet01_fedavg_seed42"]
    assert new_run["schema_version"] == 2
    assert new_run["distribution"] == "dirichlet_0.1"
    assert new_run["local_epochs"] == 5
    assert new_run["lr"] == pytest.approx(0.001)
    assert new_run["n_nodes"] == 3

    central = by_name["centralized_noniid_seed42"]
    assert central["kind"] == "centralized"
    assert central["strategy"] == "centralized"
    assert central["distribution"] == "non_iid_label"
    assert central["seed"] == 42
    assert central["final_accuracy"] == pytest.approx(0.9952)
    assert len(central["curve"]) == 3

    local = by_name["local_noniid_seed42"]
    assert local["kind"] == "local"
    assert local["strategy"] == "local_only"
    assert local["distribution"] == "non_iid_label"
    assert local["n_nodes"] == 3
    # curve = mean over nodes of val_accuracy at epochs 5, 10, 15
    assert len(local["curve"]) == 3
    assert local["final_accuracy"] == pytest.approx(np.mean([0.9943, 0.9948, 0.9953]), abs=1e-9)


def test_include_test_runs_flag(synthetic_results):
    runs = collect_runs(synthetic_results, include_test_runs=True, warn=False)
    assert any(r["run_name"] == "test_run_noniid" for r in runs)


def test_old_runs_estimated_new_runs_not(synthetic_runs):
    by_name = {r["run_name"]: r for r in synthetic_runs}

    old = by_name["3node_noniid_fedavg_seed42"]
    assert old["time_estimated"] is True
    assert old["comm_estimated"] is True
    total = 6042.0
    assert old["curve"][-1]["elapsed_s"] == pytest.approx(total)
    assert old["curve"][0]["elapsed_s"] == pytest.approx(total / 3.0)
    # payload * 2 * n_nodes * round / 1e6, monotone in round
    assert old["curve"][1]["cumulative_mb"] == pytest.approx(2 * old["curve"][0]["cumulative_mb"])
    assert old["curve"][0]["cumulative_mb"] > 0

    new = by_name["3node_dirichlet01_fedavg_seed42"]
    assert new["time_estimated"] is False
    assert new["comm_estimated"] is False
    assert new["curve"][-1]["elapsed_s"] == pytest.approx(3600.0)
    assert new["curve"][-1]["cumulative_mb"] == pytest.approx(6_128_344 * 2 * 3 * 3 / 1e6)
    # every metric in metrics_distributed is kept
    for metric in NEW_METRICS:
        assert metric in new["curve"][-1]
        assert metric in new["metrics_final"]


def test_load_run_accepts_file_and_directory(synthetic_results):
    run_dir = os.path.join(synthetic_results, "3node_noniid_fedavg_seed42")
    from_dir = load_run(run_dir, warn=False)
    from_file = load_run(os.path.join(run_dir, "results.json"), warn=False)
    assert from_dir["config_id"] == from_file["config_id"]
    assert from_dir["final_accuracy"] == pytest.approx(from_file["final_accuracy"])


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def test_runs_table_has_one_row_per_run(synthetic_runs):
    tables = summarize(synthetic_runs)
    runs_df = tables["runs"]
    assert len(runs_df) == len(synthetic_runs)
    for column in ("kind", "strategy", "distribution", "seed_label", "final_accuracy",
                   "best_accuracy", "total_time_s"):
        assert column in runs_df.columns
    row = runs_df[runs_df["run_name"] == "3node_noniid_fedprox_0.01_seed42"].iloc[0]
    assert row["final_accuracy"] == pytest.approx(FEDPROX_ACCS[42])
    assert row["best_accuracy"] >= row["final_accuracy"]


def test_summary_ci_matches_scipy_hand_computation(synthetic_runs):
    tables = summarize(synthetic_runs)
    summary = tables["summary"]

    prox = [r for r in synthetic_runs if r["strategy"] == "fedprox_0.01"]
    config_id = prox[0]["config_id"]
    values = np.array([r["final_accuracy"] for r in prox], dtype=float)
    n = values.size
    assert n == 3

    expected_mean = float(values.mean())
    expected_std = float(np.std(values, ddof=1))
    expected_ci = float(stats.t.ppf(0.975, n - 1) * expected_std / math.sqrt(n))

    row = summary[(summary["config_id"] == config_id) &
                  (summary["metric"] == "final_accuracy")].iloc[0]
    assert int(row["n_seeds"]) == n
    assert row["mean"] == pytest.approx(expected_mean)
    assert row["std"] == pytest.approx(expected_std)
    assert row["ci95"] == pytest.approx(expected_ci)
    assert row["ci_low"] == pytest.approx(expected_mean - expected_ci)
    assert row["ci_high"] == pytest.approx(expected_mean + expected_ci)
    assert row["min"] == pytest.approx(values.min())
    assert row["max"] == pytest.approx(values.max())

    # CI is NaN when a configuration has a single replicate
    central = [r for r in synthetic_runs if r["kind"] == "centralized"][0]
    crow = summary[(summary["config_id"] == central["config_id"]) &
                   (summary["metric"] == "final_accuracy")].iloc[0]
    assert int(crow["n_seeds"]) == 1
    assert math.isnan(float(crow["ci95"]))
    assert math.isnan(float(crow["std"]))

    # new-format runs contribute every metric at the final round
    new_run = [r for r in synthetic_runs if r["schema_version"] == 2][0]
    metrics = set(summary[summary["config_id"] == new_run["config_id"]]["metric"])
    for metric in NEW_METRICS:
        assert "final_" + metric in metrics
    assert "total_time_s" in metrics
    assert "round1_accuracy" in metrics


def test_per_round_table(synthetic_runs):
    per_round = summarize(synthetic_runs)["per_round"]
    prox = [r for r in synthetic_runs if r["strategy"] == "fedprox_0.01"]
    sub = per_round[per_round["config_id"] == prox[0]["config_id"]]
    assert sorted(sub["round"].tolist()) == [1, 2, 3]
    last = sub[sub["round"] == 3].iloc[0]
    values = [r["curve"][-1]["accuracy"] for r in prox]
    assert last["accuracy_mean"] == pytest.approx(float(np.mean(values)))
    assert last["accuracy_std"] == pytest.approx(float(np.std(values, ddof=1)))
    assert last["accuracy_ci95"] == pytest.approx(
        float(stats.t.ppf(0.975, 2) * np.std(values, ddof=1) / math.sqrt(3)))
    assert int(last["n_seeds"]) == 3


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def test_pairwise_tests_have_finite_effect_sizes(synthetic_runs):
    pairs = pairwise_tests(synthetic_runs, metric="accuracy")
    assert not pairs.empty

    noniid = pairs[pairs["distribution"] == "non_iid_label"]
    assert not noniid.empty
    names = {(r["strategy_a"], r["strategy_b"]) for _, r in noniid.iterrows()}
    assert ("FedAvg", "FedProx(mu=0.01)") in names

    row = noniid[(noniid["strategy_a"] == "FedAvg") &
                 (noniid["strategy_b"] == "FedProx(mu=0.01)")].iloc[0]
    assert int(row["n_seeds"]) == 3
    assert np.isfinite(row["cohen_d_paired"])
    assert np.isfinite(row["cohen_d_unpaired"])
    assert np.isfinite(row["ttest_p"])
    assert np.isfinite(row["wilcoxon_p"])
    assert row["mean_diff"] < 0  # FedProx is better in the synthetic data

    a = np.array([FEDAVG_ACCS[s] for s in sorted(FEDAVG_ACCS)])
    b = np.array([FEDPROX_ACCS[s] for s in sorted(FEDPROX_ACCS)])
    diffs = a - b
    expected_d = float(np.mean(diffs) / np.std(diffs, ddof=1))
    assert row["cohen_d_paired"] == pytest.approx(expected_d)
    assert row["mean_a"] == pytest.approx(float(a.mean()))
    assert row["mean_b"] == pytest.approx(float(b.mean()))

    # single-seed configurations cannot be paired -> no rows for them
    assert not ((pairs["strategy_a"] == "Centralized") &
                (pairs["strategy_b"] == "Local-only")).any()


def test_pairwise_handles_identical_runs(tmp_path):
    root = str(tmp_path / "results")
    for seed in (1, 2, 3):
        write_old_fl_run(root, "3node_iid_fedavg_seed{}".format(seed), "fedavg",
                         [0.5, 0.9, 0.99], [0.7, 0.2, 0.05], seed=seed)
        write_old_fl_run(root, "3node_iid_fedbn_seed{}".format(seed), "fedbn",
                         [0.5, 0.9, 0.99], [0.7, 0.2, 0.05], seed=seed)
    runs = collect_runs(root, warn=False)
    pairs = pairwise_tests(runs, metric="accuracy")
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert math.isnan(float(row["wilcoxon_p"]))       # all differences are zero
    assert "zero" in str(row["note"])
    assert math.isnan(float(row["cohen_d_paired"]))   # std of diffs is zero


def test_friedman(synthetic_runs):
    friedman = friedman_tests(synthetic_runs, metric="accuracy")
    assert not friedman.empty
    row = friedman[friedman["distribution"] == "non_iid_label"].iloc[0]
    assert int(row["n_strategies"]) >= 3
    assert int(row["n_seeds"]) >= 2
    assert np.isfinite(row["chi_square"])
    assert np.isfinite(row["p_value"])


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def test_make_plots_writes_non_empty_figures(synthetic_runs, tmp_path):
    out = str(tmp_path / "figs")
    written = make_plots(synthetic_runs, out, metric="accuracy")
    assert written

    expected = [
        "accuracy_vs_round_non_iid_label.png",
        "accuracy_vs_round_non_iid_label.pdf",
        "accuracy_vs_time_non_iid_label.png",
        "accuracy_vs_communication_non_iid_label.png",
        "accuracy_vs_round_dirichlet_0.1.png",
        "accuracy_vs_round_all.png",
        "final_metrics_dirichlet_0.1.png",
    ]
    for name in expected:
        path = os.path.join(out, name)
        assert os.path.isfile(path), "missing figure {}".format(name)
        assert os.path.getsize(path) > 0

    # accuracy-only (old-format) distributions get no all-metrics bar chart
    assert not os.path.isfile(os.path.join(out, "final_metrics_non_iid_label.png"))
    assert IEEE_STYLE["font.family"] == "serif"


# --------------------------------------------------------------------------- #
# end-to-end CLI
# --------------------------------------------------------------------------- #
def test_main_end_to_end_on_synthetic_tree(synthetic_results, tmp_path):
    out = str(tmp_path / "analysis")
    rc = main(["--results_dir", synthetic_results, "--output_dir", out])
    assert rc == 0

    for name in ("runs.csv", "summary_table.csv", "summary_table.md",
                 "per_round_table.csv", "pairwise_tests.csv", "pairwise_tests.md",
                 "friedman.csv", "accuracy_vs_round_non_iid_label.png"):
        path = os.path.join(out, name)
        assert os.path.isfile(path), "missing output {}".format(name)
        assert os.path.getsize(path) > 0

    with open(os.path.join(out, "summary_table.md"), encoding="utf-8") as fh:
        markdown = fh.read()
    assert "±" in markdown
    assert "FedProx(mu=0.01)" in markdown

    import pandas as pd
    runs_df = pd.read_csv(os.path.join(out, "runs.csv"))
    assert len(runs_df) == 14


def test_main_no_plots_and_options(synthetic_results, tmp_path):
    out = str(tmp_path / "noplots")
    rc = main([
        "--results_dir", synthetic_results,
        "--output_dir", out,
        "--no_plots",
        "--missing_seed", "999",
        "--payload_bytes", "6100000",
        "--local_epochs_equiv", "5",
        "--metric", "accuracy",
    ])
    assert rc == 0
    assert os.path.isfile(os.path.join(out, "runs.csv"))
    assert not [f for f in os.listdir(out) if f.endswith(".png")]

    import pandas as pd
    runs_df = pd.read_csv(os.path.join(out, "runs.csv"))
    unseeded = runs_df[runs_df["run_name"] == "3node_noniid_fedavg"].iloc[0]
    assert int(unseeded["seed"]) == 999  # --missing_seed applied


def test_main_rejects_output_inside_results(synthetic_results):
    with pytest.raises(SystemExit):
        main(["--results_dir", synthetic_results,
              "--output_dir", os.path.join(synthetic_results, "analysis")])


def test_main_on_real_repository_results(tmp_path):
    """Read-only smoke test over the real results/ tree; writes only to tmp_path."""
    real_results = os.path.join(_REPO_ROOT, "results")
    if not os.path.isdir(real_results):
        pytest.skip("no results/ directory in the repository")

    out = str(tmp_path / "real")
    cwd = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        rc = main(["--results_dir", "results", "--output_dir", out])
    finally:
        os.chdir(cwd)
    assert rc == 0

    import pandas as pd
    runs_df = pd.read_csv(os.path.join(out, "runs.csv"))
    assert len(runs_df) > 0
    assert set(runs_df["kind"]) <= {"fl", "centralized", "local"}
    assert not runs_df["run_name"].astype(str).str.startswith("test_").any()
    assert os.path.isfile(os.path.join(out, "summary_table.md"))
