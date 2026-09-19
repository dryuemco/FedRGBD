"""analyze_results.py / export_latex_tables.py under the model-selection protocol.

Schema-3 FL runs carry validation and test metrics for every round.  The
analysis must report the test metrics of the round with the lowest weighted
validation loss (earliest on ties) -- never a maximum over rounds, never the
final round -- and keep v1 rows (no per-round test metrics) in their own,
explicitly labelled column.
"""

import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.analyze_results import (  # noqa: E402
    HEADLINE_NONE,
    HEADLINE_SELECTED,
    HEADLINE_V1,
    collect_runs,
    load_run,
    main,
    make_plots,
    pairwise_tests,
    select_fl_round,
    summarize,
)
from scripts.export_latex_tables import export  # noqa: E402

NODES = ("node_a", "node_b", "node_c")
N_VAL = {"node_a": 100, "node_b": 300, "node_c": 50}
N_TEST = {"node_a": 120, "node_b": 80, "node_c": 400}


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def schema3_payload(strategy, dist, seed, val_losses, test_accs, val_accs=None,
                    round_wall=1000.0, test_overhead=40.0, payload_bytes=6_000_000):
    """A results.json as written by the schema-3 server (3 clients, same loss per client)."""
    n_rounds = len(val_losses)
    val_accs = val_accs or [0.7 + 0.05 * r for r in range(n_rounds)]
    rounds, elapsed, excl = [], 0.0, 0.0
    for i in range(n_rounds):
        elapsed += round_wall
        excl += round_wall - test_overhead
        clients = {}
        for node in NODES:
            clients[node] = {
                "loss": val_losses[i], "val_loss": val_losses[i], "accuracy": val_accs[i],
                "val_accuracy": val_accs[i], "val_n_examples": N_VAL[node],
                "num_examples": N_VAL[node],
                "test_accuracy": test_accs[i], "test_balanced_accuracy": test_accs[i] - 0.01,
                "test_mcc": 2 * test_accs[i] - 1, "test_loss": 1.0 - test_accs[i],
                "test_n_examples": N_TEST[node],
                "test_confusion_matrix": [[10, 2], [3, 15]],
                "eval_time_s": 50.0, "val_eval_time_s": 48.0, "test_eval_time_s": 40.0,
                "eval_wall_s": 90.0,
            }
        aggregate = {
            "accuracy": val_accs[i], "loss": val_losses[i], "val_accuracy": val_accs[i],
            "val_loss": val_losses[i],
            "test_accuracy": test_accs[i], "test_balanced_accuracy": test_accs[i] - 0.01,
            "test_mcc": 2 * test_accs[i] - 1, "test_loss": 1.0 - test_accs[i],
            "test_n_examples_total": sum(N_TEST.values()), "test_tp": 45, "test_fp": 6,
            "test_fn": 9, "test_tn": 30, "test_eval_time_s_max": 40.0,
            "pooled_test_accuracy": test_accs[i],
        }
        rounds.append({
            "round": i + 1,
            "fit": {"clients": {}, "aggregate": {}, "elapsed_s": elapsed - 100.0},
            "evaluate": {"clients": clients, "aggregate": aggregate, "elapsed_s": elapsed,
                         "elapsed_excl_test_s": excl},
            "weighted_val_loss": val_losses[i],
            "timing": {"round_wall_s": round_wall, "test_eval_overhead_s": test_overhead,
                       "round_time_s": round_wall - test_overhead, "elapsed_excl_test_s": excl},
            "cumulative_communication_bytes": payload_bytes * 2 * 3 * (i + 1),
        })
    return {
        "strategy": strategy, "num_rounds": n_rounds, "min_clients": 3, "seed": seed,
        "total_time_s": elapsed + 30.0, "timestamp": "2026-10-01T10:00:00",
        "losses_distributed": [{"round": i + 1, "loss": v} for i, v in enumerate(val_losses)],
        "metrics_distributed": {
            "accuracy": [{"round": i + 1, "value": v} for i, v in enumerate(val_accs)],
            # Flower history also carries the test aggregates; analysis must ignore them
            "test_accuracy": [{"round": i + 1, "value": v} for i, v in enumerate(test_accs)],
        },
        "results_schema_version": 3, "proximal_mu": 0.0, "tags": [dist],
        "client_config": {n: {"local_epochs": 5, "lr": 0.001, "batch_size": 8,
                              "data_dir": "data/processed/{}/{}".format(dist, n)} for n in NODES},
        "model_payload_bytes": payload_bytes,
        "total_communication_bytes": payload_bytes * 2 * 3 * n_rounds,
        "rounds": rounds,
        "model_selection": {"rule": "min_weighted_val_loss_earliest_round",
                            "selected_round": int(np.argmin(val_losses)) + 1},
        "total_test_eval_overhead_s": test_overhead * n_rounds,
        "total_time_excl_test_s": elapsed + 30.0 - test_overhead * n_rounds,
    }


def write_run(root, name, payload):
    _write(os.path.join(root, name, "results.json"), payload)
    return os.path.join(root, name)


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
def test_headline_is_the_test_metric_of_the_min_val_loss_round(tmp_path):
    # val loss is lowest at round 2; test accuracy peaks at round 3; final round is 4
    payload = schema3_payload("fedavg", "iid", 42, val_losses=[0.6, 0.2, 0.3, 0.25],
                              test_accs=[0.80, 0.85, 0.95, 0.90])
    run = load_run(write_run(str(tmp_path), "rev_iid_fedavg_seed42", payload), warn=False)
    assert run["headline_source"] == HEADLINE_SELECTED
    assert run["selected_round"] == 2
    assert run["selected_val_loss"] == pytest.approx(0.2)
    assert run["selected_test_metrics"]["accuracy"] == pytest.approx(0.85)   # not 0.95, not 0.90
    assert run["selected_test_metrics"]["mcc"] == pytest.approx(0.70)
    assert run["selected_test_metrics"]["n_examples"] == sum(N_TEST.values())
    assert run["selected_test_metrics"]["tp"] == 45
    assert "best_accuracy" not in run


def test_ties_go_to_the_earlier_round(tmp_path):
    payload = schema3_payload("fedavg", "iid", 42, val_losses=[0.5, 0.2, 0.2],
                              test_accs=[0.7, 0.8, 0.9])
    run = load_run(write_run(str(tmp_path), "rev_iid_fedavg_seed42", payload), warn=False)
    assert run["selected_round"] == 2
    assert run["selected_test_metrics"]["accuracy"] == pytest.approx(0.8)


def test_selection_is_recomputed_from_client_val_losses_weighted_by_val_size():
    """Round 1: small clients low, big client high; round 2 the reverse."""
    payload = schema3_payload("fedavg", "iid", 42, val_losses=[0.4, 0.4], test_accs=[0.8, 0.9])
    r1, r2 = (payload["rounds"][i]["evaluate"]["clients"] for i in (0, 1))
    for node, (l1, l2) in {"node_a": (0.1, 0.6), "node_b": (0.6, 0.3),
                           "node_c": (0.1, 0.6)}.items():
        r1[node]["val_loss"] = l1
        r2[node]["val_loss"] = l2
    # weighted: r1 = (100*.1 + 300*.6 + 50*.1)/450 = 0.433; r2 = (60 + 90 + 30)/450 = 0.4
    # unweighted means would be r1 = 0.267 < r2 = 0.5
    sel = select_fl_round(payload)
    assert sel["val_loss_by_round"][1] == pytest.approx(195.0 / 450)
    assert sel["val_loss_by_round"][2] == pytest.approx(180.0 / 450)
    assert sel["selected_round"] == 2


def test_test_metrics_never_influence_selection():
    base = schema3_payload("fedavg", "iid", 42, val_losses=[0.6, 0.2, 0.3, 0.25],
                           test_accs=[0.80, 0.85, 0.95, 0.90])
    expected = select_fl_round(base)["selected_round"]
    rng = np.random.RandomState(0)
    for _ in range(10):
        poisoned = copy.deepcopy(base)
        for entry in poisoned["rounds"]:
            ev = entry["evaluate"]
            for row in list(ev["clients"].values()) + [ev["aggregate"]]:
                for key in list(row):
                    if key.startswith(("test_", "pooled_test_")) and \
                            isinstance(row[key], (int, float)):
                        row[key] = float(rng.rand() * 10)
        assert select_fl_round(poisoned)["selected_round"] == expected


def test_declared_selection_mismatch_warns_and_uses_the_recomputed_round(tmp_path, capsys):
    payload = schema3_payload("fedavg", "iid", 42, val_losses=[0.6, 0.2, 0.3],
                              test_accs=[0.8, 0.85, 0.95])
    payload["model_selection"]["selected_round"] = 3          # wrong on purpose
    run = load_run(write_run(str(tmp_path), "rev_iid_fedavg_seed42", payload), warn=True)
    assert run["selected_round"] == 2
    assert "recomputed" in capsys.readouterr().out


def test_no_finite_val_loss_gives_no_headline(tmp_path):
    payload = schema3_payload("fedavg", "iid", 42, val_losses=[0.6, 0.2],
                              test_accs=[0.8, 0.9])
    for entry in payload["rounds"]:
        for row in entry["evaluate"]["clients"].values():
            row["val_loss"] = row["loss"] = None     # json_safe writes a NaN loss as null
    run = load_run(write_run(str(tmp_path), "rev_iid_fedavg_seed42", payload), warn=False)
    assert run["headline_source"] == HEADLINE_NONE
    assert run["selected_round"] is None and not run["selected_test_metrics"]


# --------------------------------------------------------------------------- #
# test metrics stay out of everything else
# --------------------------------------------------------------------------- #
def test_curves_and_final_metrics_never_contain_test_keys(tmp_path):
    payload = schema3_payload("fedavg", "iid", 42, val_losses=[0.6, 0.2, 0.3],
                              test_accs=[0.8, 0.85, 0.95])
    run = load_run(write_run(str(tmp_path), "rev_iid_fedavg_seed42", payload), warn=False)
    for point in run["curve"]:
        assert not any(k.startswith(("test_", "pooled_test_")) for k in point), point
    assert not any(k.startswith(("test_", "pooled_test_")) for k in run["metrics_final"])
    # the per-round curve is validation accuracy
    assert [p["accuracy"] for p in run["curve"]] == pytest.approx([0.70, 0.75, 0.80])


def test_reported_time_excludes_the_test_pass(tmp_path):
    payload = schema3_payload("fedavg", "iid", 42, val_losses=[0.6, 0.2, 0.3],
                              test_accs=[0.8, 0.85, 0.95], round_wall=1000.0, test_overhead=40.0)
    run = load_run(write_run(str(tmp_path), "rev_iid_fedavg_seed42", payload), warn=False)
    assert run["total_time_s"] == pytest.approx(3030.0 - 120.0)
    assert run["total_time_raw_s"] == pytest.approx(3030.0)
    assert [p["elapsed_s"] for p in run["curve"]] == pytest.approx([960.0, 1920.0, 2880.0])
    assert run["time_estimated"] is False


# --------------------------------------------------------------------------- #
# tables: separate columns, protocol-aware statistics
# --------------------------------------------------------------------------- #
@pytest.fixture()
def mixed_results(tmp_path):
    """Revision (schema 3) and v1 runs of the same strategies, distribution and seeds."""
    root = str(tmp_path / "results")
    for seed, shift in ((42, 0.0), (123, 0.01), (456, 0.02)):
        write_run(root, "rev_non_iid_label_fedavg_seed{}".format(seed), schema3_payload(
            "fedavg", "non_iid_label", seed, [0.6, 0.2, 0.3], [0.80 + shift, 0.85 + shift, 0.9]))
        write_run(root, "rev_non_iid_label_fedprox_0.01_seed{}".format(seed), schema3_payload(
            "fedprox_0.01", "non_iid_label", seed, [0.6, 0.5, 0.1],
            [0.70 + shift, 0.75, 0.88 + shift]))
        for strategy, acc in (("fedavg", 0.99), ("fedprox_0.01", 0.98)):
            _write(os.path.join(root, "3node_noniid_{}_seed{}".format(strategy, seed),
                                "results.json"), {
                "strategy": strategy, "num_rounds": 3, "min_clients": 3, "seed": seed,
                "total_time_s": 6000.0,
                "losses_distributed": [{"round": r, "loss": 0.1} for r in (1, 2, 3)],
                "metrics_distributed": {"accuracy": [{"round": r, "value": acc - shift}
                                                     for r in (1, 2, 3)]},
            })
    return root


def test_summary_keeps_selected_test_and_v1_final_round_apart(mixed_results):
    runs = collect_runs(mixed_results, warn=False)
    tables = summarize(runs)
    summary = tables["summary"]
    rev = summary[summary["label"].str.contains("{group}", regex=False)]
    v1 = summary[summary["label"].str.contains("{image}", regex=False)]
    assert set(rev["metric"]) >= {"selected_test_accuracy", "selected_test_mcc",
                                  "selected_round", "selected_val_loss"}
    comm = "final_cumulative_mb"                 # communication volume, not a metric
    assert not any(m.startswith(("v1_final_round", "final_")) and m != comm
                   for m in rev["metric"])
    assert set(v1["metric"]) >= {"v1_final_round_accuracy"}
    assert not any(m.startswith(("selected_", "final_")) and m != comm for m in v1["metric"])

    fedavg = rev[(rev["strategy"] == "fedavg") & (rev["metric"] == "selected_test_accuracy")]
    assert float(fedavg["mean"].iloc[0]) == pytest.approx(np.mean([0.85, 0.86, 0.87]))
    rounds = rev[(rev["strategy"] == "fedavg") & (rev["metric"] == "selected_round")]
    assert float(rounds["mean"].iloc[0]) == 2.0

    runs_df = tables["runs"]
    assert "best_accuracy" not in runs_df.columns
    assert set(runs_df["headline_source"]) == {HEADLINE_SELECTED, HEADLINE_V1}


def test_pairwise_statistics_never_pair_v1_with_revision_runs(mixed_results):
    runs = collect_runs(mixed_results, warn=False)
    pairs = pairwise_tests(runs, metric="accuracy")
    assert set(pairs["protocol"]) == {"group", "image"}
    rev = pairs[pairs["protocol"] == "group"].iloc[0]
    v1 = pairs[pairs["protocol"] == "image"].iloc[0]
    # revision: selected-round test accuracy (FedAvg round 2, FedProx round 3)
    assert rev["mean_a"] == pytest.approx(np.mean([0.85, 0.86, 0.87]))
    assert rev["mean_b"] == pytest.approx(np.mean([0.88, 0.89, 0.90]))
    # v1: final-round validation accuracy, untouched by the revision runs
    assert v1["mean_a"] == pytest.approx(np.mean([0.99, 0.98, 0.97]))


def test_per_client_selected_table(mixed_results):
    runs = collect_runs(mixed_results, warn=False)
    per_client = summarize(runs)["per_client_selected"]
    assert set(per_client["node"]) == set(NODES)
    assert set(per_client["protocol"]) == {"group"}
    row = per_client[(per_client["strategy"] == "fedavg") & (per_client["node"] == "node_a")
                     & (per_client["metric"] == "selected_test_accuracy")].iloc[0]
    assert row["mean"] == pytest.approx(np.mean([0.85, 0.86, 0.87]))
    assert int(row["n_seeds"]) == 3


def test_main_and_export_end_to_end(mixed_results, tmp_path):
    analysis = str(tmp_path / "analysis")
    assert main(["--results_dir", mixed_results, "--output_dir", analysis]) == 0
    assert os.path.isfile(os.path.join(analysis, "per_client_selected.csv"))
    assert os.path.isfile(os.path.join(analysis, "final_metrics_non_iid_label.png"))

    tables = str(tmp_path / "tables")
    written = {os.path.basename(p) for p in export(analysis, tables, warn=False)}
    assert "summary_selected_test_accuracy.tex" in written
    assert "summary_v1_final_round_accuracy.tex" in written
    assert "time.tex" in written
    with open(os.path.join(tables, "time.tex"), encoding="utf-8") as fh:
        time_text = fh.read()
    assert "Test acc." in time_text and "86.00" in time_text      # FedAvg selected-round mean
    assert "excluding the report-only test evaluation" in time_text
    with open(os.path.join(tables, "full_metrics_non_iid_label.tex"), encoding="utf-8") as fh:
        full = fh.read()
    assert "0.8600" in full                                         # FL row = selected test
    assert "(image-level)" not in full                              # v1 FL rows have no test
    with open(os.path.join(tables, "pairwise_tests.tex"), encoding="utf-8") as fh:
        pw = fh.read()
    assert "(group-level)" in pw and "(image-level, v1)" in pw
    summary = pd.read_csv(os.path.join(analysis, "summary_table.csv"))
    assert "best_accuracy" not in set(summary["metric"])


def test_bar_chart_uses_selected_round_test_metrics(mixed_results, tmp_path):
    runs = collect_runs(mixed_results, warn=False)
    out = str(tmp_path / "figs")
    make_plots(runs, out, metric="accuracy")
    assert os.path.isfile(os.path.join(out, "final_metrics_non_iid_label.png"))
