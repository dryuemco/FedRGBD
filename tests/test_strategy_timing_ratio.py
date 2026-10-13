"""Wall-clock ratio FedProx(0.01) / FedAvg (docs/STRATEGY_TIMING_RATIO.md) --
synthetic results.json, plus the committed output against the real results/ tree."""

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from scripts import strategy_timing_ratio as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_run(run_dir, round_times, total):
    os.makedirs(run_dir)
    rounds = [{"round": r, "timing": {"round_time_s": t}} for r, t in round_times.items()]
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"num_rounds": len(round_times), "rounds": rounds,
                   "total_time_excl_test_s": total, "total_time_s": total + 50.0}, f)


def _tree(tmp_path, factor=1.5, seeds=st.SEEDS):
    """FedProx = factor x FedAvg in every round; FedAvg round 1 = 1.1 x its steady round."""
    root = str(tmp_path / "results")
    for config, R, dir_of in st.CONFIGS:
        for tag, _ in st.PARTITIONS:
            for seed in seeds:
                avg = {r: 1000.0 + seed % 5 + (100.0 if r == 1 else 0.0) for r in range(1, R + 1)}
                prox = {r: factor * t for r, t in avg.items()}
                _write_run(dir_of(root, tag, "fedavg", seed), avg, sum(avg.values()))
                _write_run(dir_of(root, tag, "fedprox_0.01", seed), prox, sum(prox.values()))
    return root


def test_every_cell_and_quantity(tmp_path):
    runs, ratios = st.compare(_tree(tmp_path), B=300)
    assert len(runs) == 2 * 2 * 5 * 2
    main = ratios[ratios.quantity != "fedavg_rounds_in_fedprox_budget"]
    assert len(main) == 2 * 2 * len(st.QUANTITIES)
    assert (main.n_pairs == 5).all() and (main.max_distinct_resamples == 126).all()
    for q in ("steady", "round1", "total"):
        assert np.allclose(main[main.quantity == q].ratio, 1.5)
    # FedProx round 1 against FedAvg's steady round: 1.5 x 1.1 x steady / steady
    cross = main[main.quantity == "prox_round1_vs_avg_steady"]
    assert np.allclose(cross.ratio, 1.5 * 1100.0 / 1000.0, rtol=1e-2)
    assert all(len(r.split()) == 5 for r in main.per_seed_ratios)


def test_budget_is_r_times_the_total_ratio(tmp_path):
    _, ratios = st.compare(_tree(tmp_path), B=300)
    for config, R, _ in st.CONFIGS:
        for _, part in st.PARTITIONS:
            sel = ratios[(ratios.power_config == config) & (ratios.partition == part)]
            tot = sel[sel.quantity == "total"].iloc[0]
            bud = sel[sel.quantity == "fedavg_rounds_in_fedprox_budget"].iloc[0]
            for col in ("ratio", "ci_low", "ci_high"):
                assert math.isclose(bud[col], R * tot[col])


def test_a_seed_enters_only_when_both_runs_exist(tmp_path):
    root = _tree(tmp_path)
    os.remove(os.path.join(st.CONFIGS[0][2](root, "iid", "fedavg", 789), "results.json"))
    _, ratios = st.compare(root, B=300)
    het_iid = ratios[(ratios.power_config == "heterogeneous") & (ratios.partition == "iid")
                     & (ratios.quantity == "steady")].iloc[0]
    assert het_iid.n_pairs == 4 and "789" not in het_iid.paired_seeds.split()


def test_missing_round_refuses(tmp_path):
    root = str(tmp_path / "results")
    _write_run(os.path.join(root, "x"), {1: 1.0, 3: 1.0}, 2.0)
    with pytest.raises(ValueError):
        st.run_times(os.path.join(root, "x"), 3)


@pytest.mark.skipif(not os.path.isfile(os.path.join(ROOT, "results", "rev_iid_fedavg_seed42",
                                                    "results.json")),
                    reason="needs the real results/ tree")
def test_committed_output_is_the_real_results(tmp_path):
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        assert st.main(["--output_dir", str(tmp_path)]) == 0
    finally:
        os.chdir(cwd)
    for name in ("runs.csv", "ratios.csv"):
        fresh = pd.read_csv(tmp_path / name)
        committed = pd.read_csv(os.path.join(ROOT, "analysis", "strategy_timing", name))
        pd.testing.assert_frame_equal(fresh, committed)
