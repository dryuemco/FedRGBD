"""Timing comparison MAXN / heterogeneous (docs/CROSS_CONFIG_COMPARISON.md (c)) --
synthetic results.json only."""

import json
import math
import os

import numpy as np

from scripts import cross_config_timing as cct

NODES = cct.NODES


def _write_run(run_dir, round_times, fit_walls, total=None):
    """round_times: {round: T_round}; fit_walls: {round: {node: fit_wall_s}}."""
    os.makedirs(run_dir)
    rounds = [{"round": r, "timing": {"round_time_s": t},
               "fit": {"clients": {n: {"fit_wall_s": fit_walls[r][n]} for n in NODES}}}
              for r, t in round_times.items()]
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"num_rounds": len(round_times), "rounds": rounds,
                   "total_time_s": total if total is not None else sum(round_times.values())}, f)


def _tree(tmp_path, scale=0.7, seeds=(42, 123, 456, 789, 1011)):
    """Every cell of both families; MAXN = scale x heterogeneous, seed-dependent cold start.
    Heterogeneous straggler node_c, MAXN straggler node_b."""
    root = str(tmp_path / "results")
    het_walls = {"node_a": 900.0, "node_b": 890.0, "node_c": 1340.0}
    max_walls = {"node_a": 850.0, "node_b": 905.0, "node_c": 870.0}
    for fam in cct.FAMILIES:
        for tag, _ in fam["partitions"]:
            for strategy, _ in fam["strategies"]:
                for seed in fam["seeds"]:
                    if seed not in seeds:
                        continue
                    R = fam["reference_rounds"]
                    het = {r: 1000.0 + r for r in range(1, R + 1)}
                    het[1] = 1100.0 + seed % 7
                    _write_run(cct.heterogeneous_dir(root, tag, strategy, seed, R), het,
                               {r: het_walls for r in het})
                    mx = {r: scale * (1000.0 + min(r, 3)) for r in range(1, 11)}
                    mx[1] = scale * (1100.0 + seed % 7) * (1.1 if seed == 42 else 1.0)
                    path = cct.maxn_dir(root, tag, strategy, seed)
                    if not os.path.isdir(path):
                        _write_run(path, mx, {r: max_walls for r in mx})
    return root


def test_ratios_per_family_and_quantity(tmp_path):
    runs, ratios, strag = cct.compare(_tree(tmp_path), B=500)
    got = {(r.family, r.partition, r.strategy, r.quantity) for r in ratios.itertuples()}
    assert len(got) == 4 * 3 + 2 * 2
    assert set(ratios[ratios.family == "rounds_1_3"].quantity) == {"primary", "secondary", "round1"}
    assert set(ratios[ratios.family == "ten_rounds"].quantity) == {"steady", "round1"}
    prim = ratios[(ratios.family == "rounds_1_3") & (ratios.quantity == "primary")]
    assert np.allclose(prim.ratio, 0.7)                       # median of rounds 2-3 in both
    assert (prim.n_pairs == 5).all() and (prim.max_distinct_resamples == 126).all()
    ten = ratios[ratios.family == "ten_rounds"]
    assert (ten.n_pairs == 3).all() and (ten.max_distinct_resamples == 10).all()


def test_steady_state_is_a_median_never_a_mean_or_total_over_r(tmp_path):
    root = str(tmp_path / "results")
    rt = {1: 1500.0, 2: 900.0, 3: 910.0, 4: 2000.0, 5: 905.0}
    _write_run(os.path.join(root, "x"), rt, {r: dict.fromkeys(NODES, 1.0) for r in rt})
    rounds = cct.load_rounds(os.path.join(root, "x"))
    assert cct.round_time(rounds, (1,)) == 1500.0
    assert cct.round_time(rounds, (2, 3, 4, 5)) == 907.5
    assert cct.round_time(rounds, (2, 3, 4, 5)) != sum(rt.values()) / len(rt)


def test_geometric_mean_reciprocal_and_per_seed_ratios(tmp_path):
    r = [0.6, 0.7, 0.8]
    a = cct.geometric_ratio(r, "k", B=200)
    b = cct.geometric_ratio([1 / x for x in r], "k", B=200)
    assert math.isclose(a["ratio"] * b["ratio"], 1.0)
    assert math.isclose(a["ratio"], (0.6 * 0.7 * 0.8) ** (1 / 3))
    assert a["ci_low"] <= a["ratio"] <= a["ci_high"]
    assert cct.geometric_ratio(r, "k", B=200) == a           # keyed generator: reproducible
    _, ratios, _ = cct.compare(_tree(tmp_path), B=200)
    row = ratios[(ratios.family == "rounds_1_3") & (ratios.partition == "iid")
                 & (ratios.strategy == "FedAvg") & (ratios.quantity == "round1")].iloc[0]
    per_seed = [float(x) for x in row.per_seed_ratios.split()]
    assert len(per_seed) == 5 and math.isclose(per_seed[0], 0.77, rel_tol=1e-3)  # seed 42


def test_stragglers_counted_per_configuration_and_round_set(tmp_path):
    _, _, strag = cct.compare(_tree(tmp_path), B=100)
    het = strag[strag.power_config == "heterogeneous"]
    mx = strag[strag.power_config == "maxn"]
    assert (het.straggler_node_c == het.n_rounds_counted).all()
    assert (mx.straggler_node_b == mx.n_rounds_counted).all()
    cell = strag[(strag.family == "rounds_1_3") & (strag.partition == "iid")
                 & (strag.strategy == "FedAvg")]
    assert set(zip(cell.power_config, cell.rounds)) == {
        ("heterogeneous", "1"), ("heterogeneous", "2-3"),
        ("maxn", "1"), ("maxn", "2-3"), ("maxn", "2-10")}
    assert set(cell[cell.rounds == "2-10"].n_rounds_counted) == {45}


def test_missing_pairs_are_listed_not_invented(tmp_path):
    _, ratios, _ = cct.compare(_tree(tmp_path, seeds=(42, 123)), B=100)
    r13 = ratios[ratios.family == "rounds_1_3"]
    assert (r13.n_pairs == 2).all() and set(r13.paired_seeds) == {"42 123"}
    empty = cct.compare(str(tmp_path / "nothing"), B=100)[1]
    assert (empty.n_pairs == 0).all() and len(empty) == 16


def test_markdown_has_no_verdict_words(tmp_path):
    _, ratios, strag = cct.compare(_tree(tmp_path), B=100)
    md = cct.to_markdown(ratios, strag).lower()
    assert "indicative" in md
    for word in ("equivalent", "no difference", "no detectable", "significant", "faster", "slower"):
        assert word not in md
