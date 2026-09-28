"""Cross-configuration comparison (docs/CROSS_CONFIG_COMPARISON.md) and the
within-configuration determinism gate of scripts/run_matrix.py -- synthetic data only."""

import json
import os

import numpy as np
import pytest

from scripts import cross_config_comparison as ccc
from scripts import run_matrix

NODES = ccc.NODES


def _node_data(node, n_groups=4, per_group=10):
    rng = np.random.default_rng(NODES.index(node))
    paths, labels, groups = [], [], []
    for g in range(n_groups):
        for i in range(per_group):
            y = 0 if g == 0 else 1 if g == 1 else int(rng.random() < 0.6)
            paths.append("%s/%s_g%d_%02d.jpg" % ("Fire" if y else "No_Fire", node, g, i))
            labels.append(y)
            groups.append("%s_g%d" % (node, g))
    return np.array(paths), np.array(labels, dtype=np.uint8), groups


DATA = {n: _node_data(n) for n in NODES}
GROUPS = {p: g for n in NODES for p, g in zip(DATA[n][0], DATA[n][2])}


def _write_run(run_dir, val_losses, skill_by_round, seed):
    """A fake FL run: results.json with val_loss_by_round, per-round test predictions."""
    os.makedirs(os.path.join(run_dir, "predictions"))
    rounds = [{"round": r, "fit": {"clients": {n: {"train_loss": 0.1 * r} for n in NODES}}}
              for r in val_losses]
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"rounds": rounds, "model_selection": {
            "val_loss_by_round": {str(r): v for r, v in val_losses.items()},
            "selected_round": min(val_losses, key=val_losses.get)}}, f)
    rng = np.random.default_rng(seed)
    for r, skill in skill_by_round.items():
        for n in NODES:
            paths, labels, _ = DATA[n]
            right = rng.random(len(labels)) < skill
            margin = np.where(labels == 1, 1.0, -1.0) * np.where(right, 1.0, -1.0)
            np.savez(os.path.join(run_dir, "predictions", "r%03d_%s_test.npz" % (r, n)),
                     path=paths, label=labels, logit_margin=margin.astype(np.float32),
                     p_fire=(1 / (1 + np.exp(-margin))).astype(np.float32),
                     format_version=np.array(1))


def _tree(tmp_path, maxn_skill, het_skill, seeds=(42, 123, 456), cells=None):
    root = str(tmp_path / "results")
    cells = cells or [(t, s) for t, _ in ccc.PARTITIONS for s, _ in ccc.STRATEGIES]
    for tag, strategy in cells:
        for seed in seeds:
            # MAXN: rounds 1-3 chosen by val loss (round 2 lowest among them); round 7
            # has an even lower loss and a perfect model -- it must be ignored
            vl = {r: 1.0 for r in range(1, 11)}
            vl[2], vl[7] = 0.5, 0.1
            skills = {r: maxn_skill for r in range(1, 11)}
            skills[7] = 1.0
            _write_run(ccc.maxn_dir(root, tag, strategy, seed), vl, skills, seed)
            _write_run(ccc.heterogeneous_dir(root, tag, strategy, seed),
                       {1: 0.9, 2: 0.4, 3: 0.8}, {1: het_skill, 2: het_skill, 3: het_skill},
                       seed + 1)
    return root


def test_selection_is_restricted_to_rounds_1_to_3(tmp_path):
    root = _tree(tmp_path, 0.8, 0.8, seeds=(42,), cells=[("iid", "fedavg")])
    assert ccc.selected_round_1_3(ccc.maxn_dir(root, "iid", "fedavg", 42)) == 2
    runs, comps = ccc.compare(root, lambda p: GROUPS, B=200)
    assert set(runs.selected_round) == {2}
    assert (runs[runs.power_config == "maxn"].balanced_accuracy < 0.99).all()  # not round 7


def test_ties_go_to_the_earlier_round(tmp_path):
    d = str(tmp_path / "r")
    _write_run(d, {1: 0.5, 2: 0.5, 3: 0.7}, {1: 0.8}, 0)
    assert ccc.selected_round_1_3(d) == 1


def test_difference_sign_verdicts_and_holm(tmp_path):
    root = _tree(tmp_path, maxn_skill=0.95, het_skill=0.6)
    runs, comps = ccc.compare(root, lambda p: GROUPS, B=500)
    first = comps[comps.family == "rounds_1_3"]
    assert len(first) == 4 and (first.n_pairs == 3).all()
    assert (first["diff"] > 0).all()
    assert set(first.verdict) <= {ccc.HIGHER, ccc.NO_DIFFERENCE}
    assert (first.holm_m == 4).all()
    assert (first.p_holm >= first.p_boot - 1e-12).all()
    ten = comps[comps.family == "ten_rounds"]              # no heterogeneous r10 run here
    assert len(ten) == 2 and (ten.n_pairs == 0).all()
    md = ccc.comparisons_markdown(comps)
    assert "equivalent" not in md and "## rounds_1_3" in md and "## ten_rounds" in md


def test_ten_round_family_selects_over_all_rounds_with_its_own_holm(tmp_path):
    root = _tree(tmp_path, maxn_skill=0.7, het_skill=0.7, seeds=(42, 123, 456),
                 cells=[("noniid", "fedavg")])
    for strategy in ("fedavg", "fedbn"):
        for seed in (42, 123, 456):
            if strategy == "fedbn":           # MAXN FedBN runs of the same shape
                vl = {r: 1.0 for r in range(1, 11)}
                vl[2], vl[7] = 0.5, 0.1
                skills = {r: 0.7 for r in range(1, 11)}
                skills[7] = 1.0
                _write_run(ccc.maxn_dir(root, "noniid", strategy, seed), vl, skills, seed)
            hv = {r: 1.0 for r in range(1, 11)}
            hv[9] = 0.2
            _write_run(ccc.heterogeneous_dir(root, "noniid", strategy, seed, 10), hv,
                       {r: 0.6 for r in range(1, 11)}, seed + 7)
    runs, comps = ccc.compare(root, lambda p: GROUPS, B=300)
    ten_runs = runs[runs.family == "ten_rounds"]
    assert set(ten_runs[ten_runs.power_config == "maxn"].selected_round) == {7}
    assert set(ten_runs[ten_runs.power_config == "heterogeneous"].selected_round) == {9}
    ten = comps[comps.family == "ten_rounds"]
    assert list(ten.strategy) == ["FedAvg", "FedBN"] and (ten.n_pairs == 3).all()
    assert (ten.holm_m == 2).all() and (ten["diff"] > 0).all()
    first = comps[(comps.family == "rounds_1_3") & (comps.n_pairs > 0)]
    assert (first.holm_m == 1).all()                       # families never share a Holm
    assert set(runs[runs.family == "rounds_1_3"].selected_round) <= {1, 2, 3}


def test_missing_pairs_are_listed_not_invented(tmp_path):
    root = _tree(tmp_path, 0.8, 0.8, seeds=(42,), cells=[("iid", "fedavg")])
    runs, comps = ccc.compare(root, lambda p: GROUPS, B=200)
    got = {(r.partition, r.strategy): r.n_pairs for r in comps.itertuples()}
    assert got[("iid", "FedAvg")] == 1
    assert got[("non_iid_label", "FedProx(0.01)")] == 0
    assert comps.holm_m.iloc[0] == 1


def test_verdict_rules():
    assert ccc.verdict_from_ci(0.001, 0.02) == ccc.HIGHER
    assert ccc.verdict_from_ci(0.0, 0.02) == ccc.NO_DIFFERENCE
    assert ccc.verdict_from_ci(-0.02, -0.001) == ccc.LOWER
    assert ccc.verdict_from_holm(-0.01, 0.01) == ccc.LOWER
    assert ccc.verdict_from_holm(0.01, 0.2) == ccc.NO_DIFFERENCE
    assert ccc.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_fedbn_is_not_compared():
    # no heterogeneous 3-round FedBN run exists (docs/CROSS_CONFIG_COMPARISON.md)
    assert "fedbn" not in {s for s, _ in ccc.STRATEGIES}


# ------------------------------------------------------------------ determinism gate
def _smoke(run_dir, skill=0.8, seed=0, loss=0.1):
    _write_run(run_dir, {1: 0.9, 2: 0.4}, {1: skill, 2: skill}, seed)
    p = os.path.join(run_dir, "results.json")
    d = json.load(open(p))
    d["rounds"][0]["fit"]["clients"]["node_c"]["train_loss"] = loss
    json.dump(d, open(p, "w"))


def test_determinism_gate_passes_on_identical_runs(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _smoke(a)
    _smoke(b)
    ok, why = run_matrix.check_determinism_gate({"runs": [a, b]})
    assert ok, why
    assert "6 prediction files" in why


def test_determinism_gate_fails_on_a_single_changed_logit(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _smoke(a)
    _smoke(b)
    f = os.path.join(b, "predictions", "r002_node_b_test.npz")
    d = dict(np.load(f))
    d["logit_margin"] = d["logit_margin"].copy()
    d["logit_margin"][3] = np.nextafter(d["logit_margin"][3], np.float32(10))
    np.savez(f, **d)
    ok, why = run_matrix.check_determinism_gate({"runs": [a, b]})
    assert not ok and "r002_node_b_test.npz" in why


def test_determinism_gate_fails_on_train_loss_and_missing_runs(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _smoke(a, loss=0.1)
    _smoke(b, loss=0.1 + 1e-15)
    ok, why = run_matrix.check_determinism_gate({"runs": [a, b]})
    assert not ok and "train_loss round 1 node_c" in why
    ok, why = run_matrix.check_determinism_gate({"runs": [a, str(tmp_path / "nope")]})
    assert not ok and "results.json missing" in why
    ok, why = run_matrix.check_determinism_gate({"runs": [a]})
    assert not ok


def test_the_maxn_block_declares_the_determinism_gate_and_no_identity_gate():
    gate = run_matrix.block_determinism_gate(
        "maxn_long_horizon", os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                          "configs", "experiment_matrix.yaml"))
    assert gate == {"runs": ["results/diag_smoke_maxn", "results/diag_smoke_maxn_r2"]}
