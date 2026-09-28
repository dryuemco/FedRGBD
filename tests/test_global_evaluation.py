"""Global evaluation (docs/GLOBAL_EVALUATION.md): bootstrap core, table assembly and the
reproduction gates of the predict stage -- on synthetic data only."""

import os

import numpy as np
import pytest

from scripts import global_evaluation as ge
from src.evaluation.bootstrap import (Unit, check_same_set, config_ci, seed_paired_diff_ci,
                                      shared_set_ci)
from src.evaluation.predictions import load_npz, metrics_from_predictions, pack

NODES = ge.NODES


def _node_data(node, n_groups=4, per_group=12, seed=0):
    """Paths, labels and group ids of one node's test split; groups never span nodes,
    one pure no-fire, one pure fire, the rest mixed."""
    rng = np.random.default_rng(seed + NODES.index(node))
    paths, labels, groups = [], [], []
    for g in range(n_groups):
        for i in range(per_group):
            y = 0 if g == 0 else 1 if g == 1 else int(rng.random() < 0.6)
            paths.append("%s/%s_g%d_%02d.jpg" % ("Fire" if y else "No_Fire", node, g, i))
            labels.append(y)
            groups.append("%s_g%d" % (node, g))
    return paths, np.array(labels), np.array(groups)


def _logits(labels, skill, rng):
    """Two-class logits whose margin is right with probability ~skill."""
    correct = rng.random(len(labels)) < skill
    sign = np.where(labels == 1, 1.0, -1.0) * np.where(correct, 1.0, -1.0)
    margin = sign * (0.5 + rng.random(len(labels)))
    return np.stack([np.zeros_like(margin), margin], axis=1).astype(np.float32)


def _unit(labels, margin, groups):
    return Unit(labels, margin, groups)


# ------------------------------------------------------------------ bootstrap core
def _shared_units(n_models, skills, seed=0):
    data = [_node_data(n) for n in NODES]
    labels = np.concatenate([d[1] for d in data])
    groups = np.concatenate([d[2] for d in data])
    rng = np.random.default_rng(seed)
    return [_unit(labels, _logits(labels, s, rng)[:, 1], groups) for s in skills[:n_models]]


def test_shared_set_ci_matches_config_ci_for_one_model_per_run():
    runs = [[u] for u in _shared_units(3, [0.8, 0.85, 0.9])]
    ours = shared_set_ci(runs, "k", B=500)
    ref = config_ci(runs, "pooled", "k", B=500)["balanced_accuracy"]
    assert ours["ci_low"] == pytest.approx(ref["ci_low"])
    assert ours["ci_high"] == pytest.approx(ref["ci_high"])


def test_shared_set_ci_point_is_mean_over_models_and_runs():
    units = _shared_units(4, [0.6, 0.7, 0.8, 0.9])
    runs = [units[:2], units[2:]]
    ba = [metrics_from_predictions(u.label, u.margin)["balanced_accuracy"] for u in units]
    assert shared_set_ci(runs, "k", B=200)["mean"] == pytest.approx(np.mean(ba))


def test_seed_paired_diff_identical_sides_is_exactly_zero():
    units = _shared_units(3, [0.7, 0.8, 0.9])
    runs = [[u] for u in units]
    res = seed_paired_diff_ci(runs, runs, "k", B=300)
    assert res["diff"] == 0 and res["ci_low"] == 0 and res["ci_high"] == 0
    assert res["p_boot"] == 1.0


def test_seed_paired_diff_detects_a_uniformly_better_side():
    good = _shared_units(3, [0.95, 0.95, 0.95], seed=1)
    bad = _shared_units(3, [0.55, 0.55, 0.55], seed=2)
    res = seed_paired_diff_ci([[u] for u in good], [[u] for u in bad], "k", B=2000)
    assert res["diff"] > 0.3 and res["ci_low"] > 0
    assert res["p_boot"] == pytest.approx(2 / 2001)
    flipped = seed_paired_diff_ci([[u] for u in bad], [[u] for u in good], "k", B=2000)
    assert flipped["ci_high"] < 0


def test_seed_paired_diff_is_reproducible_and_keyed():
    a = [[u] for u in _shared_units(3, [0.8, 0.82, 0.84], seed=3)]
    b = [[u] for u in _shared_units(3, [0.8, 0.81, 0.79], seed=4)]
    r1 = seed_paired_diff_ci(a, b, "same", B=400)
    assert r1 == seed_paired_diff_ci(a, b, "same", B=400)
    assert r1 != seed_paired_diff_ci(a, b, "other", B=400)


def test_models_on_different_images_are_refused():
    units = _shared_units(2, [0.8, 0.8])
    other = Unit(units[0].label[:-1], units[0].margin[:-1],
                 np.concatenate([d[2] for d in [_node_data(n) for n in NODES]])[:-1])
    with pytest.raises(ValueError, match="same held-out images"):
        check_same_set([units[0], other])
    with pytest.raises(ValueError):
        seed_paired_diff_ci([[units[0]]], [[other]], "k", B=10)


# ------------------------------------------------------------------ table assembly
@pytest.fixture
def partition(tmp_path, monkeypatch):
    """A synthetic 'iid' partition: FedAvg seeds 1-3, FedProx 1-2, centralized 1,
    local-only 1-2 with global_* prediction files on disk."""
    data = {n: _node_data(n) for n in NODES}
    groups = {p: g for n in NODES for p, g in zip(data[n][0], data[n][2])}
    excluded = {data["node_a"][0][0], data["node_b"][0][5]}
    monkeypatch.setattr("scripts.analyze_results.partition_tables",
                        lambda partition: (groups, excluded))
    monkeypatch.setattr(ge, "FAMILIES", {"heterogeneous": dict(ge.FAMILIES["heterogeneous"],
                                                               partitions=("iid",))})
    union_paths = [tuple(sorted(data[n][0])) for n in NODES]
    rng = np.random.default_rng(7)

    def units_from(logits_by_node, clean):
        out = []
        for n in NODES:
            paths, labels, gid = data[n]
            keep = np.array([p not in excluded for p in paths]) if clean else np.ones(len(paths), bool)
            out.append({"label": labels[keep], "margin": logits_by_node[n][keep, 1],
                        "group": gid[keep]})
        return out

    base = dict(protocol="group", distribution="iid", power_config="heterogeneous",
                num_rounds=3, local_epochs=5.0, lr=0.001, n_nodes=3)
    runs = []

    def add(kind, strategy, seed, skill):
        lg = {n: _logits(data[n][1], skill, rng) for n in NODES}
        run_dir = tmp_path / ("%s_%s_%d" % (kind, strategy, seed))
        (run_dir / "predictions").mkdir(parents=True)
        rec = dict(base, kind=kind, strategy=strategy, seed=seed, run_dir=str(run_dir),
                   run_name=run_dir.name, pred_units=units_from(lg, False),
                   pred_units_clean=units_from(lg, True), pred_paths=union_paths)
        if kind == "local":
            rec.update(power_config="desktop_gpu", local_epochs=15.0, lr=None, strategy="local_only")
            rec["_models"] = {}
            for k, model_node in enumerate(NODES):
                lk = {n: _logits(data[n][1], skill - 0.1 * k, rng) for n in NODES}
                rec["_models"][model_node] = lk
                for n in NODES:
                    with open(ge.global_file(str(run_dir / "predictions"), model_node, n), "wb") as f:
                        f.write(pack(data[n][0], data[n][1], lk[n]))
        runs.append(rec)
        return rec

    for s in (1, 2, 3):
        add("fl", "fedavg", s, 0.9)
    for s in (1, 2):
        add("fl", "fedprox_0.01", s, 0.85)
    add("centralized", "centralized", 1, 0.92)
    for s in (1, 2):
        add("local", "local_only", s, 0.8)
    return runs, data, excluded


def _ba(units_or_parts):
    label = np.concatenate([u["label"] for u in units_or_parts])
    margin = np.concatenate([u["margin"] for u in units_or_parts])
    return metrics_from_predictions(label, margin)["balanced_accuracy"]


def test_analyze_tables(partition):
    runs, data, excluded = partition
    models, configs, comps = ge.analyze(runs, B=300)
    # every model on the union: FL 5 runs + centralized 1 (one model each), local 2 x 3
    assert len(models[models.subset == "full"]) == 5 + 1 + 6
    assert set(models[models.method == "Local-only"].model) == set(NODES)
    assert (models[models.subset == "full"].n_images == sum(len(data[n][0]) for n in NODES)).all()
    assert (models[models.subset == "clean"].n_images
            == sum(len(data[n][0]) for n in NODES) - len(excluded)).all()

    # an FL model's global value is its pooled figure
    fl1 = next(r for r in runs if r["kind"] == "fl" and r["strategy"] == "fedavg" and r["seed"] == 1)
    row = models[(models.method == "FedAvg") & (models.seed == 1) & (models.subset == "full")]
    assert float(row[ge.METRIC].iloc[0]) == pytest.approx(_ba(fl1["pred_units"]))

    # a local node model on the WHOLE union, not on its own split
    loc1 = next(r for r in runs if r["kind"] == "local" and r["seed"] == 1)
    lk = loc1["_models"]["node_b"]
    whole = [{"label": data[n][1], "margin": lk[n][:, 1]} for n in NODES]
    row = models[(models.method == "Local-only") & (models.seed == 1) & (models.model == "node_b")
                 & (models.subset == "full")]
    assert float(row[ge.METRIC].iloc[0]) == pytest.approx(_ba(whole))

    # pairing: FedAvg has seeds 1-3, local-only 1-2 -> the difference uses 1 and 2 only
    c = comps[(comps.strategy == "FedAvg") & (comps.subset == "full")].iloc[0]
    assert c.paired_seeds == "1 2" and c.n_pairs == 2
    fl_vals = models[(models.method == "FedAvg") & (models.subset == "full") & (models.seed < 3)]
    loc_vals = models[(models.method == "Local-only") & (models.subset == "full")]
    want = fl_vals[ge.METRIC].mean() - loc_vals.groupby("seed")[ge.METRIC].mean().mean()
    assert c["diff"] == pytest.approx(want)
    assert c.ci_low <= c["diff"] <= c.ci_high
    # per-configuration means use all of a configuration's seeds
    cfg = configs[(configs.method == "FedAvg") & (configs.subset == "full")].iloc[0]
    assert cfg.n_seeds == 3
    assert set(comps.subset) == {"full", "clean"} and len(comps) == 4


def test_analyze_refuses_a_local_model_on_a_different_union(partition):
    runs, data, _ = partition
    loc = next(r for r in runs if r["kind"] == "local")
    path = ge.global_file(os.path.join(loc["run_dir"], "predictions"), "node_a", "node_c")
    d = load_npz(path)
    with open(path, "wb") as f:                    # one image fewer on node_c
        f.write(pack(list(d["path"][:-1]), d["label"][:-1],
                     np.stack([np.zeros(len(d["path"]) - 1), d["logit_margin"][:-1]], axis=1)))
    with pytest.raises(ValueError, match="union differs"):
        ge.analyze(runs, B=50)


def test_analyze_needs_the_predict_stage(partition):
    runs, _, _ = partition
    loc = next(r for r in runs if r["kind"] == "local")
    os.remove(ge.global_file(os.path.join(loc["run_dir"], "predictions"), "node_b", "node_a"))
    with pytest.raises(SystemExit, match="predict stage"):
        ge.analyze(runs, B=50)


# ------------------------------------------------------------------ predict-stage gates
@pytest.fixture
def fake_local_run(tmp_path, monkeypatch):
    """A local-only run dir whose 'models' are fixed logit tables, so the gates can be
    exercised without torch inference."""
    import torch

    import scripts.predict_from_checkpoint as pfc
    import src.models.mobilenetv3_multimodal as mm

    data = {n: _node_data(n) for n in NODES}
    rng = np.random.default_rng(11)
    logits = {m: {n: _logits(data[n][1], 0.85, rng) for n in NODES} for m in NODES}
    run_dir = tmp_path / "rev_iid_local_seed42"
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True)
    results = {}
    for m in NODES:
        (run_dir / m).mkdir()
        (run_dir / m / "model_selected.pt").write_bytes(b"")
        with open(pred_dir / ("selected_%s_test.npz" % m), "wb") as f:
            f.write(pack(data[m][0], data[m][1], logits[m][m]))
        cross = {}
        for n in NODES:
            if n != m:
                met = metrics_from_predictions(data[n][1], logits[m][n][:, 1])
                cross[n] = {"test_metrics": {k: round(float(met[k]), 6)
                                             for k in ("accuracy", "balanced_accuracy")}}
        results[m] = {"selected_epoch": 3, "batch_size": 8, "cross_eval": cross,
                      "data_dir": os.path.join("data", "processed", "iid", m)}

    class Model:
        def __init__(self):
            self.node = None

        def load_state_dict(self, state):
            self.node = state

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(torch, "load", lambda path, **kw: os.path.basename(os.path.dirname(path)))
    monkeypatch.setattr(mm, "create_model", lambda **kw: Model())
    monkeypatch.setattr(pfc, "models_of", lambda rd: [
        (m, [results[m]["data_dir"]], str(run_dir / m / "model_selected.pt"), results[m])
        for m in NODES])
    monkeypatch.setattr(pfc, "predict", lambda model, data_dir, split, batch, device: pack(
        data[os.path.basename(data_dir)][0], data[os.path.basename(data_dir)][1],
        logits[model.node][os.path.basename(data_dir)]))
    return str(run_dir), results, logits


def test_predict_writes_all_nine_files_when_everything_reproduces(fake_local_run):
    run_dir, _, _ = fake_local_run
    assert ge.predict_run(run_dir, "cpu") == []
    pred = os.path.join(run_dir, "predictions")
    assert all(os.path.isfile(ge.global_file(pred, m, n)) for m in NODES for n in NODES)


def test_predict_stops_when_the_logged_cross_eval_is_not_reproduced(fake_local_run):
    run_dir, results, _ = fake_local_run
    results["node_b"]["cross_eval"]["node_c"]["test_metrics"]["balanced_accuracy"] += 1e-3
    problems = ge.predict_run(run_dir, "cpu")
    assert problems and "node_b model on node_c" in problems[0] and "cross_eval" in problems[0]
    pred = os.path.join(run_dir, "predictions")
    assert not any(os.path.isfile(ge.global_file(pred, "node_b", n)) for n in NODES)


def test_predict_stops_when_own_node_predictions_are_not_reproduced(fake_local_run):
    run_dir, _, logits = fake_local_run
    logits["node_a"]["node_a"][:, 1] *= -1                  # the model no longer matches
    problems = ge.predict_run(run_dir, "cpu")
    assert problems and "does not reproduce selected_node_a_test.npz" in problems[0]


# ------------------------------------------------------------------ interpretation rule
def test_verdict_from_ci_is_the_declared_rule():
    assert ge.verdict_from_ci(0.001, 0.05) == ge.IMPROVES
    assert ge.verdict_from_ci(-0.01, 0.05) == ge.NO_DIFFERENCE
    assert ge.verdict_from_ci(0.0, 0.05) == ge.NO_DIFFERENCE      # lower bound must be > 0
    assert ge.verdict_from_ci(-0.05, -0.001) == ge.WORSE
    assert ge.verdict_from_ci(-0.05, 0.0) == ge.NO_DIFFERENCE


def test_holm_matches_the_step_down_definition():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1, cumulative max
    assert ge.holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert ge.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert ge.holm([0.2]) == pytest.approx([0.2])


def test_apply_rule_corrects_within_family_and_subset_only():
    import pandas as pd

    rows = []
    for family, m in (("heterogeneous", 3), ("maxn", 2)):
        for subset in ("full", "clean"):
            for k in range(m):
                rows.append({"family": family, "subset": subset, "partition": "p%d" % k,
                             "strategy": "FedAvg", "diff": 0.02 if k else -0.02,
                             "ci_low": 0.001 if k else -0.03, "ci_high": 0.04 if k else -0.001,
                             "p_boot": 0.02})
    out = ge.apply_rule(pd.DataFrame(rows))
    het = out[(out.family == "heterogeneous") & (out.subset == "full")]
    assert (het.holm_m == 3).all() and het.p_holm.tolist() == pytest.approx([0.06] * 3)
    assert (het.verdict_holm == ge.NO_DIFFERENCE).all()          # 0.06 >= 0.05
    assert het.verdict.tolist() == [ge.WORSE, ge.IMPROVES, ge.IMPROVES]
    mx = out[(out.family == "maxn") & (out.subset == "full")]
    assert (mx.holm_m == 2).all() and mx.p_holm.tolist() == pytest.approx([0.04, 0.04])
    assert mx.verdict_holm.tolist() == [ge.WORSE, ge.IMPROVES]


def test_markdown_lists_every_comparison(partition):
    runs, _, _ = partition
    _, _, comps = ge.analyze(runs, B=200)
    md = ge.comparisons_markdown(ge.apply_rule(comps))
    assert md.count("| iid | ") == len(comps) == 4
    assert "Holm m = 2" in md
