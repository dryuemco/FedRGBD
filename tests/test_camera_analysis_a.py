"""Camera experiment, question (a): the scene-paired bootstrap and the declared analysis.

bootstrap.scene_paired_contrast_ci (different images of the same scenes) and
scripts/camera_analysis_a.py (families, Holm, the hierarchy contrast, exact phrases), on
synthetic out-of-fold prediction files -- no training.
"""

import csv
import json
import os

import numpy as np
import pytest

from scripts import camera_analysis_a as ca
from scripts.camera_loso import EXPERIMENT, oof_name, run_dir_name, target_nodes
from src.evaluation.bootstrap import Unit, scene_paired_contrast_ci, seed_paired_diff_ci

NODES = ("node_a", "node_b", "node_c")
SCENES = ["s%02d" % i for i in range(1, 9)]
SEEDS = (42, 123, 456)


def _unit(rng, n_per_scene=20, acc=0.8, scenes=SCENES):
    labels, margins, groups = [], [], []
    for s in scenes:
        y = np.r_[np.zeros(n_per_scene // 2, int), np.ones(n_per_scene - n_per_scene // 2, int)]
        correct = rng.random(len(y)) < acc
        pred = np.where(correct, y, 1 - y)
        labels.append(y)
        margins.append(np.where(pred == 1, 1.0, -1.0) * (0.5 + rng.random(len(y))))
        groups += [s] * len(y)
    return Unit(np.concatenate(labels), np.concatenate(margins), np.array(groups))


def _clone_other_images(u):
    """Same per-scene confusion counts on a different image order (different images)."""
    order = np.random.default_rng(9).permutation(len(u.label))
    return Unit(u.label[order], u.margin[order], u.gid[u.g][order])


# --------------------------------------------------------------------------- bootstrap
def test_identical_sides_give_zero_and_degenerate_interval():
    rng = np.random.default_rng(0)
    runs = [[_unit(rng, acc=a)] for a in (0.6, 0.75, 0.9)]
    res = scene_paired_contrast_ci([(1, runs), (-1, runs)], "t", B=500)
    assert res["diff"] == 0 and res["ci_low"] == 0 and res["ci_high"] == 0
    assert res["p_boot"] == 1.0 and res["n_pairs"] == 3 and res["n_clusters"] == len(SCENES)


def test_pairing_is_kept_across_seeds_and_scenes():
    # side A is side B on different images with the same per-scene counts, while the seeds
    # differ strongly: only a paired resample (seeds AND scenes) gives exactly zero
    rng = np.random.default_rng(1)
    b = [[_unit(rng, acc=a)] for a in (0.55, 0.7, 0.95)]
    a = [[_clone_other_images(r[0])] for r in b]
    res = scene_paired_contrast_ci([(1, a), (-1, b)], "pair", B=1000)
    assert res["diff"] == pytest.approx(0) and res["ci_low"] == pytest.approx(0)
    assert res["ci_high"] == pytest.approx(0)
    # the existing same-set function refuses these (different images): new function needed
    with pytest.raises(ValueError):
        seed_paired_diff_ci([[_unit(rng)]], [[_unit(np.random.default_rng(5), n_per_scene=10)]],
                            "x", B=10)


def test_detects_a_real_shift_and_is_reproducible():
    rng = np.random.default_rng(2)
    good = [[_unit(rng, acc=0.9)] for _ in SEEDS]
    bad = [[_unit(rng, acc=0.6)] for _ in SEEDS]
    r1 = scene_paired_contrast_ci([(1, bad), (-1, good)], "shift", B=2000)
    r2 = scene_paired_contrast_ci([(1, bad), (-1, good)], "shift", B=2000)
    assert r1 == r2
    assert r1["ci_high"] < 0 and r1["diff"] < 0 and r1["p_boot"] < 0.01
    r3 = scene_paired_contrast_ci([(1, bad), (-1, good)], "other key", B=2000)
    assert r3["ci_low"] != r1["ci_low"] or r3["ci_high"] != r1["ci_high"]


def test_contrast_point_is_the_linear_combination_of_the_differences():
    rng = np.random.default_rng(3)
    u = {(x, y): [[_unit(rng, acc=0.6 + 0.3 * rng.random())] for _ in SEEDS]
         for x in NODES for y in NODES}

    def d(x, y):
        return scene_paired_contrast_ci([(1, u[(x, y)]), (-1, u[(x, x)])], "d", B=200)["diff"]

    terms = []
    for x, y in ca.CROSS:
        terms += [(0.25, u[(x, y)]), (-0.25, u[(x, x)])]
    for x, y in ca.WITHIN:
        terms += [(-0.5, u[(x, y)]), (0.5, u[(x, x)])]
    k = scene_paired_contrast_ci(terms, "k", B=500)
    want = np.mean([d(x, y) for x, y in ca.CROSS]) - np.mean([d(x, y) for x, y in ca.WITHIN])
    assert k["diff"] == pytest.approx(want, abs=1e-12)
    assert k["ci_low"] <= k["ci_high"]


def test_scene_sets_and_seed_counts_must_match():
    rng = np.random.default_rng(4)
    a = [[_unit(rng)] for _ in SEEDS]
    with pytest.raises(ValueError, match="same clusters"):
        scene_paired_contrast_ci([(1, a), (-1, [[_unit(rng, scenes=SCENES[:-1])] for _ in SEEDS])],
                                 "x", B=10)
    with pytest.raises(ValueError, match="same number of runs"):
        scene_paired_contrast_ci([(1, a), (-1, a[:2])], "x", B=10)


# --------------------------------------------------------------------------- rule
def test_verdict_phrases_exact():
    assert ca.PAIR_POS == "higher on the target sensor"
    assert ca.PAIR_NEG == "lower on the target sensor"
    assert ca.NO_DIFFERENCE == "no detectable difference"
    assert ca.K_NEG == "the cross-technology shift costs more than the within-family shift"
    assert ca.K_POS == "the within-family shift costs more than the cross-technology shift"
    v = ca.verdict_from_ci
    assert v(0.01, 0.05, ca.PAIR_POS, ca.PAIR_NEG) == "higher on the target sensor"
    assert v(-0.05, -0.01, ca.PAIR_POS, ca.PAIR_NEG) == "lower on the target sensor"
    assert v(-0.05, 0.01, ca.PAIR_POS, ca.PAIR_NEG) == "no detectable difference"
    assert v(0.0, 0.01, ca.PAIR_POS, ca.PAIR_NEG) == "no detectable difference"
    # K < 0 (the cross-technology pairs lose more) -> the cross-technology phrase
    assert v(-0.05, -0.01, ca.K_POS, ca.K_NEG) == ca.K_NEG
    assert v(0.01, 0.05, ca.K_POS, ca.K_NEG) == ca.K_POS


def test_holm_family_rule():
    rows = [{"diff": d, "ci_low": lo, "ci_high": hi, "p_boot": p} for d, lo, hi, p in (
        (-0.10, -0.15, -0.05, 0.001), (-0.04, -0.08, -0.001, 0.02), (0.02, -0.01, 0.05, 0.2),
        (0.05, 0.01, 0.09, 0.009), (0.00, -0.02, 0.02, 0.9), (-0.01, -0.03, 0.01, 0.5))]
    out = ca.apply_family_rule(rows)
    assert [r["holm_m"] for r in out] == [6] * 6
    assert [r["p_holm"] for r in out] == pytest.approx([0.006, 0.08, 0.6, 0.045, 1.0, 1.0])
    assert [r["verdict_holm"] for r in out] == [
        "lower on the target sensor", "no detectable difference", "no detectable difference",
        "higher on the target sensor", "no detectable difference", "no detectable difference"]
    # the interval verdict is reported alongside and may differ from the Holm headline
    assert out[1]["verdict"] == "lower on the target sensor"


# --------------------------------------------------------------------------- end to end
def _write_fixture(root):
    """labels.csv + oof files for rgb, rgb_d, ir (loso) and rgb (random), 3 seeds.

    Cross-technology targets (ZED <-> RealSense) are made much worse than within-family."""
    labels = os.path.join(root, "labels.csv")
    ids = ["%s_%s_d100_%04d" % (s, lab, i) for s in SCENES for lab in ("fire", "no_fire")
           for i in range(15)]
    with open(labels, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "node", "scene", "label", "distance_m", "capture_id", "frame_index",
                    "valid", "exclusion_reason"])
        for n in NODES:
            for fid in sorted(ids):
                s, lab = fid.split("_")[0], ("no_fire" if "_no_fire_" in fid else "fire")
                w.writerow([fid, n, s, lab, "1.000", fid[:-5], int(fid[-4:]), 1, ""])
    rng = np.random.default_rng(7)
    for modality in ("rgb", "rgb_d", "ir"):
        for x in target_nodes(modality):
            for seed in SEEDS:
                d = os.path.join(root, "runs", run_dir_name(modality, x, seed))
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "results.json"), "w") as f:
                    json.dump({"experiment": EXPERIMENT,
                               "config": {"modality": modality, "source": x, "seed": seed}}, f)
                protos = ("loso", "random") if modality == "rgb" else ("loso",)
                for proto in protos:
                    for y in target_nodes(modality):
                        cross = (x == "node_c") != (y == "node_c")
                        acc = 0.95 if x == y else (0.6 if cross else 0.9)
                        if proto == "random":
                            acc = min(0.99, acc + 0.05)
                        fid = np.array(sorted(ids))
                        lab = np.array([0 if "_no_fire_" in i else 1 for i in fid])
                        ok = rng.random(len(fid)) < acc
                        m = np.where(np.where(ok, lab, 1 - lab) == 1, 1.0, -1.0) \
                            * (0.5 + rng.random(len(fid)))
                        np.savez_compressed(os.path.join(d, oof_name(x, y, proto)), path=fid,
                                            label=lab.astype(np.uint8),
                                            logit_margin=m.astype(np.float32),
                                            p_fire=(1 / (1 + np.exp(-m))).astype(np.float32),
                                            format_version=np.array(1))
    return labels


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("cama"))
    labels = _write_fixture(root)
    out = os.path.join(root, "analysis")
    assert ca.main(["--results_root", os.path.join(root, "runs"), "--labels_csv", labels,
                    "--output_dir", out, "--B", "1000"]) == 0
    return out


def test_outputs_families_and_contrast(analysis):
    import pandas as pd
    pairs = pd.read_csv(os.path.join(analysis, "pairs.csv"))
    for fam in ("A-RGB", "A-RGBD"):
        g = pairs[pairs.family == fam]
        assert len(g) == 6 and set(g.holm_m) == {6}
        assert set(g.seeds) == {"42 123 456"}
        cross = g[(g.source == "node_c") | (g.target == "node_c")]
        assert set(cross.verdict_holm) == {"lower on the target sensor"}
        assert (cross["diff"] < -0.2).all()
    ir = pairs[pairs.family == "IR"]
    assert len(ir) == 2 and set(ir.verdict) == {"descriptive (no verdict)"}
    k = pd.read_csv(os.path.join(analysis, "hierarchy_contrast.csv")).iloc[0]
    assert k.K < 0 and k.ci_high < 0
    assert k.verdict == "the cross-technology shift costs more than the within-family shift"
    m = pd.read_csv(os.path.join(analysis, "ba_matrix.csv"))
    assert len(m[(m.protocol == "loso") & (m.modality == "rgb")]) == 9
    assert len(m[(m.protocol == "loso") & (m.modality == "ir")]) == 4
    assert len(m[(m.protocol == "random") & (m.modality == "rgb")]) == 9
    assert ((m.ci_low <= m.balanced_accuracy) & (m.balanced_accuracy <= m.ci_high)).all()


def test_markdown_wording(analysis):
    with open(os.path.join(analysis, "camera_a.md"), encoding="utf-8") as f:
        text = f.read()
    assert "equivalent" not in text.lower()
    assert "Holm m = 6" in text and "Contrast A-hierarchy" in text
    assert "random frame-level split" in text


def test_foreign_frame_ids_are_rejected(tmp_path):
    labels = _write_fixture(str(tmp_path))
    runs = ca.collect(str(tmp_path / "runs"))
    path = os.path.join(runs[("rgb", "node_a", 42)], oof_name("node_a", "node_b"))
    d = dict(np.load(path))
    d["path"] = d["path"].copy()
    d["path"][0] = "s99_fire_d100_0000"
    np.savez_compressed(path, **d)
    with pytest.raises(ValueError, match="kept frames"):
        ca.load_units(runs, labels)


def test_loso_table_is_generated_from_the_analysis(analysis, tmp_path):
    import pandas as pd
    from scripts import camera_export_loso_table as cet

    text = cet.tabular(analysis)
    rows = [l for l in text.splitlines() if r"$\rightarrow$" in l and "&" in l]
    assert [l.split(" & ")[0] for l in rows] == [
        r"D435if $\rightarrow$ D435i", r"D435if $\rightarrow$ ZED 2i",
        r"D435i $\rightarrow$ D435if", r"D435i $\rightarrow$ ZED 2i",
        r"ZED 2i $\rightarrow$ D435if", r"ZED 2i $\rightarrow$ D435i"]
    pairs = pd.read_csv(os.path.join(analysis, "pairs.csv"))
    r = pairs[(pairs["family"] == "A-RGB") & (pairs["source"] == "node_a")
              & (pairs["target"] == "node_c")].iloc[0]
    cells = rows[1].split(" & ")
    assert cells[3].startswith("%+.1f [" % (100 * r["diff"]))
    assert cells[4] == "%.3f" % r["p_holm"] and cells[5].startswith(r["verdict_holm"])
    assert "Contrast $K$" in text and text.startswith(cet.GENERATED)
    assert cet.SYNTHETIC not in text and cet.SYNTHETIC in cet.tabular(analysis, synthetic=True)
    with pytest.raises(SystemExit, match="never written under paper"):
        cet.main(["--analysis_dir", analysis, "--synthetic", "--output",
                  os.path.join(cet.REPO, "paper", "tables", "x.tex")])
    out = str(tmp_path / "t.tex")
    assert cet.main(["--analysis_dir", analysis, "--output", out]) == 0
