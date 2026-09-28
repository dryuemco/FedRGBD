"""Camera experiment, question (b) (docs/CAMERA_EXPERIMENT_PREREG.md section 6): fold
materialisation, the camera_sensor_skew block and its analysis -- synthetic data only."""

import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from scripts import camera_analysis_b as cab  # noqa: E402
from scripts import camera_fl_prepare as prep  # noqa: E402
from scripts import print_revision_commands as prc  # noqa: E402
from scripts import run_matrix  # noqa: E402
from scripts.camera_labels import LABEL_COLUMNS  # noqa: E402
from scripts.camera_manifests import FL_FILE, render  # noqa: E402
from src.evaluation.predictions import pack  # noqa: E402

NODES = ("node_a", "node_b", "node_c")
SCENES = ["s%02d" % i for i in range(1, 8)]              # 7 scenes -> 5 folds, two of size 2
NATIVE = {"node_a": (64, 48), "node_b": (48, 64), "node_c": (100, 40)}
FRAMES = 3
INVALID = {("node_b", "s02_fire_d100_0001"), ("node_c", "s05_no_fire_d100_0002")}


# --------------------------------------------------------------------------- fixtures
def _label_rows():
    rows = []
    for node in NODES:
        for scene in SCENES:
            for label in ("fire", "no_fire"):
                for i in range(FRAMES):
                    fid = "%s_%s_d100_%04d" % (scene, label, i)
                    bad = (node, fid) in INVALID
                    rows.append({"id": fid, "node": node, "scene": scene, "label": label,
                                 "distance_m": "1.000", "capture_id": fid[:-5],
                                 "frame_index": i, "valid": 0 if bad else 1,
                                 "exclusion_reason": "constant" if bad else ""})
    return rows


def _write_inputs(root, rows=None, image_order=None):
    """labels.csv, fl_folds.csv (the pre-registered generator) and raw images."""
    rows = _label_rows() if rows is None else rows
    labels = os.path.join(root, "data", "raw", "camera", "labels.csv")
    os.makedirs(os.path.dirname(labels), exist_ok=True)
    with open(labels, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(LABEL_COLUMNS)
        for r in rows:
            w.writerow([r[c] for c in LABEL_COLUMNS])
    folds = os.path.join(root, "data", "splits_camera", FL_FILE)
    os.makedirs(os.path.dirname(folds), exist_ok=True)
    with open(folds, "wb") as f:
        f.write(render(SCENES)[FL_FILE])
    items = [(r["node"], r["id"]) for r in rows]
    for node, fid in (image_order(items) if image_order else items):
        path = os.path.join(root, "data", "raw", "camera", node, fid + "_rgb.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        seed = int(hashlib.md5((node + fid).encode()).hexdigest()[:8], 16)
        px = np.random.default_rng(seed).integers(0, 255, NATIVE[node][::-1] + (3,), dtype=np.uint8)
        Image.fromarray(px).save(path)
    return labels, folds


def _prepare(root, *extra):
    return prep.main(["--labels_csv", os.path.join(root, "data", "raw", "camera", "labels.csv"),
                      "--folds_csv", os.path.join(root, "data", "splits_camera", FL_FILE),
                      "--raw_dir", os.path.join(root, "data", "raw", "camera"),
                      "--output_root", os.path.join(root, "data", "processed"),
                      "--counts_csv", os.path.join(root, "data", "splits_camera",
                                                   "fl_materialised_manifest.csv")] + list(extra))


def _tree(root):
    """{relative path: sha256} of every file below root."""
    out = {}
    for dirpath, _d, files in os.walk(root):
        for name in files:
            p = os.path.join(dirpath, name)
            with open(p, "rb") as f:
                out[os.path.relpath(p, root).replace("\\", "/")] = hashlib.sha256(f.read()).hexdigest()
    return out


def _fold_of_scene():
    return {k: int(v) for k, v in (line.split(",") for line in
                                   render(SCENES)[FL_FILE].decode().splitlines()[1:])}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("cam"))
    _write_inputs(root)
    assert _prepare(root, "--clean") == 0
    return root


# --------------------------------------------------------------------------- materialisation
def test_splits_are_disjoint_by_scene_across_all_nodes(built):
    fos = _fold_of_scene()
    for f in range(5):
        fold_dir = os.path.join(built, "data", "processed", "camera_fold%d" % f)
        scenes = {s: set() for s in prep.SPLITS}
        for node in NODES:
            for split in prep.SPLITS:
                for cls in ("Fire", "No_Fire"):
                    d = os.path.join(fold_dir, node, split, cls)
                    scenes[split] |= {n[:3] for n in os.listdir(d)}
        assert not (scenes["train"] & scenes["val"]) and not (scenes["train"] & scenes["test"])
        assert not (scenes["val"] & scenes["test"])
        assert scenes["test"] == {s for s, k in fos.items() if k == f}
        assert scenes["val"] == {s for s, k in fos.items() if k == (f + 1) % 5}
        assert scenes["train"] == set(SCENES) - scenes["test"] - scenes["val"]


def test_each_scene_is_a_test_scene_in_exactly_one_fold(built):
    seen = []
    for f in range(5):
        for node in NODES:
            for cls in ("Fire", "No_Fire"):
                d = os.path.join(built, "data", "processed", "camera_fold%d" % f, node, "test", cls)
                seen += [(node, n) for n in os.listdir(d)]
    valid = [(r["node"], r["id"] + ".png") for r in _label_rows() if r["valid"]]
    assert sorted(seen) == sorted(valid)


def test_each_node_holds_only_its_own_sensor_frames_preprocessed(built):
    rows = _label_rows()
    for f in range(5):
        for node in NODES:
            node_dir = os.path.join(built, "data", "processed", "camera_fold%d" % f, node)
            got = sorted(n[:-4] for _d, _s, files in os.walk(node_dir) for n in files)
            want = sorted(r["id"] for r in rows if r["node"] == node and r["valid"])
            assert got == want                           # every valid own frame, nothing else
            assert not any((node, i) in INVALID for i in got)
        img = Image.open(os.path.join(built, "data", "processed", "camera_fold%d" % f, "node_c",
                                      "train", "Fire", os.listdir(os.path.join(
                                          built, "data", "processed", "camera_fold%d" % f,
                                          "node_c", "train", "Fire"))[0]))
        assert img.size == (224, 224) and img.format == "PNG"


def test_pixels_are_camera_preprocess_rgb_224(built):
    from src.data.camera_preprocess import rgb_224
    fos = _fold_of_scene()
    fid = "s03_fire_d100_0000"
    f = fos["s03"]
    src = Image.open(os.path.join(built, "data", "raw", "camera", "node_a", fid + "_rgb.png"))
    out = Image.open(os.path.join(built, "data", "processed", "camera_fold%d" % f, "node_a",
                                  "test", "Fire", fid + ".png"))
    assert np.array_equal(np.asarray(out), np.asarray(rgb_224(src)))


def test_counts_manifest_matches_disk(built):
    df = pd.read_csv(os.path.join(built, "data", "splits_camera", "fl_materialised_manifest.csv"))
    assert len(df) == 5 * 3 * 3 * 2
    for r in df.itertuples():
        d = os.path.join(built, "data", "processed", "camera_fold%d" % r.fold, r.node, r.split,
                         getattr(r, "_4"))
        assert len(os.listdir(d)) == r.n_images
    for f in range(5):
        with open(os.path.join(built, "data", "processed", "camera_fold%d" % f, "manifest.csv"),
                  "rb") as fh:
            md5 = hashlib.md5(fh.read()).hexdigest()
        assert set(df[df.fold == f].fold_manifest_md5) == {md5}
    assert df.n_images.sum() == 5 * sum(1 for r in _label_rows() if r["valid"])


def test_materialisation_is_independent_of_creation_order(built, tmp_path):
    other = str(tmp_path / "other")
    rows = _label_rows()
    random.Random(7).shuffle(rows)                              # labels.csv in another order
    _write_inputs(other, rows=rows, image_order=lambda items: list(reversed(items)))
    assert _prepare(other) == 0
    a = _tree(os.path.join(built, "data", "processed"))
    b = _tree(os.path.join(other, "data", "processed"))
    assert a == b
    assert _tree(os.path.join(built, "data", "splits_camera")) == \
        _tree(os.path.join(other, "data", "splits_camera"))


def test_verify_and_idempotence(built, tmp_path, capsys):
    root = str(tmp_path / "v")
    shutil.copytree(built, root)
    assert _prepare(root, "--verify") == 0
    before = _tree(os.path.join(root, "data", "processed"))
    assert _prepare(root) == 0                                        # idempotent
    assert _tree(os.path.join(root, "data", "processed")) == before
    stray = os.path.join(root, "data", "processed", "camera_fold0", "node_a", "test", "Fire", "x.png")
    Image.new("RGB", (224, 224)).save(stray)
    assert _prepare(root, "--verify") == 1
    with pytest.raises(SystemExit, match="--clean"):
        _prepare(root)
    assert _prepare(root, "--clean") == 0
    assert _prepare(root, "--verify") == 0
    victim = os.path.join(root, "data", "processed", "camera_fold2", "node_b", "train")
    shutil.rmtree(victim)
    assert _prepare(root, "--verify") == 1
    assert _prepare(root) == 0                                        # fills in what is missing
    assert _prepare(root, "--verify") == 0
    assert _tree(os.path.join(root, "data", "processed")) == before


def test_a_node_can_materialise_only_its_own_frames(built, tmp_path):
    root = str(tmp_path / "nb")
    _write_inputs(root)
    shutil.rmtree(os.path.join(root, "data", "raw", "camera", "node_a"))   # node_b has no A frames
    assert _prepare(root, "--nodes", "node_b") == 0
    assert _prepare(root, "--verify", "--nodes", "node_b") == 0
    for f in range(5):
        base = os.path.join(root, "data", "processed", "camera_fold%d" % f)
        assert sorted(os.listdir(base)) == ["manifest.csv", "node_b"]
        with open(os.path.join(base, "manifest.csv"), "rb") as a, \
                open(os.path.join(built, "data", "processed", "camera_fold%d" % f,
                                  "manifest.csv"), "rb") as b:
            assert a.read() == b.read()                         # same md5 on every node


def test_refuses_missing_inputs(tmp_path):
    root = str(tmp_path / "m")
    labels, folds = _write_inputs(root)
    os.remove(folds)
    with pytest.raises(SystemExit, match="fl_folds.csv is missing"):
        _prepare(root)
    _write_inputs(root)
    os.remove(labels)
    with pytest.raises(SystemExit, match="labels.csv is missing"):
        _prepare(root)


def test_refuses_folds_that_are_not_the_preregistered_ones(tmp_path):
    root = str(tmp_path / "f")
    _, folds = _write_inputs(root)
    lines = open(folds, encoding="utf-8").read().splitlines()
    scene, fold = lines[1].split(",")
    lines[1] = "%s,%d" % (scene, (int(fold) + 1) % 5)
    with open(folds, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with pytest.raises(SystemExit, match="pre-registered"):
        _prepare(root)


def test_refuses_a_scene_missing_a_class_on_a_node(tmp_path):
    root = str(tmp_path / "c")
    rows = _label_rows()
    for r in rows:
        if r["node"] == "node_c" and r["scene"] == "s04" and r["label"] == "fire":
            r["valid"] = 0
    _write_inputs(root, rows=rows)
    with pytest.raises(ValueError, match="s04"):
        _prepare(root)


# --------------------------------------------------------------------------- the block
def _revision():
    return prc.apply_all_seeds(prc.load_config(prc.DEFAULT_CONFIG)["revision"])


def test_block_is_45_maxn_runs_with_unambiguous_dirs():
    runs = prc.expand_all(_revision(), prc.CAMERA_BLOCK)[prc.CAMERA_BLOCK]
    assert len(runs) == 45
    assert {(r.strategy, r.split, r.seed) for r in runs} == {
        (s, "camera_fold%d" % f, seed) for s in ("fedavg", "fedprox_0.01", "fedbn")
        for f in range(5) for seed in (42, 123, 456)}
    for r in runs:
        assert r.kind == "fl" and r.power_config == "maxn"
        assert (r.rounds, r.local_epochs, r.lr, r.batch_size) == (10, 5, 0.001, 8)
        assert r.output_dir == "results/pc_maxn/rev_%s_%s_r10_seed%d" % (r.split, r.strategy, r.seed)
        for node, cmd in zip(NODES, r.client_commands()):
            assert "--data_dir data/processed/%s/%s " % (r.split, node) in cmd
    assert len({r.output_dir for r in runs}) == 45
    other = {r.output_dir for name in prc.BLOCK_ORDER
             for r in prc.expand_all(_revision(), name)[name]}
    assert not other & {r.output_dir for r in runs}


def test_block_is_outside_the_flame_matrix_and_its_gate():
    assert prc.CAMERA_BLOCK not in prc.BLOCK_ORDER
    assert prc.CAMERA_BASELINE_BLOCK not in prc.BLOCK_ORDER
    maxn = prc.expand_all(_revision(), "maxn_long_horizon")["maxn_long_horizon"]
    assert len(maxn) == 30 and not any("camera" in r.output_dir for r in maxn)
    assert run_matrix.block_determinism_gate(prc.CAMERA_BLOCK) is None
    assert run_matrix.block_determinism_gate("maxn_long_horizon") == {
        "runs": ["results/diag_smoke_maxn", "results/diag_smoke_maxn_r2"]}


def test_default_output_never_contains_the_camera_block(capsys):
    assert prc.main(["--all_seeds", "--format", "bash", "--no_skip_existing"]) == 0
    assert "camera" not in capsys.readouterr().out


def test_run_matrix_accepts_the_emitted_script(tmp_path, capsys):
    assert prc.main(["--all_seeds", "--block", prc.CAMERA_BLOCK, "--format", "bash",
                     "--no_skip_existing", "--power_config", "maxn"]) == 0
    path = tmp_path / "cam.sh"
    path.write_text(capsys.readouterr().out, encoding="utf-8")
    runs = run_matrix.parse_block(str(path))
    want = prc.expand_all(_revision(), prc.CAMERA_BLOCK)[prc.CAMERA_BLOCK]
    assert [r["out_dir"] for r in runs] == [r.output_dir for r in want]
    for got, run in zip(runs, want):
        assert got["server"] == run.server_command()
        assert got["clients"] == dict(zip(NODES, run.client_commands()))
        assert got["gates"] == []
    run_matrix.check_namespace(runs, "maxn")
    with pytest.raises(SystemExit):
        run_matrix.check_namespace(runs, "heterogeneous")
    assert run_matrix.block_splits(runs) == ["camera_fold%d" % f for f in range(5)]


def test_block_without_power_config_defaults_to_maxn_and_refuses_heterogeneous(capsys):
    assert prc.main(["--all_seeds", "--block", prc.CAMERA_BLOCK, "--format", "bash",
                     "--no_skip_existing"]) == 0
    out = capsys.readouterr().out
    assert "# power_config: maxn" in out
    assert out.count("--output_dir results/pc_maxn/rev_camera_fold") == 45
    with pytest.raises(SystemExit, match="declares power_config 'maxn'"):
        prc.main(["--all_seeds", "--block", prc.CAMERA_BLOCK, "--format", "bash",
                  "--power_config", "heterogeneous"])


def test_desktop_baselines(tmp_path, capsys):
    runs = prc.expand_all(_revision(), prc.CAMERA_BASELINE_BLOCK)[prc.CAMERA_BASELINE_BLOCK]
    assert len(runs) == 30
    assert {(r.baseline_type, r.split, r.seed) for r in runs} == {
        (t, "camera_fold%d" % f, s) for t in ("centralized", "local_only")
        for f in range(5) for s in (42, 123, 456)}
    for r in runs:
        assert r.kind == "baseline" and r.epochs == 50 and r.lr == 0.001 and r.batch_size == 8
        name = "centralized" if r.baseline_type == "centralized" else "local"
        assert r.output_dir == "results/rev_baselines_camera/rev_%s_%s_r10_seed%d" % (
            r.split, name, r.seed)
        assert r.post_commands == ["python3 scripts/predict_from_checkpoint.py " + r.output_dir]
    assert prc.main(["--all_seeds", "--block", prc.CAMERA_BASELINE_BLOCK, "--format", "bash",
                     "--no_skip_existing"]) == 0
    out = capsys.readouterr().out
    assert "DESKTOP GPU BASELINES" in out
    assert out.count("scripts/train_local.py --batch --cross_eval") == 15
    assert out.count("scripts/train_centralized.py") == 15
    path = tmp_path / "b.sh"
    path.write_text(out, encoding="utf-8")
    with pytest.raises(SystemExit, match="desktop GPU"):
        run_matrix.parse_block(str(path))


def test_every_other_block_is_byte_identical_to_the_committed_generator(tmp_path):
    """The committed print_revision_commands.py (before the camera block) and the current
    one print the same bytes for every existing block and for the default output."""
    try:
        old = subprocess.run(["git", "show", "HEAD:scripts/print_revision_commands.py"],
                             cwd=_REPO, capture_output=True, text=True, encoding="utf-8")
    except OSError:
        pytest.skip("git not available")
    if old.returncode != 0 or "CAMERA_BLOCK" in old.stdout:
        pytest.skip("no pre-camera generator in HEAD to compare with")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "print_revision_commands.py").write_text(old.stdout, encoding="utf-8")
    cfg = os.path.join(_REPO, "configs", "experiment_matrix.yaml")

    def out(script, *args):
        r = subprocess.run([sys.executable, str(script), "--config", cfg, "--no_skip_existing"]
                           + list(args), cwd=_REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout

    new = os.path.join(_REPO, "scripts", "print_revision_commands.py")
    variants = [("--format", "text"), ("--all_seeds", "--format", "bash")]
    variants += [("--all_seeds", "--block", b, "--format", "bash") for b in prc.BLOCK_ORDER]
    for args in variants:
        assert out(new, *args) == out(scripts / "print_revision_commands.py", *args), args


# --------------------------------------------------------------------------- analysis
def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _logits(labels, skill, rng):
    correct = rng.random(len(labels)) < skill
    sign = np.where(labels == 1, 1.0, -1.0) * np.where(correct, 1.0, -1.0)
    margin = sign * (0.5 + rng.random(len(labels)))
    return np.stack([np.zeros_like(margin), margin], axis=1).astype(np.float32)


def _test_split(node, fold, rows, fos):
    sel = [r for r in rows if r["node"] == node and r["valid"] and fos[r["scene"]] == fold]
    paths = ["%s/%s.png" % ("Fire" if r["label"] == "fire" else "No_Fire", r["id"]) for r in sel]
    labels = np.array([1 if r["label"] == "fire" else 0 for r in sel])
    return paths, labels


def _pred(pred_dir, name, node, fold, rows, fos, skill, rng, drop=0):
    paths, labels = _test_split(node, fold, rows, fos)
    if drop:
        paths, labels = paths[drop:], labels[drop:]
    os.makedirs(pred_dir, exist_ok=True)
    with open(os.path.join(pred_dir, name), "wb") as f:
        f.write(pack(["x/" + p for p in paths], labels, _logits(labels, skill, rng)))


def _fl_results(strategy, seed, val_losses):
    rounds = []
    for i, v in enumerate(val_losses):
        rounds.append({"round": i + 1, "evaluate": {
            "clients": {n: {"val_loss": v, "val_n_examples": 10, "test_accuracy": 0.5}
                        for n in NODES},
            "aggregate": {"val_loss": v, "test_accuracy": 0.5}}})
    return {"strategy": strategy, "seed": seed, "num_rounds": len(val_losses),
            "results_schema_version": 3, "model_payload_bytes": 1000, "rounds": rounds,
            "tags": ["camera"]}


def _baseline_json(experiment, seed, extra=None):
    d = {"experiment": experiment, "seed": seed, "epochs": 2, "selected_epoch": 1,
         "history": [{"epoch": 1, "val_loss": 0.4, "test_metrics": {"accuracy": 0.5}},
                     {"epoch": 2, "val_loss": 0.6, "test_metrics": {"accuracy": 0.5}}],
         "model_selection": {"val_loss_by_epoch": {"1": 0.4, "2": 0.6}}}
    d.update(extra or {})
    return d


SKILL = {"FedAvg": 0.95, "FedProx(0.01)": 0.93, "FedBN": 0.62, "Local-only": 0.6,
         "Centralized": 0.95}


def _camera_results(root, skill=SKILL, drop=None):
    """Every run of the block and its baselines, as the real runs lay them out."""
    rows = _label_rows()
    _write_inputs_light(root, rows)
    fos = _fold_of_scene()
    runs = cab.expected_runs(os.path.join(_REPO, "configs", "experiment_matrix.yaml"))
    rng = np.random.default_rng(0)
    for method, by in runs.items():
        for (fold, seed), rel in sorted(by.items()):
            d = os.path.join(root, rel)
            pred = os.path.join(d, "predictions")
            cut = (drop or {}).get((method, fold, seed), 0)
            if method in ("FedAvg", "FedProx(0.01)", "FedBN"):
                strategy = dict(cab.FL_METHODS)[method]
                _write_json(os.path.join(d, "results.json"), _fl_results(strategy, seed, [0.5, 0.3, 0.4]))
                for node in NODES:
                    # round 1 (not selected) is garbage: another fold's images
                    _pred(pred, "r001_%s_test.npz" % node, node, (fold + 1) % 5, rows, fos, 0.5, rng)
                    _pred(pred, "r002_%s_test.npz" % node, node, fold, rows, fos, skill[method],
                          rng, drop=cut if node == "node_a" else 0)
            elif method == "Centralized":
                _write_json(os.path.join(d, "results.json"), _baseline_json("centralized", seed))
                for node in NODES:
                    _pred(pred, "selected_%s_test.npz" % node, node, fold, rows, fos, skill[method], rng)
            else:
                _write_json(os.path.join(d, "summary.json"), {"experiment": "local_only_batch",
                                                              "seed": seed})
                for node in NODES:
                    _write_json(os.path.join(d, node, "results.json"),
                                _baseline_json("local_only", seed, {"node_name": node}))
                    _pred(pred, "selected_%s_test.npz" % node, node, fold, rows, fos,
                          skill[method], rng, drop=cut if node == "node_b" else 0)
    return root


def _write_inputs_light(root, rows):
    labels = os.path.join(root, "data", "raw", "camera", "labels.csv")
    os.makedirs(os.path.dirname(labels), exist_ok=True)
    with open(labels, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(LABEL_COLUMNS)
        for r in rows:
            w.writerow([r[c] for c in LABEL_COLUMNS])
    folds = os.path.join(root, "data", "splits_camera", FL_FILE)
    os.makedirs(os.path.dirname(folds), exist_ok=True)
    with open(folds, "wb") as f:
        f.write(render(SCENES)[FL_FILE])


def _load(root):
    return cab.load_all(root, os.path.join(_REPO, "configs", "experiment_matrix.yaml"),
                        os.path.join(root, "data", "raw", "camera", "labels.csv"),
                        os.path.join(root, "data", "splits_camera", FL_FILE))


@pytest.fixture(scope="module")
def camera_results(tmp_path_factory):
    return _camera_results(str(tmp_path_factory.mktemp("camres")))


def test_fold_concatenation_covers_each_scene_once(camera_results):
    preds, frames, folds, _ = _load(camera_results)
    assert set(preds) == set(cab.METHOD_ORDER)
    for method, by_seed in preds.items():
        assert sorted(by_seed) == [42, 123, 456]
        for nodes in by_seed.values():
            for node in NODES:
                keys = nodes[node]["key"].tolist()
                assert keys == sorted(fid for (n, fid) in frames if n == node)
                assert set(nodes[node]["scene"]) == set(SCENES)
    assert cab.check_image_set(preds, frames) == {
        n: sum(1 for r in _label_rows() if r["node"] == n and r["valid"]) for n in NODES}


def test_the_selected_round_is_used(camera_results):
    """Round 1 holds another fold's images; loading it would fail the fold check."""
    rel = cab.expected_runs(os.path.join(_REPO, "configs", "experiment_matrix.yaml"))["FedBN"][(3, 42)]
    files = cab.selected_test_files(os.path.join(camera_results, rel))
    assert sorted(files) == list(NODES)
    assert all(os.path.basename(p).startswith("r002_") for p in files.values())


def test_a_different_image_set_is_refused(tmp_path):
    root = _camera_results(str(tmp_path / "d"), drop={("Local-only", 2, 123): 1})
    preds, frames, _, _ = _load(root)
    with pytest.raises(ValueError, match="same image set"):
        cab.check_image_set(preds, frames)
    with pytest.raises(ValueError):
        cab.analyze(preds, B=50)


def test_missing_baseline_predictions_are_refused(tmp_path):
    root = _camera_results(str(tmp_path / "m"))
    rel = cab.expected_runs(os.path.join(_REPO, "configs", "experiment_matrix.yaml"))["Local-only"][(0, 42)]
    shutil.rmtree(os.path.join(root, rel, "predictions"))
    with pytest.raises(ValueError, match="predict_from_checkpoint"):
        _load(root)


def test_analysis_end_to_end(camera_results, tmp_path):
    out = str(tmp_path / "b")
    assert cab.main(["--root", camera_results, "--config",
                     os.path.join(_REPO, "configs", "experiment_matrix.yaml"),
                     "--output_dir", out, "--B", "300"]) == 0
    comps = pd.read_csv(os.path.join(out, "camera_b_comparisons.csv"))
    assert len(comps) == 2 * 2 * 3
    assert (comps.holm_m == 3).all()
    allowed = {"B-local": {"federation improves on training each sensor alone",
                           "federation worse than training each sensor alone",
                           "no detectable difference"},
               "B-central": {"higher than centralized", "lower than centralized",
                             "no detectable difference"}}
    for r in comps.itertuples():
        assert r.verdict in allowed[r.family] and r.verdict_holm in allowed[r.family]
        assert r.reference == ("Local-only" if r.family == "B-local" else "Centralized")
    primary = comps[(comps.family == "B-local") & (comps.aggregation == "pooled")].set_index("strategy")
    assert primary.loc["FedAvg", "verdict_holm"] == "federation improves on training each sensor alone"
    central = comps[(comps.family == "B-central") & (comps.aggregation == "pooled")].set_index("strategy")
    assert central.loc["FedBN", "verdict_holm"] == "lower than centralized"
    # Holm within each (family, aggregation), never across families
    for _, g in comps.groupby(["family", "aggregation"]):
        np.testing.assert_allclose(g.p_holm.to_numpy(), cab.holm(g.p_boot.to_numpy()))
    md = open(os.path.join(out, "camera_b_comparisons.md"), encoding="utf-8").read()
    assert "equivalent" not in md.replace("never equivalence", "")
    runs = pd.read_csv(os.path.join(out, "camera_b_runs.csv"))
    assert len(runs) == 5 * 3
    assert len(set(runs.n_images)) == 1
    methods = pd.read_csv(os.path.join(out, "camera_b_methods.csv"))
    assert set(methods.aggregation) == {"pooled", "clientmean"}
    info = json.load(open(os.path.join(out, "camera_b_image_set.json")))
    assert info["scenes"] == SCENES


def test_pooled_diff_is_seed_paired_diff_ci_with_scene_clusters(camera_results):
    preds, _, _, _ = _load(camera_results)
    _, _, comps = cab.analyze(preds, B=200)
    row = comps[(comps.family == "B-local") & (comps.aggregation == "pooled")
                & (comps.strategy == "FedAvg")].iloc[0]
    from src.evaluation.bootstrap import seed_paired_diff_ci
    a = [[cab.pooled_unit(preds["FedAvg"][s])] for s in (42, 123, 456)]
    b = [[cab.pooled_unit(preds["Local-only"][s])] for s in (42, 123, 456)]
    assert a[0][0].G == len(SCENES)                           # the scene is the cluster
    ref = seed_paired_diff_ci(a, b, "camera_b|B-local|FedAvg-Local-only|pooled", B=200)
    assert row["diff"] == pytest.approx(ref["diff"])
    assert row.ci_low == pytest.approx(ref["ci_low"]) and row.p_boot == pytest.approx(ref["p_boot"])


def test_clientmean_point_is_the_unweighted_client_mean(camera_results):
    preds, _, _, _ = _load(camera_results)
    runs, _, comps = cab.analyze(preds, B=100)
    fl = runs[runs.method == "FedAvg"].set_index("seed")
    for seed, r in fl.iterrows():
        assert r.clientmean_balanced_accuracy == pytest.approx(
            np.mean([r["%s_balanced_accuracy" % n] for n in NODES]))
    lo = runs[runs.method == "Local-only"].set_index("seed")
    want = np.mean([fl.loc[s, "clientmean_balanced_accuracy"] - lo.loc[s, "clientmean_balanced_accuracy"]
                    for s in (42, 123, 456)])
    got = comps[(comps.family == "B-local") & (comps.aggregation == "clientmean")
                & (comps.strategy == "FedAvg")].iloc[0]["diff"]
    assert got == pytest.approx(want)


def test_rule_phrases_and_holm_per_family():
    comps = pd.DataFrame([
        {"family": fam, "aggregation": "pooled", "strategy": s, "diff": d, "ci_low": lo,
         "ci_high": hi, "p_boot": p}
        for fam in ("B-local", "B-central")
        for s, d, lo, hi, p in (("FedAvg", 0.05, 0.01, 0.09, 0.02),
                                ("FedProx(0.01)", -0.04, -0.08, -0.01, 0.01),
                                ("FedBN", 0.01, -0.01, 0.03, 0.30))])
    out = cab.apply_rule(comps)
    local = out[out.family == "B-local"].set_index("strategy")
    central = out[out.family == "B-central"].set_index("strategy")
    assert local.loc["FedAvg", "verdict"] == "federation improves on training each sensor alone"
    assert local.loc["FedProx(0.01)", "verdict"] == "federation worse than training each sensor alone"
    assert local.loc["FedBN", "verdict"] == "no detectable difference"
    assert central.loc["FedAvg", "verdict"] == "higher than centralized"
    assert central.loc["FedProx(0.01)", "verdict"] == "lower than centralized"
    # Holm m = 3: 0.01 -> 0.03, 0.02 -> 0.04, 0.30 -> 0.30; per family, not m = 6
    assert local.loc["FedProx(0.01)", "p_holm"] == pytest.approx(0.03)
    assert local.loc["FedAvg", "p_holm"] == pytest.approx(0.04)
    assert local.loc["FedAvg", "verdict_holm"] == "federation improves on training each sensor alone"
    assert central.loc["FedProx(0.01)", "verdict_holm"] == "lower than centralized"
    assert (out.holm_m == 3).all()
    # the interval says "improves" but Holm does not reach 0.05 -> the headline says so
    comps.loc[comps.strategy == "FedAvg", "p_boot"] = 0.03
    out = cab.apply_rule(comps).set_index(["family", "strategy"])
    assert out.loc[("B-local", "FedAvg"), "verdict"] == "federation improves on training each sensor alone"
    assert out.loc[("B-local", "FedAvg"), "verdict_holm"] == "no detectable difference"
    with pytest.raises(ValueError, match="exactly 3"):
        cab.apply_rule(comps[comps.strategy != "FedBN"])
