"""Camera experiment: labels.csv, the pre-registered exclusions and the scene manifests.

docs/CAMERA_EXPERIMENT_PREREG.md sec. 2 ("Exclusions"), 5 and 6.  Synthetic capture trees
following the capture contract are built in tmp dirs; ``build_tree`` is reused by
test_camera_loso.py.
"""

import csv
import hashlib
import json
import os

import numpy as np
import pytest
from PIL import Image

from scripts import camera_labels as cl
from scripts import camera_manifests as cm

NODES = ("node_a", "node_b", "node_c")
SIZES = {"node_a": (40, 30), "node_b": (48, 32), "node_c": (44, 33)}   # different sensors


def capture_id(scene, label, cm_):
    return "%s_%s_d%d" % (scene, label, cm_)


def write_frame(node_dir, node, cap, scene, label, dist_cm, idx, rng):
    fid = "%s_%04d" % (cap, idx)
    w, h = SIZES[node]
    rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    if label == "fire":                      # learnable class signal
        rgb[..., 0] = np.maximum(rgb[..., 0], 200)
    Image.fromarray(rgb, "RGB").save(os.path.join(node_dir, fid + "_rgb.png"))
    depth = (1000 * dist_cm // 100 + rng.integers(0, 300, size=(h, w))).astype(np.uint16)
    Image.fromarray(depth).save(os.path.join(node_dir, fid + "_depth.png"))
    if node != "node_c":
        ir = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
        Image.fromarray(ir, "L").save(os.path.join(node_dir, fid + "_ir.png"))
    meta = {"frame_id": fid, "capture_id": cap, "scene": scene, "label": label,
            "distance_m": dist_cm / 100.0, "frame_index": idx, "node": node,
            "serial": "X" + node, "timestamp_unix": 1.79e9 + idx * 0.2}
    with open(os.path.join(node_dir, fid + "_meta.json"), "w") as f:
        json.dump(meta, f)


def build_tree(root, scenes=("s01", "s02", "s03"), n_frames=30, distances=(100,),
               extra=(), reverse=False, seed=0):
    """Every scene x class x distance on every node, ``n_frames`` each.

    ``extra``: additional (scene, label, cm) captures.  ``reverse`` writes the files in
    the opposite order (the output must not depend on it)."""
    rng = np.random.default_rng(seed)
    caps = [(s, lab, d) for s in scenes for lab in ("fire", "no_fire") for d in distances]
    caps += list(extra)
    jobs = [(node, c, i) for node in NODES for c in caps for i in range(n_frames)]
    if reverse:
        jobs = jobs[::-1]
    for node in NODES:
        os.makedirs(os.path.join(root, node, "_captures"), exist_ok=True)
    frames = {}
    for node, (s, lab, d), i in jobs:
        # content is keyed by the frame, not by the write order
        frng = np.random.default_rng([seed, NODES.index(node), int(s[1:]),
                                      int(lab == "fire"), d, i])
        write_frame(os.path.join(root, node), node, capture_id(s, lab, d), s, lab, d, i, frng)
        frames[(node, capture_id(s, lab, d))] = frames.get((node, capture_id(s, lab, d)), 0) + 1
    for (node, cap), n in sorted(frames.items(), reverse=reverse):
        with open(os.path.join(root, node, "_captures", cap + ".json"), "w") as f:
            json.dump({"capture_id": cap, "n_frames_written": n}, f)
    return root


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_labels(root, splits):
    assert cl.main(["--data_dir", root, "--splits_dir", splits]) == 0
    return read_csv(os.path.join(root, "labels.csv")), read_csv(os.path.join(splits, "exclusions.csv"))


# --------------------------------------------------------------------------- labels
@pytest.fixture(scope="module")
def clean_tree(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("cam") / "camera")
    return build_tree(root)


def test_clean_tree_all_valid_and_sorted(clean_tree, tmp_path):
    rows, excl = run_labels(clean_tree, str(tmp_path / "splits"))
    assert list(rows[0]) == list(cl.LABEL_COLUMNS)
    assert len(rows) == 3 * 3 * 2 * 30
    assert all(r["valid"] == "1" and r["exclusion_reason"] == "" for r in rows)
    assert [(r["node"], r["id"]) for r in rows] == sorted((r["node"], r["id"]) for r in rows)
    assert excl == []
    r = next(x for x in rows if x["id"] == "s02_no_fire_d100_0007" and x["node"] == "node_b")
    assert (r["scene"], r["label"], r["capture_id"], r["frame_index"], r["distance_m"]) == \
        ("s02", "no_fire", "s02_no_fire_d100", "7", "1.000")


def test_labels_csv_is_readable_by_custom_dataset(clean_tree, tmp_path):
    from src.data.custom_dataset import build_label_map, load_frame_index
    run_labels(clean_tree, str(tmp_path / "splits"))
    index = load_frame_index(clean_tree, labels_csv=os.path.join(clean_tree, "labels.csv"),
                             nodes=list(NODES))
    assert len(index) == 3 * 3 * 2 * 30
    assert build_label_map(r["label_name"] for r in index) == {"no_fire": 0, "fire": 1}
    assert cl.LABEL_INDEX == {"no_fire": 0, "fire": 1}


def test_frame_rules_missing_unreadable_constant(tmp_path):
    root = build_tree(str(tmp_path / "camera"), n_frames=32)
    nb = os.path.join(root, "node_b")
    os.remove(os.path.join(nb, "s01_fire_d100_0000_rgb.png"))              # missing
    with open(os.path.join(nb, "s01_fire_d100_0001_rgb.png"), "wb") as f:  # unreadable
        f.write(b"not a png")
    const = np.full((32, 48, 3), 120, np.uint8)
    const[0, 0] = 121                                                      # std << 2
    Image.fromarray(const).save(os.path.join(nb, "s02_fire_d100_0005_rgb.png"))
    almost = np.full((32, 48, 3), 100, np.uint8)
    almost[:, :24, 1] = 110                                                # channel G std 5
    Image.fromarray(almost).save(os.path.join(nb, "s02_fire_d100_0006_rgb.png"))

    rows, excl = run_labels(root, str(tmp_path / "splits"))
    by = {(r["node"], r["id"]): r for r in rows}
    assert by[("node_b", "s01_fire_d100_0000")]["exclusion_reason"] == "rgb_missing"
    assert by[("node_b", "s01_fire_d100_0001")]["exclusion_reason"] == "rgb_unreadable"
    assert by[("node_b", "s02_fire_d100_0005")]["exclusion_reason"] == "rgb_nearly_constant"
    assert by[("node_b", "s02_fire_d100_0006")]["valid"] == "1"   # one channel varies
    # 30 valid frames remain in each capture: nothing else is dropped
    assert sum(r["valid"] == "0" for r in rows) == 3
    assert {(e["level"], e["key"], e["reason"]) for e in excl} == {
        ("frame", "node_b/s01_fire_d100_0000", "rgb_missing"),
        ("frame", "node_b/s01_fire_d100_0001", "rgb_unreadable"),
        ("frame", "node_b/s02_fire_d100_0005", "rgb_nearly_constant")}


def test_capture_below_30_on_one_camera_dropped_on_all(tmp_path):
    # s03 has two fire distances; dropping one keeps the class, so the scene stays
    root = build_tree(str(tmp_path / "camera"), extra=[("s03", "fire", 200)])
    os.remove(os.path.join(root, "node_c", "s03_fire_d200_0004_rgb.png"))
    rows, excl = run_labels(root, str(tmp_path / "splits"))
    cap = [r for r in rows if r["capture_id"] == "s03_fire_d200"]
    assert len(cap) == 3 * 30 and all(r["valid"] == "0" for r in cap)
    assert {r["exclusion_reason"] for r in cap if r["node"] != "node_c"} == {"capture_dropped"}
    assert ("capture", "s03_fire_d200") in {(e["level"], e["key"]) for e in excl}
    reason = next(e["reason"] for e in excl if e["key"] == "s03_fire_d200")
    assert "node_c (29)" in reason and "node_a" not in reason
    assert all(r["valid"] == "1" for r in rows if r["capture_id"] == "s03_fire_d100")
    assert cl.kept_scenes(cl.read_labels(os.path.join(root, "labels.csv"))) == ["s01", "s02", "s03"]


def test_scene_losing_a_class_on_any_camera_is_dropped(tmp_path):
    root = build_tree(str(tmp_path / "camera"), scenes=("s01", "s02", "s03", "s04"))
    # s02: its only no_fire capture falls below 30 valid frames on node_a
    for i in range(5):
        os.remove(os.path.join(root, "node_a", "s02_no_fire_d100_%04d_rgb.png" % i))
    # s04: its no_fire capture was never recorded on node_b (only a capture record)
    for name in os.listdir(os.path.join(root, "node_b")):
        if name.startswith("s04_no_fire"):
            os.remove(os.path.join(root, "node_b", name))
    rows, excl = run_labels(root, str(tmp_path / "splits"))
    s02 = [r for r in rows if r["scene"] == "s02"]
    assert all(r["valid"] == "0" for r in s02)
    assert {r["exclusion_reason"] for r in s02 if r["label"] == "fire"} == {"scene_dropped"}
    scene_excl = {e["key"]: e["reason"] for e in excl if e["level"] == "scene"}
    assert set(scene_excl) == {"s02", "s04"}
    assert "no_fire" in scene_excl["s02"]
    assert all(r["valid"] == "0" for r in rows if r["scene"] == "s04")
    assert cl.kept_scenes(cl.read_labels(os.path.join(root, "labels.csv"))) == ["s01", "s03"]


def test_scene_with_one_class_only_is_dropped():
    rows = [{"id": "s09_fire_d100_%04d" % i, "node": n, "scene": "s09", "label": "fire",
             "distance_m": 1.0, "capture_id": "s09_fire_d100", "frame_index": i, "valid": 1,
             "exclusion_reason": ""} for n in NODES for i in range(30)]
    out, excl, summary = cl.apply_exclusions(rows, [], NODES)
    assert summary["scenes_kept"] == []
    assert [e["level"] for e in excl] == ["scene"]


def test_meta_contradicting_the_frame_id_raises(tmp_path):
    root = build_tree(str(tmp_path / "camera"), scenes=("s01",))
    p = os.path.join(root, "node_a", "s01_fire_d100_0003_meta.json")
    with open(p) as f:
        meta = json.load(f)
    meta["label"] = "no_fire"
    with open(p, "w") as f:
        json.dump(meta, f)
    with pytest.raises(ValueError, match="contradicts"):
        cl.build(root)


def test_labels_independent_of_file_creation_order(tmp_path):
    a = build_tree(str(tmp_path / "a" / "camera"))
    b = build_tree(str(tmp_path / "b" / "camera"), reverse=True)
    run_labels(a, str(tmp_path / "a" / "splits"))
    run_labels(b, str(tmp_path / "b" / "splits"))
    with open(os.path.join(a, "labels.csv"), "rb") as fa, \
            open(os.path.join(b, "labels.csv"), "rb") as fb:
        blob = fa.read()
        assert blob == fb.read()
    assert b"\r\n" not in blob


# --------------------------------------------------------------------------- manifests
SCENES = ["s%02d" % i for i in range(1, 18)]


def test_loso_fold_properties():
    folds = cm.loso_folds(SCENES[::-1])
    assert [f["test_scene"] for f in folds] == sorted(SCENES)       # each tested exactly once
    for i, f in enumerate(folds):
        assert f["fold"] == i
        assert f["val_scene"] != f["test_scene"]
        assert f["val_scene"] == sorted(SCENES)[(i + 1) % len(SCENES)]
    assert folds[-1]["val_scene"] == "s01"                           # wraps
    with pytest.raises(ValueError):
        cm.loso_folds(["s01", "s02"])


def test_fl_folds_seeded_permutation_round_robin():
    folds = cm.fl_folds(SCENES)
    perm = np.random.default_rng(20260928).permutation(np.array(sorted(SCENES)))
    assert folds == {str(s): i % 5 for i, s in enumerate(perm)}
    counts = np.bincount(list(folds.values()), minlength=5)
    assert counts.max() - counts.min() <= 1 and counts.sum() == len(SCENES)


def test_manifests_depend_on_scene_ids_only_and_check_mode(tmp_path):
    rng = np.random.default_rng(1)
    a = cm.render(SCENES)
    b = cm.render(list(rng.permutation(SCENES)) + ["s03"])          # order, duplicates
    assert a == b
    d = str(tmp_path / "splits")
    files = cm.write(SCENES, d)
    for name in (cm.LOSO_FILE, cm.FL_FILE):
        with open(os.path.join(d, name), "rb") as f:
            blob = f.read()
        assert hashlib.sha256(blob).hexdigest() in files[cm.SHA_FILE].decode()
        assert b"\r\n" not in blob
    assert set(cm.check(SCENES, d).values()) == {"identical"}
    assert cm.check(SCENES[:-1], d)[cm.LOSO_FILE] == "different"
    with open(os.path.join(d, cm.FL_FILE), "ab") as f:
        f.write(b"s99,0\n")
    assert cm.check(SCENES, d)[cm.FL_FILE] == "different"


def test_manifest_cli_from_labels(clean_tree, tmp_path):
    splits = str(tmp_path / "splits")
    run_labels(clean_tree, splits)
    labels = os.path.join(clean_tree, "labels.csv")
    assert cm.main(["--labels_csv", labels, "--splits_dir", splits]) == 0
    assert read_csv(os.path.join(splits, "loso_folds.csv")) == [
        {"fold": "0", "test_scene": "s01", "val_scene": "s02"},
        {"fold": "1", "test_scene": "s02", "val_scene": "s03"},
        {"fold": "2", "test_scene": "s03", "val_scene": "s01"}]
    assert cm.main(["--labels_csv", labels, "--splits_dir", splits, "--check"]) == 0
    assert cm.main(["--scenes", "s01", "s02", "s04", "--splits_dir", splits, "--check"]) == 1
