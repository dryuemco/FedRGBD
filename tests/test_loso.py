"""Tests for the leave-one-scene-out (LOSO) cross-sensor evaluation.

Covers ``src/data/custom_dataset.py`` (frame index, label sources, modality
channel counts, depth scaling) and ``scripts/cross_sensor_loso.py`` (fold
construction, scene independence, results.json schema, pooled random-split
baseline).  Everything runs on CPU against a synthetic 16-pixel capture tree.
"""

import csv
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from scripts import cross_sensor_loso as loso
from src.data.custom_dataset import (
    MODALITY_CHANNELS,
    CustomRGBDDataset,
    build_label_map,
    filter_index,
    load_frame_index,
    scenes_in,
)

NODES = ["node_a", "node_b", "node_c"]
SCENES = ["kitchen", "lab", "corridor"]
LABELS = ["no_fire", "fire"]
FRAMES_PER_COMBO = 2
IMG = 16


# --------------------------------------------------------------------------- #
# synthetic capture tree
# --------------------------------------------------------------------------- #
def _write_frame(node_dir, frame_id, scene, label, rng, with_ir=True, meta_labels=True):
    rgb = rng.randint(0, 256, size=(IMG, IMG, 3), dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(os.path.join(node_dir, f"{frame_id}_rgb.png"))

    # 16-bit depth in millimetres, class-dependent so the model can learn
    base = 800 if label == "fire" else 4200
    depth = (base + rng.randint(0, 200, size=(IMG, IMG))).astype(np.uint16)
    Image.fromarray(depth).save(os.path.join(node_dir, f"{frame_id}_depth.png"))

    if with_ir:
        ir = rng.randint(0, 256, size=(IMG, IMG), dtype=np.uint8)
        Image.fromarray(ir, mode="L").save(os.path.join(node_dir, f"{frame_id}_ir.png"))

    meta = {"frame_id": frame_id, "timestamp_ms": 1000.0 + rng.rand(),
            "depth_min_mm": int(depth.min()), "depth_max_mm": int(depth.max())}
    if meta_labels:
        meta.update({"scene": scene, "label": label})
    with open(os.path.join(node_dir, f"{frame_id}_meta.json"), "w") as f:
        json.dump(meta, f)


def _build_tree(root, nodes=NODES, meta_labels=True, filename_pattern=False,
                ir_nodes=("node_a", "node_b")):
    rng = np.random.RandomState(0)
    os.makedirs(root, exist_ok=True)
    for node in nodes:
        node_dir = os.path.join(root, node)
        os.makedirs(node_dir, exist_ok=True)
        counter = 0
        for scene in SCENES:
            for label in LABELS:
                for k in range(FRAMES_PER_COMBO):
                    if filename_pattern:
                        frame_id = f"{scene}_{label}_{k:03d}"
                    else:
                        frame_id = f"{counter:05d}"
                    _write_frame(node_dir, frame_id, scene, label, rng,
                                 with_ir=node in ir_nodes, meta_labels=meta_labels)
                    counter += 1
    return root


@pytest.fixture(scope="module")
def capture_root(tmp_path_factory):
    return _build_tree(str(tmp_path_factory.mktemp("custom") / "custom"))


# --------------------------------------------------------------------------- #
# frame index / label sources
# --------------------------------------------------------------------------- #
def test_index_from_meta_json(capture_root):
    index = load_frame_index(capture_root)
    expected = len(NODES) * len(SCENES) * len(LABELS) * FRAMES_PER_COMBO
    assert len(index) == expected
    assert scenes_in(index) == sorted(SCENES)
    assert {r["node"] for r in index} == set(NODES)
    assert {r["label_source"] for r in index} == {"meta_json"}
    assert {r["label"] for r in index} == {0, 1}
    rec = index[0]
    assert set(rec) >= {"id", "node", "scene", "label", "has_depth", "has_ir", "path_rgb"}
    assert rec["uid"] == f"{rec['node']}/{rec['id']}"
    assert rec["has_depth"] is True
    # node_c (ZED) has no IR stream
    assert all(not r["has_ir"] for r in index if r["node"] == "node_c")
    assert all(r["has_ir"] for r in index if r["node"] in ("node_a", "node_b"))


def test_label_map_follows_repo_convention(capture_root):
    index = load_frame_index(capture_root)
    mapping = build_label_map(r["label_name"] for r in index)
    assert mapping == {"no_fire": 0, "fire": 1}


def test_index_from_labels_csv(tmp_path):
    root = _build_tree(str(tmp_path / "nolabels"), nodes=["node_a", "node_b"],
                       meta_labels=False)
    csv_path = str(tmp_path / "labels.csv")
    rows = []
    counter = 0
    for scene in SCENES:
        for label in LABELS:
            for _ in range(FRAMES_PER_COMBO):
                rows.append({"id": f"{counter:05d}", "scene": scene, "label": label})
                counter += 1
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "scene", "label"])
        writer.writeheader()
        writer.writerows(rows)

    index = load_frame_index(root, labels_csv=csv_path)
    assert len(index) == 2 * len(SCENES) * len(LABELS) * FRAMES_PER_COMBO
    assert {r["label_source"] for r in index} == {"labels_csv"}
    assert scenes_in(index) == sorted(SCENES)


def test_index_from_filename_pattern(tmp_path):
    root = _build_tree(str(tmp_path / "byname"), nodes=["node_a"],
                       meta_labels=False, filename_pattern=True)
    index = load_frame_index(root)
    assert {r["label_source"] for r in index} == {"filename"}
    assert scenes_in(index) == sorted(SCENES)
    assert {r["label_name"] for r in index} == set(LABELS)


def test_missing_label_source_raises_explicit_error(tmp_path):
    root = _build_tree(str(tmp_path / "unlabelled"), nodes=["node_a"], meta_labels=False)
    with pytest.raises(ValueError) as exc:
        load_frame_index(root)
    msg = str(exc.value)
    assert "Could not determine (scene, label)" in msg
    assert "labels.csv" in msg and "meta.json" in msg
    assert "<scene>_<label>_<n>" in msg
    assert "frame_id" in msg  # reports the meta.json keys actually found
    assert "node_a/00000" in msg


def test_bad_labels_csv_columns(tmp_path, capture_root):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,klass\n00000,fire\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_frame_index(capture_root, labels_csv=str(bad))
    with pytest.raises(FileNotFoundError):
        load_frame_index(capture_root, labels_csv=str(tmp_path / "nope.csv"))


def test_empty_root_raises(tmp_path):
    empty = tmp_path / "empty"
    (empty / "node_a").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no '\\*_rgb.png' frames"):
        load_frame_index(str(empty))


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("modality,channels", sorted(MODALITY_CHANNELS.items()))
def test_all_modalities_load_with_right_channel_count(capture_root, modality, channels):
    index = load_frame_index(capture_root)
    usable = filter_index(index, modality=modality)
    assert usable, f"no usable frames for {modality}"
    if modality in ("ir", "rgb_d_ir"):  # ZED node has no IR
        assert {r["node"] for r in usable} == {"node_a", "node_b"}
    ds = CustomRGBDDataset(capture_root, ids=[r["uid"] for r in usable],
                           modality=modality, img_size=IMG, train=False, index=index)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and x.dtype == torch.float32
    assert tuple(x.shape) == (channels, IMG, IMG)
    assert y in (0, 1)
    assert torch.isfinite(x).all()
    assert len(ds) == len(usable)


def test_missing_modality_stream_raises(capture_root):
    index = load_frame_index(capture_root)
    zed = [r for r in index if r["node"] == "node_c"]
    with pytest.raises(FileNotFoundError, match="lack the files required"):
        CustomRGBDDataset(capture_root, ids=[r["uid"] for r in zed],
                          modality="rgb_d_ir", img_size=IMG, index=index)


def test_depth_normalisation_range(capture_root):
    index = load_frame_index(capture_root)
    ds = CustomRGBDDataset(capture_root, ids=None, modality="rgb_d", img_size=IMG,
                           train=False, index=index, max_depth_m=10.0)
    x, _ = ds[0]
    depth_channel = x[3]
    # depth in [0, max_depth_m] -> [0,1] -> (v - 0.5) / 0.25  =>  [-2, 2]
    assert float(depth_channel.min()) >= -2.0 - 1e-5
    assert float(depth_channel.max()) <= 2.0 + 1e-5


def test_unknown_modality_and_unknown_id(capture_root):
    index = load_frame_index(capture_root)
    with pytest.raises(ValueError, match="unknown modality"):
        CustomRGBDDataset(capture_root, ids=None, modality="lidar", index=index)
    with pytest.raises(KeyError):
        CustomRGBDDataset(capture_root, ids=["node_a/does_not_exist"],
                          modality="rgb", img_size=IMG, index=index)


def test_train_augmentation_is_deterministic(capture_root):
    index = load_frame_index(capture_root)
    kwargs = dict(ids=None, modality="rgb_d", img_size=IMG, index=index, seed=7)
    a = CustomRGBDDataset(capture_root, train=True, **kwargs)[3][0]
    b = CustomRGBDDataset(capture_root, train=True, **kwargs)[3][0]
    assert torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# LOSO script
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def loso_run(capture_root, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("loso_out"))
    rc = loso.main([
        "--data_dir", capture_root,
        "--train_nodes", "node_a", "node_b",
        "--test_nodes", "node_c",
        "--modality", "rgb_d",
        "--epochs", "1", "--batch_size", "4", "--img_size", str(IMG),
        "--seed", "42", "--no_pretrained",
        "--pooled_random_split",
        "--output_dir", out,
    ])
    assert rc == 0
    with open(os.path.join(out, "results.json")) as f:
        return json.load(f)


def test_results_schema(loso_run):
    assert loso_run["experiment"] == "cross_sensor_loso"
    assert loso_run["protocol"] == "leave_one_scene_out"
    assert loso_run["results_schema_version"] == 2
    assert {"folds", "summary", "config", "timestamp", "total_time_s"} <= set(loso_run)
    cfg = loso_run["config"]
    assert cfg["modality"] == "rgb_d" and cfg["in_channels"] == 4
    assert cfg["train_nodes"] == ["node_a", "node_b"]
    assert cfg["test_nodes"] == ["node_c"]
    assert loso_run["label_sources"] == {"meta_json": loso_run["n_frames"]}


def test_one_fold_per_scene(loso_run):
    folds = loso_run["folds"]
    assert len(folds) == len(SCENES)
    assert sorted(f["held_out_scene"] for f in folds) == sorted(SCENES)
    assert [f["fold"] for f in folds] == list(range(1, len(SCENES) + 1))


def test_held_out_scene_never_in_training_ids(loso_run):
    for fold in loso_run["folds"]:
        held = fold["held_out_scene"]
        assert held not in fold["train_scenes"]
        train_ids = set(fold["train_ids"])
        assert train_ids, "fold recorded no training ids"
        # every training frame belongs to a train node and another scene
        for uid in train_ids:
            node = uid.split("/")[0]
            assert node in fold["train_nodes"]
        for split in ("cross_camera", "same_camera"):
            eval_ids = set(fold["eval"][split]["ids"])
            assert eval_ids, f"{split} evaluation set is empty"
            assert not (eval_ids & train_ids), (
                f"fold {held}: held-out-scene frames leaked into training")


def test_fold_metrics_and_timing(loso_run):
    for fold in loso_run["folds"]:
        assert fold["train_time_s"] >= 0 and fold["fold_time_s"] >= 0
        assert len(fold["history"]) == 1
        for split in ("cross_camera", "same_camera"):
            entry = fold["eval"][split]
            assert entry["status"] == "ok"
            metrics = entry["metrics"]
            for key in ("accuracy", "balanced_accuracy", "precision", "recall",
                        "specificity", "f1", "macro_f1", "mcc", "confusion_matrix",
                        "n_examples", "loss"):
                assert key in metrics
            assert 0.0 <= metrics["accuracy"] <= 1.0
        assert fold["eval"]["cross_camera"]["nodes"] == ["node_c"]
        assert fold["eval"]["same_camera"]["nodes"] == ["node_a", "node_b"]


def test_summary_keys_and_aggregation(loso_run):
    summary = loso_run["summary"]
    assert {"n_folds", "scenes", "train_nodes", "test_nodes", "modality",
            "cross_camera", "same_camera", "loso_time_s", "total_time_s"} <= set(summary)
    assert summary["n_folds"] == len(SCENES)
    for split in ("cross_camera", "same_camera"):
        agg = summary[split]
        assert agg["n_folds"] == len(SCENES)
        assert sorted(agg["scenes"]) == sorted(SCENES)
        for key in ("accuracy", "balanced_accuracy", "f1", "mcc", "loss"):
            stats = agg[key]
            assert {"mean", "std", "min", "max", "values"} <= set(stats)
            assert len(stats["values"]) == len(SCENES)
            assert stats["min"] <= stats["mean"] <= stats["max"] + 1e-9
            assert stats["std"] >= 0.0


def test_pooled_random_split_baseline(loso_run):
    baseline = loso_run["pooled_random_split"]
    assert baseline["protocol"] == "frame_level_random_split"
    # the point of the baseline: every scene is present in train AND test
    train_scenes = {u for u in baseline["train_ids"]}
    assert train_scenes
    for split in ("cross_camera", "same_camera"):
        entry = baseline["eval"][split]
        assert entry["status"] == "ok"
        assert entry["n_examples"] > 0
        assert set(entry["ids"]).isdisjoint(set(baseline["train_ids"]))
    assert "pooled_random_split" in loso_run["summary"]
    assert loso_run["summary"]["pooled_random_split"]["cross_camera"]["accuracy"] is not None


def test_pooled_baseline_test_set_reuses_training_scenes(capture_root, tmp_path):
    """The old protocol's test frames come from the same scenes as training."""
    out = str(tmp_path / "prs")
    rc = loso.main([
        "--data_dir", capture_root,
        "--train_nodes", "node_a", "--test_nodes", "node_b",
        "--modality", "rgb", "--epochs", "1", "--batch_size", "4",
        "--img_size", str(IMG), "--no_pretrained", "--pooled_random_split",
        "--scenes", "kitchen", "lab",
        "--output_dir", out,
    ])
    assert rc == 0
    with open(os.path.join(out, "results.json")) as f:
        res = json.load(f)
    assert len(res["folds"]) == 2
    assert sorted(res["summary"]["scenes"]) == ["kitchen", "lab"]

    index = {r["uid"]: r for r in load_frame_index(capture_root)}
    baseline = res["pooled_random_split"]
    train_scenes = {index[u]["scene"] for u in baseline["train_ids"]}
    test_scenes = {index[u]["scene"] for u in baseline["eval"]["cross_camera"]["ids"]}
    assert test_scenes and test_scenes <= train_scenes  # scene leakage, by design


def test_same_node_mode(capture_root, tmp_path):
    out = str(tmp_path / "same")
    rc = loso.main([
        "--data_dir", capture_root,
        "--train_nodes", "node_a", "--same_node",
        "--modality", "rgb_d_ir", "--epochs", "1", "--batch_size", "4",
        "--img_size", str(IMG), "--no_pretrained",
        "--output_dir", out,
    ])
    assert rc == 0
    with open(os.path.join(out, "results.json")) as f:
        res = json.load(f)
    assert res["config"]["same_node"] is True
    assert res["config"]["test_nodes"] == ["node_a"]
    assert res["config"]["in_channels"] == 5
    assert len(res["folds"]) == len(SCENES)
    for fold in res["folds"]:
        assert fold["eval"]["cross_camera"]["ids"] == fold["eval"]["same_camera"]["ids"]


def test_requires_test_nodes(capture_root, tmp_path):
    with pytest.raises(SystemExit):
        loso.main(["--data_dir", capture_root, "--train_nodes", "node_a",
                   "--output_dir", str(tmp_path / "x")])


def test_single_scene_rejected(capture_root, tmp_path):
    with pytest.raises(SystemExit, match="needs >= 2 scenes"):
        loso.main(["--data_dir", capture_root, "--train_nodes", "node_a",
                   "--test_nodes", "node_b", "--scenes", "lab",
                   "--epochs", "1", "--img_size", str(IMG), "--no_pretrained",
                   "--output_dir", str(tmp_path / "y")])


# --------------------------------------------------------------------------- #
# experiment matrix block
# --------------------------------------------------------------------------- #
def test_experiment_matrix_block():
    import yaml

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "configs", "experiment_matrix.yaml")) as f:
        cfg = yaml.safe_load(f)
    block = cfg["revision"]["cross_sensor_loso"]
    assert block["script"] == "scripts/cross_sensor_loso.py"
    assert block["protocol"] == "leave_one_scene_out"
    assert block["seeds"] == [42, 123, 456]
    assert set(block["modalities"]) == set(MODALITY_CHANNELS)
    assert block["total_runs"] == len(block["configurations"]) * len(block["seeds"])
    parser = loso.build_parser()
    for cmd in block["commands"]:
        assert cmd.startswith("python3 scripts/cross_sensor_loso.py ")
        argv = cmd.split()[2:]
        args = parser.parse_args(argv)  # every documented command must parse
        assert args.output_dir.startswith("results/loso_")
    for conf in block["configurations"]:
        assert conf["train_nodes"]
        assert conf.get("same_node") or conf.get("test_nodes")
