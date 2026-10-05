"""Camera experiment, question (a): the leave-one-scene-out runner (scripts/camera_loso.py)
and the camera preprocessing option of CustomRGBDDataset.

Tiny synthetic tree (3 scenes x 2 classes x 30 frames per camera, 32-px inputs, 1-2
epochs, random init) so everything runs on CPU in seconds.
"""

import json
import os

import numpy as np
import pytest
import torch

from scripts import camera_labels as cl
from scripts import camera_loso as loso
from scripts import camera_manifests as cm
from src.data.custom_dataset import CustomRGBDDataset
from src.evaluation.predictions import load_npz
from tests.test_camera_labels_manifests import build_tree

IMG = 32
NODES = ("node_a", "node_b", "node_c")


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    base = tmp_path_factory.mktemp("camloso")
    root = build_tree(str(base / "camera"))
    splits = str(base / "splits")
    assert cl.main(["--data_dir", root, "--splits_dir", splits]) == 0
    assert cm.main(["--labels_csv", os.path.join(root, "labels.csv"),
                    "--splits_dir", splits]) == 0
    from scripts import camera_preprocess_frames as cpf
    from src.data.camera_preprocess import PreprocessedStore
    pre, manifest = str(base / "camera_224"), str(base / "preprocessed_manifest.csv")
    assert cpf.main(["--raw_dir", root, "--output_dir", pre, "--manifest", manifest,
                     "--img_size", str(IMG)]) == 0
    return {"root": root, "splits": splits, "labels": os.path.join(root, "labels.csv"),
            "base": base, "pre": pre, "manifest": manifest,
            "store": PreprocessedStore(pre, manifest)}


def _argv(tree, out, *extra):
    return ["--data_dir", tree["root"], "--splits_dir", tree["splits"], "--output_root", out,
            "--preprocessed_dir", tree["pre"], "--preprocessed_manifest", tree["manifest"],
            "--epochs", "1", "--img_size", str(IMG), "--no_pretrained", *extra]


@pytest.fixture(scope="module")
def rgb_run(tree):
    out = str(tree["base"] / "out")
    assert loso.main(_argv(tree, out, "--sources", "node_a", "--modalities", "rgb", "--seeds",
                           "42", "--protocols", "loso", "random", "--random_folds", "3")) == 0
    return os.path.join(out, loso.run_dir_name("rgb", "node_a", 42))


def _kept(tree, node):
    return sorted(r["id"] for r in cl.read_labels(tree["labels"])
                  if r["node"] == node and r["valid"])


# --------------------------------------------------------------------------- dataset option
def test_camera_preprocess_option_shapes_and_cache(tree):
    index, _ = loso.camera_index(tree["root"], tree["labels"], "rgb_d")
    uids = [r["uid"] for r in index[:3]]
    for modality, ch in (("rgb", 3), ("rgb_d", 4)):
        cache = {}
        a = CustomRGBDDataset(tree["root"], ids=uids, modality=modality, img_size=IMG,
                              index=index, preprocess="camera", cache=cache,
                              preprocessed=tree["store"])
        b = CustomRGBDDataset(tree["root"], ids=uids, modality=modality, img_size=IMG,
                              index=index, preprocess="camera", preprocessed=tree["store"])
        x, y = a[0]
        assert tuple(x.shape) == (ch, IMG, IMG) and y in (0, 1)
        assert torch.equal(a[1][0], b[1][0]) and torch.equal(a[1][0], a[1][0])  # cache exact
        assert cache
    ir_index, _ = loso.camera_index(tree["root"], tree["labels"], "ir")
    assert {r["node"] for r in ir_index} == {"node_a", "node_b"}
    ds = CustomRGBDDataset(tree["root"], ids=[ir_index[0]["uid"]], modality="ir", img_size=IMG,
                           index=ir_index, preprocess="camera", preprocessed=tree["store"])
    assert tuple(ds[0][0].shape) == (1, IMG, IMG)
    with pytest.raises(ValueError, match="preprocess"):
        CustomRGBDDataset(tree["root"], ids=uids, modality="rgb", index=index, preprocess="x")


def test_camera_depth_channel_uses_camera_preprocess(tree):
    from src.data.camera_preprocess import depth_224
    from PIL import Image
    index, _ = loso.camera_index(tree["root"], tree["labels"], "rgb_d")
    rec = index[0]
    ds = CustomRGBDDataset(tree["root"], ids=[rec["uid"]], modality="rgb_d", img_size=IMG,
                           index=index, preprocess="camera", preprocessed=tree["store"])
    want = depth_224(np.asarray(Image.open(rec["path_depth"])), size=IMG)
    assert np.allclose(ds[0][0][3].numpy(), (want - 0.5) / 0.25, atol=1e-6)


def test_camera_mode_reads_only_the_preprocessed_files(tree, tmp_path):
    """Amendment 2: no native frame is decoded; no manifest, no data; a changed file is refused."""
    import shutil
    from src.data.camera_preprocess import PreprocessedStore, rgb_224
    from PIL import Image
    index, _ = loso.camera_index(tree["root"], tree["labels"], "rgb")
    rec = index[0]
    with pytest.raises(ValueError, match="Amendment 2"):
        CustomRGBDDataset(tree["root"], ids=[rec["uid"]], index=index, img_size=IMG,
                          preprocess="camera")
    with pytest.raises(ValueError, match="px"):
        CustomRGBDDataset(tree["root"], ids=[rec["uid"]], index=index, img_size=IMG * 2,
                          preprocess="camera", preprocessed=tree["store"])
    ds = CustomRGBDDataset(tree["root"], ids=[rec["uid"]], index=index, img_size=IMG,
                           preprocess="camera", preprocessed=tree["store"])
    want = np.asarray(rgb_224(Image.open(rec["path_rgb"]), size=IMG), dtype=np.float32) / 255.0
    got = ds[0][0].numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) +         np.array([0.485, 0.456, 0.406])
    assert np.allclose(got, want, atol=1e-6)
    pre = str(tmp_path / "pre")
    shutil.copytree(tree["pre"], pre)
    store = PreprocessedStore(pre, tree["manifest"])
    Image.fromarray(np.zeros((IMG, IMG, 3), np.uint8)).save(store.path(rec["node"], rec["id"], "rgb"))
    bad = CustomRGBDDataset(tree["root"], ids=[rec["uid"]], index=index, img_size=IMG,
                            preprocess="camera", preprocessed=store)
    with pytest.raises(ValueError, match="md5"):
        bad[0]


def test_single_channel_adaptation_equals_grey_rgb():
    from src.models.mobilenetv3_multimodal import create_model
    torch.manual_seed(0)
    rgb = create_model(num_classes=2, in_channels=3, pretrained=False).eval()
    import copy
    one = loso.adapt_single_channel(copy.deepcopy(rgb)).eval()
    x = torch.randn(2, 1, IMG, IMG)
    with torch.no_grad():
        assert torch.allclose(one(x), rgb(x.repeat(1, 3, 1, 1)), atol=1e-5)


# --------------------------------------------------------------------------- runner
def test_out_of_fold_complete_for_every_target(rgb_run, tree):
    for target in NODES:
        for proto in ("loso", "random"):
            d = load_npz(os.path.join(rgb_run, loso.oof_name("node_a", target, proto)))
            ids = [str(p) for p in d["path"]]
            assert ids == _kept(tree, target)                 # every kept frame exactly once
            labels = {r["id"]: cl.LABEL_INDEX[r["label"]] for r in cl.read_labels(tree["labels"])
                      if r["node"] == target}
            assert [int(v) for v in d["label"]] == [labels[i] for i in ids]
            assert np.all(np.isfinite(d["logit_margin"]))


def test_results_json_folds(rgb_run, tree):
    with open(os.path.join(rgb_run, "results.json")) as f:
        res = json.load(f)
    assert res["experiment"] == "camera_loso"
    c = res["config"]
    assert (c["source"], c["modality"], c["seed"], c["epochs"]) == ("node_a", "rgb", 42, 1)
    assert c["targets"] == list(NODES) and c["preprocess"] == "camera_preprocess"
    folds = res["loso"]["folds"]
    assert [f["test_scene"] for f in folds] == ["s01", "s02", "s03"]
    for f in folds:
        assert f["val_scene"] != f["test_scene"]
        assert f["test_scene"] not in f["train_scenes"] and f["val_scene"] not in f["train_scenes"]
        assert f["selected_epoch"] == 1 and len(f["val_loss_by_epoch"]) == 1
        assert f["n_test"] == {n: 60 for n in NODES}
    assert "commit" in res["loso"] and res["loso"]["total_time_s"] >= 0
    assert len(res["random"]["folds"]) == 3


def test_complete_runs_are_skipped(rgb_run, tree, capsys):
    out = os.path.dirname(rgb_run)
    before = os.path.getmtime(os.path.join(rgb_run, "oof_node_a_on_node_b.npz"))
    assert loso.main(_argv(tree, out, "--sources", "node_a", "--seeds", "42")) == 0
    assert "complete, skipped" in capsys.readouterr().out
    assert os.path.getmtime(os.path.join(rgb_run, "oof_node_a_on_node_b.npz")) == before


def test_ir_only_realsense_targets(tree):
    out = str(tree["base"] / "out_ir")
    assert loso.main(_argv(tree, out, "--sources", "node_a", "node_c", "--modalities", "ir",
                           "--seeds", "42")) == 0
    run = os.path.join(out, loso.run_dir_name("ir", "node_a", 42))
    assert sorted(f for f in os.listdir(run) if f.endswith(".npz")) == \
        ["oof_node_a_on_node_a.npz", "oof_node_a_on_node_b.npz"]
    assert not os.path.exists(os.path.join(out, loso.run_dir_name("ir", "node_c", 42)))


def test_excluded_frames_never_used(tree, tmp_path):
    import shutil
    root = str(tmp_path / "camera")
    shutil.copytree(tree["root"], root)
    splits = str(tmp_path / "splits")
    os.remove(os.path.join(root, "node_b", "s02_fire_d100_0003_rgb.png"))
    assert cl.main(["--data_dir", root, "--splits_dir", splits]) == 0
    index, _ = loso.camera_index(root, os.path.join(root, "labels.csv"), "rgb")
    assert "node_b/s02_fire_d100_0003" not in {r["uid"] for r in index}
    # the capture now has 29 valid frames on node_b: dropped on all cameras, scene too
    assert "s02" not in {r["scene"] for r in index}
    # the manifests must be regenerated from the new kept scenes: stale ones are refused
    with pytest.raises(SystemExit):
        loso.loso_folds_for(tree["splits"], os.path.join(root, "labels.csv"))


def test_selection_never_reads_test_frames(tree, monkeypatch):
    index, _ = loso.camera_index(tree["root"], tree["labels"], "rgb")
    folds = loso.loso_folds_for(tree["splits"], tree["labels"])
    cfg = {"data_dir": tree["root"], "modality": "rgb", "seed": 42, "img_size": IMG,
           "batch_size": 8, "epochs": 2, "lr": 1e-3, "pretrained": False, "cache": {},
           "preprocessed": tree["store"]}
    loaded, phase = [], {"on": False}
    orig_get = CustomRGBDDataset.__getitem__

    def spy(self, idx):
        if phase["on"]:
            loaded[-1].add(self.records[idx]["scene"])
        return orig_get(self, idx)

    orig_sel = loso.train_and_select
    calls = []

    def wrapped(train, val, cfg_, device):
        calls.append({r["scene"] for r in train} | {r["scene"] for r in val})
        loaded.append(set())
        phase["on"] = True
        try:
            model, sel, hist = orig_sel(train, val, cfg_, device)
        finally:
            phase["on"] = False
        # the declared rule: lowest validation loss, earliest on ties
        losses = [h["val_loss"] for h in hist]
        assert sel == 1 + int(np.argmin(losses))
        assert abs(loso.val_loss(model, val, cfg_, device) - losses[sel - 1]) < 1e-5
        return model, sel, hist

    monkeypatch.setattr(CustomRGBDDataset, "__getitem__", spy)
    monkeypatch.setattr(loso, "train_and_select", wrapped)
    preds, log = loso.run_loso(index, folds, cfg, torch.device("cpu"), "node_a")
    assert len(calls) == len(folds) == 3
    for f, given, seen in zip(folds, calls, loaded):
        assert f["test_scene"] not in given and f["test_scene"] not in seen
        assert seen == given
    loso.check_out_of_fold(preds, index, "rgb")


def test_check_out_of_fold_detects_gaps_and_duplicates(tree):
    index, _ = loso.camera_index(tree["root"], tree["labels"], "rgb")
    ids = {t: [r["id"] for r in index if r["node"] == t] for t in NODES}
    preds = {t: {"id": list(v)} for t, v in ids.items()}
    loso.check_out_of_fold(preds, index, "rgb")
    preds["node_b"]["id"] = ids["node_b"][1:]
    with pytest.raises(RuntimeError, match="cover"):
        loso.check_out_of_fold(preds, index, "rgb")
    preds["node_b"]["id"] = ids["node_b"] + ids["node_b"][:1]
    with pytest.raises(RuntimeError, match="more than once"):
        loso.check_out_of_fold(preds, index, "rgb")


def test_random_split_folds_shared_across_cameras(tree):
    index, _ = loso.camera_index(tree["root"], tree["labels"], "rgb")
    a = loso.random_folds(index, 5, 42)
    assert a == loso.random_folds(list(reversed(index)), 5, 42)
    assert set(a) == {r["id"] for r in index}          # one fold per frame id, all cameras
    for label in ("fire", "no_fire"):
        counts = np.bincount([a[i] for i in a if ("_%s_" % label) in i and
                              not (label == "fire" and "_no_fire_" in i)], minlength=5)
        assert counts.max() - counts.min() <= 1
