#!/usr/bin/env python3
"""FedRGBD -- camera experiment, question (a): leave-one-scene-out cross-sensor runs.

Implements ``docs/CAMERA_EXPERIMENT_PREREG.md`` section 5 (read it first).  For a source
sensor X, a modality (rgb primary, rgb_d secondary, ir descriptive) and a seed, every fold
of ``data/splits_camera/loso_folds.csv`` (one per kept scene s; validation scene = the next
scene in sorted order; training scenes = the other S - 2):

* train MobileNetV3-Small (ImageNet-pretrained) on X's frames of the training scenes,
  15 epochs, Adam 1e-3, batch 8;
* select the epoch with the lowest validation loss on X's frames of the validation scene,
  earlier on ties (``src/evaluation/model_selection.select_round``; the selection code
  never receives a test frame -- the held-out scene is predicted only after it);
* predict the held-out scene as recorded by EACH sensor (ir: node_a and node_b only).

Every kept frame of every scene is thus predicted exactly once per (source, target, seed)
-- checked before anything is written -- and saved in the repository's prediction format
(``src/evaluation/predictions.py``; ``path`` is the frame id, the file name carries the
target) as ``<out>/oof_<source>_on_<target>.npz``, with ``<out>/results.json`` (config,
per-fold selected epoch and validation losses, timings, commit).

Only frames with ``valid == 1`` in ``labels.csv`` are used (the pre-registered exclusions of
``scripts/camera_labels.py``); a valid frame lacking a stream the modality needs (depth /
IR file) is left out of that modality and counted in results.json.  Images go through
``src/data/camera_preprocess.py`` (``CustomRGBDDataset(preprocess="camera")``).

``--protocols random`` additionally runs the descriptive frame-level random split (the v1
protocol R3.3 criticised) on the same frames: frame ids -- shared by the three cameras of
one capture -- are dealt, per label, into K folds with ``numpy.random.default_rng(seed)``;
fold k is tested, fold (k + 1) mod K validates, the rest trains, so every scene appears in
training and test; written as ``oof_random_<source>_on_<target>.npz``.

    python scripts/camera_loso.py --modalities rgb rgb_d --seeds 42 123 456
    python scripts/camera_loso.py --modalities ir --sources node_a node_b
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.camera_labels import NODES, SENSORS, kept_scenes, read_labels  # noqa: E402
from scripts.camera_manifests import LOSO_FILE, check as check_manifests  # noqa: E402
from scripts.camera_manifests import read_loso_folds  # noqa: E402

EXPERIMENT = "camera_loso"
RESULTS_SCHEMA_VERSION = 1
MODALITIES = ("rgb", "rgb_d", "ir")
IR_NODES = ("node_a", "node_b")
SEEDS = (42, 123, 456)
CLASS_TO_IDX = {"no_fire": 0, "fire": 1}


def target_nodes(modality: str) -> Tuple[str, ...]:
    return IR_NODES if modality == "ir" else NODES


def run_dir_name(modality: str, source: str, seed: int) -> str:
    return "%s_%s_seed%d" % (modality, source, seed)


def oof_name(source: str, target: str, protocol: str = "loso") -> str:
    prefix = "oof_random_" if protocol == "random" else "oof_"
    return "%s%s_on_%s.npz" % (prefix, source, target)


# --------------------------------------------------------------------------- index
def camera_index(data_dir: str, labels_csv: str, modality: str
                 ) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Valid frames of labels.csv usable for ``modality`` (frame-index records for
    ``CustomRGBDDataset``), sorted by (node, id); and the per-node count of valid frames
    left out because a stream file the modality needs is missing."""
    from src.data.custom_dataset import required_streams

    streams = required_streams(modality)
    records, missing = [], {}
    for r in read_labels(labels_csv):
        if not int(r["valid"]):
            continue
        node, fid = str(r["node"]), str(r["id"])
        if node not in target_nodes(modality):
            continue
        base = os.path.join(data_dir, node, fid)
        paths = {s: base + "_%s.png" % s for s in ("rgb", "depth", "ir")}
        if not all(os.path.isfile(paths[s]) for s in streams):
            missing[node] = missing.get(node, 0) + 1
            continue
        records.append({
            "uid": "%s/%s" % (node, fid), "id": fid, "node": node, "scene": str(r["scene"]),
            "label_name": str(r["label"]), "label": CLASS_TO_IDX[str(r["label"])],
            "label_source": "labels_csv", "capture_id": str(r["capture_id"]),
            "has_depth": os.path.isfile(paths["depth"]), "has_ir": os.path.isfile(paths["ir"]),
            "path_rgb": paths["rgb"],
            "path_depth": paths["depth"] if os.path.isfile(paths["depth"]) else None,
            "path_ir": paths["ir"] if os.path.isfile(paths["ir"]) else None,
            "path_meta": None,
        })
    records.sort(key=lambda r: (r["node"], r["id"]))
    return records, missing


# --------------------------------------------------------------------------- model
def adapt_single_channel(model):
    """1-channel first conv whose kernel is the sum of the 3-channel kernel over its input
    channels: exactly the 3-channel network fed the same image in all three channels."""
    import torch
    import torch.nn as nn

    old = model.features[0][0]
    new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.features[0][0] = new
    return model


def build_model(modality: str, pretrained: bool):
    from src.data.custom_dataset import MODALITY_CHANNELS
    from src.models.mobilenetv3_multimodal import create_model

    channels = MODALITY_CHANNELS[modality]
    if channels == 1 and pretrained:
        # create_model copies the ImageNet kernel into channels [:3], which a 1-channel
        # conv does not have; build the RGB network and fold its kernel instead.
        return adapt_single_channel(create_model(num_classes=2, in_channels=3, pretrained=True))
    return create_model(num_classes=2, in_channels=channels, pretrained=pretrained)


def _dataset(records, cfg, train: bool):
    from src.data.custom_dataset import CustomRGBDDataset
    return CustomRGBDDataset(root=cfg["data_dir"], ids=[r["uid"] for r in records],
                             modality=cfg["modality"], img_size=cfg["img_size"], train=train,
                             index=records, class_to_idx=CLASS_TO_IDX, seed=cfg["seed"],
                             preprocess="camera", cache=cfg.get("cache"))


def _loader(records, cfg, train: bool):
    import torch
    from torch.utils.data import DataLoader

    ds = _dataset(records, cfg, train)
    if not train:
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    # a last batch of one image would break BatchNorm in training mode: drop only that one
    drop_last = len(ds) % cfg["batch_size"] == 1
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
                      generator=g, drop_last=drop_last)


def predict(model, records, cfg, device) -> np.ndarray:
    """(N, 2) logits in record order."""
    import torch

    if not records:
        return np.zeros((0, 2), np.float32)
    out = []
    model.eval()
    with torch.no_grad():
        for x, _ in _loader(records, cfg, train=False):
            out.append(model(x.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def val_loss(model, records, cfg, device) -> float:
    """Mean cross-entropy over the validation frames."""
    from src.evaluation.predictions import margins, per_image_loss

    logits = predict(model, records, cfg, device)
    labels = np.array([r["label"] for r in records])
    return float(per_image_loss(labels, margins(logits)).mean())


def train_and_select(train_records, val_records, cfg, device):
    """Train ``cfg["epochs"]`` epochs; return the model of the selected epoch.

    The declared rule: lowest validation loss, earlier on ties.  This function is given
    the training and validation frames only; it never sees a test frame.
    -> (model, selected_epoch, history)
    """
    import torch
    import torch.nn as nn

    from scripts.train_local import set_seed, train_one_epoch
    from src.evaluation.model_selection import select_round

    if not train_records or not val_records:
        raise ValueError("empty training (%d) or validation (%d) set"
                         % (len(train_records), len(val_records)))
    set_seed(cfg["seed"])
    loader = _loader(train_records, cfg, train=True)
    model = build_model(cfg["modality"], cfg["pretrained"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()
    states, history = {}, []
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        vloss = val_loss(model, val_records, cfg, device)
        states[epoch] = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": round(float(train_loss), 6),
                        "val_loss": vloss, "epoch_time_s": round(time.perf_counter() - t0, 3)})
    selected = select_round({h["epoch"]: h["val_loss"] for h in history})
    if selected is None:
        raise RuntimeError("no epoch with a finite validation loss")
    model.load_state_dict(states[selected])
    return model, selected, history


# --------------------------------------------------------------------------- protocols
def loso_folds_for(splits_dir: str, labels_csv: str) -> List[Dict[str, object]]:
    """The committed LOSO folds, after checking they match the kept scenes of labels.csv."""
    scenes = kept_scenes(read_labels(labels_csv))
    try:
        status = check_manifests(scenes, splits_dir)
    except ValueError as exc:
        raise SystemExit("ERROR: %s" % exc)
    if status.get(LOSO_FILE) != "identical":
        raise SystemExit("ERROR: %s is %s for the kept scenes of %s -- run "
                         "scripts/camera_manifests.py (never edit the manifests by hand)"
                         % (os.path.join(splits_dir, LOSO_FILE), status.get(LOSO_FILE),
                            labels_csv))
    return read_loso_folds(os.path.join(splits_dir, LOSO_FILE))


def random_folds(records: Sequence[Dict[str, object]], k: int, seed: int) -> Dict[str, int]:
    """Frame id -> fold for the random frame-level split (ids shared across cameras)."""
    by_label: Dict[str, set] = {}
    for r in records:
        by_label.setdefault(str(r["label_name"]), set()).add(str(r["id"]))
    rng = np.random.default_rng(seed)
    out = {}
    for label in sorted(by_label):
        ids = np.array(sorted(by_label[label]), dtype=str)
        for i, fid in enumerate(rng.permutation(ids)):
            out[str(fid)] = i % k
    return out


def _run_folds(folds, index, cfg, device, source: str):
    """folds: list of (fold_info, train, val, {target: test records})."""
    preds = {t: {"id": [], "label": [], "logits": []} for t in target_nodes(cfg["modality"])}
    fold_log = []
    for info, train, val, tests in folds:
        t0 = time.perf_counter()
        model, selected, history = train_and_select(train, val, cfg, device)
        train_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        n_test = {}
        for target, recs in tests.items():
            logits = predict(model, recs, cfg, device)
            preds[target]["id"] += [r["id"] for r in recs]
            preds[target]["label"] += [r["label"] for r in recs]
            preds[target]["logits"].append(logits)
            n_test[target] = len(recs)
        entry = dict(info)
        entry.update({"n_train": len(train), "n_val": len(val), "n_test": n_test,
                      "selected_epoch": selected,
                      "val_loss_by_epoch": [h["val_loss"] for h in history],
                      "history": history, "train_time_s": round(train_time, 3),
                      "eval_time_s": round(time.perf_counter() - t1, 3)})
        fold_log.append(entry)
        print("    fold %-12s train %5d  val %4d  test %s  selected epoch %d  (%.0fs)"
              % (info.get("test_scene", info.get("fold")), len(train), len(val),
                 "/".join(str(n_test[t]) for t in n_test), selected, train_time))
        del model
    return preds, fold_log


def check_out_of_fold(preds, index, modality: str) -> None:
    """Every kept frame of every target predicted exactly once."""
    for target in target_nodes(modality):
        want = sorted(r["id"] for r in index if r["node"] == target)
        got = preds[target]["id"]
        if len(got) != len(set(got)):
            raise RuntimeError("%s: a frame was predicted more than once" % target)
        if sorted(got) != want:
            raise RuntimeError("%s: out-of-fold predictions do not cover the kept frames "
                               "exactly (%d predicted, %d kept)" % (target, len(got), len(want)))


def run_loso(index, folds, cfg, device, source: str):
    work = []
    for f in folds:
        test, val = f["test_scene"], f["val_scene"]
        train = [r for r in index if r["node"] == source and r["scene"] not in (test, val)]
        vrec = [r for r in index if r["node"] == source and r["scene"] == val]
        tests = {t: [r for r in index if r["node"] == t and r["scene"] == test]
                 for t in target_nodes(cfg["modality"])}
        info = {"fold": f["fold"], "test_scene": test, "val_scene": val,
                "train_scenes": sorted({r["scene"] for r in train})}
        work.append((info, train, vrec, tests))
    return _run_folds(work, index, cfg, device, source)


def run_random(index, cfg, device, source: str, k: int):
    assign = random_folds(index, k, cfg["seed"])
    work = []
    for fold in range(k):
        val_fold = (fold + 1) % k
        train = [r for r in index if r["node"] == source and assign[r["id"]] not in (fold, val_fold)]
        vrec = [r for r in index if r["node"] == source and assign[r["id"]] == val_fold]
        tests = {t: [r for r in index if r["node"] == t and assign[r["id"]] == fold]
                 for t in target_nodes(cfg["modality"])}
        work.append(({"fold": fold, "val_fold": val_fold}, train, vrec, tests))
    return _run_folds(work, index, cfg, device, source)


def write_oof(preds, out_dir: str, source: str, protocol: str) -> Dict[str, str]:
    from src.evaluation.predictions import FORMAT_VERSION, _sigmoid, margins

    written = {}
    for target, p in preds.items():
        order = np.argsort(np.array(p["id"], dtype=str), kind="mergesort")
        logits = np.concatenate(p["logits"]) if p["logits"] else np.zeros((0, 2), np.float32)
        m = margins(logits)[order]
        path = os.path.join(out_dir, oof_name(source, target, protocol))
        np.savez_compressed(path, path=np.array(p["id"], dtype=str)[order],
                            label=np.array(p["label"], dtype=np.uint8)[order], logit_margin=m,
                            p_fire=_sigmoid(m).astype(np.float32),
                            format_version=np.array(FORMAT_VERSION))
        written[target] = os.path.basename(path)
    return written


# --------------------------------------------------------------------------- run
def _sha256(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_commit() -> Dict[str, object]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=20)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=REPO, capture_output=True, text=True, timeout=20)
        if head.returncode != 0:
            return {"commit": None, "dirty": None}
        return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _complete(res: Dict[str, object], out_dir: str, source: str, modality: str,
              protocol: str) -> bool:
    return protocol in res and all(
        os.path.isfile(os.path.join(out_dir, oof_name(source, t, protocol)))
        for t in target_nodes(modality))


def run_one(source: str, modality: str, seed: int, args, device, folds, cache) -> str:
    out_dir = os.path.join(args.output_root, run_dir_name(modality, source, seed))
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "results.json")
    res: Dict[str, object] = {}
    if os.path.isfile(res_path):
        with open(res_path, encoding="utf-8") as f:
            res = json.load(f)
    index, missing = camera_index(args.data_dir, args.labels_csv, modality)
    cfg = {"data_dir": args.data_dir, "modality": modality, "seed": seed,
           "img_size": args.img_size, "batch_size": args.batch_size, "epochs": args.epochs,
           "lr": args.lr, "pretrained": not args.no_pretrained, "cache": cache}
    res.update({
        "experiment": EXPERIMENT, "results_schema_version": RESULTS_SCHEMA_VERSION,
        "config": {"source": source, "source_sensor": SENSORS[source], "modality": modality,
                   "seed": seed, "targets": list(target_nodes(modality)),
                   "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                   "optimizer": "Adam", "img_size": args.img_size,
                   "pretrained": not args.no_pretrained, "preprocess": "camera_preprocess",
                   "selection": "lowest validation loss on the source's validation-scene "
                                "frames, earlier on ties",
                   "data_dir": args.data_dir, "labels_csv": args.labels_csv,
                   "labels_csv_sha256": _sha256(args.labels_csv),
                   "loso_folds_sha256": _sha256(os.path.join(args.splits_dir, LOSO_FILE)),
                   "random_folds": args.random_folds},
        "n_frames": {t: sum(1 for r in index if r["node"] == t) for t in target_nodes(modality)},
        "n_valid_frames_missing_modality_stream": missing,
    })
    for protocol in args.protocols:
        if _complete(res, out_dir, source, modality, protocol) and not args.force:
            print("  %s %s: complete, skipped" % (os.path.basename(out_dir), protocol))
            continue
        print("  %s %s" % (os.path.basename(out_dir), protocol))
        t0 = time.perf_counter()
        if protocol == "loso":
            preds, log = run_loso(index, folds, cfg, device, source)
        else:
            preds, log = run_random(index, cfg, device, source, args.random_folds)
        check_out_of_fold(preds, index, modality)
        written = write_oof(preds, out_dir, source, protocol)
        import torch
        res[protocol] = {"folds": log, "files": written,
                         "total_time_s": round(time.perf_counter() - t0, 3),
                         "device": str(device), "hostname": socket.gethostname(),
                         "torch": torch.__version__, "python": platform.python_version(),
                         "timestamp": datetime.now().isoformat(), **git_commit()}
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", default=os.path.join("data", "raw", "camera"))
    p.add_argument("--labels_csv", default=None, help="default: <data_dir>/labels.csv")
    p.add_argument("--splits_dir", default=os.path.join("data", "splits_camera"))
    p.add_argument("--sources", nargs="+", default=list(NODES), choices=list(NODES))
    p.add_argument("--modalities", nargs="+", default=["rgb"], choices=list(MODALITIES))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--protocols", nargs="+", default=["loso"], choices=["loso", "random"])
    p.add_argument("--random_folds", type=int, default=5,
                   help="K of the descriptive frame-level random split")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=224, help="224 = prereg; smaller only for tests")
    p.add_argument("--no_pretrained", action="store_true",
                   help="random init instead of ImageNet weights (tests only)")
    p.add_argument("--output_root", default=os.path.join("results", "camera_a"))
    p.add_argument("--force", action="store_true", help="re-run complete runs")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    import torch

    args = build_parser().parse_args(argv)
    args.labels_csv = args.labels_csv or os.path.join(args.data_dir, "labels.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = loso_folds_for(args.splits_dir, args.labels_csv)
    print("camera LOSO: %d folds, device %s, modalities %s, seeds %s"
          % (len(folds), device, args.modalities, args.seeds))
    cache: Dict = {}
    for modality in args.modalities:
        for source in args.sources:
            if source not in target_nodes(modality):
                print("  %s has no %s stream, skipped" % (source, modality))
                continue
            for seed in args.seeds:
                run_one(source, modality, seed, args, device, folds, cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
