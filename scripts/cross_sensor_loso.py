"""FedRGBD — Leave-One-Scene-Out (LOSO) cross-sensor evaluation.

Reviewer 3 (NCAA-D-26-02211) observed that the cross-camera experiments split
frames of the *same* five scenes between training and testing, so the reported
accuracy may partly reflect scene similarity rather than cross-sensor
generalisation.  This script provides the scene-independent protocol:

    for each scene S:
        train on   {train_nodes} x {all scenes except S}
        evaluate on {test_nodes}  x {S}          -> cross-camera LOSO
        evaluate on {train_nodes} x {S}          -> same-camera LOSO

and reports the full metric set per fold plus mean +/- std over folds.  With
``--pooled_random_split`` it additionally re-runs the *old* frame-level random
split on exactly the same data, so the paper can put random-split and LOSO
numbers side by side in one table.

Examples
--------
Cross-camera (train on the two RealSense nodes, test on the ZED node):

  python3 scripts/cross_sensor_loso.py \
      --data_dir data/raw/custom --labels_csv data/raw/custom/labels.csv \
      --train_nodes node_a node_b --test_nodes node_c \
      --modality rgb_d --epochs 15 --batch_size 8 --lr 1e-3 --seed 42 \
      --pooled_random_split \
      --output_dir results/loso_ab_to_c_rgb_d_seed42

Same-camera control (scene-independent, but no sensor shift):

  python3 scripts/cross_sensor_loso.py \
      --data_dir data/raw/custom --labels_csv data/raw/custom/labels.csv \
      --train_nodes node_a --same_node --modality rgb_d \
      --output_dir results/loso_a_same_rgb_d_seed42
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_local import json_metrics, set_seed, train_one_epoch  # noqa: E402
from src.data.custom_dataset import (  # noqa: E402
    MODALITIES,
    MODALITY_CHANNELS,
    CustomRGBDDataset,
    build_label_map,
    class_distribution,
    filter_index,
    load_frame_index,
    scenes_in,
)
from src.evaluation.metrics import METRIC_KEYS, MetricAccumulator, format_metrics  # noqa: E402

RESULTS_SCHEMA_VERSION = 2

#: metrics aggregated as mean +/- std across folds
SUMMARY_METRIC_KEYS: List[str] = list(METRIC_KEYS) + ["loss"]


def evaluate(model, data_loader, criterion, device, num_classes: int = 2):
    """Same contract as ``scripts.train_local.evaluate`` (``(loss, acc, metrics)``),
    with ``num_classes`` exposed so a non-binary label vocabulary also works."""
    sum_criterion = nn.CrossEntropyLoss(reduction="sum")
    acc = MetricAccumulator(num_classes=num_classes, positive_class=1)
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            acc.update(outputs, labels, sum_criterion(outputs, labels).item())

    metrics = acc.compute()
    avg_loss = float(metrics["loss"]) if metrics["loss"] is not None else 0.0
    accuracy = float(metrics["accuracy"]) if metrics["accuracy"] is not None else 0.0
    metrics["loss"] = avg_loss
    return avg_loss, accuracy, metrics


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _uids(records: Sequence[dict]) -> List[str]:
    return [str(r["uid"]) for r in records]


def _make_loader(records, cfg, train: bool, shuffle: bool):
    """DataLoader over a subset of the frame index (Jetson-safe settings)."""
    ds = CustomRGBDDataset(
        root=cfg["data_dir"],
        ids=_uids(records),
        modality=cfg["modality"],
        img_size=cfg["img_size"],
        train=train,
        index=cfg["index"],
        class_to_idx=cfg["class_to_idx"],
        max_depth_m=cfg["max_depth_m"],
        seed=cfg["seed"],
    )
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(cfg["seed"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle,
                        num_workers=0, pin_memory=False, generator=generator)
    return ds, loader


def _train_model(train_records, cfg, device):
    """Fresh model + Adam/CrossEntropy loop, mirroring scripts/train_local.py."""
    from src.models.mobilenetv3_multimodal import create_model

    set_seed(cfg["seed"])
    _, train_loader = _make_loader(train_records, cfg, train=True, shuffle=True)

    model = create_model(
        num_classes=cfg["num_classes"],
        in_channels=MODALITY_CHANNELS[cfg["modality"]],
        pretrained=cfg["pretrained"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    history = []
    start = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        epoch_time = time.perf_counter() - epoch_start
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "epoch_time_s": round(epoch_time, 3),
            "elapsed_s": round(time.perf_counter() - start, 3),
        })
        print(f"      epoch {epoch:2d}/{cfg['epochs']}: "
              f"train_loss={train_loss:.4f}, time={epoch_time:.1f}s")

    return model, criterion, history, time.perf_counter() - start


def _eval_split(model, criterion, records, cfg, device, name, nodes):
    """Evaluate on a frame subset; empty subsets are recorded, not crashed on."""
    entry: Dict[str, object] = {
        "split": name,
        "nodes": list(nodes),
        "n_examples": len(records),
        "class_distribution": class_distribution(records),
        "ids": _uids(records) if cfg["record_ids"] else None,
    }
    if not records:
        entry.update({"status": "empty", "loss": None, "accuracy": None, "metrics": None,
                      "eval_time_s": 0.0})
        return entry
    _, loader = _make_loader(records, cfg, train=False, shuffle=False)
    t0 = time.perf_counter()
    loss, acc, metrics = evaluate(model, loader, criterion, device, cfg["num_classes"])
    entry.update({
        "status": "ok",
        "loss": round(loss, 6),
        "accuracy": round(acc, 6),
        "metrics": json_metrics(metrics),
        "eval_time_s": round(time.perf_counter() - t0, 3),
    })
    print(f"      {name:<13} ({len(records):4d} frames): "
          f"loss={loss:.4f}, {format_metrics(metrics)}")
    return entry


def _aggregate(folds: Sequence[dict], split: str) -> Dict[str, object]:
    """mean/std/min/max over the folds whose ``split`` evaluation is non-empty."""
    per_metric: Dict[str, List[float]] = {}
    used = []
    total_examples = 0
    for fold in folds:
        entry = fold["eval"].get(split) or {}
        if entry.get("status") != "ok":
            continue
        used.append(fold["held_out_scene"])
        total_examples += int(entry["n_examples"])
        metrics = entry["metrics"] or {}
        for key in SUMMARY_METRIC_KEYS:
            val = metrics.get(key)
            if isinstance(val, (int, float)):
                per_metric.setdefault(key, []).append(float(val))

    out: Dict[str, object] = {
        "n_folds": len(used),
        "scenes": used,
        "total_eval_examples": total_examples,
    }
    for key, values in per_metric.items():
        arr = np.asarray(values, dtype=np.float64)
        out[key] = {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "values": [round(v, 6) for v in values],
        }
    return out


# --------------------------------------------------------------------------- #
# protocols
# --------------------------------------------------------------------------- #
def run_loso(index, cfg, device) -> List[dict]:
    """One fold per scene: train on the other scenes, test on the held-out one."""
    folds = []
    for i, scene in enumerate(cfg["scenes"], start=1):
        train_records = [r for r in index
                         if r["node"] in cfg["train_nodes"] and r["scene"] != scene]
        cross_records = [r for r in index
                         if r["node"] in cfg["test_nodes"] and r["scene"] == scene]
        same_records = [r for r in index
                        if r["node"] in cfg["train_nodes"] and r["scene"] == scene]

        print(f"\n  [fold {i}/{len(cfg['scenes'])}] held-out scene: {scene}")
        print(f"    train: {len(train_records)} frames from "
              f"{sorted({r['node'] for r in train_records})}, "
              f"scenes={sorted({r['scene'] for r in train_records})}")
        if not train_records:
            raise ValueError(
                f"fold {scene!r}: no training frames left for nodes "
                f"{cfg['train_nodes']} — LOSO needs at least two scenes per "
                "training node."
            )

        fold_start = time.perf_counter()
        model, criterion, history, train_time = _train_model(train_records, cfg, device)
        eval_entries = {
            "cross_camera": _eval_split(model, criterion, cross_records, cfg, device,
                                        "cross_camera", cfg["test_nodes"]),
            "same_camera": _eval_split(model, criterion, same_records, cfg, device,
                                       "same_camera", cfg["train_nodes"]),
        }
        fold = {
            "fold": i,
            "held_out_scene": scene,
            "train_nodes": list(cfg["train_nodes"]),
            "test_nodes": list(cfg["test_nodes"]),
            "train_scenes": sorted({str(r["scene"]) for r in train_records}),
            "n_train": len(train_records),
            "train_class_distribution": class_distribution(train_records),
            "train_ids": _uids(train_records) if cfg["record_ids"] else None,
            "eval": eval_entries,
            "history": history,
            "train_time_s": round(train_time, 3),
            "eval_time_s": round(sum(float(e["eval_time_s"]) for e in eval_entries.values()), 3),
            "fold_time_s": round(time.perf_counter() - fold_start, 3),
        }
        folds.append(fold)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return folds


def pooled_random_split(index, cfg, device) -> dict:
    """The *old* (leaky) protocol: a frame-level random split over all scenes.

    Frames of every scene appear in both the training and the test part, which
    is exactly what Reviewer 3 flagged.  The split is stratified by
    ``(node, label)`` so both classes stay present on the small custom capture.
    """
    rng = np.random.RandomState(cfg["seed"])
    groups: Dict[tuple, List[dict]] = {}
    for r in index:
        groups.setdefault((r["node"], r["label_name"]), []).append(r)

    train_part, test_part = [], []
    for key in sorted(groups):
        recs = sorted(groups[key], key=lambda r: r["uid"])
        order = rng.permutation(len(recs))
        n_test = max(1, int(round(len(recs) * cfg["test_frac"]))) if len(recs) > 1 else 0
        for pos, j in enumerate(order):
            (test_part if pos < n_test else train_part).append(recs[j])

    train_records = [r for r in train_part if r["node"] in cfg["train_nodes"]]
    cross_records = [r for r in test_part if r["node"] in cfg["test_nodes"]]
    same_records = [r for r in test_part if r["node"] in cfg["train_nodes"]]

    print(f"\n  [pooled random split] test_frac={cfg['test_frac']} "
          f"(frames of every scene appear in train AND test)")
    print(f"    train: {len(train_records)} frames, "
          f"cross-camera test: {len(cross_records)}, same-camera test: {len(same_records)}")
    if not train_records:
        raise ValueError("pooled random split: empty training set")

    start = time.perf_counter()
    model, criterion, history, train_time = _train_model(train_records, cfg, device)
    eval_entries = {
        "cross_camera": _eval_split(model, criterion, cross_records, cfg, device,
                                    "cross_camera", cfg["test_nodes"]),
        "same_camera": _eval_split(model, criterion, same_records, cfg, device,
                                   "same_camera", cfg["train_nodes"]),
    }
    result = {
        "protocol": "frame_level_random_split",
        "description": ("old protocol: frames from the same scenes are divided "
                        "between training and testing (scene leakage)"),
        "test_frac": cfg["test_frac"],
        "stratified_by": ["node", "label"],
        "train_nodes": list(cfg["train_nodes"]),
        "test_nodes": list(cfg["test_nodes"]),
        "scenes": list(cfg["scenes"]),
        "n_train": len(train_records),
        "train_class_distribution": class_distribution(train_records),
        "train_ids": _uids(train_records) if cfg["record_ids"] else None,
        "eval": eval_entries,
        "history": history,
        "train_time_s": round(train_time, 3),
        "total_time_s": round(time.perf_counter() - start, 3),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Leave-one-scene-out cross-sensor evaluation on the custom "
                    "RGB-D captures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", default="data/raw/custom",
                   help="Custom capture root containing node_* directories")
    p.add_argument("--labels_csv", default=None,
                   help="CSV with columns id,scene,label (optional node column); "
                        "only needed when {id}_meta.json has no scene/label keys "
                        "and the frame ids do not follow <scene>_<label>_<n>")
    p.add_argument("--train_nodes", nargs="+", default=["node_a"],
                   help="Node directories used for training")
    p.add_argument("--test_nodes", nargs="+", default=None,
                   help="Node directories used for the cross-camera evaluation")
    p.add_argument("--same_node", action="store_true",
                   help="Same-camera LOSO: test nodes == train nodes")
    p.add_argument("--modality", default="rgb_d", choices=MODALITIES)
    p.add_argument("--scenes", nargs="+", default=None,
                   help="Restrict the evaluation to this subset of scenes")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img_size", type=int, default=224,
                   help="Input resolution (224 = paper setting; smaller only for tests)")
    p.add_argument("--max_depth_m", type=float, default=10.0,
                   help="Depth clipping range in metres before normalisation")
    p.add_argument("--no_pretrained", action="store_true",
                   help="Random init instead of ImageNet weights (CPU unit tests only)")
    p.add_argument("--pooled_random_split", action="store_true",
                   help="Additionally run the old frame-level random split on the "
                        "same data, for a random-split vs LOSO comparison")
    p.add_argument("--test_frac", type=float, default=0.2,
                   help="Test fraction of the pooled random-split baseline")
    p.add_argument("--no_record_ids", action="store_true",
                   help="Do not store per-fold frame id lists in results.json")
    p.add_argument("--output_dir", default="results/loso",
                   help="Directory for results.json (use results/loso_<...> for real runs)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.same_node:
        test_nodes = list(args.train_nodes)
    elif args.test_nodes:
        test_nodes = list(args.test_nodes)
    else:
        raise SystemExit("ERROR: pass --test_nodes node_x [...] or --same_node")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hostname = socket.gethostname()
    involved = sorted(set(args.train_nodes) | set(test_nodes))

    print("=" * 66)
    print("  FedRGBD — Leave-One-Scene-Out cross-sensor evaluation")
    print(f"  Host: {hostname}, Device: {device}")
    print(f"  Data: {args.data_dir}  modality={args.modality}")
    print(f"  Train nodes: {args.train_nodes}   Test nodes: {test_nodes}"
          f"{'  (same-camera)' if args.same_node else ''}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}, "
          f"Seed: {args.seed}, img_size: {args.img_size}")
    print("=" * 66)

    full_index = load_frame_index(args.data_dir, args.labels_csv)
    selected = filter_index(full_index, nodes=involved, scenes=args.scenes)
    if not selected:
        raise SystemExit(
            f"ERROR: no frames for nodes {involved}"
            + (f" and scenes {args.scenes}" if args.scenes else "")
            + f". Available nodes: {sorted({r['node'] for r in full_index})}, "
              f"scenes: {scenes_in(full_index)}"
        )

    usable = filter_index(selected, modality=args.modality)
    dropped = len(selected) - len(usable)
    if dropped:
        by_node: Dict[str, int] = {}
        usable_uids = {r["uid"] for r in usable}
        for r in selected:
            if r["uid"] not in usable_uids:
                by_node[str(r["node"])] = by_node.get(str(r["node"]), 0) + 1
        print(f"  WARNING: {dropped} frame(s) dropped — missing streams for "
              f"modality {args.modality!r}: {by_node} "
              "(the ZED node has no IR stream)")
    if not usable:
        raise SystemExit(
            f"ERROR: no frame provides every stream needed by modality "
            f"{args.modality!r} for nodes {involved}"
        )

    scenes = args.scenes if args.scenes else scenes_in(usable)
    scenes = [s for s in scenes if s in set(scenes_in(usable))]
    if len(scenes) < 2:
        raise SystemExit(
            f"ERROR: leave-one-scene-out needs >= 2 scenes, found {scenes}. "
            "Check the scene annotation (meta.json / --labels_csv)."
        )

    class_to_idx = build_label_map(r["label_name"] for r in usable)
    num_classes = max(2, len(set(class_to_idx.values())))

    print(f"  Frames: {len(usable)} usable / {len(selected)} selected")
    print(f"  Scenes ({len(scenes)}): {scenes}")
    print(f"  Classes: {class_to_idx}  distribution={class_distribution(usable)}")
    for node in involved:
        sub = filter_index(usable, nodes=[node])
        print(f"    {node}: {len(sub)} frames, scenes={scenes_in(sub)}, "
              f"classes={class_distribution(sub)}")

    cfg = {
        "data_dir": args.data_dir,
        "index": usable,
        "modality": args.modality,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "max_depth_m": args.max_depth_m,
        "pretrained": not args.no_pretrained,
        "class_to_idx": class_to_idx,
        "num_classes": num_classes,
        "train_nodes": list(args.train_nodes),
        "test_nodes": test_nodes,
        "scenes": scenes,
        "test_frac": args.test_frac,
        "record_ids": not args.no_record_ids,
    }

    total_start = time.perf_counter()
    folds = run_loso(usable, cfg, device)
    loso_time = time.perf_counter() - total_start

    summary: Dict[str, object] = {
        "n_folds": len(folds),
        "scenes": list(scenes),
        "train_nodes": list(args.train_nodes),
        "test_nodes": test_nodes,
        "same_node": bool(args.same_node),
        "modality": args.modality,
        "n_frames": len(usable),
        "cross_camera": _aggregate(folds, "cross_camera"),
        "same_camera": _aggregate(folds, "same_camera"),
        "loso_time_s": round(loso_time, 2),
    }

    results: Dict[str, object] = {
        "experiment": "cross_sensor_loso",
        "protocol": "leave_one_scene_out",
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "hostname": hostname,
        "device": str(device),
        "config": {
            "data_dir": args.data_dir,
            "labels_csv": args.labels_csv,
            "train_nodes": list(args.train_nodes),
            "test_nodes": test_nodes,
            "same_node": bool(args.same_node),
            "modality": args.modality,
            "in_channels": MODALITY_CHANNELS[args.modality],
            "scenes": list(scenes),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "img_size": args.img_size,
            "max_depth_m": args.max_depth_m,
            "pretrained": not args.no_pretrained,
            "num_classes": num_classes,
            "class_to_idx": class_to_idx,
            "pooled_random_split": bool(args.pooled_random_split),
            "test_frac": args.test_frac,
        },
        "label_sources": {
            src: sum(1 for r in usable if r["label_source"] == src)
            for src in sorted({str(r["label_source"]) for r in usable})
        },
        "n_frames": len(usable),
        "n_frames_dropped_for_modality": dropped,
        "class_distribution": class_distribution(usable),
        "folds": folds,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }

    if args.pooled_random_split:
        baseline = pooled_random_split(usable, cfg, device)
        results["pooled_random_split"] = baseline
        summary["pooled_random_split"] = {
            split: {
                "n_examples": baseline["eval"][split]["n_examples"],
                "accuracy": baseline["eval"][split]["accuracy"],
                "metrics": baseline["eval"][split]["metrics"],
            }
            for split in ("cross_camera", "same_camera")
        }

    summary["total_time_s"] = round(time.perf_counter() - total_start, 2)
    results["total_time_s"] = summary["total_time_s"]

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 66}")
    print(f"  LOSO complete — {len(folds)} folds in {summary['total_time_s']:.1f}s")
    for split in ("cross_camera", "same_camera"):
        agg = summary[split]
        if agg.get("accuracy"):
            print(f"    {split:<13} accuracy = {agg['accuracy']['mean']:.4f} "
                  f"+/- {agg['accuracy']['std']:.4f} over {agg['n_folds']} folds")
    if args.pooled_random_split:
        prs = summary["pooled_random_split"]["cross_camera"]
        print(f"    pooled random split (old protocol) cross-camera accuracy = "
              f"{prs['accuracy']}")
    print(f"  Results: {results_path}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
