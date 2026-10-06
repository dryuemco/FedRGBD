"""
FedRGBD — Captured Data Splitter for Cross-Sensor Experiments (v2)
==================================================================
Handles two capture formats:
  - RealSense (Node A, B): flat files — 00000_rgb.png, 00000_depth.png, 00000_ir.png
  - ZED (Node C): subdirectories — rgb/000001.png, depth/000001.png

Output structure (matches FlameDataset format):
    data/processed/captures/node_a/
        train/Fire/    train/No_Fire/
        test/Fire/     test/No_Fire/
        val/Fire/      val/No_Fire/    (copy of test)

Usage:
    python3 scripts/prepare_captures.py --base_dir data/raw/captures --output_dir data/processed/captures
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path


def find_rgb_frames(scene_dir):
    """Find all RGB frames in a scene directory, handling both formats.
    
    Returns list of Path objects pointing to RGB PNG files.
    """
    scene_dir = Path(scene_dir)
    
    # Format 1: ZED — rgb/ subdirectory
    rgb_subdir = scene_dir / "rgb"
    if rgb_subdir.exists():
        return sorted(rgb_subdir.glob("*.png"))
    
    # Format 2: RealSense — flat XXXXX_rgb.png files
    flat_files = sorted(scene_dir.glob("*_rgb.png"))
    if flat_files:
        return flat_files
    
    return []


def prepare_node(node_dir, output_dir, train_ratio=0.8, seed=42):
    """Split a single node's captures into train/test."""
    random.seed(seed)
    node_name = node_dir.name
    out = Path(output_dir) / node_name

    # Collect all RGB frames with labels
    all_frames = []

    for scene_dir in sorted(node_dir.iterdir()):
        if not scene_dir.is_dir():
            continue

        rgb_files = find_rgb_frames(scene_dir)
        if not rgb_files:
            print(f"  WARNING: No RGB frames in {scene_dir.name}, skipping")
            continue

        # Determine class from directory name
        dirname = scene_dir.name.lower()
        if "_fire_" in dirname and "_no_fire_" not in dirname:
            class_label = "Fire"
        elif "_no_fire_" in dirname:
            class_label = "No_Fire"
        else:
            print(f"  WARNING: Cannot determine class for {scene_dir.name}, skipping")
            continue

        for f in rgb_files:
            all_frames.append((f, class_label, scene_dir.name))

        print(f"    {scene_dir.name}: {len(rgb_files)} frames → {class_label}")

    if not all_frames:
        print(f"  ERROR: No frames found for {node_name}")
        return

    # Split per scene to ensure balanced representation
    scenes = {}
    for path, label, scene_id in all_frames:
        if scene_id not in scenes:
            scenes[scene_id] = []
        scenes[scene_id].append((path, label))

    train_frames = []
    test_frames = []

    for scene_id, frames in scenes.items():
        random.shuffle(frames)
        n_train = int(len(frames) * train_ratio)
        train_frames.extend(frames[:n_train])
        test_frames.extend(frames[n_train:])

    # Create output directories
    for split in ["train", "test"]:
        for cls in ["Fire", "No_Fire"]:
            (out / split / cls).mkdir(parents=True, exist_ok=True)

    # Copy files
    def copy_frames(frames, split_name):
        counts = {"Fire": 0, "No_Fire": 0}
        for src_path, label in frames:
            counts[label] += 1
            # Unique name: scene + original filename
            parent_scene = src_path.parent.name
            # If parent is "rgb" subdir, go one more level up for scene name
            if parent_scene == "rgb":
                parent_scene = src_path.parent.parent.name
            new_name = f"{parent_scene}_{src_path.name}"
            dst = out / split_name / label / new_name
            shutil.copy2(src_path, dst)
        return counts

    train_counts = copy_frames(train_frames, "train")
    test_counts = copy_frames(test_frames, "test")

    print(f"  {node_name}: train={sum(train_counts.values())} "
          f"(Fire={train_counts['Fire']}, No_Fire={train_counts['No_Fire']}), "
          f"test={sum(test_counts.values())} "
          f"(Fire={test_counts['Fire']}, No_Fire={test_counts['No_Fire']})")

    # Create val split (copy of test for FlameDataset compatibility)
    val_dir = out / "val"
    if val_dir.exists():
        shutil.rmtree(val_dir)
    shutil.copytree(out / "test", val_dir)

    # Save split metadata
    meta = {
        "node": node_name,
        "total_frames": len(all_frames),
        "train": sum(train_counts.values()),
        "test": sum(test_counts.values()),
        "train_counts": train_counts,
        "test_counts": test_counts,
        "scenes": list(scenes.keys()),
        "train_ratio": train_ratio,
        "seed": seed,
    }
    with open(out / "split_info.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="data/raw/captures",
                        help="Base directory with node_a/, node_b/, node_c/ captures")
    parser.add_argument("--output_dir", default="data/processed/captures",
                        help="Output directory for train/test splits")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = Path(args.base_dir)
    print("=" * 60)
    print("  FedRGBD — Prepare Captured Data for Cross-Sensor Experiments")
    print(f"  Source: {base}")
    print(f"  Output: {args.output_dir}")
    print(f"  Train/Test ratio: {args.train_ratio}/{1-args.train_ratio:.1f}")
    print("=" * 60)

    for node_id in ["node_a", "node_b", "node_c"]:
        node_dir = base / node_id
        if not node_dir.exists():
            print(f"  {node_id}: NOT FOUND, skipping")
            continue
        print(f"\n  Processing {node_id}:")
        prepare_node(node_dir, args.output_dir, args.train_ratio, args.seed)

    print(f"\nDone. Data ready at: {args.output_dir}/")


if __name__ == "__main__":
    main()
