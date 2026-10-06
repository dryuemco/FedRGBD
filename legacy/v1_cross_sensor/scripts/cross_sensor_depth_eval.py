"""
FedRGBD — Cross-Sensor Generalization: Depth-Only
====================================================
Same as cross_sensor_eval.py but uses depth maps instead of RGB.
This is where sensor heterogeneity should manifest.

Usage:
    python3 scripts/cross_sensor_depth_eval.py --all --seed 42 \
        --output_dir results/cross_sensor_depth
"""

import argparse
import json
import os
import random
import socket
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, ".")
from src.data.depth_dataset import DepthDataset
from src.models.mobilenetv3_multimodal import create_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for p, l in zip(predicted, labels):
                if l == 1 and p == 1: tp += 1
                elif l == 0 and p == 1: fp += 1
                elif l == 0 and p == 0: tn += 1
                elif l == 1 and p == 0: fn += 1

    accuracy = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "loss": round(total_loss / max(total, 1), 6),
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total,
    }


def train_and_evaluate(train_dir, train_name, test_dirs, test_names,
                       epochs, batch_size, lr, seed, device):
    set_seed(seed)

    train_ds = DepthDataset(train_dir, split="train")
    val_ds = DepthDataset(train_dir, split="val")

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    print(f"  Train data ({train_name}): {len(train_ds)} depth samples")
    print(f"  Class distribution: {train_ds.get_class_distribution()}")

    # Use pretrained=False since depth input distribution is very different from ImageNet
    model = create_model(num_classes=2, in_channels=3, pretrained=False)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_history = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.train()
        epoch_loss = 0
        epoch_samples = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
            epoch_samples += images.size(0)

        val_result = evaluate(model, val_loader, criterion, device)
        train_history.append({
            "epoch": epoch,
            "train_loss": round(epoch_loss / max(epoch_samples, 1), 6),
            "val_accuracy": val_result["accuracy"],
        })
        print(f"    Epoch {epoch}/{epochs}: train_loss={epoch_loss/max(epoch_samples,1):.4f}, "
              f"val_acc={val_result['accuracy']:.4f}")

    train_time = time.perf_counter() - start

    # Cross-sensor evaluation
    print(f"\n  Cross-sensor evaluation (trained on {train_name} depth):")
    cross_results = {}

    for test_dir, test_name in zip(test_dirs, test_names):
        test_ds = DepthDataset(test_dir, split="test")
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=0, pin_memory=False)
        result = evaluate(model, test_loader, criterion, device)
        cross_results[test_name] = result

        marker = " <-- same sensor" if test_name == train_name else ""
        print(f"    -> {test_name}: acc={result['accuracy']:.4f}, "
              f"f1={result['f1']:.4f}, n={result['total']}{marker}")

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "train_sensor": train_name,
        "modality": "depth_only",
        "train_samples": len(train_ds),
        "train_time_s": round(train_time, 2),
        "train_history": train_history,
        "cross_results": cross_results,
    }


def run_full_matrix(base_dir, output_dir, epochs, batch_size, lr, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hostname = socket.gethostname()

    nodes = {
        "D435if": os.path.join(base_dir, "node_a"),
        "D435i": os.path.join(base_dir, "node_b"),
        "ZED2i": os.path.join(base_dir, "node_c"),
    }

    for name, path in nodes.items():
        if not os.path.exists(os.path.join(path, "train")):
            print(f"ERROR: {path}/train not found. Run prepare_depth_captures.py first.")
            return

    test_dirs = list(nodes.values())
    test_names = list(nodes.keys())

    print("=" * 60)
    print(f"  FedRGBD — Cross-Sensor Depth-Only Experiment")
    print(f"  Host: {hostname}, Device: {device}")
    print(f"  Sensors: {', '.join(nodes.keys())}")
    print(f"  Modality: DEPTH ONLY")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, Seed: {seed}")
    print("=" * 60)

    all_results = {}
    total_start = time.perf_counter()

    for train_name, train_dir in nodes.items():
        print(f"\n{'='*60}")
        print(f"  TRAINING ON: {train_name} (depth)")
        print(f"{'='*60}")

        result = train_and_evaluate(
            train_dir=train_dir,
            train_name=train_name,
            test_dirs=test_dirs,
            test_names=test_names,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            device=device,
        )
        all_results[train_name] = result

    total_time = time.perf_counter() - total_start

    # Build matrix
    matrix = {}
    for train_name in nodes:
        matrix[train_name] = {}
        for test_name in nodes:
            matrix[train_name][test_name] = all_results[train_name]["cross_results"][test_name]["accuracy"]

    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "experiment": "cross_sensor_depth_only",
        "hostname": hostname,
        "seed": seed,
        "epochs": epochs,
        "total_time_s": round(total_time, 2),
        "accuracy_matrix": matrix,
        "detailed_results": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print matrix
    print(f"\n{'='*60}")
    print("  DEPTH-ONLY CROSS-SENSOR ACCURACY MATRIX")
    print(f"{'='*60}")
    header = "Train / Test"
    print(f"{header:<12}", end="")
    for test_name in nodes:
        print(f"{test_name:>10}", end="")
    print()
    print("-" * 42)
    for train_name in nodes:
        print(f"{train_name:<12}", end="")
        for test_name in nodes:
            acc = matrix[train_name][test_name]
            marker = "*" if train_name == test_name else " "
            print(f"{acc*100:>9.2f}{marker}", end="")
        print()
    print("  * = same sensor (train & test)")

    same_sensor = [matrix[s][s] for s in nodes]
    diff_sensor = [matrix[t][s] for t in nodes for s in nodes if t != s]
    gap = np.mean(same_sensor) - np.mean(diff_sensor)

    print(f"\n  Same-sensor mean:  {np.mean(same_sensor)*100:.2f}%")
    print(f"  Cross-sensor mean: {np.mean(diff_sensor)*100:.2f}%")
    print(f"  Generalization gap: {gap*100:.2f} pp")
    print(f"\n  Results: {results_path}")
    print(f"  Total time: {total_time:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--base_dir", default="data/processed/captures_depth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/cross_sensor_depth")
    args = parser.parse_args()

    if args.all:
        run_full_matrix(
            base_dir=args.base_dir,
            output_dir=os.path.join(args.output_dir, f"seed{args.seed}"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
    else:
        print("Use --all to run the full 3x3 depth cross-sensor matrix")


if __name__ == "__main__":
    main()
