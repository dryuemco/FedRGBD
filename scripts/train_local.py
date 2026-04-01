"""FedRGBD — Local-Only Training Baseline (Lower Bound).

Each node trains independently on its own data without any federation.
Run this script on EACH node separately (or on one node iterating over data dirs).

Usage (run on Node A for all nodes' data):
  python3 scripts/train_local.py \
    --data_dir data/processed/iid/node_a --node_name node_a \
    --seed 42 --output_dir results/local_iid_seed42

  python3 scripts/train_local.py \
    --data_dir data/processed/non_iid_label/node_a --node_name node_a \
    --seed 42 --output_dir results/local_noniid_seed42

Or use the batch runner to train all nodes sequentially on one device:
  python3 scripts/train_local.py --batch \
    --data_dirs data/processed/iid/node_a data/processed/iid/node_b data/processed/iid/node_c \
    --seed 42 --output_dir results/local_iid_seed42
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
from src.data.dataset import FlameDataset
from src.models.mobilenetv3_multimodal import create_model


def set_seed(seed):
    """Set all random seeds for reproducibility."""
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
    """Evaluate model on given data loader."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / max(total_samples, 1)


def train_single_node(data_dir, node_name, epochs, batch_size, lr, seed, output_dir,
                      cross_eval_dirs=None):
    """Train a model on a single node's data."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hostname = socket.gethostname()

    print(f"\n{'=' * 60}")
    print(f"  Local Training — {node_name}")
    print(f"  Host: {hostname}, Device: {device}")
    print(f"  Data: {data_dir}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}, Seed: {seed}")
    print(f"{'=' * 60}")

    # Load data
    train_ds = FlameDataset(data_dir, split="train")
    val_ds = FlameDataset(data_dir, split="val")
    test_ds = FlameDataset(data_dir, split="test")

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Class distribution: {train_ds.get_class_distribution()}")

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=False,
                              generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0, pin_memory=False)

    # Create fresh model (same init for each node with same seed)
    model = create_model(num_classes=2, in_channels=3, pretrained=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = []
    start_total = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.perf_counter() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "epoch_time_s": round(epoch_time, 1),
        }
        history.append(record)

        marker = " ← FL Round equivalent" if epoch % 5 == 0 else ""
        print(f"  Epoch {epoch:2d}/{epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_acc={val_acc:.4f}, time={epoch_time:.1f}s{marker}")

    total_time = time.perf_counter() - start_total

    # Final test evaluation (on own test set)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n  {node_name} Test (own data): loss={test_loss:.4f}, accuracy={test_acc:.4f}")

    # Cross-evaluation: test this node's model on other nodes' test sets
    cross_eval_results = {}
    if cross_eval_dirs:
        print(f"\n  Cross-node evaluation:")
        for other_dir in cross_eval_dirs:
            other_name = os.path.basename(other_dir)
            if other_name == node_name:
                continue
            try:
                other_test_ds = FlameDataset(other_dir, split="test")
                other_loader = DataLoader(other_test_ds, batch_size=batch_size,
                                          shuffle=False, num_workers=0, pin_memory=False)
                other_loss, other_acc = evaluate(model, other_loader, criterion, device)
                cross_eval_results[other_name] = {
                    "test_loss": round(other_loss, 6),
                    "test_accuracy": round(other_acc, 6),
                }
                print(f"    → {other_name}: loss={other_loss:.4f}, acc={other_acc:.4f}")
            except Exception as e:
                print(f"    → {other_name}: FAILED ({e})")

    # Save results
    node_output_dir = os.path.join(output_dir, node_name)
    os.makedirs(node_output_dir, exist_ok=True)

    results = {
        "experiment": "local_only",
        "node_name": node_name,
        "hostname": hostname,
        "data_dir": data_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "total_time_s": round(total_time, 2),
        "final_test_loss": round(test_loss, 6),
        "final_test_accuracy": round(test_acc, 6),
        "cross_eval": cross_eval_results,
        "history": history,
        "fl_round_equivalents": {
            f"round_{r}": history[r * 5 - 1] if r * 5 <= len(history) else None
            for r in range(1, 4)
        },
        "train_class_distribution": train_ds.get_class_distribution(),
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(node_output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    model_path = os.path.join(node_output_dir, "model_final.pt")
    torch.save(model.state_dict(), model_path)

    print(f"\n  {node_name} completed in {total_time:.1f}s")
    print(f"  Results: {results_path}")

    # Clean up GPU memory for next node
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Local-only training baseline")
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: train all nodes sequentially on this device")

    # Single node mode
    parser.add_argument("--data_dir", default=None,
                        help="Path to single node's data dir")
    parser.add_argument("--node_name", default=None,
                        help="Node identifier (node_a, node_b, node_c)")

    # Batch mode
    parser.add_argument("--data_dirs", nargs="+", default=None,
                        help="Paths to all node data dirs (batch mode)")

    # Common
    parser.add_argument("--epochs", type=int, default=15,
                        help="Training epochs per node (3 rounds × 5 epochs = 15)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/local")
    parser.add_argument("--cross_eval", action="store_true",
                        help="Evaluate each model on all other nodes' test sets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.batch:
        # Batch mode: train all nodes sequentially
        if not args.data_dirs:
            print("ERROR: --data_dirs required in batch mode")
            return

        print("=" * 60)
        print(f"  FedRGBD — Local-Only Training (Batch Mode)")
        print(f"  Nodes: {len(args.data_dirs)}")
        print(f"  Epochs per node: {args.epochs}")
        print(f"  Seed: {args.seed}")
        print("=" * 60)

        cross_dirs = args.data_dirs if args.cross_eval else None
        all_results = {}
        total_start = time.perf_counter()

        for data_dir in args.data_dirs:
            node_name = os.path.basename(data_dir)
            result = train_single_node(
                data_dir=data_dir,
                node_name=node_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                output_dir=args.output_dir,
                cross_eval_dirs=cross_dirs,
            )
            all_results[node_name] = result

        total_time = time.perf_counter() - total_start

        # Save summary
        summary = {
            "experiment": "local_only_batch",
            "seed": args.seed,
            "epochs": args.epochs,
            "total_time_s": round(total_time, 2),
            "nodes": {
                name: {
                    "final_test_accuracy": r["final_test_accuracy"],
                    "final_test_loss": r["final_test_loss"],
                    "cross_eval": r.get("cross_eval", {}),
                }
                for name, r in all_results.items()
            },
            "mean_test_accuracy": round(
                np.mean([r["final_test_accuracy"] for r in all_results.values()]), 6
            ),
            "timestamp": datetime.now().isoformat(),
        }

        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"  ALL NODES COMPLETE — Total time: {total_time:.1f}s")
        for name, r in all_results.items():
            print(f"    {name}: accuracy={r['final_test_accuracy']:.4f}")
        print(f"    Mean accuracy: {summary['mean_test_accuracy']:.4f}")
        print(f"  Summary: {summary_path}")
        print(f"{'=' * 60}")

    else:
        # Single node mode
        if not args.data_dir or not args.node_name:
            print("ERROR: --data_dir and --node_name required (or use --batch mode)")
            return

        cross_dirs = None  # No cross-eval in single mode by default
        train_single_node(
            data_dir=args.data_dir,
            node_name=args.node_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            output_dir=args.output_dir,
            cross_eval_dirs=cross_dirs,
        )


if __name__ == "__main__":
    main()
