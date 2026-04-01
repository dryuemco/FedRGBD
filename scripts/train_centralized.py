"""FedRGBD — Centralized Training Baseline (Upper Bound).

Pools all node data and trains a single model on one device.
This represents the best achievable performance without FL constraints.

Usage:
  python3 scripts/train_centralized.py \
    --data_dirs data/processed/iid/node_a data/processed/iid/node_b data/processed/iid/node_c \
    --seed 42 --output_dir results/centralized_iid_seed42

  python3 scripts/train_centralized.py \
    --data_dirs data/processed/non_iid_label/node_a data/processed/non_iid_label/node_b data/processed/non_iid_label/node_c \
    --seed 42 --output_dir results/centralized_noniid_seed42
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
from torch.utils.data import DataLoader, ConcatDataset

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


def main():
    parser = argparse.ArgumentParser(description="Centralized training baseline")
    parser.add_argument("--data_dirs", nargs="+", required=True,
                        help="Paths to all node data dirs (will be pooled)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Total training epochs (3 rounds × 5 local_epochs = 15)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/centralized")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hostname = socket.gethostname()

    print("=" * 60)
    print(f"  FedRGBD — Centralized Training Baseline")
    print(f"  Host: {hostname}")
    print(f"  Data dirs: {args.data_dirs}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {device}")
    print("=" * 60)

    # Pool all training data from all nodes
    print("\nLoading and pooling data from all nodes...")
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for data_dir in args.data_dirs:
        node_name = os.path.basename(data_dir)
        train_ds = FlameDataset(data_dir, split="train")
        val_ds = FlameDataset(data_dir, split="val")
        test_ds = FlameDataset(data_dir, split="test")
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        test_datasets.append(test_ds)
        print(f"  {node_name}: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Concatenate all splits
    pooled_train = ConcatDataset(train_datasets)
    pooled_val = ConcatDataset(val_datasets)
    pooled_test = ConcatDataset(test_datasets)

    print(f"\n  Pooled totals: train={len(pooled_train)}, val={len(pooled_val)}, "
          f"test={len(pooled_test)}")

    # DataLoaders
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(pooled_train, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False,
                              generator=g)
    val_loader = DataLoader(pooled_val, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(pooled_test, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=False)

    # Create model
    model = create_model(num_classes=2, in_channels=3, pretrained=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\nTraining...")
    history = []
    start_total = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
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

        # Print at FL-equivalent "rounds" (every 5 epochs = 1 FL round)
        marker = " ← FL Round equivalent" if epoch % 5 == 0 else ""
        print(f"  Epoch {epoch:2d}/{args.epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_acc={val_acc:.4f}, time={epoch_time:.1f}s{marker}")

    total_time = time.perf_counter() - start_total

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n  Final Test: loss={test_loss:.4f}, accuracy={test_acc:.4f}")

    # Also evaluate on each node's test set individually
    per_node_results = {}
    for i, data_dir in enumerate(args.data_dirs):
        node_name = os.path.basename(data_dir)
        node_test_loader = DataLoader(test_datasets[i], batch_size=args.batch_size,
                                      shuffle=False, num_workers=0, pin_memory=False)
        node_loss, node_acc = evaluate(model, node_test_loader, criterion, device)
        per_node_results[node_name] = {
            "test_loss": round(node_loss, 6),
            "test_accuracy": round(node_acc, 6),
        }
        print(f"  {node_name} Test: loss={node_loss:.4f}, accuracy={node_acc:.4f}")

    # Save results
    results = {
        "experiment": "centralized",
        "hostname": hostname,
        "data_dirs": args.data_dirs,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "total_time_s": round(total_time, 2),
        "final_test_loss": round(test_loss, 6),
        "final_test_accuracy": round(test_acc, 6),
        "per_node_test": per_node_results,
        "history": history,
        "fl_round_equivalents": {
            f"round_{r}": history[r * 5 - 1] if r * 5 <= len(history) else None
            for r in range(1, 4)  # Rounds 1, 2, 3
        },
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save model checkpoint
    model_path = os.path.join(args.output_dir, "model_final.pt")
    torch.save(model.state_dict(), model_path)

    print(f"\nCentralized training completed in {total_time:.1f}s")
    print(f"Results saved to {results_path}")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
