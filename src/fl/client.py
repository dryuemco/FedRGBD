"""FedRGBD — Flower FL Client with FedAvg, FedProx, and FedBN support (v2)."""

import argparse
import os
import random
import socket
import time

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader

import flwr as fl

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


def get_bn_indices(model):
    """Get state_dict indices that belong to BatchNorm layers.
    
    Uses isinstance() on actual modules - more reliable than shape heuristics.
    """
    bn_layer_names = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_layer_names.add(name)
    
    bn_indices = set()
    state_dict_keys = list(model.state_dict().keys())
    for i, key in enumerate(state_dict_keys):
        parent = key.rsplit('.', 1)[0]
        if parent in bn_layer_names:
            bn_indices.add(i)
    
    return bn_indices


class FedRGBDClient(fl.client.NumPyClient):
    def __init__(self, data_dir, batch_size=16, lr=0.001, local_epochs=5,
                 device="cuda", seed=42):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.hostname = socket.gethostname()
        self.batch_size = batch_size
        self.lr = lr
        self.local_epochs = local_epochs
        self.seed = seed
        self._fedbn_mode = False

        set_seed(seed)

        # Load data
        print(f"  [{self.hostname}] Loading data from {data_dir}...")
        self.train_ds = FlameDataset(data_dir, split="train")
        self.val_ds = FlameDataset(data_dir, split="val")
        self.test_ds = FlameDataset(data_dir, split="test")

        g = torch.Generator()
        g.manual_seed(seed)

        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size,
                                       shuffle=True, num_workers=0, pin_memory=False,
                                       generator=g)
        self.val_loader = DataLoader(self.val_ds, batch_size=batch_size,
                                     shuffle=False, num_workers=0, pin_memory=False)

        # Create model
        self.model = create_model(num_classes=2, in_channels=3, pretrained=True)
        self.model.to(self.device)

        # Name-based BN detection (matches server fedbn_strategy.py)
        self._bn_key_indices = get_bn_indices(self.model)

        print(f"  [{self.hostname}] Train: {len(self.train_ds)}, Val: {len(self.val_ds)}, "
              f"Test: {len(self.test_ds)}")
        print(f"  [{self.hostname}] Device: {self.device}, BN params: {len(self._bn_key_indices)}")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Load global parameters. In FedBN mode, skip BatchNorm layers."""
        keys = list(self.model.state_dict().keys())
        current_state = self.model.state_dict()

        if self._fedbn_mode:
            new_state = OrderedDict()
            for i, key in enumerate(keys):
                if i in self._bn_key_indices:
                    # Keep local BN parameters
                    new_state[key] = current_state[key]
                else:
                    # Cast to the original dtype of the parameter
                    orig_dtype = current_state[key].dtype
                    new_state[key] = torch.tensor(parameters[i]).to(orig_dtype)
        else:
            new_state = OrderedDict()
            for i, key in enumerate(keys):
                orig_dtype = current_state[key].dtype
                new_state[key] = torch.tensor(parameters[i]).to(orig_dtype)

        self.model.load_state_dict(new_state, strict=True)

    def fit(self, parameters, config):
        if config.get("fedbn", False):
            if not self._fedbn_mode:
                self._fedbn_mode = True
                print(f"  [{self.hostname}] FedBN mode: keeping {len(self._bn_key_indices)} "
                      f"BN params local")

        self.set_parameters(parameters)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        proximal_mu = config.get("proximal_mu", 0.0)
        if proximal_mu > 0:
            global_params = [val.clone().detach().cpu() for val in self.model.parameters()]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        total_loss = 0
        total_samples = 0
        start = time.perf_counter()

        for epoch in range(self.local_epochs):
            epoch_loss = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                if proximal_mu > 0:
                    proximal_term = 0.0
                    for local_param, global_param in zip(self.model.parameters(), global_params):
                        proximal_term += ((local_param - global_param.to(self.device)) ** 2).sum()
                    loss = loss + (proximal_mu / 2.0) * proximal_term

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
            total_loss += epoch_loss

        train_time = time.perf_counter() - start
        avg_loss = total_loss / max(total_samples, 1)

        if self._fedbn_mode:
            strategy_str = "FedBN"
        elif proximal_mu > 0:
            strategy_str = f"FedProx(mu={proximal_mu})"
        else:
            strategy_str = "FedAvg"

        print(f"  [{self.hostname}] Fit: {self.local_epochs} epochs, "
              f"loss={avg_loss:.4f}, time={train_time:.1f}s, strategy={strategy_str}")

        if proximal_mu > 0:
            del global_params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.get_parameters(config={}), len(self.train_ds), {
            "train_loss": avg_loss,
            "train_time": train_time,
            "hostname": self.hostname,
            "strategy": strategy_str,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        print(f"  [{self.hostname}] Eval: loss={avg_loss:.4f}, acc={accuracy:.4f}")

        return avg_loss, total, {"accuracy": accuracy, "hostname": self.hostname}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="192.168.1.4:8080")
    parser.add_argument("--data_dir", default="data/processed/iid/node_a")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  FedRGBD FL Client — {socket.gethostname()}")
    print(f"  Server: {args.server}")
    print(f"  Data: {args.data_dir}")
    print(f"  Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Local epochs: {args.local_epochs}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    client = FedRGBDClient(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        local_epochs=args.local_epochs,
        seed=args.seed,
    )

    fl.client.start_client(
        server_address=args.server,
        client=client.to_client(),
    )
    print("Client finished!")


if __name__ == "__main__":
    main()
