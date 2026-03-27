"""FedRGBD — Flower FL Client for fire classification on Jetson."""

import argparse
import socket
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader

import flwr as fl

import sys
sys.path.insert(0, ".")
from src.data.dataset import FlameDataset
from src.models.mobilenetv3_multimodal import create_model


class FedRGBDClient(fl.client.NumPyClient):
    def __init__(self, data_dir, batch_size=16, lr=0.001, local_epochs=5, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.hostname = socket.gethostname()
        self.batch_size = batch_size
        self.lr = lr
        self.local_epochs = local_epochs

        # Load data
        print(f"  [{self.hostname}] Loading data from {data_dir}...")
        self.train_ds = FlameDataset(data_dir, split="train")
        self.val_ds = FlameDataset(data_dir, split="val")
        self.test_ds = FlameDataset(data_dir, split="test")

        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size,
                                       shuffle=True, num_workers=0, pin_memory=False)
        self.val_loader = DataLoader(self.val_ds, batch_size=batch_size,
                                     shuffle=False, num_workers=0, pin_memory=False)

        # Create model
        self.model = create_model(num_classes=2, in_channels=3, pretrained=True)
        self.model.to(self.device)

        print(f"  [{self.hostname}] Train: {len(self.train_ds)}, Val: {len(self.val_ds)}, Test: {len(self.test_ds)}")
        print(f"  [{self.hostname}] Device: {self.device}")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

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
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
            total_loss += epoch_loss

        train_time = time.perf_counter() - start
        avg_loss = total_loss / max(total_samples, 1)
        print(f"  [{self.hostname}] Fit: {self.local_epochs} epochs, "
              f"loss={avg_loss:.4f}, time={train_time:.1f}s")

        return self.get_parameters(config={}), len(self.train_ds), {
            "train_loss": avg_loss,
            "train_time": train_time,
            "hostname": self.hostname,
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
    parser.add_argument("--server", default="192.168.1.8:8080")
    parser.add_argument("--data_dir", default="data/processed/iid/node_a")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--local_epochs", type=int, default=5)
    args = parser.parse_args()

    print("=" * 50)
    print(f"  FedRGBD FL Client — {socket.gethostname()}")
    print(f"  Server: {args.server}")
    print(f"  Data: {args.data_dir}")
    print("=" * 50)

    client = FedRGBDClient(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        local_epochs=args.local_epochs,
    )

    fl.client.start_client(
        server_address=args.server,
        client=client.to_client(),
    )
    print("Client finished!")


if __name__ == "__main__":
    main()
