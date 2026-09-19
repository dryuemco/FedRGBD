"""FedRGBD — Flower FL Client with FedAvg, FedProx, and FedBN support (v3).

v3 (NCAA revision): ``evaluate()`` returns the full metric set from
``src/evaluation/metrics.py`` (accuracy, balanced accuracy, precision, recall,
specificity, F1, MCC, ROC-AUC, confusion matrix); ``fit()`` and ``evaluate()``
report wall-clock time and the model payload bytes sent/received so the server
can persist per-client, per-round cost measurements.  The model is unchanged.

Model selection: every round ``evaluate()`` scores the global model on the
validation split (``val_*`` keys, plus the unprefixed legacy keys) and then on
the test split (``test_*`` keys, report only).  Selection uses validation only;
see ``src/evaluation/model_selection.py``.
"""

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
from src.evaluation.metrics import MetricAccumulator, format_metrics, to_flower_metrics
from src.models.mobilenetv3_multimodal import create_model


def payload_bytes(parameters):
    """Bytes of a list of NumPy arrays (= bytes serialised on the wire, excluding gRPC framing)."""
    return int(sum(np.asarray(p).nbytes for p in parameters))


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
                 device="cuda", seed=42, node_name=None, pretrained=True,
                 img_size=224):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.hostname = socket.gethostname()
        self.data_dir = data_dir
        # node_a / node_b / node_c — used by the server to key per-client metrics
        self.node_name = node_name or os.path.basename(os.path.normpath(data_dir))
        self.batch_size = batch_size
        self.lr = lr
        self.local_epochs = local_epochs
        self.seed = seed
        self._fedbn_mode = False

        set_seed(seed)

        # Load data
        print(f"  [{self.hostname}] Loading data from {data_dir}...")
        self.train_ds = FlameDataset(data_dir, split="train", img_size=img_size)
        self.val_ds = FlameDataset(data_dir, split="val", img_size=img_size)
        self.test_ds = FlameDataset(data_dir, split="test", img_size=img_size)

        g = torch.Generator()
        g.manual_seed(seed)

        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size,
                                       shuffle=True, num_workers=0, pin_memory=False,
                                       generator=g)
        self.val_loader = DataLoader(self.val_ds, batch_size=batch_size,
                                     shuffle=False, num_workers=0, pin_memory=False)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size,
                                      shuffle=False, num_workers=0, pin_memory=False)

        # Create model (architecture unchanged; pretrained=False only for CPU unit tests)
        self.model = create_model(num_classes=2, in_channels=3, pretrained=pretrained)
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
        fit_start = time.perf_counter()
        bytes_down = payload_bytes(parameters)
        server_round = int(config.get("server_round", 0))

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

        new_parameters = self.get_parameters(config={})
        bytes_up = payload_bytes(new_parameters)
        fit_wall = time.perf_counter() - fit_start

        return new_parameters, len(self.train_ds), {
            # --- keys kept from v2 ---
            "train_loss": avg_loss,
            "train_time": train_time,
            "hostname": self.hostname,
            "strategy": strategy_str,
            # --- v3: identity / config echo (lets the server persist per-client rows) ---
            "node_name": self.node_name,
            "data_dir": self.data_dir,
            "server_round": server_round,
            "local_epochs": int(self.local_epochs),
            "lr": float(self.lr),
            "batch_size": int(self.batch_size),
            "proximal_mu": float(proximal_mu),
            "num_examples": len(self.train_ds),
            # --- v3: cost measurements ---
            "fit_time_s": train_time,          # pure local-training time
            "fit_wall_s": fit_wall,            # incl. parameter loading / serialisation
            "payload_bytes_down": bytes_down,  # global model received for this round
            "payload_bytes_up": bytes_up,      # local model returned to the server
        }

    def evaluate(self, parameters, config):
        """Evaluate the round's global model on the validation AND the test split.

        Model-selection protocol (paper, Section III): only the validation split
        drives anything.  The returned loss and example count -- which Flower
        aggregates into ``losses_distributed`` and which the server's selection
        rule reads -- are the validation ones, and the unprefixed metric keys are
        the validation metrics exactly as before.  The test split is evaluated
        *after* validation, with the model in eval mode and no gradient, and is
        only reported, under ``test_*`` keys; it feeds no training, weighting or
        selection decision.

        Timers: ``val_eval_time_s`` / ``test_eval_time_s`` are the two loader
        passes; ``eval_time_s`` is the evaluation cost that counts toward the
        reported round time (parameter loading + validation, *excluding* the test
        pass); ``eval_wall_s`` is the whole call including the test pass.
        """
        eval_start = time.perf_counter()
        bytes_down = payload_bytes(parameters)
        server_round = int(config.get("server_round", 0))
        self.set_parameters(parameters)

        val_start = time.perf_counter()
        val_metrics = self.evaluate_loader(self.val_loader)
        val_eval_time = time.perf_counter() - val_start
        eval_time = time.perf_counter() - eval_start  # reported: excludes the test pass

        test_start = time.perf_counter()
        test_metrics = self.evaluate_loader(self.test_loader)
        test_eval_time = time.perf_counter() - test_start
        eval_wall = time.perf_counter() - eval_start

        total = int(val_metrics["n_examples"])
        avg_loss = float(val_metrics["loss"]) if val_metrics["loss"] is not None else 0.0
        print(f"  [{self.hostname}] Eval r{server_round} (val): loss={avg_loss:.4f}, "
              f"{format_metrics(val_metrics)}, time={val_eval_time:.1f}s")
        print(f"  [{self.hostname}] Eval r{server_round} (test, report only): "
              f"{format_metrics(test_metrics)}, time={test_eval_time:.1f}s")

        flat = to_flower_metrics(val_metrics)               # v3 keys = validation, unchanged
        flat.update(to_flower_metrics(val_metrics, prefix="val_"))
        flat.update(to_flower_metrics(test_metrics, prefix="test_"))
        flat.update({
            "hostname": self.hostname,
            "node_name": self.node_name,
            "server_round": server_round,
            "eval_split": "val",               # the split behind loss / num_examples / unprefixed keys
            "num_examples": total,
            "eval_time_s": eval_time,
            "val_eval_time_s": val_eval_time,
            "test_eval_time_s": test_eval_time,
            "eval_wall_s": eval_wall,
            "payload_bytes_down": bytes_down,
        })
        return avg_loss, total, flat

    def evaluate_loader(self, loader):
        """Run the model over ``loader`` and return the full metrics dict (incl. ``loss``)."""
        criterion = nn.CrossEntropyLoss(reduction="sum")
        acc = MetricAccumulator(num_classes=2, positive_class=1)
        self.model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss_sum = criterion(outputs, labels).item()
                acc.update(outputs, labels, loss_sum)
        return acc.compute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="192.168.1.10:8080")
    parser.add_argument("--data_dir", default="data/processed/iid/node_a")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--node_name", default=None,
                        help="Client identifier for results.json (default: basename of --data_dir)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  FedRGBD FL Client — {socket.gethostname()}")
    print(f"  Server: {args.server}")
    print(f"  Data: {args.data_dir}")
    print(f"  Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Local epochs: {args.local_epochs}")
    print(f"  Seed: {args.seed}")
    print(f"  Eval: val (selection) + test (report only), every round")
    print("=" * 60)

    client = FedRGBDClient(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        local_epochs=args.local_epochs,
        seed=args.seed,
        node_name=args.node_name,
    )

    fl.client.start_client(
        server_address=args.server,
        client=client.to_client(),
    )
    print("Client finished!")


if __name__ == "__main__":
    main()
