"""FedRGBD — Flower FL Server with FedAvg, FedProx, and FedBN support (v3).

v3 (NCAA revision): every round the server now persists, in ``results.json``,

* per-client evaluation metrics (accuracy, balanced accuracy, precision,
  recall/sensitivity, specificity, F1, MCC, ROC-AUC, confusion matrix, loss),
* the weighted-global aggregate of those metrics **and** the "pooled" metrics
  computed from the summed confusion matrix of all clients,
* per-client fit/eval wall-clock time and model payload bytes (up/down), and
  the cumulative communication volume of the run,
* the server-side elapsed time at the end of every fit / evaluate phase.

All keys written by the previous version (``strategy``, ``num_rounds``,
``min_clients``, ``seed``, ``total_time_s``, ``timestamp``,
``losses_distributed``, ``metrics_distributed``) are kept unchanged so the
existing analysis code and the old result files remain compatible.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import flwr as fl
from flwr.common import Metrics
from flwr.server.strategy import FedAvg, FedProx

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# FedBN strategy (local import)
from fedbn_strategy import FedBN  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    METRIC_KEYS,
    confusion_matrix_from_flat,
    metrics_from_confusion_matrix,
)

RESULTS_SCHEMA_VERSION = 2

# metric keys that are averaged with num_examples weights
WEIGHTED_KEYS = set(METRIC_KEYS) | {"loss", "train_loss"}
# keys that are summed over clients
SUMMED_PREFIXES = ("payload_bytes", "cm_", "support_")
SUMMED_KEYS = {"num_examples", "tp", "fp", "fn", "tn"}
# keys for which mean AND max are reported (wall-clock)
TIME_SUFFIXES = ("_time_s", "_wall_s", "train_time")
# keys never aggregated (identity / config echo)
IDENTITY_KEYS = {"hostname", "node_name", "data_dir", "server_round", "strategy", "eval_split",
                 "local_epochs", "lr", "batch_size", "proximal_mu", "confusion_matrix_json",
                 "n_examples"}  # n_examples duplicates num_examples (summed as num_examples_total)


def set_seed(seed):
    """Set all random seeds for reproducibility on server side."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def weighted_average(metrics):
    """Aggregate accuracy across clients (v2 behaviour, kept for reference/tests)."""
    accuracies = [num * m["accuracy"] for num, m in metrics]
    totals = [num for num, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(totals)}


def model_payload_bytes() -> Optional[int]:
    """Bytes of one full model transfer (all state_dict tensors, as sent by the client)."""
    try:
        from src.models.mobilenetv3_multimodal import create_model
        model = create_model(num_classes=2, in_channels=3, pretrained=False)
        return int(sum(v.numel() * v.element_size() for v in model.state_dict().values()))
    except Exception as exc:  # pragma: no cover - torchvision missing etc.
        print(f"  [server] could not compute model payload size: {exc}")
        return None


# --------------------------------------------------------------------------- #
# per-round recorder
# --------------------------------------------------------------------------- #
class RoundRecorder:
    """Stateful metric aggregation that also remembers every client's raw metrics.

    Flower calls ``fit_metrics_aggregation_fn`` / ``evaluate_metrics_aggregation_fn``
    once per round with ``[(num_examples, metrics), ...]``.  The client echoes
    ``server_round`` and ``node_name`` in its metrics, which lets us key rows
    without subclassing the strategy.  Falls back to a call counter / hostname
    for older clients.
    """

    def __init__(self, start_time: Optional[float] = None):
        self.start_time = start_time if start_time is not None else time.perf_counter()
        self.rounds: Dict[int, dict] = {}
        self.client_config: Dict[str, dict] = {}
        self.cumulative_bytes = 0
        self._fit_calls = 0
        self._eval_calls = 0

    # -- helpers ---------------------------------------------------------- #
    def elapsed(self) -> float:
        return round(time.perf_counter() - self.start_time, 3)

    @staticmethod
    def _client_key(m: Metrics, idx: int) -> str:
        return str(m.get("node_name") or m.get("hostname") or f"client_{idx}")

    @staticmethod
    def _round_of(metrics: List[Tuple[int, Metrics]], counter: int) -> int:
        rounds = [int(m["server_round"]) for _, m in metrics if int(m.get("server_round", 0)) > 0]
        return max(rounds) if rounds else counter

    @staticmethod
    def _client_row(num_examples: int, m: Metrics) -> dict:
        """Readable per-client row: nested confusion matrix, no flattened duplicates."""
        row = {}
        cm = confusion_matrix_from_flat(m)
        for k, v in m.items():
            if k.startswith("cm_") or k == "confusion_matrix_json":
                continue
            if isinstance(v, (np.generic,)):
                v = v.item()
            row[k] = v
        row["num_examples"] = int(num_examples)
        if cm is not None:
            row["confusion_matrix"] = cm.tolist()
        return row

    @staticmethod
    def aggregate(metrics: List[Tuple[int, Metrics]]) -> Dict[str, float]:
        """Weighted-global aggregate + pooled confusion-matrix metrics (flat, Flower-safe)."""
        total = float(sum(n for n, _ in metrics)) or 1.0
        agg: Dict[str, float] = {}
        keys = set()
        for _, m in metrics:
            keys |= set(m.keys())
        for key in sorted(keys):
            vals = [(n, m[key]) for n, m in metrics if key in m and isinstance(m[key], (int, float)) and not isinstance(m[key], bool)]
            if not vals or key in IDENTITY_KEYS:
                continue
            if key in SUMMED_KEYS or key.startswith(SUMMED_PREFIXES):
                total_v = int(sum(v for _, v in vals))
                if key == "num_examples":
                    agg["num_examples_total"] = total_v
                elif key.startswith("payload_bytes"):
                    agg[key + "_total"] = total_v
                else:
                    agg[key] = total_v
            elif key.endswith(TIME_SUFFIXES):
                xs = [v for _, v in vals]
                agg[key + "_mean"] = float(np.mean(xs))
                agg[key + "_max"] = float(np.max(xs))
            else:  # weighted mean (metrics, loss, per-class values, ...)
                w = float(sum(n for n, _ in vals)) or 1.0
                agg[key] = float(sum(n * v for n, v in vals) / w)

        # pooled metrics from the summed confusion matrix
        cms = [confusion_matrix_from_flat(m) for _, m in metrics]
        cms = [c for c in cms if c is not None]
        if cms:
            k = max(c.shape[0] for c in cms)
            pooled_cm = np.zeros((k, k), dtype=np.int64)
            for c in cms:
                pooled_cm[: c.shape[0], : c.shape[1]] += c
            pooled = metrics_from_confusion_matrix(pooled_cm)
            for key in METRIC_KEYS:
                if key in pooled and pooled[key] is not None:
                    agg[f"pooled_{key}"] = float(pooled[key])
        return agg

    # -- Flower callbacks ------------------------------------------------- #
    def fit_aggregation(self, metrics: List[Tuple[int, Metrics]]) -> Dict[str, float]:
        self._fit_calls += 1
        rnd = self._round_of(metrics, self._fit_calls)
        clients = {}
        round_bytes = 0
        for i, (n, m) in enumerate(metrics):
            key = self._client_key(m, i)
            clients[key] = self._client_row(n, m)
            cfg = {k: m[k] for k in ("local_epochs", "lr", "batch_size", "data_dir", "hostname") if k in m}
            if cfg:
                self.client_config.setdefault(key, {}).update(cfg)
            round_bytes += int(m.get("payload_bytes_up", 0)) + int(m.get("payload_bytes_down", 0))
        agg = self.aggregate(metrics)
        self.cumulative_bytes += round_bytes
        entry = self.rounds.setdefault(rnd, {"round": rnd})
        entry["fit"] = {"clients": clients, "aggregate": agg, "elapsed_s": self.elapsed()}
        entry["cumulative_communication_bytes"] = self.cumulative_bytes
        return agg

    def evaluate_aggregation(self, metrics: List[Tuple[int, Metrics]]) -> Dict[str, float]:
        self._eval_calls += 1
        rnd = self._round_of(metrics, self._eval_calls)
        clients = {}
        round_bytes = 0
        for i, (n, m) in enumerate(metrics):
            key = self._client_key(m, i)
            clients[key] = self._client_row(n, m)
            round_bytes += int(m.get("payload_bytes_down", 0))
        agg = self.aggregate(metrics)
        self.cumulative_bytes += round_bytes
        entry = self.rounds.setdefault(rnd, {"round": rnd})
        entry["evaluate"] = {"clients": clients, "aggregate": agg, "elapsed_s": self.elapsed()}
        entry["cumulative_communication_bytes"] = self.cumulative_bytes
        # Flower history requires "accuracy" (v2 key) — guarantee it exists
        if "accuracy" not in agg:
            agg["accuracy"] = weighted_average(metrics)["accuracy"] if all("accuracy" in m for _, m in metrics) else 0.0
        return agg

    def to_list(self) -> List[dict]:
        return [self.rounds[r] for r in sorted(self.rounds)]


# --------------------------------------------------------------------------- #
# strategy factory
# --------------------------------------------------------------------------- #
def make_fit_config_fn(strategy_name):
    """Create on_fit_config_fn that passes strategy info to clients."""
    def fit_config(server_round: int):
        config = {"server_round": server_round}
        if strategy_name == "fedbn":
            config["fedbn"] = True
        return config
    return fit_config


def make_evaluate_config_fn():
    def evaluate_config(server_round: int):
        return {"server_round": server_round}
    return evaluate_config


def parse_mu(name: str) -> float:
    if name.startswith("fedprox"):
        return float(name.split("_")[-1]) if "_" in name else 0.01
    return 0.0


def get_strategy(name, min_clients=3, recorder: Optional[RoundRecorder] = None, **kwargs):
    """Create FL strategy by name."""
    common = dict(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=recorder.evaluate_aggregation if recorder else weighted_average,
        fit_metrics_aggregation_fn=recorder.fit_aggregation if recorder else None,
        on_fit_config_fn=make_fit_config_fn(name),
        on_evaluate_config_fn=make_evaluate_config_fn(),
    )
    common.update(kwargs)

    if name == "fedavg":
        return FedAvg(**common)
    elif name.startswith("fedprox"):
        return FedProx(proximal_mu=parse_mu(name), **common)
    elif name == "fedbn":
        return FedBN(**common)
    else:
        return FedAvg(**common)


# --------------------------------------------------------------------------- #
# results serialisation
# --------------------------------------------------------------------------- #
def json_safe(obj):
    """Make a value strictly JSON-serialisable (RFC 8259).

    ``json.dump`` happily writes the bare tokens ``NaN`` / ``Infinity``, which
    every non-Python JSON reader rejects.  A diverging local round (large
    ``proximal_mu``, high lr) produces exactly that via ``train_loss``, so the
    whole run's results.json would be unreadable outside Python.  Non-finite
    floats become ``None``; numpy scalars/arrays become Python types.
    """
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, bytes):  # pragma: no cover - no client sends bytes today
        return obj.decode("utf-8", "replace")
    return obj


def build_results(args, history, total_time: float, recorder: RoundRecorder,
                  payload_bytes: Optional[int]) -> dict:
    """Assemble results.json — v2 keys first (unchanged), v3 keys appended."""
    results = {
        "strategy": args.strategy,
        "num_rounds": args.rounds,
        "min_clients": args.min_clients,
        "seed": args.seed,
        "total_time_s": round(total_time, 2),
        "timestamp": datetime.now().isoformat(),
        "losses_distributed": [
            {"round": i + 1, "loss": loss}
            for i, (_, loss) in enumerate(history.losses_distributed)
        ] if history.losses_distributed else [],
        "metrics_distributed": {
            key: [{"round": i + 1, "value": val}
                  for i, (_, val) in enumerate(values)]
            for key, values in history.metrics_distributed.items()
        } if history.metrics_distributed else {},
    }
    # ---- v3 additions ----
    results.update({
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "proximal_mu": parse_mu(args.strategy),
        "tags": list(args.tag or []),
        "address": args.address,
        "client_config": recorder.client_config,
        "model_payload_bytes": payload_bytes,
        "total_communication_bytes": recorder.cumulative_bytes,
        "metrics_distributed_fit": {
            key: [{"round": i + 1, "value": val} for i, (_, val) in enumerate(values)]
            for key, values in getattr(history, "metrics_distributed_fit", {}).items()
        },
        "rounds": recorder.to_list(),
    })
    return json_safe(results)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="fedavg",
                        help="fedavg, fedprox_<mu> (e.g. fedprox_0.01), fedbn")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--address", default="0.0.0.0:8080")
    parser.add_argument("--output_dir", default="results/fl_run")
    parser.add_argument("--min_clients", type=int, default=3,
                        help="Minimum number of clients (2 or 3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", nargs="*", default=None,
                        help="Free-form tags stored in results.json, e.g. --tag dirichlet_0.1 sub0.05")
    args = parser.parse_args(argv)

    # Set seed
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    recorder = RoundRecorder()
    strategy = get_strategy(args.strategy, min_clients=args.min_clients, recorder=recorder)
    payload = model_payload_bytes()

    print("=" * 60)
    print(f"  FedRGBD FL Server")
    print(f"  Strategy: {args.strategy}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Address: {args.address}")
    print(f"  Output: {args.output_dir}")
    print(f"  Min clients: {args.min_clients}")
    print(f"  Seed: {args.seed}")
    if args.tag:
        print(f"  Tags: {args.tag}")
    if payload:
        print(f"  Model payload: {payload / 1e6:.2f} MB per transfer")
    print(f"  Waiting for {args.min_clients} clients...")
    print("=" * 60)

    start = time.perf_counter()
    recorder.start_time = start

    history = fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    total_time = time.perf_counter() - start

    results = build_results(args, history, total_time, recorder, payload)
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFL completed in {total_time:.1f}s")
    print(f"Total communication: {recorder.cumulative_bytes / 1e6:.1f} MB")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
