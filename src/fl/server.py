"""FedRGBD — Flower FL Server with logging and seed support."""

import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import flwr as fl
from flwr.common import Metrics
from flwr.server.strategy import FedAvg, FedProx


def set_seed(seed):
    """Set all random seeds for reproducibility on server side."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def weighted_average(metrics):
    """Aggregate accuracy across clients."""
    accuracies = [num * m["accuracy"] for num, m in metrics]
    totals = [num for num, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(totals)}


def get_strategy(name, min_clients=3, **kwargs):
    """Create FL strategy by name."""
    common = dict(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    common.update(kwargs)

    if name == "fedavg":
        return FedAvg(**common)
    elif name.startswith("fedprox"):
        mu = float(name.split("_")[-1]) if "_" in name else 0.01
        return FedProx(proximal_mu=mu, **common)
    else:
        return FedAvg(**common)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="fedavg",
                        help="fedavg, fedprox_0.01, fedprox_0.1")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--address", default="0.0.0.0:8080")
    parser.add_argument("--output_dir", default="results/fl_run")
    parser.add_argument("--min_clients", type=int, default=3,
                        help="Minimum number of clients (2 or 3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    strategy = get_strategy(args.strategy, min_clients=args.min_clients)

    print("=" * 50)
    print(f"  FedRGBD FL Server")
    print(f"  Strategy: {args.strategy}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Address: {args.address}")
    print(f"  Output: {args.output_dir}")
    print(f"  Min clients: {args.min_clients}")
    print(f"  Seed: {args.seed}")
    print(f"  Waiting for {args.min_clients} clients...")
    print("=" * 50)

    start = time.perf_counter()

    history = fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    total_time = time.perf_counter() - start

    # Save results
    results = {
        "strategy": args.strategy,
        "num_rounds": args.rounds,
        "min_clients": args.min_clients,
        "seed": args.seed,
        "total_time_s": round(total_time, 2),
        "timestamp": datetime.now().isoformat(),
        "losses_distributed": [
            {"round": i+1, "loss": loss}
            for i, (_, loss) in enumerate(history.losses_distributed)
        ] if history.losses_distributed else [],
        "metrics_distributed": {
            key: [{"round": i+1, "value": val}
                  for i, (_, val) in enumerate(values)]
            for key, values in history.metrics_distributed.items()
        } if history.metrics_distributed else {},
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFL completed in {total_time:.1f}s")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
