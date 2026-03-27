"""FedRGBD — Flower FL Server with logging."""

import argparse
import json
import os
import time
from datetime import datetime

import flwr as fl
from flwr.common import Metrics
from flwr.server.strategy import FedAvg, FedProx


def weighted_average(metrics):
    """Aggregate accuracy across clients."""
    accuracies = [num * m["accuracy"] for num, m in metrics]
    totals = [num for num, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(totals)}


def get_strategy(name, **kwargs):
    """Create FL strategy by name."""
    common = dict(
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    strategy = get_strategy(args.strategy)

    print("=" * 50)
    print(f"  FedRGBD FL Server")
    print(f"  Strategy: {args.strategy}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Address: {args.address}")
    print(f"  Output: {args.output_dir}")
    print("  Waiting for 2 clients...")
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
