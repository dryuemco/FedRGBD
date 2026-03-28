import flwr as fl
from flwr.server.strategy import FedAvg

def main():
    strategy = FedAvg(
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )
    print("=" * 50)
    print("  FedRGBD - FL Hello World Server")
    print("  Waiting for 3 clients...")
    print("  Server: 0.0.0.0:8080")
    print("=" * 50)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
    print("\nFL Hello World completed! 3 rounds OK.")

if __name__ == "__main__":
    main()
