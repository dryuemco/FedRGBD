import argparse
import socket
import torch
import torch.nn as nn
import flwr as fl
from collections import OrderedDict

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

class HelloClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = TinyModel()
        self.hostname = socket.gethostname()
        print(f"  Client on: {self.hostname} | CUDA: {torch.cuda.is_available()}")

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        x, y = torch.randn(32, 10), torch.randint(0, 2, (32,))
        opt = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.model.train()
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(self.model(x), y)
        loss.backward()
        opt.step()
        print(f"  [{self.hostname}] Train loss: {loss.item():.4f}")
        return self.get_parameters(config={}), 32, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        x, y = torch.randn(16, 10), torch.randint(0, 2, (16,))
        self.model.eval()
        with torch.no_grad():
            loss = nn.CrossEntropyLoss()(self.model(x), y)
        print(f"  [{self.hostname}] Eval loss: {loss.item():.4f}")
        return loss.item(), 16, {"loss": loss.item()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="192.168.1.4:8080")
    args = parser.parse_args()
    print(f"  Connecting to {args.server}...")
    fl.client.start_client(
        server_address=args.server,
        client=HelloClient().to_client(),
    )
    print("Client finished!")
