"""End-to-end Flower run on localhost (CPU, synthetic data, 2 clients, 2 rounds).

This is an integration test of ``src/fl/server.py`` + ``src/fl/client.py`` with
the real Flower 1.13.1 gRPC stack — NOT an experiment.  It checks that the
per-round, per-client metrics reach ``results.json`` in the v3 layout while the
v2 keys stay intact.  The server runs in the test process (Flower installs a
signal handler on the client side, so clients run as subprocesses).
Runtime ≈ 15-30 s per strategy.
"""

import json
import os
import socket
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src", "fl"))

IMG = 16

CLIENT_SNIPPET = textwrap.dedent("""
    import sys, torch
    sys.path.insert(0, {repo!r})
    import flwr as fl
    from src.fl.client import FedRGBDClient
    torch.set_num_threads(1)
    client = FedRGBDClient({data_dir!r}, batch_size=4, lr=1e-3, local_epochs=1, device="cpu",
                           seed=42, pretrained=False, img_size={img})
    fl.client.start_client(server_address="127.0.0.1:{port}", client=client.to_client())
""")


def _make_node(root, seed):
    rng = np.random.RandomState(seed)
    for split in ("train", "val", "test"):
        for cls in ("Fire", "No_Fire"):
            d = os.path.join(root, split, cls)
            os.makedirs(d, exist_ok=True)
            for i in range(4):
                arr = rng.randint(0, 255, (IMG, IMG, 3), dtype=np.uint8)
                Image.fromarray(arr).save(os.path.join(d, f"{i}.png"))
    return root


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("strategy", ["fedprox_0.01", "fedbn"])
def test_two_client_run_writes_v3_results(tmp_path, strategy):
    import server as fl_server

    nodes = [_make_node(str(tmp_path / f"node_{c}"), i) for i, c in enumerate("ab")]
    port = _free_port()
    out_dir = tmp_path / "run"
    rounds = 2

    procs = []
    for data_dir in nodes:
        code = CLIENT_SNIPPET.format(repo=_REPO, data_dir=data_dir, img=IMG, port=port)
        procs.append(subprocess.Popen([sys.executable, "-c", code], cwd=_REPO,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    try:
        fl_server.main(["--strategy", strategy, "--rounds", str(rounds), "--address", f"127.0.0.1:{port}",
                        "--output_dir", str(out_dir), "--min_clients", "2", "--seed", "42",
                        "--tag", "unit_test"])
    finally:
        outputs = []
        for p in procs:
            try:
                out, _ = p.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
                out, _ = p.communicate()
            outputs.append(out)
    for p, out in zip(procs, outputs):
        assert p.returncode == 0, out[-3000:]

    res = json.loads((out_dir / "results.json").read_text())
    # v2 keys
    assert res["strategy"] == strategy and res["num_rounds"] == rounds and res["seed"] == 42
    assert [d["round"] for d in res["losses_distributed"]] == [1, 2]
    assert [d["round"] for d in res["metrics_distributed"]["accuracy"]] == [1, 2]
    # v3 keys
    assert res["results_schema_version"] == 2 and res["tags"] == ["unit_test"]
    assert res["proximal_mu"] == (0.01 if strategy.startswith("fedprox") else 0.0)
    assert set(res["client_config"]) == {"node_a", "node_b"}
    assert res["client_config"]["node_a"]["local_epochs"] == 1
    assert len(res["rounds"]) == rounds
    r1 = res["rounds"][0]
    assert r1["round"] == 1
    assert set(r1["evaluate"]["clients"]) == {"node_a", "node_b"}
    for node_row in r1["evaluate"]["clients"].values():
        for k in ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc", "loss",
                  "eval_time_s", "payload_bytes_down", "num_examples", "confusion_matrix"):
            assert k in node_row, k
    for node_row in r1["fit"]["clients"].values():
        for k in ("train_loss", "fit_time_s", "payload_bytes_up", "payload_bytes_down", "num_examples"):
            assert k in node_row, k
        assert node_row["payload_bytes_up"] == res["model_payload_bytes"]
        if strategy.startswith("fedprox"):
            assert node_row["proximal_mu"] == 0.01 and node_row["strategy"].startswith("FedProx")
        else:
            assert node_row["strategy"] == "FedBN"
    assert "pooled_accuracy" in r1["evaluate"]["aggregate"]
    assert r1["evaluate"]["aggregate"]["accuracy"] == pytest.approx(res["metrics_distributed"]["accuracy"][0]["value"])
    assert 0 < r1["fit"]["elapsed_s"] <= r1["evaluate"]["elapsed_s"] <= res["rounds"][1]["fit"]["elapsed_s"]
    # 2 clients × (fit down + fit up + eval down) per round
    assert res["rounds"][1]["cumulative_communication_bytes"] == 2 * 3 * rounds * res["model_payload_bytes"]
    assert res["total_communication_bytes"] == res["rounds"][-1]["cumulative_communication_bytes"]
    assert "NaN" not in (out_dir / "results.json").read_text()
