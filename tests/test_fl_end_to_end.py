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
    assert res["results_schema_version"] == 3 and res["tags"] == ["unit_test"]
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

    # schema 3: validation and test metric sets, per client and weighted-global
    for node_row in r1["evaluate"]["clients"].values():
        for k in ("val_accuracy", "val_loss", "val_mcc", "val_n_examples", "val_confusion_matrix",
                  "test_accuracy", "test_loss", "test_mcc", "test_n_examples",
                  "test_confusion_matrix", "val_eval_time_s", "test_eval_time_s", "eval_wall_s"):
            assert k in node_row, k
        assert node_row["accuracy"] == node_row["val_accuracy"]
        assert node_row["num_examples"] == node_row["val_n_examples"]
    agg1 = r1["evaluate"]["aggregate"]
    for k in ("val_accuracy", "test_accuracy", "test_balanced_accuracy", "pooled_test_accuracy",
              "test_n_examples_total", "val_n_examples_total"):
        assert k in agg1, k
    # Flower's distributed loss is the validation loss
    assert res["losses_distributed"][0]["loss"] == pytest.approx(agg1["val_loss"])
    # model selection from validation losses; timing without the test pass
    sel = res["model_selection"]
    assert sel["selected_round"] in (1, 2)
    losses = {int(r): v for r, v in sel["val_loss_by_round"].items()}
    assert sel["selected_round"] == min(losses, key=lambda r: (losses[r], r))
    assert sel["selected_round_test"]["accuracy"] == pytest.approx(
        res["rounds"][sel["selected_round"] - 1]["evaluate"]["aggregate"]["test_accuracy"])
    for entry in res["rounds"]:
        t = entry["timing"]
        assert t["round_time_s"] == pytest.approx(t["round_wall_s"] - t["test_eval_overhead_s"],
                                                  abs=2e-3)
        assert t["test_eval_overhead_s"] > 0
    assert res["total_time_excl_test_s"] < res["total_time_s"]

    # and the analysis reads it back under the declared rule
    from scripts.analyze_results import load_run
    run = load_run(str(out_dir), warn=False)
    assert run["headline_source"] == "selected_round_test"
    assert run["selected_round"] == sel["selected_round"]
