"""End-to-end Flower run on localhost (CPU, synthetic data, 2 clients, 2 rounds).

This is an integration test of ``src/fl/server.py`` + ``src/fl/client.py`` with
the real Flower 1.13.1 gRPC stack — NOT an experiment.  It checks that the
per-round, per-client metrics reach ``results.json`` in the v3 layout while the
v2 keys stay intact.  Server and clients run as subprocesses, launched in the
order scripts/run_matrix.py uses (server, wait for its port, clients); a
client started before the server dies, which is tested too.
Runtime ≈ 10-15 s per strategy.
"""

import json
import os
import socket
import subprocess
import sys
import textwrap
import time

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


def test_client_started_before_the_server_dies(tmp_path):
    """flwr 1.13.1's default gRPC-bidi client does NOT retry a refused first connection
    (its RetryInvoker is unused on that transport), whatever the start_client docstring
    says about max_retries=None.  This is why scripts/run_matrix.py starts the server
    first and waits for its port."""
    node = _make_node(str(tmp_path / "node_a"), 0)
    code = CLIENT_SNIPPET.format(repo=_REPO, data_dir=node, img=IMG, port=_free_port())
    r = subprocess.run([sys.executable, "-c", code], cwd=_REPO, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, timeout=120)
    assert r.returncode != 0
    assert "UNAVAILABLE" in r.stdout


@pytest.mark.parametrize("strategy", ["fedprox_0.01", "fedbn"])
def test_two_client_run_writes_v3_results(tmp_path, strategy):
    """Launch order as in scripts/run_matrix.py: server, wait for its port, clients."""
    from scripts.run_matrix import wait_for_port

    nodes = [_make_node(str(tmp_path / f"node_{c}"), i) for i, c in enumerate("ab")]
    port = _free_port()
    out_dir = tmp_path / "run"
    rounds = 2

    # output goes to files: an unread PIPE fills up with Flower's logging and blocks the
    # processes while this test polls them
    logs = [open(tmp_path / name, "w+") for name in ("server.log", "client0.log", "client1.log")]
    server = subprocess.Popen(
        [sys.executable, "src/fl/server.py", "--strategy", strategy, "--rounds", str(rounds),
         "--address", f"127.0.0.1:{port}", "--output_dir", str(out_dir), "--min_clients", "2",
         "--seed", "42", "--tag", "unit_test"],
        cwd=_REPO, stdout=logs[0], stderr=subprocess.STDOUT)
    procs, exited = [], {}
    try:
        ready, why = wait_for_port("127.0.0.1", port, timeout=120, server=server, poll=0.2)
        assert ready, why
        for i, data_dir in enumerate(nodes):
            code = CLIENT_SNIPPET.format(repo=_REPO, data_dir=data_dir, img=IMG, port=port)
            procs.append(subprocess.Popen([sys.executable, "-c", code], cwd=_REPO,
                                          stdout=logs[1 + i], stderr=subprocess.STDOUT))
        # record the real shutdown order (run_matrix treats rc=0 clients as finished)
        deadline = time.time() + 300
        while len(exited) < 3 and time.time() < deadline:
            for name, p in [("server", server)] + [("client%d" % i, p) for i, p in enumerate(procs)]:
                if name not in exited and p.poll() is not None:
                    exited[name] = time.time()
            time.sleep(0.05)
    finally:
        for p in procs + [server]:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=60)
    outputs = []
    for f in logs:
        f.seek(0)
        outputs.append(f.read())
        f.close()
    for p, out in zip([server] + procs, outputs):
        assert p.returncode == 0, out[-3000:]
    # clients end normally no later than the server (Flower disconnects them, then the
    # server writes results.json and exits)
    assert max(exited["client0"], exited["client1"]) <= exited["server"] + 0.5, exited

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

    # per-image predictions: one file per round, client and split; none in results.json
    from src.evaluation.predictions import load_npz, metrics_from_predictions
    pred_dir = out_dir / "predictions"
    expected = {"README.md"} | {"r%03d_%s_%s.npz" % (r, n, s) for r in (1, 2)
                                for n in ("node_a", "node_b") for s in ("val", "test")}
    assert set(os.listdir(pred_dir)) == expected
    assert "_npz" not in (out_dir / "results.json").read_text()
    for entry in res["rounds"]:
        agg = entry["evaluate"]["aggregate"]
        acc, n = 0.0, 0
        for node in ("node_a", "node_b"):
            d = load_npz(str(pred_dir / ("r%03d_%s_test.npz" % (entry["round"], node))))
            acc += metrics_from_predictions(d["label"], d["logit_margin"])["accuracy"] * len(d["label"])
            n += len(d["label"])
        assert acc / n == pytest.approx(agg["test_accuracy"], abs=1e-12)

    # and the analysis reads it back under the declared rule
    from scripts.analyze_results import load_run
    run = load_run(str(out_dir), warn=False)
    assert run["headline_source"] == "selected_round_test"
    assert run["selected_round"] == sel["selected_round"]
