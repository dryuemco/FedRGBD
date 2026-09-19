"""run_matrix.execute_run with real processes (no SSH, no Flower, CPU only).

Stand-ins for the FL server and clients reproduce the testbed behaviour that
broke the first unattended run: a client makes ONE connection attempt and dies
on "Connection refused" (flwr 1.13.1's gRPC-bidi client does not retry).  The
runner must therefore start the server first, wait until its port accepts TCP
connections, and only then start the clients; and it must end a run as soon as
a client dies instead of waiting for the timeout.
"""

import os
import socket
import sys
import textwrap
import time

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from scripts import run_matrix  # noqa: E402

FAKE_SERVER = textwrap.dedent("""
    import socket, sys, time
    port, delay, run_s, rc, stamp = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), \\
        int(sys.argv[4]), sys.argv[5]
    time.sleep(delay)
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port)); s.listen(16); s.settimeout(0.1)
    open(stamp, "w").write(repr(time.time()))
    print("server listening", flush=True)
    end, conns = time.time() + run_s, []
    while time.time() < end:
        try:
            conns.append(s.accept()[0])
        except OSError:
            pass
    print("server done", flush=True)
    sys.exit(rc)
""")

FAKE_CLIENT = textwrap.dedent("""
    import socket, sys, time
    port, mode, stamp = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    open(stamp, "w").write(repr(time.time()))
    try:
        c = socket.create_connection(("127.0.0.1", port), timeout=2)   # ONE attempt, like flwr
    except OSError as e:
        print("grpc StatusCode.UNAVAILABLE Connection refused: %s" % e, flush=True)
        sys.exit(1)
    print("client connected", flush=True)
    if mode == "die":
        print("Traceback (most recent call last):", flush=True)
        print("RuntimeError: boom in fit()", flush=True)
        sys.exit(1)
    if mode == "hang":
        time.sleep(3600)
    time.sleep(float(mode))            # normal client: finish after <mode> seconds
    sys.exit(0)
""")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Point run_matrix at local stand-in processes; returns a configure() helper."""
    server_py = tmp_path / "fake_server.py"
    client_py = tmp_path / "fake_client.py"
    server_py.write_text(FAKE_SERVER)
    client_py.write_text(FAKE_CLIENT)
    port = _free_port()
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(run_matrix, "LOG_DIR", str(logs))
    monkeypatch.setattr(run_matrix, "SERVER_PORT", port)
    monkeypatch.setattr(run_matrix, "NODES", {
        n: {"host": "127.0.0.1", "user": "u", "repo": "/r", "venv": "/v", "local": n == "node_a"}
        for n in ("node_a", "node_b", "node_c")})
    monkeypatch.setattr(run_matrix, "kill_stragglers", lambda: None)
    state = {"server": ("0", "5", "0"), "clients": {}, "client_calls": []}

    def fake_start_server(run, log_path):
        delay, run_s, rc = state["server"]
        out = open(log_path, "ab")
        argv = [sys.executable, str(server_py), str(port), delay, run_s, rc,
                str(tmp_path / "server.stamp")]
        return run_matrix._popen(argv, out), out

    def fake_run_on(node, inner, log_path):
        state["client_calls"].append(node)
        out = open(log_path, "ab")
        argv = [sys.executable, str(client_py), str(port), state["clients"].get(node, "1"),
                str(tmp_path / ("%s.stamp" % node))]
        return run_matrix._popen(argv, out), out

    monkeypatch.setattr(run_matrix, "start_server", fake_start_server)
    monkeypatch.setattr(run_matrix, "run_on", fake_run_on)
    run = {"out_dir": "results/rev_iid_fedavg_seed42", "server": "srv",
           "clients": {"node_a": "a", "node_b": "b", "node_c": "c"}}
    state.update(run=run, tmp=tmp_path, logs=logs)
    return state


def _stamp(state, name):
    return float((state["tmp"] / ("%s.stamp" % name)).read_text())


def _runner_log(state):
    return (state["logs"] / "run_matrix.log").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# the stand-in behaves like the testbed client
# --------------------------------------------------------------------------- #
def test_stand_in_client_dies_when_started_before_the_server(harness, tmp_path):
    """Sanity check of the model: the pre-fix order (clients first) kills the client."""
    import subprocess

    port = run_matrix.SERVER_PORT
    r = subprocess.run([sys.executable, str(tmp_path / "fake_client.py"), str(port), "1",
                        str(tmp_path / "x.stamp")], stdout=subprocess.PIPE, text=True)
    assert r.returncode == 1
    assert "UNAVAILABLE" in r.stdout


# --------------------------------------------------------------------------- #
# 1. server first, clients only once the port accepts connections
# --------------------------------------------------------------------------- #
def test_clients_start_only_after_the_server_accepts_connections(harness):
    harness["server"] = ("1.5", "4", "0")      # listens only after 1.5 s, then runs 4 s
    rc, why = run_matrix.execute_run(harness["run"], 1, ready_timeout=30, run_timeout=60,
                                     poll=0.2)
    assert (rc, why) == (0, "server exited")
    listening = _stamp(harness, "server")
    for node in ("node_a", "node_b", "node_c"):
        assert _stamp(harness, node) >= listening, node        # never before the listen
    for log in harness["logs"].glob("*_node_*.log"):
        text = log.read_text()
        assert "client connected" in text and "refused" not in text, log.name
    assert "server accepting connections" in _runner_log(harness)


def test_server_that_never_listens_fails_the_run_without_starting_clients(harness):
    harness["server"] = ("3600", "1", "0")      # never gets to listen
    t0 = time.time()
    rc, why = run_matrix.execute_run(harness["run"], 1, ready_timeout=2, run_timeout=60,
                                     poll=0.2)
    assert rc == -2 and "did not accept connections within 2s" in why
    assert harness["client_calls"] == []
    assert time.time() - t0 < 15
    assert "SERVER NOT READY" in _runner_log(harness)


def test_server_crashing_before_it_listens_is_reported(harness, monkeypatch):
    def crashing_server(run, log_path):
        out = open(log_path, "ab")
        argv = [sys.executable, "-c", "import sys; print('crash'); sys.exit(3)"]
        return run_matrix._popen(argv, out), out

    monkeypatch.setattr(run_matrix, "start_server", crashing_server)
    t0 = time.time()
    rc, why = run_matrix.execute_run(harness["run"], 1, ready_timeout=60, run_timeout=60,
                                     poll=0.2)
    assert rc == -2 and "rc=3" in why
    assert time.time() - t0 < 15                 # noticed the exit, did not wait 60 s
    assert harness["client_calls"] == []


# --------------------------------------------------------------------------- #
# 2. fail fast on a dead client
# --------------------------------------------------------------------------- #
def test_a_dead_client_ends_the_run_at_once_with_its_log_tail(harness):
    harness["server"] = ("0", "600", "0")       # would run 10 min
    harness["clients"] = {"node_a": "hang", "node_b": "die", "node_c": "hang"}
    t0 = time.time()
    rc, why = run_matrix.execute_run(harness["run"], 1, ready_timeout=30, run_timeout=3600,
                                     poll=0.2)
    elapsed = time.time() - t0
    assert rc == -3 and "node_b" in why and "rc=1" in why
    assert elapsed < 20, elapsed                 # not the 10-min server, not the timeout
    log = _runner_log(harness)
    assert "CLIENT DIED" in log and "RuntimeError: boom in fit()" in log


def test_clients_ending_normally_before_the_server_are_not_a_failure(harness):
    """Flower disconnects the clients first, then the server writes results and exits."""
    harness["server"] = ("0", "3", "0")
    harness["clients"] = {"node_a": "0.5", "node_b": "0.5", "node_c": "0.5"}
    rc, why = run_matrix.execute_run(harness["run"], 1, ready_timeout=30, run_timeout=60,
                                     poll=0.2)
    assert (rc, why) == (0, "server exited")
    assert "all clients finished" in _runner_log(harness)


def test_server_hanging_after_all_clients_finished_is_ended(harness):
    harness["server"] = ("0", "600", "0")
    harness["clients"] = {"node_a": "0.2", "node_b": "0.2", "node_c": "0.2"}
    rc, why = run_matrix.execute_run(harness["run"], 1, ready_timeout=30, run_timeout=3600,
                                     poll=0.2, finish_grace=2)
    assert rc == -4 and "after every client finished" in why


def test_timeout_still_applies(harness):
    harness["server"] = ("0", "600", "0")
    harness["clients"] = {"node_a": "hang", "node_b": "hang", "node_c": "hang"}
    rc, why = run_matrix.execute_run(harness["run"], 1, ready_timeout=30, run_timeout=2,
                                     poll=0.2)
    assert rc == -1 and "TIMEOUT" in why


# --------------------------------------------------------------------------- #
# the address the runner polls is the address the clients dial
# --------------------------------------------------------------------------- #
def test_client_server_address_must_match_the_configured_node_a(monkeypatch):
    monkeypatch.setattr(run_matrix, "SERVER_PORT", 8080)
    monkeypatch.setattr(run_matrix, "NODES", {"node_a": {"host": "192.168.1.10"}})
    ok = [{"clients": {"node_a": "python3 src/fl/client.py --server 192.168.1.10:8080 --x"}}]
    run_matrix.check_server_address(ok)
    bad = [{"clients": {"node_a": "python3 src/fl/client.py --server 192.168.1.4:8080 --x"}}]
    with pytest.raises(SystemExit, match="192.168.1.4:8080"):
        run_matrix.check_server_address(bad)
