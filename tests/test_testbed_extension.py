"""The 2026-09-27 testbed extension: low_data seeds 789/1011, the MAXN ten-round
block with its bitwise identity gates, energy logging, and block notifications.

The identity gates are hard gates.  These tests pin that a mismatch

* is detected from the per-image prediction files and the aggregated validation
  loss, bit for bit (one flipped float is a failure, a half-written file is not);
* ends the block with STOPPING, is never retried, and leaves a marker;
* keeps the block from starting again, and keeps analyze_results from analysing
  the run, until a human removes the marker.
"""

import json
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from scripts import block_report  # noqa: E402
from scripts import print_revision_commands as prc  # noqa: E402
from scripts import run_matrix  # noqa: E402
from scripts.analyze_results import IdentityGateError, collect_runs  # noqa: E402

MAXN = {"node_a": "MAXN_SUPER", "node_b": "MAXN_SUPER", "node_c": "MAXN_SUPER"}


def _revision():
    return prc.apply_all_seeds(prc.load_config(prc.DEFAULT_CONFIG)["revision"])


# --------------------------------------------------------------------------- #
# matrix definition
# --------------------------------------------------------------------------- #
def test_low_data_gains_seeds_789_and_1011_in_the_main_namespace():
    runs = prc.expand_all(_revision(), "low_data")["low_data"]
    assert len(runs) == 40
    new = [r for r in runs if r.seed in (789, 1011)]
    assert len(new) == 16
    assert all(r.output_dir.startswith("results/rev_") for r in runs)   # pools with 42/123/456
    assert all(r.power_config is None for r in runs)


def test_maxn_block_is_30_ten_round_runs_in_pc_maxn():
    runs = prc.expand_all(_revision(), "maxn_long_horizon")["maxn_long_horizon"]
    assert len(runs) == 30
    assert {r.strategy for r in runs} == {"fedavg", "fedprox_0.01", "fedbn"}
    assert {r.dist for r in runs} == {"iid", "noniid"}
    assert {r.seed for r in runs} == {42, 123, 456, 789, 1011}
    assert all(r.rounds == 10 and r.local_epochs == 5 and r.lr == 0.001 for r in runs)
    assert all(r.output_dir.startswith("results/pc_maxn/rev_") and r.output_dir.count("pc_maxn") == 1
               for r in runs)
    assert all(r.power_config == "maxn" for r in runs)


def test_identity_gates_are_exactly_the_two_declared_sets():
    runs = prc.expand_all(_revision(), "maxn_long_horizon")["maxn_long_horizon"]
    n = 0
    for r in runs:
        want = []
        if r.strategy in ("fedavg", "fedprox_0.01"):
            want.append(("results/rev_%s_%s_seed%d" % (r.dist, r.strategy, r.seed), [1, 2, 3]))
        if r.strategy in ("fedavg", "fedbn") and r.dist == "noniid" and r.seed in (42, 123, 456):
            want.append(("results/rev_noniid_%s_r10_seed%d" % (r.strategy, r.seed),
                         list(range(1, 11))))
        assert r.identity == want, r.output_dir
        n += len(want)
        for ref, _ in r.identity:                      # the references are committed runs
            assert os.path.isfile(os.path.join(_REPO, ref, "results.json")), ref
    assert n == 26


def test_the_maxn_block_cannot_be_emitted_for_the_heterogeneous_testbed(capsys):
    with pytest.raises(SystemExit, match="declares power_config 'maxn'"):
        prc.main(["--all_seeds", "--block", "maxn_long_horizon", "--format", "bash",
                  "--power_config", "heterogeneous"])
    assert prc.main(["--all_seeds", "--block", "maxn_long_horizon", "--format", "bash",
                     "--no_skip_existing", "--power_config", "maxn"]) == 0
    out = capsys.readouterr().out
    assert out.count(">>> IDENTITY GATE:") == 26
    assert "results/pc_maxn/pc_maxn" not in out
    # every block at once, heterogeneous named explicitly: the maxn block is left out
    assert prc.main(["--all_seeds", "--format", "bash", "--no_skip_existing",
                     "--power_config", "heterogeneous"]) == 0
    captured = capsys.readouterr()
    assert "pc_maxn" not in captured.out
    assert "skipping block 'maxn_long_horizon'" in captured.err


def test_run_matrix_parses_the_gates_it_must_enforce(tmp_path, capsys):
    assert prc.main(["--all_seeds", "--block", "maxn_long_horizon", "--format", "bash",
                     "--no_skip_existing", "--power_config", "maxn"]) == 0
    path = tmp_path / "b.sh"
    path.write_text(capsys.readouterr().out, encoding="utf-8")
    runs = run_matrix.parse_block(str(path))
    assert len(runs) == 30
    by = {r["out_dir"]: r["gates"] for r in runs}
    assert by["results/pc_maxn/rev_noniid_fedavg_r10_seed42"] == [
        ("results/rev_noniid_fedavg_seed42", [1, 2, 3]),
        ("results/rev_noniid_fedavg_r10_seed42", list(range(1, 11)))]
    assert by["results/pc_maxn/rev_iid_fedbn_r10_seed42"] == []
    run_matrix.check_namespace(runs, "maxn")


# --------------------------------------------------------------------------- #
# bitwise comparison
# --------------------------------------------------------------------------- #
def _write_preds(run_dir, rounds, seed=0, flip=None):
    """Prediction files for every client/split/round; ``flip=(round, node, split)``
    changes one logit margin by one ulp."""
    os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)
    for rnd in rounds:
        for node in run_matrix.PRED_NODES:
            for split in run_matrix.PRED_SPLITS:
                rng = np.random.default_rng(hash((seed, rnd, node, split)) % 2**32)
                m = rng.normal(size=50).astype(np.float32)
                if flip == (rnd, node, split):
                    m[7] = np.nextafter(m[7], np.float32(np.inf))
                np.savez(run_matrix.pred_file(run_dir, rnd, node, split),
                         path=np.array(["Fire/%d.jpg" % i for i in range(50)]),
                         label=(np.arange(50) % 2).astype(np.uint8),
                         logit_margin=m, p_fire=(1 / (1 + np.exp(-m))).astype(np.float32),
                         format_version=np.int64(1))


def _write_results(run_dir, losses):
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"model_selection": {"selected_round": 1},
                   "rounds": [{"round": i + 1, "weighted_val_loss": v} for i, v in enumerate(losses)]},
                  f)


def test_identical_rounds_pass_and_one_ulp_fails(tmp_path):
    ref, got = str(tmp_path / "ref"), str(tmp_path / "got")
    _write_preds(ref, [1, 2, 3])
    _write_preds(got, [1, 2, 3])
    assert run_matrix.compare_round(got, ref, 2)[0] == "same"
    _write_preds(got, [2], flip=(2, "node_c", "test"))
    state, detail = run_matrix.compare_round(got, ref, 2)
    assert state == "differ" and "node_c test" in detail and "logit_margin" in detail


def test_missing_or_half_written_files_are_pending_not_a_failure(tmp_path):
    ref, got = str(tmp_path / "ref"), str(tmp_path / "got")
    _write_preds(ref, [1])
    os.makedirs(os.path.join(got, "predictions"))
    assert run_matrix.compare_round(got, ref, 1)[0] == "pending"
    _write_preds(got, [1])
    f = run_matrix.pred_file(got, 1, "node_b", "val")
    data = open(f, "rb").read()
    open(f, "wb").write(data[: len(data) // 2])                  # the server is mid-write
    assert run_matrix.compare_round(got, ref, 1)[0] == "pending"


def test_monitor_catches_the_first_mismatching_round_as_it_appears(tmp_path, monkeypatch):
    monkeypatch.setattr(run_matrix, "LOG_DIR", str(tmp_path))
    ref, got = str(tmp_path / "ref"), str(tmp_path / "got")
    _write_preds(ref, [1, 2, 3])
    mon = run_matrix.IdentityMonitor({"out_dir": got, "gates": [(ref, [1, 2, 3])]})
    os.makedirs(os.path.join(got, "predictions"))
    assert mon.poll() is None
    _write_preds(got, [1])
    assert mon.poll() is None and mon.done == [(ref, 1)]
    _write_preds(got, [2], flip=(2, "node_a", "val"))
    reason = mon.poll()
    assert reason and "round 2 node_a val" in reason


def test_final_check_also_requires_the_aggregated_validation_loss(tmp_path):
    ref, got = str(tmp_path / "ref"), str(tmp_path / "got")
    _write_preds(ref, [1, 2, 3])
    _write_preds(got, [1, 2, 3, 4])
    _write_results(ref, [2.4, 1.1, 0.27])
    _write_results(got, [2.4, 1.1, 0.27, 0.2])
    run = {"out_dir": got, "gates": [(ref, [1, 2, 3])]}
    assert run_matrix.final_identity_check(run) == (True, "3 gated round(s) bitwise identical")
    _write_results(got, [2.4, 1.1, 0.27000000000000002 + 1e-15, 0.2])
    ok, why = run_matrix.final_identity_check(run)
    assert not ok and "round 3 aggregated validation loss" in why
    os.remove(run_matrix.pred_file(got, 2, "node_b", "test"))
    ok, why = run_matrix.final_identity_check({"out_dir": got, "gates": [(ref, [1, 2, 3])]})
    assert not ok and "round 2 missing" in why


def test_block_refuses_to_start_when_a_reference_is_incomplete(tmp_path):
    ref = str(tmp_path / "ref")
    _write_preds(ref, [1, 2])
    _write_results(ref, [1, 1])
    with pytest.raises(SystemExit, match="lacks r003_node_a_val"):
        run_matrix.check_gate_references([{"out_dir": "x", "gates": [(ref, [1, 2, 3])]}])


# --------------------------------------------------------------------------- #
# main loop: a failed gate stops the block, is not retried, and sticks
# --------------------------------------------------------------------------- #
@pytest.fixture
def block(tmp_path, monkeypatch):
    """A two-run gated block in tmp_path, with every testbed contact stubbed out."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_matrix.os, "chdir", lambda p: None)
    monkeypatch.setattr(run_matrix, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(run_matrix.time, "sleep", lambda s: None)
    nodes = {n: {"host": "192.168.1.10" if n == "node_a" else "h", "user": "u", "repo": "/r",
                 "venv": "/v", "local": n == "node_a"} for n in run_matrix.NODE_NAMES}
    monkeypatch.setattr(run_matrix, "load_testbed", lambda path: (nodes, 8080))
    monkeypatch.setattr(run_matrix, "load_power_modes", lambda path, pc: dict(MAXN))
    monkeypatch.setattr(run_matrix, "check_power", lambda expected: ([], dict(MAXN)))

    def preflight(strict_commit=True, splits=(), power=None, measured=None):
        if measured is not None:
            measured.update(MAXN)
        return []
    monkeypatch.setattr(run_matrix, "preflight", preflight)
    for i in (1, 2):
        _write_preds("results/rev_iid_fedavg_seed%d" % i, [1, 2, 3], seed=i)
        _write_results("results/rev_iid_fedavg_seed%d" % i, [2.0, 1.0, 0.5])
    lines = ["#!/bin/bash"]
    for i in (1, 2):
        out = "results/pc_maxn/rev_iid_fedavg_r10_seed%d" % i
        lines += ["echo '--- %s ---'" % out,
                  "echo '>>> IDENTITY GATE: results/rev_iid_fedavg_seed%d rounds 1,2,3'" % i]
        for node in run_matrix.NODE_NAMES:
            lines += ["echo '>>> START ON %s (x):'" % node,
                      "echo '  python3 src/fl/client.py --server 192.168.1.10:8080 "
                      "--data_dir data/processed/iid/%s --seed %d'" % (node, i)]
        lines += ["python3 src/fl/server.py --strategy fedavg --rounds 10 --seed %d "
                  "--output_dir %s --tag iid" % (i, out)]
    (tmp_path / "block.sh").write_text("\n".join(lines) + "\n")
    calls = []
    state = {"flip": None, "rc": None}

    def fake_execute(run, attempt, *a, **kw):
        calls.append((run["out_dir"], attempt))
        i = int(run["out_dir"][-1])
        if state["rc"] is not None:
            return state["rc"]
        _write_preds(run["out_dir"], range(1, 11), seed=i,
                     flip=state["flip"] if i == 1 else None)
        _write_results(run["out_dir"], [2.0, 1.0, 0.5] + [0.4] * 7)
        return 0, "server exited"
    monkeypatch.setattr(run_matrix, "execute_run", fake_execute)
    monkeypatch.setattr(sys, "argv", ["run_matrix.py", "--script", "block.sh",
                                      "--power_config", "maxn"])
    return {"calls": calls, "state": state, "tmp": tmp_path}


def _log(tmp):
    return (tmp / "logs" / "run_matrix.log").read_text()


def test_passing_gates_let_the_block_run_and_record_power(block):
    with pytest.raises(SystemExit) as e:
        run_matrix.main()
    assert e.value.code == 0
    assert [c[0][-1] for c in block["calls"]] == ["1", "2"]
    data = json.load(open("results/pc_maxn/rev_iid_fedavg_r10_seed1/results.json"))
    assert data["power"]["power_config"] == "maxn"
    assert "3 gated round(s) bitwise identical" in _log(block["tmp"])


def test_a_failed_gate_stops_the_block_without_retry_and_blocks_a_relaunch(block):
    block["state"]["flip"] = (3, "node_b", "test")
    with pytest.raises(SystemExit) as e:
        run_matrix.main()
    assert e.value.code == 1
    assert block["calls"] == [("results/pc_maxn/rev_iid_fedavg_r10_seed1", 1)]   # no retry, no run 2
    marker = "results/pc_maxn/rev_iid_fedavg_r10_seed1/" + run_matrix.GATE_MARKER
    assert "round 3 node_b test" in json.load(open(marker))["reason"]
    assert "STOPPING the block: IDENTITY GATE FAILED" in _log(block["tmp"])
    # the analysis refuses the run, and so does a relaunch of the block
    with pytest.raises(IdentityGateError):
        collect_runs("results", warn=False)
    block["calls"].clear()
    with pytest.raises(SystemExit) as e:
        run_matrix.main()
    assert e.value.code == 1 and block["calls"] == []
    assert "unresolved identity-gate failure" in _log(block["tmp"])


def test_a_mid_run_gate_failure_is_final_too(block):
    block["state"]["rc"] = (-5, "identity gate failed: round 1 node_a val: array 'logit_margin'")
    with pytest.raises(SystemExit) as e:
        run_matrix.main()
    assert e.value.code == 1
    assert len(block["calls"]) == 1
    assert os.path.isfile("results/pc_maxn/rev_iid_fedavg_r10_seed1/" + run_matrix.GATE_MARKER)


def test_execute_run_ends_the_run_when_the_monitor_reports(tmp_path, monkeypatch):
    """Through the real process loop, with stand-ins (see test_run_matrix_process)."""
    from tests.test_run_matrix_process import FAKE_CLIENT, FAKE_SERVER, _free_port
    server_py, client_py = tmp_path / "s.py", tmp_path / "c.py"
    server_py.write_text(FAKE_SERVER)
    client_py.write_text(FAKE_CLIENT)
    port = _free_port()
    monkeypatch.setattr(run_matrix, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(run_matrix, "SERVER_PORT", port)
    monkeypatch.setattr(run_matrix, "NODES", {
        n: {"host": "127.0.0.1", "user": "u", "repo": "/r", "venv": "/v", "local": n == "node_a"}
        for n in run_matrix.NODE_NAMES})
    monkeypatch.setattr(run_matrix, "kill_stragglers", lambda: None)
    monkeypatch.setattr(run_matrix, "start_server", lambda run, lp: (run_matrix._popen(
        [sys.executable, str(server_py), str(port), "0", "600", "0", str(tmp_path / "s")],
        open(lp, "ab")), open(lp, "ab")))
    monkeypatch.setattr(run_matrix, "run_on", lambda node, inner, lp: (run_matrix._popen(
        [sys.executable, str(client_py), str(port), "hang", str(tmp_path / node)],
        open(lp, "ab")), open(lp, "ab")))

    class Mon:
        n = 0

        def poll(self):
            self.n += 1
            return "round 1 node_a val differs" if self.n >= 3 else None
    times = {}
    rc, why = run_matrix.execute_run({"out_dir": "x", "server": "s",
                                      "clients": {n: n for n in run_matrix.NODE_NAMES}},
                                     1, ready_timeout=30, run_timeout=3600, poll=0.2,
                                     monitor=Mon(), times=times)
    assert rc == -5 and why == "identity gate failed: round 1 node_a val differs"
    assert "server_start" in times


# --------------------------------------------------------------------------- #
# energy
# --------------------------------------------------------------------------- #
TEGRA = """\
09-27-2026 23:15:26 RAM 980/7620MB (lfb 77x4MB) SWAP 33/3810MB (cached 0MB) CPU [0%@729,0%@729] GR3D_FREQ 0% cpu@49.625C VDD_IN 5417mW/5417mW VDD_CPU_GPU_CV 481mW/481mW VDD_SOC 2608mW/2608mW
09-27-2026 23:15:27 RAM 980/7620MB (lfb 77x4MB) SWAP 33/3810MB (cached 0MB) CPU [0%@729,0%@729] GR3D_FREQ 0% cpu@49.468C VDD_IN 5377mW/5397mW VDD_CPU_GPU_CV 481mW/481mW VDD_SOC 2608mW/2608mW
09-27-2026 23:15:28 RAM 980/7620MB (lfb 77x4MB) SWAP 33/3810MB (cached 0MB) CPU [off,off] GR3D_FREQ 0% cpu@49.468C VDD_IN 12000mW/7598mW VDD_CPU_GPU_CV 481mW/481mW VDD_SOC 2608mW/2608mW
garbage line
"""


def test_tegrastats_lines_parse_to_instantaneous_board_power():
    s = run_matrix.parse_tegrastats(TEGRA)
    assert [mw for _, mw in s] == [5417, 5377, 12000]          # instantaneous, not the average
    assert s[1][0] - s[0][0] == 1.0


def test_energy_integrates_only_the_run_window():
    s = run_matrix.parse_tegrastats(TEGRA)
    t0 = s[0][0]
    e = run_matrix.integrate_energy(s, t0 + 1, t0 + 2, 1.0)
    assert e["n_samples"] == 2 and e["mean_W"] == pytest.approx(8.6885, abs=1e-3)
    # energy = mean power x window, so a missing sample is not counted as zero power
    assert e["energy_J"] == pytest.approx(8.7)
    e = run_matrix.integrate_energy(s, t0, t0 + 10, 1.0)
    assert e["coverage"] == pytest.approx(0.3) and e["energy_J"] == pytest.approx(75.98, abs=0.1)


def test_remote_stamps_are_shifted_by_the_nodes_own_zone():
    base = run_matrix.parse_tegrastats(TEGRA)
    shifted = run_matrix.parse_tegrastats(TEGRA, tz_shift=10800.0)     # node 3 h ahead
    assert [t for t, _ in shifted] == [t - 10800.0 for t, _ in base]


def test_kill_patterns_never_match_their_own_command_line():
    """pkill -f runs inside `bash -c "<cmd>"`; a pattern matching that shell's own
    command line would kill the shell.  The [s] bracket prevents it."""
    import re
    lg = run_matrix.EnergyLogger("rev_x", 1, "20260927_000000", 100)
    monkey_nodes = {"node_a": {"repo": "/home/u/FedRGBD"}}
    old = dict(run_matrix.NODES)
    run_matrix.NODES.clear()
    run_matrix.NODES.update(monkey_nodes)
    try:
        pat = lg._pattern("node_a")
        target = ("timeout 1900 tegrastats --interval 1000 --logfile "
                  "/home/u/FedRGBD/logs/energy/rev_x_try1_20260927_000000_node_a.tegrastats")
        assert re.search(pat, target)
        assert not re.search(pat, "bash -c pkill -f '%s' || true" % pat)
    finally:
        run_matrix.NODES.clear()
        run_matrix.NODES.update(old)
    generic = "tegrastat[s] --interval [0-9]+ --logfile .*/logs/energy/"
    assert re.search(generic, target)
    assert not re.search(generic, "tegrastats --interval 1000")        # someone else's instance


# --------------------------------------------------------------------------- #
# block notifications
# --------------------------------------------------------------------------- #
def test_a_grown_block_is_reported_again_and_old_reports_are_adopted(tmp_path):
    out = str(tmp_path / "low_data")
    os.makedirs(out)
    with open(os.path.join(out, "STATUS.md"), "w") as f:
        f.write("# Block\n\n- runs expected: **24**\n")
    runs24 = ["results/rev_%d" % i for i in range(24)]
    runs40 = ["results/rev_%d" % i for i in range(40)]
    assert block_report.reported_for(out, runs24)             # adopted, not re-fired
    assert os.path.isfile(os.path.join(out, "complete.json"))
    assert not block_report.reported_for(out, runs40)         # the grown set is new
    block_report.mark_reported(out, runs40)
    assert block_report.reported_for(out, runs40)


def test_newly_complete_is_printed_once(tmp_path, monkeypatch, capsys):
    status = {"low_data": {"expected": 2, "done": ["a", "b"], "missing": [], "complete": True},
              "maxn_long_horizon": {"expected": 30, "done": [], "missing": ["x"], "complete": False}}
    monkeypatch.setattr(block_report, "block_status", lambda repo: status)
    monkeypatch.setattr(block_report, "run_pipeline", lambda repo, name, out: ["ok"])
    monkeypatch.setattr(block_report, "write_status",
                        lambda repo, name, st, out, lines: os.path.join(out, "STATUS.md"))
    block_report.main(["--repo", str(tmp_path)])
    out = capsys.readouterr().out
    assert "NEWLY_COMPLETE low_data 2" in out and "COMPLETE low_data 2" in out
    block_report.main(["--repo", str(tmp_path)])
    out = capsys.readouterr().out
    assert "NEWLY_COMPLETE" not in out
    # every complete block is listed on every pass (the fetch job keeps its own state)
    assert "COMPLETE low_data 2" in out and "maxn_long_horizon" not in out


# --------------------------------------------------------------------------- #
# review fixes: a failed gate can never be taken for a finished run
# --------------------------------------------------------------------------- #
def test_failed_gate_renames_results_and_is_found_by_a_namespace_scan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_matrix, "LOG_DIR", str(tmp_path / "logs"))
    out = "results/pc_maxn/rev_noniid_fedbn_r10_seed42"
    os.makedirs(out)
    _write_results(out, [1.0])
    run_matrix.fail_gate({"out_dir": out, "gates": []}, "round 10 differs")
    assert not os.path.exists(os.path.join(out, "results.json"))        # not "existing"
    assert os.path.isfile(os.path.join(out, "results.gate_failed.json"))
    # a relaunch through --block regenerates the script without this run; the
    # namespace scan still finds it
    assert run_matrix.unresolved_gate_failures([], "results/pc_maxn") == [out]
    assert not block_report.is_finished(out)


def test_block_report_never_counts_a_marked_run(tmp_path):
    d = str(tmp_path / "rev_x")
    os.makedirs(d)
    _write_results(d, [1.0])
    assert block_report.is_finished(d)
    open(os.path.join(d, run_matrix.GATE_MARKER), "w").write("{}")
    assert not block_report.is_finished(d)


def test_analysis_refuses_a_marker_even_without_results_json(tmp_path):
    root = tmp_path / "results"
    (root / "pc_maxn" / "rev_x" / "predictions").mkdir(parents=True)
    (root / "pc_maxn" / "rev_x" / run_matrix.GATE_MARKER).write_text("{}")
    with pytest.raises(IdentityGateError):
        collect_runs(str(root), warn=False)


def test_a_retry_is_judged_only_on_its_own_prediction_files(tmp_path, monkeypatch):
    monkeypatch.setattr(run_matrix, "INVALID_DIR", str(tmp_path / "logs" / "invalid_runs"))
    ref, got = str(tmp_path / "ref"), str(tmp_path / "results" / "pc_maxn" / "rev_y")
    _write_preds(ref, [1, 2, 3])
    _write_preds(got, [1, 2])                                    # left by a failed attempt 1
    run = {"out_dir": got, "gates": [(ref, [1, 2, 3])]}
    assert run_matrix.gated_rounds_on_disk_differ(run) is None
    moved = run_matrix.set_aside_predictions(run, 2)
    assert moved and not os.path.exists(os.path.join(got, "predictions"))
    assert sorted(os.listdir(moved))[0].startswith("r001_")
    mon = run_matrix.IdentityMonitor(run)
    assert mon.poll() is None and mon.done == []                 # nothing stale is accepted
    _write_preds(got, [1], flip=(1, "node_a", "val"))
    assert "round 1 node_a val" in mon.poll()
    _write_preds(got, [1, 2], flip=(2, "node_b", "test"))
    assert "round 2 node_b test" in run_matrix.gated_rounds_on_disk_differ(run)


def test_a_retry_after_a_differing_earlier_attempt_is_a_gate_failure(block):
    """Attempt 1 dies (rc -3) after writing a mismatching round: no retry."""
    calls = block["calls"]

    def dying(run, attempt, *a, **kw):
        calls.append((run["out_dir"], attempt))
        _write_preds(run["out_dir"], [1], seed=int(run["out_dir"][-1]), flip=(1, "node_c", "val"))
        return -3, "client node_b exited with rc=1 before the server"
    run_matrix.execute_run = dying
    with pytest.raises(SystemExit) as e:
        run_matrix.main()
    assert e.value.code == 1
    assert calls == [("results/pc_maxn/rev_iid_fedavg_r10_seed1", 1)]
    assert os.path.isfile("results/pc_maxn/rev_iid_fedavg_r10_seed1/" + run_matrix.GATE_MARKER)
    assert "IDENTITY GATE FAIL in the failed attempt" in _log(block["tmp"])


def test_no_other_block_may_write_a_gated_maxn_directory(capsys):
    with pytest.raises(SystemExit, match="belongs to block 'maxn_long_horizon'"):
        prc.main(["--all_seeds", "--block", "long_horizon_fedbn", "--format", "bash",
                  "--power_config", "maxn"])


def test_the_maxn_block_report_exports_maxn_tables(tmp_path, monkeypatch):
    seen = []

    class P:
        returncode = 0
        stdout = "ok"
        stderr = ""
    monkeypatch.setattr(block_report.subprocess, "run", lambda cmd, **kw: seen.append(cmd) or P())
    block_report.run_pipeline(_REPO, "maxn_long_horizon", str(tmp_path))
    assert seen[1][-2:] == ["--power_config", "maxn"]
    seen.clear()
    block_report.run_pipeline(_REPO, "low_data", str(tmp_path))
    assert "--power_config" not in seen[1]
