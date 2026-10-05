"""scripts/resume_after_reboot.py: restart only an interrupted block, never after an end."""

import json
import os

import pytest

from scripts import resume_after_reboot as rar
from scripts import run_matrix

ARGV = ["--block", "maxn_long_horizon", "--power_config", "maxn"]


def _start(argv=ARGV, pid=4242):
    return "[2026-09-28 10:00:00] run_matrix start (pid %d): %s" % (pid, json.dumps(argv))


def _log(*lines):
    return "\n".join(lines) + "\n"


def test_interrupted_mid_run_is_the_only_restartable_state():
    text = _log(_start(), "[t] block script: logs/b.sh", "[t] pre-flight OK",
                "[t] === [1/30] results/pc_maxn/rev_iid_fedavg_r10_seed42 ===",
                "[t]     client up on node_c (pid 5380)")
    state, argv, _ = rar.classify(text)
    assert state == "interrupted" and argv == ARGV


@pytest.mark.parametrize("end,state", [
    ("[t] block finished: 30 ok, 0 failed, 0 not attempted", "finished"),
    ("[t] STOPPING the block: IDENTITY GATE FAILED for x: y", "gate_failed"),
    ("[t]     IDENTITY GATE FAIL: round 1 differs -- ending the run now", "gate_failed"),
    ("[t] STOPPING the block: x did not produce a valid result after two identical attempts.",
     "stopped"),
    ("[t] PRE-FLIGHT FAIL -- not starting:", "preflight_failed"),
    ("[t] PRE-FLIGHT FAIL -- determinism gate (a vs b): x. The block does not start", "preflight_failed"),
    ("[t]     pre-flight FAIL before this run:", "preflight_failed"),
    ("[t] nothing to do", "finished"),
])
def test_any_recorded_end_prevents_a_restart(end, state):
    got, _, why = rar.classify(_log(_start(), "[t] pre-flight OK", end))
    assert got == state and "recorded end" in why


def test_only_the_last_invocation_counts():
    older = [_start(pid=1), "[t] block finished: 16 ok, 0 failed, 0 not attempted"]
    assert rar.classify(_log(*older, _start(pid=2), "[t] pre-flight OK"))[0] == "interrupted"
    newer_end = [_start(pid=1), "[t] pre-flight OK", _start(pid=2), "[t] STOPPING the block: x"]
    assert rar.classify(_log(*newer_end))[0] == "stopped"


def test_a_log_without_start_lines_is_never_restarted():
    # logs written before the start line existed (e.g. the 5b gate failure)
    text = _log("[t] block script: logs/b.sh", "[t] === [1/30] x ===")
    assert rar.classify(text)[0] == "unknown"
    assert rar.classify("")[0] == "unknown"
    assert rar.classify(_log("[t] run_matrix start (pid 1): not json"))[0] == "unknown"


def test_options_and_block_script_path():
    o = rar.invocation_options(ARGV + ["--energy"])
    assert (o.block, o.power_config, o.script) == ("maxn_long_horizon", "maxn", None)
    assert rar.block_script(o) == os.path.join("logs", "block_maxn_long_horizon_pc_maxn.sh")
    o = rar.invocation_options(["--block", "low_data", "--power_config", "heterogeneous"])
    assert rar.block_script(o) == os.path.join("logs", "block_low_data.sh")
    o = rar.invocation_options(["--script", "logs/x.sh", "--power_config", "maxn"])
    assert rar.block_script(o) == "logs/x.sh"


def _block_script(path, out_dirs):
    lines = []
    for d in out_dirs:
        lines.append("echo '--- %s ---'" % d)
        for node, ip in (("node_a", "10"), ("node_b", "7"), ("node_c", "6")):
            lines += ["echo '>>> START ON %s (192.168.1.%s):'" % (node, ip),
                      "echo '  python3 src/fl/client.py --server 192.168.1.10:8080 "
                      "--data_dir data/processed/iid/%s'" % node]
        lines += ["python3 src/fl/server.py --strategy fedavg --rounds 2 --seed 42 "
                  "--min_clients 3 --output_dir %s --tag iid" % d]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_partial_runs_are_moved_finished_and_gated_ones_are_not(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    done, partial, gated, absent, bad = ("results/pc_maxn/rev_%s" % n for n in
                                         ("done", "partial", "gated", "absent", "bad"))
    for d in (done, partial, gated, bad):
        os.makedirs(os.path.join(d, "predictions"))
    with open(os.path.join(done, "results.json"), "w") as f:
        json.dump({"model_selection": {"selected_round": 2}}, f)
    with open(os.path.join(bad, "results.json"), "w") as f:
        f.write("{truncated")
    open(os.path.join(partial, "predictions", "r001_node_a_val.npz"), "w").close()
    open(os.path.join(gated, run_matrix.GATE_MARKER), "w").close()
    script = tmp_path / "b.sh"
    _block_script(script, [done, partial, gated, absent, bad])
    found = rar.partial_run_dirs(str(script))
    assert sorted(found) == sorted([partial, bad])
    moved = rar.move_partial(found, "20260928_101500")
    assert not os.path.exists(partial) and os.path.isdir(done) and os.path.isdir(gated)
    dest = os.path.join("results", "_interrupted", "20260928_101500", "pc_maxn", "rev_partial")
    assert (partial, dest) in moved
    assert os.path.isfile(os.path.join(dest, "predictions", "r001_node_a_val.npz"))


def test_dry_run_changes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)          # main() chdirs to REPO; restore cwd afterwards
    monkeypatch.setattr(rar, "REPO", str(tmp_path))
    os.makedirs(tmp_path / "logs")
    (tmp_path / "logs" / "run_matrix.log").write_text(_log(_start(), "[t] pre-flight OK"))
    monkeypatch.setattr(rar, "run_matrix_alive", lambda: False)
    assert rar.main(["--dry_run"]) == 0
    text = (tmp_path / "logs" / "resume.log").read_text()
    assert text.startswith("RESUME [") and "interrupted" in text and "nothing done" in text


def test_a_finished_block_is_reported_not_restarted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # main() chdirs to REPO; restore cwd afterwards
    monkeypatch.setattr(rar, "REPO", str(tmp_path))
    os.makedirs(tmp_path / "logs")
    (tmp_path / "logs" / "run_matrix.log").write_text(
        _log(_start(), "[t] block finished: 30 ok, 0 failed, 0 not attempted"))
    assert rar.main([]) == 0
    assert "no restart: last block state is finished" in (tmp_path / "logs" / "resume.log").read_text()


# --------------------------------------------------------------------------- #
# clock synchronisation before the pre-flight (POST_5B_CHECKLIST c1)
# --------------------------------------------------------------------------- #
FAKE_NODES = {"node_a": {"local": True, "venv": "/v"}, "node_b": {"local": False},
              "node_c": {"local": False}}


def _interrupted_repo(tmp_path, monkeypatch, sync_answers, offsets=None):
    """An interrupted block with fake nodes.  ``sync_answers[node]`` is the list of
    successive NTPSynchronized answers (the last one repeats).  Records every
    subprocess.run (the pre-flight and tmux) instead of running it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rar, "REPO", str(tmp_path))
    os.makedirs(tmp_path / "logs")
    (tmp_path / "logs" / "run_matrix.log").write_text(_log(_start(), "[t] pre-flight OK"))
    monkeypatch.setattr(rar, "run_matrix_alive", lambda: False)
    monkeypatch.setattr(run_matrix, "load_testbed", lambda path: (FAKE_NODES, 8080))
    monkeypatch.setattr(rar, "wait_for_nodes", lambda *a: True)
    monkeypatch.setattr(rar.time, "sleep", lambda s: None)
    polls = {n: 0 for n in FAKE_NODES}
    events = []

    def ntp(node):
        answers = sync_answers[node]
        a = answers[min(polls[node], len(answers) - 1)]
        polls[node] += 1
        events.append(("ntp", node, a))
        return a

    def run(argv, **kw):
        events.append(("run", argv))
        class R:                                  # the pre-flight fails: stop right there
            returncode, stdout = 1, "PRE-FLIGHT FAIL -- fake"
        return R()

    monkeypatch.setattr(rar, "ntp_synchronised", ntp)
    monkeypatch.setattr(rar, "measure_offset",
                        lambda n: (offsets or {"node_b": 0.0123, "node_c": -0.004})[n])
    monkeypatch.setattr(rar.subprocess, "run", run)
    return events


def test_no_preflight_or_restart_before_every_clock_says_yes(tmp_path, monkeypatch):
    k = 3
    events = _interrupted_repo(tmp_path, monkeypatch, {
        "node_a": ["yes"], "node_b": ["yes"], "node_c": ["no"] * k + ["yes"]})
    assert rar.main(["--poll", "1"]) == 1               # stops at the fake failing pre-flight
    runs = [i for i, e in enumerate(events) if e[0] == "run"]
    assert len(runs) == 1 and "--check_only" in events[runs[0]][1]
    before = [e for e in events[:runs[0]] if e[0] == "ntp" and e[1] == "node_c"]
    assert [e[2] for e in before] == ["no"] * k + ["yes"]
    log = (tmp_path / "logs" / "resume.log").read_text()
    assert "waiting for synchronised clocks" in log
    assert "clocks: node_a sync=yes, node_b +12 ms, node_c -4 ms" in log
    assert log.index("clocks: node_a") < log.index("no restart: pre-flight failed")


def test_clock_timeout_means_no_restart_and_a_log_line(tmp_path, monkeypatch):
    events = _interrupted_repo(tmp_path, monkeypatch, {
        "node_a": ["yes"], "node_b": ["no"], "node_c": ["yes"]})
    assert rar.main(["--clock_timeout", "0", "--poll", "1"]) == 1
    assert not [e for e in events if e[0] == "run"]       # no pre-flight, no tmux
    log = (tmp_path / "logs" / "resume.log").read_text()
    assert "clocks not synchronised after 0 s: node_a sync=yes, node_b sync=no, node_c sync=yes" in log
    assert "no restart: clocks not synchronised" in log
    assert "restarted" not in log


def test_clocks_line_format_and_unmeasurable_offset():
    line = rar.clocks_line("yes", {"node_b": 0.0004, "node_c": None})
    assert line == "clocks: node_a sync=yes, node_b +0 ms, node_c offset n/a"
    assert rar.clocks_line("yes", {"node_b": 1.25, "node_c": -0.3}) == \
        "clocks: node_a sync=yes, node_b +1250 ms, node_c -300 ms"


def test_offset_is_the_best_of_the_round_trips(monkeypatch):
    # three samples; the one with the shortest round trip decides
    clock = iter([0.0, 0.5,   10.0, 10.1,   20.0, 20.9])
    remote = iter(["0.30", "10.25", "21.0"])
    monkeypatch.setattr(rar.time, "time", lambda: next(clock))
    monkeypatch.setattr(rar, "_node_command", lambda node, cmd: next(remote) + "\n")
    off = rar.measure_offset("node_b", n_samples=3)
    assert abs(off - (10.25 - 10.05)) < 1e-9


def test_dry_run_reports_clocks_without_waiting(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rar, "REPO", str(tmp_path))
    os.makedirs(tmp_path / "logs")
    (tmp_path / "logs" / "run_matrix.log").write_text(
        _log(_start(), "[t] block finished: 30 ok, 0 failed, 0 not attempted"))
    monkeypatch.setattr(run_matrix, "load_testbed", lambda path: (FAKE_NODES, 8080))
    monkeypatch.setattr(rar, "ntp_synchronised", lambda n: "no" if n == "node_c" else "yes")
    monkeypatch.setattr(rar, "measure_offset", lambda n: 0.002)
    monkeypatch.setattr(rar.time, "sleep", lambda s: pytest.fail("a dry run never waits"))
    assert rar.main(["--dry_run"]) == 0
    log = (tmp_path / "logs" / "resume.log").read_text()
    assert "dry run: no restart: last block state is finished" in log
    assert "dry run: NTPSynchronized node_a=yes, node_b=yes, node_c=no" in log
    assert "clocks: node_a sync=yes, node_b +2 ms, node_c +2 ms" in log
