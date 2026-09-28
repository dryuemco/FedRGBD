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
