"""scripts/fetch_results.ps1: time limits and pass bookkeeping (added 2026-10-08).

On the night of 2026-10-07 the hourly passes connected to node_a and then hung until
the task's 30-minute limit killed them, with nothing in fetch.log after "fetch start"
and no pop-up. These tests drive the real script (Windows PowerShell + Windows
OpenSSH) against local stand-ins:

* a fake SSH server that sends its banner and then never answers, so ssh.exe sits in
  the key exchange -- the stage a connect timeout does not cover;
* a closed port, for an unreachable node.

Pop-ups are logged as [POPUP] lines (FEDRGBD_FETCH_NO_POPUP=1); log and state go to a
temporary directory. Windows only.
"""

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime, timedelta

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO, "scripts", "fetch_results.ps1")
_SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"

pytestmark = pytest.mark.skipif(
    os.name != "nt" or not shutil.which("powershell.exe") or not os.path.isfile(_SSH),
    reason="needs Windows PowerShell and the Windows OpenSSH client")


class StallingSshServer:
    """Accepts, sends an OpenSSH banner, then reads and never answers."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        self.conns = []
        self._stop = False
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        self.sock.settimeout(0.2)
        while not self._stop:
            try:
                c, _ = self.sock.accept()
            except OSError:
                continue
            c.sendall(b"SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.10\r\n")
            self.conns.append(c)

    def close(self):
        self._stop = True
        for c in self.conns:
            c.close()
        self.sock.close()


def _closed_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _run(tmp_path, port, call_timeout_s=3, pass_timeout_s=60, state=None):
    cfg = {"host": "127.0.0.1", "port": port, "user": "jetson-a",
           "remote_repo": "/home/jetson-a/FedRGBD", "connect_timeout_s": 5,
           "call_timeout_s": call_timeout_s, "pass_timeout_s": pass_timeout_s,
           "stale_after_minutes": 150, "stale_alert_repeat_minutes": 180, "python": None}
    cfg_path = tmp_path / "fetch.config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    if state is not None:
        (log_dir / "fetch_pass.state.json").write_text(json.dumps(state), encoding="utf-8")
    env = dict(os.environ, FEDRGBD_FETCH_NO_POPUP="1")
    t0 = time.monotonic()
    p = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", _SCRIPT, "-ConfigPath", str(cfg_path), "-LogDir", str(log_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=180)
    seconds = time.monotonic() - t0
    log = (log_dir / "fetch.log").read_text(encoding="utf-8-sig")
    st = json.loads((log_dir / "fetch_pass.state.json").read_text(encoding="utf-8-sig"))
    return p.returncode, log, st, seconds


def _iso(dt):
    return dt.astimezone().isoformat()


def test_a_hung_ssh_session_is_killed_and_reported(tmp_path):
    srv = StallingSshServer()
    try:
        rc, log, st, seconds = _run(tmp_path, srv.port, call_timeout_s=3)
    finally:
        srv.close()
    assert rc == 3
    assert seconds < 60
    assert "[ERROR] FetchResults pass of" in log
    assert "aborted: ssh to jetson-a@127.0.0.1 (true): no answer within 3 s" in log
    popups = [l for l in log.splitlines() if "[POPUP]" in l]
    assert len(popups) == 1 and "no answer within 3 s" in popups[0]
    assert "Last successful pass: none recorded" in popups[0]
    assert st["last_end_status"] == "timeout" and "last_success" not in st


def test_the_pass_limit_caps_every_call(tmp_path):
    srv = StallingSshServer()
    try:
        rc, log, st, seconds = _run(tmp_path, srv.port, call_timeout_s=120, pass_timeout_s=4)
    finally:
        srv.close()
    assert rc == 3 and seconds < 60
    assert "the pass reached its 4 s limit" in log


def test_a_pass_that_never_finished_is_reported_by_the_next(tmp_path):
    started = datetime.now() - timedelta(hours=1)
    state = {"tracking_since": _iso(started - timedelta(days=1)),
             "last_success": _iso(started - timedelta(minutes=60)),
             "last_end": _iso(started - timedelta(minutes=59)), "last_end_status": "ok",
             "last_start": _iso(started), "last_start_pid": 999999}
    rc, log, st, _ = _run(tmp_path, _closed_port(), state=state)
    assert rc == 1                                    # unreachable
    msg = "FetchResults pass started %s never finished" % started.strftime("%Y-%m-%d %H:%M")
    assert any("[POPUP]" in l and msg in l for l in log.splitlines())
    assert st["last_end_status"] == "unreachable"


def test_an_old_last_success_raises_one_popup_per_repeat_interval(tmp_path):
    now = datetime.now()
    state = {"tracking_since": _iso(now - timedelta(days=1)),
             "last_success": _iso(now - timedelta(hours=5)),
             "last_start": _iso(now - timedelta(minutes=50)),
             "last_end": _iso(now - timedelta(minutes=49)), "last_end_status": "unreachable"}
    rc, log, st, _ = _run(tmp_path, _closed_port(), state=state)
    stale = [l for l in log.splitlines() if "[POPUP]" in l and "no successful fetch for 5 h" in l]
    assert len(stale) == 1 and "last pass ended: unreachable" in stale[0]
    assert "last_stale_alert" in st
    rc, log, st, _ = _run(tmp_path, _closed_port())      # next pass: no repeat yet
    assert len([l for l in log.splitlines() if "[POPUP]" in l and "no successful fetch" in l]) == 1
