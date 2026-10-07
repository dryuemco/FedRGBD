#!/usr/bin/env python3
"""FedRGBD -- start one camera-experiment capture on all three nodes at once (node_a).

``docs/CAMERA_EXPERIMENT_PREREG.md`` sec. 2: the three cameras record simultaneously;
node_a starts every capture on all three nodes at a common scheduled wall-clock time.
SSH logins to node_b have been measured at 2-4 s, so a capture never starts on command
arrival: the start is scheduled ``--lead`` seconds ahead (default 15), every node gets
the same ``--start_at``, opens its camera, and waits for that time.  Remote commands go
over one OpenSSH ControlMaster connection per node, opened before the start is
scheduled.

    # reachability, camera SDK per node, clock offset vs node_a, camera detected
    python scripts/camera_capture_session.py --check

    # one capture: (scene, label, distance) on node_a/node_b (RealSense) and node_c (ZED);
    # no-fire first (it fixes the exposure lock of this scene x distance), then fire
    python scripts/camera_capture_session.py --scene s01 --label no_fire --distance_m 2 \\
        --notes "source=none; distance_measured_m=2.01; flame_height_cm=0"
    python scripts/camera_capture_session.py --scene s01 --label fire --distance_m 2 \\
        --notes "source=torch1; distance_measured_m=2.01; flame_height_cm=12"

    # print the exact commands, run nothing
    python scripts/camera_capture_session.py --scene s01 --label fire --distance_m 2 --dry_run

After the nodes finish, every node's ``data/raw/camera/<node>/_captures/<capture_id>.json``
is read back and checked: n_frames_written == requested, the record belongs to this
session (same start_scheduled_unix), and start_actual within 1.0 s of the schedule on
every node; the record is complete, the camera was on USB 3.x, and the capture ran under
the exposure lock with auto exposure off in every colour frame (prereg sec. 13 draft).
``--notes`` is required and parsed before anything starts.  One line per node, then PASS
or FAIL.  Every session is appended to
``data/raw/camera/session_log.jsonl`` on node_a.  Hosts, users, repos and venvs come from
``configs/testbed.local.yaml`` (``scripts/run_matrix.load_testbed``).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.run_matrix import TESTBED_LOCAL, load_testbed  # noqa: E402
from src.data.camera_capture_common import (  # noqa: E402
    CAPTURES_DIR, DEFAULT_FPS, DEFAULT_FRAMES, DEFAULT_ROOT, LABELS, NODE_NAMES,
    NODE_SENSORS, NOTES_FORMAT, START_TOLERANCE_S, make_capture_id, notes_problems,
    usb_problems, validate_distance,
)

#: node -> capture script (inside the node's venv, from the node's repo root)
CAPTURE_SCRIPT = {
    "node_a": "src/data/realsense_capture.py",
    "node_b": "src/data/realsense_capture.py",
    "node_c": "src/data/zed_capture.py",
}
#: the camera SDK each node must import
EXPECTED_SDK = {"node_a": "pyrealsense2", "node_b": "pyrealsense2", "node_c": "pyzed"}
#: the camera each node must have (S/N as the SDK reports it; docs/HARDWARE_SETUP.md)
EXPECTED_SERIAL = {"node_a": "239722070442", "node_b": "405622076256", "node_c": "35201583"}
PROBE_CMD = "python src/data/camera_capture_common.py --probe"
CLOCK_CMD = 'python3 -c "import time; print(repr(time.time()))"'
CONTROL_PATH = "~/.ssh/cm-fedrgbd-%r@%h:%p"
#: 25 s: the nodes open the camera and, for a no-fire capture, let auto exposure settle
#: before the exposure lock is read (prereg sec. 13 draft); 15 s left no margin for that
DEFAULT_LEAD_S = 25.0
SESSION_LOG = "session_log.jsonl"


# --------------------------------------------------------------------------- #
# command construction
# --------------------------------------------------------------------------- #
def ssh_opts(persist_s: int = 600) -> List[str]:
    return ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4",
            "-o", "ControlMaster=auto", "-o", "ControlPath=%s" % CONTROL_PATH,
            "-o", "ControlPersist=%d" % persist_s]


def node_shell(cfg: Dict, inner: str) -> str:
    """cd into the node's repo, activate its venv, run ``inner``."""
    return ("cd %s && source %s/bin/activate && %s"
            % (shlex.quote(str(cfg["repo"])), shlex.quote(str(cfg["venv"])), inner))


def node_argv(cfg: Dict, inner: str, in_repo: bool = True) -> List[str]:
    """argv running ``inner`` on the node: bash locally (node_a), ssh otherwise."""
    shell = node_shell(cfg, inner) if in_repo else inner
    if cfg.get("local"):
        return ["bash", "-lc", shell]
    return (["ssh"] + ssh_opts() + ["%s@%s" % (cfg["user"], cfg["host"]),
                                   "bash -lc %s" % shlex.quote(shell)])


def master_argv(cfg: Dict) -> List[str]:
    """Open (or reuse) the persistent ControlMaster connection to a remote node."""
    return ["ssh"] + ssh_opts() + ["%s@%s" % (cfg["user"], cfg["host"]), "true"]


def capture_inner(node: str, scene: str, label: str, distance_m: float, frames: int,
                  fps: float, start_at: float, root: str, retake: bool = False,
                  notes: str = "", no_lock: bool = False) -> str:
    parts = ["python", CAPTURE_SCRIPT[node],
             "--scene", scene, "--label", label, "--distance_m", "%.1f" % distance_m,
             "--frames", str(int(frames)), "--fps", "%g" % fps,
             "--start_at", "%.3f" % start_at, "--root", root, "--node", node]
    if retake:
        parts.append("--retake")
    if notes:
        parts += ["--notes", notes]
    if no_lock:
        parts.append("--no_lock")
    return " ".join(shlex.quote(p) for p in parts)


def build_commands(nodes: Dict[str, Dict], scene: str, label: str, distance_m: float,
                   frames: int, fps: float, start_at: float, root: str,
                   retake: bool = False, notes: str = "",
                   no_lock: bool = False) -> Dict[str, List[str]]:
    return {n: node_argv(nodes[n], capture_inner(n, scene, label, distance_m, frames, fps,
                                                 start_at, root, retake, notes, no_lock))
            for n in NODE_NAMES}


def record_relpath(root: str, node: str, capture_id: str) -> str:
    return "/".join([root.replace(os.sep, "/").rstrip("/"), node, CAPTURES_DIR,
                     capture_id + ".json"])


# --------------------------------------------------------------------------- #
# verification (pure)
# --------------------------------------------------------------------------- #
def verify_records(records: Dict[str, Optional[Dict]], start_scheduled: float,
                   n_requested: int, tol_s: float = START_TOLERANCE_S,
                   returncodes: Optional[Dict[str, Optional[int]]] = None,
                   expected_serials: Dict[str, str] = EXPECTED_SERIAL,
                   require_lock: bool = True) -> Tuple[bool, Dict[str, str]]:
    """-> (all nodes pass, {node: one-line summary}).

    A node passes iff its capture record exists, belongs to this session (its
    start_scheduled_unix equals the one sent), has n_frames_written == requested, a
    start_actual_unix within ``tol_s`` of the schedule, was recorded by the node's
    expected camera (serial), and its command exited 0; and (prereg sec. 13 draft) its
    status is complete, its camera was on a USB 3.x link, and -- unless ``require_lock``
    is False (``--no_lock``, e5 tests only) -- it ran under an exposure lock with auto
    exposure off in every colour frame.
    """
    ok_all = True
    lines: Dict[str, str] = {}
    for node in NODE_NAMES:
        rec = records.get(node)
        problems = []
        rc = (returncodes or {}).get(node)
        if rc not in (0, None):
            problems.append("exit code %s" % rc)
        if rec is None:
            problems.append("no capture record")
            desc = "-"
        else:
            written = rec.get("n_frames_written")
            sched = rec.get("start_scheduled_unix")
            actual = rec.get("start_actual_unix")
            if sched is None or abs(float(sched) - float(start_scheduled)) > 1e-3:
                problems.append("record is not from this session (scheduled %s)" % sched)
            if written != n_requested:
                problems.append("%s of %d frames" % (written, n_requested))
            if str(rec.get("serial")) != expected_serials[node]:
                problems.append("camera S/N %s, expected %s"
                                % (rec.get("serial"), expected_serials[node]))
            if rec.get("status") != "complete":
                problems.append("status %s" % rec.get("status"))
            usb = rec.get("usb") or {}
            if usb.get("problems") is None:
                problems.append("USB type not recorded")
            elif usb["problems"]:
                problems.append("; ".join(usb["problems"]))
            if require_lock:
                if not rec.get("exposure_lock"):
                    problems.append("no exposure lock")
                elif rec.get("ae_on_frames") or rec.get("ae_unknown_frames"):
                    problems.append("auto exposure on in %s / unreported in %s frames"
                                    % (rec.get("ae_on_frames"), rec.get("ae_unknown_frames")))
            if actual is None:
                problems.append("no start_actual")
                delay = "n/a"
            else:
                d = float(actual) - float(start_scheduled)
                delay = "%+.3f s" % d
                if abs(d) > tol_s:
                    problems.append("start %s off schedule (limit %.1f s)" % (delay, tol_s))
            desc = "%s/%d frames, start %s, %s %s" % (
                written, n_requested, delay, rec.get("camera_model"), rec.get("serial"))
        verdict = "PASS" if not problems else "FAIL (%s)" % "; ".join(problems)
        ok_all = ok_all and not problems
        lines[node] = "%-7s %-18s %s  %s" % (node, NODE_SENSORS[node], desc, verdict)
    return ok_all, lines


def clock_offset(samples: Sequence[Tuple[float, float, float]]) -> Tuple[float, float]:
    """(t_local_before, t_remote, t_local_after) samples -> (offset_s, rtt_s).

    offset = remote - midpoint(local before, local after) of the sample with the
    smallest round trip (the least queueing, so the tightest bound).
    """
    if not samples:
        raise ValueError("no clock samples")
    best = min(samples, key=lambda s: s[2] - s[0])
    t0, remote, t1 = best
    return remote - (t0 + t1) / 2.0, t1 - t0


def clock_gate(offsets: Dict[str, Optional[float]],
               limit_s: float = START_TOLERANCE_S) -> List[str]:
    """Problems with the node clocks, [] if every offset vs node_a is within ``limit_s``.

    The capture is scheduled on node_a's clock and each node starts on its own, so the
    pairing across cameras rests on the nodes agreeing (prereg sec. 2: NTP-synchronised,
    second-level synchronisation enough).  Right after a reboot a node can be minutes off
    until NTP syncs (node_a: ~22 min on 2026-09-28), so an unmeasured or larger offset
    stops the capture before anything is scheduled.  ``None`` = could not be measured.
    """
    problems = []
    for node, off in offsets.items():
        if off is None:
            problems.append("%s: clock offset could not be measured" % node)
        elif abs(off) > limit_s:
            problems.append("%s: clock %+.3f s off node_a (limit %.1f s); wait for NTP "
                            "(timedatectl: System clock synchronized: yes)"
                            % (node, off, limit_s))
    return problems


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def _run(argv: List[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=timeout)


def open_masters(nodes: Dict[str, Dict], log=print) -> bool:
    ok = True
    for n in NODE_NAMES:
        if nodes[n].get("local"):
            continue
        t0 = time.time()
        try:
            r = _run(master_argv(nodes[n]), timeout=30)
            good = r.returncode == 0
        except subprocess.TimeoutExpired:
            good, r = False, None
        log("  ssh %s: %s (%.2f s)" % (n, "connected" if good else "FAILED: %s"
                                         % ((r.stdout or "").strip()[:200] if r else "timeout"),
                                         time.time() - t0))
        ok = ok and good
    return ok


def measure_offset(cfg: Dict, n_samples: int = 5) -> Tuple[float, float]:
    if cfg.get("local"):
        return 0.0, 0.0
    argv = ["ssh"] + ssh_opts() + ["%s@%s" % (cfg["user"], cfg["host"]), CLOCK_CMD]
    samples = []
    for _ in range(n_samples):
        t0 = time.time()
        r = _run(argv, timeout=20)
        t1 = time.time()
        if r.returncode == 0:
            samples.append((t0, float(r.stdout.strip().splitlines()[-1]), t1))
    return clock_offset(samples)


def read_record(cfg: Dict, root: str, node: str, capture_id: str) -> Optional[Dict]:
    rel = record_relpath(root, node, capture_id)
    if cfg.get("local"):
        path = rel if os.path.isabs(rel) else os.path.join(_REPO, rel)
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    r = _run(node_argv(cfg, "cat %s" % shlex.quote(rel)), timeout=30)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except ValueError:
        return None


def append_session_log(root: str, entry: Dict) -> str:
    base = root if os.path.isabs(root) else os.path.join(_REPO, root)
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, SESSION_LOG)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return path


def run_check(nodes: Dict[str, Dict], dry_run: bool = False, log=print) -> int:
    if dry_run:
        for n in NODE_NAMES:
            cfg = nodes[n]
            if not cfg.get("local"):
                log("%s master : %s" % (n, " ".join(shlex.quote(a) for a in master_argv(cfg))))
            log("%s probe  : %s" % (n, " ".join(shlex.quote(a)
                                                 for a in node_argv(cfg, PROBE_CMD))))
        return 0
    if not open_masters(nodes, log=log):
        log("FAIL: ssh")
        return 1
    ok = True
    for n in NODE_NAMES:
        cfg = nodes[n]
        problems = []
        try:
            r = _run(node_argv(cfg, PROBE_CMD), timeout=60)
            probe = json.loads((r.stdout or "").strip().splitlines()[-1])
        except Exception as e:  # noqa: BLE001
            probe, problems = {}, ["probe failed: %s" % e]
        sdk = probe.get(EXPECTED_SDK[n], {})
        cams = sdk.get("cameras") or []
        if probe and not sdk.get("import"):
            problems.append("%s does not import: %s" % (EXPECTED_SDK[n], sdk.get("error")))
        elif probe and not cams:
            problems.append("no camera detected")
        elif probe and [str(c.get("serial")) for c in cams] != [EXPECTED_SERIAL[n]]:
            problems.append("cameras %s, expected exactly S/N %s"
                            % ([c.get("serial") for c in cams], EXPECTED_SERIAL[n]))
        elif probe:
            problems += usb_problems({"usb_type": cams[0].get("usb_type"),
                                      "usb_speed_mbps": cams[0].get("usb_speed_mbps")})
        try:
            off, rtt = measure_offset(cfg)
            clock = "offset %+.1f ms (rtt %.1f ms)" % (off * 1e3, rtt * 1e3)
            problems += clock_gate({n: off})
        except Exception as e:  # noqa: BLE001
            clock, problems = "offset n/a", problems + ["clock: %s" % e]
        others = [k for k in ("pyrealsense2", "pyzed") if probe.get(k, {}).get("import")]
        log("%-7s %-18s sdk=%s cameras=%s %s  %s" % (
            n, NODE_SENSORS[n], ",".join(others) or "none",
            json.dumps(cams), clock, "OK" if not problems else "FAIL (%s)"
            % "; ".join(problems)))
        ok = ok and not problems
    log("PASS" if ok else "FAIL")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene")
    p.add_argument("--label", choices=LABELS)
    p.add_argument("--distance_m", type=float)
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--lead", type=float, default=DEFAULT_LEAD_S,
                   help="seconds between scheduling and the common start (default %(default)s)")
    p.add_argument("--root", default=DEFAULT_ROOT.replace(os.sep, "/"),
                   help="capture root, relative to each node's repo (default %(default)s)")
    p.add_argument("--retake", action="store_true",
                   help="pass --retake to every node (old capture moved to _retakes/)")
    p.add_argument("--notes", default="",
                   help="required, parsed: %s; no-fire: source=none, flame_height_cm=0"
                   % NOTES_FORMAT.replace("%", "%%"))
    p.add_argument("--no_lock", action="store_true",
                   help="TEST ONLY (e5): no exposure lock on any node; never a study capture")
    p.add_argument("--timeout", type=float, default=None,
                   help="seconds to wait for the nodes (default lead + 3 x capture + 120)")
    p.add_argument("--testbed", default=TESTBED_LOCAL)
    p.add_argument("--check", action="store_true",
                   help="ssh, camera SDK, clock offset and camera per node; no capture")
    p.add_argument("--dry_run", action="store_true", help="print the commands, run nothing")
    return p


def main(argv=None, clock=time.time, log=print) -> int:
    args = build_parser().parse_args(argv)
    nodes, _ = load_testbed(args.testbed)
    if args.check:
        return run_check(nodes, dry_run=args.dry_run, log=log)
    if not (args.scene and args.label and args.distance_m is not None):
        log("--scene, --label and --distance_m are required (or use --check)")
        return 2
    capture_id = make_capture_id(args.scene, args.label, args.distance_m)
    distance = validate_distance(args.distance_m)
    bad_notes = notes_problems(args.notes, args.label)
    if bad_notes:
        for p in bad_notes:
            log("FAIL: " + p)
        log("FAIL: --notes incomplete; nothing was started")
        return 2

    if args.dry_run:
        start_at = round(clock() + args.lead, 3)
        log("# capture %s, start_at %.3f (now + %.0f s; recomputed on a real run)"
            % (capture_id, start_at, args.lead))
        for n in NODE_NAMES:
            if not nodes[n].get("local"):
                log("%s master : %s" % (n, " ".join(shlex.quote(a)
                                                    for a in master_argv(nodes[n]))))
        cmds = build_commands(nodes, args.scene, args.label, distance, args.frames,
                              args.fps, start_at, args.root, args.retake, args.notes,
                              args.no_lock)
        for n in NODE_NAMES:
            log("%s capture: %s" % (n, " ".join(shlex.quote(a) for a in cmds[n])))
        return 0

    log("capture %s: opening ssh connections" % capture_id)
    if not open_masters(nodes, log=log):
        log("FAIL: could not reach every node; nothing was started")
        return 1
    offsets: Dict[str, Optional[float]] = {}
    for n in NODE_NAMES:
        try:
            offsets[n] = measure_offset(nodes[n])[0]
        except Exception:  # noqa: BLE001
            offsets[n] = None
    log("clock offsets vs node_a: %s" % ", ".join(
        "%s %s" % (n, "n/a" if o is None else "%+.1f ms" % (o * 1e3))
        for n, o in offsets.items()))
    clock_problems = clock_gate(offsets)
    if clock_problems:
        for p in clock_problems:
            log("FAIL: " + p)
        log("FAIL: clocks not synchronised; nothing was started")
        return 1
    start_at = round(clock() + args.lead, 3)
    cmds = build_commands(nodes, args.scene, args.label, distance, args.frames, args.fps,
                          start_at, args.root, args.retake, args.notes, args.no_lock)
    log_dir = os.path.join(_REPO, "logs", "camera")
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(clock()))
    procs, outs = {}, {}
    for n in NODE_NAMES:
        outs[n] = open(os.path.join(log_dir, "%s_%s_%s.log" % (stamp, capture_id, n)), "ab")
        procs[n] = subprocess.Popen(cmds[n], stdout=outs[n], stderr=subprocess.STDOUT)
    log("started at %.3f, capture begins at %.3f (in %.1f s)"
        % (clock(), start_at, start_at - clock()))
    timeout = args.timeout if args.timeout is not None else (
        args.lead + 3.0 * args.frames / args.fps + 120.0)
    deadline = clock() + timeout
    rcs: Dict[str, Optional[int]] = {}
    for n in NODE_NAMES:
        try:
            rcs[n] = procs[n].wait(timeout=max(1.0, deadline - clock()))
        except subprocess.TimeoutExpired:
            procs[n].kill()
            rcs[n] = procs[n].wait()
            log("%s: timed out after %.0f s, killed" % (n, timeout))
        outs[n].close()
    records = {n: read_record(nodes[n], args.root, n, capture_id) for n in NODE_NAMES}
    ok, lines = verify_records(records, start_at, args.frames, returncodes=rcs,
                               require_lock=not args.no_lock)
    for n in NODE_NAMES:
        log(lines[n])
    if args.no_lock:
        log("NOTE: --no_lock -- a test capture, never a study capture")
    log("PASS" if ok else "FAIL  (logs: %s)" % log_dir)
    append_session_log(args.root, {
        "logged_unix": clock(), "host": socket.gethostname(), "capture_id": capture_id,
        "scene": args.scene, "label": args.label, "distance_m": distance,
        "frames": args.frames, "fps": args.fps, "lead_s": args.lead,
        "start_scheduled_unix": start_at, "retake": bool(args.retake),
        "clock_offset_s": offsets,
        "notes": args.notes, "verdict": "PASS" if ok else "FAIL",
        "no_lock": bool(args.no_lock),
        "nodes": {n: {"returncode": rcs.get(n), "summary": lines[n],
                      "record": records[n] and {k: records[n].get(k) for k in (
                          "n_frames_written", "start_actual_unix", "end_unix", "serial",
                          "camera_model", "status", "notes", "notes_parsed", "usb",
                          "exposure_lock", "ae_on_frames", "ae_unknown_frames")}}
                  for n in NODE_NAMES},
    })
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
