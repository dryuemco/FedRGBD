#!/usr/bin/env python3
"""FedRGBD -- A5 "desktop simulation sensitivity analysis" (docs/A5_DIRICHLET_DRAWS_PREREG.md
sections 3b and 6).

18 cells: alpha in {0.1, 0.5, 1} x draw in {ps42 (= the main study's ``dirichlet_<alpha>``),
ps123, ps456} x {FedAvg, FedProx(mu=0.01)}; 10 rounds, 5 local epochs, lr 1e-3, batch 8,
training seed 42.  The cells run one after another on the desktop GPU; each cell runs the
UNCHANGED ``src/fl/server.py`` and three ``src/fl/client.py`` processes on localhost
(node names node_a/b/c).  Nothing here changes the training code.

Data: the partitions are replayed from ``data/splits`` with ``--from_manifest`` (CLAUDE.md
rule 2) into ``--data_root`` (outside data/processed, so the desktop baselines' tree is
untouched), and each replayed ``manifest.csv`` must have the md5 of
``analysis/leakage/P0_SUMMARY.md`` before any cell starts.

Results: ``results/desktop_sim/sim_<partition>_<strategy>_r10_seed42/`` (results.json and
predictions/ as the testbed writes them, plus ``sim_info.json`` with the label, GPU,
library versions, commit, wall-clock).  ``scripts/analyze_results.py`` never reads that tree.

Pause: if ``--stop_file`` exists, no new cell starts (a running cell finishes) -- the
camera experiment's desktop work has priority (A5-2).

    python scripts/desktop_fl_sim.py replay --data_root D:/fedrgbd_sim/processed
    python scripts/desktop_fl_sim.py run --data_root D:/fedrgbd_sim/processed      # all 18
    python scripts/desktop_fl_sim.py run --data_root ... --cells dirichlet_0.1_ps123:fedavg
    python scripts/desktop_fl_sim.py list
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

LABEL = "desktop simulation sensitivity analysis"
ALPHAS = ("0.1", "0.5", "1")
DRAWS = ("ps42", "ps123", "ps456")
STRATEGIES = ("fedavg", "fedprox_0.01")
NODES = ("node_a", "node_b", "node_c")
HYPER = {"rounds": 10, "local_epochs": 5, "lr": 0.001, "batch_size": 8, "seed": 42}
SPLITS_DIR = os.path.join("data", "splits")
RESULTS_ROOT = os.path.join("results", "desktop_sim")
DEFAULT_PORT = 8091
DEFAULT_STOP_FILE = os.path.join("logs", "desktop_sim", "STOP")
#: lanes sharing the GPU while a cell ran (recorded: wall-clock is not comparable across)
LANES = 1


def partition(alpha: str, draw: str) -> str:
    """The data split name: ps42 is the main study's draw (no suffix)."""
    return "dirichlet_%s" % alpha if draw == "ps42" else "dirichlet_%s_%s" % (alpha, draw)


def tag(alpha: str, draw: str) -> str:
    return "dirichlet%s_%s" % (alpha, draw)


def cells() -> List[Tuple[str, str, str]]:
    """(alpha, draw, strategy) in run order: alpha, then draw, then strategy."""
    return [(a, d, s) for a in ALPHAS for d in DRAWS for s in STRATEGIES]


def out_dir(alpha: str, draw: str, strategy: str, root: str = RESULTS_ROOT) -> str:
    return os.path.join(root, "sim_%s_%s_r%d_seed%d" % (tag(alpha, draw), strategy,
                                                       HYPER["rounds"], HYPER["seed"]))


def server_argv(python: str, alpha: str, draw: str, strategy: str, port: int,
                output: str, rounds: int) -> List[str]:
    return [python, "src/fl/server.py", "--strategy", strategy, "--rounds", str(rounds),
            "--seed", str(HYPER["seed"]), "--min_clients", "3",
            "--address", "127.0.0.1:%d" % port, "--output_dir", output,
            "--tag", tag(alpha, draw)]


def client_argv(python: str, data_root: str, split: str, node: str, port: int,
                local_epochs: int) -> List[str]:
    return [python, "src/fl/client.py", "--server", "127.0.0.1:%d" % port,
            "--data_dir", os.path.join(data_root, split, node),
            "--batch_size", str(HYPER["batch_size"]), "--seed", str(HYPER["seed"]),
            "--local_epochs", str(local_epochs), "--lr", str(HYPER["lr"]),
            "--node_name", node]


# --------------------------------------------------------------------------- data
def needed_splits() -> List[str]:
    return sorted({partition(a, d) for a, d, _ in cells()})


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_problems(data_root: str, splits: Sequence[str]) -> List[str]:
    from scripts.run_matrix import expected_digests
    want = expected_digests()
    out = []
    for s in splits:
        p = os.path.join(data_root, s, "manifest.csv")
        if s not in want:
            out.append("%s: no reference md5 in P0_SUMMARY.md" % s)
        elif not os.path.isfile(p):
            out.append("%s: %s missing -- run `replay`" % (s, p))
        elif md5(p) != want[s]:
            out.append("%s: manifest md5 %s != P0_SUMMARY %s" % (s, md5(p)[:8], want[s][:8]))
    return out


def replay(data_root: str, splits: Sequence[str], python: str,
           raw: str = os.path.join("data", "raw", "flame_dataset")) -> int:
    """Replay only ``splits`` from data/splits (a copy of their manifests in a temp dir)."""
    tmp = tempfile.mkdtemp(prefix="fedrgbd_sim_manifests_")
    try:
        for s in splits:
            shutil.copy2(os.path.join(SPLITS_DIR, s + ".csv.gz"), tmp)
        shutil.copy2(os.path.join(SPLITS_DIR, "split_stats.json"), tmp)
        rc = subprocess.run([python, "src/data/data_splitter.py", "--from_manifest", tmp,
                             "--data_dir", raw, "--output_dir", data_root,
                             "--link_mode", "hardlink", "--clean", "--verify"],
                            cwd=REPO).returncode
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rc


# --------------------------------------------------------------------------- one cell
def wait_port(port: int, timeout: float, proc: subprocess.Popen) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            return False
        with socket.socket() as s:
            s.settimeout(1.0)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1.0)
    return False


def git_commit() -> Optional[str]:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, stdout=subprocess.PIPE,
                       text=True)
    return r.stdout.strip() or None


def run_cell(alpha: str, draw: str, strategy: str, data_root: str, python: str,
             port: int = DEFAULT_PORT, results_root: str = RESULTS_ROOT,
             rounds: int = HYPER["rounds"], local_epochs: int = HYPER["local_epochs"],
             log_dir: Optional[str] = None, timeout_s: float = 12 * 3600,
             split: Optional[str] = None) -> Dict:
    """One cell.  ``split`` overrides the partition (validation runs only)."""
    split = split or partition(alpha, draw)
    out = out_dir(alpha, draw, strategy, results_root)
    if rounds != HYPER["rounds"] or local_epochs != HYPER["local_epochs"] or split != partition(alpha, draw):
        out += "_VALIDATION_r%d_e%d_%s" % (rounds, local_epochs, split)
    if os.path.isfile(os.path.join(out, "results.json")):
        return {"cell": out, "status": "skipped (complete)"}
    log_dir = log_dir or os.path.join(REPO, "logs", "desktop_sim")
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = os.path.basename(out)
    t0 = time.perf_counter()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    logs = {}
    srv_log = open(os.path.join(log_dir, "%s_%s_server.log" % (name, stamp)), "wb")
    server = subprocess.Popen(server_argv(python, alpha, draw, strategy, port, out, rounds),
                              cwd=REPO, stdout=srv_log, stderr=subprocess.STDOUT)
    clients = {}
    try:
        if not wait_port(port, 120, server):
            raise RuntimeError("server did not accept connections on 127.0.0.1:%d" % port)
        for node in NODES:
            logs[node] = open(os.path.join(log_dir, "%s_%s_%s.log" % (name, stamp, node)), "wb")
            clients[node] = subprocess.Popen(
                client_argv(python, data_root, split, node, port, local_epochs),
                cwd=REPO, stdout=logs[node], stderr=subprocess.STDOUT)
        rc = server.wait(timeout=timeout_s)
        crc = {n: p.wait(timeout=300) for n, p in clients.items()}
    finally:
        for p in list(clients.values()) + [server]:
            if p.poll() is None:
                p.kill()
        for f in list(logs.values()) + [srv_log]:
            f.close()
    seconds = time.perf_counter() - t0
    ok = rc == 0 and all(v == 0 for v in crc.values()) and \
        os.path.isfile(os.path.join(out, "results.json"))
    info = {"label": LABEL, "prereg": "docs/A5_DIRICHLET_DRAWS_PREREG.md sec. 3b, A5-2",
            "alpha": alpha, "draw": draw, "partition": split, "strategy": strategy,
            "rounds": rounds, "local_epochs": local_epochs, "lr": HYPER["lr"],
            "batch_size": HYPER["batch_size"], "seed": HYPER["seed"],
            "server_rc": rc, "client_rc": crc, "ok": ok, "started": started,
            "wall_clock_s": round(seconds, 1), "parallel_lanes": LANES, "port": port,
            "commit": git_commit(),
            "host": socket.gethostname(), "python": platform.python_version(),
            "gpu": _gpu_name(python), "data_root": os.path.abspath(data_root)}
    if os.path.isdir(out):
        with open(os.path.join(out, "sim_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
    return {"cell": out, "status": "ok" if ok else "FAILED", "seconds": round(seconds, 1)}


def _gpu_name(python: str) -> Optional[str]:
    code = ("import torch\n"
            "try:\n    name = torch.cuda.get_device_name(0)\n"
            "except Exception as e:\n    name = 'cpu (%s)' % type(e).__name__\n"
            "print(torch.__version__, name)")
    r = subprocess.run([python, "-c", code], stdout=subprocess.PIPE,
                       stderr=subprocess.DEVNULL, text=True)
    return r.stdout.strip() or None


# --------------------------------------------------------------------------- CLI
def parse_cells(specs: Optional[Sequence[str]]) -> List[Tuple[str, str, str]]:
    if not specs:
        return cells()
    lookup = {"%s:%s" % (partition(a, d), s): (a, d, s) for a, d, s in cells()}
    bad = [x for x in specs if x not in lookup]
    if bad:
        raise SystemExit("unknown cell(s) %s; choose from %s" % (bad, sorted(lookup)))
    return [lookup[x] for x in specs]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    rp = sub.add_parser("replay")
    rp.add_argument("--data_root", required=True)
    rp.add_argument("--python", default=sys.executable)
    rp.add_argument("--extra_splits", nargs="*", default=[],
                    help="also replay these (e.g. a _sub split for a validation cell)")
    r = sub.add_parser("run")
    r.add_argument("--data_root", required=True)
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--cells", nargs="*", default=None, metavar="SPLIT:STRATEGY")
    r.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help="lane i uses port + i")
    r.add_argument("--lanes", type=int, default=1,
                   help="cells run in this many parallel lanes on the one GPU (each lane "
                        "sequential); the training code and settings are unchanged")
    r.add_argument("--results_root", default=RESULTS_ROOT)
    r.add_argument("--stop_file", default=DEFAULT_STOP_FILE)
    r.add_argument("--rounds", type=int, default=HYPER["rounds"],
                   help="VALIDATION only: other values write *_VALIDATION_* directories")
    r.add_argument("--local_epochs", type=int, default=HYPER["local_epochs"],
                   help="VALIDATION only")
    r.add_argument("--validation_split", default=None,
                   help="VALIDATION only: run on this split instead (e.g. dirichlet_0.1_sub0.01)")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        for a, d, s in cells():
            print("%-22s %-13s %s" % (partition(a, d), s, out_dir(a, d, s)))
        return 0
    if args.cmd == "replay":
        splits = needed_splits() + list(args.extra_splits)
        rc = replay(args.data_root, splits, args.python)
        bad = digest_problems(args.data_root, splits)
        for b in bad:
            print("DIGEST FAIL: " + b)
        print("replay rc=%d, digests %s" % (rc, "OK" if not bad else "FAIL"))
        return 0 if rc == 0 and not bad else 1
    validation = (args.rounds != HYPER["rounds"] or args.local_epochs != HYPER["local_epochs"]
                  or args.validation_split)
    if validation and os.path.normpath(args.results_root) == os.path.normpath(RESULTS_ROOT):
        raise SystemExit("a validation run never writes to %s; pass --results_root" % RESULTS_ROOT)
    todo = parse_cells(args.cells)
    splits = sorted({args.validation_split or partition(a, d) for a, d, _ in todo})
    bad = digest_problems(args.data_root, splits)
    if bad:
        for b in bad:
            print("PRE-FLIGHT FAIL: " + b)
        return 1
    global LANES
    LANES = max(1, args.lanes)
    results = run_lanes(todo, LANES, lambda a, d, s, port: run_cell(
        a, d, s, args.data_root, args.python, port, args.results_root, args.rounds,
        args.local_epochs, split=args.validation_split), args.port, args.stop_file)
    return 1 if any(r["status"] == "FAILED" for r in results) else 0


def lane_plan(todo: Sequence[Tuple[str, str, str]], lanes: int) -> List[List[Tuple[str, str, str]]]:
    """Cells dealt round-robin to ``lanes`` lanes, in run order (each lane sequential)."""
    return [list(todo[i::lanes]) for i in range(lanes)]


def run_lanes(todo, lanes: int, run_one, base_port: int, stop_file: str) -> List[Dict]:
    """Each lane runs its cells one after another on its own port (base_port + lane);
    before every cell the stop file is checked, so no lane starts a new cell once it
    exists (a running cell finishes)."""
    import threading
    results: List[Dict] = []
    lock = threading.Lock()

    def lane(i, cells_):
        for a, d, s in cells_:
            if os.path.exists(stop_file):
                print("lane %d: stop file %s present: no new cell (camera work has priority)"
                      % (i, stop_file), flush=True)
                return
            res = run_one(a, d, s, base_port + i)
            with lock:
                results.append(res)
                print("[%s] lane %d %s: %s %s" % (time.strftime("%H:%M:%S"), i, res["cell"],
                                                  res["status"], res.get("seconds", "")),
                      flush=True)

    threads = [threading.Thread(target=lane, args=(i, c)) for i, c in
               enumerate(lane_plan(todo, lanes)) if c]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


if __name__ == "__main__":
    sys.exit(main())
