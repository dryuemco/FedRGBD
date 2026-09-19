#!/usr/bin/env python3
"""FedRGBD — split the desktop-GPU baseline block into parallel lanes.

The baseline scripts use ``num_workers=0``, so a run is bound by single-threaded
JPEG decoding and uses ~15 % of the RTX 5090; several runs are therefore run
side by side.  This script expands ``baselines_extension`` exactly as
``print_revision_commands.py --all_seeds`` does, optionally under a different
``--baseline_root``, and deals the runs into ``--lanes`` shell scripts
(longest-first onto the least-loaded lane, full-partition runs being ~6x the
cost of the 5 % / 1 % low-data runs).  Each lane skips finished runs
(``results.json`` for centralized, ``summary.json`` for local-only), so a lane
can be restarted, and writes one log per run plus a ``[START]/[DONE]/[FAIL]``
line per run to ``logs/<prefix>_<k>.out``.

    python scripts/make_desktop_lanes.py --baseline_root results/rev_baselines_sel \
        --lanes 6 --prefix lane_sel
    for k in 0 1 2 3 4 5; do bash logs/lane_sel_$k.sh > logs/lane_sel_$k.out 2>&1 & done
"""

import argparse
import os
import sys
from typing import List, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts import print_revision_commands as prc  # noqa: E402

GPU_VENV_SCRIPTS = "/c/Users/CORSAIR/venvs/fedrgbd-gpu/Scripts"
REPO_POSIX = "/c/Users/CORSAIR/projects/FedRGBD"


def run_cost(run) -> float:
    """Relative cost: low-data (subsampled) runs are ~1/6 of a full-partition run."""
    return 1.0 if "_sub" not in run.dist else 0.17


def done_marker(run) -> str:
    name = "summary.json" if run.baseline_type != "centralized" else "results.json"
    return run.output_dir + "/" + name


def lane_script(runs, lane: int, prefix: str) -> str:
    out = ["#!/bin/bash",
           f"# desktop GPU baseline lane {lane} (scripts/make_desktop_lanes.py)",
           "export PYTHONUTF8=1",
           f'export PATH="{GPU_VENV_SCRIPTS}:$PATH"',
           f"cd {REPO_POSIX}",
           "mkdir -p logs"]
    for run in runs:
        name = os.path.basename(run.output_dir)
        log = f"logs/{prefix}_{name}.log"
        out += [
            f'if [ -f {done_marker(run)} ]; then echo "[SKIP] {name}"; else',
            f'  echo "[START] {name} $(date +%H:%M:%S)"',
            f'  {run.baseline_command()} > {log} 2>&1 && echo "[DONE] {name} $(date +%H:%M:%S)"'
            f' || echo "[FAIL] {name} $(date +%H:%M:%S) exit $?"',
            "fi",
        ]
    return "\n".join(out) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=prc.DEFAULT_CONFIG)
    p.add_argument("--baseline_root", default=None,
                   help="as in print_revision_commands.py (default: results/)")
    p.add_argument("--lanes", type=int, default=4)
    p.add_argument("--prefix", default="lane")
    p.add_argument("--out_dir", default="logs")
    args = p.parse_args(argv)

    cfg = prc.load_config(args.config)["revision"]
    cfg = dict(cfg, baselines_extension=dict(cfg["baselines_extension"], all_seeds=True))
    runs_by_block = prc.expand_all(cfg, "baselines_extension")
    if args.baseline_root:
        prc.rebase_baselines(runs_by_block, args.baseline_root)
    runs = runs_by_block["baselines_extension"]

    lanes: List[list] = [[] for _ in range(args.lanes)]
    load = [0.0] * args.lanes
    for run in sorted(runs, key=run_cost, reverse=True):
        k = min(range(args.lanes), key=lambda i: load[i])
        lanes[k].append(run)
        load[k] += run_cost(run)

    os.makedirs(args.out_dir, exist_ok=True)
    for k, lane_runs in enumerate(lanes):
        path = os.path.join(args.out_dir, f"{args.prefix}_{k}.sh")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(lane_script(lane_runs, k, args.prefix))
        print(f"{path}: {len(lane_runs)} runs, relative load {load[k]:.2f}")
    print(f"{len(runs)} runs in {args.lanes} lanes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
