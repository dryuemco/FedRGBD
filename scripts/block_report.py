#!/usr/bin/env python3
"""FedRGBD -- when a block of the Jetson matrix is complete, run the mechanical
pipeline over it into a scratch directory.

Called by ``scripts/fetch_results.ps1`` after it pulls new runs. It does two
things and nothing else:

1. Decides whether a block is complete. A block's expected run directories come
   from ``configs/experiment_matrix.yaml`` via ``print_revision_commands`` -- the
   same source the node runs from -- rather than from parsing
   ``logs/run_matrix.log``. A block is complete when every one of its runs exists
   locally with a ``results.json`` that parses and carries
   ``model_selection.selected_round``.
2. For a newly complete block, runs ``analyze_results.py`` and
   ``export_latex_tables.py`` over the whole of ``results/`` and writes the output
   to ``scratch/block_reports/<block>/``, plus a short ``STATUS.md``.

**It never writes to ``analysis/`` or ``paper/tables/``**, and it never commits
anything. ``scratch/`` is gitignored. Everything derived from the federated runs
stays out of the repository until the matrix is finished and Node A commits the
results (see ``CLAUDE.md`` and the Jetson-block protocol).

There is deliberately no interpretation here: the status file reports counts,
selected rounds and where the tables were written, and stops.

    python scripts/block_report.py --repo . [--block seed_extension] [--force]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys

REPO_DEFAULT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def expected_runs(repo: str):
    """{block: [run_dir, ...]} straight from the committed experiment matrix."""
    sys.path.insert(0, repo)
    import scripts.print_revision_commands as prc

    cfg = prc.load_config(os.path.join(repo, "configs", "experiment_matrix.yaml"))
    revision = cfg.get("revision", cfg)
    blocks = prc.expand_all(revision)
    return {name: [r.output_dir for r in blocks.get(name, [])] for name in prc.BLOCK_ORDER}


def is_finished(run_dir: str) -> bool:
    """A run counts as finished only if its results.json parses and was selected."""
    path = os.path.join(run_dir, "results.json")
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (ValueError, OSError):
        return False                      # still being written
    block = data.get("model_selection")
    return isinstance(block, dict) and block.get("selected_round") is not None


def block_status(repo: str):
    out = {}
    for name, runs in expected_runs(repo).items():
        done, missing = [], []
        for rel in runs:
            path = rel if os.path.isabs(rel) else os.path.join(repo, rel)
            (done if is_finished(path) else missing).append(rel)
        out[name] = {"expected": len(runs), "done": done, "missing": missing,
                     "complete": bool(runs) and not missing}
    return out


def run_pipeline(repo: str, block: str, out_dir: str) -> list:
    """analyze_results + export_latex_tables into out_dir. Returns log lines."""
    python = sys.executable
    analysis_dir = os.path.join(out_dir, "analysis")
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    lines = []
    for cmd in (
        [python, os.path.join(repo, "scripts", "analyze_results.py"),
         "--results_dir", os.path.join(repo, "results"), "--output_dir", analysis_dir],
        [python, os.path.join(repo, "scripts", "export_latex_tables.py"),
         "--analysis_dir", analysis_dir, "--output_dir", tables_dir],
    ):
        proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        tail = (proc.stdout or "").strip().splitlines()[-3:]
        lines.append("%s -> exit %d" % (os.path.basename(cmd[1]), proc.returncode))
        lines.extend("    " + t for t in tail)
        if proc.returncode != 0:
            lines.append("    STDERR: " + (proc.stderr or "").strip()[-500:])
            break
    return lines


def write_status(repo: str, block: str, status: dict, out_dir: str, log_lines: list) -> str:
    rows = []
    for rel in sorted(status["done"]):
        path = rel if os.path.isabs(rel) else os.path.join(repo, rel)
        try:
            with open(os.path.join(path, "results.json"), encoding="utf-8") as fh:
                data = json.load(fh)
            sel = (data.get("model_selection") or {}).get("selected_round")
            rounds = data.get("num_rounds")
            secs = data.get("total_time_s")
            rows.append("| `%s` | %s | %s | %.1f |" % (
                os.path.basename(rel), sel, rounds, (secs or 0) / 3600.0))
        except (ValueError, OSError):
            rows.append("| `%s` | ? | ? | ? |" % os.path.basename(rel))

    text = [
        "# Block `%s` -- complete" % block,
        "",
        "Generated %s by `scripts/block_report.py`." % _dt.datetime.now().isoformat(timespec="seconds"),
        "",
        "**This is a mechanical report, not an interpretation.** It records what finished and",
        "where the regenerated tables were written. Nothing here is committed: `scratch/` is",
        "gitignored, and results derived from the federated runs stay out of the repository",
        "until the matrix is done and Node A commits them.",
        "",
        "- runs expected: **%d**" % status["expected"],
        "- runs finished: **%d**" % len(status["done"]),
        "- analysis: `%s`" % os.path.relpath(os.path.join(out_dir, "analysis"), repo).replace("\\", "/"),
        "- tables: `%s`" % os.path.relpath(os.path.join(out_dir, "tables"), repo).replace("\\", "/"),
        "",
        "| run | selected round | rounds | wall-clock (h) |",
        "|---|---|---|---|",
    ] + rows + [
        "",
        "## Pipeline",
        "",
        "```",
    ] + log_lines + [
        "```",
        "",
        "Next step is yours: this file does not analyse the block.",
        "",
    ]
    path = os.path.join(out_dir, "STATUS.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(text))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--block", default=None, help="only consider this block")
    ap.add_argument("--force", action="store_true",
                    help="regenerate even if the report already exists")
    ap.add_argument("--status_only", action="store_true",
                    help="print per-block progress and exit, running nothing")
    args = ap.parse_args(argv)

    repo = os.path.abspath(args.repo)
    status = block_status(repo)

    if args.status_only:
        for name, st in status.items():
            print("%-22s %3d/%-3d %s" % (name, len(st["done"]), st["expected"],
                                         "COMPLETE" if st["complete"] else ""))
        return 0

    root = os.path.join(repo, "scratch", "block_reports")
    any_run = False
    for name, st in status.items():
        if args.block and name != args.block:
            continue
        if not st["complete"]:
            continue
        out_dir = os.path.join(root, name)
        marker = os.path.join(out_dir, "STATUS.md")
        if os.path.exists(marker) and not args.force:
            continue                                  # already reported
        os.makedirs(out_dir, exist_ok=True)
        print("block %s complete (%d runs); running the pipeline" % (name, st["expected"]))
        lines = run_pipeline(repo, name, out_dir)
        path = write_status(repo, name, st, out_dir, lines)
        print("wrote %s" % os.path.relpath(path, repo).replace("\\", "/"))
        any_run = True
    if not any_run:
        incomplete = ["%s %d/%d" % (n, len(s["done"]), s["expected"])
                      for n, s in status.items() if not s["complete"] and s["done"]]
        print("no newly complete block" + ("; in progress: " + ", ".join(incomplete) if incomplete else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
