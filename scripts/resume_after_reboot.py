#!/usr/bin/env python3
"""FedRGBD -- after a reboot of node_a, resume a block that a power loss interrupted.

Run from a user crontab ``@reboot`` entry on node_a (no sudo).  It restarts nothing
unless ``logs/run_matrix.log`` says the last block invocation was *interrupted*: it
started and has no end.  Any recorded end -- ``block finished``, ``STOPPING``, a
failed pre-flight, an identity-gate failure, ``nothing to do`` -- means a human
decides, and nothing is restarted.

    1. classify the last invocation (the lines after the last ``run_matrix start``);
    2. wait until all three nodes answer over SSH;
    3. run the pre-flight (``run_matrix.py --check_only`` with the invocation's
       power configuration);
    4. move every run directory of the block that has no valid results.json and no
       gate marker to ``results/_interrupted/<timestamp>/`` (nothing is deleted);
    5. restart the same ``run_matrix.py`` command in a new tmux session.

Every decision is a line starting with ``RESUME`` in ``logs/resume.log``, which the
desktop fetch task scans and raises as a pop-up.

    python3 scripts/resume_after_reboot.py            # what the @reboot entry runs
    python3 scripts/resume_after_reboot.py --dry_run  # classify and report only
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from scripts import run_matrix  # noqa: E402

START = 'run_matrix start'
RESUME_LOG = os.path.join('logs', 'resume.log')
INTERRUPTED_ROOT = os.path.join('results', '_interrupted')

#: (substring, state) in order of precedence; any of them is a recorded end
END_MARKERS = (
    ('IDENTITY GATE FAIL', 'gate_failed'),
    ('STOPPING', 'stopped'),
    ('PRE-FLIGHT FAIL', 'preflight_failed'),
    ('pre-flight FAIL', 'preflight_failed'),
    ('block finished:', 'finished'),
    ('nothing to do', 'finished'),
)


def last_invocation(log_text: str) -> Tuple[Optional[List[str]], List[str]]:
    """-> (argv of the last ``run_matrix start`` line or None, the lines after it)."""
    lines = log_text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        pos = lines[i].find(START)
        if pos >= 0:
            payload = lines[i][lines[i].find(':', pos) + 1:].strip()
            try:
                argv = json.loads(payload)
            except ValueError:
                return None, lines[i + 1:]
            return (argv if isinstance(argv, list) else None), lines[i + 1:]
    return None, []


def classify(log_text: str) -> Tuple[str, Optional[List[str]], str]:
    """-> (state, argv, reason).  state is 'interrupted' only when the last invocation
    started and nothing after it records an end."""
    argv, after = last_invocation(log_text)
    if argv is None:
        return 'unknown', None, 'no parsable "%s" line in run_matrix.log' % START
    for marker, state in END_MARKERS:
        for line in after:
            if marker in line:
                return state, argv, 'recorded end: %s' % line.strip()
    return 'interrupted', argv, 'the last invocation started and has no recorded end'


def invocation_options(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--block')
    ap.add_argument('--script')
    ap.add_argument('--power_config')
    ap.add_argument('--testbed', default=run_matrix.TESTBED_LOCAL)
    ap.add_argument('--allow_commit_mismatch', action='store_true')
    opts, _ = ap.parse_known_args(argv)
    return opts


def block_script(opts: argparse.Namespace) -> str:
    """The bash file the interrupted invocation ran (what run_matrix.main wrote)."""
    if opts.script:
        return opts.script
    suffix = '' if opts.power_config == run_matrix.DEFAULT_POWER_CONFIG else '_pc_%s' % opts.power_config
    return os.path.join(run_matrix.LOG_DIR, 'block_%s%s.sh' % (opts.block, suffix))


def partial_run_dirs(script: str) -> List[str]:
    """Run dirs of the block that exist, have no valid results.json and no gate marker."""
    out = []
    for run in run_matrix.parse_block(script):
        d = run['out_dir']
        if not os.path.isdir(d) or os.path.isfile(os.path.join(d, run_matrix.GATE_MARKER)):
            continue
        ok, _ = run_matrix.result_ok(d)
        if not ok:
            out.append(d)
    return out


def move_partial(dirs: List[str], stamp: str) -> List[Tuple[str, str]]:
    moved = []
    for d in dirs:
        rel = os.path.relpath(d, 'results')
        dest = os.path.join(INTERRUPTED_ROOT, stamp, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(d, dest)
        moved.append((d, dest))
    return moved


class Reporter:
    def __init__(self, path: str = RESUME_LOG, echo: bool = True):
        self.path, self.echo = path, echo

    def __call__(self, msg: str) -> None:
        line = 'RESUME [%s] %s' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg)
        if self.echo:
            print(line, flush=True)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'a') as f:
            f.write(line + '\n')


def wait_for_nodes(timeout: int, poll: int, say) -> bool:
    deadline = time.time() + timeout
    remote = [n for n in run_matrix.NODE_NAMES if not run_matrix.NODES[n]['local']]
    missing = remote
    while True:
        missing = []
        for n in remote:
            try:
                rc = run_matrix.ssh(n, 'true', timeout=20).returncode
            except subprocess.TimeoutExpired:
                rc = -1
            if rc != 0:
                missing.append(n)
        if not missing:
            return True
        if time.time() > deadline:
            say('nodes not reachable after %d s: %s' % (timeout, ', '.join(missing)))
            return False
        time.sleep(poll)


def run_matrix_alive() -> bool:
    r = subprocess.run(['pgrep', '-f', 'scripts/run_matrix.py'], stdout=subprocess.PIPE, text=True)
    return any(int(p) != os.getpid() for p in r.stdout.split())


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dry_run', action='store_true', help='classify and report, change nothing')
    ap.add_argument('--wait_timeout', type=int, default=3600,
                    help='seconds to wait for all three nodes to answer')
    ap.add_argument('--poll', type=int, default=30)
    args = ap.parse_args(argv)
    os.chdir(REPO)
    say = Reporter()
    log_path = os.path.join(run_matrix.LOG_DIR, 'run_matrix.log')
    text = open(log_path, encoding='utf-8', errors='replace').read() if os.path.isfile(log_path) else ''
    state, inv, why = classify(text)
    if state != 'interrupted':
        say('no restart: last block state is %s (%s)' % (state, why))
        return 0
    if run_matrix_alive():
        say('no restart: a run_matrix.py process is already running')
        return 0
    opts = invocation_options(inv)
    say('last block state is interrupted: run_matrix.py %s' % ' '.join(inv))
    if args.dry_run:
        say('dry run: would wait for the nodes, run the pre-flight, move partial run dirs '
            'and restart; nothing done')
        return 0

    nodes, _port = run_matrix.load_testbed(opts.testbed)
    run_matrix.NODES.clear()
    run_matrix.NODES.update(nodes)
    say('waiting for all three nodes (up to %d s)' % args.wait_timeout)
    if not wait_for_nodes(args.wait_timeout, args.poll, say):
        say('no restart: nodes unreachable')
        return 1
    say('all three nodes answer')

    check = ['python3', 'scripts/run_matrix.py', '--check_only', '--testbed', opts.testbed]
    if opts.power_config:
        check += ['--power_config', opts.power_config]
    if opts.allow_commit_mismatch:
        check.append('--allow_commit_mismatch')
    pf = subprocess.run(check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    detail = ' | '.join(l.strip() for l in pf.stdout.splitlines() if l.strip())
    if pf.returncode != 0:
        say('no restart: pre-flight failed: %s' % detail)
        return 1
    say('pre-flight OK')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    script = block_script(opts)
    if not os.path.isfile(script):
        say('no restart: block script %s not found' % script)
        return 1
    for src, dest in move_partial(partial_run_dirs(script), stamp):
        say('moved partial run %s -> %s' % (src, dest))

    session = 'fedrgbd_resume_%s' % stamp
    out = os.path.join(run_matrix.LOG_DIR, 'resume_%s.out' % stamp)
    venv = run_matrix.NODES['node_a']['venv']
    inner = 'cd %s && source %s/bin/activate && python3 scripts/run_matrix.py %s >> %s 2>&1' % (
        shlex.quote(REPO), shlex.quote(venv), ' '.join(shlex.quote(a) for a in inv), shlex.quote(out))
    r = subprocess.run(['tmux', 'new-session', '-d', '-s', session, 'bash -lc %s' % shlex.quote(inner)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        say('restart FAILED: tmux: %s' % r.stdout.strip())
        return 1
    say('restarted in tmux session %s, output %s' % (session, out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
