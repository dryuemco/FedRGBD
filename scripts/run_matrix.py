#!/usr/bin/env python3
"""FedRGBD -- run a revision block unattended on the 3-node Jetson testbed.

``print_revision_commands.py --format bash`` prints the server command and tells
the operator to start three clients by hand inside a 10 s window.  That is fine
for one run and unusable for a 20-run, 46-hour block.  This runner parses the
same generated script and drives the whole block:

    for each run:
        pre-flight  : nodes reachable, repo at the same commit, RAM/disk free,
                      port 8080 clear, no stale python processes, GUI off
                      (multi-user.target) on every node, every split the
                      run reads has the manifest md5 of
                      analysis/leakage/P0_SUMMARY.md on every node, and every
                      node's `nvpmodel -q` power mode is the one declared for
                      --power_config in testbed.local.yaml
        start       : the server on node_a; once node_a:8080 accepts TCP
                      connections, the clients (node_b / node_c over SSH,
                      node_a locally).  Clients must not start earlier: the
                      default gRPC-bidi Flower client does not retry a
                      refused first connection and dies.
        wait        : until the server exits; a client exiting non-zero
                      before that ends the run at once
        check       : results/<run>/results.json exists and parses, and the
                      model_selection block is present; the power modes are
                      read again and written into results.json ("power"). A
                      mode that changed during the run invalidates it: the
                      run directory is moved to logs/invalid_runs/ so it is
                      neither analysed nor skipped as done
        identity    : for runs with ">>> IDENTITY GATE:" lines, the per-image
                      predictions of the listed rounds must be bitwise
                      identical to the reference run (checked while the run
                      executes and again at the end, with the aggregated
                      validation loss).  A mismatch ends the run and STOPS the
                      block -- no retry -- and leaves IDENTITY_GATE_FAILED.json
                      in the run directory; the block refuses to start again
                      until a human has resolved it
        energy      : with --energy, tegrastats logs board input power (VDD_IN)
                      on every node for the duration of the run; the energy of
                      the run window is written into results.json ("energy")
        on failure  : ONE retry with identical parameters, then stop the block

Deliberately NOT adaptive.  It never changes a batch size, never skips a seed,
never edits a command.  A run either completes exactly as specified or the block
stops and waits for a human.  Experiment comparability depends on that.

Run it on Node A, inside tmux:

    tmux new -s fedrgbd
    python3 scripts/run_matrix.py --block seed_extension --power_config maxn
    # detach with Ctrl-b d ; reattach with: tmux attach -t fedrgbd

Requires passwordless SSH from Node A to Node B and Node C (see --check_only), and
configs/testbed.local.yaml with your nodes' hosts, users and paths and the
expected power mode of every node per power configuration: copy
configs/testbed.example.yaml and fill it in (the copy is gitignored).

Power configurations.  ``--power_config`` is required.  ``heterogeneous`` is
the main matrix (results/rev_*); any other configuration writes to its own
namespace, results/pc_<name>/rev_* (print_revision_commands.py
--power_config), so its runs are never skipped because a run of another
configuration exists, and analyze_results.py never pools across them.
"""

import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime

# --- testbed description -----------------------------------------------------
# Node A is the server and also runs a client locally.
# Hosts, users and paths live in configs/testbed.local.yaml (gitignored; the repo
# is public).  Copy configs/testbed.example.yaml and fill it in.  Loaded by main().
TESTBED_LOCAL = os.path.join('configs', 'testbed.local.yaml')
TESTBED_EXAMPLE = os.path.join('configs', 'testbed.example.yaml')
NODE_NAMES = ('node_a', 'node_b', 'node_c')
NODE_FIELDS = ('host', 'user', 'repo', 'venv', 'local')
NODES = {}
SERVER_PORT = 8080
MIN_FREE_MB = 4000      # refuse to start a run with less available RAM than this
MIN_FREE_DISK_MB = 2000

LOG_DIR = 'logs'
#: where a run invalidated after the fact (power mode changed mid-run) is moved:
#: outside results/, so it is neither analysed nor counted as done
INVALID_DIR = os.path.join(LOG_DIR, 'invalid_runs')

#: must equal print_revision_commands.POWER_CONFIGS / DEFAULT_POWER_CONFIG
#: (pinned by tests/test_run_matrix.py; this script does not import the generator)
POWER_CONFIGS = ('heterogeneous', 'maxn')
DEFAULT_POWER_CONFIG = 'heterogeneous'


def power_root(power_config):
    """results for the main matrix, results/pc_<name> for any other configuration."""
    return 'results' if power_config == DEFAULT_POWER_CONFIG else 'results/pc_%s' % power_config


#: the camera experiment (docs/CAMERA_EXPERIMENT_PREREG.md) is separate from FLAME by
#: dataset: its testbed blocks write to results/camera/<power_config>/, never to a FLAME
#: namespace, and no FLAME block may write there.  Must equal
#: print_revision_commands.CAMERA_TESTBED_BLOCKS / CAMERA_RESULTS (tests/test_run_matrix.py).
CAMERA_TESTBED_BLOCKS = ('camera_sensor_skew', 'camera_determinism_smoke')
CAMERA_RESULTS = 'results/camera'


def namespace_root(power_config, block=None):
    """Where the runs of ``block`` under ``power_config`` must write."""
    if block in CAMERA_TESTBED_BLOCKS:
        return '%s/%s' % (CAMERA_RESULTS, power_config)
    return power_root(power_config)


def load_testbed(path=TESTBED_LOCAL):
    """-> (nodes, server_port) from the operator's local testbed file.

    Fails with instructions when the file is missing or still holds placeholders:
    node users and paths are never stored in the (public) repository.
    """
    if not os.path.isfile(path):
        raise SystemExit(
            'testbed config %s not found.\n'
            'Copy the example and fill in the hosts, users and paths of your own nodes '
            '(the copy is gitignored):\n'
            '    cp %s %s\n'
            '    $EDITOR %s' % (path, TESTBED_EXAMPLE, TESTBED_LOCAL, TESTBED_LOCAL))
    import yaml
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    nodes = cfg.get('nodes') or {}
    problems = []
    if sorted(nodes) != sorted(NODE_NAMES):
        problems.append('nodes must be exactly %s, found %s'
                        % (', '.join(NODE_NAMES), ', '.join(sorted(nodes)) or 'none'))
    for name in NODE_NAMES:
        node = nodes.get(name) or {}
        missing = [k for k in NODE_FIELDS if k not in node]
        if missing:
            problems.append('%s: missing %s' % (name, ', '.join(missing)))
        placeholders = [k for k, v in node.items() if '<' in str(v) or '>' in str(v)]
        if placeholders:
            problems.append('%s: placeholder value(s) not filled in: %s'
                            % (name, ', '.join(placeholders)))
    if nodes and [n for n in NODE_NAMES if (nodes.get(n) or {}).get('local')] != ['node_a']:
        problems.append("exactly node_a must have 'local: true' (run_matrix.py runs on node_a)")
    if problems:
        raise SystemExit('invalid testbed config %s:\n  - %s' % (path, '\n  - '.join(problems)))
    return ({n: {k: nodes[n][k] for k in NODE_FIELDS} for n in NODE_NAMES},
            int(cfg.get('server_port', 8080)))


def load_power_modes(path, power_config):
    """{node: expected nvpmodel mode name} of ``power_config`` from the testbed file.

    The runner refuses to start without it: per-round wall-clock is a reported
    result, and the main matrix ran with modes nobody had harmonised.
    """
    import yaml
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    table = cfg.get('power_modes') or {}
    modes = table.get(power_config) if isinstance(table, dict) else None
    if not isinstance(modes, dict):
        raise SystemExit(
            '%s declares no power_modes.%s -- the expected `nvpmodel -q` mode of every '
            'node is required; see %s' % (path, power_config, TESTBED_EXAMPLE))
    problems = []
    for name in NODE_NAMES:
        value = modes.get(name)
        if value is None or not str(value).strip():
            problems.append('power_modes.%s.%s: missing' % (power_config, name))
        elif '<' in str(value) or '>' in str(value):
            problems.append('power_modes.%s.%s: placeholder value not filled in'
                            % (power_config, name))
        elif power_config == 'maxn' and not str(value).strip().upper().startswith('MAXN'):
            problems.append('power_modes.maxn.%s is %r, which is not a MAXN mode'
                            % (name, value))
    extra = sorted(set(modes) - set(NODE_NAMES))
    if extra:
        problems.append('power_modes.%s: unknown node(s) %s' % (power_config, ', '.join(extra)))
    if problems:
        raise SystemExit('invalid power modes in %s:\n  - %s' % (path, '\n  - '.join(problems)))
    return {n: str(modes[n]).strip() for n in NODE_NAMES}


def ts():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(msg):
    line = '[%s] %s' % (ts(), msg)
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, 'run_matrix.log'), 'a') as f:
        f.write(line + '\n')


# --- parsing the generated block --------------------------------------------
RE_RUN = re.compile(r"^echo '--- (results/\S+) ---'")
RE_NODE = re.compile(r"^echo '>>> START ON (node_[abc]) \(")
RE_CMD = re.compile(r"^echo '\s*(python3 src/fl/client\.py .*)'$")
RE_SERVER = re.compile(r"^(python3 src/fl/server\.py .*)$")
RE_GATE = re.compile(r"^echo '>>> IDENTITY GATE: (results/\S+) rounds ([0-9,]+)'$")


def parse_block(path):
    """-> [{'out_dir':…, 'clients': {node: cmd}, 'server': cmd}]"""
    runs, cur, pending_node = [], None, None
    # the generated header contains a non-ASCII dash; never let the locale decide
    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.rstrip('\r\n')
            if line.startswith('python3 scripts/train_'):
                raise SystemExit('%s contains centralized/local-only baselines; those run on '
                                 'the desktop GPU (scripts/make_desktop_lanes.py), not on the '
                                 'testbed' % path)
            m = RE_RUN.match(line)
            if m:
                if cur:
                    runs.append(cur)
                cur = {'out_dir': m.group(1), 'clients': {}, 'server': None, 'gates': []}
                pending_node = None
                continue
            if cur is None:
                continue
            m = RE_GATE.match(line)
            if m:
                cur['gates'].append((m.group(1), [int(r) for r in m.group(2).split(',')]))
                continue
            m = RE_NODE.match(line)
            if m:
                pending_node = m.group(1)
                continue
            m = RE_CMD.match(line)
            if m and pending_node:
                cur['clients'][pending_node] = m.group(1)
                pending_node = None
                continue
            m = RE_SERVER.match(line)
            if m:
                cur['server'] = m.group(1)
    if cur:
        runs.append(cur)
    bad = [r['out_dir'] for r in runs if not r['server'] or len(r['clients']) != 3]
    if bad:
        raise SystemExit('parse error: incomplete run definition for %s' % ', '.join(bad))
    return runs


# --- remote helpers ----------------------------------------------------------
def ssh(node, command, timeout=60, capture=True):
    cfg = NODES[node]
    argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=4',
            '%s@%s' % (cfg['user'], cfg['host']), command]
    return subprocess.run(argv, timeout=timeout,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None,
                          text=True)


def node_shell(node, inner):
    """Wrap a repo command in cd + venv activation for that node."""
    cfg = NODES[node]
    return ('cd %s && source %s/bin/activate && %s'
            % (shlex.quote(cfg['repo']), shlex.quote(cfg['venv']), inner))


def _popen(argv, out):
    """Background process in its own session, so the whole group can be killed."""
    return subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT,
                            start_new_session=True)


def _kill(proc, sig=signal.SIGTERM):
    if proc.poll() is not None:
        return
    try:
        if hasattr(os, 'killpg'):
            os.killpg(os.getpgid(proc.pid), sig)
        else:  # pragma: no cover - Windows (CPU tests only)
            proc.kill()
    except (OSError, ProcessLookupError):
        pass


def run_on(node, inner, log_path):
    """Start a client (background Popen). Local for node_a, SSH otherwise."""
    cfg = NODES[node]
    out = open(log_path, 'ab')
    if cfg['local']:
        argv = ['bash', '-lc', node_shell(node, inner)]
    else:
        argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=4',
                '%s@%s' % (cfg['user'], cfg['host']),
                'bash -lc %s' % shlex.quote(node_shell(node, inner))]
    return _popen(argv, out), out


def start_server(run, log_path):
    """Start the FL server on node_a (this machine)."""
    out = open(log_path, 'ab')
    return _popen(['bash', '-lc', node_shell('node_a', run['server'])], out), out


def wait_for_port(host, port, timeout, server=None, poll=1.0):
    """Block until host:port accepts a TCP connection.

    -> (True, 'ready') or (False, reason).  Gives up early when the server
    process exits.  Flower's default gRPC-bidi client does NOT retry a refused
    first connection (flwr 1.13.1 ignores max_retries on that transport): a
    client started before the server listens dies with UNAVAILABLE, so clients
    are only started once this returns True.
    """
    deadline = time.time() + timeout
    while True:
        if server is not None and server.poll() is not None:
            return False, 'server exited with rc=%s before accepting connections' % server.poll()
        try:
            with socket.create_connection((host, port), timeout=2):
                return True, 'ready'
        except OSError:
            pass
        if time.time() >= deadline:
            return False, '%s:%d did not accept connections within %ds' % (host, port, timeout)
        time.sleep(poll)


def log_tail(path, lines=15):
    try:
        with open(path, 'rb') as f:
            text = f.read().decode('utf-8', 'replace')
    except OSError:
        return '(no log)'
    return '\n'.join('        | ' + l for l in text.rstrip().splitlines()[-lines:])


# --- pre-flight --------------------------------------------------------------
#: the authoritative split digests (CLAUDE.md rule 2): every node's
#: data/processed/<split>/manifest.csv must have exactly these md5s
P0_SUMMARY = os.path.join('analysis', 'leakage', 'P0_SUMMARY.md')
RE_DIGEST = re.compile(r'^\|\s*(\S+)/manifest\.csv\s*\|\s*\d+\s*\|\s*([0-9a-f]{32})\s*\|')
RE_SPLIT = re.compile(r'--data_dir data/processed/(\S+)/node_[abc](?:\s|$)')
RE_ROUNDS = re.compile(r'--rounds (\d+)')


def expected_digests(path=P0_SUMMARY, camera_path=None):
    """{split: md5} from the 'Split digests' table of P0_SUMMARY.md, plus -- once
    scripts/camera_fl_prepare.py has written it -- ``camera_fold<f>`` from the
    ``fold_manifest_md5`` column of data/splits_camera/fl_materialised_manifest.csv."""
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = RE_DIGEST.match(line)
            if m:
                out[m.group(1)] = m.group(2)
    out.update(camera_digests(CAMERA_COUNTS if camera_path is None else camera_path))
    return out


#: committed by scripts/camera_fl_prepare.py: per fold / node / split / class counts and
#: the md5 of data/processed/camera_fold<f>/manifest.csv (identical on every node)
CAMERA_COUNTS = os.path.join('data', 'splits_camera', 'fl_materialised_manifest.csv')


def camera_digests(path=CAMERA_COUNTS):
    """{camera_fold<f>: md5}; {} if the file does not exist; a fold listed with two
    different digests raises (the committed record would contradict itself)."""
    import csv
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            split, md5 = 'camera_fold%d' % int(row['fold']), row['fold_manifest_md5'].strip()
            if out.setdefault(split, md5) != md5:
                raise SystemExit('%s lists two manifest md5s for %s' % (path, split))
    return out


def block_splits(runs):
    """Every data split the clients of these runs read."""
    return sorted({m.group(1) for r in runs for cmd in r['clients'].values()
                   for m in [RE_SPLIT.search(cmd)] if m})


def node_facts(node, splits):
    """-> (systemd default target, {split: md5 or None}) for one node."""
    cfg = NODES[node]
    files = ' '.join('data/processed/%s/manifest.csv' % s for s in splits)
    cmd = ('systemctl get-default; cd %s && md5sum %s 2>&1 || true'
           % (shlex.quote(cfg['repo']), files))
    if cfg['local']:
        out = subprocess.run(['bash', '-lc', cmd], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True).stdout
    else:
        out = ssh(node, cmd).stdout or ''
    lines = out.strip().splitlines()
    target = lines[0].strip() if lines else ''
    digests = {s: None for s in splits}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 2 and re.fullmatch(r'[0-9a-f]{32}', parts[0]):
            for s in splits:
                if parts[1].lstrip('*') == 'data/processed/%s/manifest.csv' % s:
                    digests[s] = parts[0]
    return target, digests


def check_testbed(splits):
    """CLAUDE.md: GUI off on every node (per-round wall-clock is a reported result) and
    the replayed partition identical to data/splits on every node (rule 2)."""
    problems = []
    want = expected_digests()
    unknown = [s for s in splits if s not in want]
    if unknown:
        problems.append('no reference md5 in %s for split(s): %s' % (P0_SUMMARY, ', '.join(unknown)))
    for node in NODES:
        try:
            target, digests = node_facts(node, splits)
        except subprocess.TimeoutExpired:
            problems.append('%s: timed out reading systemd target / manifests' % node)
            continue
        if target != 'multi-user.target':
            problems.append('%s: default target is %r, not multi-user.target (GUI must be '
                            'off on all three nodes)' % (node, target))
        for s in splits:
            if digests.get(s) is None:
                problems.append('%s: data/processed/%s/manifest.csv missing -- replay the '
                                'partition with --from_manifest' % (node, s))
            elif s in want and digests[s] != want[s]:
                problems.append('%s: %s manifest md5 %s != %s (P0_SUMMARY) -- partition '
                                'differs; never re-derive it, replay data/splits'
                                % (node, s, digests[s][:8], want[s][:8]))
    return problems


RE_POWER_MODE = re.compile(r'NV Power Mode:\s*(\S+)')


def node_power_mode(node):
    """-> the mode name `nvpmodel -q` reports on ``node``, or None if unreadable."""
    cmd = 'nvpmodel -q 2>&1 || true'
    try:
        if NODES[node]['local']:
            out = subprocess.run(['bash', '-lc', cmd], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, timeout=60).stdout
        else:
            out = ssh(node, cmd).stdout or ''
    except subprocess.TimeoutExpired:
        return None
    m = RE_POWER_MODE.search(out or '')
    return m.group(1) if m else None


def check_power(expected):
    """-> (problems, {node: measured mode or None}) against ``expected`` {node: mode}."""
    problems, measured = [], {}
    for node in NODES:
        mode = node_power_mode(node)
        measured[node] = mode
        if mode is None:
            problems.append('%s: could not read the power mode (nvpmodel -q)' % node)
        elif mode != expected.get(node):
            problems.append('%s: power mode %s, expected %s (testbed.local.yaml)'
                            % (node, mode, expected.get(node)))
    return problems, measured


def local_commit():
    r = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, text=True)
    return r.stdout.strip()


def preflight(strict_commit=True, splits=(), power=None, measured=None):
    """-> list of problems.  ``power``: {node: expected mode}; the measured modes are
    written into the ``measured`` dict when one is passed."""
    problems = check_testbed(list(splits)) if splits else []
    if power is not None:
        power_problems, modes = check_power(power)
        problems.extend(power_problems)
        if measured is not None:
            measured.clear()
            measured.update(modes)
    want = local_commit()
    for node, cfg in NODES.items():
        if cfg['local']:
            free = int(subprocess.run(
                ["bash", "-lc", "free -m | awk '/^Mem:/{print $7}'"],
                stdout=subprocess.PIPE, text=True).stdout.strip() or 0)
            disk = int(subprocess.run(
                ["bash", "-lc", "df -m %s | tail -1 | awk '{print $4}'" % shlex.quote(cfg['repo'])],
                stdout=subprocess.PIPE, text=True).stdout.strip() or 0)
            head = want
        else:
            r = ssh(node, "free -m | awk '/^Mem:/{print $7}'; "
                          "df -m %s | tail -1 | awk '{print $4}'; "
                          "cd %s && git rev-parse HEAD"
                          % (shlex.quote(cfg['repo']), shlex.quote(cfg['repo'])))
            if r.returncode != 0:
                problems.append('%s: ssh failed: %s' % (node, (r.stdout or '').strip()[:200]))
                continue
            parts = (r.stdout or '').split()
            if len(parts) < 3:
                problems.append('%s: unexpected pre-flight output: %r' % (node, r.stdout))
                continue
            free, disk, head = int(parts[0]), int(parts[1]), parts[2]
        if free < MIN_FREE_MB:
            problems.append('%s: only %d MB RAM available (need %d)' % (node, free, MIN_FREE_MB))
        if disk < MIN_FREE_DISK_MB:
            problems.append('%s: only %d MB disk free' % (node, disk))
        if strict_commit and head != want:
            problems.append('%s: repo at %s, expected %s' % (node, head[:8], want[:8]))
    # server port must be free.  SO_REUSEADDR: the previous run's server leaves
    # TIME_WAIT sockets on 8080 for ~60 s; without it this probe reports "in use"
    # after every run (the gRPC server itself binds with SO_REUSEADDR), while a
    # live listener still makes the bind fail.
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(('0.0.0.0', SERVER_PORT))
    except OSError:
        problems.append('port %d already in use on node_a' % SERVER_PORT)
    finally:
        s.close()
    # no stale FL processes anywhere
    for node, cfg in NODES.items():
        cmd = "pgrep -fa 'src/fl/(client|server).py' || true"
        if cfg['local']:
            out = subprocess.run(['bash', '-lc', cmd], stdout=subprocess.PIPE, text=True).stdout
        else:
            out = (ssh(node, cmd).stdout or '')
        if out.strip():
            problems.append('%s: stale FL process: %s' % (node, out.strip().splitlines()[0][:120]))
    return problems


# Page-cache release before every run, without sudo.  On the Jetson, CPU and GPU share
# the 8 GB; after a manifest replay the page cache held ~6.4 GB on node_a and its client
# died with CUDA out of memory at model load (2026-10-07, 02:28), before the kernel had
# reclaimed the cache.  Touching anonymous memory up to MemAvailable minus a reserve makes
# the kernel drop clean cache pages; the memory is released at once.  Data is unchanged.
PAGE_CACHE_RESERVE_MB = 900
RELEASE_PAGE_CACHE = r'''python3 - <<'PY'
import mmap
def mem(key):
    for line in open('/proc/meminfo'):
        if line.startswith(key + ':'):
            return int(line.split()[1]) * 1024
before = mem('MemFree')
n = mem('MemAvailable') - %d * 1024 * 1024
if n > 0:
    m = mmap.mmap(-1, n)
    for off in range(0, n, 4096):
        m[off] = 1
    m.close()
print('PAGECACHE %%d %%d' %% (before >> 20, mem('MemFree') >> 20))
PY''' % PAGE_CACHE_RESERVE_MB
RE_PAGECACHE = re.compile(r'PAGECACHE (\d+) (\d+)')


def release_page_cache():
    """-> {node: (MemFree MB before, after) or None}; run on every node before a run."""
    out = {}
    for node, cfg in NODES.items():
        try:
            if cfg['local']:
                text = subprocess.run(['bash', '-c', RELEASE_PAGE_CACHE], stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, timeout=180).stdout
            else:
                text = ssh(node, RELEASE_PAGE_CACHE, timeout=180).stdout or ''
        except subprocess.TimeoutExpired:
            text = ''
        m = RE_PAGECACHE.search(text or '')
        out[node] = (int(m.group(1)), int(m.group(2))) if m else None
    return out


def kill_stragglers():
    for node, cfg in NODES.items():
        # our own tegrastats only (their logfile lives under logs/energy/)
        cmd = ("pkill -f 'src/fl/(client|server).py' || true; "
               "pkill -f 'tegrastat[s] --interval [0-9]+ --logfile .*/logs/energy/' || true")
        if cfg['local']:
            subprocess.run(['bash', '-lc', cmd])
        else:
            try:
                ssh(node, cmd, timeout=30)
            except subprocess.TimeoutExpired:
                pass


# --- one run -----------------------------------------------------------------
RE_CLIENT_SERVER = re.compile(r'--server (\S+)')


def check_server_address(runs):
    """Every client must dial node_a's configured host:port, the address the
    runner polls.  A mismatch (config vs generated commands) would start clients
    against a server that is not there."""
    want = '%s:%d' % (NODES['node_a']['host'], SERVER_PORT)
    bad = sorted({m.group(1) for r in runs for cmd in r['clients'].values()
                  for m in [RE_CLIENT_SERVER.search(cmd)] if m and m.group(1) != want})
    if bad:
        raise SystemExit('clients dial %s but node_a is configured as %s (%s); fix '
                         'configs/testbed.local.yaml or the generator'
                         % (', '.join(bad), want, TESTBED_LOCAL))
def check_namespace(runs, power_config, block=None):
    """Every run must write into the namespace of ``power_config`` -- a script
    generated for another configuration (--script) would otherwise mix them -- and of its
    dataset: camera blocks only into results/camera/<power_config>/, FLAME never there."""
    root = namespace_root(power_config, block) + '/'
    camera = block in CAMERA_TESTBED_BLOCKS
    bad = [r['out_dir'] for r in runs
           if not r['out_dir'].startswith(root)
           or (power_config == DEFAULT_POWER_CONFIG and r['out_dir'].startswith('results/pc_'))
           or (not camera and r['out_dir'].startswith(CAMERA_RESULTS + '/'))]
    if bad:
        raise SystemExit('%d run(s) are outside %s, the namespace of power_config %s '
                         '(first: %s); regenerate the script with --power_config %s'
                         % (len(bad), root, power_config, bad[0], power_config))


# --- within-configuration determinism gate ------------------------------------
# docs/CROSS_CONFIG_COMPARISON.md: a block may declare `determinism_gate: {runs: [a, b]}`,
# two smoke runs of the same command under the block's power configuration.  The block
# starts only if they are bitwise identical: every prediction file (same set of files,
# every array with the same dtype, shape and bytes), every client's per-round training
# loss and the aggregated validation loss of every round.
EXPERIMENT_CONFIG = os.path.join('configs', 'experiment_matrix.yaml')


def block_determinism_gate(block, config_path=EXPERIMENT_CONFIG):
    """-> the `determinism_gate` a block declares, or None."""
    import yaml
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    return ((cfg.get('revision') or {}).get(block) or {}).get('determinism_gate')


def _smoke_record(run_dir):
    with open(os.path.join(run_dir, 'results.json')) as f:
        d = json.load(f)
    losses = {(int(r['round']), node): c.get('train_loss')
              for r in d.get('rounds') or [] for node, c in (r.get('fit') or {}).get('clients', {}).items()}
    val = {str(k): v for k, v in ((d.get('model_selection') or {}).get('val_loss_by_round') or {}).items()}
    return losses, val


def runs_bitwise_identical(dir_a, dir_b):
    """-> (True, summary) if the two runs are bitwise identical, else (False, first difference)."""
    import numpy as np
    for d in (dir_a, dir_b):
        if not os.path.isfile(os.path.join(d, 'results.json')):
            return False, '%s: results.json missing' % d
    la, va = _smoke_record(dir_a)
    lb, vb = _smoke_record(dir_b)
    if not la or not va:
        return False, '%s: no per-round training or validation losses' % dir_a
    if la != lb:
        k = sorted(set(la) | set(lb))
        first = next(x for x in k if la.get(x) != lb.get(x))
        return False, 'train_loss round %d %s: %r vs %r' % (first[0], first[1], la.get(first), lb.get(first))
    if va != vb:
        return False, 'aggregated val loss differs: %r vs %r' % (va, vb)
    pa, pb = os.path.join(dir_a, 'predictions'), os.path.join(dir_b, 'predictions')
    fa = sorted(f for f in os.listdir(pa) if f.endswith('.npz')) if os.path.isdir(pa) else []
    fb = sorted(f for f in os.listdir(pb) if f.endswith('.npz')) if os.path.isdir(pb) else []
    if not fa or fa != fb:
        return False, 'prediction file sets differ (%d vs %d files)' % (len(fa), len(fb))
    for name in fa:
        with np.load(os.path.join(pa, name)) as A, np.load(os.path.join(pb, name)) as B:
            if sorted(A.files) != sorted(B.files):
                return False, '%s: arrays differ' % name
            for k in A.files:
                x, y = A[k], B[k]
                if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                    return False, '%s: array %r is not bitwise identical' % (name, k)
    return True, '%d prediction files, %d client-round losses and %d validation losses identical' % (
        len(fa), len(la), len(va))


def check_determinism_gate(gate):
    """-> (ok, why) for a declared `determinism_gate`."""
    runs = list((gate or {}).get('runs') or [])
    if len(runs) != 2:
        return False, 'determinism_gate must name exactly two runs, got %r' % (runs,)
    return runs_bitwise_identical(runs[0], runs[1])


def result_ok(out_dir):
    path = os.path.join(out_dir, 'results.json')
    if not os.path.isfile(path):
        return False, 'results.json missing'
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        return False, 'results.json unreadable: %r' % e
    ms = d.get('model_selection')
    if not ms or ms.get('selected_round') is None:
        return False, 'model_selection missing or empty'
    return True, 'selected_round=%s' % ms.get('selected_round')


def record_power(out_dir, power_config, expected, before, after):
    """Write the power configuration and both measurements into results.json.

    -> (True, why) or (False, why).  If any node's mode after the run differs from
    the expected one (or cannot be read), the run's wall-clock is not attributable
    to ``power_config``: the run directory is moved to logs/invalid_runs/, so the
    block does not count it as done and analyze_results.py never sees it.
    """
    changed = sorted(n for n in expected if after.get(n) != expected[n])
    if changed:
        why = 'power mode changed during the run: %s' % ', '.join(
            '%s %s -> %s' % (n, before.get(n), after.get(n)) for n in changed)
        return False, why + '; ' + quarantine(out_dir)
    path = os.path.join(out_dir, 'results.json')
    try:
        with open(path) as f:
            data = json.load(f)
        data['power'] = {
            'power_config': power_config,
            'expected': dict(expected),
            'measured_before': dict(before),
            'measured_after': dict(after),
            'source': 'nvpmodel -q, read by scripts/run_matrix.py before and after the run',
        }
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except (OSError, ValueError) as e:
        return False, 'could not record the power modes (%r); %s' % (e, quarantine(out_dir))
    return True, 'power modes recorded (%s)' % ', '.join(
        '%s=%s' % (n, after[n]) for n in sorted(after))


def quarantine(out_dir):
    """Move a run directory out of results/ (see record_power); -> what was done."""
    if not os.path.isdir(out_dir):
        return 'nothing to move'
    os.makedirs(INVALID_DIR, exist_ok=True)
    parts = [p for p in re.split(r'[\\/]+', os.path.normpath(out_dir)) if p]
    name = '__'.join(parts[-2:]).replace(':', '')        # e.g. pc_maxn__rev_iid_fedavg_seed42
    dest = os.path.join(INVALID_DIR, '%s__%s' % (name, datetime.now().strftime('%Y%m%d_%H%M%S')))
    os.rename(out_dir, dest)
    return 'run directory moved to %s' % dest


# --- identity gates ----------------------------------------------------------
GATE_MARKER = 'IDENTITY_GATE_FAILED.json'
PRED_NODES = ('node_a', 'node_b', 'node_c')
PRED_SPLITS = ('val', 'test')
PRED_ARRAYS = ('path', 'label', 'logit_margin', 'p_fire')


def pred_file(run_dir, rnd, node, split):
    return os.path.join(run_dir, 'predictions', 'r%03d_%s_%s.npz' % (rnd, node, split))


def _load_pred(path):
    """-> {array: raw bytes/dtype/shape} or None if the file is absent or not complete.

    The server writes the .npz files non-atomically; a half-written zip has no
    central directory and fails to load, so a failure here means "not yet"."""
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as d:
            return {k: (d[k].dtype.str, d[k].shape, d[k].tobytes()) for k in PRED_ARRAYS}
    except Exception:
        return None


def compare_round(run_dir, ref_dir, rnd):
    """-> (state, detail) with state 'same', 'differ' or 'pending'."""
    for node in PRED_NODES:
        for split in PRED_SPLITS:
            got = _load_pred(pred_file(run_dir, rnd, node, split))
            if got is None:
                return 'pending', 'r%03d_%s_%s not written yet' % (rnd, node, split)
            ref = _load_pred(pred_file(ref_dir, rnd, node, split))
            if ref is None:
                return 'differ', 'reference %s has no complete r%03d_%s_%s' % (ref_dir, rnd, node, split)
            for k in PRED_ARRAYS:
                if got[k] != ref[k]:
                    return 'differ', ('round %d %s %s: array %r is not bitwise identical to %s'
                                      % (rnd, node, split, k, ref_dir))
    return 'same', 'round %d identical' % rnd


def check_gate_references(runs):
    """Refuse to start when a reference run a gate needs is missing or incomplete."""
    bad = []
    for r in runs:
        for ref, rounds in r.get('gates', []):
            if not os.path.isfile(os.path.join(ref, 'results.json')):
                bad.append('%s: reference %s has no results.json' % (r['out_dir'], ref))
                continue
            for rnd in rounds:
                for node in PRED_NODES:
                    for split in PRED_SPLITS:
                        if _load_pred(pred_file(ref, rnd, node, split)) is None:
                            bad.append('%s: reference %s lacks r%03d_%s_%s.npz'
                                       % (r['out_dir'], ref, rnd, node, split))
    if bad:
        raise SystemExit('identity-gate references incomplete, not starting:\n  - '
                         + '\n  - '.join(bad[:20]))


def unresolved_gate_failures(runs, root=None):
    """Run dirs holding IDENTITY_GATE_FAILED.json: every run of the parsed block, and --
    independent of what the generator skipped -- every run dir under ``root``."""
    found = {r['out_dir'] for r in runs
             if os.path.isfile(os.path.join(r['out_dir'], GATE_MARKER))}
    if root and os.path.isdir(root):
        for entry in sorted(os.listdir(root)):
            path = os.path.join(root, entry)
            if os.path.isfile(os.path.join(path, GATE_MARKER)):
                found.add(path.replace('\\', '/'))
    return sorted(found)


def fail_gate(run, reason):
    """Record a gate failure so that nothing can mistake the run for a finished one:
    the marker is written and results.json (if any) becomes results.gate_failed.json,
    so the generator does not skip the run as existing, the fetch job does not take
    it, block_report does not count it and analyze_results refuses the namespace.
    -> True if a results.json existed and was renamed, False if there was none."""
    write_gate_marker(run, reason)
    path = os.path.join(run['out_dir'], 'results.json')
    if os.path.isfile(path):
        os.replace(path, os.path.join(run['out_dir'], 'results.gate_failed.json'))
        return True
    return False


def gate_stop_message(out_dir, reason, renamed):
    """The STOPPING line of a gate failure; says what happened to results.json."""
    kept = ('results.json renamed to results.gate_failed.json' if renamed else
            'no results.json had been written; the per-round prediction files are kept')
    return ('STOPPING the block: IDENTITY GATE FAILED for %s: %s. Not retried. The run '
            'directory keeps its output (%s) and %s; the block will not start again until '
            'a human has resolved it.' % (out_dir, reason, kept, GATE_MARKER))


def set_aside_predictions(run, attempt):
    """Move a gated run's prediction files out of the way before an attempt, so the
    live gate judges only files that this attempt wrote.  Nothing is deleted."""
    pred = os.path.join(run['out_dir'], 'predictions')
    if not os.path.isdir(pred) or not os.listdir(pred):
        return None
    os.makedirs(INVALID_DIR, exist_ok=True)
    parts = [p for p in re.split(r'[\\/]+', os.path.normpath(run['out_dir'])) if p]
    dest = os.path.join(INVALID_DIR, '%s__before_try%d__%s' % (
        '__'.join(parts[-2:]).replace(':', ''), attempt, datetime.now().strftime('%Y%m%d_%H%M%S')))
    os.rename(pred, dest)
    return dest


def gated_rounds_on_disk_differ(run):
    """Before a retry: the reason if any gated round already on disk differs."""
    for ref, rounds in run.get('gates', []):
        for rnd in rounds:
            state, detail = compare_round(run['out_dir'], ref, rnd)
            if state == 'differ':
                return detail
    return None


class IdentityMonitor:
    """Checks gated rounds as soon as their prediction files are complete."""

    def __init__(self, run):
        self.out_dir = run['out_dir']
        self.todo = [(ref, rnd) for ref, rounds in run.get('gates', []) for rnd in rounds]
        self.done = []

    def poll(self):
        """-> None, or the reason of the first mismatch."""
        still = []
        for ref, rnd in self.todo:
            state, detail = compare_round(self.out_dir, ref, rnd)
            if state == 'differ':
                return detail
            if state == 'same':
                self.done.append((ref, rnd))
                log('    identity gate: round %d == %s' % (rnd, ref))
            else:
                still.append((ref, rnd))
        self.todo = still
        return None


def final_identity_check(run):
    """Every gated round, at the end: predictions and aggregated validation loss.

    -> (ok, detail).  Rounds whose files are still missing count as a failure."""
    out_dir = run['out_dir']
    try:
        with open(os.path.join(out_dir, 'results.json')) as f:
            got = {r['round']: r.get('weighted_val_loss') for r in json.load(f).get('rounds', [])}
    except Exception as e:
        return False, 'results.json unreadable for the identity check: %r' % e
    for ref, rounds in run.get('gates', []):
        try:
            with open(os.path.join(ref, 'results.json')) as f:
                want = {r['round']: r.get('weighted_val_loss') for r in json.load(f).get('rounds', [])}
        except Exception as e:
            return False, 'reference %s unreadable: %r' % (ref, e)
        for rnd in rounds:
            state, detail = compare_round(out_dir, ref, rnd)
            if state != 'same':
                return False, detail if state == 'differ' else 'round %d missing: %s' % (rnd, detail)
            if got.get(rnd) is None or got.get(rnd) != want.get(rnd):
                return False, ('round %d aggregated validation loss %r != %r in %s'
                               % (rnd, got.get(rnd), want.get(rnd), ref))
    n = sum(len(rounds) for _, rounds in run.get('gates', []))
    return True, '%d gated round(s) bitwise identical' % n


def write_gate_marker(run, reason):
    os.makedirs(run['out_dir'], exist_ok=True)
    with open(os.path.join(run['out_dir'], GATE_MARKER), 'w') as f:
        json.dump({'reason': reason, 'gates': run.get('gates', []), 'time': ts()}, f, indent=2)


# --- energy (tegrastats) ------------------------------------------------------
ENERGY_INTERVAL_MS = 1000
RE_TEGRA = re.compile(r'^(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d) .*?\bVDD_IN (\d+)mW/')


def parse_tegrastats(text, tz_shift=0.0):
    """-> [(epoch_seconds, milliwatts)] from tegrastats output lines.

    tegrastats stamps are the node's LOCAL time without a zone; ``time.mktime`` reads
    them in node_a's zone, so ``tz_shift`` (seconds, see EnergyLogger.clock_offsets)
    removes the difference between the node's zone and node_a's."""
    out = []
    for line in text.splitlines():
        m = RE_TEGRA.match(line.strip())
        if m:
            t = time.mktime(time.strptime(m.group(1), '%m-%d-%Y %H:%M:%S')) - tz_shift
            out.append((t, int(m.group(2))))
    return out


def integrate_energy(samples, t0, t1, interval_s):
    """Energy (J) of the window [t0, t1] as the mean board power of the samples inside
    it times the window length -- a missing sample is not counted as zero power.
    ``coverage`` (samples x interval / window) is the quality indicator."""
    inside = [mw for t, mw in samples if t0 <= t <= t1]
    mean_w = sum(inside) / 1000.0 / len(inside) if inside else None
    window = t1 - t0
    return {'energy_J': round(mean_w * window, 1) if mean_w is not None else None,
            'n_samples': len(inside),
            'mean_W': round(mean_w, 3) if mean_w is not None else None,
            'window_s': round(window, 1),
            'coverage': round(len(inside) * interval_s / window, 3) if window > 0 else None}


class EnergyLogger:
    """tegrastats on every node for one run.  Each instance is wrapped in `timeout`
    so it can never outlive the run, and stopped by its unique log path."""

    def __init__(self, tag, attempt, stamp, max_s):
        self.name = '%s_try%d_%s' % (tag, attempt, stamp)
        self.max_s = int(max_s)
        self.procs = {}
        self.paths = {}
        self.offsets = {}
        self.tz_shift = {}

    def _path(self, node):
        return '%s/logs/energy/%s_%s.tegrastats' % (NODES[node]['repo'], self.name, node)

    def _pattern(self, node):
        # "tegrastat[s]" matches tegrastats but not this pkill's own command line
        return 'tegrastat[s] --interval %d --logfile %s' % (ENERGY_INTERVAL_MS, self._path(node))

    def clock_offsets(self):
        """Per node: clock offset (node epoch minus node_a epoch, s) and the zone shift of
        its local-time stamps as read by node_a's ``time.mktime``, from one
        `date '+%s.%N|%m-%d-%Y %H:%M:%S'` bracketed by node_a's clock."""
        for node, cfg in NODES.items():
            cmd = "date '+%s.%N|%m-%d-%Y %H:%M:%S'"
            t0 = time.time()
            try:
                if cfg['local']:
                    out = subprocess.run(['bash', '-lc', cmd], stdout=subprocess.PIPE,
                                         text=True, timeout=30).stdout
                else:
                    out = ssh(node, cmd, timeout=30).stdout
                t1 = time.time()
                epoch, local = (out or '').strip().split('|')
                epoch = float(epoch)
                self.offsets[node] = 0.0 if cfg['local'] else round(epoch - (t0 + t1) / 2.0, 3)
                self.tz_shift[node] = round(
                    time.mktime(time.strptime(local, '%m-%d-%Y %H:%M:%S')) - int(epoch), 0)
            except (ValueError, subprocess.TimeoutExpired):
                self.offsets[node] = None
                self.tz_shift[node] = None

    def start(self):
        self.clock_offsets()
        for node, cfg in NODES.items():
            path = self._path(node)
            self.paths[node] = path
            inner = ('mkdir -p %s && exec timeout %d tegrastats --interval %d --logfile %s'
                     % (shlex.quote(os.path.dirname(path)), self.max_s, ENERGY_INTERVAL_MS,
                        shlex.quote(path)))
            out = open(os.devnull, 'wb')
            if cfg['local']:
                argv = ['bash', '-lc', inner]
            else:
                argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                        '%s@%s' % (cfg['user'], cfg['host']), 'bash -lc %s' % shlex.quote(inner)]
            self.procs[node] = (_popen(argv, out), out)

    def stop(self):
        for node, cfg in NODES.items():
            cmd = "pkill -f %s || true" % shlex.quote(self._pattern(node))
            try:
                if cfg['local']:
                    subprocess.run(['bash', '-lc', cmd], timeout=30)
                else:
                    ssh(node, cmd, timeout=30)
            except subprocess.TimeoutExpired:
                pass
        for node, (proc, out) in self.procs.items():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _kill(proc)
            out.close()

    def collect(self, out_dir, t0, t1):
        """Copy the logs into <run>/energy/ and return the results.json block."""
        dest_dir = os.path.join(out_dir, 'energy')
        os.makedirs(dest_dir, exist_ok=True)
        nodes = {}
        for node, cfg in NODES.items():
            dest = os.path.join(dest_dir, '%s.tegrastats' % node)
            try:
                if cfg['local']:
                    subprocess.run(['cp', self.paths[node], dest], check=True, timeout=60)
                else:
                    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                    '%s@%s:%s' % (cfg['user'], cfg['host'], self.paths[node]), dest],
                                   check=True, timeout=120)
                if self.offsets.get(node) is None or self.tz_shift.get(node) is None:
                    raise ValueError('clock offset of %s could not be measured' % node)
                with open(dest) as f:
                    samples = parse_tegrastats(f.read(), self.tz_shift[node])
            except Exception as e:
                nodes[node] = {'error': repr(e)[:200]}
                continue
            off = self.offsets[node]
            nodes[node] = integrate_energy(samples, t0 + off, t1 + off, ENERGY_INTERVAL_MS / 1000.0)
            nodes[node]['clock_offset_s'] = off
            nodes[node]['tz_shift_s'] = self.tz_shift[node]
        complete = all('error' not in v and v.get('energy_J') is not None
                       and (v.get('coverage') or 0) >= 0.95 for v in nodes.values())
        total = sum(v.get('energy_J') or 0.0 for v in nodes.values()) if complete else None
        return {'source': 'tegrastats VDD_IN (board input power), %d ms interval' % ENERGY_INTERVAL_MS,
                'window': 'server start to server exit, node_a clock; node clocks corrected by '
                          'the measured offset',
                'window_start': t0, 'window_end': t1,
                'nodes': nodes, 'total_energy_J': round(total, 1) if total is not None else None,
                'complete': complete}


def record_energy(out_dir, block):
    path = os.path.join(out_dir, 'results.json')
    with open(path) as f:
        data = json.load(f)
    data['energy'] = block
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def execute_run(run, attempt, ready_timeout, run_timeout, poll=5.0, finish_grace=600,
                monitor=None, times=None):
    """Server first, clients once the port accepts, then watch all four processes.

    -> (rc, reason).  rc is the server's exit code, or negative when the runner
    ended the run: -1 timeout, -2 server never became ready, -3 a client died,
    -4 the server did not exit after every client had finished, -5 an identity
    gate failed (``monitor.poll()`` returned a reason).  ``times`` (a dict) gets
    the node_a wall-clock of server start and server exit.

    A client that exits with a non-zero code before the server exits ends the run
    at once: without it the server can never complete.  Exit code 0 is a client's
    normal end (the server disconnects the clients, then writes results.json and
    exits), so it only starts the ``finish_grace`` window for the server.
    """
    tag = os.path.basename(run['out_dir'])
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    server, clients, handles = None, [], []
    try:
        slog = os.path.join(LOG_DIR, '%s_try%d_server_%s.log' % (tag, attempt, stamp))
        server, sh = start_server(run, slog)
        handles.append(sh)
        if times is not None:
            times['server_start'] = time.time()
        log('    server started (pid %d); waiting for %s:%d to accept connections'
            % (server.pid, NODES['node_a']['host'], SERVER_PORT))
        ready, why = wait_for_port(NODES['node_a']['host'], SERVER_PORT, ready_timeout,
                                   server=server)
        if not ready:
            log('    SERVER NOT READY: %s\n%s' % (why, log_tail(slog)))
            return -2, why
        log('    server accepting connections')

        for node in ('node_a', 'node_b', 'node_c'):
            lp = os.path.join(LOG_DIR, '%s_try%d_%s_%s.log' % (tag, attempt, node, stamp))
            p, h = run_on(node, run['clients'][node], lp)
            clients.append((node, p, lp))
            handles.append(h)
            log('    client up on %s (pid %d)' % (node, p.pid))

        # run timeout and finish grace on the monotonic clock: an NTP step after a reboot
        # (2026-09-28: ~22 min) must not end a run early; server_start/server_exit stay
        # wall-clock because the energy log is aligned to tegrastats timestamps
        start = time.monotonic()
        all_done_at = None
        while server.poll() is None:
            for node, p, lp in clients:
                rc = p.poll()
                if rc is not None and rc != 0:
                    why = 'client %s exited with rc=%s before the server' % (node, rc)
                    log('    CLIENT DIED: %s -- ending the run now. Last lines of its log '
                        '(%s):\n%s' % (why, lp, log_tail(lp)))
                    return -3, why
            if all_done_at is None and all(p.poll() == 0 for _, p, _ in clients):
                all_done_at = time.monotonic()
                log('    all clients finished; waiting for the server to exit')
            if all_done_at is not None and time.monotonic() - all_done_at > finish_grace:
                why = 'server still running %ds after every client finished' % finish_grace
                log('    %s\n%s' % (why, log_tail(slog)))
                return -4, why
            if time.monotonic() - start > run_timeout:
                why = 'TIMEOUT after %ds' % run_timeout
                log('    %s -- killing the run' % why)
                return -1, why
            if monitor is not None:
                reason = monitor.poll()
                if reason:
                    log('    IDENTITY GATE FAIL: %s -- ending the run now' % reason)
                    return -5, 'identity gate failed: ' + reason
            time.sleep(poll)
        if times is not None:
            times['server_exit'] = time.time()
        return server.returncode, 'server exited'
    finally:
        if server is not None:
            _kill(server, getattr(signal, 'SIGKILL', signal.SIGTERM))   # no SIGKILL on Windows
        for _, p, _ in clients:
            _kill(p)
        time.sleep(3)
        kill_stragglers()
        for h in handles:
            h.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--block', required=False,
                    help='block name passed to print_revision_commands.py')
    ap.add_argument('--power_config', required=True, choices=POWER_CONFIGS,
                    help='power modes the block runs under; every node must report the '
                         'mode declared for it under power_modes in testbed.local.yaml. '
                         'Runs of any configuration but %s go to results/pc_<name>/'
                         % DEFAULT_POWER_CONFIG)
    ap.add_argument('--script', help='use an already-generated bash file instead (it must '
                                     'have been generated with --all_seeds)')
    ap.add_argument('--server_ready_timeout', type=int, default=180,
                    help='seconds to wait for the server to accept TCP connections on '
                         'node_a before the clients are started (the run fails if it never '
                         'does)')
    ap.add_argument('--finish_grace', type=int, default=600,
                    help='seconds the server may keep running after every client has '
                         'exited normally')
    ap.add_argument('--gap', type=int, default=30, help='seconds between runs')
    ap.add_argument('--round_timeout', type=int, default=4 * 3600,
                    help='per-round allowance: a run is killed after rounds x this many '
                         'seconds (default 4 h/round; the slowest planned rounds, FedProx on '
                         'Dirichlet 0.5, are estimated at ~2.5 h)')
    ap.add_argument('--run_timeout', type=int, default=None,
                    help='fixed per-run limit in seconds, overriding --round_timeout')
    ap.add_argument('--check_only', action='store_true',
                    help='run the pre-flight checks and exit')
    ap.add_argument('--dry_run', action='store_true', help='list the runs and exit')
    ap.add_argument('--allow_commit_mismatch', action='store_true')
    ap.add_argument('--energy', action='store_true',
                    help='log board input power (tegrastats VDD_IN) on every node during each '
                         'run and record the energy of the run window in results.json')
    ap.add_argument('--testbed', default=TESTBED_LOCAL,
                    help='node hosts/users/paths (default: %s; copy %s to create it)'
                         % (TESTBED_LOCAL, TESTBED_EXAMPLE))
    args = ap.parse_args()

    # every path below (logs/, results/, scripts/, analysis/) is repo-relative
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(LOG_DIR, exist_ok=True)
    if not (args.check_only or args.dry_run):
        # scripts/resume_after_reboot.py finds the last invocation by this line and
        # restarts it only if nothing after it records an end (block finished,
        # STOPPING, a failed pre-flight, a gate failure)
        log('run_matrix start (pid %d): %s' % (os.getpid(), json.dumps(sys.argv[1:])))
    global SERVER_PORT
    nodes, SERVER_PORT = load_testbed(args.testbed)
    NODES.clear()
    NODES.update(nodes)
    power = load_power_modes(args.testbed, args.power_config)
    if args.block == 'baselines_extension':
        raise SystemExit('baselines_extension runs on the desktop GPU '
                         '(scripts/make_desktop_lanes.py), not on the testbed')

    if args.check_only:
        problems = preflight(strict_commit=not args.allow_commit_mismatch,
                             splits=sorted(expected_digests()), power=power)
        if problems:
            print('PRE-FLIGHT FAIL')
            for p in problems:
                print('  -', p)
            sys.exit(1)
        print('PRE-FLIGHT OK -- all three nodes reachable, same commit, resources free, '
              'GUI off, all %d split manifests match P0_SUMMARY, power modes %s (%s)'
              % (len(expected_digests()), args.power_config,
                 ', '.join('%s=%s' % (n, power[n]) for n in NODE_NAMES)))
        return

    if args.script:
        script = args.script
    else:
        if not args.block:
            raise SystemExit('need --block or --script')
        suffix = '' if args.power_config == DEFAULT_POWER_CONFIG else '_pc_%s' % args.power_config
        script = os.path.join(LOG_DIR, 'block_%s%s.sh' % (args.block, suffix))
        with open(script, 'w') as f:
            subprocess.run(['python3', 'scripts/print_revision_commands.py',
                            '--all_seeds', '--block', args.block, '--format', 'bash',
                            '--power_config', args.power_config],
                           stdout=f, check=True)

    runs = parse_block(script)
    check_server_address(runs)
    check_namespace(runs, args.power_config, args.block)
    failed_gates = unresolved_gate_failures(runs, namespace_root(args.power_config, args.block))
    if failed_gates:
        log('PRE-FLIGHT FAIL -- unresolved identity-gate failure(s), not starting: %s. A human '
            'must decide what they mean before the block may continue.' % ', '.join(failed_gates))
        sys.exit(1)
    check_gate_references(runs)
    det_gate = block_determinism_gate(args.block) if args.block else None
    if det_gate:
        det_ok, det_why = check_determinism_gate(det_gate)
        pair = ' vs '.join(map(str, det_gate.get('runs') or []))
        if not det_ok:
            log('PRE-FLIGHT FAIL -- determinism gate (%s): %s. The block does not start '
                '(docs/CROSS_CONFIG_COMPARISON.md).' % (pair, det_why))
            sys.exit(1)
        log('determinism gate OK (%s): %s' % (pair, det_why))
    todo = [r for r in runs if not os.path.isfile(os.path.join(r['out_dir'], 'results.json'))]
    log('block script: %s (power_config %s: %s)'
        % (script, args.power_config, ', '.join('%s=%s' % (n, power[n]) for n in NODE_NAMES)))
    log('%d run(s) defined, %d still to do' % (len(runs), len(todo)))
    for r in todo:
        log('   - %s' % r['out_dir'])
    if args.dry_run:
        return
    if not todo:
        log('nothing to do')
        return

    splits = block_splits(todo)
    problems = preflight(strict_commit=not args.allow_commit_mismatch, splits=splits,
                         power=power)
    if problems:
        log('PRE-FLIGHT FAIL -- not starting:')
        for p in problems:
            log('   - %s' % p)
        sys.exit(1)
    log('pre-flight OK')

    done = failed = 0
    for i, run in enumerate(todo, 1):
        log('=== [%d/%d] %s ===' % (i, len(todo), run['out_dir']))
        ok = False
        preflight_failed = False
        gate_failed = None
        gate_renamed = False
        m = RE_ROUNDS.search(run['server'])
        timeout = args.run_timeout or int(m.group(1) if m else 3) * args.round_timeout
        for attempt in (1, 2):
            if attempt == 2:
                log('    retry with identical parameters')
                time.sleep(60)
            before = {}
            problems = preflight(strict_commit=not args.allow_commit_mismatch,
                                 splits=block_splits([run]), power=power, measured=before)
            if problems:
                log('    pre-flight FAIL before this run:')
                for p in problems:
                    log('      - %s' % p)
                preflight_failed = True
                break
            freed = release_page_cache()
            log('    page cache released (MemFree MB before -> after): %s'
                % ', '.join('%s %s' % (n, '%d -> %d' % f if f else 'unreadable')
                            for n, f in freed.items()))
            if run.get('gates'):
                if attempt > 1:
                    reason = gated_rounds_on_disk_differ(run)
                    if reason:
                        log('    IDENTITY GATE FAIL in the failed attempt\'s rounds: %s' % reason)
                        gate_renamed = fail_gate(run, reason)
                        gate_failed = reason
                        ok = False
                        break
                moved = set_aside_predictions(run, attempt)
                if moved:
                    log('    earlier prediction files moved to %s' % moved)
            monitor = IdentityMonitor(run) if run.get('gates') else None
            energy = None
            if args.energy:
                energy = EnergyLogger(os.path.basename(run['out_dir']), attempt,
                                      datetime.now().strftime('%Y%m%d_%H%M%S'), timeout + 1800)
                energy.start()
            times = {}
            # the logged per-run duration: monotonic, so an NTP step after a reboot cannot
            # inflate it (2026-09-28: ~22 min); the log line timestamps stay wall-clock
            t0 = time.monotonic()
            try:
                rc, ended = execute_run(run, attempt, args.server_ready_timeout, timeout,
                                        finish_grace=args.finish_grace, monitor=monitor,
                                        times=times)
            finally:
                if energy is not None:
                    energy.stop()
            dt = time.monotonic() - t0
            if rc == -5:
                gate_renamed = fail_gate(run, ended)
                gate_failed = ended
                ok = False
                break
            ok, why = result_ok(run['out_dir'])
            if ok and run.get('gates'):
                gate_ok, gate_why = final_identity_check(run)
                if not gate_ok:
                    gate_renamed = fail_gate(run, gate_why)
                    log('    IDENTITY GATE FAIL: %s' % gate_why)
                    gate_failed = gate_why
                    ok = False          # a finished run whose gate failed is NOT done
                    break
                why = '%s; %s' % (why, gate_why)
            if ok:
                _, after = check_power(power)
                ok, power_why = record_power(run['out_dir'], args.power_config, power,
                                             before, after)
                why = '%s; %s' % (why, power_why)
            if ok and energy is not None and 'server_exit' in times:
                try:
                    block = energy.collect(run['out_dir'], times['server_start'],
                                           times['server_exit'])
                    record_energy(run['out_dir'], block)
                    why = ('%s; energy %.0f kJ' % (why, block['total_energy_J'] / 1000.0)
                           if block['complete'] else '%s; energy INCOMPLETE (see results.json)' % why)
                except Exception as e:           # energy is auxiliary: never fails a run
                    why = '%s; energy not recorded (%r)' % (why, e)
            log('    %s (rc=%s), %.1f min, result: %s (%s)'
                % (ended, rc, dt / 60.0, 'OK' if ok else 'BAD', why))
            if ok:
                break
        if ok:
            done += 1
        else:
            failed += 1
            if gate_failed:
                log(gate_stop_message(run['out_dir'], gate_failed, gate_renamed))
            elif preflight_failed:
                log('STOPPING the block: pre-flight failed before %s (see above). Fix the '
                    'cause, then re-run this command; finished runs are skipped '
                    'automatically.' % run['out_dir'])
            else:
                log('STOPPING the block: %s did not produce a valid result after two '
                    'identical attempts. Fix the cause, then re-run this command; '
                    'finished runs are skipped automatically.' % run['out_dir'])
            break
        if i < len(todo):
            time.sleep(args.gap)

    log('block finished: %d ok, %d failed, %d not attempted'
        % (done, failed, len(todo) - done - failed))
    log('next: python3 scripts/analyze_results.py --results_dir results --output_dir analysis')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
