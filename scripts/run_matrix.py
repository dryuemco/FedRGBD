#!/usr/bin/env python3
"""FedRGBD -- run a revision block unattended on the 3-node Jetson testbed.

``print_revision_commands.py --format bash`` prints the server command and tells
the operator to start three clients by hand inside a 10 s window.  That is fine
for one run and unusable for a 20-run, 46-hour block.  This runner parses the
same generated script and drives the whole block:

    for each run:
        pre-flight  : nodes reachable, repo at the same commit, RAM/disk free,
                      port 8080 clear, no stale python processes, GUI off
                      (multi-user.target) on every node, and every split the
                      run reads has the manifest md5 of
                      analysis/leakage/P0_SUMMARY.md on every node
        start       : the server on node_a; once node_a:8080 accepts TCP
                      connections, the clients (node_b / node_c over SSH,
                      node_a locally).  Clients must not start earlier: the
                      default gRPC-bidi Flower client does not retry a
                      refused first connection and dies.
        wait        : until the server exits; a client exiting non-zero
                      before that ends the run at once
        check       : results/<run>/results.json exists and parses, and the
                      model_selection block is present
        on failure  : ONE retry with identical parameters, then stop the block

Deliberately NOT adaptive.  It never changes a batch size, never skips a seed,
never edits a command.  A run either completes exactly as specified or the block
stops and waits for a human.  Experiment comparability depends on that.

Run it on Node A, inside tmux:

    tmux new -s fedrgbd
    python3 scripts/run_matrix.py --block seed_extension
    # detach with Ctrl-b d ; reattach with: tmux attach -t fedrgbd

Requires passwordless SSH from Node A to Node B and Node C (see --check_only), and
configs/testbed.local.yaml with your nodes' hosts, users and paths: copy
configs/testbed.example.yaml and fill it in (the copy is gitignored).
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
                cur = {'out_dir': m.group(1), 'clients': {}, 'server': None}
                pending_node = None
                continue
            if cur is None:
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


def expected_digests(path=P0_SUMMARY):
    """{split: md5} from the 'Split digests' table of P0_SUMMARY.md."""
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = RE_DIGEST.match(line)
            if m:
                out[m.group(1)] = m.group(2)
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


def local_commit():
    r = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, text=True)
    return r.stdout.strip()


def preflight(strict_commit=True, splits=()):
    problems = check_testbed(list(splits)) if splits else []
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


def kill_stragglers():
    for node, cfg in NODES.items():
        cmd = "pkill -f 'src/fl/(client|server).py' || true"
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


def execute_run(run, attempt, ready_timeout, run_timeout, poll=5.0, finish_grace=600):
    """Server first, clients once the port accepts, then watch all four processes.

    -> (rc, reason).  rc is the server's exit code, or negative when the runner
    ended the run: -1 timeout, -2 server never became ready, -3 a client died,
    -4 the server did not exit after every client had finished.

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

        start = time.time()
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
                all_done_at = time.time()
                log('    all clients finished; waiting for the server to exit')
            if all_done_at is not None and time.time() - all_done_at > finish_grace:
                why = 'server still running %ds after every client finished' % finish_grace
                log('    %s\n%s' % (why, log_tail(slog)))
                return -4, why
            if time.time() - start > run_timeout:
                why = 'TIMEOUT after %ds' % run_timeout
                log('    %s -- killing the run' % why)
                return -1, why
            time.sleep(poll)
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
    ap.add_argument('--testbed', default=TESTBED_LOCAL,
                    help='node hosts/users/paths (default: %s; copy %s to create it)'
                         % (TESTBED_LOCAL, TESTBED_EXAMPLE))
    args = ap.parse_args()

    # every path below (logs/, results/, scripts/, analysis/) is repo-relative
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(LOG_DIR, exist_ok=True)
    global SERVER_PORT
    nodes, SERVER_PORT = load_testbed(args.testbed)
    NODES.clear()
    NODES.update(nodes)
    if args.block == 'baselines_extension':
        raise SystemExit('baselines_extension runs on the desktop GPU '
                         '(scripts/make_desktop_lanes.py), not on the testbed')

    if args.check_only:
        problems = preflight(strict_commit=not args.allow_commit_mismatch,
                             splits=sorted(expected_digests()))
        if problems:
            print('PRE-FLIGHT FAIL')
            for p in problems:
                print('  -', p)
            sys.exit(1)
        print('PRE-FLIGHT OK -- all three nodes reachable, same commit, resources free, '
              'GUI off, all %d split manifests match P0_SUMMARY' % len(expected_digests()))
        return

    if args.script:
        script = args.script
    else:
        if not args.block:
            raise SystemExit('need --block or --script')
        script = os.path.join(LOG_DIR, 'block_%s.sh' % args.block)
        with open(script, 'w') as f:
            subprocess.run(['python3', 'scripts/print_revision_commands.py',
                            '--all_seeds', '--block', args.block, '--format', 'bash'],
                           stdout=f, check=True)

    runs = parse_block(script)
    check_server_address(runs)
    todo = [r for r in runs if not os.path.isfile(os.path.join(r['out_dir'], 'results.json'))]
    log('block script: %s' % script)
    log('%d run(s) defined, %d still to do' % (len(runs), len(todo)))
    for r in todo:
        log('   - %s' % r['out_dir'])
    if args.dry_run:
        return
    if not todo:
        log('nothing to do')
        return

    splits = block_splits(todo)
    problems = preflight(strict_commit=not args.allow_commit_mismatch, splits=splits)
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
        m = RE_ROUNDS.search(run['server'])
        timeout = args.run_timeout or int(m.group(1) if m else 3) * args.round_timeout
        for attempt in (1, 2):
            if attempt == 2:
                log('    retry with identical parameters')
                time.sleep(60)
            problems = preflight(strict_commit=not args.allow_commit_mismatch,
                                 splits=block_splits([run]))
            if problems:
                log('    pre-flight FAIL before this run:')
                for p in problems:
                    log('      - %s' % p)
                preflight_failed = True
                break
            t0 = time.time()
            rc, ended = execute_run(run, attempt, args.server_ready_timeout, timeout,
                                    finish_grace=args.finish_grace)
            dt = time.time() - t0
            ok, why = result_ok(run['out_dir'])
            log('    %s (rc=%s), %.1f min, result: %s (%s)'
                % (ended, rc, dt / 60.0, 'OK' if ok else 'BAD', why))
            if ok:
                break
        if ok:
            done += 1
        else:
            failed += 1
            if preflight_failed:
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
