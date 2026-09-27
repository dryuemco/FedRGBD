#!/usr/bin/env python3
"""FedRGBD -- smoke test and overhead measurement of tegrastats energy logging.

Runs on ONE Jetson node, alone (nothing else may be running).  It trains the FL
client's model on the node's own training split with the client's settings
(MobileNetV3-Small, Adam 1e-3, batch 8, num_workers=0 -- the same CPU-bound JPEG
data path as ``src/fl/client.py``) and times fixed-size blocks of batches with
tegrastats off (A) and on (B).

Design, so that a 1 % effect can be resolved rather than guessed:

* the page cache is warmed first by reading every file of the split once, so the
  timed blocks run in the FL client's steady state (after its first epoch), not as
  a cold random-read workload;
* blocks come in ABBA quadruples, which cancel any linear drift (thermal, clocks)
  exactly; every block -- A or B -- is preceded by the same idle pause, during
  which tegrastats is started for a B block;
* the effect is estimated from the paired differences within each quadruple:
  d_q = mean(B) / mean(A) - 1, reported with its mean, SD and 95 % t interval.

Verdict, declared before measuring:

* ``negligible``      the 95 % CI of the relative throughput change lies inside
                      +/-1 % AND tegrastats uses at most 1 % of one CPU core AND
                      the log carries VDD_IN samples;
* ``not negligible``  the CI lies entirely outside +/-1 %, or the CPU rule fails;
* ``inconclusive``    anything else: extend with --quads, do not decide on it.

tegrastats is started under ``timeout`` and stopped in a ``finally``, so it can
never outlive the script.  Writes one JSON file; does not touch results/.

    python3 scripts/energy_smoke.py --data_dir data/processed/iid/node_c \\
        --out logs/energy/smoke_node_c.json
"""

import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

RE_TEGRA = re.compile(r'^(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d) .*?\bVDD_IN (\d+)mW/')
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 19: 2.093, 24: 2.064, 29: 2.045}


def t975(df):
    keys = sorted(k for k in T975 if k <= df)
    return T975[keys[-1]] if keys else float('nan')


def proc_cpu_seconds(pid):
    """utime + stime of a process, in seconds (None if it is gone).

    /proc/<pid>/stat fields 14 and 15 (1-based); after splitting off everything up
    to the closing parenthesis of the command name, field 3 (state) is index 0, so
    utime and stime are indices 11 and 12."""
    try:
        with open('/proc/%d/stat' % pid) as f:
            fields = f.read().rsplit(')', 1)[1].split()
        ticks = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
        return (int(fields[11]) + int(fields[12])) / float(ticks)
    except (OSError, IndexError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batches', type=int, default=80, help='batches per timed block')
    ap.add_argument('--quads', type=int, default=6, help='ABBA quadruples')
    ap.add_argument('--warmup', type=int, default=40)
    ap.add_argument('--pause', type=float, default=2.0, help='idle seconds before every block')
    ap.add_argument('--interval_ms', type=int, default=1000)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from src.data.dataset import FlameDataset
    from src.models.mobilenetv3_multimodal import create_model

    torch.manual_seed(0)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = FlameDataset(args.data_dir, split='train', img_size=224)
    # warm the page cache: the FL client is in this state after its first epoch
    t_read = time.time()
    n_bytes = 0
    for sample in getattr(ds, 'samples', []):
        path = sample[0] if isinstance(sample, (tuple, list)) else sample
        with open(path, 'rb') as f:
            n_bytes += len(f.read())
    t_read = time.time() - t_read
    g = torch.Generator()
    g.manual_seed(0)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=False, generator=g)
    model = create_model(num_classes=2, in_channels=3, pretrained=False).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    it = iter(loader)

    def batches(n):
        nonlocal it
        seen = 0
        for _ in range(n):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            seen += len(y)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        return seen

    batches(args.warmup)
    logdir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(logdir, exist_ok=True)
    order = ['A', 'B', 'B', 'A'] * args.quads
    budget = int(3 * args.batches * 8 / 20.0 + 60)          # generous per-block bound (s)
    blocks, tegra = [], []
    for k, mode in enumerate(order):
        proc = None
        logfile = None
        try:
            if mode == 'B':
                logfile = os.path.join(logdir, 'smoke_%s_block%02d.tegrastats' % (socket.gethostname(), k))
                proc = subprocess.Popen(['timeout', str(budget), 'tegrastats', '--interval',
                                         str(args.interval_ms), '--logfile', logfile],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(args.pause)                          # the same idle before A and B
            tpid = None
            if proc is not None:
                try:
                    out = subprocess.run(['pgrep', '-P', str(proc.pid)], stdout=subprocess.PIPE,
                                         text=True, timeout=10).stdout.split()
                    tpid = int(out[0]) if out else None
                except (subprocess.SubprocessError, ValueError):
                    tpid = None
                cpu0, w0 = (proc_cpu_seconds(tpid) if tpid else None), time.time()
            t0 = time.perf_counter()
            n = batches(args.batches)
            dt = time.perf_counter() - t0
            blocks.append({'mode': mode, 'images': n, 'seconds': round(dt, 3), 'ips': round(n / dt, 4)})
            if proc is not None:
                cpu1, w1 = (proc_cpu_seconds(tpid) if tpid else None), time.time()
                tegra.append({'cpu_pct_of_one_core': (round(100.0 * (cpu1 - cpu0) / (w1 - w0), 3)
                                                      if cpu0 is not None and cpu1 is not None else None)})
        finally:
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
        if logfile:
            with open(logfile) as f:
                samples = [(m.group(1), int(m.group(2))) for m in map(RE_TEGRA.match, f) if m]
            tegra[-1].update({'n_samples': len(samples),
                              'mean_VDD_IN_W': (round(sum(v for _, v in samples) / 1000.0 / len(samples), 3)
                                                if samples else None),
                              'logfile': logfile})

    ips = [b['ips'] for b in blocks]
    diffs = []
    for q in range(args.quads):
        a = (ips[4 * q] + ips[4 * q + 3]) / 2.0
        b = (ips[4 * q + 1] + ips[4 * q + 2]) / 2.0
        diffs.append(b / a - 1.0)
    m = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)) if len(diffs) > 1 else float('nan')
    half = t975(len(diffs) - 1) * sd / math.sqrt(len(diffs)) if len(diffs) > 1 else float('nan')
    lo, hi = m - half, m + half
    cpu = [t['cpu_pct_of_one_core'] for t in tegra if t.get('cpu_pct_of_one_core') is not None]
    logged = bool(tegra) and all(t.get('n_samples', 0) > 0 for t in tegra)
    cpu_ok = bool(cpu) and max(cpu) <= 1.0
    if lo >= -0.01 and hi <= 0.01 and cpu_ok and logged:
        verdict = 'negligible'
    elif (hi < -0.01 or lo > 0.01) or (cpu and not cpu_ok) or not logged:
        verdict = 'not negligible'
    else:
        verdict = 'inconclusive'
    a_ips = [b['ips'] for b in blocks if b['mode'] == 'A']
    out = {
        'host': socket.gethostname(), 'data_dir': args.data_dir, 'timezone': time.strftime('%Z%z'),
        'device': str(dev), 'cache_warm_read_s': round(t_read, 1), 'cache_warm_bytes': n_bytes,
        'blocks': blocks, 'tegrastats': tegra,
        'warm_ips_off_mean': round(sum(a_ips) / len(a_ips), 3),
        'quad_relative_change': [round(d, 5) for d in diffs],
        'relative_change_mean': round(m, 5), 'relative_change_sd': round(sd, 5),
        'relative_change_ci95': [round(lo, 5), round(hi, 5)],
        'max_tegrastats_cpu_pct_of_one_core': max(cpu) if cpu else None,
        'vdd_in_logged': logged, 'verdict': verdict,
        'rule': 'negligible iff the 95% CI of the relative throughput change lies within +/-1% '
                'and tegrastats uses <= 1% of one core and VDD_IN is logged; declared before '
                'measuring',
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ('host', 'warm_ips_off_mean', 'relative_change_mean',
                                          'relative_change_ci95', 'max_tegrastats_cpu_pct_of_one_core',
                                          'vdd_in_logged', 'verdict')}))


if __name__ == '__main__':
    main()
