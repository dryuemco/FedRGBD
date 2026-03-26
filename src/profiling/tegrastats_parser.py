"""
FedRGBD — Tegrastats Parser
============================
Parses tegrastats output from NVIDIA Jetson Orin Nano for energy profiling.

tegrastats output format (Jetson Orin Nano, JetPack 6.x):
  RAM 2345/7620MB (lfb 154x4MB) SWAP 0/3810MB (cached 0MB)
  CPU [12%@1510,15%@1510,8%@1510,10%@1510,5%@1510,7%@1510]
  GR3D_FREQ 30% VIC_FREQ 0% GEN_FREQ 0%
  GPU 2500/15000 CPU 1800/15000 SOC 1200/15000 CV 0/15000 VDDRQ 800/15000 SYS5V 3500/15000

Power values are in milliwatts (mW). The format is: current/average.

Usage:
    # Start logging (run in separate terminal during FL training)
    tegrastats --interval 100 --logfile tegrastats_log.txt

    # Parse after experiment
    python3 src/profiling/tegrastats_parser.py tegrastats_log.txt --output results/experiment_4/

    # Parse and get summary statistics
    python3 src/profiling/tegrastats_parser.py tegrastats_log.txt --summary
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_line(line):
    """Parse a single tegrastats output line into a dict of metrics."""
    metrics = {}

    # RAM usage: RAM XXXX/YYYYMB
    ram_match = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
    if ram_match:
        metrics['ram_used_mb'] = int(ram_match.group(1))
        metrics['ram_total_mb'] = int(ram_match.group(2))

    # SWAP usage: SWAP X/YMB
    swap_match = re.search(r'SWAP\s+(\d+)/(\d+)MB', line)
    if swap_match:
        metrics['swap_used_mb'] = int(swap_match.group(1))
        metrics['swap_total_mb'] = int(swap_match.group(2))

    # CPU usage: CPU [XX%@FREQ,YY%@FREQ,...]
    cpu_match = re.search(r'CPU\s+\[([^\]]+)\]', line)
    if cpu_match:
        cpu_cores = cpu_match.group(1).split(',')
        usages = []
        freqs = []
        for core in cpu_cores:
            core_match = re.match(r'(\d+)%@(\d+)', core.strip())
            if core_match:
                usages.append(int(core_match.group(1)))
                freqs.append(int(core_match.group(2)))
            elif core.strip() == 'off':
                usages.append(0)
                freqs.append(0)
        metrics['cpu_usage_percent'] = np.mean(usages) if usages else 0
        metrics['cpu_freq_mhz'] = np.mean(freqs) if freqs else 0
        metrics['cpu_num_active'] = sum(1 for u in usages if u > 0)

    # GPU frequency: GR3D_FREQ XX%
    gpu_freq_match = re.search(r'GR3D_FREQ\s+(\d+)%', line)
    if gpu_freq_match:
        metrics['gpu_freq_percent'] = int(gpu_freq_match.group(1))

    # Power rails (mW): GPU XXXX/YYYY CPU XXXX/YYYY SOC XXXX/YYYY etc.
    # Jetson Orin Nano power rail format: RAIL current/average
    power_rails = ['GPU', 'CPU', 'SOC', 'CV', 'VDDRQ', 'SYS5V']
    for rail in power_rails:
        # Match the power rail pattern (avoid matching CPU usage brackets)
        # Power rails appear after the CPU [] block
        pattern = rf'(?<!\[)\b{rail}\s+(\d+)/(\d+)\b'
        power_match = re.search(pattern, line)
        if power_match:
            metrics[f'power_{rail.lower()}_mw'] = int(power_match.group(1))
            metrics[f'power_{rail.lower()}_avg_mw'] = int(power_match.group(2))

    # Total board power (sum of rails)
    power_current_keys = [k for k in metrics if k.startswith('power_') and not k.endswith('_avg_mw')]
    if power_current_keys:
        metrics['power_total_mw'] = sum(metrics[k] for k in power_current_keys)
        metrics['power_total_w'] = metrics['power_total_mw'] / 1000.0

    # Temperature: various thermal zones
    temp_patterns = [
        (r'cpu@([\d.]+)C', 'temp_cpu_c'),
        (r'gpu@([\d.]+)C', 'temp_gpu_c'),
        (r'soc2@([\d.]+)C', 'temp_soc_c'),
        (r'tj@([\d.]+)C', 'temp_tj_c'),
    ]
    for pattern, name in temp_patterns:
        temp_match = re.search(pattern, line)
        if temp_match:
            metrics[name] = float(temp_match.group(1))

    return metrics


def parse_logfile(filepath):
    """Parse an entire tegrastats log file into a DataFrame."""
    records = []

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            metrics = parse_line(line)
            if metrics:
                metrics['line_num'] = line_num
                records.append(metrics)

    if not records:
        print(f"WARNING: No valid tegrastats data found in {filepath}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df


def compute_summary(df, interval_ms=100):
    """Compute summary statistics from parsed tegrastats data."""
    if df.empty:
        return {}

    duration_s = len(df) * (interval_ms / 1000.0)

    summary = {
        'duration_s': round(duration_s, 2),
        'num_samples': len(df),
        'sampling_interval_ms': interval_ms,
    }

    # Power statistics
    if 'power_total_w' in df.columns:
        summary['power_avg_w'] = round(df['power_total_w'].mean(), 3)
        summary['power_max_w'] = round(df['power_total_w'].max(), 3)
        summary['power_min_w'] = round(df['power_total_w'].min(), 3)
        summary['power_std_w'] = round(df['power_total_w'].std(), 3)
        # Energy = average power × time
        summary['energy_total_wh'] = round(
            summary['power_avg_w'] * duration_s / 3600.0, 4
        )

    # GPU power specifically
    if 'power_gpu_mw' in df.columns:
        summary['gpu_power_avg_w'] = round(df['power_gpu_mw'].mean() / 1000.0, 3)
        summary['gpu_power_max_w'] = round(df['power_gpu_mw'].max() / 1000.0, 3)

    # RAM statistics
    if 'ram_used_mb' in df.columns:
        summary['ram_avg_mb'] = round(df['ram_used_mb'].mean(), 1)
        summary['ram_peak_mb'] = int(df['ram_used_mb'].max())

    # CPU statistics
    if 'cpu_usage_percent' in df.columns:
        summary['cpu_usage_avg_percent'] = round(df['cpu_usage_percent'].mean(), 1)

    # GPU frequency
    if 'gpu_freq_percent' in df.columns:
        summary['gpu_freq_avg_percent'] = round(df['gpu_freq_percent'].mean(), 1)

    # Temperature
    for temp_col in ['temp_cpu_c', 'temp_gpu_c', 'temp_tj_c']:
        if temp_col in df.columns:
            summary[f'{temp_col}_avg'] = round(df[temp_col].mean(), 1)
            summary[f'{temp_col}_max'] = round(df[temp_col].max(), 1)

    return summary


def main():
    parser = argparse.ArgumentParser(description="FedRGBD Tegrastats Parser")
    parser.add_argument("logfile", type=str, help="Path to tegrastats log file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for parsed CSV and summary JSON")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary statistics")
    parser.add_argument("--interval", type=int, default=100,
                        help="Tegrastats sampling interval in ms (default: 100)")

    args = parser.parse_args()

    filepath = Path(args.logfile)
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    print(f"Parsing: {filepath}")
    df = parse_logfile(filepath)

    if df.empty:
        print("No data parsed. Check the log file format.")
        sys.exit(1)

    print(f"Parsed {len(df)} samples")

    if args.summary:
        summary = compute_summary(df, interval_ms=args.interval)
        print("\n--- Summary ---")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save parsed data as CSV
        csv_path = output_path / f"{filepath.stem}_parsed.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

        # Save summary as JSON
        summary = compute_summary(df, interval_ms=args.interval)
        json_path = output_path / f"{filepath.stem}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
