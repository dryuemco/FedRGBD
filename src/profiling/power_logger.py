"""
FedRGBD — Per-Round Power Logger
==================================
Context manager that runs tegrastats in the background during an FL round
and computes energy consumption for that round.

Usage:
    from src.profiling.power_logger import PowerLogger

    logger = PowerLogger(interval_ms=100)

    # During each FL round:
    with logger.measure_round(round_num=1) as measurement:
        # ... local training happens here ...
        pass

    print(measurement)  # {'round': 1, 'duration_s': 12.3, 'energy_wh': 0.051, ...}

    # Save all rounds
    logger.save_results("results/experiment_4/power_log.json")
"""

import json
import os
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from .tegrastats_parser import parse_logfile, compute_summary


class PowerLogger:
    """Logs power consumption per FL round using tegrastats."""

    def __init__(self, interval_ms=100):
        """
        Args:
            interval_ms: tegrastats sampling interval in milliseconds.
                         100ms gives good resolution without too much overhead.
        """
        self.interval_ms = interval_ms
        self.round_results = []
        self._temp_dir = tempfile.mkdtemp(prefix="fedrgbd_power_")

    @contextmanager
    def measure_round(self, round_num, extra_info=None):
        """Context manager to measure power during an FL round.

        Args:
            round_num: FL round number
            extra_info: dict of extra info to attach (e.g., strategy name)

        Yields:
            dict that will be populated with measurement results after the block.
        """
        result = {
            "round": round_num,
            "extra": extra_info or {},
        }

        logfile = os.path.join(self._temp_dir, f"round_{round_num:03d}.txt")

        # Start tegrastats in background
        proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms),
             "--logfile", logfile],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid  # Create new process group for clean kill
        )

        start_time = time.perf_counter()

        try:
            yield result
        finally:
            end_time = time.perf_counter()
            wall_time = end_time - start_time

            # Stop tegrastats
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass

            # Small delay to ensure file is flushed
            time.sleep(0.2)

            # Parse the log
            result["duration_s"] = round(wall_time, 3)

            if os.path.exists(logfile) and os.path.getsize(logfile) > 0:
                try:
                    df = parse_logfile(logfile)
                    summary = compute_summary(df, interval_ms=self.interval_ms)
                    result.update(summary)
                except Exception as e:
                    result["parse_error"] = str(e)
            else:
                result["parse_error"] = "tegrastats log file empty or missing"

            self.round_results.append(result)

    def get_results(self):
        """Return all round measurements."""
        return self.round_results

    def save_results(self, filepath):
        """Save all round measurements to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.round_results, f, indent=2)

        print(f"Power log saved: {path} ({len(self.round_results)} rounds)")

    def print_summary(self):
        """Print a summary table of all rounds."""
        if not self.round_results:
            print("No rounds logged yet.")
            return

        print(f"\n{'Round':>6} | {'Duration(s)':>11} | {'Avg Power(W)':>12} | "
              f"{'Energy(Wh)':>10} | {'Peak RAM(MB)':>12} | {'GPU Freq(%)':>11}")
        print("-" * 80)

        for r in self.round_results:
            print(f"{r.get('round', '?'):>6} | "
                  f"{r.get('duration_s', 0):>11.2f} | "
                  f"{r.get('power_avg_w', 0):>12.3f} | "
                  f"{r.get('energy_total_wh', 0):>10.5f} | "
                  f"{r.get('ram_peak_mb', 0):>12} | "
                  f"{r.get('gpu_freq_avg_percent', 0):>11.1f}")

    def cleanup(self):
        """Remove temporary log files."""
        import shutil
        try:
            shutil.rmtree(self._temp_dir)
        except Exception:
            pass
