# Declared analysis: wall-clock ratio FedProx(0.01) / FedAvg

Declared 2026-10-13 and committed and pushed before any value below was computed. The
author approved the proposal of `docs/SESSION_HANDOFF_20261013.md` section 4.3 on
2026-10-13. Output `analysis/strategy_timing/`, script `scripts/strategy_timing_ratio.py`.

**Declared after the times were seen.** Every run used here was finished, and the
per-strategy times were in the paper: `tab:time` and `tab:time_maxn` give, per
strategy, the mean, SD and 95 % CI over seeds of the total time, the round-1 time and
the steady-state round time. A ratio of those means can be read off the tables. No
seed-paired ratio, no per-seed ratio and no interval of a ratio had been computed. The
paper states this ("declared on 2026-10-13, after the times were seen").

It fills the timing `\todo`s of `sec:res_time`, `sec:implications` and
`sec:generalise`. The v1 ratio is not reused.

## Statistic

The method declared for the MAXN / heterogeneous timing comparison
(`docs/CROSS_CONFIG_COMPARISON.md` (c) and its amendment), applied to the two strategies:

* Per cell and quantity, per seed s: r_s = T_FedProx,s / T_FedAvg,s, the two runs of the
  same power configuration, partition and seed. A seed enters only when both runs exist.
* The reported ratio is the geometric mean exp(mean_s log r_s), so FedProx / FedAvg and
  FedAvg / FedProx are exact reciprocals.
* Interval: seed bootstrap, the paired seeds resampled with replacement with each pair
  kept together, B = 10,000, 95 % percentile interval. Generator
  `BASE_SEED + crc32("timing|strategy|" + power_config + "|" + partition + "|" + quantity)`.
* **Every ratio is printed with its per-seed ratios r_s** next to the geometric mean and
  the interval.
* **The interval is indicative.** n paired seeds give at most C(2n-1, n) distinct
  resamples (126 for five seeds), and the report says so wherever it prints one.
* Descriptive only: no verdicts, no tests, no Holm. No ratio is described as "equal",
  "equivalent" or "no difference".

## Per-run quantities (timing reporting rule, CLAUDE.md rule 12)

T_round is the test-free round time `rounds[].timing.round_time_s`.

* `steady` -- **the per-round overhead factor (primary):** the median T_round over
  rounds 2..R of each run.
* `round1` -- T_round of round 1 (the cold start), its own ratio.
* `total` -- the run's total time excluding the report-only test evaluation
  (`total_time_excl_test_s`, the "Time" column of `tab:time`), R rounds.
* `prox_round1_vs_avg_steady` -- FedProx's round-1 T_round over FedAvg's steady-state
  T_round of the same seed: how many FedAvg rounds FedProx's first round costs.

**Derived, not a separate statistic:** the FedAvg rounds that FedProx's R-round budget
buys = R x the `total` ratio. Its interval is R x the `total` interval.

## Cells

* **Heterogeneous** (the main matrix, `results/rev_*`, three rounds): IID and label skew,
  FedProx (mu = 0.01) vs FedAvg, seeds 42, 123, 456, 789, 1011. **Primary configuration:**
  the paper's timing statements refer to these cells.
* **MAXN_SUPER** (block 5b, `results/pc_maxn/rev_*_r10`, ten rounds): the same two
  partitions, strategies and seeds. Reported separately and never pooled with the
  heterogeneous cells.

Nothing else is computed: no other mu, partition, strategy or round budget.
