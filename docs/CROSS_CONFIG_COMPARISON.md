# MAXN block: determinism gate and declared cross-configuration comparison

Declared 2026-09-28, after the identity gate of block 5b (`maxn_long_horizon`) failed and
before 5b was launched again. It replaces the cross-configuration identity gates declared
on 2026-09-27 (`docs/REVISION_CHANGES.md`, "New runs after the matrix"). Nothing in this
file may be changed after the first 5b run finishes; a change has to be disclosed in the
paper and the response letter.

## Why the identity gate had to go

On 2026-09-28 04:58 the first 5b run (`results/pc_maxn/rev_iid_fedavg_r10_seed42`, all
three nodes at MAXN_SUPER) failed its gate in round 1: every value of all six round-1
prediction files differed from the heterogeneous matrix's `rev_iid_fedavg_seed42`
(max |logit-margin difference| 1.79, no prediction changed sign). The block stopped as
designed.

Diagnosis (desktop side, then two smoke tests; evidence in `analysis/determinism/`):

* The code does not explain it: no file under `src/` changed between the commit the
  reference ran at (`e85945d`, from node_a's reflog and `run_matrix.log`; `results.json`
  records no commit) and `1025f92`; client commands are identical; no package changed on
  any node since 2026-09-19. `cudnn.deterministic = True`, `cudnn.benchmark = False`
  (`src/fl/client.py`), so kernel choice is not timing-based.
* The number of rounds does not explain it: at the heterogeneous modes the three
  10-round label-skew runs reproduce the 3-round runs bitwise in rounds 1-3.
* What the power modes change (same `nvpmodel` file on all three nodes): node_a
  15W -> MAXN_SUPER changes clocks only (6 CPU cores, TPC mask 240 in both); node_b was
  MAXN_SUPER in both; node_c 7W -> MAXN_SUPER goes from TPC mask 252 (2 of 4 TPCs, 4 SMs)
  and 4 online CPU cores to TPC mask 240 (8 SMs) and 6 cores, plus clocks.
* **Test 1** -- the smoke of `results/smoke_r1` (IID rho = 0.01, FedAvg, 2 rounds, seed 42)
  run twice with all three nodes at MAXN_SUPER, node code `1025f92`:
  `diag_smoke_maxn` and `diag_smoke_maxn_r2` are **bitwise identical** to each other (all
  12 prediction files, every client's training loss, the aggregated validation loss).
  Against `smoke_r1` the round-1 training loss of node_a and node_b is equal to full
  float precision, node_c's is not (0.0939630424718676 vs 0.09717213229435272); from
  there every file differs.
* **Test 2** -- the same smoke with node_c back at 7W, node_a and node_b at MAXN_SUPER
  (`diag_smoke_c7w`): **bitwise identical to `smoke_r1`** in all 12 files, every training
  loss and both aggregated validation losses.

So training is bitwise deterministic within a fixed power configuration, clock changes
alone (node_a) do not change it, and node_c's 7W mode does. 7W changes both the number
of active SMs and the number of online CPU cores; these tests do not separate the two.
The likely mechanism is that cuDNN/cuBLAS choose kernels by heuristics that depend on
the SM count, even in deterministic mode. Consequence: no MAXN run in which node_c
trains can be bitwise identical to the heterogeneous matrix, so the gate could never
pass.

## (a) Hard gate: within-configuration determinism

`maxn_long_horizon` declares `determinism_gate: {runs: [results/diag_smoke_maxn,
results/diag_smoke_maxn_r2]}` in `configs/experiment_matrix.yaml`. Before the block
starts, `scripts/run_matrix.py` checks that the two runs are bitwise identical -- the same
set of prediction files, every array with the same dtype, shape and bytes, every client's
per-round training loss and every round's aggregated validation loss -- and refuses to
start otherwise (`PRE-FLIGHT FAIL -- determinism gate`). Both runs were made at
MAXN_SUPER on all three nodes (checked by the driver before each run) with the node code
`1025f92`; `src/` is unchanged in every later commit up to this declaration. The check
passes today (12 files, 6 client-round losses, 2 validation losses).

The cross-configuration identity gates of 2026-09-27 are removed from the block. The gate
machinery of `run_matrix.py` stays for any block that declares one.

## (b) Declared analysis: MAXN vs heterogeneous, rounds 1-3

Not a gate. Reported whatever it shows. Code: `scripts/cross_config_comparison.py`,
output `analysis/cross_config/`.

* **Cells.** IID and label skew x FedAvg and FedProx(mu = 0.01): the cells of 5b that
  have a heterogeneous 3-round run for every seed (42, 123, 456, 789, 1011). A seed
  enters a cell when both runs exist. **FedBN is not compared**: the heterogeneous matrix
  has no 3-round FedBN run for these partitions (only label-skew 10-round runs, seeds
  42/123/456). Adding a FedBN comparison is a change to this declaration.
* **Per run.** The model of the round in {1, 2, 3} with the lowest aggregated validation
  loss (the declared selection rule restricted to the rounds both configurations have;
  earlier round on ties). For the heterogeneous 3-round run this is its own selected
  round. Value: balanced accuracy pooled over the union of the three clients' test splits
  (CLAUDE.md rule 8, primary aggregation), all held-out images.
* **Difference.** D = mean over paired seeds of (MAXN - heterogeneous), with the
  stratified cluster bootstrap of `seed_paired_diff_ci` (one sequence resample of the
  shared test set applied to both sides, one draw of seed indices applied to both sides;
  B = 10,000; 95 % percentile interval; two-sided bootstrap p as in
  `docs/GLOBAL_EVALUATION.md`).
* **Verdict**, on the interval [L, U] of D:

  | interval | verdict |
  |---|---|
  | L > 0 | "higher at MAXN_SUPER" |
  | L <= 0 <= U | "no detectable difference" |
  | U < 0 | "lower at MAXN_SUPER" |

  Holm correction across the compared cells (one family, m = 4 once every cell has
  pairs): adjusted p < 0.05 -> "higher"/"lower" by the sign of D, otherwise "no
  detectable difference". Both verdicts are reported; the Holm verdict is the headline.
  The wording is fixed: never "equivalent" or "no effect" (this is not an equivalence
  test).

**Amendment (2026-09-28, committed and pushed before 5b launched): a second family,
`ten_rounds`.** For label skew x FedAvg and FedBN, seeds 42, 123 and 456, the MAXN run is
also compared over all ten rounds with the heterogeneous `long_horizon_fedbn` run of the
same cell and seed (`results/rev_noniid_<strategy>_r10_seed<S>`, node_a 15W, node_b
MAXN_SUPER, node_c 7W). Each run's model is the round in {1, ..., 10} with the lowest
aggregated validation loss (the declared selection rule, unrestricted; earlier round on
ties). Same statistic (pooled balanced accuracy, seed-paired difference MAXN -
heterogeneous, stratified cluster bootstrap, B = 10,000), same three verdict phrases, and
its own Holm family (m = 2), never pooled with `rounds_1_3` (m = 4). The `rounds_1_3`
comparison is unchanged and still covers these cells' rounds 1-3 for FedAvg; FedBN
enters only this family. Reported whatever it shows.

**What existed when this was declared.** The heterogeneous 3-round results of every cell
and the heterogeneous 10-round `long_horizon_fedbn` results (published in `analysis/`). Of the MAXN side: only the four smoke runs above (IID
rho = 0.01, 2 rounds, not a 5b cell) and round 1 of the failed run, whose prediction
files were compared with the reference (differences above) and whose node_a client log
printed its round-1 metrics on node_a (val and test balanced accuracy 0.5000: the
round-1 global model predicted no-fire for every node_a image); no prediction in any of
the six files changed sign relative to the reference. No selected-round or seed-paired
statistic of any 5b cell existed.

**The failed run.** `results/pc_maxn/rev_iid_fedavg_r10_seed42` (round-1 prediction files,
`IDENTITY_GATE_FAILED.json`, no `results.json`) is kept unchanged as evidence and enters
no analysis. Before 5b is relaunched it is moved, unchanged, out of the `pc_maxn`
namespace to `results/_gate_failed/` (a relaunch refuses while its marker is in the
namespace, and its directory is the first run's output directory). It stays on node_a:
the fetch job copies only `results/rev_*` and `results/pc_<name>/rev_*`, and the
analysis runs on the desktop copy (were it ever copied, its marker would make
`analyze_results.py` refuse to run).

## (c) Declared analysis: timing, MAXN vs heterogeneous (2026-09-29)

Declared 2026-09-29 and committed and pushed while 5b was still running (26 of its 30 runs
unfinished), before any ratio below had been computed. Descriptive only: **no verdicts,
no tests, no Holm**, and no ratio is ever described as "equal", "equivalent" or
"no difference". Per-run quantities follow the timing reporting rule (CLAUDE.md rule 12):
T_round is the test-free round time `rounds[].timing.round_time_s`, and a steady-state
value is a median of T_round over the stated rounds. Output
`analysis/cross_config_timing/`.

* **Statistic.** For a cell and a quantity, per paired seed s,
  r_s = T_MAXN,s / T_het,s. The reported ratio is the geometric mean of r_s over the paired
  seeds, exp(mean_s log r_s), so MAXN / het and het / MAXN are exact reciprocals. Its
  interval is a bootstrap over seeds: the paired seeds are resampled with replacement,
  keeping each seed's pair together, B = 10,000, 95 % percentile interval, generator keyed
  like the other bootstraps (`BASE_SEED + crc32("timing|" + cell + "|" + quantity)`).
  Every cell is reported with its number of paired seeds; a seed enters a cell only when
  both runs exist.
* **Family `rounds_1_3` (cells as in (b)).** IID and label skew x FedAvg and FedProx
  (mu = 0.01), seeds 42/123/456/789/1011. The MAXN run is the ten-round 5b run, and the
  heterogeneous run is the three-round main-matrix run.
  - *Primary:* steady state over rounds 2-3 in BOTH configurations (the median of T_round
    over rounds 2 and 3, i.e. the same round positions).
  - *Secondary:* MAXN steady state over rounds 2-10 against heterogeneous rounds 2-3.
  - *Round 1:* the ratio of the round-1 T_round (the cold start), reported on its own and
    never combined with the steady-state ratios.
* **Family `ten_rounds`.** Label skew x FedAvg and FedBN, seeds 42/123/456: the MAXN
  ten-round run against the heterogeneous ten-round `long_horizon_fedbn` run
  (`results/rev_noniid_<strategy>_r10_seed<S>`).
  - *Steady state:* rounds 2-10 in both configurations.
  - *Round 1:* its own ratio.
* **Straggler.** In each round, the straggler is the client with the largest
  `fit_wall_s`, because the fit phase ends when the slowest client returns. For each
  configuration and cell, the report counts how often each node is the straggler:
  - over the rounds entering the steady state, i.e. rounds 2-3 for the primary of
    `rounds_1_3` and rounds 2-10 otherwise;
  - separately in round 1;
  - with the median fit_wall_s of each node over the same rounds.

  The straggler is reported, never tested.
* **What had been seen when this was declared.** Nothing below was computed. Seen before:
  - the heterogeneous totals in `tab:time`;
  - a rough per-round figure for both configurations (total / R, noted while 5b was
    monitored);
  - the round wall-clock of the four finished IID FedAvg MAXN runs (all ten rounds,
    printed during the clock-jump check of 2026-09-29);
  - the heterogeneous round-1 and rounds-2-3 values of IID FedAvg and IID FedProx in a
    scratch export of `tab:time` the same day.

  No MAXN / heterogeneous ratio, straggler count or bootstrap had been computed.
