# FedRGBD — project instructions

Federated learning on a 3-node Jetson Orin Nano 8 GB cluster with heterogeneous RGB-D cameras.
Manuscript **NCAA-D-26-02211**, *Neural Computing and Applications*, **major revision due
2026-11-13**. All revision work happens on the branch `revision-ncaa`.

## Read these first

| File | What it is |
|---|---|
| `docs/REVISION_PLAN_TR.md` | The working plan (Turkish). §5 status, §5.1/§5.2 findings, **§6 the Jetson handoff: run order, time estimates, which table each block fills** |
| `docs/REVISION_CHANGES.md` | File-by-file record of every change made for the revision (English) |
| `docs/RESPONSE_TO_REVIEWERS.md` | Per-reviewer-comment answers; says what is DONE and what is PENDING EXPERIMENT |
| `docs/DESKTOP_GPU_BASELINES.md` | Desktop GPU environment and the baseline block that already ran |

## Environment

- **Jetson**: `./setup_jetson.sh` (JetPack 6.x, NVIDIA PyTorch wheel, Flower 1.13.1). Python 3.10
  syntax, `numpy==1.26.4`, `batch_size=8` — these are hardware constraints, do not "modernise" them.
- **Desktop (Windows, RTX 5090)**: venv `~/venvs/fedrgbd` (CPU, tests) and `~/venvs/fedrgbd-gpu`
  (CUDA 12.8, baselines). System Python and WSL have no torch.
- **Testbed network** (wired Gigabit Ethernet, static): node_a = `192.168.1.10` (also the Flower
  server, `:8080`), node_b = `192.168.1.7`, node_c = `192.168.1.6`.
- **FLAME layout**: the Kaggle archive unpacks as `Training/Training/{Fire,No_Fire}` and
  `Test/Test/{Fire,No_Fire}`, but the `data/splits` manifests expect a flat `{Fire,No_Fire}`
  layout. On a (re)built node, move all files into `data/raw/flame_dataset/Fire` and
  `data/raw/flame_dataset/No_Fire` (30155 + 17837) before running `--from_manifest`, otherwise the
  replay aborts with missing sources.
- **GUI off on all three nodes** (`sudo systemctl set-default multi-user.target`): the desktop
  session costs ~2.5 GB of the 8 GB, and the nodes must be comparable because per-round wall-clock
  is a reported result. As of 2026-09-19 all three nodes run with GUI off and have ~6.8 GB of the
  8 GB available, so they are comparable for timing.
- **v1 ran over WiFi (802.11ac), the revision over wired GbE.** v1 and revision wall-clock numbers
  are not comparable; revision timings supersede the v1 ones.
- **Sudo scope on the testbed (granted 2026-09-28).** The assistant may run exactly two
  privileged commands, and only on node_c: `sudo -n nvpmodel -m 2` (MAXN_SUPER) and
  `sudo -n nvpmodel -m 3` (7W), via `/etc/sudoers.d/nvpmodel`, and never while any
  run_matrix or FL process is alive on any node. Switching reboots node_c; afterwards verify
  `nvpmodel -q` on all three nodes and that WiFi is still disabled. No other sudo, ever --
  every other power-mode change, reboot or system change is the user's.
- **Power mode changes the numbers, not only the timing.** Training is bitwise
  deterministic within a fixed power configuration, but node_c's 7W mode (2 of 4 GPU TPCs,
  4 CPU cores) trains differently from MAXN_SUPER; clock changes alone (node_a 15W vs MAXN)
  do not (`docs/CROSS_CONFIG_COMPARISON.md`, `analysis/determinism/`).
- Tests: `python -m pytest tests -q -k "not end_to_end"` (~250 CPU tests, seconds).
  On the Jetson nodes the ROS 2 pytest plugins break collection; use
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q -k "not end_to_end" -p no:cacheprovider`.

## Hard rules

1. **Never commit `*.eml`.** The reviewer/editor correspondence is confidential and this repo is public.
2. **Never re-derive the data partition.** `find_images()` depends on `os.walk` order, so two
   machines can produce *different* splits. The authoritative partition is committed in
   `data/splits/` and is replayed:
   ```bash
   python src/data/data_splitter.py --from_manifest data/splits \
       --data_dir data/raw/flame_dataset --output_dir data/processed \
       --link_mode hardlink --clean --verify
   ```
   Then check the manifest MD5s against the table in `analysis/leakage/P0_SUMMARY.md`; they must
   be identical on all three nodes.
3. **Always pass `--all_seeds`** to `scripts/print_revision_commands.py`. Runs from before the
   leakage-safe re-split used a different partition and are not comparable.
4. **Never type a number into the paper by hand.** Run `scripts/analyze_results.py` then
   `scripts/export_latex_tables.py`; the tables come from `analysis/`.
5. **Do not touch `results/3node_*`, `results/centralized_*`, `results/local_*`** — those are the
   v1 (image-level split) results kept as the published reference. New runs go to `results/rev_*`.
6. The default (no-flag) code path of `data_splitter.py` is pinned bit-identical to the original
   submission by a regression test. Changes must stay behind a flag.
7. **The clean-subset rule is pre-registered and frozen.** Fixed 2026-09-19, before any
   clean-subset metric was computed (full-set baseline results and one FL run existed then):
   a val/test image is excluded iff a training image of the same partition (all nodes) is within
   dHash ≤ 12 or pHash ≤ 10 (`scripts/clean_subset.py`, lists in
   `analysis/leakage/clean_subset/`, `RULE.json`). Headline metrics are reported on all held-out
   images and on the clean subset. Never change the thresholds, hashes or lists after seeing
   results; `python scripts/clean_subset.py --check` must report "identical". If the rule ever
   has to change, disclose it in the paper and the response letter.
8. **Balanced accuracy is the declared primary metric; accuracy is secondary.** Fixed
   2026-09-22, after the baseline references existed but **before any federated run of the
   revision** (say it that way — it is not a pre-registration like rule 7, and the paper states
   the difference in §"Declared Primary Metric"). For every partition and method: balanced
   accuracy leads, MCC is the second summary statistic, accuracy is reported alongside but
   never ranks methods on its own. **Why (decided 2026-09-28, general grounds):** FLAME is
   imbalanced (62.8 % fire), the per-node class distributions of the non-IID partitions are
   strongly skewed and each node's test split inherits them, and balanced accuracy is the
   standard metric for imbalanced classification -- accuracy rewards majority-class
   prediction, the very failure federation should fix. The metric is NOT switched now that
   results exist (switching after seeing results is what the rule prevents); only the
   rationale changed. The label-skew accuracy inversion below prompted the rule on 22 Sep
   but is **no longer its justification** -- under the pooled aggregation it disappears.
   The paper's rationale passage carries a `\todo` for the narrative stage. The original illustration --
   local-only beats centralized on accuracy (95.5 vs 94.3 %) and loses on balanced accuracy
   (88.6 vs 94.7 %) -- mixed two aggregations (88.6 is a client mean, 94.7 pooled): under the
   pooled aggregation below it is 95.3 vs 94.7, under the client mean 88.6 vs 93.8. The
   paper's narrative about it is pending (`\todo`s). Column order lives in
   `FULL_METRIC_ORDER` / `DEFAULT_METRICS` in `scripts/export_latex_tables.py`; do not reorder
   accuracy back to the front.
   **Aggregation over the three nodes, one definition for every row.** Fixed 2026-09-27,
   AFTER all 98 federated runs existed and after an analysis of them showed that several
   comparisons depend on it -- NOT pre-registered, which is why both are always reported:
   *primary* = pooled over the union of the held-out images (FL: the global model on every
   client's test split; local-only: each node's own model on its own split; centralized: its
   pooled test set -- `check_prediction_unions` verifies it is image for image the same set),
   `selected_test_<m>` in `analysis/`; *secondary* = unweighted mean over the clients,
   `selected_test_clientmean_<m>`. Both carry the stratified cluster bootstrap (sequences
   resampled across nodes for the pooled figure, within each client for the mean).
   Comparisons whose conclusion flips between them: `scripts/aggregation_sensitivity.py` ->
   `analysis/aggregation_flips.md`. Never report the test-size-weighted client mean that the
   FL runs log themselves (kept only as `selected_test_metrics_logged`).

9. **Global evaluation is a pre-registered second perspective** (declared 2026-09-28,
   `docs/GLOBAL_EVALUATION.md`): every model on the union of the three nodes' test splits
   -- local-only as the mean over the three node models run on the whole union -- with a
   seed-paired FL - local-only difference, cluster bootstrap, clean subset as robustness,
   a fixed three-way verdict rule and Holm within each family. The personalised pooled
   figure of rule 8 stays primary. Definition, code and rule were committed and pushed
   before any local-only global value was aggregated or compared -- NOT "before its
   numbers existed": the per-node cross-evaluations were logged automatically during
   baseline training (2026-09-19/20) and the FL/centralized global values are the known
   pooled figures; use the doc's Provenance wording. Never change them afterwards, never
   add or drop a comparison, never paraphrase the verdict phrases. `scripts/global_evaluation.py`.

10. **MAXN block gate and cross-configuration comparison are declared**
   (2026-09-28, `docs/CROSS_CONFIG_COMPARISON.md`, committed before 5b relaunched). Hard
   gate: the two MAXN smoke runs named in `determinism_gate` must be bitwise identical.
   MAXN vs heterogeneous is a declared analysis in two Holm families -- rounds 1-3
   (FedAvg/FedProx x IID/label skew) and all ten rounds (label skew x FedAvg/FedBN, seeds
   42/123/456, vs long_horizon_fedbn; amendment committed before 5b launched) -- `scripts/cross_config_comparison.py`, reported whatever it shows, never a
   gate, never "equivalent". Do not change it after the first 5b run finishes.

## Model selection (declared rule — use this wording, do not paraphrase it)

The reported model is the one from the round (baselines: epoch) with the lowest validation
loss, aggregated across clients weighted by client validation-set size; ties are broken
toward the earlier round. Test metrics are computed every round for logging but never
influence model selection, which uses the aggregated validation loss only; the reported test
metrics are those of the selected round. Never report a maximum over rounds, and never write
that the test set is "used once": it is evaluated every round. Implementation:
`src/evaluation/model_selection.py`.

## Why the protocol changed

A near-duplicate audit of FLAME found that **99.6 % of validation/test images had a near-duplicate
in some node's training split** under the original random image-level split: the dataset is video
frames, and 265 groups cover 99.7 % of the 47,992 images. The revision therefore assigns whole
sequences (groups) to one node and one of train/val/test. 22 groups contain both fire and no-fire
frames (42 % of the data) and are kept together regardless of label. Under the audited protocol the
references drop from ~99.6 % to **93.4 % (centralized IID) and 86.9 % (local-only IID)** — a
6.5-point band, which is what makes the reviewers' question — when does federation help —
measurable at all.

These two numbers, and every other reference number in the paper, come from
`analysis/summary_table.csv` (protocol `group`), written by `scripts/analyze_results.py`; never
re-type them from here, and re-read them after any run. The same runs reported at their **final
epoch** instead (protocol `group_final_epoch`, kept for comparison) give 90.9 % and 78.3 %, a
12.6-point band: final-epoch reporting more than doubles the apparent advantage of pooling,
because the local-only models are the unstable ones across epochs. Report the `group` numbers;
cite `group_final_epoch` only when the point *is* the reporting protocol.

## Current state (2026-09-17)

Everything that does not need the testbed is done: leakage audit, re-split, all 62
centralized/local-only baseline runs, bibliography, paper text, response letter.
**Remaining: 98 federated runs on the Jetsons (~220 h, see §6.2 of the plan)**, plus three
author-only items (funding wording, testbed photo, scene labels for the cross-sensor experiment).
Paper placeholders awaiting those runs are red `\todo{...}` / `\PHs` macros in `paper/main.tex`.

## Running a block

```bash
python scripts/print_revision_commands.py --all_seeds --block seed_extension --format bash > run.sh
# server on node A, clients on all three; finished runs are skipped automatically
bash run.sh
python scripts/analyze_results.py --results_dir results --output_dir analysis
python scripts/export_latex_tables.py --analysis_dir analysis --output_dir paper/tables
```

On the testbed use `scripts/run_matrix.py --block <b> --power_config <heterogeneous|maxn>`
(required). After a reboot of node_a, a crontab `@reboot` entry runs
`scripts/resume_after_reboot.py`, which restarts a block only if `logs/run_matrix.log` shows
it was interrupted (no recorded end); every decision is in `logs/resume.log`. The main matrix ran with unharmonised power modes (node_a 15W, node_b MAXN_SUPER,
node_c 7W) and is `heterogeneous`, `results/rev_*`; any other configuration writes to
`results/pc_<name>/rev_*` and is never pooled with it (`docs/REVISION_CHANGES.md`, "Power
configurations").

Block order and cost: `docs/REVISION_PLAN_TR.md` §6.2. After each block, update the matching
table listed in §6.3 and commit.
