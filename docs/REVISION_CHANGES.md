# NCAA-D-26-02211 — Major Revision: Code Changes (branch `revision-ncaa`)

Scope: **hardware-independent** revision work only.  No FL experiment was run,
no result under `results/` was created, modified or removed, and the model
(`src/models/mobilenetv3_multimodal.py`) is untouched.  Every new code path is
covered by CPU unit tests on tiny synthetic data (`tests/`).

Compatibility constraints kept: Python 3.10 syntax, `numpy==1.26.4`,
PyTorch 2.5, Flower 1.13.1, `batch_size=8` defaults for the Jetson Orin Nano
(8 GB).  `results.json` keeps every previously written key.

## Mapping to reviewer comments

| Reviewer request | Change |
|------------------|--------|
| R3: related frames / same-sequence images may be in both train and test; use a sequence- or source-level split | `scripts/analyze_flame_leakage.py` (near-duplicate clustering + leakage audit) and `data_splitter.py --group_file` (whole groups stay on one node and in one split) |
| R3, R5: report balanced accuracy, macro-F1, sensitivity, specificity, precision, recall, MCC, ROC-AUC, per-client confusion matrices, globally and per client | `src/evaluation/metrics.py`, wired into the FL client/server and both baseline scripts; per-client + weighted-global + pooled metrics per round in `results.json` |
| R3: compare methods by elapsed time and communication cost, not only by round | per-round `fit_time_s` / `eval_time_s`, server `elapsed_s`, payload bytes per client and cumulative MB in `results.json`; accuracy-vs-time and accuracy-vs-MB plots in `scripts/analyze_results.py` |
| R3, R5: more seeds, confidence intervals, cautious statistics | 5-seed block in the revision matrix; `analyze_results.py` gives mean ± std, 95 % t-CI, paired Cohen's d, Wilcoxon, paired t, Friedman |
| R3: a more demanding condition (fewer local samples, stronger heterogeneity) | `--subsample_frac 0.05 0.01` (low-data) and `--dirichlet_alpha 0.1 0.5 1.0` (stronger / different label skew) partitions |
| R5: conclusions under different degrees and types of non-IID | Dirichlet sweep block |
| R5: hyperparameter sensitivity (μ, epochs, lr) | μ grid {0.001, 0.01, 0.05, 0.1, 0.5}, local epochs {1, 2, 5}, lr {1e-4, 1e-3} blocks |
| R3, R5: FedBN conclusions limited to 3 rounds | 10-round FedBN (and FedAvg reference) block |

Not addressed here because they need hardware or manuscript edits: scene-independent
(leave-one-scene-out) cross-sensor evaluation, tying sensor heterogeneity into the FL
run, new references / title change / wavelet future-work paragraph.

## File-by-file changes

### New: `scripts/analyze_flame_leakage.py`
* Walks `--data_dir` (Fire / No_Fire class folders, same rule as the splitter).
* Perceptual hash per image: `dhash` (default), `phash` (DCT, NumPy-only) or `ahash`, 64 bit.
  Optional multiprocessing (`--workers`), hash cache (`hashes_<method>8.npz`).
* Near-duplicate clustering at `--threshold` Hamming bits via multi-index hashing
  (pigeonhole over `threshold+1` chunks, exact-hash collapsing, block-wise popcount) and
  SciPy connected components — no O(N²) scan; 50 k hashes cluster in a few seconds.
* Writes `groups.json` (`{"groups": {gid: [relative paths]}}`), `groups.csv`
  (`path,label,group_id,group_size,hash_hex`) and `leakage_report.json` (group-size
  histogram, exact duplicates, cross-label groups, per-class stats).
* `--processed_dir`: audits an existing `data/processed` tree and reports, per split and
  node, the fraction of val/test images with a near-duplicate in any node's train split
  and the number of groups spanning several nodes.
* `load_group_file()` is the shared reader used by the splitter.

### Rewritten: `src/data/data_splitter.py`
* **Default behaviour unchanged**: without new flags the partition is bit-identical to the
  original (same `random` call order and count arithmetic) — pinned by
  `tests/test_splitter.py::test_default_path_identical_to_original`, which imports the
  pre-revision module from `git show main:` and compares full assignments.
* Unit abstraction: a *unit* is a list of paths that must stay together (size 1 without a
  group file; units are built per class, cross-label groups are counted).
* `--group_file PATH` (JSON/CSV): whole groups assigned to one node and one of
  train/val/test, for iid, non_iid_label, dirichlet and subsample splits.
* `--dirichlet_alpha A [A ...]`: per-class `Dirichlet(alpha)` proportions over the nodes
  from `numpy.random.RandomState(seed)`, units assigned by cumulative image count;
  `--dirichlet_min_size` (default 10) redraws until every node has enough images; output
  `dirichlet_<alpha>/`; `split_stats.json` gets `class_counts`, `dirichlet_alpha`,
  `dirichlet_proportions`.
* `--subsample_frac F [F ...]`: after splitting, every base split (iid, non_iid_label,
  each dirichlet) is emitted as `<split>_sub<F>/` with the `--subsample_splits`
  (default `train`) reduced per node, stratified per class, at unit level, seeded with a
  CRC32-stable seed.
* `--skip_base_splits`, `--link_mode {symlink,hardlink,copy}` (symlink falls back to copy
  on Windows/unsupported FS), `manifest.csv` per split, `_meta` block and `class_counts`
  table for every split in `split_stats.json`.

### New: `src/evaluation/metrics.py`
* `compute_metrics(logits, labels)` → accuracy, balanced accuracy, precision, recall
  (sensitivity), specificity, F1, macro-F1/precision/recall/specificity, MCC, ROC-AUC
  (Mann–Whitney with ties), confusion matrix, per-class table, support.  Pure NumPy,
  accepts torch tensors, multi-class capable (macro / OvR).
* `metrics_from_confusion_matrix()` (used server-side for pooled metrics),
  `MetricAccumulator` (batch-wise), `to_flower_metrics()` (flattens to Flower scalars,
  drops undefined metrics so no NaN reaches JSON), `confusion_matrix_from_flat()`.

### Modified: `src/fl/client.py`
* `evaluate()` returns the full metric set (flattened), `eval_time_s`,
  `payload_bytes_down`, `server_round`, `node_name`, `num_examples`.
* `fit()` additionally returns `fit_time_s`, `fit_wall_s`, `payload_bytes_up/down`,
  `server_round`, `node_name`, `data_dir`, `local_epochs`, `lr`, `batch_size`,
  `proximal_mu`, `num_examples` (v2 keys `train_loss`, `train_time`, `hostname`,
  `strategy` kept).
* Constructor: `node_name`, `pretrained`, `img_size`, `eval_split` (defaults = paper
  behaviour; `pretrained=False`/small `img_size` only used by CPU tests).  CLI:
  `--node_name`, `--eval_split {val,test}`.

### Modified: `src/fl/server.py`
* `RoundRecorder`: stateful `fit_metrics_aggregation_fn` / `evaluate_metrics_aggregation_fn`
  that keys rows by the client's `node_name` (fallback hostname), detects the round from the
  echoed `server_round` (fallback call counter), computes the num-examples-weighted global
  aggregate, sums confusion matrices → pooled metrics, mean/max of timing keys, payload
  totals and cumulative communication bytes, and stamps the server elapsed time.
* `on_evaluate_config_fn` passes `server_round`; `fit_metrics_aggregation_fn` is now set
  (also used by FedBN).  `--tag` CLI arg; `model_payload_bytes` computed from the model.
* `results.json`: v2 keys unchanged; adds `results_schema_version: 2`, `proximal_mu`,
  `tags`, `client_config`, `model_payload_bytes`, `total_communication_bytes`,
  `metrics_distributed_fit`, `rounds[]` (per-round `fit`/`evaluate` with `clients`,
  `aggregate`, `elapsed_s`, and `cumulative_communication_bytes`).

### Modified: `scripts/train_local.py`, `scripts/train_centralized.py`
* `evaluate()` returns `(loss, accuracy, metrics)`; history records add `val_metrics`,
  `train_time_s`, `eval_time_s`, `elapsed_s`; results add `final_test_metrics` and
  `results_schema_version: 2`; cross-eval / per-node test add `test_metrics`;
  local batch `summary.json` adds `final_test_metrics` per node.
* `--no_pretrained`, `--img_size` (tests only; defaults = paper), `main(argv)`.

### New: `scripts/analyze_results.py`
* Loads every `results/**/results.json` (old accuracy-only FL files, new schema-2 FL
  files, centralized and local-only runs); derives strategy, μ, distribution, seed,
  rounds, local epochs, lr and node count from the JSON first and the directory name as
  fallback (`--missing_seed` for files without a seed).
* Per-round curves with elapsed time and cumulative MB; for old files both are
  *estimated* (from `total_time_s` and the model payload size) and flagged as such.
* Outputs under `--output_dir` (default `analysis/`, never `results/`): `runs.csv`,
  `summary_table.{csv,md}` (n, mean, std, 95 % t-CI, min, max per config and metric),
  `per_round_table.csv`, `pairwise_tests.{csv,md}` (paired/unpaired Cohen's d, Wilcoxon,
  paired t; pingouin when available, SciPy fallback), `friedman.csv`, and plots
  `accuracy_vs_round_<dist>`, `accuracy_vs_time_<dist>`,
  `accuracy_vs_communication_<dist>`, `final_metrics_<dist>` (PNG + PDF, IEEE style).

### Modified: `configs/experiment_matrix.yaml`; new `scripts/print_revision_commands.py`
* `revision:` block with `seed_extension` (5 seeds FedAvg vs FedProx 0.01 × IID/non-IID),
  `dirichlet_skew`, `low_data`, `long_horizon_fedbn` (10 rounds), `mu_grid`,
  `local_epochs`, `learning_rate`, `baselines_extension`, plus `data_preparation` and
  `analysis` command blocks and run-count totals.
* `print_revision_commands.py` expands the blocks into concrete, resumable server/client
  (or baseline) command lines following
  `results/rev_<dist>_<strategy>[_ep<E>][_lr<LR>][_r<R>]_seed<S>` (`--block`,
  `--format {text,bash}`, `--skip_existing`, `--all_seeds`).  Blocks total 151 new
  runs (163 logical cells; default-valued cells of the μ / epoch / lr sweeps share a
  directory with the seed-extension block and are deduplicated by `--skip_existing`).
* **Comparability caveat**: the seed-extension block reuses the three existing seeds
  by default.  That is only valid while `data/processed` is the original image-level
  partition.  After re-splitting with `--group_file` (leakage-safe protocol) the
  partition changes and all five seeds must be rerun: use `--all_seeds`.

### Other
* `scripts/run_experiment.sh` passes `--tag "$DIST"` and honours `ROUNDS=<n>`.
* `requirements.txt`: `pytest` (tests).  `.gitignore`: hash cache, `*.eml`.
* `README.md`, `docs/REVIEWER_GUIDE.md`, `data/README.md` updated.

## Tests (`python3 -m pytest tests -q`)

| File | What it checks |
|------|----------------|
| `test_metrics.py` | every metric against scikit-learn (binary + multi-class), edge cases, accumulator, Flower flattening round-trip |
| `test_leakage.py` | synthetic near-duplicate images are grouped correctly, hashing/clustering primitives, group files, audit of a processed tree with and without leakage |
| `test_splitter.py` | bit-identity of the default partition with the original code (2 and 3 nodes), group integrity across nodes and splits, Dirichlet determinism / class-count table / min-size, subsampling, stats and manifest contract |
| `test_fl_server_recorder.py` | weighted + pooled aggregation, per-client rows, round detection, v2 keys in `results.json`, strategy wiring |
| `test_client_and_baselines.py` | client `fit`/`evaluate` metrics, payload bytes, FedProx/FedBN paths; `train_local.py` and `train_centralized.py` end-to-end on synthetic data |
| `test_fl_end_to_end.py` | real two-client Flower 1.13.1 run on localhost (FedProx and FedBN), full v3 `results.json` |
| `test_analyze_results.py` | old and new result formats, CI computation, pairwise tests, plots, and a read-only run on the real `results/` tree |
| `test_revision_matrix.py` | matrix contents match the revision spec; command expansion and directory naming |

## Notes for the authors
* The two files mentioned as "provided" (`scripts/analyze_flame_leakage.py`, updated
  `data_splitter.py`) were not found in the working directory or the repository; both were
  implemented from the description above.  Drop-in replacements will be picked up by the
  same tests.
* Existing `data/processed` trees produced by the v1 splitter are *not* group-safe; rerun
  the splitter with `--group_file` before the revision experiments.
* Old result files carry no timing/communication per round; the analysis script marks
  those curves as estimated.  New runs record them exactly.
