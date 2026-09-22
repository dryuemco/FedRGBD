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

| R3: scene-independent cross-sensor evaluation | `scripts/cross_sensor_loso.py` + `src/data/custom_dataset.py` (leave-one-scene-out, with a pooled random-split baseline for comparison) |
| R1, R5: title, scope statements, equation citations, 2025-26 references, metaheuristic HPO paragraph, wavelet future work | `paper/main.tex` revised draft (all new numbers are red placeholders) and `docs/RESPONSE_TO_REVIEWERS.md` |

Still requiring hardware or author input: every new experiment (see the revision matrix),
scene labels for the custom captures (`labels.csv`), tying sensor heterogeneity
directly into an FL run, and downloading the raw FLAME data to the desktop (Kaggle
credentials) so that the P0 leakage/re-split chain can run (see the 2026-09-17 section).

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
* **Merged from the authors' own script** (see "Notes for the authors"):
  `--hash both` (clusters the *union* of the dHash edges at `--threshold` and the pHash
  edges at `--phash_threshold`, default 10; both hash arrays are cached separately, and
  the union grouping always contains the single-hash grouping);
  `--sweep 4 6 8 10 12` (threshold-sensitivity table with n_groups /
  n_nontrivial_groups / images_in_nontrivial_groups / largest_group /
  fraction_with_near_duplicate per threshold in
  `leakage_report.json["threshold_sweep"]`, printed as a table, without changing which
  threshold writes `groups.json`);
  `exact_duplicate_files_md5` (byte-identical duplicates, computed in the *same* pass as
  the perceptual hashes so the ~48 k files are read once, cached in `md5s.npz`,
  `--no_md5` to skip);
  `--sequence_heuristic` (the authors' filename frame-number heuristic, extended into a
  statement the paper can use: what share of pairs of consecutive frame numbers ends up
  in the same near-duplicate group, globally and per class);
  `--examples N` -> `example_groups.txt` with the N largest groups and their members;
  `unreadable_files.txt` for files that could not be hashed;
  and `load_group_file()` now also reads the authors' inverted layouts
  (`{"groups": {path: gid}}` and a bare `{path: gid}` mapping, detected by value type),
  relativising absolute paths against `--data_dir` and otherwise falling back to
  `<ClassDir>/<basename>`.

### Rewritten: `src/data/data_splitter.py`
* **Default behaviour unchanged**: without new flags the partition is bit-identical to the
  original (same `random` call order and count arithmetic) — pinned by
  `tests/test_splitter.py::test_default_path_identical_to_original`, which imports the
  pre-revision module from `git show main:` and compares full assignments.
* **Group mode with oversized units (2026-09-17, after the first real FLAME run).** The real
  group file makes whole video sequences single units: 47,863 of 47,992 images fall into
  265 groups, the largest unit holds 4,341 images of one class.  The first re-split on the
  real data therefore left node_b of the IID split with **zero** validation images, gave a
  Dirichlet(0.1) node 12 images, and turned the "5 %" low-data subsample into 69 % on one
  node (one unit taken = one whole sequence).  Three changes, all gated on
  `is_grouped()` (at least one unit of size > 1), so the default path stays bit-identical:
  1. **largest-first greedy assignment** (`assign_units_greedy`, LPT rule = the authors'
     own "largest remaining deficit" rule with a size-descending order) fills the *same*
     image quotas the image-level arithmetic produces — equal thirds for IID
     (`_equal_quotas`), the original 80/–/20 label-skew numbers (`label_skew_quotas`,
     ~88.5 % fire on node_b), `p × N` for Dirichlet, 70/15/15 per node and class — with whole
     units.  Every bucket deviates from its quota by at most the largest unit it received;
     the achieved counts are in `split_stats.json` and must be quoted in the paper.
  2. **image-level subsampling inside the node's own units** for `--subsample_frac` in
     group mode: exact fraction, survivors keep their `group_id`, dropped images go
     nowhere, so no group ever spans two nodes or two of train/val/test.  The low-data
     regime therefore reduces *frames per client*, not sequences (say so in the paper).
  3. **`--verify` writes the fresh `split_stats.json` before checking**: the verifier
     compares manifests against the stats file on disk, so with a stale file from the
     previous partition (the normal `--clean` re-split case) it reported a spurious
     `VERIFY FAIL` on every count.  `_meta` now also records `grouped`, `assignment`
     (`greedy_largest_first` / `sequential_cut`) and `subsample_level`.
  4. **Cross-label groups are one bundle.**  The second real run passed `--verify` but the
     independent audit still found up to 72 % of a node's test images leaking: 22 groups
     (20,006 images, the largest 4,341 Fire + 583 No_Fire) contain frames of both labels — a
     FLAME sequence is a video in which the fire appears and disappears — and the splitter
     treated the Fire part and the No_Fire part as independent units.  `bundle_units()` pairs
     the two per-class units of a group and `assign_bundles_greedy()` places the bundle as a
     whole (composition-weighted deficit over the per-class quotas; identical to the unit
     rule for pure groups); nodes, train/val/test and Dirichlet all use it in group mode.
     Both verifiers (`scripts/verify_splits.py::check_manifest` and the splitter's fallback)
     now also check group integrity **ignoring the label**, so this class of leak can no
     longer pass `--verify`.
  Tests: `tests/test_splitter_grouped.py` (FLAME-shaped synthetic groups: no empty bucket,
  quota arithmetic, exact subsample, stale-stats verify) and `tests/test_splitter_bundles.py`
  (cross-label bundles stay on one node and in one bucket; label-split group rejected by both
  verifiers), plus the tightened subsample assertion in `tests/test_splitter.py`.
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
* `--clean`: deletes each `<output_dir>/<split>/` tree before rewriting it.  **Required
  when re-splitting into an existing `data/processed`** — the splitter only overwrites
  files with the same basename, so without it images that move to another node or to
  another train/val/test bucket are left behind by the previous partition and the tree
  holds both at once (train images silently leak into val/test).  Without `--clean` a
  non-empty target directory produces a loud warning.  Every re-split command in
  `README.md`, `data/README.md`, `docs/REVIEWER_GUIDE.md`, `docs/REVISION_PLAN_TR.md` and
  `configs/experiment_matrix.yaml:revision.data_preparation` now passes it.
* **Merged from the authors' own splitter**: `--verify` (their `verify_no_leak()` as a
  post-write self-check) re-reads the manifests and asserts that no `(group_id, label)`
  unit spans nodes or train/val/test and that the manifest counts equal `split_stats.json`;
  it reuses `scripts/verify_splits.py` when importable (so it also catches duplicated
  rows, images shared by two nodes and basename collisions) and falls back to a local
  minimal check otherwise, prints `VERIFY PASS`/`VERIFY FAIL`, records `_meta.verify_ok`
  and exits non-zero on FAIL.  Their per-node `train/val/test=a/b/c` summary line was
  also adopted, and `--group_file` accepts their inverted group-file layout.  Their
  *part-first* group partitioning (train/val/test split at group level first, nodes
  inside each part) was deliberately **not** adopted: it changes the partition design and
  would break the bit-identical default path pinned by `tests/test_splitter.py`.

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
| `test_leakage.py` | synthetic near-duplicate images are grouped correctly, hashing/clustering primitives, group files, audit of a processed tree with and without leakage, `--hash both` (union groups contain the dHash groups, both caches written), `--sweep` (monotone, leaves `groups.json` untouched), MD5 exact duplicates, the filename frame-number heuristic, `example_groups.txt`, and the inverted group-file layout (equal to the standard layout, relative and absolute paths) |
| `test_splitter.py` | bit-identity of the default partition with the original code (2 and 3 nodes), group integrity across nodes and splits, Dirichlet determinism / class-count table / min-size, subsampling, stats and manifest contract, `--clean`, `--verify` (PASS on a fresh split, FAIL + exit 1 on a group spanning nodes / train+test / a stats mismatch) and the inverted group-file layout producing the same partition |
| `test_fl_server_recorder.py` | weighted + pooled aggregation, per-client rows, round detection, v2 keys in `results.json`, strategy wiring |
| `test_client_and_baselines.py` | client `fit`/`evaluate` metrics, payload bytes, FedProx/FedBN paths; `train_local.py` and `train_centralized.py` end-to-end on synthetic data |
| `test_fl_end_to_end.py` | real two-client Flower 1.13.1 run on localhost (FedProx and FedBN), full v3 `results.json` |
| `test_analyze_results.py` | old and new result formats, CI computation, pairwise tests, plots, and a read-only run on the real `results/` tree |
| `test_revision_matrix.py` | matrix contents match the revision spec; command expansion and directory naming |
| `test_run_p0_pipeline.py` | the one-command P0 driver end-to-end on synthetic data: leaky v1 split detected, group-safe re-split, zero leakage after, resumable skips, `--dry_run`, missing-data exit code 2, dataset flattening |

## Notes for the authors
* The two files mentioned as "provided" (`scripts/analyze_flame_leakage.py`, updated
  `data_splitter.py`) **were received and have been merged**.  The repository versions
  stay the base (they are supersets: multi-index hashing instead of the O(N²) pairwise
  scan, unit-level Dirichlet/subsample partitions, manifests, the bit-identical default
  path) and every extra of the authors' files that adds information was ported into them:
  `--hash both`, `--sweep`, the MD5 exact-duplicate count, the filename frame-number
  heuristic (`--sequence_heuristic`), `example_groups.txt` (`--examples`), the inverted
  `{path: gid}` group-file layout, the authors' `verify_no_leak()` as
  `data_splitter.py --verify`, and their per-node `train/val/test` summary line.
  Deliberately not adopted: the authors' *part-first* group split (train/val/test chosen
  before the nodes) and their greedy "largest remaining deficit" group assignment, because
  they are a different partition design — the default path must stay bit-identical to the
  submitted one (`tests/test_splitter.py::test_default_path_identical_to_original`), and
  the group-level path deliberately reuses the original label-skew arithmetic.  Their
  `realpath`-based group-file keys were replaced by `--data_dir`-relative keys (the same
  information, but portable across machines and symlinked dataset roots).
* Existing `data/processed` trees produced by the v1 splitter are *not* group-safe; rerun
  the splitter with `--group_file` before the revision experiments.
* Old result files carry no timing/communication per round; the analysis script marks
  those curves as estimated.  New runs record them exactly.

## Leakage sensitivity, pre-registered clean subset, cluster bootstrap, per-image predictions (2026-09-19)

* **Partition kept** (dHash τ = 8). Coarser groupings (dHash ≤ 8 ∪ pHash ≤ 3/4/6) were built and
  compared in scratch: they shrink the residual near-neighbour band but never close it, and they
  worsen balance and single-sequence dominance.
* **Pre-registration** (commit 873b389, pushed before any clean-subset metric existed):
  `scripts/clean_subset.py`, `analysis/leakage/phash.csv.gz`, `analysis/leakage/clean_subset/`
  (`RULE.json`, excluded lists for all 15 partitions, `nearest_train_distance.csv/.tex`). Rule:
  exclude held-out images within dHash ≤ 12 or pHash ≤ 10 of a training image of the same
  partition. CLAUDE.md hard rule 7 freezes it. `--check` reproduces the files byte for byte.
* **Per-image predictions** (`src/evaluation/predictions.py`): the FL client packs
  `path, label, logit_margin, p_fire` for val and test after the test pass (timer
  `pred_pack_time_s`, excluded from the reported round time together with the server's file
  writes); the server strips the `pred_<split>_npz` payloads before aggregation and writes
  `results/<run>/predictions/r<round>_<node>_<split>.npz` + README (`*.npz` gitignored).
  `scripts/predict_from_checkpoint.py` regenerates baseline predictions from `model_selected.pt`
  and checks them against the logged metrics.
* **Cluster bootstrap** (`src/evaluation/bootstrap.py`): hierarchical over seeds and held-out
  sequences (group_id), B = 1000, vectorised (exact against materialised resamples).
  `analyze_results.py` uses it for the selected-test metrics of every configuration whose runs
  all have predictions (`ci_method` column), adds `selected_test_clean_<m>`, and checks that the
  predictions reproduce the logged metrics (`pred_check_max_diff`). Other CIs stay the t-interval
  over seeds. `scripts/heldout_dominance.py` writes `analysis/leakage/heldout_dominance.csv`.
* **Bug found by the new tests:** the first server version stripped every `pred_*` key, including
  the float timer `pred_pack_time_s`, which would have left packing time inside the reported round
  time; only the `_npz` payloads are stripped now.
* Paper: leakage guarantee stated as relative to its definition, `tab:nn_distance`, new
  subsection "Pre-Registered Clean-Subset Analysis", bootstrap CIs in the statistics section,
  ninth limitation (node C). Response letter: note on the leakage guarantee with the numbers.

## Model-selection protocol (2026-09-19, before the 98-run matrix)

Declared rule (paper §III, "Model Selection and Use of the Test Split"): the reported
model is the one from the round with the lowest validation loss, aggregated across
clients weighted by client validation-set size; ties go to the earlier round. Test
metrics are computed every round for logging but never influence model selection, which
uses the aggregated validation loss only; the reported test metrics are those of the
selected round.  Previously `analyze_results.py` reported `best_accuracy` as the maximum
validation accuracy over rounds, i.e. a post-hoc choice of round by the reported metric.

* New `src/evaluation/model_selection.py`: `weighted_val_loss()`, `select_round()` --
  the single implementation used by the server and the analysis.  A round in which any
  client's validation loss is non-finite is not eligible.
* `src/fl/client.py`: `evaluate()` scores the validation split, then the test split.
  Returned loss / `num_examples` and the unprefixed metric keys are validation (as
  before); new `val_*` and `test_*` metric sets incl. `*_n_examples` and confusion
  counts; timers `val_eval_time_s`, `test_eval_time_s`, `eval_time_s` (loading +
  validation, excludes the test pass) and `eval_wall_s`.  `--eval_split` removed.
* `src/fl/server.py` (results schema 3): per-namespace aggregation (`test_*` weighted by
  test-set size, `pooled_val_*`, `pooled_test_*`, `*_n_examples_total`), nested
  `val_/test_confusion_matrix` per client, per-round `weighted_val_loss` and `timing`
  (`round_time_s` = round wall-clock minus the test pass's critical-path share
  `max_k(eval_wall_s) - max_k(eval_wall_s - test_eval_time_s)`), `elapsed_excl_test_s`,
  and top-level `model_selection`, `total_time_excl_test_s`,
  `total_test_eval_overhead_s`, `timing_definition`.  All v2/v3 keys kept.
* `scripts/analyze_results.py`: `best_accuracy` removed.  `select_fl_round()` recomputes
  the selection from per-client validation losses and then reads that round's test
  metrics.  Headline metric families never share a column: `selected_test_<m>`
  (revision FL), `final_<m>` (baselines, final-epoch test), `v1_final_round_accuracy`
  (FL runs without per-round test metrics).  Curves and `metrics_final` never contain
  test keys.  `total_time_s` of schema-3 runs is the test-free time
  (`total_time_raw_s` keeps the raw one).  New `per_client_selected.csv`.
  **Bug fix:** pairwise tests and the Friedman test were keyed by distribution only, so
  v1 and revision runs with the same strategy, distribution and seed were averaged
  together (the committed `analysis/pairwise_tests.csv` / `friedman.csv` are affected;
  the paper's `tab:stats` predates the revision baselines and is not).  They are now keyed
  by (protocol, distribution); the v1 values of `tab:stats` and the Friedman sentence are reproduced exactly.
* `scripts/export_latex_tables.py`: full-metric tables use `selected_test_<m>` for FL
  rows; `time.tex` gains the selected-round test accuracy; pairwise / Friedman tables are
  grouped by protocol.
* Paper: new §III subsection with the rule; evaluation and profiling text; captions of
  the revision tables ("Test ... at the Selected Round"); the `tab:v1_ci` footnote labels the v1
  federated rows as final-round validation accuracy; limitations updated.
* Tests: `tests/test_model_selection.py`, `tests/test_fl_server_selection.py`,
  `tests/test_analyze_selection.py`, new client tests; `tests/test_fl_end_to_end.py`
  checks schema 3 over a real Flower run.

## Hardware-independent follow-up (2026-09-17)

Everything below was done on the desktop without the Jetson testbed; 227 CPU tests pass
(`python -m pytest tests -q -k "not end_to_end"`), `latexmk` builds `paper/main.tex` with no errors.

* **Bibliography verified.** All 15 `% VERIFY` entries in `paper/main.tex` were checked against
  Crossref / arXiv; three were upgraded from preprint to the published record (Banerjee et al. ->
  Euro-Par 2025, LNCS 15900, pp. 264-278; Zhang et al. -> IEEE ICASSP 2025; Borazjani et al. ->
  IEEE Trans. Artif. Intell. 7(9), 2026). The R1.2 list in `docs/RESPONSE_TO_REVIEWERS.md` was
  aligned. No entry failed verification.
* **Paper stubs removed.** The empty v1 subsections *Modality Ablation*, *Resource Profiling* and
  *Network Constraint Sensitivity* (not requested by any reviewer, not runnable in this revision)
  were deleted; the introduction/related-work sentences promising energy measurements were
  reworded to wall-clock/communication; a Limitations paragraph and a Future Work sentence
  declare the three measurements out of scope. The response letter (R3.1 and the pending-
  experiments table) was updated accordingly.
* **v1 numbers cross-checked.** `scripts/analyze_results.py` -> `analysis/` and
  `scripts/export_latex_tables.py` -> `paper/tables/` were run on the existing `results/`.
  `tab:v1_ci`, `tab:time` and `tab:stats` match the regenerated values exactly; one prose count
  in the statistics section was wrong (two, not three, IID pairs reach p<0.05) and was fixed.
* **New: `scripts/run_p0_leakage_and_split.py`.** One resumable command for plan sections 2.1-2.4:
  optional Kaggle download + layout flattening, audit of the existing `data/processed`, raw-data
  audit (`--sweep`, `--sequence_heuristic`, `--examples`, MD5), a decision summary
  (`analysis/leakage/P0_SUMMARY.md`, RE-SPLIT REQUIRED / negligible), group-safe re-split with
  `--clean --verify`, post-split zero-leakage assertion and manifest MD5 listing. `--dry_run`,
  `--force`, exit code 2 when the dataset is missing. Documented in `README.md` and `data/README.md`.
* **New: desktop GPU environment** for the `baselines_extension` block: the `fedrgbd-gpu` venv
  under the user's `venvs` folder (torch 2.11.0+cu128, torchvision 0.26, numpy 2.5; deviates
  from the Jetson pins because the RTX 5090 needs CUDA 12.8 / sm_120). `setup_desktop_windows.ps1`
  reproduces it; `docs/DESKTOP_GPU_BASELINES.md` records versions, verification and the exact run
  sequence (`PYTHONUTF8=1` is required on Windows because the training scripts print Unicode arrows).
  `train_centralized.py` / `train_local.py` now record `"device"` in their results JSON.
* **P0 executed on the real FLAME data (same day).**  The author downloaded the Kaggle archive
  (47,992 JPEGs, Training/Test layout flattened to `Fire/` + `No_Fire/`); the v1 image-level
  split was regenerated with seed 42 (per-node counts identical to the paper) and audited, then
  the group-safe re-split was produced with `--link_mode hardlink --dirichlet_min_size 200`.
  Measured: 265 non-trivial groups cover 99.73 % of the images (largest 4,924); under the v1
  split 99.6 % of val/test images have a near-duplicate in some node's train split and
  235 / 219 groups span nodes; under the group-level protocol all 15 partitions audit at
  L = 0 with no group spanning two nodes.  `paper/main.tex` Table `tab:leakage` (dataset
  statistics, threshold sweep, both protocols) and the new Table `tab:group_counts` (achieved
  per-node counts) are filled with these numbers; Sections III-D/E/F describe the
  label-agnostic groups, the largest-first assignment, `n_min = 200` and the frame-level
  low-data subsample; the response letter R3.2 reports the audit.  Outputs are committed under
  `analysis/leakage/` (`P0_SUMMARY.md`, `leakage_report.json`, `groups.{json,csv}`, the
  `v1_audit/` and `post_split_audit/` reports); hash caches are git-ignored.
* **Baseline block executed on the desktop GPU (same day, 12:15-20:23).**  62 runs under the
  group-level split: centralized and local-only for IID and label skew (5 seeds each), the three
  Dirichlet partitions and the four low-data partitions (3 seeds each), four parallel lanes,
  ~46 min per full run.  `results/rev_*_{centralized,local}_seed*` are committed.  Findings:
  centralized 90.9 % (IID) / 94.2 % (skew) and local-only 78.3 % / 94.1 % test accuracy (v1:
  99.5-99.7 %); local-only balanced accuracy 58.7 % at Dirichlet 0.1; the low-data regime costs
  the local model little under IID because the frame-level subsample keeps most sequences.
  `analyze_results.py` gained a `protocol` dimension (`{group}` / `{image}`) and node-averaged
  full metrics for local-only batches; `print_revision_commands.py --all_seeds` also covers the
  baseline block and treats `summary.json` as a finished run.  Paper: `tab:protocol_effect`,
  `tab:fullmetrics`, `tab:dirichlet`, `tab:lowdata` baseline rows and the IV-C/E/F prose are
  filled; a Limitations paragraph states that group-level results are conditional on the
  held-out sequences of the partition seed.  Response letter: paragraph after R3.4.
* **The partition is now shipped, not re-derived** (`data_splitter.py --export_manifests` /
  `--from_manifest`, `data/splits/`).  `find_images()` discovers images with `os.walk`, whose
  order is not guaranteed across machines, so running the splitter independently on each Jetson
  could produce *different* partitions -- breaking the federated protocol and the leakage
  guarantee at once.  `--export_manifests DIR` writes one gzipped `manifest.csv` per split plus
  `split_stats.json` (1.1 MB for all 15 FLAME splits, tracked in git, `mtime=0` so re-exports
  are byte-stable); `--from_manifest DIR` rebuilds the tree from it with no RNG at all and
  reproduces every manifest MD5 in `analysis/leakage/P0_SUMMARY.md`.  Verified against the real
  archive and covered by `tests/test_splitter_manifest_replay.py` (10 tests, including a
  reversed-`os.walk` run that still reproduces the digests).
* **`CLAUDE.md`** added at the repo root so that an agent starting on a Jetson has the
  conventions, the hard rules (never commit `*.eml`, never re-derive the partition, always
  `--all_seeds`, never hand-type a number into the paper) and the run recipe in front of it.
* Not done (needs the author): funding / AI-disclosure wording; testbed photo; scene labels for
  the cross-sensor experiment.  Everything else that remains is a Jetson run.

### RNG streams and the evaluation passes (verified 2026-09-19)

Iterating a `DataLoader` that has no generator of its own draws one number from the
global torch generator per pass, even with `shuffle=False` (checked in torch 2.5.1 and
2.11); the training loaders use their own seeded generator and draw nothing from it.
MobileNetV3-Small's classifier has a `Dropout` layer, which does draw from the global
generator, so every such pass shifts the dropout masks of later training.

* **Validation pass: deliberately not guarded.** The v1 client (`main:src/fl/client.py`)
  and the schema-2 baselines already iterated the validation loader every round/epoch,
  so validation shifted the dropout stream in v1 exactly as it does now: one draw per
  pass, independent of the data and of the split size. Keeping it unguarded keeps the
  revision's RNG consumption identical to v1's; it is **not** an additional difference
  between v1 and revision runs.
* **Test pass: guarded** (`report_only()`, commit 94e0349). Without the guard, the new
  per-round/per-epoch test pass would have added one draw per pass and made training
  differ from a run without it. The unguarded version (c98e34b) never produced
  experiment results: no FL run and no baseline was run with it.

## Reference conditions move to the declared selection rule (2026-09-21)

The 62 centralized / local-only baselines were re-run so that they follow the model-selection
rule of `src/evaluation/model_selection.py` (commit 94e0349); the results landed in
`results/rev_baselines_sel/` and had never been folded into `analysis/`. `analysis/runs.csv`
still held only the older final-epoch runs, so every reference number in the paper was a
final-epoch number.

* **`analysis/` regenerated** with `scripts/analyze_results.py --results_dir results`, which
  recurses into `results/rev_baselines_sel/`. The runs now carry two protocol labels: `group`
  (the declared selection rule, reported) and `group_final_epoch` (the older final-epoch
  reporting of the same configurations, kept for comparison). 97 -> 159 runs.
* **Reference numbers changed substantially.** IID centralized 90.9 -> 93.4 %, IID local-only
  78.3 -> 86.9 %, so the centralized-to-local-only band narrows from 12.6 to 6.5 points. Under
  label skew the local-only *accuracy* now exceeds the centralized one (95.5 vs 94.3 %) while
  its balanced accuracy does not (88.6 vs 94.7 %) -- an accuracy inversion, where before the
  two merely tied.
* **The effect is asymmetric and is a finding about reporting**, not about federation: the
  local-only models are the unstable ones across epochs (node C's local-only IID baseline
  swings between 0.96 and 0.42 validation accuracy in consecutive epochs at a training loss
  below 0.05), so a final-epoch snapshot penalises them far more often than it penalises the
  pooled model. Stated in Section IV-C of the paper and in the response letter.
* **`paper/main.tex`**: the group-level cells of `tab:protocol_effect` and the baseline rows of
  `tab:fullmetrics` updated from `analysis/summary_table.csv`; both footnotes now say that
  every row uses the selection rule. The two sentences whose claim depends on the size of the
  band carry `\todo{Revisit after the FL results...}` -- the narrative has to be written against
  where the federated strategies land inside the 6.5-point band, not before.
* **New: `tests/test_paper_numbers.py`.** These two tables cannot `\input` a generated file
  (they hold `\PH` placeholder rows for the federated strategies), so the test re-derives every
  hand-typed cell from `analysis/summary_table.csv` and fails on any disagreement; a guard test
  also fails if 90.9 / 78.3 reappear anywhere that does not name them as the superseded
  final-epoch protocol. This is what keeps rule 4 true for the two inline tables.
* **`analysis/leakage/` and `paper/tables/`**: `full_metrics_*.tex` are produced again (the
  selection-rule runs supply the full metric set) and `summary_selected_test_accuracy.tex` is
  new.

## The grouping threshold as a trade-off, and the BatchNorm hypothesis (2026-09-21)

* **New: `scripts/grouping_tradeoff.py`** (+ `tests/test_grouping_tradeoff.py`). Rebuilds all
  five partitions in memory under coarser groupings -- the connected components of the dHash
  edges at tau=8 unioned with the pHash edges at 3, 4 and 6 bits -- and measures the three
  quantities the choice trades off: the dHash 9-10 residual of the held-out images, the
  smallest held-out set any client is left with, and the number of per-node held-out class
  sets in which one sequence supplies >=90 % of the images. Inputs are the committed hashes
  (`analysis/leakage/groups.csv`, `phash.csv.gz`), so it needs neither the raw images nor a
  re-hash, and it never writes to `data/splits` or `data/processed`.
  Outputs `analysis/leakage/grouping_tradeoff.{csv,tex}`; `export_latex_tables.py` copies the
  `.tex` into `paper/tables/` so rule 4 keeps one pipeline into the paper.

  | grouping | non-triv. groups | dHash 9-10 residual | min held-out/client | sets >=90 % one seq. |
  |---|---|---|---|---|
  | tau=8 (kept) | 265 | 188-476 | 137 | 2/55 |
  | tau=8 + pHash<=6 | 152 | 65-91 | 116 | 8/53 |
  | tau=8 + pHash<=4 | 207 | 127-408 | 137 | 7/53 |
  | tau=8 + pHash<=3 | 247 | 125-520 | 137 | 5/56 |

* **`--check-shipped` verifies the comparison is honest.** `data_splitter.find_images` uses
  `os.walk`, whose order is not reproducible, while the hash tables are sorted by path, so the
  rebuild could in principle have produced a different partition from the shipped one. It does
  not: the `tau=8` row is identical to the shipped manifests on every metric. Keep the flag in
  the loop on any re-run -- if that equality breaks, the table has silently become a comparison
  between two different partitions rather than between two groupings.
* **tau=8 is presented as a justified choice, never as an optimum** (Sections III-C and V-A,
  and the response letter). pHash<=6 really does shrink the residual (188-476 -> 65-91), and we
  say so; it costs balance (137 -> 116) and dominance (2/55 -> 8/53). The residual is not even
  monotone in the merge threshold: Dirichlet alpha=1 goes 473 -> 406 at pHash<=4 but 473 -> 520
  at pHash<=3, because re-partitioning reshuffles which images are held out. An optimality
  claim would be refuted by this one table.
* **BatchNorm running statistics, as an untested hypothesis only** (ninth limitation). The
  local-only IID baseline of node C holds a training loss of 0.006-0.049 across all fifteen
  epochs while its validation accuracy swings from 0.96 at epoch 9 to 0.42 at epoch 15, test
  following at r = 0.82, and the late-epoch models are worse on *every* held-out sequence, not
  only the dominant one -- so it reads as a shift in the model's overall state rather than as
  overfitting to one scene. The suggested explanation, that the batch-normalisation running
  statistics are noisy at batch size 8 (verified: all 124 baseline `results.json` use
  `batch_size` 8) and drift between epochs, is stated explicitly as untested and is *not*
  connected to the round-1 label-skew result or to FedBN; neither link has been tested.

## Federated runs are reproducible at a fixed seed (2026-09-22)

Two Jetson runs of the same commit (e6c7ef7) at the same seed (42, `iid_sub0.01`,
FedAvg, 2 rounds) trained **identically on every client** -- same `train_loss`
(0.11324341938775163), same per-client val/test accuracy -- and still diverged at the
server:

| | run A | run B |
|---|---|---|
| round 1 aggregated val loss | 4.4800187735908255 | 4.480026002017449 |
| round 2 aggregated val loss | 0.7300890427681955 | 0.6840349276794742 |

**Cause.** Flower hands `aggregate_fit` / `aggregate_evaluate` the client results in
*arrival* order, and both of its helpers fold in list order:

    aggregate():         reduce(np.add, layer_updates) / num_examples_total
    weighted_loss_avg(): sum(num_examples * loss for ...) / total

Floating-point addition is not associative, so whichever Jetson reported first changed
the global model at ~1e-7 in round 1; the perturbed model changes every client's next
local optimum, and by round 2 the difference is 6 %.

**Fix** (`src/fl/aggregation_order.py`, new): sort the results by a stable client
identifier before aggregating.

* The key is the **node name** the client already echoes in its fit and evaluate
  metrics (`src/fl/client.py`), *not* Flower's `cid`, which is assigned per connection
  and differs between runs -- sorting on `cid` would fix nothing.
* `DeterministicClientOrder` mixin applied to `FedAvgOrdered` / `FedProxOrdered`
  (`src/fl/server.py`) and to `FedBN`; FedBN overrides `aggregate_fit`, so it sorts
  inside it and takes only `aggregate_evaluate` from the mixin.
* `RoundRecorder.fit_aggregation` / `evaluate_aggregation` sort on entry, which covers
  the weighted means in `aggregate()`, `round_val_loss` (the model-selection input) and
  the order of the per-client rows written to `results.json`.
* Audited for other arrival-order dependencies: the pooled confusion matrices are
  integer sums (order-independent); `pop_predictions` keys files by client and split and
  now receives sorted metrics; `weighted_average` is order-dependent in isolation but is
  left unchanged for v2 parity because every call path now feeds it sorted metrics.

**Deliberately not done:** `cudnn.deterministic` / `use_deterministic_algorithms`. They
can slow training, and per-round wall-clock is a reported result of this paper.

**`results.json` is unchanged**: it records `"strategy": args.strategy`, i.e. the CLI
name (`fedavg` / `fedprox_<mu>` / `fedbn`), never the new class names, so
`analyze_results.py` groups new runs with the old ones. Pinned by
`tests/test_aggregation_order.py`.

**Tests** (`tests/test_aggregation_order.py`, 16): every strategy aggregates all six
permutations of three clients to **bitwise-identical** parameters and losses
(`tobytes()` / `float.hex()`, not approximate equality -- 1e-7 is what became 6 %).
`test_control_unsorted_aggregation_is_order_dependent` aggregates *unsorted* and asserts
the order does change the result, so the suite cannot pass vacuously if the fixtures
stop exercising the bug. `tests/test_fl_server_recorder.py` no longer pins the exact
strategy class name; it asserts the base class and the mixin instead.

The paper's reproducibility sentence is written but the measured outcome of the
fixed-seed repeat test on the testbed is still open; it carries a `\todo` in
`paper/main.tex` and lands with that result.

## Declared primary metric, bootstrap CIs for the references, predictions format (2026-09-22)

### Balanced accuracy is the declared primary metric

Fixed 2026-09-22, **after** the reference results existed and **before any federated run of
the revision**. This is deliberately *not* called a pre-registration -- the clean-subset rule
(hard rule 7) was fixed before any metric of its kind existed; this one was prompted by a
result. The paper says so in the new methodology subsection "Declared Primary Metric", and so
does the response letter.

* Balanced accuracy leads for every partition and method, MCC second, accuracy reported but
  never ranking methods on its own. CLAUDE.md hard rule 8.
* **Why:** under label and Dirichlet skew each node's test split inherits that node's class
  proportions, so accuracy rewards majority-class prediction -- the failure federation should
  fix. The references demonstrate it: local-only beats centralized on accuracy (95.5 vs
  94.3 %) and loses on balanced accuracy (88.6 vs 94.7 %), so the two rankings disagree.
* Column order: `FULL_METRIC_ORDER` and `DEFAULT_METRICS` in `scripts/export_latex_tables.py`;
  `tab:fullmetrics` in the paper reordered to Bal.\ acc. / MCC / Acc. / ...; the abstract now
  leads with balanced accuracy and states the inversion. `tests/test_paper_numbers.py` maps
  column position to metric, so it also pins the new order.
* The gap-dependent narrative is untouched and still carries its `\todo` markers.

### Reference CIs now use the same cluster bootstrap as the federated runs

`scripts/predict_from_checkpoint.py` regenerated per-image predictions for all 62
selection-rule baselines from their `model_selected.pt`; every run reproduced its logged
selected-epoch metrics exactly. `analyze_results.py` then switches a configuration to the
sequence-level bootstrap automatically once every run in it has predictions, so no code change
was needed for that part. **No table mixes CI methods**: the selected-test tables are
uniformly `cluster_bootstrap_B1000`, and the `t_seeds` tables (`final_*`, `total_time_s`,
`v1_*`) contain only protocols that have no predictions by construction.

Effect on balanced accuracy (t-interval over seeds -> cluster bootstrap):

| partition | kind | mean | t-interval | cluster bootstrap | width |
|---|---|---|---|---|---|
| IID | centralized | 0.9374 | [0.920, 0.954] | [0.894, 0.958] | x1.90 |
| IID | local-only | 0.8449 | [0.826, 0.864] | [0.781, 0.879] | x2.57 |
| label skew | centralized | 0.9471 | [0.936, 0.958] | [0.920, 0.969] | x2.18 |
| label skew | local-only | 0.8864 | [0.864, 0.908] | [0.835, 0.919] | x1.91 |
| Dir. 0.1 | local-only | 0.5622 | [0.392, 0.732] | [0.553, 0.809] | x0.75 |
| Dir. 1 | centralized | 0.8903 | [0.741, **1.040**] | [0.822, 0.968] | x0.49 |

At five seeds the bootstrap is about twice as wide -- the understatement it exists to correct.
At three seeds it is often narrower, because the $t$ interval is dominated by
$t_{0.975,2}=4.30$ and can leave the unit interval entirely, as at Dirichlet 1.

### Bug found and fixed: the pooled bootstrap used only the first node

Checking the before/after exposed CIs that **did not contain their own point estimate** (label
skew, centralized: mean 0.9471, CI [0.953, 0.998]).

`run_replicates` implements `aggregation="pooled"` as `reps[0]` and documents it as
"centralized: one unit", but `_bootstrap_cis` handed it **one unit per node prediction file**.
Every centralized CI was therefore bootstrapped from `node_a` alone while its point estimate
pooled all three nodes; under label skew node_a is 80 % fire, which is why the interval sat
above the mean.

Fixed in `scripts/analyze_results.py` (`_units_for`), **not** in `src/evaluation/bootstrap.py`:
the caller was violating the callee's documented contract, and `src/` is frozen for the Jetson
block. The per-node units are concatenated into one, exactly as `_aggregate_point` already does
for the point estimate; near-duplicate groups never span nodes, so concatenation cannot merge
two sequences. After the fix all 360 bootstrap rows contain their point estimate, none is NaN,
and no rate metric leaves [0, 1].

Only `"pooled"` was affected: `"mean"` (local-only) and `"weighted"` (FL) already used every
unit, so federated runs were never mis-estimated.

### Predictions format documented in git

`docs/PREDICTIONS_FORMAT.md` is the committed counterpart of the README that
`src/evaluation/predictions.py` writes into each `predictions/` directory (now gitignored, like
the `.npz` files). `tests/test_predictions.py` pins the doc's format version, file-name
patterns and all four array rows to `README_TEXT`, so the two cannot drift apart.

### Reproducibility paragraph

The measured outcome of the fixed-seed repeat test is now in the paper (Section III-I): on the
testbed at commit e85945d two FedAvg runs agreed to all printed digits on the aggregated
validation loss of each round, and all twelve per-image prediction files were bitwise
identical. The scope is stated honestly -- FedAvg in that configuration; the fixed aggregation
order applies to all strategies and the unit tests cover all three, but FedProx and FedBN were
not separately re-run twice on hardware. The smoke-test runs are not committed to `results/`.

## The cluster bootstrap is stratified by class composition (2026-09-22)

The Dirichlet 0.1 local-only balanced accuracy sat at the very edge of its own interval
(mean 0.5622, CI [0.553, 0.809]) -- the same symptom as the pooled bug, and the same kind of
cause.

**Diagnosis.** `Unit.weights` resampled sequences uniformly, so a resample could contain no
sequence carrying one of the classes, and `metrics_from_counts` computes

    bal = (rec1 * has1 + rec0 * has0) / (has1 + has0)

which silently reports balanced accuracy as the recall of the *surviving* class when one is
missing -- biasing replicates upward. Measured fraction of resamples that dropped a class:

| run | unit | sequences | minority-class sequences | dropped a class |
|---|---|---|---|---|
| dirichlet0.1 local seed42 | node_c | 30 | **1** (211 no-fire images) | **37.8 %** |
| dirichlet0.1 local seed42 | node_b | 31 | 2 (55 no-fire images) | 14.4 % |
| iid local seed42 | node_a | 22 | 2 (1011 no-fire images) | 13.5 % |

Not a Dirichlet-only problem: the IID partition has a node with 1011 no-fire images in two
sequences.

**Fix.** Sequences are split into three strata by class *composition* -- only class 0, only
class 1, both -- and each stratum is resampled with replacement to its own size. A mixed
sequence carries both classes by construction, so if a class occurs anywhere in a unit it
occurs in every resample; measured dropout is now 0.0 % everywhere.

Stratifying by a sequence's **majority** class would not have worked, and this is worth
recording: FLAME sequences are videos in which the fire appears and disappears, so a node's
entire minority class can live inside sequences that are majority the other class. Two of the
three nodes above have *no* no-fire-dominant sequence at all, so the minority stratum would
have been empty exactly where it was needed.

**Which rows moved** (360 bootstrap rows): 179 moved by more than 0.01, 32 by more than 0.05,
15 by more than 0.10; no row is unchanged. The movement is concentrated in the local-only rows,
whose per-node units are small enough to lose a class; centralized rows, which bootstrap one
pooled unit, barely move.

| partition | kind | median move (balanced accuracy) |
|---|---|---|
| dirichlet_0.1 | local | 0.1174 |
| iid_sub0.01 | local | 0.0329 |
| iid_sub0.05 | local | 0.0298 |
| iid | local | 0.0212 |
| non_iid_label | centralized | 0.0022 |

The row that prompted this: dirichlet_0.1 local-only balanced accuracy, [0.553, 0.809] ->
[0.510, 0.682], with the point estimate 0.5622 now properly interior. Across all rows, the
number whose point estimate sits in the lowest decile of its own interval went 2 -> 0.

**Permanent guards** (`tests/test_predictions.py`), so neither defect can return:
(a) every CI contains its point estimate, on the skewed shape that produced the symptom, for
both `mean` and `pooled` aggregation; (b) no resample drops a class the unit contains -- with a
control asserting the *unstratified* path still can, so the guard cannot pass vacuously;
(c) `pooled` collapses the per-node units into one while `mean`/`weighted` keep them separate.
A further test pins the strata to class composition rather than majority class.

`weights(..., stratified=False)` is kept so the effect can be measured, and is documented as
not for reported intervals.

**Note on the src/ freeze.** This changes `src/evaluation/bootstrap.py` during the Jetson block.
That module is analysis-only -- it is imported by `scripts/analyze_results.py` and the tests and
by nothing under `src/fl/` -- so it cannot affect a running federated run. The nodes are
unaffected regardless, since they do not pull during the block.

## Sequence counts per class, and a dagger on the intervals that rest on one or two videos (2026-09-22)

The stratified bootstrap keeps every class present in every replicate, but it cannot create
information the partition does not hold. When a node's minority class is carried by one or two
held-out sequences there is no between-sequence variance for that class to estimate, so the
interval is conditional on those particular videos and is too narrow as a statement about new
footage. That limit is now counted, stated and marked.

* **Methodology** (Section III-J) now names the estimator precisely: a hierarchical
  sequence-level (cluster) bootstrap *percentile* interval, B = 1000, with the sequences
  resampled stratified by class composition into the three strata *fire-only*,
  *no-fire-only* and *mixed*, each resampled to its own size -- with the one-line reason that a
  class confined to few sequences would otherwise vanish from some resamples and balanced
  accuracy would silently become the recall of the surviving class.
* **`scripts/heldout_dominance.py`** now also writes, from the committed manifests alone:
  `analysis/leakage/heldout_sequences.tex` (sequences per class, per partition and node,
  `tab:heldout_sequences`) and `analysis/leakage/scarce_minority.csv` (which
  `(partition, kind)` configurations rest on <= 2 minority-class sequences).
  `export_latex_tables.py` copies the table into `paper/tables/` and reads the flags.
* **Five of the thirty held-out cells** of the five main partitions have a minority class of at
  most two sequences. The count is invisible in the image counts, which is the point:

  | partition | node | split | minority class | images | sequences |
  |---|---|---|---|---|---|
  | dirichlet_0.1 | node_c | test | No_Fire | 211 | **1** |
  | dirichlet_0.1 | node_c | val | No_Fire | 5 | **1** |
  | dirichlet_0.1 | node_b | test | No_Fire | 55 | **2** |
  | dirichlet_0.1 | node_b | val | No_Fire | 274 | **2** |
  | iid | node_a | test | No_Fire | **1011** | **2** |

* **Flagging.** Every generated summary table marks the affected cells with `$^{\dagger}$`
  next to the CI, and the footnote explains it; `tab:fullmetrics` marks the IID local-only row.
  Only per-client configurations (`local`, `fl`) are ever flagged -- the centralized rows
  evaluate one pooled split, whose sequences come from all three nodes, and are never scarce.
* **Limitations** states what the flag means: the interval describes uncertainty about which
  frames of that footage were held out, not about a new fire from a new flight, and widening it
  is a matter of more footage rather than more seeds -- with 265 sequences in FLAME and whole
  sequences as the unit of assignment, a three-way split cannot give every node many sequences
  of its minority class.
* **Tests.** A flagged configuration must carry the dagger and an unflagged one must not, and
  the footnote must explain it. Two pre-existing test gaps surfaced and were fixed rather than
  worked around: `assert_valid_latex` treated `\ref{...}` keys as typeset text and so rejected
  a legitimate underscore in a cross-reference, and the `tab:fullmetrics` row matcher in
  `tests/test_paper_numbers.py` did not tolerate a marker on the method name.

## Scarce minority classes also bias model selection (2026-09-22)

Added to Limitations, next to the dagger explanation: the scarce cells matter for the
*selection rule*, not only for the confidence intervals, and asymmetrically. Under
Dirichlet 0.1 node_c's validation split holds **5 no-fire images in one sequence** against
1,387 fire images, so its local-only model is selected almost entirely on fire-class loss,
while a federated run selects on the validation loss aggregated across all three clients and
so sees node_a's 2,347 no-fire validation images -- **89 % of that partition's no-fire
validation data**. In the most skewed partition the selection rule is therefore better
informed for FL than for local-only. Stated as a caveat on that comparison, not as a property
of either method. No new analysis: the counts come from
`analysis/leakage/heldout_dominance.csv`.

## Round-1 collapse: interpretation plan, written before the evidence (2026-09-22)

**Status and timing.** Written on 2026-09-22, after exactly **one** federated run existed
(`rev_iid_fedavg_seed42`, IID, FedAvg, 3 rounds, seed 42) and **before** any local-epoch sweep,
FedProx or FedBN result of the block had been produced. It is recorded now so that the reading
of those results is fixed in advance rather than chosen after seeing them. It is a plan, not a
finding: one seed, one strategy, one partition.

### What was observed (the anchor)

In round 1 the aggregated model predicted **no-fire for all 14,311 held-out images** on all
three clients: `tp = 0, fp = 0` everywhere, accuracy equal to each node's no-fire base rate
(0.3517-0.4013), balanced accuracy exactly 0.5000, weighted validation loss 2.913.

The representation was **not** destroyed. Fire images still scored above no-fire images in
every cell, and ROC-AUC reached 0.9876 (node_a test) and 0.9238 (node_b test) -- ROC-AUC is the
clean evidence here because it is threshold-free and fits nothing. The maximum logit margin
over all images was negative (-0.766 to -1.906, max `p_fire` 0.13-0.32). Meanwhile every client
had reached `train_loss` 0.020-0.060 locally. The failure is therefore in the **decision
boundary / output calibration** of the averaged model, not in its features.

How much a threshold can repair is a separate question, and it must be answered honestly.
Fitting the threshold on each node's **validation** split and applying it to that node's
**test** split gives round-1 test balanced accuracy of **0.797 (node_a), 0.814 (node_b) and
0.487 (node_c)** -- node_c is *below chance*, i.e. its validation-fitted threshold does not
transfer, which is consistent with its low test ROC-AUC of 0.5721. Choosing the threshold
directly on test would instead give 0.9368 / 0.9118 / 0.7248; **those numbers are fitted on the
test set, are an upper bound rather than a result, and are recorded here only as diagnostic
evidence that the ranking survived.** They must never be reported as a metric or used to select
anything. An earlier version of this section quoted that test-fitted range (0.66-0.94) without
the distinction; it overstated the recovery, most of all on node_c.

The partition alone does not explain it: the `iid_sub0.01` smoke run uses the *same*
group-level, flight-disjoint partition and did not collapse (0.61 round-1 balanced accuracy),
differing only in having 1 % of the training data and therefore far less local drift per round.

### Operational definition of "collapse" (mechanical, no judgement)

A round-1 result **collapses** iff, for **every** client `k` and **both** held-out splits
`s in {val, test}` of that client:

    balanced_accuracy(k, s, round 1) <= 0.55   AND   max_i logit_margin(k, s, i) < 0

The second clause is what distinguishes "predicts one class everywhere" from a low balanced
accuracy arising some other way. `rev_iid_fedavg_seed42` satisfies it exactly (0.5000 in all
six cells; largest margin over all six, -0.766).

* **reduced**: the test-size-weighted pooled round-1 balanced accuracy is at least 0.05 above
  the matched FedAvg run at the same seed and partition, while at least one client still meets
  both clauses above.
* **absent**: not a collapse -- some client exceeds 0.55, or some client's maximum margin is
  positive.

Comparisons are made **within a seed**: the criterion is evaluated per run, and a change counts
only against the FedAvg run of the same seed, partition and round budget.

### What each outcome would indicate

| Outcome | Reading |
|---|---|
| Collapse present at E = 5, absent at E = 1 (same partition, seed, strategy) | **Drift magnitude.** The number of local steps before the first average is the operative variable. |
| FedProx with mu > 0 removes or reduces it at fixed E = 5 | **Drift magnitude.** The proximal term bounds the distance from the broadcast weights, so it acts on exactly that quantity. |
| FedBN removes it while FedAvg *and* FedProx do not | **Averaged BatchNorm running statistics.** FedBN differs from FedAvg precisely by keeping normalisation parameters and statistics client-local. |
| Collapse persists under all of the above | **Neither.** The mechanism is not local-drift magnitude and not BN aggregation; the diagnostic run below becomes necessary. |

Recorded caveats, so they are not discovered afterwards:

* The first two rows are **not independent** -- both reduce drift. If both hold, that is
  consistent with drift magnitude but does not identify *which* aspect of drift matters.
* FedBN changes *what is averaged* as well as the BN statistics, so row 3 implicates BN
  aggregation as a whole, not the running statistics specifically. Separating those needs the
  diagnostic run.
* A mixed outcome (e.g. absent at E = 1 *and* removed by FedBN) is evidence for both and
  resolves neither; it is not to be reported as support for whichever is written up first.
* None of these rows is established by a single seed.
* **Any threshold recalibration used in evaluating these outcomes is fitted on validation data
  and applied to test, never fitted on test.** A threshold chosen directly on the test split is
  diagnostic evidence that the ranking survived and nothing more: it is an upper bound, it is
  labelled as such wherever it is written down, and it may not be reported as a metric,
  compared against the references, or used to select a threshold, a round or a strategy. The
  distinction is not academic here -- on node_c the validation-fitted threshold yields 0.487
  test balanced accuracy against 0.725 for the test-fitted one, so the test-fitted number would
  have misrepresented a failed recalibration as a partial recovery. Prefer ROC-AUC, which makes
  the same point about ranking without fitting anything.

### Post-block diagnostic run (design only, not implemented)

Runs **only after the whole matrix is finished**, and never mixes into its results: output goes
to a top-level `diagnostics/` directory, **not** under `results/`, so no analysis script can
reach it (`iter_run_dirs` walks `results/` only) and no reported table can contain it.

**D0 -- threshold recalibration (costs no testbed time).** Offline from the already-saved
round-1 `.npz`. **The threshold is fitted on validation data and only then applied to test.**
For seed 42 that gives round-1 test balanced accuracy 0.797 / 0.814 / 0.487 on nodes A / B / C,
against 0.5000 at the model's own threshold; node_c's validation-fitted threshold does not
transfer at all.

The threshold that would be best *on test* (0.9368 / 0.9118 / 0.7248, at -5.67 / -4.62 / -7.00)
is **fitted on the test set**. It is an upper bound, not an achievable result: it is recorded
only as diagnostic evidence that the ranking survived aggregation, and it must never appear as
a reported metric, never be compared against the baselines, and never be used to choose a
threshold, a round, a strategy or anything else. Wherever it is written down it carries that
label. The threshold-free statement of the same evidence is ROC-AUC, which fits nothing and is
the form to prefer.

D0 establishes how much of the collapse is the decision boundary alone, as the reference point
for D1 and D2.

**D1 -- per-client model before aggregation.** One round of FedAvg, IID, seed 42, 5 local
epochs -- identical to the block's round 1. Each client evaluates **its own locally trained
model** on its own val and test split and writes predictions *before* returning parameters to
the server; the server then aggregates and evaluates as usual, reproducing the existing
post-aggregation numbers in the same run.
*Decides:* whether each client's model is individually well-calibrated (balanced accuracy high,
maximum margin > 0) while only their average is collapsed -- which would place the cause in
averaging -- or whether the clients are already degenerate, which would place it in local
training. At present this is only **inferred** from `train_loss`, never measured on held-out
data.

**D2 -- BatchNorm re-estimation.** Reusing D1's saved round-1 aggregated checkpoint, with no
retraining: re-estimate the BatchNorm running statistics with forward passes in training mode
over a small sample of each client's *own* training split (256 / 512 / 2048 images, no gradient
updates), then re-evaluate on the same held-out splits.
*Decides:* whether restoring the normalisation statistics alone repairs the model -- implicating
the averaged BN running statistics -- or whether the shift lives in the weights. Sampling is
per client and local; nothing is pooled, so the federated constraint is not violated.

**Cost**, from this run's measured timings (round-1 fit phase 1,402.7 s, evaluation phase 44.5 s
for val + test across the three clients in parallel; node_c is the slowest at ~109 images/s):

| | work | testbed wall-clock |
|---|---|---|
| D0 | offline, existing files | **0** |
| D1 | 1 FedAvg round (5 local epochs) + 1 extra val/test pass per client | ~1,490 s (**0.41 h**) |
| D2 | BN forward passes (<= 2,048 images/client, ~19 s) + 1 val/test pass, x3 sample sizes | ~200 s (**0.06 h**) |
| | **total, one seed** | **~0.5 h** |
| | three seeds (42, 123, 456) | **~1.4 h** |

Under 1 % of the block's ~220 h, so cost is not a reason to skip it; the reason to defer it is
only that it must not perturb the matrix.

### The margin clause proved knife-edge at the second seed (2026-09-22, added after seed 123)

`rev_iid_fedavg_seed123` was scored against the collapse criterion **exactly as written above**.
Verdict: **NOT a collapse**, because the `max logit_margin < 0` clause fails on four of the six
client/split cells.

| node | split | bal. acc. | `<= 0.55` | max margin | `< 0` | images predicted fire | clauses |
|---|---|---|---|---|---|---|---|
| node_a | val | 0.5010 | yes | +0.1515 | no | 3 / 2340 | FAIL |
| node_a | test | 0.5090 | yes | +0.3571 | no | 27 / 2519 | FAIL |
| node_b | val | 0.5000 | yes | -0.5389 | yes | 0 / 2326 | pass |
| node_b | test | 0.5003 | yes | +0.1363 | no | 1 / 2326 | FAIL |
| node_c | val | 0.5003 | yes | +0.1969 | no | 1 / 2400 | FAIL |
| node_c | test | 0.5000 | yes | -0.4151 | yes | 0 / 2400 | pass |

**The verdict diverges from the obvious reading, and the criterion is not being changed.** The
balanced-accuracy clause passes in every cell (0.5000-0.5090, against 0.5000 throughout at seed
42). What fails is the margin clause, decided by **32 images out of 14,311 (0.22 %)** whose
margins reach +0.14 to +0.36 while the bulk of the distribution sits near -4. Every one of
those 32 images is a true fire image, so the model is not mistakenly firing: it is the same
one-class behaviour as seed 42, with a handful of correct detections barely clearing zero.

Recorded consequences:

* The margin clause was written as a way of saying "predicts one class everywhere". At the
  second seed it turned out to be a **knife-edge test**: a single image at +0.14 flips the
  verdict for a whole client/split. It discriminates on a quantity far finer than the
  phenomenon it was meant to capture.
* The criterion **stands as written**. Seed 42 is a collapse under it; seed 123 is not. Both
  verdicts are reported as the criterion gives them.
* Any alternative criterion adopted later (for instance, a bound on the *fraction* of images
  predicted as the minority class rather than on the maximum margin) is **post hoc**: it was
  formulated after seeing seed 123. It must be labelled as such, reported alongside the
  original criterion and its verdicts, and never substituted for it in a way that erases the
  original reading. The reason is exactly the one that motivated writing the plan in advance:
  a threshold chosen after seeing the data it will be applied to is not a prediction.
* This is a second illustration of the same failure mode as the test-fitted threshold above --
  a quantity that looks decisive until it is checked against data it was not built on.

## Automated result collection from the testbed (2026-09-22)

Node A is chaining the remaining blocks (~6-9 days). Finished runs are pulled to this
desktop hourly, so that a report can be produced per block and so that there is a second
copy of every run -- this job is also the backup against SD-card failure on Node A.

Everything runs **on the Windows desktop, never on a Jetson**.

* **`scripts/fetch_results.ps1`** -- one pass of the fetch.
  * **Strictly read-only on the node.** The complete set of remote operations is:
    `md5sum results/rev_*/results.json`, `cat` of a run's `results.json`, `tail` of
    `logs/run_matrix.log`, and `scp` *pulls*. No git, no writes, no deletes, no process
    control, ever -- Node A is mid-block and a stray write there costs days.
  * **Only finished runs.** A run is fetched only when its remote `results.json` parses
    as JSON *and* carries `model_selection.selected_round`; a run still being written
    fails that test and is retried next pass.
  * **Never overwrites.** Identical copies (md5 of `results.json`) are skipped. A local
    copy that differs is left alone and logged as a warning.
  * Runs already **committed** (the 31 desktop-GPU baselines) legitimately differ from
    the node's own copies; they are recognised via `git ls-files` and ignored, so the one
    warning that matters -- a fetched run changing underneath us -- is not buried under
    thirty that do not.
  * **Atomic.** Each run lands in `results/.incoming_<run>` and is renamed only after its
    checksum matches, so an interrupted transfer can never leave a half-written directory
    that a later pass mistakes for a finished run.
  * Uses **Windows OpenSSH** explicitly (`C:\Windows\System32\OpenSSH\ssh.exe`): the key
    is in the Windows ssh-agent, which Git Bash's own ssh cannot see. `BatchMode=yes`
    plus a connect timeout means auth or network failure logs and exits rather than
    hanging; the scheduled task also has a 30-minute execution limit as a backstop.
  * Logs every attempt to `logs/fetch.log` (gitignored).
* **`scripts/block_report.py`** -- after new runs arrive, checks whether a block is
  complete and, if so, runs `analyze_results.py` and `export_latex_tables.py` into
  `scratch/block_reports/<block>/` with a short `STATUS.md`.
  * Completeness comes from `configs/experiment_matrix.yaml` via
    `print_revision_commands` -- the same source the node runs from -- not from parsing
    `run_matrix.log`, so it cannot be fooled by log formatting.
  * It writes **only** to `scratch/` (gitignored), never to `analysis/` or
    `paper/tables/`, and commits nothing. The status file is mechanical: counts, selected
    rounds, wall-clock, where the tables went. **No interpretation** -- that is requested
    per block.
* **`scripts/install_fetch_task.ps1` / `uninstall_fetch_task.ps1`** -- register and remove
  the hourly Task Scheduler entry `FedRGBD-FetchResults`. Registered with
  `LogonType = Interactive` on purpose: ssh-agent keys are protected per user and a task
  set to run while logged off cannot reach them. `-RunWhenLoggedOff` exists for the
  identity-file case.
* **`scripts/fetch_results.config.json`** is gitignored (it names the host and account);
  `fetch_results.config.example.json` is the committed template.

Verified by hand before scheduling: connection, dry run, a real fetch of three runs
(18 prediction files each), an idempotent second pass (0 fetched, 5 present), no staging
directories left behind, and one scheduled pass with `LastTaskResult = 0`.

**Commit rule unchanged**: no `results/rev_*` FL directory and nothing derived from the FL
runs is committed until the matrix is done and Node A commits the results. Only the
scheduler scripts are committed here.

### Fix: block_report enumerated a different run set than the node executes (2026-09-22)

`block_report.py` expanded the matrix **without** `--all_seeds`, while
`scripts/run_matrix.py` always passes it (CLAUDE.md hard rule 3: runs from before the
leakage-safe re-split used a different partition and are not comparable). The two
therefore disagreed about what a block contains:

| block | reported before | actually executed |
|---|---|---|
| `seed_extension` | 8 (new seeds 789/1011 only) | **20** (all five seeds) |
| `baselines_extension` | 50 | **62** |

Consequence, had it not been caught: `seed_extension` would have been declared complete
after its 8 new-seed runs and the analysis pipeline fired with **12 runs still missing**.

* `print_revision_commands.apply_all_seeds()` is now the single definition of that
  transformation, used by the CLI's `--all_seeds` flag and by `block_report`. Pure
  refactor -- the CLI emits the same 20 runs for `seed_extension` as before.
* `tests/test_block_report.py` (11 tests) pins the two together: for **every** block it
  compares `block_report`'s enumeration against the run set that
  `print_revision_commands.py --all_seeds --block <name> --format bash` emits -- i.e.
  against the command `run_matrix` actually runs -- and fails on any run missing from
  either side. Verified to catch the defect: reverting the fix fails 3 tests naming both
  affected blocks.
* A further test greps the fetch/report path for any expander call that does not mention
  `all_seeds`, so a second caller cannot reintroduce it.

Audit of every other caller: `scripts/make_desktop_lanes.py` and
`tests/test_run_matrix.py` both already widen the seed set correctly, but each carries
its own hand-rolled copy of the transformation -- which is the duplication that produced
this bug. They are left alone while the block is running; `apply_all_seeds` is now
available to both.

Corrected status of the seven chained blocks at the time of the fix:

    seed_extension        5/20
    dirichlet_skew        0/18
    low_data              0/24
    long_horizon_fedbn    0/6
    mu_grid               0/15
    local_epochs          0/18
    learning_rate         0/12

(`baselines_extension` reads 0/62 and is not part of the chain: its runs are the
selection-rule baselines, which already exist under `results/rev_baselines_sel/` rather
than at the top level, so that block can never be reported complete and can never fire.)

## Post-matrix cleanup list

Work that is deliberately **deferred until the Jetson matrix is finished** and Node A has
committed the FL results. Nothing here is urgent; each item is recorded at the moment it
was noticed so it is not rediscovered later. Add to this list rather than fixing mid-block
whenever a change would touch frozen paths or risk perturbing a running block.

- [ ] **Consolidate `apply_all_seeds`.** `scripts/make_desktop_lanes.py` and
      `tests/test_run_matrix.py` each carry their own hand-rolled copy of the
      `--all_seeds` transformation. Both are correct today, but that duplication is
      exactly what produced the `block_report` defect (8 runs reported against 20
      executed). Point both at `print_revision_commands.apply_all_seeds()`.
      `make_desktop_lanes.py` widens only `baselines_extension`; the shared helper also
      widens `seed_extension`, which that script never reads, so the change is
      behaviour-preserving -- confirm that with a test rather than by inspection.
- [ ] **Re-check `NOT_IN_CHAIN` in `scripts/block_report.py`.** `baselines_extension` is
      excluded because its runs live under `results/rev_baselines_sel/` rather than at the
      top level. If the baselines are ever moved or re-run at the top level, the exclusion
      becomes wrong and should be removed.
- [ ] **Run the D0/D1/D2 diagnostic** for the round-1 collapse (design and cost in the
      interpretation-plan section above). Only after the whole matrix, into a top-level
      `diagnostics/` directory, never mixed into `results/`.
- [ ] **Fold the FL results into `analysis/` and the paper.** After Node A commits them:
      re-run `analyze_results.py` and `export_latex_tables.py`, then resolve the `\todo`
      markers that are waiting on FL numbers -- the gap-dependent narrative, the round-1
      label-skew attribution, the abstract's quantitative sentences, the FedProx overhead
      factor, and the federated rows of `tab:protocol_effect` / `tab:fullmetrics`.
      `tests/test_paper_numbers.py` will name any hand-typed cell that went stale.
- [ ] **Decide the fate of the hourly fetch task.** `scripts/uninstall_fetch_task.ps1`
      removes it; the local config and `logs/fetch.log` are kept deliberately.

### Fetch job: alerting on the node's failure markers (2026-09-22)

The hourly fetch now also reads `~/chain.log` and `<repo>/logs/run_matrix.log` on Node A
(still read-only, still only `tail`) and raises an alert on `STOPPING`, `DURDU`,
`BASLAMADI` or `pre-flight FAIL`.

* An `[ALERT]` line goes to `logs/fetch.log` and a desktop pop-up appears -- `msg.exe`
  first, falling back to a **detached** message box, because a scheduled pass must never
  block waiting for someone to click OK.
* **Once per event.** Each matching line is fingerprinted (source + full line text) into
  `logs/fetch_alerts.state.json`, so an hourly pass over the same log is silent. Log lines
  carry timestamps, so a genuinely repeated event is a different line and alerts again.
  The state keeps only fingerprints still inside the tail window, so it stays bounded
  across the 6-9 day run.
* On the very first pass the state file does not exist, so pre-existing markers are
  recorded and logged at `INFO` instead of firing a pop-up for history.
* An alerting fault can never fail the fetch: the scan is wrapped, and a failure is a
  `WARN`. Fetching the runs is the job that matters.

**Trap worth knowing about: this machine runs under `tr-TR`, where case-insensitive
matching is broken for any word containing `i` or `I`.** In Turkish the capital of `i` is
`İ` and the lowercase of `I` is `ı`, so .NET's culture-aware folding treats `I` and `i` as
different letters. Verified here:

    [regex]::IsMatch('fail','FAIL', IgnoreCase)                  -> False
    [regex]::IsMatch('fail','FAIL', IgnoreCase|CultureInvariant) -> True

PowerShell's `-imatch` inherits this. **Every** marker above contains an `i` or an `I` --
`STOPPING`, `BASLAMADI`, `pre-flight FAIL` -- so the case-insensitive pattern would have
silently never fired while the job looked perfectly healthy. `Get-RegexOptions` in
`fetch_results.ps1` therefore always sets `CultureInvariant`. Any future case-insensitive
matching in this repository's PowerShell needs the same flag; `-imatch` alone is not safe
on this machine.

The all-caps markers are matched case-sensitively on purpose, so ordinary prose such as
"stopping the server cleanly" stays quiet.

Verified end to end: a temporary pattern matching an existing `chain.log` line produced
one `[ALERT]` and a pop-up on the first pass and nothing on the second.
