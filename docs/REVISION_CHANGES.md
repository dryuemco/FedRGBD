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
* Not done (needs the author): funding / AI-disclosure wording; testbed photo; copying
  `data/processed` to the Jetsons (or re-running the splitter there and comparing the manifest
  MD5s in `P0_SUMMARY.md`).  Everything else that remains is a Jetson run, and the desktop GPU
  baseline block can start now.
