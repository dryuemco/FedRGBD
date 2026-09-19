# FedRGBD — Reviewer's Guide to Reproducing Results

## Overview

This document provides step-by-step instructions for reviewers to reproduce
the experiments presented in the manuscript. All code, configurations, and
instructions are provided for full reproducibility.

## Hardware Requirements (Exact Setup)

| Component | Model | Quantity | Notes |
|-----------|-------|----------|-------|
| Edge device | NVIDIA Jetson Orin Nano Super 8GB | 3 | JetPack 6.2, 67 TOPS |
| Camera A | Intel RealSense D435if | 1 | Active IR stereo, IR cut filter |
| Camera B | Intel RealSense D435i | 1 | Active IR stereo |
| Camera C | Stereolabs ZED 2i | 1 | Passive stereo + neural depth |
| Network | Gigabit Ethernet switch | 1 | FL communication (v1 used a WiFi 802.11ac router) |

**Note:** Results on different hardware will differ due to different power profiles,
memory constraints, and sensor characteristics. This is expected and is part of
the study's contribution — real hardware produces results that simulations cannot capture.

## Software Requirements

- JetPack 6.2 (L4T 36.4.3, CUDA 12.6)
- Python 3.10
- PyTorch 2.5.0a0 (NVIDIA Jetson wheel)
- torchvision 0.20.0 (built from source)
- Flower 1.13.1
- librealsense 2.55.1 (RSUSB backend, Nodes A & B)
- ZED SDK 5.2.3 (Node C)
- numpy 1.26.4 (numpy 2.x is incompatible)

## Reproduction Steps

### Step 1: Hardware Setup (~2 hours)
```bash
# Flash JetPack 6.2 on all 3 Jetsons using NVIDIA SDK Manager
# Connect cameras: D435if → Node A, D435i → Node B, ZED 2i → Node C (all USB3)
# Connect all nodes to one Gigabit Ethernet switch (wired; v1 used WiFi)
# Assign static IPs: Node A=192.168.1.10, Node B=192.168.1.7, Node C=192.168.1.6
```

### Step 2: Software Setup (~1 hour per node)
```bash
git clone https://github.com/dryuemco/FedRGBD.git
cd FedRGBD
./setup_jetson.sh
```

### Step 3: Data Preparation (~30 minutes)
```bash
# Download FLAME dataset from Kaggle to data/raw/flame_dataset/
python3 src/data/data_splitter.py --data_dir data/raw/flame_dataset --output_dir data/processed --nodes 3 --clean
```
`--clean` removes each `data/processed/<split>/` tree before rewriting it; it is
required whenever `data/processed` already holds a partition (otherwise the
previous partition's files stay behind and leak train images into val/test).

### Step 4: Run All Experiments (~120 hours total compute, ~40 hours wall-clock)
```bash
# See scripts/run_experiment.sh for individual experiments
# Example: 3-node Non-IID FedAvg with seeds 42, 123, 456
bash scripts/run_experiment.sh fedavg non_iid 42 123 456
```

### Step 5: Generate Results
```bash
python3 scripts/generate_plots.py     # PDF/PNG figures (paper v1, hard-coded numbers)
python3 scripts/analyze_results.py --results_dir results --output_dir analysis
#   -> analysis/summary_table.{csv,md}  mean ± std, 95 % t-CI over seeds, per config
#   -> analysis/pairwise_tests.{csv,md} paired Cohen's d, Wilcoxon, paired t-test
#   -> analysis/friedman.csv            Friedman test across >= 3 strategies
#   -> analysis/accuracy_vs_{round,time,communication}_<dist>.{png,pdf}
```

## Revision Protocol (NCAA-D-26-02211, major revision)

The revision adds the following hardware-independent steps; none of them
change the model or the already-recorded results under `results/`.

### R1. Near-duplicate audit and sequence-level split
```bash
python3 scripts/analyze_flame_leakage.py --data_dir data/raw/flame_dataset \
    --processed_dir data/processed --output_dir analysis/leakage --threshold 8 --workers 4 \
    --sweep 4 6 8 10 12 --sequence_heuristic --examples 20
```
`--sweep` adds a threshold-sensitivity table (`threshold_sweep`) without
changing which threshold writes `groups.json`, `--sequence_heuristic` reports
the share of consecutive frame numbers that fall into one near-duplicate group,
`--examples` writes `example_groups.txt` for manual inspection, and
`--hash both --phash_threshold 10` clusters the union of the dHash and pHash
near-duplicate edges.
`analysis/leakage/leakage_report.json` reports, per split and node, the share
of val/test images that have a near-duplicate (Hamming ≤ 8 on a 64-bit dHash)
in *any* node's training split, plus the number of near-duplicate groups that
span several nodes.  `groups.json` is then passed to the splitter:
```bash
python3 src/data/data_splitter.py --data_dir data/raw/flame_dataset --output_dir data/processed \
    --nodes 3 --seed 42 --group_file analysis/leakage/groups.json \
    --dirichlet_alpha 0.1 0.5 1.0 --subsample_frac 0.05 0.01 --clean --verify
```
`--clean` is required (this rewrites the existing `data/processed`) and
`--verify` re-reads the manifests afterwards, failing with exit status 1 if a
near-duplicate group ended up on two nodes or in two of train/val/test.
This regenerates `iid/` and `non_iid_label/` group-aware and adds
`dirichlet_<alpha>/` (per-class Dirichlet label skew) and `<split>_sub<frac>/`
(train split reduced per node) partitions.  `data/processed/split_stats.json`
holds the per-node class-count tables.

### R2. Full metric set per client and per round
Clients now return accuracy, balanced accuracy, precision, recall
(sensitivity), specificity, F1, macro-F1, MCC, ROC-AUC and the confusion
matrix every round, together with fit/eval wall-clock time and model payload
bytes.  The server writes per-client rows, the weighted-global aggregate and
pooled (summed-confusion-matrix) metrics to `results.json` (`rounds` key;
all previous keys are unchanged).  `train_local.py` / `train_centralized.py`
record the same metrics per epoch and on the final test set.

### R3. Revision experiment matrix
```bash
python3 scripts/print_revision_commands.py            # every run, resumable
python3 scripts/print_revision_commands.py --block mu_grid --format bash
```
The `revision:` block of `configs/experiment_matrix.yaml` defines: 5 seeds for
FedAvg vs FedProx(μ=0.01) on IID / non-IID; Dirichlet α ∈ {0.1, 0.5, 1.0};
subsample fractions {0.05, 0.01}; a 10-round FedBN run; μ ∈ {0.001, 0.01,
0.05, 0.1, 0.5}; local epochs {1, 2, 5}; learning rates {1e-4, 1e-3}; and the
matching centralized / local-only baselines.  Output directories follow
`results/rev_<dist>_<strategy>[_ep<E>][_lr<LR>][_r<R>]_seed<S>` and are parsed
by `scripts/analyze_results.py`.

### R4. Unit tests (CPU, synthetic data)
```bash
python3 -m pytest tests -q
```
Covers the metric implementation (against scikit-learn), near-duplicate
clustering and leakage audit, splitter behaviour (bit-identity with the v1
partition, group integrity, Dirichlet determinism, subsampling), the FL
client/server metric pipeline (including a real two-client Flower run on
localhost), both baseline scripts, the analysis script on old and new result
files, and the experiment matrix.

## Expected Variance

Due to stochastic training and hardware-level measurements:

- **Accuracy metrics**: ±0.5-1.5% across seeds (3 seeds per config)
- **Energy measurements**: ±5-10% due to tegrastats sampling and thermal conditions
- **Latency**: run-to-run variation from OS scheduling and thermal state; the wired link removes
  the WiFi jitter of the v1 setup, so v1 and revision timings are not comparable
- **Communication**: Deterministic (model size is fixed at 6.1 MB)

All results include 95% confidence intervals computed from 3 independent runs.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM during training | Use `batch_size=8` and `pin_memory=False` |
| RealSense camera not detected | Check USB3 connection, run `rs-enumerate-devices` |
| ZED camera not detected | Kill ZED background processes, check USB3 |
| FL connection fails | Verify IPs with `ping`, check firewall: `sudo ufw disable` |
| Node B GPU memory crash | `pkill -f python3`, wait, restart with `batch_size=8` |
| numpy error | Ensure numpy==1.26.4, not 2.x |
| tegrastats permission | Run with `sudo` or add user to appropriate group |

## Leave-one-scene-out cross-sensor evaluation

*Added for the NCAA revision (Reviewer 3).* The original cross-camera results
split frames of the **same** five scenes between training and testing, so part
of the reported cross-camera accuracy may reflect scene similarity rather than
sensor generalisation. `scripts/cross_sensor_loso.py` adds the
scene-independent protocol: one fold per scene, training on the remaining
scenes and evaluating on the held-out scene only.

Per fold it evaluates twice:

| Split | Evaluated on | Question it answers |
|-------|--------------|---------------------|
| `cross_camera` | `--test_nodes` × held-out scene | unseen camera *and* unseen scene |
| `same_camera` | `--train_nodes` × held-out scene | unseen scene, same camera (control) |

### Where the scene/class labels come from

The custom capture writer (`src/data/realsense_capture.py`) stores only
`frame_id`, `timestamp_ms` and the depth range in `{id}_meta.json`, so the
scene annotation has to be supplied explicitly. `load_frame_index()` in
`src/data/custom_dataset.py` accepts exactly three sources, in this order:

1. `"scene"` and `"label"` keys inside each `{id}_meta.json`;
2. a CSV with columns `id,scene,label` (plus an optional `node` column when
   frame ids repeat across nodes), passed with `--labels_csv`;
3. frame ids that follow `<scene>_<label>_<n>`, e.g. `lab_fire_00017` or
   `kitchen_no_fire_004`.

If no source annotates *every* frame the run aborts with a report of what was
found (which sources resolved how many frames, the `meta.json` keys present,
the first unresolved frame ids). Nothing is ever guessed silently — a
leave-one-scene-out evaluation is only meaningful with a real scene annotation.
Label names map to class indices with the repository convention
(`No_Fire = 0`, `Fire = 1`); any other vocabulary is mapped in sorted order.

### Commands

```bash
# Cross-camera, manufacturing variance (D435if -> D435i), RGB-D,
# plus the old frame-level random split on the same frames for comparison
python3 scripts/cross_sensor_loso.py \
    --data_dir data/raw/custom --labels_csv data/raw/custom/labels.csv \
    --train_nodes node_a --test_nodes node_b \
    --modality rgb_d --epochs 15 --batch_size 8 --lr 1e-3 --seed 42 \
    --img_size 224 --pooled_random_split \
    --output_dir results/loso_a_to_b_rgb_d_seed42

# Cross-technology: both RealSense nodes -> ZED node
python3 scripts/cross_sensor_loso.py \
    --data_dir data/raw/custom --labels_csv data/raw/custom/labels.csv \
    --train_nodes node_a node_b --test_nodes node_c \
    --modality rgb_d --epochs 15 --batch_size 8 --lr 1e-3 --seed 42 \
    --pooled_random_split \
    --output_dir results/loso_ab_to_c_rgb_d_seed42

# Same-camera control (scene-independent, no sensor shift), RGB+D+IR
python3 scripts/cross_sensor_loso.py \
    --data_dir data/raw/custom --labels_csv data/raw/custom/labels.csv \
    --train_nodes node_a --same_node \
    --modality rgb_d_ir --epochs 15 --batch_size 8 --lr 1e-3 --seed 42 \
    --output_dir results/loso_a_same_rgb_d_ir_seed42

# Restrict to a subset of scenes / inspect the capture tree first
python3 -m src.data.custom_dataset --data_dir data/raw/custom --modality rgb_d
python3 scripts/cross_sensor_loso.py ... --scenes kitchen lab corridor
```

Flags: `--modality {rgb,depth,ir,rgb_d,rgb_d_ir}` (channel counts 3/1/1/4/5,
matching `create_model(in_channels=...)`; ImageNet weights are reused for the
RGB channels only), `--max_depth_m 10` (depth is read as 16-bit millimetres,
converted to metres, clipped and normalised with `configs/model_config.yaml`'s
`normalize_depth`), `--no_pretrained` and `--img_size 16` for the CPU unit
tests, `--test_frac` for the random-split baseline, `--no_record_ids` to keep
`results.json` small. The ZED node has no IR stream, so `--modality ir` /
`rgb_d_ir` drops its frames and prints how many were dropped per node.

### Output

`<output_dir>/results.json` with `experiment: "cross_sensor_loso"`,
`results_schema_version: 2`, and:

- `folds[]` — one entry per scene: `held_out_scene`, `train_scenes`, `n_train`,
  `train_ids`, per-epoch `history`, `train_time_s` / `eval_time_s` /
  `fold_time_s`, and `eval.{cross_camera,same_camera}` each carrying the full
  `src/evaluation/metrics` set (accuracy, balanced accuracy, precision, recall,
  specificity, F1, macro-F1, MCC, ROC-AUC, confusion matrix, per-class support)
  plus the evaluated frame ids;
- `summary` — `mean`, `std` (sample, ddof=1), `min`, `max` and the per-fold
  `values` of every metric, separately for `cross_camera` and `same_camera`,
  plus timing;
- `pooled_random_split` (with `--pooled_random_split`) — the same model/schedule
  trained on a frame-level random split (stratified by node and label,
  `--test_frac`, scenes shared between train and test), so the paper can put
  the old random-split number next to the LOSO number in one table;
- `label_sources` — how many frames were annotated from `meta.json` /
  `labels.csv` / the filename pattern.

Unit tests (synthetic 16-px capture tree, CPU, no pretrained weights):

```bash
python3 -m pytest tests/test_loso.py -q
```
