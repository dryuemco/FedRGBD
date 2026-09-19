# FedRGBD: Multimodal Federated Learning on Edge Sensor Nodes for Visual Anomaly Detection Using RGB-D Cameras

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.13-green.svg)](https://flower.ai/)

## Overview

FedRGBD is the first empirical study of **multimodal federated learning** using real RGB-D cameras on physical edge hardware. We deploy three NVIDIA Jetson Orin Nano Super devices, each equipped with a heterogeneous depth camera (two Intel RealSense variants and one Stereolabs ZED 2i), to investigate how sensor heterogeneity, data distribution, and FL strategy affect model convergence, accuracy, and resource consumption.

## Key Contributions

1. **First multimodal FL study with real depth cameras** — physical edge nodes with real sensors, not simulated virtual clients
2. **Three-level sensor heterogeneity as natural non-IID** — manufacturing variance (D435i vs D435if), cross-technology (RealSense vs ZED 2i), and client scaling (2-node vs 3-node)
3. **Proximal term trade-off on real hardware** — FedProx μ=0.01 recovers 41.1% accuracy in round 1 under non-IID; μ=0.1 causes over-regularization
4. **Real hardware resource profiling** — per-round energy (Wh), latency (s), communication (KB) on Jetson Orin Nano Super
5. **Open-source reproducible framework** — complete code, configs, and instructions

## Hardware

| Equipment | Model | Role |
|-----------|-------|------|
| Edge Node A | NVIDIA Jetson Orin Nano Super 8GB | FL Client #1 + FL Server |
| Edge Node B | NVIDIA Jetson Orin Nano Super 8GB | FL Client #2 |
| Edge Node C | NVIDIA Jetson Orin Nano Super 8GB | FL Client #3 |
| Camera A | Intel RealSense D435if (active IR stereo) | Sensor Node A |
| Camera B | Intel RealSense D435i (active IR stereo) | Sensor Node B |
| Camera C | Stereolabs ZED 2i (passive stereo + neural depth) | Sensor Node C |
| Network | WiFi (IEEE 802.11ac) | FL communication |

## Software Stack

- JetPack 6.2 (CUDA 12.6)
- PyTorch 2.5.0a0 (NVIDIA Jetson wheel)
- torchvision 0.20.0 (built from source)
- Flower 1.13.1
- librealsense 2.55.1 (RSUSB backend)
- ZED SDK 5.2.3 (Node C)
- Python 3.10 (virtual environment)

## Current Results (seed=42)

| Config | Strategy | R1 Acc | R3 Acc | Time |
|--------|----------|--------|--------|------|
| 2-Node IID | FedAvg | 98.79% | 99.68% | 61 min |
| 2-Node Non-IID | FedAvg | 78.36% | 99.85% | 81 min |
| 3-Node IID | FedAvg | 85.45% | 99.75% | 76 min |
| 3-Node Non-IID | FedAvg | 53.66% | 99.10% | 103 min |
| 3-Node Non-IID | FedProx μ=0.01 | 94.79% | **99.49%** | 168 min |
| 3-Node Non-IID | FedProx μ=0.1 | **96.91%** | 98.35% | 155 min |

## Quick Start

### 1. Jetson Setup
```bash
git clone https://github.com/dryuemco/FedRGBD.git
cd FedRGBD
chmod +x setup_jetson.sh
./setup_jetson.sh
```

### 2. Activate Environment
```bash
source ~/fedrgbd_venv/bin/activate
cd ~/FedRGBD
```

### 3. Prepare Dataset
```bash
# Download FLAME dataset from Kaggle to data/raw/flame_dataset/
python3 src/data/data_splitter.py --data_dir data/raw/flame_dataset --output_dir data/processed --nodes 3 --clean
```
`--clean` deletes each `data/processed/<split>/` tree before rewriting it and is
**required** whenever `data/processed` already holds a partition: without it the
files of the previous partition stay behind and silently leak training images
into val/test.

### 4. Run FL Experiment
```bash
# Server (Node A — 192.168.1.10)
python3 src/fl/server.py --strategy fedavg --rounds 3 --seed 42 --output_dir results/3node_iid_fedavg_seed42

# Client (Node A)
python3 src/fl/client.py --server 192.168.1.10:8080 --data_dir data/processed/iid/node_a --batch_size 8 --seed 42

# Client (Node B — 192.168.1.7)
python3 src/fl/client.py --server 192.168.1.10:8080 --data_dir data/processed/iid/node_b --batch_size 8 --seed 42

# Client (Node C — 192.168.1.6)
python3 src/fl/client.py --server 192.168.1.10:8080 --data_dir data/processed/iid/node_c --batch_size 8 --seed 42
```

### 5. Generate Figures
```bash
python3 scripts/generate_plots.py                 # paper v1 figures (hard-coded numbers)
python3 scripts/analyze_results.py --results_dir results --output_dir analysis   # revision: tables + plots from results/**/results.json
```

## NCAA Revision Additions (branch `revision-ncaa`)

Everything below is hardware-independent code added for the major revision of
NCAA-D-26-02211 (see `docs/REVISION_CHANGES.md` for the full change list and
the mapping to reviewer comments).  The model (MobileNetV3-Small) is unchanged
and no experiment results were modified.

### Near-duplicate / leakage audit and sequence-level splits
FLAME frames are video frames; a random image-level split puts near-copies of
test images into training.  `scripts/analyze_flame_leakage.py` hashes every
image (dHash / pHash / aHash, 64 bit), clusters near-duplicates within a
Hamming threshold (multi-index hashing, no O(N²) scan) and audits an existing
`data/processed` tree for val/test → train leakage.  Its `groups.json` feeds
`src/data/data_splitter.py --group_file`, which then assigns whole groups so
no near-duplicate group is split across nodes or across train/val/test.
```bash
python3 scripts/analyze_flame_leakage.py --data_dir data/raw/flame_dataset \
    --processed_dir data/processed --output_dir analysis/leakage --threshold 8 --workers 4
python3 src/data/data_splitter.py --data_dir data/raw/flame_dataset --output_dir data/processed \
    --nodes 3 --seed 42 --group_file analysis/leakage/groups.json \
    --dirichlet_alpha 0.1 0.5 1.0 --subsample_frac 0.05 0.01 --clean --verify
```
`--clean` is required here because this re-splits an existing `data/processed`;
`--verify` re-reads the manifests afterwards and fails (exit 1) if any
near-duplicate group ended up on two nodes or in two of train/val/test.

The whole P0 chain (dataset check/Kaggle download -> v1 audit -> raw audit with
`--sweep 4 6 8 10 12 --sequence_heuristic --examples 20` -> decision summary ->
group-safe split with `--clean --verify` -> zero-leakage re-audit + manifest md5
table) runs resumably with one command:
```bash
python3 scripts/run_p0_leakage_and_split.py        # --dry_run prints the commands, --force redoes finished steps
```
It writes `analysis/leakage/P0_SUMMARY.md` and exits non-zero on `VERIFY FAIL` or
any non-zero post-split leak rate.  Downloading needs `pip install kaggle` plus a
Kaggle API token (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`);
without one the script prints the manual download steps and exits 2.

The leakage script also accepts `--hash both` (cluster the union of the dHash
edges at `--threshold` and the pHash edges at `--phash_threshold`),
`--sweep 4 6 8 10 12` (threshold-sensitivity table in
`leakage_report.json["threshold_sweep"]`), `--sequence_heuristic` (how often
consecutive frame numbers in the file names fall into one group),
`--examples N` (`example_groups.txt`) and reports byte-identical duplicates
(`exact_duplicate_files_md5`, computed in the same pass; `--no_md5` to skip).
`--group_file` additionally reads the inverted `{"groups": {path: gid}}` layout
with absolute paths.

The splitter's `--clean` (delete each `<output_dir>/<split>/` tree before rewriting
it) is required for every re-split of an existing `data/processed`, and `--verify`
re-reads the manifests afterwards.  Both scripts merge the extras of the authors'
own versions: `--hash both`, `--sweep`, the MD5 exact-duplicate count,
`--sequence_heuristic`, `example_groups.txt` (`--examples`) and the inverted
`{path: gid}` group-file layout in the leakage script, and `--verify`
(their `verify_no_leak()`) plus the per-node `train/val/test` summary line in the
splitter — see `docs/REVISION_CHANGES.md`.

### Dirichlet label skew and low-data regimes
`--dirichlet_alpha 0.1 0.5 1.0` produces `data/processed/dirichlet_<alpha>/`
(per-class Dirichlet(alpha) proportions over the 3 nodes, seeded, group-aware,
per-node class-count table written to `split_stats.json`).
`--subsample_frac 0.05 0.01` produces `<split>_sub<frac>/` variants in which
each node's *train* split is reduced to that fraction (stratified, group-aware);
val/test stay full so results remain comparable.  See `data/README.md`.

### Metrics beyond accuracy
`src/evaluation/metrics.py` computes accuracy, balanced accuracy, precision,
recall (sensitivity), specificity, F1 / macro-F1, MCC, ROC-AUC and the
confusion matrix from logits + labels (pure NumPy, validated against
scikit-learn in `tests/test_metrics.py`).  It is wired into

* `src/fl/client.py` – `evaluate()` returns every metric per client per round,
  plus `eval_time_s`, `fit_time_s` and the model payload bytes sent/received;
* `src/fl/server.py` – persists per-client rows, the num-examples-weighted
  global aggregate **and** pooled metrics from the summed confusion matrix for
  every round, plus server elapsed time and cumulative communication bytes;
* `scripts/train_local.py` / `scripts/train_centralized.py` – full metric set
  per epoch (`val_metrics`) and on the final test set (`final_test_metrics`).

`results.json` keeps every previous key (`strategy`, `num_rounds`, `seed`,
`total_time_s`, `losses_distributed`, `metrics_distributed`, …) and adds
`results_schema_version: 2`, `proximal_mu`, `tags`, `client_config`,
`model_payload_bytes`, `total_communication_bytes`, `metrics_distributed_fit`
and a `rounds` list:
```
rounds[i] = {
  "round": r,
  "fit":      {"clients": {node_a: {train_loss, fit_time_s, payload_bytes_up, payload_bytes_down, ...}},
               "aggregate": {...}, "elapsed_s": ...},
  "evaluate": {"clients": {node_a: {accuracy, balanced_accuracy, precision, recall, specificity,
                                    f1, macro_f1, mcc, roc_auc, loss, confusion_matrix, eval_time_s, ...}},
               "aggregate": {accuracy, ..., pooled_accuracy, pooled_f1, ..., cm_0_0, ...}, "elapsed_s": ...},
  "cumulative_communication_bytes": ...
}
```
New server flag: `--tag <dist> ...` stores free-form tags (e.g. `dirichlet_0.1`)
used by the analysis script.  New client flags: `--node_name`, `--eval_split`.

### Statistical analysis and plots
`scripts/analyze_results.py` reads every `results/**/results.json` (old
accuracy-only files and new ones alike) and writes to `analysis/`: per-config
tables (mean ± std, 95 % t-CI over seeds), paired Cohen's d, Wilcoxon and
paired t-tests, Friedman tests, and accuracy-vs-round, accuracy-vs-elapsed-time
and accuracy-vs-cumulative-communication (MB) plots.  For old files time and
communication are *estimated* (flagged) from `total_time_s` and the model size.

### Revision experiment matrix
`configs/experiment_matrix.yaml` → `revision:` block (5 seeds FedAvg vs
FedProx, Dirichlet alphas, subsample fractions, 10-round FedBN, μ grid,
local-epoch and learning-rate sweeps, matching baselines).
`scripts/print_revision_commands.py` expands it into resumable server/client
command lines (`--skip_existing`).

### Tests
CPU-only, tiny synthetic data, no downloads:
```bash
pip install pytest
python3 -m pytest tests -q
```

## Repository Structure

```
FedRGBD/
├── configs/                 # YAML configuration files
├── src/
│   ├── data/               # Data capture, loading, splitting
│   ├── models/             # MobileNetV3 multimodal variants
│   ├── fl/                 # Flower server, client, strategies
│   ├── profiling/          # Energy, latency, communication logging
│   └── evaluation/         # Metrics and statistical tests
├── scripts/                # Experiment runners and plot generation
├── data/                   # Raw and processed data (see data/README.md)
├── results/                # Experiment outputs (JSON)
├── paper/                  # IEEE Sensors Journal manuscript (LaTeX)
└── docs/                   # Setup guides and experiment logs
```

## Experiments

| # | Experiment | Description | Runs |
|---|-----------|-------------|------|
| 1 | FL Strategy Comparison | FedAvg, FedProx, FedBN × IID/Non-IID × 3 seeds | 36 |
| 2 | Modality Ablation | RGB / Depth / IR / RGB+D / RGB+D+IR | 15 |
| 3 | Cross-Sensor Generalization | D435if ↔ D435i ↔ ZED 2i | 27 |
| 4 | Resource Profiling | Per-round energy, latency, communication | — |
| 5 | Network Constraint | WiFi baseline, 10 Mbps, 1 Mbps, 1 Mbps + 5% loss | 12 |

## Known Issues

- `pin_memory=True` causes OOM on Jetson — use `pin_memory=False`
- Node A needs `batch_size=8` when running both server + client
- Node C ZED SDK uses GPU memory — kill background processes before FL training
- `numpy` must be 1.26.4 — numpy 2.x breaks PyTorch on Jetson

## Citation

```bibtex
@article{cogurcu2026fedrgbd,
  title={Multimodal Federated Learning on Edge Sensor Nodes for Visual Anomaly Detection Using RGB-D Cameras},
  author={{\c{C}}o{\u{g}}urcu, Yunus Emre},
  journal={IEEE Sensors Journal},
  year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
