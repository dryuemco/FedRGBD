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
python3 src/data/data_splitter.py --data_dir data/raw/flame_dataset --output_dir data/processed --nodes 3
```

### 4. Run FL Experiment
```bash
# Server (Node A — 192.168.1.4)
python3 src/fl/server.py --strategy fedavg --rounds 3 --seed 42 --output_dir results/3node_iid_fedavg_seed42

# Client (Node A)
python3 src/fl/client.py --server 192.168.1.4:8080 --data_dir data/processed/iid/node_a --batch_size 8 --seed 42

# Client (Node B — 192.168.1.5)
python3 src/fl/client.py --server 192.168.1.4:8080 --data_dir data/processed/iid/node_b --batch_size 8 --seed 42

# Client (Node C — 192.168.1.3)
python3 src/fl/client.py --server 192.168.1.4:8080 --data_dir data/processed/iid/node_c --batch_size 8 --seed 42
```

### 5. Generate Figures
```bash
python3 scripts/generate_plots.py
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
