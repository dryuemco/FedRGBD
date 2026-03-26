# FedRGBD: Multimodal Federated Learning on Edge Sensor Nodes for Visual Anomaly Detection Using RGB-D Cameras

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.x-green.svg)](https://flower.ai/)

## Overview

FedRGBD is the first empirical study of **multimodal federated learning** using real RGB-D cameras on physical edge hardware. We investigate how RGB, Depth, and IR modalities from heterogeneous Intel RealSense cameras (D435i and D455) affect federated learning performance, convergence, and resource consumption on NVIDIA Jetson Orin Nano 8GB edge devices.

## Key Contributions

1. **First multimodal FL study with real depth cameras** — not simulated virtual clients
2. **Sensor heterogeneity as natural non-IID** — D435i vs D455 produce structurally different data
3. **Modality ablation under FL** — systematic per-modality contribution analysis
4. **Real hardware resource profiling** — per-round energy (Wh), latency (ms), communication (KB)
5. **Open-source reproducible framework** — complete code, configs, and instructions

## Hardware Requirements

| Equipment | Model | Role |
|-----------|-------|------|
| Edge Node A | NVIDIA Jetson Orin Nano 8GB | FL Client #1 |
| Edge Node B | NVIDIA Jetson Orin Nano 8GB | FL Client #2 |
| Camera A | Intel RealSense D435i | Sensor Node A |
| Camera B | Intel RealSense D455 | Sensor Node B |
| Training Server | Desktop with RTX 3060+ | Pre-training only |
| Network | Gigabit Ethernet switch | FL communication |

## Quick Start

### 1. Jetson Setup
```bash
# On each Jetson Orin Nano (after flashing JetPack 6.1)
git clone https://github.com/dryuemco/FedRGBD.git
cd FedRGBD
chmod +x setup_jetson.sh
./setup_jetson.sh
```

### 2. Desktop Setup
```bash
# On the training server
git clone https://github.com/dryuemco/FedRGBD.git
cd FedRGBD
chmod +x setup_desktop.sh
./setup_desktop.sh
```

### 3. Verify Installation
```bash
# On Jetson: verify camera
rs-enumerate-devices

# On Jetson: verify CUDA + PyTorch
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# On Jetson: verify Flower
python3 -c "import flwr; print(f'Flower version: {flwr.__version__}')"
```

### 4. Run Experiments
```bash
# See scripts/run_experiment.py for single experiment
# See scripts/run_all_experiments.sh for full matrix
python3 scripts/run_experiment.py --config configs/experiment_matrix.yaml --exp 1
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
├── scripts/                # Experiment runners and utilities
├── notebooks/              # Analysis and visualization notebooks
├── data/                   # Raw and processed data (see data/README.md)
├── results/                # Experiment outputs
├── paper/                  # IEEE Sensors Journal manuscript
└── docs/                   # Setup guides and experiment logs
```

## Experiments

| # | Experiment | Runs |
|---|-----------|------|
| 1 | FL Strategy Comparison (FedAvg, FedProx, FedBN) | 36 |
| 2 | Modality Ablation (RGB / Depth / IR / RGB+D / RGB+D+IR) | 15 |
| 3 | Cross-Sensor Generalization (D435i ↔ D455) | 9 |
| 4 | Resource Profiling (energy, latency, communication) | — |
| 5 | Network Constraint Sensitivity (bandwidth limits) | 12 |

## Citation

```bibtex
@article{cogurcu2026fedrgbd,
  title={Multimodal Federated Learning on Edge Sensor Nodes for Visual Anomaly Detection Using RGB-D Cameras},
  author={Çoğurcu, Yunus Emre},
  journal={IEEE Sensors Journal},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
