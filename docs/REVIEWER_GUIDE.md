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
| Network | WiFi router (802.11ac) | 1 | FL communication |

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
# Connect all nodes to same WiFi network
# Assign static IPs: Node A=192.168.1.4, Node B=192.168.1.5, Node C=192.168.1.3
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
python3 src/data/data_splitter.py --data_dir data/raw/flame_dataset --output_dir data/processed --nodes 3
```

### Step 4: Run All Experiments (~120 hours total compute, ~40 hours wall-clock)
```bash
# See scripts/run_experiment.sh for individual experiments
# Example: 3-node Non-IID FedAvg with seeds 42, 123, 456
bash scripts/run_experiment.sh fedavg non_iid 42 123 456
```

### Step 5: Generate Results
```bash
python3 scripts/generate_plots.py     # PDF/PNG figures
```

## Expected Variance

Due to stochastic training and hardware-level measurements:

- **Accuracy metrics**: ±0.5-1.5% across seeds (3 seeds per config)
- **Energy measurements**: ±5-10% due to tegrastats sampling and thermal conditions
- **Latency**: ±10-15% due to WiFi jitter and OS scheduling
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
