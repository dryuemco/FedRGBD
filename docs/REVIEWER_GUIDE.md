# FedRGBD — Reviewer's Guide to Reproducing Results

## Overview

This document provides step-by-step instructions for reviewers to reproduce
the experiments presented in the manuscript. All code, configurations, and
instructions are provided for full reproducibility.

## Hardware Requirements (Exact Setup)

| Component | Model | Quantity |
|-----------|-------|----------|
| Edge device | NVIDIA Jetson Orin Nano 8GB (40 TOPS) | 2 |
| Camera A | Intel RealSense D435i | 1 |
| Camera B | Intel RealSense D455 | 1 |
| Network | Gigabit Ethernet switch + Cat6 cables | 1 |
| Pre-training | Any NVIDIA GPU with 12GB+ VRAM | 1 |

**Note:** Results on different hardware (e.g., Jetson AGX Orin, different cameras)
will differ due to different power profiles, memory constraints, and sensor
characteristics. This is expected and is part of the study's contribution.

## Software Requirements

- JetPack 6.1 (L4T 36.4.0) or JetPack 6.2 (L4T 36.4.3)
- Python 3.10
- PyTorch 2.5.0+ (NVIDIA Jetson wheel)
- Flower 1.13.x
- librealsense 2.55.x

## Reproduction Steps

### Step 1: Hardware Setup (~2 hours)
```bash
# Flash JetPack on both Jetsons using NVIDIA SDK Manager
# Connect cameras: D435i → Node A USB3, D455 → Node B USB3
# Connect Ethernet: Node A ↔ Switch ↔ Node B
```

### Step 2: Software Setup (~1 hour per node)
```bash
git clone https://github.com/<username>/FedRGBD.git
cd FedRGBD
./setup_jetson.sh
```

### Step 3: Data Preparation (~30 minutes)
```bash
# Download public dataset (see configs/data_config.yaml for URL)
python3 src/data/data_splitter.py --config configs/data_config.yaml
```

### Step 4: Run All Experiments (~48-72 hours total)
```bash
# Full experiment matrix
bash scripts/run_all_experiments.sh
```

### Step 5: Generate Results
```bash
python3 scripts/generate_tables.py    # LaTeX tables
python3 scripts/generate_plots.py     # PDF figures
```

## Expected Variance

Due to the stochastic nature of FL training and hardware-level measurements:

- **Accuracy metrics**: ±1-2% across seeds (3 seeds per config)
- **Energy measurements**: ±5-10% due to tegrastats sampling and thermal conditions
- **Latency**: ±10% due to OS scheduling and network jitter
- **Communication**: Should be deterministic (model size is fixed)

All results include 95% confidence intervals computed from 3 independent runs.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM during training | Reduce batch_size to 8 in configs/fl_config.yaml |
| Camera not detected | Check USB3 connection, run `rs-enumerate-devices` |
| FL connection fails | Verify IPs with `ping`, check firewall: `sudo ufw disable` |
| tegrastats permission | Run profiling with `sudo` or add user to appropriate group |
