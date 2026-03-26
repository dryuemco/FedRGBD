# FedRGBD — Data

## Directory Structure

```
data/
├── raw/
│   ├── public/          # Public dataset (downloaded)
│   └── custom/          # Custom RealSense captures
│       ├── node_a/      # D435i captures (RGB + Depth + IR)
│       └── node_b/      # D455 captures (RGB + Depth + IR)
├── processed/           # Preprocessed and split data
│   ├── iid/             # IID split
│   └── non_iid/         # Non-IID split (label skew)
└── README.md
```

## Raw Data

Raw data files are NOT tracked in git (too large). To reproduce experiments:

### Public Dataset
The exact public dataset will be confirmed on Day 4. Check `configs/data_config.yaml` for the finalized choice.

### Custom RealSense Data
Custom data is captured using `src/data/realsense_capture.py` on each Jetson node.
Each capture produces synchronized RGB, Depth, and IR frames.

## Data Format

Each sample is stored as:
- `{id}_rgb.png` — RGB image (1920×1080, resized to 224×224 during training)
- `{id}_depth.png` — Depth map (1280×720, 16-bit, resized to 224×224)
- `{id}_ir.png` — IR image (1280×720, 8-bit, resized to 224×224)
- `{id}_meta.json` — Metadata (timestamp, camera model, intrinsics, IMU data)
