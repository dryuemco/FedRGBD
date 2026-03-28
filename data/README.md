# FedRGBD — Data

## Directory Structure

```
data/
├── raw/
│   ├── flame_dataset/       # FLAME fire/nofire classification dataset
│   │   ├── Fire/            # 30,155 fire images (254×254 JPEG)
│   │   └── No_Fire/         # 17,837 no-fire images (254×254 JPEG)
│   └── custom/              # Custom RGB-D captures (Phase B)
│       ├── node_a/          # D435if captures (RGB + Depth + IR)
│       ├── node_b/          # D435i captures (RGB + Depth + IR)
│       └── node_c/          # ZED 2i captures (RGB + Depth)
├── processed/               # Preprocessed and split data
│   ├── iid/                 # IID split (~16K per node, 62.8% Fire ratio)
│   │   ├── node_a/
│   │   ├── node_b/
│   │   └── node_c/
│   └── non_iid_label/       # Non-IID label skew
│       ├── node_a/          # 80% Fire
│       ├── node_b/          # 88.5% Fire
│       └── node_c/          # 20% Fire
└── README.md
```

## Public Dataset — FLAME

- **Source:** Kaggle (smrutisanchitadas/flame-dataset-fire-classification)
- **Size:** 47,992 images (30,155 Fire + 17,837 No_Fire)
- **Resolution:** 254×254 JPEG
- **Task:** Binary classification (Fire vs No_Fire)

### Download
```bash
# Option 1: Kaggle CLI
kaggle datasets download smrutisanchitadas/flame-dataset-fire-classification
unzip flame-dataset-fire-classification.zip -d data/raw/flame_dataset/

# Option 2: Manual download from Kaggle web interface
# Place Fire/ and No_Fire/ folders in data/raw/flame_dataset/
```

### Create Splits
```bash
python3 src/data/data_splitter.py \
    --data_dir data/raw/flame_dataset \
    --output_dir data/processed \
    --nodes 3 \
    --seed 42
```

## Custom RGB-D Data (Phase B)

Custom data is captured using camera-specific scripts:
```bash
# Node A & B (RealSense)
python3 src/data/realsense_capture.py --output data/raw/custom/node_a --frames 500

# Node C (ZED) — requires ZED SDK
python3 src/data/zed_capture.py --output data/raw/custom/node_c --frames 500
```

Each capture produces synchronized frames:
- `{id}_rgb.png` — RGB image
- `{id}_depth.png` — Depth map (16-bit PNG, mm)
- `{id}_ir.png` — IR image (8-bit, RealSense only)
- `{id}_meta.json` — Timestamp, camera model, intrinsics

## Note

Raw data files are NOT tracked in git (too large). Use the download/capture
instructions above to reproduce the dataset on your setup.
