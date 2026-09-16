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
│   ├── split_stats.json     # per-split, per-node class-count tables (+ Dirichlet proportions)
│   ├── iid/                 # IID split (~16K per node, 62.8% Fire ratio)
│   │   ├── manifest.csv     # node,split,label,path,group_id for every image
│   │   ├── node_a/
│   │   ├── node_b/
│   │   └── node_c/
│   ├── non_iid_label/       # Non-IID label skew (manual, paper v1)
│   │   ├── node_a/          # 80% Fire
│   │   ├── node_b/          # 88.5% Fire
│   │   └── node_c/          # 20% Fire
│   ├── dirichlet_0.1/       # Dirichlet(alpha) label skew (revision), alpha 0.1 / 0.5 / 1
│   ├── dirichlet_0.5/
│   ├── dirichlet_1/
│   ├── iid_sub0.05/         # low-data variants: train split reduced to 5 % / 1 % per node
│   ├── iid_sub0.01/
│   └── non_iid_label_sub0.05/ ...
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

#### Paper v1 splits (image-level, unchanged)
```bash
python3 src/data/data_splitter.py \
    --data_dir data/raw/flame_dataset \
    --output_dir data/processed \
    --nodes 3 \
    --seed 42
```
Without any of the new flags the partition is bit-identical to the one used in
the first submission (same `random` call sequence for the same seed).

#### Revision: near-duplicate audit and sequence-level (group) splits
FLAME frames come from video, so consecutive frames are near-identical and a
random image-level split leaks near-copies of test images into training.
First cluster near-duplicates with a perceptual hash, then split by *group*:
```bash
# 1. hash every image (dHash, 64 bit), cluster at Hamming distance <= 8,
#    audit the existing data/processed tree for val/test -> train leakage
python3 scripts/analyze_flame_leakage.py \
    --data_dir data/raw/flame_dataset \
    --processed_dir data/processed \
    --output_dir analysis/leakage --threshold 8 --workers 4
#    -> analysis/leakage/groups.json, groups.csv, leakage_report.json

# 2. re-split so that no near-duplicate group is split across nodes or
#    across train/val/test, and add the revision partitions
python3 src/data/data_splitter.py \
    --data_dir data/raw/flame_dataset --output_dir data/processed \
    --nodes 3 --seed 42 \
    --group_file analysis/leakage/groups.json \
    --dirichlet_alpha 0.1 0.5 1.0 \
    --subsample_frac 0.05 0.01
```
Splitter options added in the revision:

| Flag | Effect |
|------|--------|
| `--group_file PATH` | assign whole near-duplicate groups (JSON or CSV from `analyze_flame_leakage.py`); applies to every split type |
| `--dirichlet_alpha A [A ...]` | per-class Dirichlet(alpha) label-skew partition over the nodes -> `dirichlet_<alpha>/`; seeded with `--seed`; `--dirichlet_min_size` (default 10) guarantees a minimum node size by redrawing |
| `--subsample_frac F [F ...]` | after splitting, reduce each node's `train` split (stratified per class, group-aware) to fraction F -> `<split>_sub<F>/`; `--subsample_splits` can also shrink val/test |
| `--skip_base_splits` | only produce Dirichlet / subsample variants |
| `--link_mode {symlink,hardlink,copy}` | symlink (default, Jetson) falls back to copy automatically |

`split_stats.json` contains, for every split, the per-node train/val/test
counts (unchanged keys) plus a `class_counts` table
(`{node: {Fire, No_Fire, total, fire_ratio}}`), the Dirichlet proportions and
the subsample settings.  Every split directory also gets a `manifest.csv`
listing node, split, label, image path and near-duplicate group id.

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
