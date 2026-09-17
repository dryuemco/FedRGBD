# `data/splits` — the authoritative FLAME partition

This directory **is** the partition used in the revised manuscript, not a recipe for it.
One gzipped `manifest.csv` per split (`node,split,label,path,group_id` for every image) plus
the `split_stats.json` that describes them. About 1.1 MB for all 15 splits, so it is tracked
in git and travels with the code.

## Why the partition is shipped instead of regenerated

`data_splitter.py` discovers images with `os.walk`, whose order is not guaranteed across
machines or filesystems. Running the splitter independently on each node could therefore
produce *different* partitions, which would break both the federated protocol (nodes training
on each other's evaluation data) and the leakage guarantee. Replaying these manifests uses no
random number generator at all and reproduces the tree byte for byte.

## Rebuild `data/processed` on a node

```bash
# 1. the raw dataset must be in place first (data/README.md)
python scripts/run_p0_leakage_and_split.py --dry_run     # shows the Kaggle instructions

# 2. replay the committed partition
python src/data/data_splitter.py --from_manifest data/splits \
    --data_dir data/raw/flame_dataset --output_dir data/processed \
    --link_mode hardlink --clean --verify
```

`--verify` re-reads the written manifests and asserts that no near-duplicate group spans two
nodes or two of train/val/test. Then compare the manifest MD5s with the "Split digests" table
in `analysis/leakage/P0_SUMMARY.md`: they must match on every node.

## What the splits are

| Split | Description |
|---|---|
| `iid` | equal per-node image count and class ratio (~62.8 % fire) |
| `non_iid_label` | manual label skew: node A 80 % fire, B 88.5 %, C 20 % |
| `dirichlet_0.1`, `dirichlet_0.5`, `dirichlet_1` | Dirichlet label skew, `--dirichlet_min_size 200` |
| `<split>_sub0.05`, `<split>_sub0.01` | low-data regime: training split reduced to 5 % / 1 % at frame level |

Produced with seed 42, three nodes, and the group file
`analysis/leakage/groups.json` (dHash, Hamming threshold 8). Achieved per-node counts are in
`split_stats.json` and in Table `tab:group_counts` of the paper.

## Regenerating this archive

Only if the partition itself changes:

```bash
python src/data/data_splitter.py --export_manifests data/splits --output_dir data/processed
```

The gzip headers carry `mtime=0`, so re-exporting an unchanged partition produces no diff.
