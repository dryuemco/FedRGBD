#!/usr/bin/env python3
"""FedRGBD -- how many training sequences the low-data subsamples keep, per node.

The low-data regime (paper Section "Low-Data Regime") subsamples each node's training
split uniformly over its *frames* (``--subsample_fraction``), stratified per class, and
leaves validation and test untouched.  Whether that removes near-duplicate frames
before it removes sequences is a property of the committed manifests, so it is measured
here rather than assumed: per partition, fraction rho and node, the training frames and
the training sequences (near-duplicate groups) of the full split and of the subsample,
the share of the node's sequences that keep at least one frame, the share of the node's
full training frames that lie in those sequences, and both counts per sequence size
(fewer than 10 frames, 10-99, 100 or more; size = training frames in the full split).
Most of a node's sequences are a few frames each, so the count share alone understates
what the subsample keeps.

Reads ``data/splits/<partition>.csv.gz`` and ``<partition>_sub<rho>.csv.gz`` (never
re-derives a split, CLAUDE.md rule 2); writes ``analysis/lowdata_sequence_coverage.csv``.

    python scripts/lowdata_sequence_coverage.py
"""

import argparse
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = ("node_a", "node_b", "node_c")
#: the partitions and fractions of the paper's low-data regime
PARTITIONS = ("iid", "non_iid_label")
FRACTIONS = ("0.05", "0.01")
#: sequence-size bands (training frames of the sequence in the full split): label, lower, upper
SIZE_BANDS = (("lt10", 1, 9), ("10_99", 10, 99), ("ge100", 100, None))


def coverage(full: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    """Per node: training frames and sequences, full vs subsample."""
    rows = []
    ft, st = full[full["split"] == "train"], sub[sub["split"] == "train"]
    for node in NODES:
        sizes = ft.loc[ft["node"] == node].groupby("group_id").size()
        gf = set(sizes.index)
        gs = set(st.loc[st["node"] == node, "group_id"])
        if not gs <= gf:
            raise ValueError("%s: the subsample holds sequences the full split does not" % node)
        kept = sizes.index.isin(sorted(gs))
        row = {"node": node,
               "train_frames_full": int((ft["node"] == node).sum()),
               "train_frames_sub": int((st["node"] == node).sum()),
               "train_sequences_full": len(gf), "train_sequences_sub": len(gs),
               "sequence_share": round(len(gs) / float(len(gf)), 4),
               "frame_share_kept_sequences": round(sizes[kept].sum() / float(sizes.sum()), 4)}
        for label, lo, hi in SIZE_BANDS:
            band = (sizes >= lo) & ((sizes <= hi) if hi is not None else True)
            row["sequences_full_" + label] = int(band.sum())
            row["sequences_sub_" + label] = int((band & kept).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits_dir", default=os.path.join(REPO, "data", "splits"))
    ap.add_argument("--out", default=os.path.join(REPO, "analysis", "lowdata_sequence_coverage.csv"))
    args = ap.parse_args(argv)
    out = []
    for part in PARTITIONS:
        full = pd.read_csv(os.path.join(args.splits_dir, part + ".csv.gz"), dtype={"group_id": str})
        for rho in FRACTIONS:
            sub = pd.read_csv(os.path.join(args.splits_dir, "%s_sub%s.csv.gz" % (part, rho)),
                              dtype={"group_id": str})
            df = coverage(full, sub)
            df.insert(0, "rho", rho)
            df.insert(0, "partition", part)
            out.append(df)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(args.out, index=False, lineterminator="\n")
    for _, r in res.iterrows():
        print("%-14s rho %-5s %s  frames %5d -> %4d   sequences %3d -> %3d (%.1f %%), "
              "holding %.1f %% of the frames; by size <10 %d/%d, 10-99 %d/%d, >=100 %d/%d"
              % (r["partition"], r["rho"], r["node"], r["train_frames_full"], r["train_frames_sub"],
                 r["train_sequences_full"], r["train_sequences_sub"], 100 * r["sequence_share"],
                 100 * r["frame_share_kept_sequences"],
                 r["sequences_sub_lt10"], r["sequences_full_lt10"],
                 r["sequences_sub_10_99"], r["sequences_full_10_99"],
                 r["sequences_sub_ge100"], r["sequences_full_ge100"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
