#!/usr/bin/env python3
"""FedRGBD -- how much of each held-out cell is a single sequence (partition property).

For every partition in data/splits and every (node, split, class) validation/test
cell: number of images, number of sequences (group_id), and the share of the cell
taken by its largest sequence.  Writes analysis/leakage/heldout_dominance.csv.

    python scripts/heldout_dominance.py
"""

import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(REPO, "data", "splits")
OUT = os.path.join(REPO, "analysis", "leakage", "heldout_dominance.csv")


def main():
    rows = []
    for name in sorted(os.listdir(SPLITS)):
        if not name.endswith(".csv.gz"):
            continue
        part = name[:-len(".csv.gz")]
        m = pd.read_csv(os.path.join(SPLITS, name), dtype={"group_id": str})
        for (node, split, label), cell in m[m.split != "train"].groupby(["node", "split", "label"]):
            sizes = cell.group_id.value_counts()
            rows.append(dict(partition=part, node=node, split=split, label=label, n=len(cell),
                             sequences=len(sizes), largest_group=sizes.index[0],
                             largest_share=round(sizes.iloc[0] / len(cell), 4)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, lineterminator="\n")
    print("wrote %s (%d cells)" % (OUT, len(df)))
    print(df[(df.partition == "iid") & (df.node == "node_c")].to_string(index=False))


if __name__ == "__main__":
    main()
