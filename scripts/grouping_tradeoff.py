#!/usr/bin/env python
"""What a coarser near-duplicate grouping buys, and what it costs.

The shipped partition groups FLAME frames by dHash at Hamming distance
:math:`\\tau = 8` (``data/splits``, ``analysis/leakage/groups.json``).  Held-out
images just above that threshold still have a close training neighbour
(``analysis/leakage/clean_subset/nearest_train_distance.csv``), so the obvious
question is whether merging groups more aggressively would remove the residual.

This script answers it by rebuilding the whole partition under coarser
groupings -- the connected components of the *union* of the dHash edges at
:math:`\\tau = 8` and the pHash edges at ``t`` -- and measuring the three
quantities that the choice trades off against each other:

``residual``
    held-out images whose nearest training image (any node) sits in the
    dHash 9--10 band, and, as an independent second measure, within pHash 8.
``balance``
    the smallest held-out set any client is left with, and the node
    image-count spread of the IID partition.
``dominance``
    the largest share a single sequence takes of any per-node held-out class
    set -- the quantity behind the "79 % of node C's validation Fire images"
    limitation.

Nothing here writes to ``data/splits`` or ``data/processed``: the candidate
partitions live in memory only.  The inputs are the committed hashes
(``analysis/leakage/groups.csv``, ``analysis/leakage/phash.csv.gz``), so the
script needs neither the raw images nor a re-hash.

**On ordering.** ``data_splitter.find_images`` discovers images with ``os.walk``,
whose order is not reproducible across machines, which is why ``data/splits`` is
replayed rather than regenerated (see ``data/splits/README.md``).  The hash
tables are sorted by path, so every row here is rebuilt from the sorted order.
``--check-shipped`` measures the shipped manifests as well, and the ``tau=8``
row comes out identical to them on every metric -- the partition that the
shipped manifests encode is reproduced exactly, so the differences between the
rows are attributable to the grouping alone.  Keep ``--check-shipped`` in the
loop when this is re-run: if that equality ever breaks, the comparison has
silently become one between two different partitions.

Usage::

    python scripts/grouping_tradeoff.py
    python scripts/grouping_tradeoff.py --phash_thresholds 6 4 3 --check-shipped
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_flame_leakage import cluster_hash_columns  # noqa: E402
from scripts.clean_subset import nearest  # noqa: E402
from src.data import data_splitter as ds  # noqa: E402

GROUPS_CSV = "analysis/leakage/groups.csv"
PHASH_CSV = "analysis/leakage/phash.csv.gz"
SPLITS_DIR = "data/splits"

# The parameters the shipped partition was produced with (data/splits/README.md).
SEED = 42
N_NODES = 3
DIRICHLET_ALPHAS = (0.1, 0.5, 1.0)
DIRICHLET_MIN_SIZE = 200
PARTITIONS = ("iid", "non_iid_label", "dirichlet_0.1", "dirichlet_0.5", "dirichlet_1")

# A per-node held-out class set smaller than this is too small for a "share of
# the set" number to mean anything; it is reported through `min_heldout` instead.
MIN_SET_FOR_DOMINANCE = 50


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def load_inputs(groups_csv: str, phash_csv: str):
    """``(paths, labels, dhash, phash)`` for all 47,992 images, sorted by path."""
    g = pd.read_csv(groups_csv)
    p = pd.read_csv(phash_csv).set_index("path")
    if len(p) != len(g):
        raise SystemExit("%s and %s cover different image sets" % (groups_csv, phash_csv))
    g = g.sort_values("path").reset_index(drop=True)
    dh = np.array([np.uint64(int(h, 16)) for h in g.hash_hex], dtype=np.uint64)
    ph = np.array([np.uint64(int(h, 16)) for h in p.phash_hex.reindex(g.path)], dtype=np.uint64)
    return list(g.path), list(g.label), dh, ph


def grouping(dh: np.ndarray, ph: np.ndarray, tau: int, phash_t: int | None) -> np.ndarray:
    """Connected components of the dHash edges at ``tau``, optionally unioned
    with the pHash edges at ``phash_t``.  ``phash_t=None`` is the shipped rule."""
    columns = {"dhash": dh}
    thresholds = {"dhash": tau}
    if phash_t is not None:
        columns["phash"] = ph
        thresholds["phash"] = phash_t
    return cluster_hash_columns(columns, thresholds)


# --------------------------------------------------------------------------- #
# rebuilding the partitions under one grouping
# --------------------------------------------------------------------------- #
def build_partitions(paths, labels, group_ids) -> dict:
    """Rebuild all five partitions in memory.  Returns ``{name: DataFrame}``.

    The ``random`` calls are issued in exactly the order of
    ``data_splitter.main`` -- IID, then label skew, then the Dirichlet alphas --
    because the global RNG state carries from one partition to the next.
    """
    group_map = {path: int(gid) for path, gid in zip(paths, group_ids)}
    fire = [p for p, lab in zip(paths, labels) if lab == "Fire"]
    nofire = [p for p, lab in zip(paths, labels) if lab != "Fire"]

    fire_units, nofire_units, gid_of, _info = ds.build_units(fire, nofire, ".", group_map)
    grouped = ds.is_grouped(fire_units, nofire_units)
    node_names = ds.NODE_NAMES[:N_NODES]

    random.seed(SEED)
    out = {}

    rf, rn = list(fire_units), list(nofire_units)
    random.shuffle(rf)
    random.shuffle(rn)
    nodes = ds.partition_iid(rf, rn, node_names, grouped=grouped, seed=SEED, gid_of=gid_of)
    out["iid"] = _tvt_frame(ds.partition_to_tvt(nodes, SEED, grouped=grouped, gid_of=gid_of),
                            gid_of)

    random.shuffle(rf)
    random.shuffle(rn)
    nodes = ds.partition_non_iid_label(rf, rn, node_names, grouped=grouped, seed=SEED,
                                       gid_of=gid_of)
    out["non_iid_label"] = _tvt_frame(
        ds.partition_to_tvt(nodes, SEED, grouped=grouped, gid_of=gid_of), gid_of)

    for alpha in DIRICHLET_ALPHAS:
        nodes, _proportions = ds.partition_dirichlet(
            fire_units, nofire_units, node_names, alpha, SEED,
            min_size=DIRICHLET_MIN_SIZE, grouped=grouped, gid_of=gid_of)
        out["dirichlet_{:g}".format(alpha)] = _tvt_frame(
            ds.partition_to_tvt(nodes, SEED, grouped=grouped, gid_of=gid_of), gid_of)

    return out


def _tvt_frame(tvt, gid_of) -> pd.DataFrame:
    """Flatten the nested tvt structure exactly as ``write_manifest`` walks it."""
    rows = []
    for node, per_class in tvt.items():
        for sp in ("train", "val", "test"):
            for key, label in (("fire", "Fire"), ("nofire", "No_Fire")):
                for path in ds.flatten(per_class[key][sp]):
                    rows.append((node, sp, label, path, gid_of.get(path, "singleton")))
    return pd.DataFrame(rows, columns=["node", "split", "label", "path", "group_id"])


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def measure(manifest: pd.DataFrame, index: dict, dh: np.ndarray, ph: np.ndarray) -> dict:
    """Residual, balance and dominance for one partition."""
    train = manifest[manifest.split == "train"]
    held = manifest[manifest.split != "train"]

    ti = np.array([index[p] for p in train.path], dtype=np.int64)
    hi = np.array([index[p] for p in held.path], dtype=np.int64)
    d = nearest(dh[hi], dh[ti])
    q = nearest(ph[hi], ph[ti])

    # balance: the smallest held-out set any client is left with
    sizes = held.groupby(["node", "split"]).size()
    node_totals = manifest.groupby("node").size()

    # dominance: how much of a per-node held-out class set one sequence takes.
    # The maximum saturates at 100 % under several groupings, so the count of
    # sets that a single sequence almost entirely covers is what discriminates.
    best_share, best_where = 0.0, ""
    n_sets, n_ge90 = 0, 0
    for (node, split, label), sub in held.groupby(["node", "split", "label"]):
        if len(sub) < MIN_SET_FOR_DOMINANCE:
            continue
        n_sets += 1
        top = sub.group_id.value_counts()
        share = float(top.iloc[0]) / len(sub)
        if share >= 0.90:
            n_ge90 += 1
        if share > best_share:
            best_share = share
            best_where = "%s/%s/%s g%s" % (node, split, label, top.index[0])

    return dict(
        n_heldout=int(len(held)),
        residual_dhash_9_10=int(((d >= 9) & (d <= 10)).sum()),
        residual_dhash_le10=int((d <= 10).sum()),
        residual_phash_le8=int((q <= 8).sum()),
        min_heldout_per_client=int(sizes.min()),
        node_images_max_over_min=round(float(node_totals.max()) / float(node_totals.min()), 2),
        n_heldout_sets=n_sets,
        n_sets_top_seq_ge90=n_ge90,
        max_sequence_share=round(best_share, 4),
        max_sequence_where=best_where,
    )


def candidate_rows(paths, labels, dh, ph, phash_thresholds) -> pd.DataFrame:
    index = {p: i for i, p in enumerate(paths)}
    rows = []
    for phash_t in [None] + list(phash_thresholds):
        name = "tau=8" if phash_t is None else "tau=8 + pHash<=%d" % phash_t
        gids = grouping(dh, ph, 8, phash_t)
        sizes = pd.Series(gids).value_counts()
        print("\n=== %s: %d groups, %d non-trivial, largest %d images ==="
              % (name, len(sizes), int((sizes > 1).sum()), int(sizes.iloc[0])))
        parts = build_partitions(paths, labels, gids)
        for part_name in PARTITIONS:
            m = measure(parts[part_name], index, dh, ph)
            m.update(grouping=name, phash_threshold=(-1 if phash_t is None else phash_t),
                     partition=part_name, n_groups=int(len(sizes)),
                     n_nontrivial_groups=int((sizes > 1).sum()),
                     largest_group=int(sizes.iloc[0]))
            rows.append(m)
            print("  %-16s residual(dHash 9-10)=%4d  pHash<=8=%4d  "
                  "min held-out/client=%4d  sets>=90%% one seq=%d/%d  max=%.1f%% (%s)"
                  % (part_name, m["residual_dhash_9_10"], m["residual_phash_le8"],
                     m["min_heldout_per_client"], m["n_sets_top_seq_ge90"],
                     m["n_heldout_sets"], 100 * m["max_sequence_share"],
                     m["max_sequence_where"]))
    cols = ["grouping", "phash_threshold", "n_groups", "n_nontrivial_groups", "largest_group",
            "partition", "n_heldout", "residual_dhash_9_10", "residual_dhash_le10",
            "residual_phash_le8", "min_heldout_per_client", "node_images_max_over_min",
            "n_heldout_sets", "n_sets_top_seq_ge90", "max_sequence_share", "max_sequence_where"]
    return pd.DataFrame(rows)[cols]


def shipped_rows(splits_dir, paths, dh, ph) -> pd.DataFrame:
    """The same metrics on the shipped manifests (os.walk ordering)."""
    index = {p: i for i, p in enumerate(paths)}
    rows = []
    for part_name in PARTITIONS:
        m = pd.read_csv(os.path.join(splits_dir, part_name + ".csv.gz"), dtype={"group_id": str})
        r = measure(m, index, dh, ph)
        r.update(grouping="tau=8 (shipped)", phash_threshold=-1, partition=part_name,
                 n_groups=-1, n_nontrivial_groups=-1, largest_group=-1)
        rows.append(r)
        print("  %-16s residual(dHash 9-10)=%4d  pHash<=8=%4d  min held-out/client=%4d  "
              "max seq share=%.1f%% (%s)"
              % (part_name, r["residual_dhash_9_10"], r["residual_phash_le8"],
                 r["min_heldout_per_client"], 100 * r["max_sequence_share"],
                 r["max_sequence_where"]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
def latex(table: pd.DataFrame) -> str:
    lines = [
        "% generated by scripts/grouping_tradeoff.py -- do not edit by hand",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Effect of a Coarser Near-Duplicate Grouping on the Residual, the Balance "
        r"and the Sequence Dominance of the Partitions. Ranges are over the Five Partitions "
        r"(IID, Label Skew, Three Dirichlet Skews)}",
        r"\label{tab:grouping_tradeoff}",
        r"\small",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"\textbf{Grouping} & \textbf{Groups} & \textbf{Largest} & \textbf{dHash 9--10} "
        r"& \textbf{Min held-out} & \textbf{Sets $\ge$90\%} \\",
        r" & (non-triv.) & group & residual & per client & one sequence \\",
        r"\midrule",
    ]
    for name, sub in table.groupby("grouping", sort=False):
        res = sub.residual_dhash_9_10
        label = name.replace("tau=8", r"$\tau{=}8$").replace("<=", r"$\le$").replace("_", r"\_")
        lines.append(r"%s & %d & %d & %d--%d & %d & %d/%d \\" % (
            label, int(sub.n_nontrivial_groups.iloc[0]), int(sub.largest_group.iloc[0]),
            int(res.min()), int(res.max()), int(sub.min_heldout_per_client.min()),
            int(sub.n_sets_top_seq_ge90.sum()), int(sub.n_heldout_sets.sum())))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\footnotesize A grouping is the connected components of the dHash edges at "
        r"$\tau{=}8$, unioned with the pHash edges at the stated threshold; groups only ever "
        r"merge. \textbf{dHash 9--10 residual}: held-out images whose nearest training image "
        r"(any node) falls in that band, summed over validation and test, range over the five "
        r"partitions. \textbf{Min held-out per client}: the smallest validation or test set any "
        r"client is left with. \textbf{Sets $\ge$90\% one sequence}: per-node held-out class "
        r"sets of at least 50 images in which a single sequence supplies 90\% or more of the "
        r"images, out of all such sets in the five partitions. Every row is rebuilt by the same "
        r"code path, and the $\tau{=}8$ row reproduces the shipped partition exactly "
        r"(\texttt{scripts/grouping\_tradeoff.py --check-shipped}).",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--groups_csv", default=GROUPS_CSV)
    ap.add_argument("--phash_csv", default=PHASH_CSV)
    ap.add_argument("--splits_dir", default=SPLITS_DIR)
    ap.add_argument("--phash_thresholds", type=int, nargs="+", default=[6, 4, 3],
                    help="pHash thresholds to union with the dHash edges (default 6 4 3)")
    ap.add_argument("--out_csv", default="analysis/leakage/grouping_tradeoff.csv")
    # written next to the other leakage artifacts; scripts/export_latex_tables.py
    # copies it into paper/tables/ so that rule 4 keeps one pipeline into the paper.
    ap.add_argument("--out_tex", default="analysis/leakage/grouping_tradeoff.tex")
    ap.add_argument("--check-shipped", dest="check_shipped", action="store_true",
                    help="also measure the shipped manifests, to show the ordering effect")
    args = ap.parse_args(argv)

    paths, labels, dh, ph = load_inputs(args.groups_csv, args.phash_csv)
    print("%d images, %d Fire" % (len(paths), sum(1 for x in labels if x == "Fire")))

    table = candidate_rows(paths, labels, dh, ph, args.phash_thresholds)

    if args.check_shipped:
        print("\n=== tau=8, shipped manifests (os.walk ordering) ===")
        table = pd.concat([table, shipped_rows(args.splits_dir, paths, dh, ph)],
                          ignore_index=True)[table.columns]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    table.to_csv(args.out_csv, index=False, lineterminator="\n")
    os.makedirs(os.path.dirname(args.out_tex), exist_ok=True)
    with open(args.out_tex, "w", newline="\n") as f:
        f.write(latex(table[table.grouping != "tau=8 (shipped)"]))
    print("\nwrote %s and %s" % (args.out_csv, args.out_tex))


if __name__ == "__main__":
    main()
