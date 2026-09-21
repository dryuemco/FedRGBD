"""Unit tests for scripts/grouping_tradeoff.py (no dataset, no images)."""

import numpy as np
import pandas as pd
import pytest

from scripts import grouping_tradeoff as gt


def _u64(bits):
    """uint64 with the given bit positions set."""
    v = 0
    for b in bits:
        v |= 1 << b
    return np.uint64(v)


def test_coarser_grouping_only_merges():
    """A pHash union can only merge dHash groups, never split them.

    This is the property the trade-off table relies on: every row is a coarsening
    of the row above it, so the comparison is between nested groupings.
    """
    rng = np.random.default_rng(0)
    dh = rng.integers(0, 2 ** 63, size=60, dtype=np.int64).astype(np.uint64)
    ph = rng.integers(0, 2 ** 63, size=60, dtype=np.int64).astype(np.uint64)
    # make a few pairs close under each hash so there is something to merge
    dh[1] = dh[0]
    ph[3] = ph[2]

    base = gt.grouping(dh, ph, 8, None)
    coarse = gt.grouping(dh, ph, 8, 6)

    # every base group is contained in exactly one coarse group
    for gid in set(base.tolist()):
        members = np.flatnonzero(base == gid)
        assert len(set(coarse[members].tolist())) == 1
    assert len(set(coarse.tolist())) <= len(set(base.tolist()))


def test_measure_residual_bands_and_dominance():
    n_pad = 60
    paths = ["train/a", "train/b"] + ["held/%d" % i for i in range(n_pad)]
    index = {p: i for i, p in enumerate(paths)}
    dh = np.zeros(len(paths), dtype=np.uint64)
    ph = np.zeros(len(paths), dtype=np.uint64)

    # training images sit at 0; give the held-out images known distances to them
    dh[index["held/0"]] = _u64(range(9))    # dHash 9  -> in the 9-10 band
    dh[index["held/1"]] = _u64(range(10))   # dHash 10 -> in the 9-10 band
    dh[index["held/2"]] = _u64(range(11))   # dHash 11 -> outside the band
    for i in range(3, n_pad):
        dh[index["held/%d" % i]] = _u64(range(20))
    ph[index["held/0"]] = _u64(range(8))    # pHash 8 -> counted by residual_phash_le8
    for i in range(1, n_pad):
        ph[index["held/%d" % i]] = _u64(range(30))

    rows = [("node_a", "train", "Fire", "train/a", "g1"),
            ("node_a", "train", "Fire", "train/b", "g1")]
    # 60 held-out Fire images in one node/split, 45 of them from a single sequence
    for i in range(n_pad):
        rows.append(("node_a", "val", "Fire", "held/%d" % i, "g9" if i < 45 else "g%d" % i))
    manifest = pd.DataFrame(rows, columns=["node", "split", "label", "path", "group_id"])

    m = gt.measure(manifest, index, dh, ph)

    assert m["n_heldout"] == n_pad
    assert m["residual_dhash_9_10"] == 2
    assert m["residual_dhash_le10"] == 2
    assert m["residual_phash_le8"] == 1
    assert m["min_heldout_per_client"] == n_pad
    assert m["max_sequence_share"] == pytest.approx(45 / n_pad)
    assert "g9" in m["max_sequence_where"]
    # 45/60 = 75 %, below the 90 % bar, so the set is counted but not flagged
    assert m["n_heldout_sets"] == 1
    assert m["n_sets_top_seq_ge90"] == 0


def test_measure_ignores_tiny_heldout_sets_for_dominance():
    """A 3-image held-out set must not report a 100 % dominance share."""
    paths = ["train/a", "held/0", "held/1", "held/2"]
    index = {p: i for i, p in enumerate(paths)}
    dh = np.zeros(len(paths), dtype=np.uint64)
    ph = np.zeros(len(paths), dtype=np.uint64)
    for p in ("held/0", "held/1", "held/2"):
        dh[index[p]] = _u64(range(20))
        ph[index[p]] = _u64(range(30))

    rows = [("node_a", "train", "Fire", "train/a", "g1")]
    rows += [("node_a", "val", "Fire", "held/%d" % i, "g9") for i in range(3)]
    manifest = pd.DataFrame(rows, columns=["node", "split", "label", "path", "group_id"])

    m = gt.measure(manifest, index, dh, ph)
    assert m["max_sequence_share"] == 0.0
    assert m["n_heldout_sets"] == 0
    assert m["n_sets_top_seq_ge90"] == 0
    assert m["min_heldout_per_client"] == 3


def test_tvt_frame_matches_write_manifest_traversal():
    """_tvt_frame must emit the same rows as data_splitter.write_manifest."""
    tvt = {"node_a": {"fire": {"train": [["Fire/1.jpg", "Fire/2.jpg"]],
                               "val": [["Fire/3.jpg"]], "test": []},
                      "nofire": {"train": [["No_Fire/1.jpg"]], "val": [], "test": []}}}
    gid_of = {"Fire/1.jpg": "7", "Fire/2.jpg": "7", "Fire/3.jpg": "8"}

    frame = gt._tvt_frame(tvt, gid_of)

    assert list(frame.columns) == ["node", "split", "label", "path", "group_id"]
    assert frame.path.tolist() == ["Fire/1.jpg", "Fire/2.jpg", "No_Fire/1.jpg", "Fire/3.jpg"]
    assert frame.split.tolist() == ["train", "train", "train", "val"]
    assert frame.group_id.tolist() == ["7", "7", "singleton", "8"]


def test_latex_reports_a_range_not_a_single_optimum():
    """The table must present ranges over partitions, not an 'optimal' threshold."""
    table = pd.DataFrame([
        dict(grouping="tau=8", phash_threshold=-1, n_groups=394, n_nontrivial_groups=265,
             largest_group=4924, partition="iid", n_heldout=14311,
             residual_dhash_9_10=224, residual_dhash_le10=224, residual_phash_le8=335,
             min_heldout_per_client=137, node_images_max_over_min=1.02,
             n_heldout_sets=18, n_sets_top_seq_ge90=2,
             max_sequence_share=0.79, max_sequence_where="node_c/val/Fire g89"),
        dict(grouping="tau=8", phash_threshold=-1, n_groups=394, n_nontrivial_groups=265,
             largest_group=4924, partition="dirichlet_0.1", n_heldout=14360,
             residual_dhash_9_10=476, residual_dhash_le10=476, residual_phash_le8=359,
             min_heldout_per_client=140, node_images_max_over_min=1.02,
             n_heldout_sets=18, n_sets_top_seq_ge90=3,
             max_sequence_share=0.81, max_sequence_where="node_a/test/Fire g12"),
    ])
    out = gt.latex(table)
    assert "224--476" in out
    assert "5/36" in out          # dominance is summed over the partitions
    assert "137" in out           # balance is the worst case over the partitions
    assert "optimal" not in out.lower()
    assert r"\label{tab:grouping_tradeoff}" in out
