"""The low-data sequence coverage (analysis/lowdata_sequence_coverage.csv) is what the
committed manifests give: recomputed here from data/splits, never typed."""

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import lowdata_sequence_coverage as cov  # noqa: E402

CSV = os.path.join(ROOT, "analysis", "lowdata_sequence_coverage.csv")


def test_committed_coverage_is_the_manifests(tmp_path):
    out = tmp_path / "cov.csv"
    assert cov.main(["--out", str(out)]) == 0
    fresh = pd.read_csv(out, dtype={"rho": str})
    committed = pd.read_csv(CSV, dtype={"rho": str})
    pd.testing.assert_frame_equal(fresh, committed)
    assert len(committed) == len(cov.PARTITIONS) * len(cov.FRACTIONS) * len(cov.NODES)


def test_coverage_counts_sequences_per_node():
    full = pd.DataFrame({"node": ["node_a"] * 4 + ["node_b"] * 2 + ["node_c"] * 2,
                         "split": ["train"] * 3 + ["test"] + ["train"] * 4,
                         "group_id": ["1", "1", "2", "9", "3", "4", "5", "6"]})
    sub = full[full["group_id"].isin(["1", "3", "5", "6"]) & (full["split"] == "train")]
    df = cov.coverage(full, sub).set_index("node")
    assert df.loc["node_a", "train_sequences_full"] == 2      # the test-only group 9 is not counted
    assert df.loc["node_a", "sequence_share"] == 0.5
    assert df.loc["node_c", "sequence_share"] == 1.0
    with pytest.raises(ValueError):
        cov.coverage(full, pd.concat([sub, pd.DataFrame(
            {"node": ["node_b"], "split": ["train"], "group_id": ["77"]})]))
