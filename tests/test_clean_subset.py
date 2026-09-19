"""The pre-registered clean-subset rule (scripts/clean_subset.py)."""

import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from scripts import clean_subset as cs  # noqa: E402

OUT = os.path.join(_REPO, "analysis", "leakage", "clean_subset")


def test_rule_is_frozen():
    """CLAUDE.md rule 7: these values must never change after results were seen."""
    assert cs.RULE["dhash_max"] == 12 and cs.RULE["phash_max"] == 10
    assert cs.RULE["hash_size"] == 8 and cs.RULE["fixed_on"] == "2026-09-19"
    committed = json.load(open(os.path.join(OUT, "RULE.json")))
    for key in ("dhash_max", "phash_max", "hash_size", "fixed_on", "text", "name"):
        assert committed[key] == cs.RULE[key], key


def test_rule_boundaries_on_synthetic_hashes():
    """Exclusion is inclusive at 12 (dHash) / 10 (pHash) and uses the NEAREST training image."""
    def h(bits):                                  # uint64 with the lowest `bits` bits set
        return np.uint64((1 << bits) - 1)

    train = np.array([h(0)], dtype=np.uint64)
    q = np.array([h(12), h(13), h(10), h(11)], dtype=np.uint64)
    d = cs.nearest(q, train)
    assert d.tolist() == [12, 13, 10, 11]
    assert ((d <= cs.RULE["dhash_max"])).tolist() == [True, False, True, True]
    assert ((d <= cs.RULE["phash_max"])).tolist() == [False, False, True, False]
    # nearest over several training images
    assert cs.nearest(np.array([h(20)], dtype=np.uint64),
                      np.array([h(0), h(18), h(40)], dtype=np.uint64)).tolist() == [2]


def test_committed_lists_are_consistent_with_the_rule():
    rule = json.load(open(os.path.join(OUT, "RULE.json")))
    for part in cs.PARTITIONS:
        ex = pd.read_csv(os.path.join(OUT, "%s_excluded.csv.gz" % part))
        assert ((ex.min_dhash <= 12) | (ex.min_phash <= 10)).all(), part
        assert set(ex.split) <= {"val", "test"}, part
        assert len(ex) == rule["excluded_counts"][part]["excluded"], part
        raw = gzip.open(os.path.join(OUT, "%s_excluded.csv.gz" % part)).read()
        import hashlib
        assert hashlib.sha256(raw).hexdigest() == rule["list_sha256"]["%s_excluded.csv.gz" % part]
        # never a training image, and every row is a held-out image of that partition
        m = pd.read_csv(os.path.join(_REPO, "data", "splits", part + ".csv.gz"))
        held = set(m[m.split != "train"].path)
        assert set(ex.path) <= held, part


def test_end_to_end_lists_reproduce_exactly():
    """Recomputing from groups.csv, phash.csv.gz and data/splits gives the committed files."""
    assert cs.main(["--check"]) == 0
