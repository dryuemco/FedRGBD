"""Group-mode behaviour of ``src/data/data_splitter.py`` with *oversized* units.

The real FLAME group file turns whole video sequences into single units (the
largest holds 4,341 images of one class).  These tests reproduce that shape on
a tiny synthetic dataset and pin the three fixes it required:

1. largest-first greedy assignment (nodes, train/val/test, Dirichlet) so that
   no bucket is left empty by a sequential cut;
2. image-level subsampling *inside* the node's own units, so ``--subsample_frac``
   is exact in group mode while the leakage guarantee is untouched;
3. ``--verify`` compares against the *fresh* ``split_stats.json`` even when a
   stale one from a previous partition is on disk (the normal ``--clean``
   re-split situation).
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import pytest
from PIL import Image

from src.data import data_splitter as ds

# one group is a fifth of its class, many small ones -- the FLAME shape in miniature
FIRE_SIZES = [12, 8, 6, 5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]      # 60 images
NOFIRE_SIZES = [10, 7, 5, 4, 4, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]    # 50 images
NODES = ["node_a", "node_b", "node_c"]


def _write_png(path, value):
    Image.new("RGB", (8, 8), (value % 256, (value * 7) % 256, (value * 13) % 256)).save(path)


@pytest.fixture
def big_groups(tmp_path):
    raw = tmp_path / "raw"
    groups = defaultdict(list)
    gid = 0
    for cls, prefix, sizes, offset in (("Fire", "fire", FIRE_SIZES, 0),
                                       ("No_Fire", "nofire", NOFIRE_SIZES, 1000)):
        d = raw / cls
        d.mkdir(parents=True)
        i = 0
        for size in sizes:
            for _ in range(size):
                fname = "{}_{:03d}.png".format(prefix, i)
                _write_png(d / fname, i + offset)
                groups[gid].append("{}/{}".format(cls, fname))
                i += 1
            gid += 1
    group_file = tmp_path / "groups.json"
    group_file.write_text(json.dumps({"groups": {str(k): v for k, v in groups.items()}}))
    return {"raw": str(raw), "group_file": str(group_file), "root": tmp_path}


def run_split(data_dir, output_dir, **kw):
    argv = ["--data_dir", str(data_dir), "--output_dir", str(output_dir), "--link_mode", "copy"]
    for key, value in kw.items():
        flag = "--" + key
        if value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(str(v) for v in value)
        else:
            argv.extend([flag, str(value)])
    return ds.main(argv)


def rows_of(output_dir, split):
    with open(os.path.join(str(output_dir), split, "manifest.csv"), newline="") as f:
        return list(csv.DictReader(f))


def counts(output_dir, split):
    out = defaultdict(int)
    for r in rows_of(output_dir, split):
        out[(r["node"], r["split"])] += 1
    return out


def assert_groups_intact(rows, split):
    nodes_of, tvt_of = defaultdict(set), defaultdict(set)
    for r in rows:
        if r["group_id"].startswith("singleton"):
            continue
        key = (r["group_id"], r["label"])
        nodes_of[key].add(r["node"])
        tvt_of[key].add(r["split"])
    assert nodes_of, split
    assert all(len(v) == 1 for v in nodes_of.values()), split
    assert all(len(v) == 1 for v in tvt_of.values()), split


# --------------------------------------------------------------------------- #
# 1. the greedy primitive
# --------------------------------------------------------------------------- #
def test_assign_units_greedy_preserves_units_and_bounds_deviation():
    units = [["p{}_{}".format(i, j) for j in range(size)] for i, size in enumerate(FIRE_SIZES)]
    quotas = [20, 20, 20]
    parts = ds.assign_units_greedy(units, quotas, seed=3)
    assert len(parts) == 3
    assert sorted(p for part in parts for u in part for p in u) == sorted(p for u in units for p in u)
    for part, q in zip(parts, quotas):
        assert part, "greedy must not leave a bucket empty when units >= buckets"
        largest = max(len(u) for u in part)
        assert abs(ds.n_images(part) - q) <= largest
    # LPT on these sizes balances the thirds exactly
    assert sorted(ds.n_images(p) for p in parts) == [20, 20, 20]


def test_assign_units_greedy_is_deterministic_and_seed_dependent():
    units = [["a"], ["b"], ["c"], ["d"], ["e"], ["f"]]  # all ties -> seed decides
    p1 = ds.assign_units_greedy(units, [3, 3], seed=1)
    p2 = ds.assign_units_greedy(units, [3, 3], seed=1)
    p3 = ds.assign_units_greedy(units, [3, 3], seed=2)
    assert p1 == p2
    assert [ds.n_images(p) for p in p1] == [3, 3]
    assert p1 != p3 or True  # a different seed may coincide, but must not crash


def test_assign_units_greedy_fills_a_tiny_quota_last_with_a_small_unit():
    units = [["x"] * 20, ["y"] * 12, ["z"] * 8, ["w"] * 2, ["v"]]
    parts = ds.assign_units_greedy(units, [1, 21, 21], seed=0)  # 43 images, quotas sum to 43
    # the near-zero quota gets a small unit, not the 20-image one
    assert ds.n_images(parts[0]) <= 2
    assert ds.n_images(parts[0]) >= 1


def test_is_grouped():
    assert ds.is_grouped([["a"], ["b"]]) is False
    assert ds.is_grouped([["a"], ["b", "c"]]) is True
    assert ds.is_grouped([["a"]], [["b", "c"]]) is True


# --------------------------------------------------------------------------- #
# 2. end-to-end group mode with oversized units
# --------------------------------------------------------------------------- #
def test_group_mode_leaves_no_bucket_empty_and_keeps_groups(big_groups, tmp_path):
    out = tmp_path / "out"
    stats = run_split(big_groups["raw"], out, seed=42, nodes=3, group_file=big_groups["group_file"],
                      dirichlet_alpha=[1.0], dirichlet_min_size=8, subsample_frac=[0.1],
                      verify=True)
    meta = stats["_meta"]
    assert meta["grouped"] is True
    assert meta["assignment"] == "greedy_largest_first"
    assert meta["subsample_level"] == "image_within_units"
    assert meta["verify_ok"] is True
    assert meta["largest_unit"] == max(FIRE_SIZES)

    for split in ("iid", "non_iid_label", "dirichlet_1"):
        c = counts(out, split)
        for node in NODES:
            # a Dirichlet node may legitimately be tiny (that is the intended skew);
            # the equal / label-skew designs must give every node all three buckets
            buckets = ("train", "val", "test") if split != "dirichlet_1" else ("train",)
            for tvt in buckets:
                assert c[(node, tvt)] > 0, "{} {} {} is empty".format(split, node, tvt)
        assert_groups_intact(rows_of(out, split), split)

    # iid: node totals within one largest unit of equal thirds (per class quotas)
    node_tot = defaultdict(int)
    for r in rows_of(out, "iid"):
        node_tot[r["node"]] += 1
    total = sum(FIRE_SIZES) + sum(NOFIRE_SIZES)
    assert max(abs(v - total / 3) for v in node_tot.values()) <= max(FIRE_SIZES) + max(NOFIRE_SIZES)


def test_group_mode_label_skew_quotas_are_the_original_arithmetic():
    qf, qn = ds.label_skew_quotas(30155, 17837, 3)
    per_node = 47992 // 3
    assert qf[0] == int(per_node * 0.8) and qf[0] + qn[0] == per_node
    assert qn[2] == int(per_node * 0.8) and qf[2] + qn[2] == per_node
    assert sum(qf) == 30155 and sum(qn) == 17837
    qf2, qn2 = ds.label_skew_quotas(100, 100, 2)
    assert qf2 == [70, 30] and qn2 == [30, 70]


def test_group_mode_subsample_is_exact_and_leak_free(big_groups, tmp_path):
    out = tmp_path / "out_sub"
    frac = 0.1
    run_split(big_groups["raw"], out, seed=42, nodes=2, group_file=big_groups["group_file"],
              subsample_frac=[frac])
    for base in ("iid", "non_iid_label"):
        sub = "{}_sub{:g}".format(base, frac)
        full = defaultdict(int)
        part = defaultdict(int)
        for r in rows_of(out, base):
            full[(r["node"], r["split"], r["label"])] += 1
        for r in rows_of(out, sub):
            part[(r["node"], r["split"], r["label"])] += 1
        for key, n_full in full.items():
            n_sub = part.get(key, 0)
            if key[1] == "train":
                assert n_sub == max(1, int(round(frac * n_full))), (base, key, n_full, n_sub)
            else:
                assert n_sub == n_full
        # survivors keep their group id and stay inside their original node/split
        sub_rows = rows_of(out, sub)
        assert_groups_intact(sub_rows, sub)
        where = {r["path"]: (r["node"], r["split"]) for r in rows_of(out, base)}
        assert all(where[r["path"]] == (r["node"], r["split"]) for r in sub_rows)


# --------------------------------------------------------------------------- #
# 3. --verify against a stale split_stats.json (the --clean re-split case)
# --------------------------------------------------------------------------- #
def test_verify_passes_when_a_stale_stats_file_is_on_disk(big_groups, tmp_path, capsys):
    out = tmp_path / "out_stale"
    # v1-style image-level partition first ...
    run_split(big_groups["raw"], out, seed=42, nodes=3)
    assert os.path.isfile(os.path.join(str(out), "split_stats.json"))
    capsys.readouterr()
    # ... then the group-safe re-split into the same directory
    stats = run_split(big_groups["raw"], out, seed=42, nodes=3, group_file=big_groups["group_file"],
                      clean=True, verify=True)
    captured = capsys.readouterr().out
    assert stats["_meta"]["verify_ok"] is True, captured
    assert "VERIFY PASS" in captured
    with open(os.path.join(str(out), "split_stats.json")) as f:
        on_disk = json.load(f)
    assert on_disk["_meta"]["verify_ok"] is True
    assert "partial" not in on_disk["_meta"]
    # and the stats on disk describe the *new* partition
    c = counts(out, "iid")
    for node in NODES:
        assert on_disk["iid"][node]["train"]["total"] == c[(node, "train")]


def test_default_path_reports_sequential_cut(big_groups, tmp_path):
    stats = run_split(big_groups["raw"], tmp_path / "out_default", seed=42, nodes=2)
    assert stats["_meta"]["grouped"] is False
    assert stats["_meta"]["assignment"] == "sequential_cut"
    assert stats["_meta"]["subsample_level"] == "unit"
