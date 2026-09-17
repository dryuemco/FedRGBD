"""Cross-label groups in group mode: one bundle, one node, one of train/val/test.

On FLAME 22 near-duplicate groups contain frames of *both* labels (a video in
which fire appears and disappears); they hold 20,006 of the 47,992 images.
Treating the Fire part and the No_Fire part as independent units put them on
different nodes / in train and test, and the label-agnostic leakage audit
flagged up to 72 % of a node's test images.  These tests pin the fix:
``bundle_units`` + ``assign_bundles_greedy`` keep a cross-label group together,
and both verifiers (``scripts/verify_splits.py`` and the splitter's fallback)
reject a manifest in which a group is split by label.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import pytest
from PIL import Image

from src.data import data_splitter as ds
from scripts import verify_splits as vs

NODES = ["node_a", "node_b", "node_c"]


def _write_png(path, value):
    Image.new("RGB", (8, 8), (value % 256, (value * 7) % 256, (value * 13) % 256)).save(path)


@pytest.fixture
def mixed_groups(tmp_path):
    """Two big cross-label groups + pure groups + singletons, FLAME-like in miniature."""
    raw = tmp_path / "raw"
    (raw / "Fire").mkdir(parents=True)
    (raw / "No_Fire").mkdir(parents=True)
    groups = defaultdict(list)
    spec = [  # (gid, n_fire, n_nofire)
        (0, 14, 4), (1, 6, 9),                     # cross-label
        (2, 8, 0), (3, 5, 0), (4, 3, 0), (5, 2, 0), (6, 2, 0),
        (7, 0, 7), (8, 0, 5), (9, 0, 3), (10, 0, 2), (11, 0, 2),
    ]
    fi = ni = 0
    for gid, nf, nn in spec:
        for _ in range(nf):
            name = "fire_{:03d}.png".format(fi)
            _write_png(raw / "Fire" / name, fi)
            groups[gid].append("Fire/" + name)
            fi += 1
        for _ in range(nn):
            name = "nofire_{:03d}.png".format(ni)
            _write_png(raw / "No_Fire" / name, 500 + ni)
            groups[gid].append("No_Fire/" + name)
            ni += 1
    for k in range(6):  # singletons, not in the group file
        _write_png(raw / "Fire" / "fire_s{}.png".format(k), 900 + k)
        _write_png(raw / "No_Fire" / "nofire_s{}.png".format(k), 950 + k)
    gf = tmp_path / "groups.json"
    gf.write_text(json.dumps({"groups": {str(k): v for k, v in groups.items()}}))
    return {"raw": str(raw), "group_file": str(gf), "cross": {"0", "1"}}


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


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def test_bundle_units_pairs_cross_label_groups():
    gid_of = {"f1": "7", "f2": "7", "f3": "8", "n1": "7", "n2": "9", "s": "singleton_0"}
    fire = [["f1", "f2"], ["f3"], ["s"]]
    nofire = [["n1"], ["n2"]]
    bundles = ds.bundle_units(fire, nofire, gid_of)
    assert (["f1", "f2"], ["n1"]) in bundles          # paired by gid 7
    assert (["f3"], []) in bundles
    assert (["s"], []) in bundles                      # singleton stays alone
    assert ([], ["n2"]) in bundles
    assert len(bundles) == 4
    # without gid_of nothing is paired
    assert len(ds.bundle_units(fire, nofire, None)) == 5


def test_assign_bundles_greedy_keeps_bundle_whole_and_meets_quotas():
    bundles = [(["a"] * 10, ["b"] * 4), (["c"] * 6, []), ([], ["d"] * 6), (["e"] * 2, ["f"] * 2)]
    parts = ds.assign_bundles_greedy(bundles, quotas_fire=[9, 9], quotas_nofire=[6, 6], seed=1)
    assert len(parts) == 2
    # every bundle's two halves sit in the same bucket
    for fire_part, nofire_part in bundles:
        homes = {b for b, (fu, nu) in enumerate(parts)
                 if (fire_part and fire_part in fu) or (nofire_part and nofire_part in nu)}
        assert len(homes) == 1, (fire_part, nofire_part, homes)
    tot_f = [ds.n_images(fu) for fu, _ in parts]
    tot_n = [ds.n_images(nu) for _, nu in parts]
    assert sum(tot_f) == 18 and sum(tot_n) == 12
    # the 14-image mixed bundle goes first; the rest balances the other bucket
    assert max(tot_f) - min(tot_f) <= 10 and max(tot_n) - min(tot_n) <= 6


def test_assign_bundles_greedy_matches_units_greedy_for_pure_bundles():
    units = [["p{}_{}".format(i, j) for j in range(n)] for i, n in enumerate([9, 7, 5, 3, 3, 2, 1])]
    by_units = ds.assign_units_greedy(units, [10, 10, 10], seed=5)
    by_bundles = ds.assign_bundles_greedy([(u, []) for u in units], [10, 10, 10], [0, 0, 0], seed=5)
    assert [fu for fu, _ in by_bundles] == by_units


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_cross_label_groups_stay_on_one_node_and_in_one_bucket(mixed_groups, tmp_path):
    out = tmp_path / "out"
    stats = run_split(mixed_groups["raw"], out, seed=42, nodes=3,
                      group_file=mixed_groups["group_file"], dirichlet_alpha=[0.5, 1.0],
                      dirichlet_min_size=4, subsample_frac=[0.5], verify=True)
    assert stats["_meta"]["cross_label_groups"] == 2
    assert stats["_meta"]["verify_ok"] is True
    for split in ("iid", "non_iid_label", "dirichlet_0.5", "dirichlet_1", "iid_sub0.5"):
        where = defaultdict(set)
        labels = defaultdict(set)
        for r in rows_of(out, split):
            if r["group_id"].startswith("singleton"):
                continue
            where[r["group_id"]].add((r["node"], r["split"]))
            labels[r["group_id"]].add(r["label"])
        for gid in mixed_groups["cross"]:
            assert labels[gid] == {"Fire", "No_Fire"}, (split, gid)
            assert len(where[gid]) == 1, "{}: cross-label group {} spread over {}".format(
                split, gid, sorted(where[gid]))
        assert all(len(v) == 1 for v in where.values()), split


def test_cross_label_group_split_by_label_is_rejected_by_both_verifiers(tmp_path):
    rows = [
        ["node_a", "train", "Fire", "Fire/a.png", "7"],
        ["node_a", "train", "Fire", "Fire/b.png", "7"],
        ["node_b", "test", "No_Fire", "No_Fire/c.png", "7"],   # same group, other label
        ["node_b", "train", "No_Fire", "No_Fire/d.png", "8"],
    ]
    os.makedirs(os.path.join(str(tmp_path), "iid"))
    with open(os.path.join(str(tmp_path), "iid", "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "split", "label", "path", "group_id"])
        w.writerows(rows)
    dict_rows = [dict(zip(["node", "split", "label", "path", "group_id"], r)) for r in rows]

    report = vs.Report()
    vs.check_manifest("iid", dict_rows, report)
    messages = " ".join(i["message"] for i in report.errors)
    assert "labels are ignored" in messages
    assert "across nodes" in messages and "train/val/test" in messages

    problems = ds._local_verify_split("iid", dict_rows, stats=None)
    assert any("across labels" in p and "spans nodes" in p for p in problems)
    assert any("across labels" in p and "train" in p and "test" in p and "nodes" not in p for p in problems)
    # the (group_id, label) unit checks alone would have passed
    assert not any("(label" in p for p in problems)
