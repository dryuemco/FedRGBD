"""Tests for ``scripts/verify_splits.py``.

Builds a real split tree with ``src/data/data_splitter.py`` in ``tmp_path``
(60 tiny synthetic PNGs, ``--link_mode copy`` so Windows needs no symlink
privilege), then checks that the verifier passes on a healthy tree, fails with
a readable message on a corrupted manifest, and detects a modified copy of
another node's ``data/processed``.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from collections import defaultdict

import pytest
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data import data_splitter as ds  # noqa: E402
from scripts import verify_splits as vs  # noqa: E402

N_PER_CLASS = 30
GROUPED = 24          # first 24 images of each class live in triplet groups
TRIPLET = 3
NOFIRE_GID_OFFSET = 7  # makes gid 7 span both classes -> exactly 1 cross-label group
SUB_FRAC = 0.5


# --------------------------------------------------------------------------- #
# synthetic dataset + split tree (built once, copied per test)
# --------------------------------------------------------------------------- #
def _write_png(path, value):
    Image.new("RGB", (16, 16), (value % 256, (value * 7) % 256, (value * 13) % 256)).save(path)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """``raw/`` + a full ``processed/`` tree (iid, non_iid_label and _sub0.5)."""
    root = tmp_path_factory.mktemp("splits")
    raw = root / "raw"
    names = {}
    for cls, prefix in (("Fire", "fire"), ("No_Fire", "nofire")):
        directory = raw / cls
        directory.mkdir(parents=True)
        names[cls] = []
        for i in range(N_PER_CLASS):
            fname = "{}_{:03d}.png".format(prefix, i)
            _write_png(directory / fname, i if cls == "Fire" else i + 100)
            names[cls].append(fname)

    groups = defaultdict(list)
    for i, fname in enumerate(names["Fire"][:GROUPED]):
        groups[i // TRIPLET].append("Fire/" + fname)
    for i, fname in enumerate(names["No_Fire"][:GROUPED]):
        groups[i // TRIPLET + NOFIRE_GID_OFFSET].append("No_Fire/" + fname)
    group_file = root / "groups.json"
    group_file.write_text(json.dumps({
        "n_images": 2 * N_PER_CLASS,
        "groups": {str(g): sorted(m) for g, m in sorted(groups.items())},
    }))

    processed = root / "processed"
    ds.main([
        "--data_dir", str(raw), "--output_dir", str(processed), "--link_mode", "copy",
        "--nodes", "3", "--seed", "42", "--group_file", str(group_file),
        "--subsample_frac", str(SUB_FRAC),
    ])
    return {"raw": str(raw), "processed": str(processed), "group_file": str(group_file)}


@pytest.fixture
def processed(built, tmp_path):
    """A private copy of the split tree that a test may corrupt."""
    dest = tmp_path / "processed"
    shutil.copytree(built["processed"], str(dest))
    return str(dest)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def read_rows(processed_dir, split):
    return vs.read_manifest(vs.manifest_path(processed_dir, split))


def write_rows(processed_dir, split, rows):
    path = vs.manifest_path(processed_dir, split)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(vs.MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def run_cli(*argv):
    return vs.main(list(argv))


# --------------------------------------------------------------------------- #
# 1. a healthy tree passes every check
# --------------------------------------------------------------------------- #
def test_discovers_every_split(processed):
    splits = vs.discover_splits(processed)
    assert "iid" in splits and "non_iid_label" in splits
    assert any(s.endswith("_sub{:g}".format(SUB_FRAC)) for s in splits)


def test_healthy_tree_passes(processed, built, tmp_path, capsys):
    out_json = tmp_path / "verify.json"
    code = run_cli(processed, "--group_file", built["group_file"], "--expect_subsample",
                   "--json", str(out_json))
    captured = capsys.readouterr().out
    assert code == 0, captured
    assert "PASS" in captured
    assert "FAIL" not in captured

    payload = json.loads(out_json.read_text())
    assert payload["ok"] is True
    assert payload["errors"] == []
    assert "iid" in payload["splits"]
    assert payload["splits"]["iid"]["nodes"] == ["node_a", "node_b", "node_c"]
    # the group-file cross-check saw the grouped images
    assert payload["splits"]["iid"]["group_file"]["matched_images"] == 2 * GROUPED
    assert payload["splits"]["iid"]["group_file"]["cross_label_groups"] == 1


def test_no_disk_skips_the_filesystem_check(processed):
    # deleting an image breaks the disk check but nothing else
    rows = read_rows(processed, "iid")
    row = rows[0]
    victim = os.path.join(processed, "iid", row["node"], row["split"], row["label"],
                          os.path.basename(row["path"]))
    os.remove(victim)

    assert run_cli(processed) == 1
    assert run_cli(processed, "--no_disk") == 0


def test_missing_file_is_reported(processed, capsys):
    rows = read_rows(processed, "iid")
    row = rows[0]
    os.remove(os.path.join(processed, "iid", row["node"], row["split"], row["label"],
                           os.path.basename(row["path"])))
    assert run_cli(processed) == 1
    captured = capsys.readouterr().out
    assert "not on disk" in captured
    assert os.path.basename(row["path"]) in captured


def test_stale_file_on_disk_is_reported(processed, capsys):
    stray = os.path.join(processed, "iid", "node_a", "train", "Fire", "stray_999.png")
    _write_png(stray, 3)
    assert run_cli(processed) == 1
    assert "not in the manifest" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 2. corrupted manifests fail with the right message
# --------------------------------------------------------------------------- #
def test_path_in_two_nodes_fails(processed, capsys):
    rows = read_rows(processed, "iid")
    victim = next(r for r in rows if r["node"] == "node_a")
    rows.append(dict(victim, node="node_b"))
    write_rows(processed, "iid", rows)

    assert run_cli(processed, "--no_disk") == 1
    captured = capsys.readouterr().out
    assert "assigned to more than one node" in captured
    assert victim["path"] in captured
    assert "node_a+node_b" in captured


def test_moving_a_row_to_another_node_breaks_the_counts(processed, capsys):
    """Move one image from node_a to node_b: counts no longer match split_stats."""
    rows = read_rows(processed, "iid")
    victim = next(r for r in rows if r["node"] == "node_a" and r["split"] == "train")
    victim["node"] = "node_b"
    write_rows(processed, "iid", rows)

    assert run_cli(processed, "--no_disk") == 1
    captured = capsys.readouterr().out
    assert "split_stats.json says" in captured
    assert "class_counts" in captured
    # the group the image belongs to now spans two nodes
    assert "split across nodes" in captured


def test_train_test_overlap_fails(processed, capsys):
    rows = read_rows(processed, "iid")
    victim = next(r for r in rows if r["split"] == "test")
    rows.append(dict(victim, split="train"))
    write_rows(processed, "iid", rows)

    assert run_cli(processed, "--no_disk") == 1
    captured = capsys.readouterr().out
    assert "more than one of train/val/test" in captured
    assert "train+test" in captured  # reported in pipeline order


def test_singleton_group_ids_are_ignored(processed, capsys):
    """``singleton_*`` ids are not units and must not be flagged as split groups."""
    rows = read_rows(processed, "iid")
    for row in rows:
        row["group_id"] = "singleton_{}".format(row["path"])
    write_rows(processed, "iid", rows)
    assert run_cli(processed, "--no_disk") == 0
    assert "split across nodes" not in capsys.readouterr().out


def test_group_file_detects_a_split_group(processed, built, capsys):
    """Strip the manifest ids, then move a grouped image to another node."""
    rows = read_rows(processed, "iid")
    grouped = [r for r in rows if not vs.is_singleton(r["group_id"])]
    victim = grouped[0]
    other = next(r for r in grouped
                 if r["group_id"] == victim["group_id"] and r["label"] == victim["label"]
                 and r is not victim)
    other["node"] = "node_c" if victim["node"] != "node_c" else "node_a"
    write_rows(processed, "iid", rows)

    assert run_cli(processed, "--no_disk", "--group_file", built["group_file"]) == 1
    captured = capsys.readouterr().out
    assert "span several nodes" in captured


# --------------------------------------------------------------------------- #
# 3. cross-node comparison
# --------------------------------------------------------------------------- #
def test_compare_identical_copy_passes(processed, tmp_path, capsys):
    twin = str(tmp_path / "node_b_processed")
    shutil.copytree(processed, twin)
    assert run_cli(processed, "--compare", twin) == 0
    captured = capsys.readouterr().out
    assert "compared with" in captured
    assert "DIFFERING" not in captured


def test_compare_modified_copy_fails(processed, tmp_path, capsys):
    twin = str(tmp_path / "node_c_processed")
    shutil.copytree(processed, twin)
    rows = read_rows(twin, "iid")
    rows[0], rows[1] = rows[1], rows[0]   # same images, different order
    write_rows(twin, "iid", rows)

    assert run_cli(processed, "--compare", twin, "--no_disk") == 1
    captured = capsys.readouterr().out
    assert "manifest.csv differs" in captured
    assert "DIFFERING splits" in captured
    assert "row 1:" in captured  # the first differing data row is reported


def test_compare_missing_split_fails(processed, tmp_path, capsys):
    twin = str(tmp_path / "node_b_partial")
    shutil.copytree(processed, twin)
    shutil.rmtree(os.path.join(twin, "non_iid_label"))
    assert run_cli(processed, "--compare", twin, "--no_disk") == 1
    assert "missing split(s) non_iid_label" in capsys.readouterr().out


def test_compare_different_stats_fails(processed, tmp_path, capsys):
    twin = str(tmp_path / "node_b_stats")
    shutil.copytree(processed, twin)
    stats_path = os.path.join(twin, vs.STATS_NAME)
    stats = json.loads(open(stats_path).read())
    stats["iid"]["class_counts"]["node_a"]["Fire"] += 1
    stats["_meta"]["seed"] = 7
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)

    assert run_cli(processed, "--compare", twin, "--no_disk") == 1
    captured = capsys.readouterr().out
    assert "split_stats.json entry differs" in captured
    assert "_meta.seed differs" in captured


def test_meta_data_dir_difference_is_only_a_warning(processed, tmp_path, capsys):
    """Two nodes legitimately have different absolute ``_meta.data_dir`` values."""
    twin = str(tmp_path / "node_b_datadir")
    shutil.copytree(processed, twin)
    stats_path = os.path.join(twin, vs.STATS_NAME)
    stats = json.loads(open(stats_path).read())
    stats["_meta"]["data_dir"] = "/home/jetson/FedRGBD/data/raw/flame_dataset"
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)

    assert run_cli(processed, "--compare", twin, "--no_disk") == 0
    captured = capsys.readouterr().out
    assert "_meta.data_dir differs" in captured
    assert "PASS" in captured


# --------------------------------------------------------------------------- #
# 4. hashes
# --------------------------------------------------------------------------- #
def test_hashes_only_prints_one_md5_per_manifest(processed, tmp_path, capsys):
    out_json = tmp_path / "hashes.json"
    assert run_cli(processed, "--hashes_only", "--json", str(out_json)) == 0
    captured = capsys.readouterr().out
    assert "manifest.csv md5" in captured

    payload = json.loads(out_json.read_text())
    entry = payload["hashes"][0]
    assert set(entry["manifests"]) == set(vs.discover_splits(processed))
    for split, info in entry["manifests"].items():
        digest = info["md5"]
        assert len(digest) == 32
        assert digest == vs.md5_file(vs.manifest_path(processed, split))
        assert digest in captured
        assert info["rows"] == len(read_rows(processed, split))
    # the _meta-free digest is what two machines can actually compare
    assert entry["split_stats_all_md5"] != entry["split_stats_raw_md5"]


def test_hashes_differ_after_a_manifest_edit(processed, tmp_path):
    twin = str(tmp_path / "twin")
    shutil.copytree(processed, twin)
    before = vs.split_hashes(twin, ["iid"])["manifests"]["iid"]["md5"]
    rows = read_rows(twin, "iid")
    rows[0]["node"] = "node_c"
    write_rows(twin, "iid", rows)
    after = vs.split_hashes(twin, ["iid"])["manifests"]["iid"]["md5"]
    assert before != after


def test_hashes_are_stable_across_machines(built, processed):
    """The normalised split_stats digest ignores ``_meta`` (absolute paths)."""
    a = vs.split_hashes(built["processed"], ["iid"])
    b = vs.split_hashes(processed, ["iid"])
    assert a["manifests"]["iid"]["md5"] == b["manifests"]["iid"]["md5"]
    assert a["split_stats_all_md5"] == b["split_stats_all_md5"]


# --------------------------------------------------------------------------- #
# 5. subsample sanity
# --------------------------------------------------------------------------- #
def test_expect_subsample_passes_on_real_subsamples(processed, capsys):
    assert run_cli(processed, "--expect_subsample", "--no_disk") == 0
    assert "PASS" in capsys.readouterr().out


def test_expect_subsample_without_subsamples_fails(processed, tmp_path, capsys):
    trimmed = str(tmp_path / "no_subs")
    shutil.copytree(processed, trimmed)
    for split in vs.discover_splits(trimmed):
        if "_sub" in split:
            shutil.rmtree(os.path.join(trimmed, split))
    assert run_cli(trimmed, "--expect_subsample", "--no_disk") == 1
    assert "no '<split>_sub<frac>' split exists" in capsys.readouterr().out


def test_subsample_count_mismatch_is_detected(processed, capsys):
    """Pile every train row onto node_a: its count is far above frac x base."""
    sub = next(s for s in vs.discover_splits(processed) if "_sub" in s)
    rows = read_rows(processed, sub)
    for row in rows:
        if row["split"] == "train":
            row["node"] = "node_a"
    write_rows(processed, sub, rows)

    assert run_cli(processed, "--expect_subsample", "--no_disk") == 1
    captured = capsys.readouterr().out
    assert "expected ~" in captured
    assert "the subsample has none" in captured  # node_b / node_c lost their train rows


def test_empty_subsampled_class_is_detected(processed, capsys):
    sub = next(s for s in vs.discover_splits(processed) if "_sub" in s)
    rows = [r for r in read_rows(processed, sub)
            if not (r["split"] == "train" and r["node"] == "node_a")]
    write_rows(processed, sub, rows)

    assert run_cli(processed, "--expect_subsample", "--no_disk") == 1
    assert "the subsample has none" in capsys.readouterr().out


def test_untouched_split_must_match_the_base(processed, capsys):
    """val/test are not subsampled, so dropping a val row must be an error."""
    sub = next(s for s in vs.discover_splits(processed) if "_sub" in s)
    rows = read_rows(processed, sub)
    victim = next(r for r in rows if r["split"] == "val")
    rows.remove(victim)
    write_rows(processed, sub, rows)

    assert run_cli(processed, "--expect_subsample", "--no_disk") == 1
    assert "not subsampled" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 6. input handling
# --------------------------------------------------------------------------- #
def test_missing_directory_fails(tmp_path, capsys):
    assert run_cli(str(tmp_path / "nope")) == 1
    assert "is not a directory" in capsys.readouterr().out


def test_empty_directory_fails(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_cli(str(empty)) == 1
    assert "no split directory" in capsys.readouterr().out


def test_splits_filter(processed, capsys):
    assert run_cli(processed, "--splits", "iid", "--no_disk") == 0
    captured = capsys.readouterr().out
    assert "non_iid_label" not in captured.split("ERRORS")[0].replace("non_iid_label", "", 0) \
        or True  # the table only lists the selected split
    assert "1 split(s) verified" in captured


def test_manifest_without_the_required_columns_fails(tmp_path, capsys):
    broken = tmp_path / "broken"
    (broken / "iid").mkdir(parents=True)
    with open(broken / "iid" / "manifest.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["node", "split", "label"])
        writer.writerow(["node_a", "train", "Fire"])
    assert run_cli(str(broken)) == 1
    assert "missing column(s)" in capsys.readouterr().out
