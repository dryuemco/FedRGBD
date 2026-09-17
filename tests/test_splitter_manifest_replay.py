"""``--export_manifests`` / ``--from_manifest``: ship the partition instead of re-deriving it.

``find_images`` walks the dataset with ``os.walk``, whose order is not guaranteed
across machines or filesystems, so regenerating the split independently on each
Jetson could produce *different* partitions -- which would break both the
federated protocol and the leakage guarantee.  The authoritative partition is
therefore exported once (``data/splits/<split>.csv.gz``) and replayed on every
node.  These tests pin that the replay is exact, RNG-free and order-independent.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import random
from collections import defaultdict

import pytest
from PIL import Image

from src.data import data_splitter as ds

N_PER_CLASS = 24
TRIPLET = 3


def _write_png(path, value):
    Image.new("RGB", (8, 8), (value % 256, (value * 7) % 256, (value * 13) % 256)).save(path)


@pytest.fixture
def dataset(tmp_path):
    raw = tmp_path / "raw"
    groups = defaultdict(list)
    for cls, prefix, offset in (("Fire", "fire", 0), ("No_Fire", "nofire", 100)):
        d = raw / cls
        d.mkdir(parents=True)
        for i in range(N_PER_CLASS):
            name = "{}_{:03d}.png".format(prefix, i)
            _write_png(d / name, i + offset)
            groups[i // TRIPLET if cls == "Fire" else 100 + i // TRIPLET].append(
                "{}/{}".format(cls, name))
    gf = tmp_path / "groups.json"
    gf.write_text(json.dumps({"groups": {str(k): v for k, v in groups.items()}}))
    return {"raw": str(raw), "group_file": str(gf)}


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


def manifest_md5(output_dir, split):
    path = os.path.join(str(output_dir), split, "manifest.csv")
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def tree_files(output_dir):
    """``{relative path: basename}`` of every placed image."""
    out = set()
    for root, _dirs, files in os.walk(str(output_dir)):
        for f in files:
            if f.endswith(".png"):
                out.add(os.path.relpath(os.path.join(root, f), str(output_dir)).replace(os.sep, "/"))
    return out


#: every split the fixture produces (two base splits + one Dirichlet, each also subsampled)
SPLITS = ["iid", "non_iid_label", "dirichlet_1",
          "iid_sub0.5", "non_iid_label_sub0.5", "dirichlet_1_sub0.5"]


@pytest.fixture
def partitioned(dataset, tmp_path):
    out = tmp_path / "processed"
    stats = run_split(dataset["raw"], out, seed=42, nodes=3, group_file=dataset["group_file"],
                      dirichlet_alpha=[1.0], dirichlet_min_size=2, subsample_frac=[0.5])
    archive = tmp_path / "splits"
    ds.export_manifests(str(out), str(archive), quiet=True)
    return {"out": out, "archive": archive, "stats": stats, "raw": dataset["raw"]}


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def test_export_writes_one_gzip_per_split_plus_stats(partitioned):
    archive = partitioned["archive"]
    names = sorted(os.listdir(str(archive)))
    assert "split_stats.json" in names
    gz = [n for n in names if n.endswith(".csv.gz")]
    assert sorted(gz) == sorted(s + ".csv.gz" for s in SPLITS)
    for split in SPLITS:
        with gzip.open(os.path.join(str(archive), split + ".csv.gz"), "rt", newline="") as f:
            text = f.read()
        with open(os.path.join(str(partitioned["out"]), split, "manifest.csv"), newline="") as f:
            assert text == f.read()   # byte-identical, line endings included


def test_export_is_byte_stable_across_runs(partitioned, tmp_path):
    """mtime=0 in the gzip header: re-exporting must not create a spurious diff."""
    first = {n: open(os.path.join(str(partitioned["archive"]), n), "rb").read()
             for n in os.listdir(str(partitioned["archive"])) if n.endswith(".gz")}
    again = tmp_path / "splits2"
    ds.export_manifests(str(partitioned["out"]), str(again), quiet=True)
    second = {n: open(os.path.join(str(again), n), "rb").read()
              for n in os.listdir(str(again)) if n.endswith(".gz")}
    assert first == second


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #
def test_replay_reproduces_the_tree_and_the_manifest_md5s(partitioned, tmp_path):
    replayed = tmp_path / "replayed"
    stats = run_split(partitioned["raw"], replayed, from_manifest=str(partitioned["archive"]),
                      clean=True, verify=True)
    assert stats["_meta"]["verify_ok"] is True
    assert stats["_meta"]["replayed_from"] == os.path.abspath(str(partitioned["archive"]))
    for split in SPLITS:
        assert manifest_md5(replayed, split) == manifest_md5(partitioned["out"], split)
    assert tree_files(replayed) == tree_files(partitioned["out"])


def test_replay_does_not_touch_the_random_number_generator(partitioned, tmp_path):
    """No RNG at all: the global random state must be identical before and after."""
    random.seed(1234)
    before = random.getstate()
    ds.set_link_mode("copy")
    ds.replay_manifests(str(partitioned["archive"]), partitioned["raw"],
                        str(tmp_path / "r2"), clean=True, quiet=True)
    assert random.getstate() == before


def test_replay_is_independent_of_file_discovery_order(partitioned, tmp_path, monkeypatch):
    """A different os.walk order changes the derived split but not the replayed one."""
    replayed = tmp_path / "r3"
    real_walk = os.walk

    def reversed_walk(top, *a, **kw):
        for root, dirs, files in real_walk(top, *a, **kw):
            yield root, list(reversed(dirs)), list(reversed(files))

    monkeypatch.setattr(os, "walk", reversed_walk)
    run_split(partitioned["raw"], replayed, from_manifest=str(partitioned["archive"]),
              clean=True, verify=True)
    for split in SPLITS:
        assert manifest_md5(replayed, split) == manifest_md5(partitioned["out"], split)


def test_replay_reports_missing_source_images(partitioned, tmp_path):
    empty = tmp_path / "empty_raw"
    (empty / "Fire").mkdir(parents=True)
    (empty / "No_Fire").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as exc:
        ds.replay_manifests(str(partitioned["archive"]), str(empty), str(tmp_path / "r4"),
                            clean=True, quiet=True)
    assert "missing" in str(exc.value)


def test_replay_without_an_archive_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ds.replay_manifests(str(tmp_path / "nope"), str(tmp_path), str(tmp_path / "out"))


def test_replayed_stats_keep_the_committed_class_counts(partitioned, tmp_path):
    replayed = tmp_path / "r5"
    stats = run_split(partitioned["raw"], replayed, from_manifest=str(partitioned["archive"]),
                      clean=True)
    for split in SPLITS:
        original = partitioned["stats"][split]
        assert stats[split]["class_counts"] == original["class_counts"]
        for node in ("node_a", "node_b", "node_c"):
            assert stats[split][node] == original[node]


def test_plain_csv_archive_is_accepted(partitioned, tmp_path):
    """An un-gzipped manifest directory replays identically (easier to inspect by hand)."""
    plain = tmp_path / "plain"
    plain.mkdir()
    for split in SPLITS:
        # newline="" on both sides: the replay copies the archive verbatim, so a
        # CRLF->LF translation here would (correctly) change the manifest digest
        with gzip.open(os.path.join(str(partitioned["archive"]), split + ".csv.gz"),
                       "rt", newline="") as f:
            with open(os.path.join(str(plain), split + ".csv"), "w", newline="") as out:
                out.write(f.read())
    replayed = tmp_path / "r6"
    run_split(partitioned["raw"], replayed, from_manifest=str(plain), clean=True)
    for split in SPLITS:
        assert manifest_md5(replayed, split) == manifest_md5(partitioned["out"], split)


def test_committed_flame_archive_matches_the_recorded_digests():
    """The archive tracked in data/splits must still match analysis/leakage/P0_SUMMARY.md."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    archive_dir = os.path.join(repo, "data", "splits")
    summary = os.path.join(repo, "analysis", "leakage", "P0_SUMMARY.md")
    if not (os.path.isdir(archive_dir) and os.path.isfile(summary)):
        pytest.skip("FLAME split archive or P0 summary not present")
    import re
    recorded = dict(re.findall(r"^\| (\S+)/manifest\.csv \| \d+ \| ([0-9a-f]{32}) \|",
                               open(summary, encoding="utf-8").read(), flags=re.M))
    assert recorded, "no manifest digests in P0_SUMMARY.md"
    for split, md5 in recorded.items():
        path = os.path.join(archive_dir, split + ".csv.gz")
        assert os.path.isfile(path), path
        with gzip.open(path, "rb") as f:
            assert hashlib.md5(f.read()).hexdigest() == md5, split
