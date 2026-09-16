"""Tests for ``src/data/data_splitter.py``.

Covers the three new features (group-aware splitting, Dirichlet label skew,
per-node subsampling) and — most importantly — pins the *default* code path to
the original image-level implementation taken from ``git show
main:src/data/data_splitter.py``, so the published FedRGBD splits stay
reproducible.

Everything runs on CPU with ~60 synthetic 16x16 PNGs in ``tmp_path`` and uses
``--link_mode copy`` so the suite works on Windows without symlink privileges.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict

import pytest
from PIL import Image

from src.data import data_splitter as ds

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_PER_CLASS = 30
GROUPED = 24          # first 24 images of each class live in triplet groups
TRIPLET = 3
NOFIRE_GID_OFFSET = 7  # makes gid 7 span both classes -> exactly 1 cross-label group


# --------------------------------------------------------------------------- #
# synthetic dataset
# --------------------------------------------------------------------------- #
def _write_png(path, value):
    Image.new("RGB", (16, 16), (value % 256, (value * 7) % 256, (value * 13) % 256)).save(path)


@pytest.fixture
def dataset(tmp_path):
    """``tmp/raw/{Fire,No_Fire}`` with 30 tiny PNGs each + a triplet group file."""
    raw = tmp_path / "raw"
    names = {}
    for cls, prefix in (("Fire", "fire"), ("No_Fire", "nofire")):
        d = raw / cls
        d.mkdir(parents=True)
        names[cls] = []
        for i in range(N_PER_CLASS):
            fname = "{}_{:03d}.png".format(prefix, i)
            _write_png(d / fname, i if cls == "Fire" else i + 100)
            names[cls].append(fname)

    groups = defaultdict(list)
    for i, fname in enumerate(names["Fire"][:GROUPED]):
        groups[i // TRIPLET].append("Fire/" + fname)
    for i, fname in enumerate(names["No_Fire"][:GROUPED]):
        groups[i // TRIPLET + NOFIRE_GID_OFFSET].append("No_Fire/" + fname)

    group_file = tmp_path / "groups.json"
    group_file.write_text(json.dumps({
        "n_images": 2 * N_PER_CLASS,
        "groups": {str(g): sorted(m) for g, m in sorted(groups.items())},
    }))

    return {"raw": str(raw), "group_file": str(group_file), "root": tmp_path, "names": names}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def read_manifest(output_dir, split):
    """manifest.csv -> list of dict rows."""
    path = os.path.join(output_dir, split, "manifest.csv")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def assignment_from_manifest(output_dir, split):
    """-> ``{(node, split, label): sorted([rel_path, ...])}``"""
    out = defaultdict(list)
    for row in read_manifest(output_dir, split):
        out[(row["node"], row["split"], row["label"])].append(row["path"])
    return {k: sorted(v) for k, v in out.items()}


def counts_from_manifest(output_dir, split):
    """-> ``{(node, split, label): n}``"""
    return {k: len(v) for k, v in assignment_from_manifest(output_dir, split).items()}


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


# --------------------------------------------------------------------------- #
# 1. the default path must reproduce the ORIGINAL implementation exactly
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def original_module(tmp_path_factory):
    """Import ``main:src/data/data_splitter.py`` (pre-rewrite) from a temp file."""
    try:
        src = subprocess.check_output(
            ["git", "show", "main:src/data/data_splitter.py"], cwd=REPO_ROOT)
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        pytest.skip("cannot read the original implementation from git: {}".format(exc))

    path = tmp_path_factory.mktemp("orig") / "orig_data_splitter.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("orig_data_splitter", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_assignment(original_module, data_dir, output_dir, seed, nodes, monkeypatch):
    """Run the original ``main()`` with ``link_files`` replaced by a recorder.

    Returns ``{(split, node, tvt, label): sorted([abspath, ...])}`` — the exact
    file assignment the published splits were produced with.
    """
    recorded = defaultdict(list)

    def recorder(file_list, dest_dir, class_name):
        parts = os.path.normpath(dest_dir).split(os.sep)
        split_name, node, tvt = parts[-3], parts[-2], parts[-1]
        recorded[(split_name, node, tvt, class_name)].extend(
            os.path.abspath(p) for p in file_list)

    monkeypatch.setattr(original_module, "link_files", recorder)
    monkeypatch.setattr(sys, "argv", [
        "data_splitter.py", "--data_dir", str(data_dir), "--output_dir", str(output_dir),
        "--seed", str(seed), "--nodes", str(nodes)])
    original_module.main()
    return {k: sorted(v) for k, v in recorded.items()}


def new_assignment(data_dir, output_dir, splits=("iid", "non_iid_label")):
    """Same shape as :func:`original_assignment`, read back from manifest.csv."""
    out = defaultdict(list)
    for split in splits:
        for row in read_manifest(output_dir, split):
            full = os.path.abspath(os.path.join(str(data_dir), row["path"]))
            out[(split, row["node"], row["split"], row["label"])].append(full)
    return {k: sorted(v) for k, v in out.items()}


@pytest.mark.parametrize("nodes", [2, 3])
def test_default_path_identical_to_original(dataset, tmp_path, monkeypatch,
                                            original_module, nodes):
    seed = 42
    orig_out = tmp_path / "out_orig_{}".format(nodes)
    new_out = tmp_path / "out_new_{}".format(nodes)

    expected = original_assignment(original_module, dataset["raw"], orig_out,
                                   seed, nodes, monkeypatch)
    run_split(dataset["raw"], new_out, seed=seed, nodes=nodes)
    actual = new_assignment(dataset["raw"], new_out)

    assert expected, "the original implementation recorded nothing"
    assert set(expected) == set(actual)
    for key in sorted(expected):
        assert expected[key] == actual[key], "mismatch at {}".format(key)

    # and the reported statistics must match too
    orig_stats = json.loads((orig_out / "split_stats.json").read_text())
    new_stats = json.loads((new_out / "split_stats.json").read_text())
    for split in ("iid", "non_iid_label"):
        for node in orig_stats[split]:
            assert orig_stats[split][node] == new_stats[split][node]


# --------------------------------------------------------------------------- #
# 2. --group_file: no group crosses a node or a train/val/test boundary
# --------------------------------------------------------------------------- #
def test_group_file_keeps_groups_together(dataset, tmp_path):
    out = tmp_path / "out_groups"
    stats = run_split(dataset["raw"], out, seed=7, nodes=3,
                      group_file=dataset["group_file"],
                      dirichlet_alpha=[0.5], dirichlet_min_size=1)

    assert stats["_meta"]["grouped_images_matched"] == 2 * GROUPED
    assert stats["_meta"]["cross_label_groups"] == 1  # gid 7 spans Fire and No_Fire

    splits = ["iid", "non_iid_label", "dirichlet_0.5"]
    for split in splits:
        rows = read_manifest(out, split)
        assert len(rows) == 2 * N_PER_CLASS
        # a unit is (group_id, label): a cross-label group is deliberately two units
        nodes_of = defaultdict(set)
        tvt_of = defaultdict(set)
        for row in rows:
            if row["group_id"].startswith("singleton"):
                continue
            key = (row["group_id"], row["label"])
            nodes_of[key].add(row["node"])
            tvt_of[key].add(row["split"])
        assert nodes_of, "no grouped images in split {}".format(split)
        for key, nodeset in nodes_of.items():
            assert len(nodeset) == 1, "{}: group {} spans nodes {}".format(split, key, nodeset)
        for key, tvtset in tvt_of.items():
            assert len(tvtset) == 1, "{}: group {} spans splits {}".format(split, key, tvtset)

        # every grouped triplet stayed whole
        sizes = defaultdict(int)
        for row in rows:
            if not row["group_id"].startswith("singleton"):
                sizes[(row["group_id"], row["label"])] += 1
        assert sorted(sizes.values()) == [TRIPLET] * len(sizes)


def test_group_file_csv_and_fallback_matching(dataset, tmp_path):
    """CSV group files and ``<ClassDir>/<basename>`` fallback matching both work."""
    csv_path = tmp_path / "groups.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "group_id"])
        for i, fname in enumerate(dataset["names"]["Fire"][:GROUPED]):
            # deliberately prefixed path -> only the <ClassDir>/<basename> fallback matches
            w.writerow(["Training/Fire/" + fname, i // TRIPLET])

    out = tmp_path / "out_csv"
    stats = run_split(dataset["raw"], out, seed=3, nodes=2, group_file=str(csv_path))
    assert stats["_meta"]["grouped_images_matched"] == GROUPED
    assert stats["_meta"]["largest_unit"] == TRIPLET


# --------------------------------------------------------------------------- #
# 3. --dirichlet_alpha
# --------------------------------------------------------------------------- #
def test_dirichlet_determinism_and_class_counts(dataset, tmp_path):
    kw = dict(nodes=3, skip_base_splits=True, dirichlet_alpha=[0.1, 1.0],
              dirichlet_min_size=1)
    a = tmp_path / "d_a"
    b = tmp_path / "d_b"
    c = tmp_path / "d_c"
    stats_a = run_split(dataset["raw"], a, seed=42, **kw)
    run_split(dataset["raw"], b, seed=42, **kw)
    run_split(dataset["raw"], c, seed=1234, **kw)

    assert sorted(k for k in stats_a if not k.startswith("_")) == ["dirichlet_0.1", "dirichlet_1"]

    for split in ("dirichlet_0.1", "dirichlet_1"):
        assert assignment_from_manifest(a, split) == assignment_from_manifest(b, split)
    assert any(assignment_from_manifest(a, s) != assignment_from_manifest(c, s)
               for s in ("dirichlet_0.1", "dirichlet_1"))

    for split in ("dirichlet_0.1", "dirichlet_1"):
        block = stats_a[split]
        table = block["class_counts"]
        assert set(table) == {"node_a", "node_b", "node_c"}
        assert sum(v["Fire"] for v in table.values()) == N_PER_CLASS
        assert sum(v["No_Fire"] for v in table.values()) == N_PER_CLASS
        assert sum(v["total"] for v in table.values()) == 2 * N_PER_CLASS
        assert set(block["dirichlet_proportions"]) == {"Fire", "No_Fire"}
        for props in block["dirichlet_proportions"].values():
            assert len(props) == 3
            assert abs(sum(props) - 1.0) < 1e-5
        assert block["seed"] == 42
        # the per-node train/val/test stats still sum to the class_counts totals
        for node, counts in table.items():
            assert sum(block[node][sp]["total"] for sp in ("train", "val", "test")) == counts["total"]

    assert stats_a["dirichlet_0.1"]["dirichlet_alpha"] == 0.1
    assert stats_a["dirichlet_1"]["dirichlet_alpha"] == 1.0


def test_dirichlet_min_size_honoured(dataset, tmp_path):
    out = tmp_path / "d_min"
    stats = run_split(dataset["raw"], out, seed=42, nodes=3, skip_base_splits=True,
                      dirichlet_alpha=[0.1], dirichlet_min_size=8)
    for counts in stats["dirichlet_0.1"]["class_counts"].values():
        assert counts["total"] >= 8


def test_dirichlet_min_size_impossible_raises(dataset, tmp_path):
    with pytest.raises(ValueError) as err:
        run_split(dataset["raw"], tmp_path / "d_bad", seed=42, nodes=3,
                  skip_base_splits=True, dirichlet_alpha=[0.1],
                  dirichlet_min_size=1000)
    assert "dirichlet_min_size" in str(err.value)


def test_dirichlet_respects_groups_via_units(dataset, tmp_path):
    out = tmp_path / "d_groups"
    run_split(dataset["raw"], out, seed=5, nodes=2, skip_base_splits=True,
              group_file=dataset["group_file"], dirichlet_alpha=[0.3],
              dirichlet_min_size=1)
    nodes_of = defaultdict(set)
    for row in read_manifest(out, "dirichlet_0.3"):
        if not row["group_id"].startswith("singleton"):
            nodes_of[(row["group_id"], row["label"])].add(row["node"])
    assert nodes_of
    assert all(len(v) == 1 for v in nodes_of.values())


# --------------------------------------------------------------------------- #
# 4. --subsample_frac
# --------------------------------------------------------------------------- #
def test_subsample_train_only_and_deterministic(dataset, tmp_path):
    frac = 0.5
    a = tmp_path / "s_a"
    b = tmp_path / "s_b"
    kw = dict(seed=42, nodes=2, subsample_frac=[frac])
    stats = run_split(dataset["raw"], a, **kw)
    run_split(dataset["raw"], b, **kw)

    for base in ("iid", "non_iid_label"):
        sub = "{}_sub{:g}".format(base, frac)
        assert sub in stats
        full_counts = counts_from_manifest(a, base)
        sub_counts = counts_from_manifest(a, sub)

        for (node, tvt, label), n_full in full_counts.items():
            n_sub = sub_counts.get((node, tvt, label), 0)
            if tvt == "train":
                # without a group file every unit is one image -> exact
                assert n_sub == round(frac * n_full) or (n_full > 0 and n_sub == 1)
            else:
                assert n_sub == n_full, "val/test must stay full by default"

        # deterministic across runs
        assert assignment_from_manifest(a, sub) == assignment_from_manifest(b, sub)

        block = stats[sub]
        assert block["subsample_frac"] == frac
        assert block["subsample_splits"] == ["train"]
        assert block["source_split"] == base
        assert set(block["class_counts"]) == {"node_a", "node_b"}


def test_subsample_applies_to_dirichlet_and_unit_granularity(dataset, tmp_path):
    out = tmp_path / "s_dir"
    frac = 0.2
    stats = run_split(dataset["raw"], out, seed=42, nodes=2,
                      group_file=dataset["group_file"], dirichlet_alpha=[0.5],
                      dirichlet_min_size=1, subsample_frac=[frac])
    for base in ("iid", "non_iid_label", "dirichlet_0.5"):
        sub = "{}_sub{:g}".format(base, frac)
        assert sub in stats, sorted(stats)
        assert os.path.exists(os.path.join(out, sub, "manifest.csv"))
        full_counts = counts_from_manifest(out, base)
        sub_counts = counts_from_manifest(out, sub)
        for (node, tvt, label), n_full in full_counts.items():
            n_sub = sub_counts.get((node, tvt, label), 0)
            if tvt == "train":
                # unit granularity: within one (triplet-sized) unit of the target
                assert 1 <= n_sub <= frac * n_full + TRIPLET
            else:
                assert n_sub == n_full
        # groups still intact after subsampling
        nodes_of = defaultdict(set)
        tvt_of = defaultdict(set)
        for row in read_manifest(out, sub):
            if row["group_id"].startswith("singleton"):
                continue
            key = (row["group_id"], row["label"])
            nodes_of[key].add(row["node"])
            tvt_of[key].add(row["split"])
        assert all(len(v) == 1 for v in nodes_of.values())
        assert all(len(v) == 1 for v in tvt_of.values())


def test_subsample_can_reduce_val_and_test(dataset, tmp_path):
    out = tmp_path / "s_all"
    stats = run_split(dataset["raw"], out, seed=42, nodes=2, skip_base_splits=True,
                      dirichlet_alpha=[1.0], dirichlet_min_size=1,
                      subsample_frac=[0.5], subsample_splits=["train", "val", "test"])
    sub = "dirichlet_1_sub0.5"
    assert stats[sub]["subsample_splits"] == ["train", "val", "test"]
    full_total = sum(v["total"] for v in stats["dirichlet_1"]["class_counts"].values())
    sub_total = sum(v["total"] for v in stats[sub]["class_counts"].values())
    assert sub_total < full_total


# --------------------------------------------------------------------------- #
# 5. output contract: split_stats.json keys + manifest.csv + link modes
# --------------------------------------------------------------------------- #
def test_split_stats_and_manifest_contract(dataset, tmp_path):
    out = tmp_path / "out_contract"
    stats = run_split(dataset["raw"], out, seed=42, nodes=3,
                      group_file=dataset["group_file"],
                      dirichlet_alpha=[0.5], dirichlet_min_size=1,
                      subsample_frac=[0.1])

    on_disk = json.loads((out / "split_stats.json").read_text())
    assert on_disk == stats

    expected = {"iid", "non_iid_label", "dirichlet_0.5",
                "iid_sub0.1", "non_iid_label_sub0.1", "dirichlet_0.5_sub0.1"}
    assert expected <= set(stats)

    for split in expected:
        block = stats[split]
        assert "class_counts" in block
        for node in ("node_a", "node_b", "node_c"):
            assert node in block
            for tvt in ("train", "val", "test"):
                cell = block[node][tvt]
                assert set(cell) == {"fire", "nofire", "total", "fire_ratio"}
                assert cell["total"] == cell["fire"] + cell["nofire"]
                # the per-node/<split> directory tree exists with both class dirs
                for cls in ("Fire", "No_Fire"):
                    assert os.path.isdir(os.path.join(out, split, node, tvt, cls))
        manifest = os.path.join(out, split, "manifest.csv")
        assert os.path.exists(manifest)
        with open(manifest, newline="") as f:
            header = next(csv.reader(f))
        assert header == ["node", "split", "label", "path", "group_id"]

    meta = stats["_meta"]
    for key in ("seed", "nodes", "node_names", "data_dir", "n_fire", "n_nofire",
                "link_mode", "group_file", "grouped_images_matched",
                "cross_label_groups", "n_units", "largest_unit"):
        assert key in meta
    assert meta["n_fire"] == meta["n_nofire"] == N_PER_CLASS

    # the base splits still cover the whole dataset exactly once
    for split in ("iid", "non_iid_label", "dirichlet_0.5"):
        paths = [row["path"] for row in read_manifest(out, split)]
        assert len(paths) == len(set(paths)) == 2 * N_PER_CLASS


def test_copy_link_mode_produces_real_files(dataset, tmp_path):
    out = tmp_path / "out_copy"
    run_split(dataset["raw"], out, seed=42, nodes=2)
    placed = os.path.join(out, "iid", "node_a", "train", "Fire")
    files = sorted(os.listdir(placed))
    assert files
    with Image.open(os.path.join(placed, files[0])) as im:
        assert im.size == (16, 16)


# --------------------------------------------------------------------------- #
# unit-level invariants that guarantee the default path stays identical
# --------------------------------------------------------------------------- #
def test_take_units_equals_slicing_for_singletons():
    units = [[str(i)] for i in range(17)]
    for target in range(0, 18):
        taken, rest = ds.take_units(units, target)
        assert taken == units[:target]
        assert rest == units[target:]


def test_split_units_into_equals_split_into_for_singletons():
    for n_items in (0, 1, 5, 17, 60):
        items = [str(i) for i in range(n_items)]
        units = [[i] for i in items]
        for n in (2, 3):
            got = [[p for u in part for p in u] for part in ds.split_units_into(units, n)]
            assert got == ds.split_into(items, n)


def test_split_units_matches_split_list_for_singletons():
    items = [str(i) for i in range(37)]
    by_unit = ds.split_units([[i] for i in items], seed=42)
    by_path = ds.split_list(items, seed=42)
    for sp in ("train", "val", "test"):
        assert [p for u in by_unit[sp] for p in u] == by_path[sp]


def test_stable_seed_is_process_independent():
    code = ("import sys; sys.path.insert(0, %r);"
            "from src.data.data_splitter import _stable_seed;"
            "print(_stable_seed(42, 'iid|0.05'))" % REPO_ROOT)
    env = dict(os.environ, PYTHONHASHSEED="12345")
    got = subprocess.check_output([sys.executable, "-c", code], env=env).decode().strip()
    assert int(got) == ds._stable_seed(42, "iid|0.05")
