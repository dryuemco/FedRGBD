"""Tests for scripts/analyze_flame_leakage.py.

Builds synthetic near-duplicate "photo bursts" (a smooth low-frequency base
image plus a small brightness shift and a 1-pixel translation, which is what
dHash is designed to tolerate) under a fake ``raw/Fire`` / ``raw/No_Fire``
tree, and checks that the perceptual-hash clustering pipeline recovers the
exact grouping, that the group files round-trip, and that the processed-tree
leakage audit correctly flags train/test near-duplicates.
"""

import itertools
import json
import os
import shutil
from collections import defaultdict

import numpy as np
import pytest
from PIL import Image

from scripts.analyze_flame_leakage import (
    audit_processed,
    cluster_hash_columns,
    cluster_hashes,
    compute_hash_columns,
    compute_hashes,
    dataset_statistics,
    find_images,
    format_sweep_table,
    hamming_matrix,
    hash_image,
    load_group_file,
    near_duplicate_pairs,
    run,
    sequence_heuristic,
    write_example_groups,
    write_group_files,
)

N_BASES = 12
THRESHOLD = 8


# --------------------------------------------------------------------------- #
# synthetic raw dataset
# --------------------------------------------------------------------------- #
def _low_freq_base(rs, size=48, small=6):
    """A smooth low-frequency grayscale image (stable under dHash)."""
    low = rs.randint(40, 216, size=(small, small)).astype(np.uint8)
    img = Image.fromarray(low, mode="L").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _save_rgb(arr2d, path):
    rgb = np.stack([arr2d] * 3, axis=-1)
    Image.fromarray(rgb, mode="RGB").save(path, quality=95)


def _build_raw_dataset(root):
    """Create N_BASES near-duplicate triplets (+singletons) under root/Fire, root/No_Fire.

    Returns ``(raw_dir, expected_group_of)`` where ``expected_group_of`` maps
    every *relative* path of a non-singleton image to its base index (int).
    """
    raw_dir = os.path.join(root, "raw")
    fire_dir = os.path.join(raw_dir, "Fire")
    nofire_dir = os.path.join(raw_dir, "No_Fire")
    os.makedirs(fire_dir, exist_ok=True)
    os.makedirs(nofire_dir, exist_ok=True)

    rs = np.random.RandomState(0)
    expected_group_of = {}

    for b in range(N_BASES):
        cls_dir, cls_name = (fire_dir, "Fire") if b < N_BASES // 2 else (nofire_dir, "No_Fire")
        arr = _low_freq_base(rs)

        p0 = f"base{b}_orig.jpg"
        _save_rgb(arr, os.path.join(cls_dir, p0))

        arr_bright = np.clip(arr.astype(np.int16) + 5, 0, 255).astype(np.uint8)
        p1 = f"base{b}_bright.jpg"
        _save_rgb(arr_bright, os.path.join(cls_dir, p1))

        arr_shift = np.pad(arr, ((0, 0), (1, 0)), mode="edge")[:, :-1]
        p2 = f"base{b}_shift.jpg"
        _save_rgb(arr_shift, os.path.join(cls_dir, p2))

        for p in (p0, p1, p2):
            expected_group_of[f"{cls_name}/{p}"] = b

    # unrelated singletons: high-frequency noise, far from every base in hash space
    for s in range(4):
        noise = rs.randint(0, 256, size=(48, 48)).astype(np.uint8)
        cls_dir, cls_name = (fire_dir, "Fire") if s % 2 == 0 else (nofire_dir, "No_Fire")
        _save_rgb(noise, os.path.join(cls_dir, f"singleton{s}.jpg"))

    return raw_dir, expected_group_of


@pytest.fixture
def flame_raw(tmp_path):
    return _build_raw_dataset(str(tmp_path))


# --------------------------------------------------------------------------- #
# hamming_matrix / near_duplicate_pairs on hand-made hashes
# --------------------------------------------------------------------------- #
def test_hamming_matrix_handmade():
    a = np.array([0, 1, 3], dtype=np.uint64)
    b = np.array([0, 2, 255], dtype=np.uint64)
    got = hamming_matrix(a, b)
    expected = np.array([[bin(x ^ y).count("1") for y in b] for x in a], dtype=np.uint8)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == np.uint8


def test_hamming_matrix_self_zero_diagonal():
    rs = np.random.RandomState(1)
    a = rs.randint(0, 2**63, size=10, dtype=np.int64).astype(np.uint64)
    got = hamming_matrix(a, a)
    assert np.all(np.diag(got) == 0)


def _brute_force_pairs(hashes, threshold):
    n = len(hashes)
    pairs = set()
    for i, j in itertools.combinations(range(n), 2):
        d = bin(int(hashes[i]) ^ int(hashes[j])).count("1")
        if d <= threshold:
            pairs.add((i, j))
    return pairs


def _pairs_as_set(rows, cols):
    return {(min(int(i), int(j)), max(int(i), int(j))) for i, j in zip(rows, cols)}


def _components_from_pairs(n, pairs):
    """Union-find connected components from an edge set -> component id per node."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return [find(i) for i in range(n)]


def _same_partition(labels_a, labels_b):
    n = len(labels_a)
    for i in range(n):
        for j in range(n):
            if (labels_a[i] == labels_a[j]) != (labels_b[i] == labels_b[j]):
                return False
    return True


@pytest.mark.parametrize("threshold", [0, 4, 8, 16])
def test_near_duplicate_pairs_matches_bruteforce_random(threshold):
    # near_duplicate_pairs is not required to return every pairwise edge (it
    # uses multi-index hashing to avoid an O(N^2) scan and only returns enough
    # edges to reconstruct connectivity), so we check (a) every returned edge
    # is actually within `threshold` (soundness) and (b) the connected
    # components it induces match the brute-force ground truth (completeness).
    rs = np.random.RandomState(2)
    hashes = rs.randint(0, 2**63, size=40, dtype=np.int64).astype(np.uint64)
    rows, cols = near_duplicate_pairs(hashes, threshold)
    got = _pairs_as_set(rows, cols)
    for i, j in got:
        assert bin(int(hashes[i]) ^ int(hashes[j])).count("1") <= threshold

    expected = _brute_force_pairs(hashes, threshold)
    got_labels = _components_from_pairs(len(hashes), got)
    expected_labels = _components_from_pairs(len(hashes), expected)
    assert _same_partition(got_labels, expected_labels)


def test_near_duplicate_pairs_exact_duplicate_collapsing():
    # indices 0,1,2 share the exact same hash (0); index 3 (value=1) is 1 bit
    # from that; index 4 (value=3) is 1 bit from index 3 (transitively chained
    # into the same component even though it is 2 bits from the hash 0
    # cluster directly); index 5 (value=255) is unrelated to everything.
    hashes = np.array([0, 0, 0, 1, 3, 255], dtype=np.uint64)
    rows, cols = near_duplicate_pairs(hashes, threshold=1)
    got = _pairs_as_set(rows, cols)
    for i, j in got:
        assert bin(int(hashes[i]) ^ int(hashes[j])).count("1") <= 1

    expected = _brute_force_pairs(hashes, threshold=1)
    got_labels = _components_from_pairs(6, got)
    expected_labels = _components_from_pairs(6, expected)
    assert _same_partition(got_labels, expected_labels)
    # {0,1,2,3,4} form one (transitively chained) component; 5 is isolated.
    assert got_labels[0] == got_labels[1] == got_labels[2] == got_labels[3] == got_labels[4]
    assert got_labels[5] != got_labels[0]
    # exact-duplicate collapsing means we should NOT need the full O(k^2)
    # clique of edges among {0,1,2} to represent that component.
    assert len(got) < len(expected)


def test_near_duplicate_pairs_threshold_zero_only_exact():
    hashes = np.array([5, 5, 6, 100], dtype=np.uint64)
    rows, cols = near_duplicate_pairs(hashes, threshold=0)
    got = _pairs_as_set(rows, cols)
    assert got == {(0, 1)}


def test_near_duplicate_pairs_empty_input():
    rows, cols = near_duplicate_pairs(np.zeros(0, dtype=np.uint64), threshold=8)
    assert len(rows) == 0
    assert len(cols) == 0


def test_cluster_hashes_deterministic():
    rs = np.random.RandomState(3)
    hashes = rs.randint(0, 2**63, size=60, dtype=np.int64).astype(np.uint64)
    g1 = cluster_hashes(hashes, THRESHOLD)
    g2 = cluster_hashes(hashes, THRESHOLD)
    np.testing.assert_array_equal(g1, g2)


def test_cluster_hashes_empty():
    out = cluster_hashes(np.zeros(0, dtype=np.uint64), THRESHOLD)
    assert len(out) == 0


def test_cluster_hashes_groups_by_component():
    hashes = np.array([0, 0, 1, 3, 500, 500], dtype=np.uint64)
    gids = cluster_hashes(hashes, threshold=1)
    # {0,1,2,3} one component, {4,5} another
    assert gids[0] == gids[1] == gids[2] == gids[3]
    assert gids[4] == gids[5]
    assert gids[0] != gids[4]


# --------------------------------------------------------------------------- #
# end-to-end grouping on the synthetic raw dataset
# --------------------------------------------------------------------------- #
def test_run_groups_bases_with_their_duplicates_only(flame_raw, tmp_path):
    raw_dir, expected_group_of = flame_raw
    out_dir = tmp_path / "leak_out"
    report = run(data_dir=raw_dir, output_dir=str(out_dir), threshold=THRESHOLD,
                 verbose=False, use_cache=False)

    mapping = load_group_file(str(out_dir / "groups.json"))

    # every expected (non-singleton) path is present
    assert set(expected_group_of) <= set(mapping)

    # each base's triplet maps to exactly one discovered group id, and no two
    # different bases share a group id.
    base_to_gids = defaultdict(set)
    gid_to_bases = defaultdict(set)
    for rel, base in expected_group_of.items():
        gid = mapping[rel]
        base_to_gids[base].add(gid)
        gid_to_bases[gid].add(base)
    for base, gids in base_to_gids.items():
        assert len(gids) == 1, f"base {base} split across groups {gids}"
    for gid, bases in gid_to_bases.items():
        assert len(bases) == 1, f"group {gid} mixes bases {bases}"

    stats = report["dataset"]
    assert stats["n_images"] == N_BASES * 3 + 4
    assert stats["n_nontrivial_groups"] == N_BASES
    assert stats["images_in_nontrivial_groups"] == N_BASES * 3
    assert stats["largest_group"] == 3
    assert stats["cross_label_groups"] == 0


def test_groups_json_and_csv_written_and_consistent(flame_raw, tmp_path):
    raw_dir, expected_group_of = flame_raw
    out_dir = tmp_path / "leak_out2"
    run(data_dir=raw_dir, output_dir=str(out_dir), threshold=THRESHOLD, verbose=False, use_cache=False)

    json_path = out_dir / "groups.json"
    csv_path = out_dir / "groups.csv"
    assert json_path.exists()
    assert csv_path.exists()

    map_json = load_group_file(str(json_path))
    map_csv = load_group_file(str(csv_path))

    # csv includes every image (incl. singletons); json only non-singleton groups
    images = find_images(raw_dir)
    assert len(map_csv) == len(images)
    assert set(map_json) <= set(map_csv)

    # non-singleton grouping agrees between the two files (ids may be permuted,
    # so compare partitions, not raw id values)
    for rel in map_json:
        base = expected_group_of[rel]
        same_base_paths = {r for r, b in expected_group_of.items() if b == base}
        assert {r for r in same_base_paths} == {
            r for r in map_json if map_json[r] == map_json[rel]
        }
        assert map_csv[rel] == map_json[rel]


def test_dataset_statistics_direct(flame_raw):
    raw_dir, expected_group_of = flame_raw
    images = find_images(raw_dir)
    paths = [os.path.join(raw_dir, p) for p, _ in images]
    hashes = compute_hashes(paths, "dhash", 8, workers=1, progress=False)
    group_ids = cluster_hashes(hashes, THRESHOLD)
    stats = dataset_statistics(images, group_ids, hashes)

    assert stats["n_images"] == len(images)
    assert stats["n_nontrivial_groups"] == N_BASES
    assert stats["images_in_nontrivial_groups"] == N_BASES * 3
    assert stats["fraction_images_with_near_duplicate"] == pytest.approx(
        (N_BASES * 3) / len(images), abs=1e-4
    )
    assert stats["cross_label_groups"] == 0
    total_per_class = sum(v["n_images"] for v in stats["per_class"].values())
    assert total_per_class == len(images)


# --------------------------------------------------------------------------- #
# audit_processed: train/test leakage on a fake processed tree
# --------------------------------------------------------------------------- #
def _copy_raw(raw_dir, rel_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    fname = os.path.basename(rel_path)
    shutil.copy2(os.path.join(raw_dir, rel_path), os.path.join(dest_dir, fname))


def test_audit_processed_flags_cross_node_train_leak(flame_raw, tmp_path):
    raw_dir, expected_group_of = flame_raw
    images = find_images(raw_dir)
    paths = [os.path.join(raw_dir, p) for p, _ in images]
    hashes = compute_hashes(paths, "dhash", 8, workers=1, progress=False)
    group_ids = cluster_hashes(hashes, THRESHOLD)

    # --- leaky tree: base0's "shift" copy sits in node_a/test while base0's
    # "orig" copy sits in node_b/train -> a same-group, cross-node leak.
    leaky_root = tmp_path / "processed_leaky" / "iid"
    _copy_raw(raw_dir, "Fire/base1_orig.jpg", str(leaky_root / "node_a" / "train" / "Fire"))
    _copy_raw(raw_dir, "Fire/base2_orig.jpg", str(leaky_root / "node_a" / "train" / "Fire"))
    _copy_raw(raw_dir, "Fire/base3_orig.jpg", str(leaky_root / "node_a" / "test" / "Fire"))
    _copy_raw(raw_dir, "Fire/base0_shift.jpg", str(leaky_root / "node_a" / "test" / "Fire"))
    _copy_raw(raw_dir, "Fire/base0_orig.jpg", str(leaky_root / "node_b" / "train" / "Fire"))

    report = audit_processed(str(tmp_path / "processed_leaky"), raw_dir, images, group_ids)
    node_a_test = report["iid"]["nodes"]["node_a"]["test"]
    assert node_a_test["n"] == 2
    assert node_a_test["with_train_duplicate_same_node"] == 0
    assert node_a_test["with_train_duplicate_any_node"] == 1
    assert node_a_test["leak_rate_same_node"] == pytest.approx(0.0)
    assert node_a_test["leak_rate_any_node"] == pytest.approx(0.5)
    assert report["iid"]["global_test_leak_rate_any_node"] == pytest.approx(0.5)

    # --- leak-free tree: node_a's test images are unrelated to any train image.
    clean_root = tmp_path / "processed_clean" / "iid"
    _copy_raw(raw_dir, "Fire/base1_orig.jpg", str(clean_root / "node_a" / "train" / "Fire"))
    _copy_raw(raw_dir, "No_Fire/base9_orig.jpg", str(clean_root / "node_b" / "train" / "No_Fire"))
    _copy_raw(raw_dir, "Fire/base3_orig.jpg", str(clean_root / "node_a" / "test" / "Fire"))
    _copy_raw(raw_dir, "No_Fire/base10_orig.jpg", str(clean_root / "node_a" / "test" / "No_Fire"))

    clean_report = audit_processed(str(tmp_path / "processed_clean"), raw_dir, images, group_ids)
    clean_test = clean_report["iid"]["nodes"]["node_a"]["test"]
    assert clean_test["with_train_duplicate_same_node"] == 0
    assert clean_test["with_train_duplicate_any_node"] == 0
    assert clean_report["iid"]["global_test_leak_rate_any_node"] == pytest.approx(0.0)


def test_audit_processed_run_integration(flame_raw, tmp_path):
    """run() with processed_dir set should embed the same audit under the report."""
    raw_dir, _ = flame_raw
    images = find_images(raw_dir)
    hashes = compute_hashes([os.path.join(raw_dir, p) for p, _ in images], "dhash", 8, 1, False)
    group_ids = cluster_hashes(hashes, THRESHOLD)

    processed_root = tmp_path / "processed" / "iid"
    _copy_raw(raw_dir, "Fire/base1_orig.jpg", str(processed_root / "node_a" / "train" / "Fire"))
    _copy_raw(raw_dir, "Fire/base3_orig.jpg", str(processed_root / "node_a" / "test" / "Fire"))

    out_dir = tmp_path / "leak_out3"
    report = run(data_dir=raw_dir, output_dir=str(out_dir), threshold=THRESHOLD,
                 processed_dir=str(tmp_path / "processed"), verbose=False, use_cache=False)
    assert "processed_splits" in report
    assert "iid" in report["processed_splits"]
    assert report["processed_splits"]["iid"]["nodes"]["node_a"]["test"]["with_train_duplicate_any_node"] == 0


# --------------------------------------------------------------------------- #
# ported from the authors' script: --hash both (union of dHash and pHash edges)
# --------------------------------------------------------------------------- #
def _partition_of(mapping):
    """``{rel: gid}`` -> ``{frozenset(members), ...}`` (id values are irrelevant)."""
    members = defaultdict(set)
    for rel, gid in mapping.items():
        members[gid].add(rel)
    return {frozenset(v) for v in members.values()}


def test_hash_both_is_a_superset_of_the_dhash_grouping(flame_raw, tmp_path):
    raw_dir, expected_group_of = flame_raw
    d_dir = tmp_path / "out_dhash"
    b_dir = tmp_path / "out_both"
    run(data_dir=raw_dir, output_dir=str(d_dir), method="dhash", threshold=THRESHOLD,
        verbose=False, use_cache=False)
    report = run(data_dir=raw_dir, output_dir=str(b_dir), method="both", threshold=THRESHOLD,
                 phash_threshold=10, verbose=False, use_cache=False)

    # groups.csv lists every image (incl. singletons), so the two partitions
    # cover the same set of paths and can be compared directly.
    dhash_map = load_group_file(str(d_dir / "groups.csv"))
    both_map = load_group_file(str(b_dir / "groups.csv"))
    assert set(dhash_map) == set(both_map)

    # union of edges => every dHash group is *contained* in one union group
    for group in _partition_of(dhash_map):
        assert len({both_map[rel] for rel in group}) == 1, (
            "dHash group {} was split by --hash both".format(sorted(group)))
    assert report["settings"]["method"] == "both"
    assert report["settings"]["phash_threshold"] == 10
    assert report["settings"]["union_of"] == ["dhash", "phash"]
    # the near-duplicate triplets survive the union grouping
    for base in set(expected_group_of.values()):
        rels = [r for r, b in expected_group_of.items() if b == base]
        assert len({both_map[r] for r in rels}) == 1


def test_hash_both_caches_both_hash_arrays(flame_raw, tmp_path):
    raw_dir, _ = flame_raw
    out_dir = tmp_path / "out_both_cache"
    run(data_dir=raw_dir, output_dir=str(out_dir), method="both", verbose=False, use_cache=True)
    assert (out_dir / "hashes_dhash8.npz").exists()
    assert (out_dir / "hashes_phash8.npz").exists()
    # a second (cached) run must reproduce the grouping exactly
    first = load_group_file(str(out_dir / "groups.csv"))
    run(data_dir=raw_dir, output_dir=str(out_dir), method="both", verbose=False, use_cache=True)
    assert _partition_of(load_group_file(str(out_dir / "groups.csv"))) == _partition_of(first)


def test_cluster_hash_columns_union_of_edges():
    # dhash links 0-1, phash links 1-2 -> one component {0,1,2}
    dh = np.array([0, 0, 255], dtype=np.uint64)
    ph = np.array([7, 0, 0], dtype=np.uint64)
    gids = cluster_hash_columns({"dhash": dh, "phash": ph}, {"dhash": 0, "phash": 0})
    assert gids[0] == gids[1] == gids[2]
    only_dhash = cluster_hashes(dh, 0)
    assert only_dhash[2] != only_dhash[0]


# --------------------------------------------------------------------------- #
# --sweep: threshold sensitivity (must not change groups.json)
# --------------------------------------------------------------------------- #
def test_threshold_sweep_reported_and_monotonic(flame_raw, tmp_path):
    raw_dir, _ = flame_raw
    plain = tmp_path / "out_plain"
    swept = tmp_path / "out_sweep"
    run(data_dir=raw_dir, output_dir=str(plain), threshold=THRESHOLD, verbose=False, use_cache=False)
    report = run(data_dir=raw_dir, output_dir=str(swept), threshold=THRESHOLD,
                 sweep=[4, 6, 8, 10, 12], verbose=False, use_cache=False)

    rows = report["threshold_sweep"]
    assert [r["threshold"] for r in rows] == [4, 6, 8, 10, 12]
    for row in rows:
        for key in ("n_groups", "n_nontrivial_groups", "images_in_nontrivial_groups",
                    "largest_group", "fraction_with_near_duplicate"):
            assert key in row
    # raising the threshold can only merge groups
    assert all(a["n_groups"] >= b["n_groups"] for a, b in zip(rows, rows[1:]))
    assert all(a["fraction_with_near_duplicate"] <= b["fraction_with_near_duplicate"]
               for a, b in zip(rows, rows[1:]))
    # the sweep does not change which threshold produces groups.json
    assert _partition_of(load_group_file(str(swept / "groups.csv"))) == \
        _partition_of(load_group_file(str(plain / "groups.csv")))
    # ... and the reported --threshold row agrees with the written grouping
    at_8 = next(r for r in rows if r["threshold"] == THRESHOLD)
    assert at_8["n_nontrivial_groups"] == report["dataset"]["n_nontrivial_groups"]
    assert at_8["largest_group"] == report["dataset"]["largest_group"]
    # and it is printed as a table
    assert "thr" in format_sweep_table(rows)
    assert format_sweep_table(rows).count("\n") == len(rows) + 1


# --------------------------------------------------------------------------- #
# MD5 exact-duplicate count
# --------------------------------------------------------------------------- #
def test_md5_exact_duplicate_count(flame_raw, tmp_path):
    raw_dir, _ = flame_raw
    shutil.copy2(os.path.join(raw_dir, "Fire", "base0_orig.jpg"),
                 os.path.join(raw_dir, "Fire", "base0_bytecopy.jpg"))
    report = run(data_dir=raw_dir, output_dir=str(tmp_path / "out_md5"), threshold=THRESHOLD,
                 verbose=False, use_cache=False)
    stats = report["dataset"]
    assert stats["exact_duplicate_files_md5"] == 1
    assert stats["n_unique_md5"] == stats["n_images"] - 1

    no_md5 = run(data_dir=raw_dir, output_dir=str(tmp_path / "out_nomd5"), threshold=THRESHOLD,
                 md5=False, verbose=False, use_cache=False)
    assert "exact_duplicate_files_md5" not in no_md5["dataset"]


def test_md5_is_cached_and_computed_in_the_hashing_pass(flame_raw, tmp_path):
    raw_dir, _ = flame_raw
    out_dir = tmp_path / "out_md5_cache"
    run(data_dir=raw_dir, output_dir=str(out_dir), verbose=False, use_cache=True)
    assert (out_dir / "md5s.npz").exists()
    images = find_images(raw_dir)
    columns, md5s, failures = compute_hash_columns(
        [os.path.join(raw_dir, p) for p, _ in images], ["dhash", "phash"], 8, 1, False, True)
    assert not failures
    assert set(columns) == {"dhash", "phash"}
    assert len(md5s) == len(images)
    # one pass, same values as the single-method helper
    np.testing.assert_array_equal(
        columns["dhash"], compute_hashes([os.path.join(raw_dir, p) for p, _ in images],
                                         "dhash", 8, 1, False))


# --------------------------------------------------------------------------- #
# filename frame-number heuristic
# --------------------------------------------------------------------------- #
def test_sequence_heuristic_on_handmade_frame_numbers():
    images = [("Fire/frame_0001.jpg", "Fire"), ("Fire/frame_0002.jpg", "Fire"),
              ("Fire/frame_0003.jpg", "Fire"), ("Fire/frame_0009.jpg", "Fire"),
              ("No_Fire/clip_1.jpg", "No_Fire"), ("No_Fire/clip_2.jpg", "No_Fire"),
              ("No_Fire/no_number.jpg", "No_Fire")]
    # frames 1-3 are one near-duplicate group, frame 9 its own, clip_1/clip_2 separate
    group_ids = np.array([0, 0, 0, 1, 2, 3, 4], dtype=np.int64)
    out = sequence_heuristic(images, group_ids, gap=1)

    assert out["n_images_with_frame_number"] == 6      # 'no_number.jpg' has none
    assert out["n_consecutive_pairs"] == 3             # (1,2) (2,3) in Fire, (1,2) in No_Fire
    assert out["n_consecutive_pairs_same_group"] == 2
    assert out["fraction_consecutive_pairs_same_group"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["per_class"]["Fire"]["consecutive_pairs"] == 2
    assert out["per_class"]["Fire"]["fraction_consecutive_pairs_same_group"] == 1.0
    assert out["per_class"]["No_Fire"]["consecutive_pairs_same_group"] == 0
    assert out["per_class"]["Fire"]["min"] == 1 and out["per_class"]["Fire"]["max"] == 9

    # a larger gap picks up frame 3 -> 9 only once the gap covers it
    assert sequence_heuristic(images, group_ids, gap=6)["n_consecutive_pairs"] == 4


def test_sequence_heuristic_is_opt_in_and_wired_into_run(flame_raw, tmp_path):
    raw_dir, _ = flame_raw
    off = run(data_dir=raw_dir, output_dir=str(tmp_path / "seq_off"), verbose=False, use_cache=False)
    assert "sequence_heuristic" not in off
    on = run(data_dir=raw_dir, output_dir=str(tmp_path / "seq_on"), sequence_heuristic_on=True,
             verbose=False, use_cache=False)
    seq = on["sequence_heuristic"]
    assert seq["gap"] == 1
    assert seq["n_images_with_frame_number"] == on["dataset"]["n_images"]
    assert 0.0 <= seq["fraction_consecutive_pairs_same_group"] <= 1.0
    # the grouping written to groups.json is unaffected by the diagnostic
    assert on["dataset"]["n_nontrivial_groups"] == off["dataset"]["n_nontrivial_groups"]


# --------------------------------------------------------------------------- #
# example_groups.txt
# --------------------------------------------------------------------------- #
def test_example_groups_lists_the_largest_groups(flame_raw, tmp_path):
    raw_dir, _ = flame_raw
    out_dir = tmp_path / "out_examples"
    report = run(data_dir=raw_dir, output_dir=str(out_dir), threshold=THRESHOLD,
                 examples=3, verbose=False, use_cache=False)
    path = out_dir / "example_groups.txt"
    assert report["example_groups"] == str(path)
    text = path.read_text()
    assert text.count("# rank ") == 3

    mapping = load_group_file(str(out_dir / "groups.csv"))
    blocks = [b for b in text.split("\n\n") if b.startswith("# rank ")]
    assert len(blocks) == 3
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln and not ln.startswith("#")]
        assert len(lines) == 3                       # the synthetic triplets
        assert len({mapping[ln] for ln in lines}) == 1

    # --examples 0 still writes the file, with no group in it
    run(data_dir=raw_dir, output_dir=str(out_dir), threshold=THRESHOLD, examples=0,
        verbose=False, use_cache=False)
    assert path.read_text().count("# rank ") == 0


# --------------------------------------------------------------------------- #
# load_group_file: the authors' inverted {path: gid} layout
# --------------------------------------------------------------------------- #
STANDARD_GROUPS = {"0": ["Fire/a.jpg", "Fire/b.jpg"],
                   "1": ["Fire/c.jpg", "No_Fire/d.jpg"],
                   "2": ["No_Fire/e.jpg"]}


def _write_json(path, payload):
    with open(str(path), "w") as f:
        json.dump(payload, f)
    return str(path)


def test_load_group_file_inverted_layout_equals_standard_layout(tmp_path):
    standard = _write_json(tmp_path / "standard.json", {"groups": STANDARD_GROUPS})
    expected = load_group_file(standard)
    assert expected == {"Fire/a.jpg": 0, "Fire/b.jpg": 0, "Fire/c.jpg": 1,
                        "No_Fire/d.jpg": 1, "No_Fire/e.jpg": 2}

    inverted = {rel: int(gid) for gid, members in STANDARD_GROUPS.items() for rel in members}

    # (a) the authors' layout: {"groups": {path: gid}} next to their metadata
    nested = _write_json(tmp_path / "inverted.json",
                         {"thresholds": {"phash": 10, "dhash": 8}, "n_groups": 3,
                          "groups": dict(inverted)})
    assert load_group_file(nested) == expected

    # (b) a bare top-level {path: gid} mapping (metadata keys ignored)
    flat = _write_json(tmp_path / "flat.json", dict(inverted, n_images=5, n_groups=3))
    assert load_group_file(flat) == expected


def test_load_group_file_inverted_absolute_paths(tmp_path):
    data_dir = tmp_path / "raw"
    (data_dir / "Fire").mkdir(parents=True)
    (data_dir / "No_Fire").mkdir(parents=True)
    expected = load_group_file(_write_json(tmp_path / "std.json", {"groups": STANDARD_GROUPS}))

    absolute = {os.path.join(str(data_dir), rel.replace("/", os.sep)): int(gid)
                for gid, members in STANDARD_GROUPS.items() for rel in members}
    path = _write_json(tmp_path / "abs.json", {"groups": absolute})

    # with --data_dir the absolute keys are relativised
    assert load_group_file(path, str(data_dir)) == expected
    # the data_dir recorded inside the file is used when the argument is omitted
    assert load_group_file(_write_json(tmp_path / "abs_meta.json",
                                       {"data_dir": str(data_dir),
                                        "groups": absolute})) == expected
    # without any data_dir the keys fall back to <ClassDir>/<basename>
    assert load_group_file(path) == expected
    # a path outside --data_dir also falls back to <ClassDir>/<basename>
    assert load_group_file(path, str(tmp_path / "somewhere_else")) == expected


def test_load_group_file_standard_layout_is_unchanged(flame_raw, tmp_path):
    raw_dir, _ = flame_raw
    out_dir = tmp_path / "out_roundtrip"
    run(data_dir=raw_dir, output_dir=str(out_dir), threshold=THRESHOLD, verbose=False,
        use_cache=False)
    mapping = load_group_file(str(out_dir / "groups.json"))
    inverted = _write_json(tmp_path / "inv.json", {"groups": {k: int(v) for k, v in mapping.items()}})
    assert load_group_file(inverted) == mapping
    assert load_group_file(inverted, raw_dir) == mapping
