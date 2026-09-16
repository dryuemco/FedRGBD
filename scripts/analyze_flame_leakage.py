"""FedRGBD — Near-duplicate / leakage analysis for the FLAME dataset.

FLAME images are frames extracted from aerial videos, so consecutive frames
are near-identical.  A random image-level split therefore places near-copies
of test images in the training set, which inflates accuracy.  This script:

1. Computes a perceptual hash (dHash / pHash / aHash) of every image.
2. Clusters images whose hashes are within a Hamming distance ``--threshold``
   into *near-duplicate groups* (exact multi-index hashing + connected
   components; no pairwise O(N²) scan).
3. Writes ``groups.json`` / ``groups.csv`` which ``src/data/data_splitter.py
   --group_file`` consumes to keep whole groups on one node and in one split.
4. Optionally audits an existing ``data/processed`` tree and reports how many
   val/test images have a near-duplicate in *any* training split
   (``leakage_report.json``).

Usage
-----
    python3 scripts/analyze_flame_leakage.py \
        --data_dir data/raw/flame_dataset \
        --processed_dir data/processed \
        --output_dir analysis/leakage --threshold 8 --workers 4

    python3 src/data/data_splitter.py --group_file analysis/leakage/groups.json ...

Only NumPy, Pillow and SciPy are required.  Runs on CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
CLASS_DIRS = {"fire": "Fire", "no_fire": "No_Fire", "nofire": "No_Fire"}

# 8-bit popcount lookup table (numpy 1.26 has no bitwise_count)
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# --------------------------------------------------------------------------- #
# image discovery
# --------------------------------------------------------------------------- #
def find_images(data_dir: str) -> List[Tuple[str, str]]:
    """Return ``[(relative_path, class_name), ...]`` sorted for determinism.

    Mirrors ``data_splitter.find_images``: the class is the *parent directory*
    name (``Fire`` / ``No_Fire`` / ``nofire``, case-insensitive).
    """
    out = []
    data_dir = os.path.abspath(data_dir)
    for root, _dirs, files in os.walk(data_dir):
        cls = CLASS_DIRS.get(os.path.basename(root).lower())
        if cls is None:
            continue
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                rel = os.path.relpath(os.path.join(root, f), data_dir).replace(os.sep, "/")
                out.append((rel, cls))
    out.sort()
    return out


# --------------------------------------------------------------------------- #
# perceptual hashes
# --------------------------------------------------------------------------- #
def _bits_to_uint64(bits: np.ndarray) -> np.uint64:
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if bits.size != 64:
        raise ValueError(f"expected 64 bits, got {bits.size}")
    packed = np.packbits(bits)  # 8 bytes, big-endian bit order
    return np.uint64(int.from_bytes(packed.tobytes(), "big"))


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix (avoids a hard scipy dependency for hashing)."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0, :] /= np.sqrt(2.0)
    return m


def hash_array(gray: np.ndarray, method: str = "dhash", hash_size: int = 8) -> np.uint64:
    """Hash an already-resized float grayscale array.

    ``gray`` must have the shape expected by the method (see :func:`hash_image`).
    """
    if method == "dhash":  # gradient hash: (hash_size, hash_size+1)
        bits = gray[:, 1:] > gray[:, :-1]
    elif method == "ahash":  # average hash: (hash_size, hash_size)
        bits = gray > gray.mean()
    elif method == "phash":  # DCT hash: (4*hash_size, 4*hash_size)
        n = gray.shape[0]
        d = _dct_matrix(n)
        dct = d @ gray @ d.T
        low = dct[:hash_size, :hash_size]
        bits = low > np.median(low)
    else:
        raise ValueError(f"unknown hash method {method!r}")
    return _bits_to_uint64(bits)


def hash_input_size(method: str, hash_size: int) -> Tuple[int, int]:
    if method == "dhash":
        return (hash_size + 1, hash_size)  # PIL (width, height)
    if method == "ahash":
        return (hash_size, hash_size)
    if method == "phash":
        return (4 * hash_size, 4 * hash_size)
    raise ValueError(method)


def hash_image(path: str, method: str = "dhash", hash_size: int = 8) -> np.uint64:
    with Image.open(path) as im:
        im = im.convert("L").resize(hash_input_size(method, hash_size), Image.LANCZOS)
        gray = np.asarray(im, dtype=np.float64)
    return hash_array(gray, method, hash_size)


def _hash_worker(args) -> Tuple[int, int]:
    idx, path, method, hash_size = args
    try:
        return idx, int(hash_image(path, method, hash_size))
    except Exception as exc:  # corrupt file → sentinel, reported later
        print(f"  WARNING: could not hash {path}: {exc}", file=sys.stderr)
        return idx, -1


def compute_hashes(paths: Sequence[str], method: str, hash_size: int, workers: int = 1,
                   progress: bool = True) -> np.ndarray:
    """Hash every path → ``uint64`` array; unreadable files get hash 0 and are flagged."""
    n = len(paths)
    hashes = np.zeros(n, dtype=np.uint64)
    jobs = [(i, p, method, hash_size) for i, p in enumerate(paths)]
    t0 = time.perf_counter()
    if workers > 1:
        with Pool(workers) as pool:
            it = pool.imap_unordered(_hash_worker, jobs, chunksize=64)
            for k, (i, h) in enumerate(it, 1):
                hashes[i] = np.uint64(max(h, 0))
                if progress and k % 5000 == 0:
                    print(f"  hashed {k}/{n} ({time.perf_counter() - t0:.0f}s)")
    else:
        for k, job in enumerate(jobs, 1):
            i, h = _hash_worker(job)
            hashes[i] = np.uint64(max(h, 0))
            if progress and k % 5000 == 0:
                print(f"  hashed {k}/{n} ({time.perf_counter() - t0:.0f}s)")
    return hashes


# --------------------------------------------------------------------------- #
# near-duplicate clustering
# --------------------------------------------------------------------------- #
def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between two uint64 arrays → (len(a), len(b)) uint8."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    return _POPCOUNT8[x.view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1).astype(np.uint8)


def near_duplicate_pairs(hashes: np.ndarray, threshold: int, block: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """All index pairs ``(i, j), i<j`` with ``hamming(h_i, h_j) <= threshold``.

    Multi-index hashing: split the 64-bit hash into ``threshold+1`` chunks; by
    the pigeonhole principle two hashes within ``threshold`` bits share at
    least one chunk exactly, so candidates are found by exact bucketing on each
    chunk and verified with the full distance.  Exact-equal hashes are collapsed
    first so a burst of identical frames does not explode the candidate set.
    """
    hashes = np.asarray(hashes, dtype=np.uint64)
    n = len(hashes)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    uniq, inverse = np.unique(hashes, return_inverse=True)
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []

    if threshold > 0 and len(uniq) > 1:
        n_chunks = min(threshold + 1, 16)
        bits_per_chunk = 64 // n_chunks
        for c in range(n_chunks):
            shift = np.uint64(c * bits_per_chunk)
            mask = np.uint64((1 << bits_per_chunk) - 1) if c < n_chunks - 1 else np.uint64((1 << (64 - c * bits_per_chunk)) - 1)
            keys = (uniq >> shift) & mask
            order = np.argsort(keys, kind="stable")
            sorted_keys = keys[order]
            starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
            ends = np.r_[starts[1:], len(sorted_keys)]
            for s, e in zip(starts, ends):
                if e - s < 2:
                    continue
                members = order[s:e]
                mh = uniq[members]
                for b0 in range(0, len(members), block):
                    ma = members[b0 : b0 + block]
                    d = hamming_matrix(uniq[ma], mh)
                    ii, jj = np.nonzero(d <= threshold)
                    gi, gj = ma[ii], members[jj]
                    keep = gi < gj  # each unordered pair once per chunk; duplicates
                    if keep.any():  # across chunks are harmless for connected_components
                        rows.append(gi[keep])
                        cols.append(gj[keep])
        # (edges between *unique* hashes)
    uniq_rows = np.concatenate(rows).astype(np.int64) if rows else np.zeros(0, dtype=np.int64)
    uniq_cols = np.concatenate(cols).astype(np.int64) if cols else np.zeros(0, dtype=np.int64)

    # expand unique-hash edges to image edges: every image maps to its unique id,
    # images with identical hashes are linked via a chain to their first occurrence
    first_occurrence = np.full(len(uniq), -1, dtype=np.int64)
    for idx in range(n - 1, -1, -1):
        first_occurrence[inverse[idx]] = idx
    dup_mask = first_occurrence[inverse] != np.arange(n)
    dup_rows = first_occurrence[inverse][dup_mask]
    dup_cols = np.arange(n)[dup_mask]

    all_rows = np.concatenate([first_occurrence[uniq_rows], dup_rows])
    all_cols = np.concatenate([first_occurrence[uniq_cols], dup_cols])
    return all_rows, all_cols


def cluster_hashes(hashes: np.ndarray, threshold: int) -> np.ndarray:
    """Connected components over near-duplicate pairs → group id per image.

    Group ids are renumbered so that they increase with the first member index.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(hashes)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    rows, cols = near_duplicate_pairs(hashes, threshold)
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    # stable renumbering by first appearance
    remap: Dict[int, int] = {}
    out = np.empty(n, dtype=np.int64)
    for i, lab in enumerate(labels.tolist()):
        out[i] = remap.setdefault(lab, len(remap))
    return out


# --------------------------------------------------------------------------- #
# group file I/O
# --------------------------------------------------------------------------- #
def build_group_table(images: Sequence[Tuple[str, str]], group_ids: np.ndarray) -> Dict[int, List[int]]:
    table: Dict[int, List[int]] = defaultdict(list)
    for idx, gid in enumerate(group_ids.tolist()):
        table[gid].append(idx)
    return dict(table)


def write_group_files(output_dir: str, images: Sequence[Tuple[str, str]], group_ids: np.ndarray,
                      hashes: np.ndarray, meta: dict, include_singletons: bool = False) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    table = build_group_table(images, group_ids)
    groups_json = {
        **meta,
        "n_images": len(images),
        "n_groups": len(table),
        "groups": {
            str(gid): [images[i][0] for i in members]
            for gid, members in sorted(table.items())
            if include_singletons or len(members) > 1
        },
    }
    json_path = os.path.join(output_dir, "groups.json")
    with open(json_path, "w") as f:
        json.dump(groups_json, f, indent=1)

    csv_path = os.path.join(output_dir, "groups.csv")
    sizes = {gid: len(m) for gid, m in table.items()}
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "group_id", "group_size", "hash_hex"])
        for idx, (rel, cls) in enumerate(images):
            gid = int(group_ids[idx])
            w.writerow([rel, cls, gid, sizes[gid], f"{int(hashes[idx]):016x}"])
    return json_path, csv_path


def load_group_file(path: str) -> Dict[str, int]:
    """Read ``groups.json`` or ``groups.csv`` → ``{relative_path: group_id}``.

    Paths use forward slashes.  Images absent from the file are singletons.
    """
    mapping: Dict[str, int] = {}
    if path.lower().endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        groups = data["groups"] if "groups" in data else data
        for gid, members in groups.items():
            for rel in members:
                mapping[rel.replace("\\", "/")] = int(gid)
    else:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                mapping[row["path"].replace("\\", "/")] = int(row["group_id"])
    return mapping


# --------------------------------------------------------------------------- #
# dataset-level statistics
# --------------------------------------------------------------------------- #
def dataset_statistics(images: Sequence[Tuple[str, str]], group_ids: np.ndarray, hashes: np.ndarray) -> dict:
    table = build_group_table(images, group_ids)
    sizes = np.array([len(m) for m in table.values()])
    nontrivial = sizes[sizes > 1]
    exact_dups = len(images) - len(np.unique(hashes))
    cross_label = 0
    for members in table.values():
        if len({images[i][1] for i in members}) > 1:
            cross_label += 1
    hist = Counter(sizes.tolist())
    per_class = {}
    for cls in sorted({c for _, c in images}):
        idx = [i for i, (_, c) in enumerate(images) if c == cls]
        gids = group_ids[idx]
        _, counts = np.unique(gids, return_counts=True)
        per_class[cls] = {
            "n_images": len(idx),
            "n_groups": int(len(counts)),
            "images_in_nontrivial_groups": int(counts[counts > 1].sum()),
        }
    return {
        "n_images": len(images),
        "n_groups": int(len(sizes)),
        "n_nontrivial_groups": int(len(nontrivial)),
        "images_in_nontrivial_groups": int(nontrivial.sum()) if len(nontrivial) else 0,
        "fraction_images_with_near_duplicate": round(float(nontrivial.sum()) / max(len(images), 1), 4) if len(nontrivial) else 0.0,
        "largest_group": int(sizes.max()) if len(sizes) else 0,
        "mean_group_size": round(float(sizes.mean()), 3) if len(sizes) else 0.0,
        "exact_duplicate_images": int(exact_dups),
        "cross_label_groups": int(cross_label),
        "group_size_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "per_class": per_class,
    }


# --------------------------------------------------------------------------- #
# audit of an existing processed split tree
# --------------------------------------------------------------------------- #
def _resolve_processed_image(link_path: str, data_dir: str, by_key: Dict[Tuple[str, str], int]) -> Optional[int]:
    """Map a processed (sym)link/copy back to the raw image index."""
    try:
        real = os.path.realpath(link_path)
        # realpath (not abspath) on both sides: --data_dir itself is often reached
        # through a symlink (data/raw -> /mnt/ssd/flame), and a mismatch here would
        # silently fall back to basename matching, which collapses same-named files
        # living in different sub-trees.
        rel = os.path.relpath(real, os.path.realpath(data_dir)).replace(os.sep, "/")
        key = ("rel", rel)
        if key in by_key:
            return by_key[key]
    except (OSError, ValueError):
        pass
    cls = CLASS_DIRS.get(os.path.basename(os.path.dirname(link_path)).lower(), "")
    return by_key.get(("name", f"{cls}/{os.path.basename(link_path)}"))


def audit_processed(processed_dir: str, data_dir: str, images: Sequence[Tuple[str, str]],
                    group_ids: np.ndarray, splits: Iterable[str] = ("train", "val", "test")) -> dict:
    """For every ``<split_name>/<node>`` measure val/test → train near-duplicate leakage."""
    by_key: Dict[Tuple[str, str], int] = {}
    for idx, (rel, cls) in enumerate(images):
        by_key[("rel", rel)] = idx
        by_key.setdefault(("name", f"{cls}/{os.path.basename(rel)}"), idx)

    report = {}
    for split_name in sorted(os.listdir(processed_dir)):
        split_root = os.path.join(processed_dir, split_name)
        if not os.path.isdir(split_root):
            continue
        nodes = sorted(d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d)))
        if not nodes:
            continue
        # collect group ids per node/split
        node_groups: Dict[str, Dict[str, List[int]]] = {}
        unresolved = 0
        for node in nodes:
            node_groups[node] = {}
            for sp in splits:
                gids = []
                sp_dir = os.path.join(split_root, node, sp)
                if not os.path.isdir(sp_dir):
                    continue
                for cls_dir in os.listdir(sp_dir):
                    full_cls = os.path.join(sp_dir, cls_dir)
                    if not os.path.isdir(full_cls):
                        continue
                    for fname in os.listdir(full_cls):
                        if not fname.lower().endswith(IMAGE_EXTS):
                            continue
                        idx = _resolve_processed_image(os.path.join(full_cls, fname), data_dir, by_key)
                        if idx is None:
                            unresolved += 1
                        else:
                            gids.append(int(group_ids[idx]))
                node_groups[node][sp] = gids
        if not any(node_groups[n] for n in nodes):
            continue

        train_any = set()
        for node in nodes:
            train_any |= set(node_groups[node].get("train", []))

        split_report = {"nodes": {}, "unresolved_files": unresolved}
        group_to_nodes: Dict[int, set] = defaultdict(set)
        for node in nodes:
            train_same = set(node_groups[node].get("train", []))
            node_rep = {}
            for sp in splits:
                gids = node_groups[node].get(sp, [])
                for g in gids:
                    group_to_nodes[g].add(node)
                if sp == "train":
                    node_rep[sp] = {"n": len(gids)}
                    continue
                n = len(gids)
                same = sum(1 for g in gids if g in train_same)
                anyn = sum(1 for g in gids if g in train_any)
                node_rep[sp] = {
                    "n": n,
                    "with_train_duplicate_same_node": same,
                    "with_train_duplicate_any_node": anyn,
                    "leak_rate_same_node": round(same / n, 4) if n else 0.0,
                    "leak_rate_any_node": round(anyn / n, 4) if n else 0.0,
                }
            split_report["nodes"][node] = node_rep
        spanning = sum(1 for s in group_to_nodes.values() if len(s) > 1)
        split_report["groups_spanning_multiple_nodes"] = spanning
        split_report["n_groups_in_split"] = len(group_to_nodes)
        # global numbers
        for sp in splits:
            if sp == "train":
                continue
            tot = sum(split_report["nodes"][nd][sp]["n"] for nd in nodes if sp in split_report["nodes"][nd])
            leak = sum(split_report["nodes"][nd][sp]["with_train_duplicate_any_node"] for nd in nodes if sp in split_report["nodes"][nd])
            split_report[f"global_{sp}_leak_rate_any_node"] = round(leak / tot, 4) if tot else 0.0
        report[split_name] = split_report
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run(data_dir: str, output_dir: str, method: str = "dhash", hash_size: int = 8, threshold: int = 8,
        processed_dir: Optional[str] = None, workers: int = 1, limit: Optional[int] = None,
        include_singletons: bool = False, use_cache: bool = True, verbose: bool = True) -> dict:
    images = find_images(data_dir)
    if limit:
        images = images[:limit]
    if not images:
        raise SystemExit(f"No images found under {data_dir}")
    if verbose:
        print(f"Found {len(images)} images in {data_dir}")

    os.makedirs(output_dir, exist_ok=True)
    cache = os.path.join(output_dir, f"hashes_{method}{hash_size}.npz")
    hashes = None
    if use_cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if len(z["hashes"]) == len(images) and list(z["paths"]) == [p for p, _ in images]:
            hashes = z["hashes"]
            if verbose:
                print(f"Loaded cached hashes from {cache}")
    if hashes is None:
        if verbose:
            print(f"Hashing with {method} (size={hash_size}, workers={workers})...")
        hashes = compute_hashes([os.path.join(data_dir, p) for p, _ in images], method, hash_size, workers, verbose)
        np.savez(cache, hashes=hashes, paths=np.array([p for p, _ in images]))

    if verbose:
        print(f"Clustering near-duplicates (Hamming <= {threshold})...")
    t0 = time.perf_counter()
    group_ids = cluster_hashes(hashes, threshold)
    stats = dataset_statistics(images, group_ids, hashes)
    stats["clustering_time_s"] = round(time.perf_counter() - t0, 2)

    meta = {"method": method, "hash_size": hash_size, "threshold": threshold, "data_dir": os.path.abspath(data_dir)}
    json_path, csv_path = write_group_files(output_dir, images, group_ids, hashes, meta, include_singletons)

    report = {"settings": meta, "dataset": stats, "group_file": json_path, "group_csv": csv_path}
    if processed_dir and os.path.isdir(processed_dir):
        if verbose:
            print(f"Auditing existing splits in {processed_dir}...")
        report["processed_splits"] = audit_processed(processed_dir, data_dir, images, group_ids)

    report_path = os.path.join(output_dir, "leakage_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print("\n=== Near-duplicate summary ===")
        for k in ("n_images", "n_groups", "n_nontrivial_groups", "images_in_nontrivial_groups",
                  "fraction_images_with_near_duplicate", "largest_group", "exact_duplicate_images", "cross_label_groups"):
            print(f"  {k:40s} {stats[k]}")
        for split_name, rep in report.get("processed_splits", {}).items():
            print(f"\n=== Split '{split_name}' ===  groups spanning >1 node: {rep['groups_spanning_multiple_nodes']}")
            for node, nrep in rep["nodes"].items():
                for sp in ("val", "test"):
                    if sp in nrep:
                        r = nrep[sp]
                        print(f"  {node:8s} {sp:5s} n={r['n']:6d}  leak(same node)={r['leak_rate_same_node']:.3f}  "
                              f"leak(any node)={r['leak_rate_any_node']:.3f}")
        print(f"\nGroup file: {json_path}\nReport:     {report_path}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="data/raw/flame_dataset")
    p.add_argument("--processed_dir", default=None, help="Existing data/processed tree to audit (optional)")
    p.add_argument("--output_dir", default="analysis/leakage")
    p.add_argument("--hash", dest="method", default="dhash", choices=["dhash", "phash", "ahash"])
    p.add_argument("--hash_size", type=int, default=8, help="8 → 64-bit hash (only 8 is supported)")
    p.add_argument("--threshold", type=int, default=8, help="max Hamming distance (of 64 bits) for near-duplicates")
    p.add_argument("--workers", type=int, default=1, help="processes for hashing")
    p.add_argument("--limit", type=int, default=None, help="only hash the first N images (debug)")
    p.add_argument("--include_singletons", action="store_true", help="also list size-1 groups in groups.json")
    p.add_argument("--no_cache", action="store_true", help="ignore cached hashes")
    args = p.parse_args(argv)
    if args.hash_size != 8:
        raise SystemExit("--hash_size must be 8 (64-bit hashes)")
    run(args.data_dir, args.output_dir, args.method, args.hash_size, args.threshold, args.processed_dir,
        args.workers, args.limit, args.include_singletons, not args.no_cache)


if __name__ == "__main__":
    main()
