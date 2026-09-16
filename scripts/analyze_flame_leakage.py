"""FedRGBD — Near-duplicate / leakage analysis for the FLAME dataset.

FLAME images are frames extracted from aerial videos, so consecutive frames
are near-identical.  A random image-level split therefore places near-copies
of test images in the training set, which inflates accuracy.  This script:

1. Computes a perceptual hash (dHash / pHash / aHash) of every image -- or,
   with ``--hash both``, dHash *and* pHash in the same pass.
2. Clusters images whose hashes are within a Hamming distance ``--threshold``
   into *near-duplicate groups* (exact multi-index hashing + connected
   components; no pairwise O(N²) scan).  ``--hash both`` clusters the **union**
   of the dHash edges (``--threshold``) and the pHash edges
   (``--phash_threshold``, default 10), so its groups always contain the
   single-hash groups.
3. Writes ``groups.json`` / ``groups.csv`` which ``src/data/data_splitter.py
   --group_file`` consumes to keep whole groups on one node and in one split,
   plus ``example_groups.txt`` with the ``--examples`` largest groups.
4. Optionally audits an existing ``data/processed`` tree and reports how many
   val/test images have a near-duplicate in *any* training split
   (``leakage_report.json``).
5. Optional diagnostics, all written into ``leakage_report.json``:
   ``--sweep 4 6 8 10 12`` (threshold sensitivity, does not change
   ``groups.json``), MD5 byte-identical duplicates (on by default, computed in
   the hashing pass; ``--no_md5`` to skip) and ``--sequence_heuristic`` (how
   often consecutive frame numbers in the file names end up in one group).

Usage
-----
    python3 scripts/analyze_flame_leakage.py \
        --data_dir data/raw/flame_dataset \
        --processed_dir data/processed \
        --output_dir analysis/leakage --threshold 8 --workers 4

    python3 scripts/analyze_flame_leakage.py --data_dir data/raw/flame_dataset \
        --hash both --threshold 8 --phash_threshold 10 \
        --sweep 4 6 8 10 12 --sequence_heuristic --examples 20

    python3 src/data/data_splitter.py --group_file analysis/leakage/groups.json ...

Only NumPy, Pillow and SciPy are required.  Runs on CPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
CLASS_DIRS = {"fire": "Fire", "no_fire": "No_Fire", "nofire": "No_Fire"}

#: ``--hash both`` clusters the union of the dHash and the pHash edges.
DUAL_METHODS = ("dhash", "phash")

#: metadata keys that may sit next to the path->gid entries of a flat group file
_GROUP_META_KEYS = frozenset({
    "n_images", "n_groups", "threshold", "phash_threshold", "thresholds",
    "hash_size", "method", "data_dir", "settings", "timestamp", "dataset",
})

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


def file_md5(path: str, chunk: int = 1 << 20) -> str:
    """MD5 of the raw file bytes (byte-identical duplicate detection)."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_worker(args):
    """``(idx, path, methods, hash_size, want_md5)`` -> ``(idx, {method: int}, md5, error)``.

    Every requested hash *and* the MD5 digest are produced in one pass over the
    file, so ``--hash both --md5`` still reads each of the ~48 k images once.
    """
    idx, path, methods, hash_size, want_md5 = args
    hashes = {m: -1 for m in methods}
    error = None
    try:
        with Image.open(path) as im:
            gray = im.convert("L")
            for method in methods:
                small = np.asarray(gray.resize(hash_input_size(method, hash_size), Image.LANCZOS),
                                   dtype=np.float64)
                hashes[method] = int(hash_array(small, method, hash_size))
    except Exception as exc:  # corrupt file -> sentinel, reported later
        error = repr(exc)
    digest = ""
    if want_md5:
        try:
            digest = file_md5(path)
        except OSError as exc:  # pragma: no cover - unreadable file
            error = error or repr(exc)
    return idx, hashes, digest, error


def compute_hash_columns(paths: Sequence[str], methods: Sequence[str], hash_size: int = 8,
                         workers: int = 1, progress: bool = True, with_md5: bool = False):
    """Hash every path with every method in ``methods`` in a single pass.

    Returns ``({method: uint64 array}, [md5, ...] or None, [(path, error), ...])``.
    Unreadable files get hash 0 (and are listed in the third return value).
    """
    methods = list(methods)
    n = len(paths)
    out = {m: np.zeros(n, dtype=np.uint64) for m in methods}
    md5s: Optional[List[str]] = [""] * n if with_md5 else None
    failures: List[Tuple[str, str]] = []
    jobs = [(i, p, methods, hash_size, with_md5) for i, p in enumerate(paths)]
    t0 = time.perf_counter()

    def consume(k, result):
        i, hashes, digest, error = result
        for m in methods:
            out[m][i] = np.uint64(max(hashes[m], 0))
        if md5s is not None:
            md5s[i] = digest
        if error is not None:
            failures.append((paths[i], error))
            print(f"  WARNING: could not hash {paths[i]}: {error}", file=sys.stderr)
        if progress and k % 5000 == 0:
            print(f"  hashed {k}/{n} ({time.perf_counter() - t0:.0f}s)")

    if workers > 1:
        with Pool(workers) as pool:
            for k, result in enumerate(pool.imap_unordered(_hash_worker, jobs, chunksize=64), 1):
                consume(k, result)
    else:
        for k, job in enumerate(jobs, 1):
            consume(k, _hash_worker(job))
    return out, md5s, failures


def compute_hashes(paths: Sequence[str], method: str, hash_size: int, workers: int = 1,
                   progress: bool = True) -> np.ndarray:
    """Hash every path -> ``uint64`` array; unreadable files get hash 0 and are flagged."""
    columns, _md5, _bad = compute_hash_columns(paths, [method], hash_size, workers, progress)
    return columns[method]


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


def cluster_from_pairs(n: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Connected components over an edge list -> group id per image.

    Group ids are renumbered so that they increase with the first member index.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if n == 0:
        return np.zeros(0, dtype=np.int64)
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    # stable renumbering by first appearance
    remap: Dict[int, int] = {}
    out = np.empty(n, dtype=np.int64)
    for i, lab in enumerate(labels.tolist()):
        out[i] = remap.setdefault(lab, len(remap))
    return out


def cluster_hashes(hashes: np.ndarray, threshold: int) -> np.ndarray:
    """Connected components over the near-duplicate pairs of one hash array."""
    n = len(hashes)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    rows, cols = near_duplicate_pairs(hashes, threshold)
    return cluster_from_pairs(n, rows, cols)


def cluster_hash_columns(columns: Dict[str, np.ndarray], thresholds: Dict[str, int]) -> np.ndarray:
    """Cluster the **union** of the near-duplicate edges of several hash methods.

    ``--hash both`` passes ``{"dhash": h_d, "phash": h_p}`` with
    ``{"dhash": --threshold, "phash": --phash_threshold}``.  Because the edge
    set is a superset of each single-method edge set, every single-method group
    is contained in one union group (groups only ever merge, never split).
    """
    if not columns:
        raise ValueError("no hash columns given")
    n = len(next(iter(columns.values())))
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    for method in sorted(columns):
        r, c = near_duplicate_pairs(columns[method], int(thresholds[method]))
        rows.append(np.asarray(r, dtype=np.int64))
        cols.append(np.asarray(c, dtype=np.int64))
    return cluster_from_pairs(n, np.concatenate(rows), np.concatenate(cols))


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


def normalise_group_path(raw: str, data_dir: Optional[str] = None) -> str:
    """Turn a group-file key into a ``--data_dir``-relative forward-slash path.

    Relative keys are returned unchanged (minus a leading ``./``).  Absolute
    keys -- which is what a group file written by the authors' own script
    contains -- are made relative to ``data_dir`` when possible and otherwise
    reduced to ``<ClassDir>/<basename>``, the fallback key both the splitter
    (``_group_lookup``) and ``verify_splits`` match on.
    """
    path = str(raw).replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if not os.path.isabs(path):
        return path
    if data_dir:
        for root in {os.path.abspath(data_dir), os.path.realpath(data_dir)}:
            try:
                rel = os.path.relpath(os.path.realpath(path), root).replace(os.sep, "/")
            except (OSError, ValueError):
                continue
            if not rel.startswith(".."):
                return rel
    parts = [p for p in path.split("/") if p]
    return "/".join(parts[-2:]) if len(parts) >= 2 else path


def _groups_block(data: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """Locate the ``{key: value}`` group block and any embedded ``data_dir``."""
    if not isinstance(data, dict):
        raise ValueError("group file must contain a JSON object")
    embedded = data.get("data_dir") if isinstance(data.get("data_dir"), str) else None
    if isinstance(data.get("groups"), dict):
        return data["groups"], embedded
    block = {k: v for k, v in data.items() if k not in _GROUP_META_KEYS}
    return block, embedded


def load_group_file(path: str, data_dir: Optional[str] = None) -> Dict[str, int]:
    """Read ``groups.json`` / ``groups.csv`` -> ``{relative_path: group_id}``.

    Three JSON layouts are accepted, detected by the *value* type of the group
    block (paths always use forward slashes, images absent from the file are
    singletons):

    * ``{"groups": {"<gid>": ["Fire/a.jpg", ...]}}`` -- written by this script;
    * ``{"groups": {"<path>": <gid>}}`` -- the inverted layout written by the
      authors' own ``analyze_flame_leakage.py`` (absolute paths);
    * a bare top-level ``{"<path>": <gid>}`` mapping (metadata keys ignored).

    ``data_dir`` is used to relativise absolute paths; when it is omitted the
    ``data_dir`` recorded inside the file is used.
    """
    mapping: Dict[str, int] = {}
    if path.lower().endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        groups, embedded = _groups_block(data)
        root = data_dir or embedded
        values = [v for v in groups.values()]
        inverted = bool(values) and not isinstance(values[0], (list, tuple))
        if inverted:  # {path: gid}
            for raw, gid in groups.items():
                mapping[normalise_group_path(raw, root)] = int(gid)
        else:  # {gid: [path, ...]}
            for gid, members in groups.items():
                for rel in members:
                    mapping[normalise_group_path(rel, root)] = int(gid)
    else:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                mapping[normalise_group_path(row["path"], data_dir)] = int(row["group_id"])
    return mapping


# --------------------------------------------------------------------------- #
# dataset-level statistics
# --------------------------------------------------------------------------- #
def dataset_statistics(images: Sequence[Tuple[str, str]], group_ids: np.ndarray, hashes: np.ndarray,
                       md5s: Optional[Sequence[str]] = None) -> dict:
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
    out = {
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
    if md5s is not None:
        # byte-identical files, independent of the perceptual hash: how many
        # images are an exact copy of an earlier one (0 = no literal duplicates)
        counts = Counter(d for d in md5s if d)
        out["exact_duplicate_files_md5"] = int(sum(c - 1 for c in counts.values() if c > 1))
        out["n_unique_md5"] = int(len(counts))
    return out


# --------------------------------------------------------------------------- #
# threshold-sensitivity sweep
# --------------------------------------------------------------------------- #
def sweep_row(images: Sequence[Tuple[str, str]], group_ids: np.ndarray, threshold: int) -> dict:
    """The five sensitivity numbers reported for one threshold."""
    sizes = np.array([len(m) for m in build_group_table(images, group_ids).values()])
    nontrivial = sizes[sizes > 1]
    in_nontrivial = int(nontrivial.sum()) if len(nontrivial) else 0
    return {
        "threshold": int(threshold),
        "n_groups": int(len(sizes)),
        "n_nontrivial_groups": int(len(nontrivial)),
        "images_in_nontrivial_groups": in_nontrivial,
        "largest_group": int(sizes.max()) if len(sizes) else 0,
        "fraction_with_near_duplicate": round(in_nontrivial / max(len(images), 1), 4),
    }


def threshold_sweep(images: Sequence[Tuple[str, str]], columns: Dict[str, np.ndarray],
                    thresholds: Sequence[int]) -> List[dict]:
    """Re-cluster at every threshold in ``thresholds`` (does NOT touch groups.json).

    With ``--hash both`` every method is clustered at the same sweep value, so
    the table shows the sensitivity of the union grouping itself.
    """
    rows = []
    for thr in sorted({int(t) for t in thresholds}):
        gids = cluster_hash_columns(columns, {m: thr for m in columns})
        rows.append(sweep_row(images, gids, thr))
    return rows


def format_sweep_table(rows: Sequence[dict]) -> str:
    """Pretty table of :func:`threshold_sweep` rows."""
    head = "  thr   n_groups  nontrivial  imgs_in_nontrivial  largest  frac_with_near_dup"
    lines = [head, "  " + "-" * (len(head) - 2)]
    for r in rows:
        lines.append("  {:>3}   {:>8}  {:>10}  {:>18}  {:>7}  {:>18.4f}".format(
            r["threshold"], r["n_groups"], r["n_nontrivial_groups"],
            r["images_in_nontrivial_groups"], r["largest_group"],
            r["fraction_with_near_duplicate"]))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# filename frame-number (sequence) heuristic -- optional diagnostic
# --------------------------------------------------------------------------- #
#: last integer in a file name, e.g. ``image_00123.jpg`` -> ``00123``
_FRAME_NUMBER_RE = re.compile(r"(\d+)(?!.*\d)")


def _frame_key(rel: str) -> Optional[Tuple[str, int]]:
    """``('Fire/image_#.jpg', 123)`` for ``Fire/image_123.jpg``; None without a number."""
    directory, base = os.path.split(rel)
    match = _FRAME_NUMBER_RE.search(base)
    if match is None:
        return None
    template = base[:match.start(1)] + "#" + base[match.end(1):]
    return ("{}/{}".format(directory, template) if directory else template, int(match.group(1)))


def sequence_heuristic(images: Sequence[Tuple[str, str]], group_ids: np.ndarray,
                       gap: int = 1) -> dict:
    """How often do consecutive *frame numbers* land in the same group?

    Ported from the authors' ``filename_sequence_stats``: the last integer of
    the file name is read as a frame number.  Here the numbers are additionally
    keyed by the enclosing directory and the name template, and every pair of
    numbers at most ``gap`` apart is checked against the near-duplicate
    clustering -- which turns the heuristic into the statement the paper needs
    ("X % of consecutive frames are near-duplicates of each other").
    """
    by_key: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    per_class_numbers: Dict[str, List[int]] = defaultdict(list)
    n_numbered = 0
    for idx, (rel, cls) in enumerate(images):
        key = _frame_key(rel)
        if key is None:
            continue
        n_numbered += 1
        by_key[key[0]].append((key[1], idx))
        per_class_numbers[cls].append(key[1])

    per_class_pairs: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    n_pairs = 0
    n_same = 0
    for entries in by_key.values():
        entries.sort()
        for (num_a, i), (num_b, j) in zip(entries, entries[1:]):
            if num_b - num_a > gap:
                continue
            n_pairs += 1
            same = int(group_ids[i]) == int(group_ids[j])
            n_same += int(same)
            cls = images[i][1]
            per_class_pairs[cls][0] += 1
            per_class_pairs[cls][1] += int(same)

    per_class = {}
    for cls in sorted(per_class_numbers):
        nums = per_class_numbers[cls]
        pairs, same = per_class_pairs.get(cls, [0, 0])
        per_class[cls] = {
            "n_with_number": len(nums),
            "min": int(min(nums)) if nums else None,
            "max": int(max(nums)) if nums else None,
            "consecutive_pairs": int(pairs),
            "consecutive_pairs_same_group": int(same),
            "fraction_consecutive_pairs_same_group": round(same / pairs, 4) if pairs else 0.0,
        }
    return {
        "gap": int(gap),
        "n_images": len(images),
        "n_images_with_frame_number": int(n_numbered),
        "n_sequences": int(len(by_key)),
        "n_consecutive_pairs": int(n_pairs),
        "n_consecutive_pairs_same_group": int(n_same),
        "fraction_consecutive_pairs_same_group": round(n_same / n_pairs, 4) if n_pairs else 0.0,
        "per_class": per_class,
    }


# --------------------------------------------------------------------------- #
# human-readable example groups
# --------------------------------------------------------------------------- #
def write_example_groups(output_dir: str, images: Sequence[Tuple[str, str]],
                         group_ids: np.ndarray, n_examples: int = 20,
                         max_members: int = 50) -> str:
    """``example_groups.txt``: the ``n_examples`` largest groups and their members."""
    table = build_group_table(images, group_ids)
    largest = sorted((kv for kv in table.items() if len(kv[1]) > 1),
                     key=lambda kv: (-len(kv[1]), kv[0]))[:max(int(n_examples), 0)]
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "example_groups.txt")
    with open(path, "w") as f:
        f.write("# {} largest near-duplicate groups (of {} groups, {} nontrivial)\n\n".format(
            len(largest), len(table), sum(1 for m in table.values() if len(m) > 1)))
        for rank, (gid, members) in enumerate(largest):
            f.write("# rank {}  group {}  size={}\n".format(rank, gid, len(members)))
            for i in members[:max_members]:
                f.write("{}\n".format(images[i][0]))
            if len(members) > max_members:
                f.write("# ... and {} more\n".format(len(members) - max_members))
            f.write("\n")
    return path


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
def methods_for(method: str) -> List[str]:
    """``"both"`` -> ``["dhash", "phash"]``; every other method -> just itself."""
    return list(DUAL_METHODS) if method == "both" else [method]


def thresholds_for(method: str, threshold: int, phash_threshold: int) -> Dict[str, int]:
    """Per-method Hamming thresholds (``--threshold`` / ``--phash_threshold``)."""
    if method == "both":
        return {"dhash": int(threshold), "phash": int(phash_threshold)}
    return {method: int(threshold)}


def load_or_compute_hashes(data_dir: str, output_dir: str, images: Sequence[Tuple[str, str]],
                           methods: Sequence[str], hash_size: int = 8, workers: int = 1,
                           use_cache: bool = True, verbose: bool = True, with_md5: bool = False):
    """Hash cache per method (``hashes_<method><size>.npz``) + optional MD5 cache.

    Every method still missing from the cache is computed in a *single* pass
    over the images, together with the MD5 digests, so the ~48 k files are read
    once even for ``--hash both --md5``.
    """
    rel_paths = [p for p, _ in images]
    key = np.array(rel_paths)
    columns: Dict[str, np.ndarray] = {}
    for method in methods:
        cache = os.path.join(output_dir, "hashes_{}{}.npz".format(method, hash_size))
        if use_cache and os.path.exists(cache):
            z = np.load(cache, allow_pickle=False)
            if len(z["hashes"]) == len(images) and list(z["paths"]) == rel_paths:
                columns[method] = z["hashes"]
                if verbose:
                    print("Loaded cached hashes from {}".format(cache))

    md5s: Optional[List[str]] = None
    md5_cache = os.path.join(output_dir, "md5s.npz")
    if with_md5 and use_cache and os.path.exists(md5_cache):
        z = np.load(md5_cache, allow_pickle=False)
        if len(z["md5s"]) == len(images) and list(z["paths"]) == rel_paths:
            md5s = [str(x) for x in z["md5s"]]
            if verbose:
                print("Loaded cached MD5 digests from {}".format(md5_cache))

    todo = [m for m in methods if m not in columns]
    need_md5 = with_md5 and md5s is None
    failures: List[Tuple[str, str]] = []
    if todo or need_md5:
        if verbose:
            print("Hashing with {} (size={}, workers={}{})...".format(
                "+".join(todo) or "md5", hash_size, workers, ", md5" if need_md5 else ""))
        fresh, digests, failures = compute_hash_columns(
            [os.path.join(data_dir, p) for p, _ in images], todo, hash_size, workers,
            verbose, need_md5)
        for method, values in fresh.items():
            columns[method] = values
            np.savez(os.path.join(output_dir, "hashes_{}{}.npz".format(method, hash_size)),
                     hashes=values, paths=key)
        if need_md5:
            md5s = digests
            np.savez(md5_cache, md5s=np.array(digests), paths=key)
    return {m: columns[m] for m in methods}, md5s, failures


def run(data_dir: str, output_dir: str, method: str = "dhash", hash_size: int = 8, threshold: int = 8,
        processed_dir: Optional[str] = None, workers: int = 1, limit: Optional[int] = None,
        include_singletons: bool = False, use_cache: bool = True, verbose: bool = True,
        phash_threshold: int = 10, sweep: Optional[Sequence[int]] = None, md5: bool = True,
        sequence_heuristic_on: bool = False, sequence_gap: int = 1, examples: int = 20) -> dict:
    images = find_images(data_dir)
    if limit:
        images = images[:limit]
    if not images:
        raise SystemExit(f"No images found under {data_dir}")
    if verbose:
        print(f"Found {len(images)} images in {data_dir}")

    os.makedirs(output_dir, exist_ok=True)
    methods = methods_for(method)
    thresholds = thresholds_for(method, threshold, phash_threshold)
    columns, md5s, failures = load_or_compute_hashes(
        data_dir, output_dir, images, methods, hash_size, workers, use_cache, verbose, md5)
    hashes = columns[methods[0]]  # the primary hash: groups.csv column, exact-dup count

    if failures:
        bad_path = os.path.join(output_dir, "unreadable_files.txt")
        with open(bad_path, "w") as f:
            for path, err in sorted(failures):
                f.write("{}\t{}\n".format(path, err))
        if verbose:
            print("  {} unreadable file(s) listed in {}".format(len(failures), bad_path))

    if verbose:
        print("Clustering near-duplicates ({})...".format(
            ", ".join("{} <= {}".format(m, thresholds[m]) for m in methods)))
    t0 = time.perf_counter()
    group_ids = cluster_hash_columns(columns, thresholds)
    stats = dataset_statistics(images, group_ids, hashes, md5s)
    stats["clustering_time_s"] = round(time.perf_counter() - t0, 2)

    meta = {"method": method, "hash_size": hash_size, "threshold": threshold,
            "data_dir": os.path.abspath(data_dir)}
    if method == "both":
        meta["phash_threshold"] = int(phash_threshold)
        meta["union_of"] = list(methods)
    json_path, csv_path = write_group_files(output_dir, images, group_ids, hashes, meta, include_singletons)
    example_path = write_example_groups(output_dir, images, group_ids, examples)

    report = {"settings": meta, "dataset": stats, "group_file": json_path, "group_csv": csv_path,
              "example_groups": example_path}
    if failures:
        report["unreadable_files"] = [{"path": p, "error": e} for p, e in sorted(failures)]

    # --- threshold-sensitivity sweep (never changes groups.json) ----------- #
    if sweep:
        if verbose:
            print("Threshold sensitivity sweep ({})...".format(
                ", ".join(str(int(t)) for t in sorted({int(t) for t in sweep}))))
        report["threshold_sweep"] = threshold_sweep(images, columns, sweep)

    # --- filename frame-number heuristic (optional diagnostic) ------------- #
    if sequence_heuristic_on:
        report["sequence_heuristic"] = sequence_heuristic(images, group_ids, sequence_gap)

    if processed_dir and os.path.isdir(processed_dir):
        if verbose:
            print(f"Auditing existing splits in {processed_dir}...")
        report["processed_splits"] = audit_processed(processed_dir, data_dir, images, group_ids)

    report_path = os.path.join(output_dir, "leakage_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print("\n=== Near-duplicate summary ===")
        keys = ["n_images", "n_groups", "n_nontrivial_groups", "images_in_nontrivial_groups",
                "fraction_images_with_near_duplicate", "largest_group", "exact_duplicate_images",
                "cross_label_groups"]
        if "exact_duplicate_files_md5" in stats:
            keys.append("exact_duplicate_files_md5")
        for k in keys:
            print(f"  {k:40s} {stats[k]}")
        if "threshold_sweep" in report:
            print("\n=== Threshold sensitivity (groups.json uses "
                  "{}) ===".format(", ".join("{}<={}".format(m, thresholds[m]) for m in methods)))
            print(format_sweep_table(report["threshold_sweep"]))
        if "sequence_heuristic" in report:
            seq = report["sequence_heuristic"]
            print("\n=== Filename frame-number heuristic (gap <= {}) ===".format(seq["gap"]))
            print("  {:40s} {}".format("n_images_with_frame_number", seq["n_images_with_frame_number"]))
            print("  {:40s} {}".format("n_consecutive_pairs", seq["n_consecutive_pairs"]))
            print("  {:40s} {} ({:.1%})".format(
                "consecutive pairs in the same group", seq["n_consecutive_pairs_same_group"],
                seq["fraction_consecutive_pairs_same_group"]))
        for split_name, rep in report.get("processed_splits", {}).items():
            print(f"\n=== Split '{split_name}' ===  groups spanning >1 node: {rep['groups_spanning_multiple_nodes']}")
            for node, nrep in rep["nodes"].items():
                for sp in ("val", "test"):
                    if sp in nrep:
                        r = nrep[sp]
                        print(f"  {node:8s} {sp:5s} n={r['n']:6d}  leak(same node)={r['leak_rate_same_node']:.3f}  "
                              f"leak(any node)={r['leak_rate_any_node']:.3f}")
        print(f"\nGroup file: {json_path}\nExamples:   {example_path}\nReport:     {report_path}")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="data/raw/flame_dataset")
    p.add_argument("--processed_dir", default=None, help="Existing data/processed tree to audit (optional)")
    p.add_argument("--output_dir", default="analysis/leakage")
    p.add_argument("--hash", dest="method", default="dhash",
                   choices=["dhash", "phash", "ahash", "both"],
                   help="'both' clusters the UNION of the dHash edges (--threshold) and the "
                        "pHash edges (--phash_threshold)")
    p.add_argument("--hash_size", type=int, default=8, help="8 -> 64-bit hash (only 8 is supported)")
    p.add_argument("--threshold", type=int, default=8, help="max Hamming distance (of 64 bits) for near-duplicates")
    p.add_argument("--phash_threshold", type=int, default=10,
                   help="pHash Hamming threshold used by --hash both (default 10)")
    p.add_argument("--sweep", type=int, nargs="+", default=None, metavar="T",
                   help="threshold-sensitivity sweep, e.g. --sweep 4 6 8 10 12; reported in "
                        "leakage_report.json['threshold_sweep'], groups.json is unaffected")
    p.add_argument("--sequence_heuristic", action="store_true",
                   help="also report how often consecutive frame numbers in the file names "
                        "land in the same near-duplicate group")
    p.add_argument("--sequence_gap", type=int, default=1,
                   help="max frame-number difference counted as 'consecutive' (default 1)")
    p.add_argument("--examples", type=int, default=20,
                   help="how many of the largest groups example_groups.txt lists (0 = none)")
    p.add_argument("--workers", type=int, default=1, help="processes for hashing")
    p.add_argument("--limit", type=int, default=None, help="only hash the first N images (debug)")
    p.add_argument("--include_singletons", action="store_true", help="also list size-1 groups in groups.json")
    p.add_argument("--no_cache", action="store_true", help="ignore cached hashes")
    md5 = p.add_mutually_exclusive_group()
    md5.add_argument("--md5", dest="md5", action="store_true", default=True,
                     help="count byte-identical duplicates (default; computed in the hashing pass)")
    md5.add_argument("--no_md5", dest="md5", action="store_false",
                     help="skip the MD5 pass (no 'exact_duplicate_files_md5' in the report)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.hash_size != 8:
        raise SystemExit("--hash_size must be 8 (64-bit hashes)")
    run(args.data_dir, args.output_dir, args.method, args.hash_size, args.threshold, args.processed_dir,
        args.workers, args.limit, args.include_singletons, not args.no_cache, True,
        args.phash_threshold, args.sweep, args.md5, args.sequence_heuristic, args.sequence_gap,
        args.examples)


if __name__ == "__main__":
    main()
