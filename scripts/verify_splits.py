#!/usr/bin/env python3
r"""FedRGBD -- verify ``data/processed`` splits, on one node and across nodes.

``src/data/data_splitter.py`` walks the raw dataset with ``os.walk`` and seeds
``random`` with ``--seed``.  The *partition* is therefore only reproducible as
long as ``os.walk`` returns the images in the same order.  When each Jetson
runs the splitter on its own copy of the dataset, a different filesystem order
silently produces a **different** partition: node A's training image may be
node B's test image, which invalidates every federated result.

This script answers two questions:

1. *Is one ``data/processed`` tree self-consistent?*  No image is used by two
   nodes or by two of train/val/test, no near-duplicate group (unit) is split,
   the ``manifest.csv`` counts agree with ``split_stats.json`` and the files
   really are on disk.
2. *Do the three nodes hold the same split?*  ``--compare`` diffs the
   manifests / per-split stats of further ``data/processed`` copies, and
   ``--hashes_only`` prints one md5 per manifest so the nodes can be compared
   without copying anything (run it on each node, compare the printed table).

Exit status is ``0`` when every check passes and ``1`` when any check fails.

Usage
-----
    # self-check of the local tree
    python3 scripts/verify_splits.py data/processed

    # md5 table only -- run on every node and compare the output by eye
    python3 scripts/verify_splits.py data/processed --hashes_only

    # compare copies fetched from the other two nodes
    python3 scripts/verify_splits.py data/processed \
        --compare /mnt/node_b/processed /mnt/node_c/processed

    # full check incl. the group file and the subsample ablations
    python3 scripts/verify_splits.py data/processed \
        --group_file analysis/leakage/groups.json --expect_subsample \
        --json analysis/verify_splits.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

TVT = ("train", "val", "test")
LABELS = ("Fire", "No_Fire")
LABEL_KEY = {"Fire": "fire", "No_Fire": "nofire"}
MANIFEST_COLUMNS = ("node", "split", "label", "path", "group_id")
MANIFEST_NAME = "manifest.csv"
STATS_NAME = "split_stats.json"

#: how many offending items a single message lists before it says "... and N more"
MAX_REPORT = 5

_SUB_RE = re.compile(r"^(?P<base>.+)_sub(?P<frac>\d+(?:\.\d+)?)$")


# --------------------------------------------------------------------------- #
# group-file loading (shared with scripts/analyze_flame_leakage.py)
# --------------------------------------------------------------------------- #
#: metadata keys that may sit next to the path->gid entries of a flat group file
_GROUP_META_KEYS = frozenset({
    "n_images", "n_groups", "threshold", "phash_threshold", "thresholds",
    "hash_size", "method", "data_dir", "settings", "timestamp", "dataset",
})


def _normalise_group_path(raw: str, data_dir: Optional[str] = None) -> str:
    """Local copy of ``analyze_flame_leakage.normalise_group_path``."""
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
    parts = [x for x in path.split("/") if x]
    return "/".join(parts[-2:]) if len(parts) >= 2 else path


def _local_load_group_file(path: str, data_dir: Optional[str] = None) -> Dict[str, int]:
    """Standalone copy of ``analyze_flame_leakage.load_group_file``.

    Accepts both ``{"groups": {"gid": [path, ...]}}`` (this repo) and the
    inverted ``{"groups": {path: gid}}`` / bare ``{path: gid}`` layout.
    """
    mapping: Dict[str, int] = {}
    if path.lower().endswith(".json"):
        with open(path) as fh:
            data = json.load(fh)
        embedded = data.get("data_dir") if isinstance(data.get("data_dir"), str) else None
        if isinstance(data.get("groups"), dict):
            groups = data["groups"]
        else:
            groups = {k: v for k, v in data.items() if k not in _GROUP_META_KEYS}
        root = data_dir or embedded
        values = list(groups.values())
        if values and not isinstance(values[0], (list, tuple)):  # {path: gid}
            for raw, gid in groups.items():
                mapping[_normalise_group_path(raw, root)] = int(gid)
        else:  # {gid: [path, ...]}
            for gid, members in groups.items():
                for rel in members:
                    mapping[_normalise_group_path(rel, root)] = int(gid)
    else:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                mapping[_normalise_group_path(row["path"], data_dir)] = int(row["group_id"])
    return mapping


def load_group_file(path: str, data_dir: Optional[str] = None) -> Dict[str, int]:
    """Canonical parser from ``scripts/``; falls back to the local copy.

    The fallback keeps this script runnable on a Jetson where numpy/PIL (which
    ``analyze_flame_leakage`` imports at module level) may be unavailable.
    """
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from scripts.analyze_flame_leakage import load_group_file as _impl  # noqa: WPS433

        return _impl(path, data_dir)
    except Exception:  # pragma: no cover - environment dependent
        return _local_load_group_file(path, data_dir)


# --------------------------------------------------------------------------- #
# report collection
# --------------------------------------------------------------------------- #
class Report(object):
    """Collects errors / warnings so every check runs before we give up."""

    def __init__(self) -> None:
        self.issues: List[Dict[str, str]] = []

    def add(self, level: str, check: str, message: str, split: Optional[str] = None) -> None:
        self.issues.append({"level": level, "check": check, "split": split or "",
                            "message": message})

    def error(self, check: str, message: str, split: Optional[str] = None) -> None:
        self.add("error", check, message, split)

    def warn(self, check: str, message: str, split: Optional[str] = None) -> None:
        self.add("warning", check, message, split)

    @property
    def errors(self) -> List[Dict[str, str]]:
        return [i for i in self.issues if i["level"] == "error"]

    @property
    def warnings(self) -> List[Dict[str, str]]:
        return [i for i in self.issues if i["level"] == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def n_errors_for(self, split: str) -> int:
        return sum(1 for i in self.errors if i["split"] == split)


def _join(items: Sequence[Any], limit: int = MAX_REPORT) -> str:
    items = list(items)
    head = ", ".join(str(i) for i in items[:limit])
    if len(items) > limit:
        head += " ... and {} more".format(len(items) - limit)
    return head


# --------------------------------------------------------------------------- #
# small IO helpers
# --------------------------------------------------------------------------- #
def md5_file(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def manifest_path(processed_dir: str, split: str) -> str:
    return os.path.join(processed_dir, split, MANIFEST_NAME)


def read_manifest(path: str) -> List[Dict[str, str]]:
    """``manifest.csv`` -> list of row dicts (raises on a missing column)."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        missing = [c for c in MANIFEST_COLUMNS if c not in fields]
        if missing:
            raise ValueError("{}: missing column(s) {}".format(path, ", ".join(missing)))
        return [dict(row) for row in reader]


def discover_splits(processed_dir: str) -> List[str]:
    """Sub-directories of ``processed_dir`` that hold a ``manifest.csv``."""
    if not os.path.isdir(processed_dir):
        return []
    return [e for e in sorted(os.listdir(processed_dir))
            if os.path.isfile(manifest_path(processed_dir, e))]


def load_split_stats(processed_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(processed_dir, STATS_NAME)
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def stats_node_keys(block: Dict[str, Any]) -> List[str]:
    """Node names inside one split's stats block (skips ``class_counts`` etc.)."""
    return sorted(k for k, v in block.items()
                  if isinstance(v, dict) and all(s in v for s in TVT))


def canonical_stats(block: Any) -> str:
    """Stable JSON text of one split's stats, used for cross-node digests."""
    return json.dumps(block, sort_keys=True, separators=(",", ":"))


def order_tvt(values: Sequence[str]) -> List[str]:
    """train/val/test in pipeline order (unknown values last, alphabetically)."""
    return sorted(values, key=lambda v: (TVT.index(v) if v in TVT else len(TVT), v))


def is_singleton(group_id: str) -> bool:
    return not group_id or str(group_id).startswith("singleton")


# --------------------------------------------------------------------------- #
# (a) manifest self-consistency
# --------------------------------------------------------------------------- #
def manifest_counts(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, str, str], int]:
    """``{(node, train|val|test, fire|nofire): n}``."""
    counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        counts[(row["node"], row["split"], LABEL_KEY.get(row["label"], row["label"]))] += 1
    return dict(counts)


def check_manifest(split: str, rows: Sequence[Dict[str, str]], report: Report) -> None:
    """No path/unit shared by two nodes or by two of train/val/test."""
    if not rows:
        report.error("manifest", "manifest.csv has no rows", split)
        return

    bad_split = sorted({r["split"] for r in rows} - set(TVT))
    if bad_split:
        report.error("manifest", "unknown split value(s): {}".format(_join(bad_split)), split)
    bad_label = sorted({r["label"] for r in rows} - set(LABELS))
    if bad_label:
        report.error("manifest", "unknown label value(s): {}".format(_join(bad_label)), split)

    dup_rows = [k for k, n in Counter(
        (r["node"], r["split"], r["label"], r["path"]) for r in rows).items() if n > 1]
    if dup_rows:
        report.error("manifest", "{} duplicated manifest row(s): {}".format(
            len(dup_rows), _join(["/".join(k) for k in sorted(dup_rows)])), split)

    nodes_of_path: Dict[str, set] = defaultdict(set)
    tvt_of_path: Dict[str, set] = defaultdict(set)
    nodes_of_unit: Dict[Tuple[str, str], set] = defaultdict(set)
    tvt_of_unit: Dict[Tuple[str, str], set] = defaultdict(set)
    nodes_of_group: Dict[str, set] = defaultdict(set)
    tvt_of_group: Dict[str, set] = defaultdict(set)
    basenames: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)

    for row in rows:
        path = row["path"]
        nodes_of_path[path].add(row["node"])
        tvt_of_path[path].add(row["split"])
        basenames[(row["node"], row["split"], row["label"])][os.path.basename(path)] += 1
        if not is_singleton(row["group_id"]):
            unit = (row["group_id"], row["label"])
            nodes_of_unit[unit].add(row["node"])
            tvt_of_unit[unit].add(row["split"])
            nodes_of_group[row["group_id"]].add(row["node"])
            tvt_of_group[row["group_id"]].add(row["split"])

    shared = sorted(p for p, n in nodes_of_path.items() if len(n) > 1)
    if shared:
        detail = ["{} -> {}".format(p, "+".join(sorted(nodes_of_path[p]))) for p in shared]
        report.error("manifest", "{} image(s) assigned to more than one node: {}".format(
            len(shared), _join(detail)), split)

    leaked = sorted(p for p, s in tvt_of_path.items() if len(s) > 1)
    if leaked:
        detail = ["{} -> {}".format(p, "+".join(order_tvt(tvt_of_path[p]))) for p in leaked]
        report.error("manifest", "{} image(s) in more than one of train/val/test: {}".format(
            len(leaked), _join(detail)), split)

    unit_nodes = sorted(u for u, n in nodes_of_unit.items() if len(n) > 1)
    if unit_nodes:
        detail = ["(group={}, label={}) -> {}".format(
            u[0], u[1], "+".join(sorted(nodes_of_unit[u]))) for u in unit_nodes]
        report.error("manifest", "{} near-duplicate group(s) split across nodes: {}".format(
            len(unit_nodes), _join(detail)), split)

    unit_tvt = sorted(u for u, s in tvt_of_unit.items() if len(s) > 1)
    if unit_tvt:
        detail = ["(group={}, label={}) -> {}".format(
            u[0], u[1], "+".join(order_tvt(tvt_of_unit[u]))) for u in unit_tvt]
        report.error("manifest", "{} near-duplicate group(s) split across train/val/test: "
                                 "{}".format(len(unit_tvt), _join(detail)), split)

    # label-agnostic: a cross-label group whose Fire part sits on one node (or in
    # train) and whose No_Fire part sits elsewhere is what the leakage audit flags
    group_nodes = sorted(g for g, n in nodes_of_group.items() if len(n) > 1)
    if group_nodes:
        detail = ["group={} -> {}".format(g, "+".join(sorted(nodes_of_group[g]))) for g in group_nodes]
        report.error("manifest", "{} near-duplicate group(s) split across nodes when labels are "
                                 "ignored (cross-label group split by label): {}".format(
                                     len(group_nodes), _join(detail)), split)
    group_tvt = sorted(g for g, s_ in tvt_of_group.items() if len(s_) > 1)
    if group_tvt:
        detail = ["group={} -> {}".format(g, "+".join(order_tvt(tvt_of_group[g]))) for g in group_tvt]
        report.error("manifest", "{} near-duplicate group(s) split across train/val/test when "
                                 "labels are ignored (cross-label group split by label): {}".format(
                                     len(group_tvt), _join(detail)), split)

    collisions = []
    for key, counter in basenames.items():
        for name, count in counter.items():
            if count > 1:
                collisions.append("{}/{}/{}/{} (x{})".format(key[0], key[1], key[2], name, count))
    if collisions:
        report.error("manifest", "{} basename collision(s) -- these images overwrite each "
                                 "other in the split tree: {}".format(
                                     len(collisions), _join(sorted(collisions))), split)


# --------------------------------------------------------------------------- #
# (a2) manifest counts vs split_stats.json
# --------------------------------------------------------------------------- #
def check_stats(split: str, rows: Sequence[Dict[str, str]], stats: Optional[Dict[str, Any]],
                report: Report) -> None:
    """Per node/split fire+nofire counts and the ``class_counts`` table."""
    if stats is None:
        report.warn("stats", "no {} next to the splits; count cross-check skipped".format(
            STATS_NAME), split)
        return
    block = stats.get(split)
    if not isinstance(block, dict):
        report.error("stats", "{} has no entry for this split".format(STATS_NAME), split)
        return

    counts = manifest_counts(rows)
    manifest_nodes = sorted({r["node"] for r in rows})
    stats_nodes = stats_node_keys(block)
    if manifest_nodes != stats_nodes:
        report.error("stats", "node mismatch: manifest has [{}], {} has [{}]".format(
            ", ".join(manifest_nodes), STATS_NAME, ", ".join(stats_nodes)), split)

    for node in sorted(set(manifest_nodes) | set(stats_nodes)):
        node_block = block.get(node)
        if not isinstance(node_block, dict):
            continue
        for tvt in TVT:
            entry = node_block.get(tvt) or {}
            for key in ("fire", "nofire"):
                expected = int(entry.get(key, 0) or 0)
                actual = counts.get((node, tvt, key), 0)
                if expected != actual:
                    report.error("stats", "{}/{}/{}: manifest has {} image(s), {} says "
                                          "{}".format(node, tvt, key, actual, STATS_NAME,
                                                      expected), split)
            total = int(entry.get("total", 0) or 0)
            actual_total = sum(counts.get((node, tvt, k), 0) for k in ("fire", "nofire"))
            if total != actual_total:
                report.error("stats", "{}/{}: manifest has {} image(s), {} total is {}".format(
                    node, tvt, actual_total, STATS_NAME, total), split)

    class_counts = block.get("class_counts")
    if not isinstance(class_counts, dict):
        report.error("stats", "{} entry has no 'class_counts' table".format(STATS_NAME), split)
        return
    for node in sorted(set(manifest_nodes) | set(class_counts)):
        entry = class_counts.get(node)
        if not isinstance(entry, dict):
            report.error("stats", "class_counts has no row for node '{}'".format(node), split)
            continue
        for label in LABELS:
            expected = int(entry.get(label, 0) or 0)
            actual = sum(counts.get((node, tvt, LABEL_KEY[label]), 0) for tvt in TVT)
            if expected != actual:
                report.error("stats", "class_counts[{}][{}]: manifest has {}, {} says "
                                      "{}".format(node, label, actual, STATS_NAME, expected),
                             split)
        expected_total = int(entry.get("total", 0) or 0)
        actual_total = sum(counts.get((node, tvt, k), 0)
                           for tvt in TVT for k in ("fire", "nofire"))
        if expected_total != actual_total:
            report.error("stats", "class_counts[{}][total]: manifest has {}, {} says "
                                  "{}".format(node, actual_total, STATS_NAME, expected_total),
                         split)


# --------------------------------------------------------------------------- #
# (b) files on disk
# --------------------------------------------------------------------------- #
def check_disk(processed_dir: str, split: str, rows: Sequence[Dict[str, str]],
               report: Report) -> None:
    """Every manifest row exists under ``<split>/<node>/<tvt>/<Label>/`` and vice versa."""
    expected: Dict[Tuple[str, str, str], set] = defaultdict(set)
    for row in rows:
        expected[(row["node"], row["split"], row["label"])].add(os.path.basename(row["path"]))

    missing_total, extra_total = 0, 0
    missing_examples: List[str] = []
    extra_examples: List[str] = []
    for (node, tvt, label), names in sorted(expected.items()):
        directory = os.path.join(processed_dir, split, node, tvt, label)
        if not os.path.isdir(directory):
            report.error("disk", "missing directory {}".format(
                os.path.join(split, node, tvt, label)), split)
            missing_total += len(names)
            missing_examples.extend(sorted(names)[:MAX_REPORT])
            continue
        on_disk = set()
        for entry in os.listdir(directory):
            full = os.path.join(directory, entry)
            if os.path.isfile(full) or os.path.islink(full):
                on_disk.add(entry)
        missing = names - on_disk
        extra = on_disk - names
        broken = sorted(n for n in (names & on_disk)
                        if os.path.islink(os.path.join(directory, n))
                        and not os.path.exists(os.path.join(directory, n)))
        missing_total += len(missing)
        extra_total += len(extra)
        missing_examples.extend(
            "/".join((node, tvt, label, n)) for n in sorted(missing)[:MAX_REPORT])
        extra_examples.extend(
            "/".join((node, tvt, label, n)) for n in sorted(extra)[:MAX_REPORT])
        if broken:
            report.error("disk", "{} broken symlink(s) under {}: {}".format(
                len(broken), os.path.join(split, node, tvt, label), _join(broken)), split)

    if missing_total:
        report.error("disk", "{} manifest image(s) are not on disk: {}".format(
            missing_total, _join(missing_examples)), split)
    if extra_total:
        report.error("disk", "{} file(s) on disk are not in the manifest (stale tree? "
                             "re-run the splitter into a clean directory): {}".format(
                                 extra_total, _join(extra_examples)), split)


# --------------------------------------------------------------------------- #
# (d) group-file cross-check
# --------------------------------------------------------------------------- #
def _gid_lookup(group_map: Dict[str, int]) -> Dict[str, int]:
    """Secondary index ``<ClassDir>/<basename> -> gid`` (mirrors the splitter)."""
    by_name: Dict[str, int] = {}
    for key, gid in group_map.items():
        parts = key.split("/")
        short = "/".join(parts[-2:]) if len(parts) >= 2 else key
        by_name.setdefault(short, gid)
    return by_name


def check_group_file(split: str, rows: Sequence[Dict[str, str]], group_map: Dict[str, int],
                     report: Report) -> Dict[str, int]:
    """No group id *from the group file* crosses nodes or train/val/test.

    A group that contains both Fire and No_Fire images is turned into one unit
    per class by the splitter, so such a group may legitimately land on two
    nodes; that case is reported as a warning, everything else as an error.
    """
    by_name = _gid_lookup(group_map)
    nodes_of: Dict[Tuple[int, str], set] = defaultdict(set)
    tvt_of: Dict[Tuple[int, str], set] = defaultdict(set)
    nodes_of_gid: Dict[int, set] = defaultdict(set)
    tvt_of_gid: Dict[int, set] = defaultdict(set)
    labels_of_gid: Dict[int, set] = defaultdict(set)
    matched = 0
    mismatched: List[str] = []
    manifest_has_gids = any(not is_singleton(r["group_id"]) for r in rows)

    for row in rows:
        rel = row["path"].replace("\\", "/")
        gid = group_map.get(rel)
        if gid is None:
            gid = by_name.get("{}/{}".format(row["label"], os.path.basename(rel)))
        if gid is None:
            continue
        matched += 1
        if manifest_has_gids and not is_singleton(row["group_id"]):
            if str(row["group_id"]) != str(gid):
                mismatched.append("{}: manifest gid {} != group file gid {}".format(
                    rel, row["group_id"], gid))
        nodes_of[(gid, row["label"])].add(row["node"])
        tvt_of[(gid, row["label"])].add(row["split"])
        nodes_of_gid[gid].add(row["node"])
        tvt_of_gid[gid].add(row["split"])
        labels_of_gid[gid].add(row["label"])

    if not manifest_has_gids:
        report.warn("group_file", "manifest.csv records no group ids (the split was built "
                                  "without --group_file); checking against the file only",
                    split)
    if mismatched:
        report.error("group_file", "{} image(s) carry a different group id than the group "
                                   "file: {}".format(len(mismatched), _join(mismatched)), split)

    bad_nodes = sorted(u for u, n in nodes_of.items() if len(n) > 1)
    if bad_nodes:
        detail = ["(group={}, label={}) -> {}".format(u[0], u[1], "+".join(sorted(nodes_of[u])))
                  for u in bad_nodes]
        report.error("group_file", "{} group/label unit(s) span several nodes: {}".format(
            len(bad_nodes), _join(detail)), split)
    bad_tvt = sorted(u for u, s in tvt_of.items() if len(s) > 1)
    if bad_tvt:
        detail = ["(group={}, label={}) -> {}".format(u[0], u[1], "+".join(order_tvt(tvt_of[u])))
                  for u in bad_tvt]
        report.error("group_file", "{} group/label unit(s) span train/val/test: {}".format(
            len(bad_tvt), _join(detail)), split)

    cross_nodes = sorted(g for g in nodes_of_gid
                         if len(nodes_of_gid[g]) > 1 and len(labels_of_gid[g]) > 1)
    cross_tvt = sorted(g for g in tvt_of_gid
                       if len(tvt_of_gid[g]) > 1 and len(labels_of_gid[g]) > 1)
    if cross_nodes or cross_tvt:
        report.warn("group_file", "{} cross-label group(s) span nodes and {} span "
                                  "train/val/test; this is by design (the splitter makes one "
                                  "unit per class): {}".format(
                                      len(cross_nodes), len(cross_tvt),
                                      _join(sorted(set(cross_nodes) | set(cross_tvt)))), split)

    return {"matched_images": matched, "groups_seen": len(nodes_of_gid),
            "cross_label_groups": sum(1 for g in labels_of_gid.values() if len(g) > 1)}


# --------------------------------------------------------------------------- #
# (e) subsample sanity
# --------------------------------------------------------------------------- #
def check_subsample(split: str, frac: float, which: Sequence[str],
                    sub_counts: Dict[Tuple[str, str, str], int],
                    base_counts: Dict[Tuple[str, str, str], int],
                    granularity: int, rtol: float, report: Report) -> None:
    """``<base>_sub<f>`` holds ~``f`` x the base count per node and class.

    The splitter subsamples whole units, so a count can overshoot by up to one
    unit; the tolerance is therefore ``max(largest_unit, rtol * expected)`` and
    a non-empty class always keeps at least one unit.
    """
    for key in sorted(set(base_counts) | set(sub_counts)):
        node, tvt, cls = key
        base = base_counts.get(key, 0)
        sub = sub_counts.get(key, 0)
        if tvt not in which:
            if base != sub:
                report.error("subsample", "{}/{}/{}: not subsampled (subsample_splits={}) "
                                          "but has {} image(s) instead of {}".format(
                                              node, tvt, cls, ",".join(which), sub, base), split)
            continue
        expected = frac * base
        tol = max(float(granularity), rtol * expected)
        if base > 0 and sub < 1:
            report.error("subsample", "{}/{}/{}: base split has {} image(s) but the "
                                      "subsample has none".format(node, tvt, cls, base), split)
            continue
        if base > 0 and expected < 1 and sub <= max(1, granularity):
            continue  # the splitter keeps >= 1 unit of a non-empty class
        if abs(sub - expected) > tol:
            report.error("subsample", "{}/{}/{}: {} image(s), expected ~{:.1f} "
                                      "({:g} x {}) +/- {:.1f}".format(
                                          node, tvt, cls, sub, expected, frac, base, tol), split)


# --------------------------------------------------------------------------- #
# (c) cross-node comparison
# --------------------------------------------------------------------------- #
def _diff_lines(path_a: str, path_b: str, limit: int = 3) -> List[str]:
    """First differing lines of two text files, as readable messages."""
    with open(path_a, "r", newline="") as fh:
        a = fh.read().splitlines()
    with open(path_b, "r", newline="") as fh:
        b = fh.read().splitlines()
    out: List[str] = []
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            out.append("row {}: '{}' != '{}'".format(i, a[i][:120], b[i][:120]))
            if len(out) >= limit:
                break
    if len(a) != len(b) and len(out) < limit:
        out.append("row count differs: {} vs {} lines".format(len(a), len(b)))
    if not out:
        only_a = set(a[1:]) - set(b[1:])
        only_b = set(b[1:]) - set(a[1:])
        if only_a or only_b:
            out.append("{} row(s) only in A, {} row(s) only in B".format(
                len(only_a), len(only_b)))
    return out


def compare_dirs(base_dir: str, base_splits: Sequence[str], other_dir: str,
                 report: Report) -> Dict[str, Any]:
    """``manifest.csv`` byte-identical and per-split stats equal across two trees."""
    result: Dict[str, Any] = {"dir": os.path.abspath(other_dir), "identical_splits": [],
                              "differing_splits": []}
    if not os.path.isdir(other_dir):
        report.error("compare", "{} is not a directory".format(other_dir))
        return result

    other_splits = discover_splits(other_dir)
    only_base = sorted(set(base_splits) - set(other_splits))
    only_other = sorted(set(other_splits) - set(base_splits))
    if only_base:
        report.error("compare", "{}: missing split(s) {}".format(other_dir, _join(only_base)))
    if only_other:
        report.error("compare", "{}: extra split(s) {}".format(other_dir, _join(only_other)))

    base_stats = load_split_stats(base_dir) or {}
    other_stats = load_split_stats(other_dir) or {}

    for split in base_splits:
        if split not in other_splits:
            result["differing_splits"].append(split)
            continue
        path_a = manifest_path(base_dir, split)
        path_b = manifest_path(other_dir, split)
        same_manifest = md5_file(path_a) == md5_file(path_b)
        if not same_manifest:
            result["differing_splits"].append(split)
            report.error("compare", "{}: manifest.csv differs from {}; first difference(s): "
                                    "{}".format(other_dir, base_dir,
                                                " | ".join(_diff_lines(path_a, path_b))), split)
        block_a = canonical_stats(base_stats.get(split))
        block_b = canonical_stats(other_stats.get(split))
        same_stats = block_a == block_b
        if not same_stats:
            if split not in result["differing_splits"]:
                result["differing_splits"].append(split)
            report.error("compare", "{}: {} entry differs (md5 {} vs {})".format(
                other_dir, STATS_NAME, md5_text(block_a)[:10], md5_text(block_b)[:10]), split)
        if same_manifest and same_stats:
            result["identical_splits"].append(split)

    meta_a = (base_stats.get("_meta") or {})
    meta_b = (other_stats.get("_meta") or {})
    for key in ("seed", "nodes", "node_names", "n_fire", "n_nofire", "n_units",
                "grouped_images_matched", "n_groups_used", "cross_label_groups",
                "largest_unit"):
        if (key in meta_a or key in meta_b) and meta_a.get(key) != meta_b.get(key):
            report.error("compare", "{}: _meta.{} differs ({!r} vs {!r})".format(
                other_dir, key, meta_a.get(key), meta_b.get(key)))
    for key in ("data_dir", "link_mode", "group_file"):
        if meta_a.get(key) != meta_b.get(key):
            report.warn("compare", "{}: _meta.{} differs ({!r} vs {!r}) -- expected when the "
                                   "trees come from different machines".format(
                                       other_dir, key, meta_a.get(key), meta_b.get(key)))
    return result


# --------------------------------------------------------------------------- #
# hashes
# --------------------------------------------------------------------------- #
def split_hashes(processed_dir: str, splits: Sequence[str]) -> Dict[str, Any]:
    """md5 of every ``manifest.csv`` plus normalised ``split_stats.json`` digests."""
    stats = load_split_stats(processed_dir)
    out: Dict[str, Any] = {"dir": os.path.abspath(processed_dir), "manifests": {},
                           "split_stats": {}}
    for split in splits:
        path = manifest_path(processed_dir, split)
        with open(path, "r", newline="") as fh:
            n_rows = max(len(fh.read().splitlines()) - 1, 0)
        out["manifests"][split] = {"md5": md5_file(path), "rows": n_rows}
        if stats is not None and split in stats:
            out["split_stats"][split] = md5_text(canonical_stats(stats[split]))
    if stats is not None:
        without_meta = {k: v for k, v in stats.items() if k != "_meta"}
        out["split_stats_all_md5"] = md5_text(canonical_stats(without_meta))
        out["split_stats_raw_md5"] = md5_file(os.path.join(processed_dir, STATS_NAME))
    return out


def print_hashes(hashes: Dict[str, Any]) -> None:
    print("processed_dir: {}".format(hashes["dir"]))
    print("{:<30} {:>8}  {}".format("split", "rows", "manifest.csv md5"))
    print("-" * 78)
    for split in sorted(hashes["manifests"]):
        entry = hashes["manifests"][split]
        print("{:<30} {:>8}  {}".format(split, entry["rows"], entry["md5"]))
    if hashes.get("split_stats"):
        print("-" * 78)
        for split in sorted(hashes["split_stats"]):
            print("{:<30} {:>8}  {}".format(split, "(stats)", hashes["split_stats"][split]))
    if "split_stats_all_md5" in hashes:
        print("-" * 78)
        print("{:<30} {:>8}  {}".format(STATS_NAME + " (no _meta)", "",
                                        hashes["split_stats_all_md5"]))
        print("{:<30} {:>8}  {}".format(STATS_NAME + " (raw bytes)", "",
                                        hashes["split_stats_raw_md5"]))
        print("  note: the raw bytes legitimately differ across machines "
              "(_meta.data_dir is absolute); compare the other digests.")


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def verify_dir(processed_dir: str, splits: Optional[Sequence[str]] = None,
               check_files: bool = True, group_map: Optional[Dict[str, int]] = None,
               expect_subsample: bool = False, subsample_rtol: float = 0.25,
               report: Optional[Report] = None) -> Dict[str, Any]:
    """Run every single-tree check; returns a JSON-able summary."""
    report = report if report is not None else Report()
    summary: Dict[str, Any] = {"dir": os.path.abspath(processed_dir), "splits": {}}

    if not os.path.isdir(processed_dir):
        report.error("input", "{} is not a directory".format(processed_dir))
        return summary

    available = discover_splits(processed_dir)
    if splits:
        for name in [s for s in splits if s not in available]:
            report.error("input", "requested split '{}' has no {}".format(name, MANIFEST_NAME))
        selected = [s for s in splits if s in available]
    else:
        selected = available
    if not selected:
        report.error("input", "no split directory with a {} found under {}".format(
            MANIFEST_NAME, processed_dir))
        return summary

    stats = load_split_stats(processed_dir)
    summary["split_stats_found"] = stats is not None
    meta = (stats or {}).get("_meta") or {}
    summary["meta"] = meta
    granularity = max(int(meta.get("largest_unit") or 1), 1)

    rows_by_split: Dict[str, List[Dict[str, str]]] = {}
    for split in selected:
        path = manifest_path(processed_dir, split)
        try:
            rows = read_manifest(path)
        except (OSError, ValueError) as exc:
            report.error("manifest", str(exc), split)
            continue
        rows_by_split[split] = rows

        check_manifest(split, rows, report)
        check_stats(split, rows, stats, report)
        if check_files:
            check_disk(processed_dir, split, rows, report)
        info: Dict[str, Any] = {
            "rows": len(rows),
            "nodes": sorted({r["node"] for r in rows}),
            "manifest_md5": md5_file(path),
            "counts": {"{}|{}|{}".format(*k): v
                       for k, v in sorted(manifest_counts(rows).items())},
        }
        if group_map:
            info["group_file"] = check_group_file(split, rows, group_map, report)
        summary["splits"][split] = info

    if expect_subsample:
        sub_splits = sorted(s for s in rows_by_split if _SUB_RE.match(s))
        if not sub_splits:
            report.error("subsample", "--expect_subsample was given but no '<split>_sub<frac>' "
                                      "split exists under {}".format(processed_dir))
        for split in sub_splits:
            block = (stats or {}).get(split) or {}
            match = _SUB_RE.match(split)
            base_split = str(block.get("source_split") or match.group("base"))
            frac = float(block.get("subsample_frac") or match.group("frac"))
            which = list(block.get("subsample_splits") or ["train"])
            summary["splits"][split]["subsample"] = {
                "base_split": base_split, "frac": frac, "splits": which}
            if base_split not in rows_by_split:
                report.warn("subsample", "base split '{}' is not available; check "
                                         "skipped".format(base_split), split)
                continue
            check_subsample(split, frac, which,
                            manifest_counts(rows_by_split[split]),
                            manifest_counts(rows_by_split[base_split]),
                            granularity, subsample_rtol, report)

    return summary


def print_report(summary: Dict[str, Any], report: Report, hashes: Optional[Dict[str, Any]],
                 compares: Sequence[Dict[str, Any]]) -> None:
    print("=" * 78)
    print("  FedRGBD split verification")
    print("=" * 78)
    print("  processed_dir : {}".format(summary.get("dir")))
    meta = summary.get("meta") or {}
    if meta:
        print("  seed / nodes  : {} / {} ({})".format(
            meta.get("seed"), meta.get("nodes"), ", ".join(meta.get("node_names") or [])))
        print("  group file    : {}".format(meta.get("group_file") or "(none)"))
        print("  largest unit  : {} image(s)".format(meta.get("largest_unit")))
    print("")
    print("  {:<26} {:>8} {:>6}  {:<34} {}".format(
        "split", "rows", "nodes", "manifest.csv md5", "status"))
    print("  " + "-" * 90)
    for split in sorted(summary.get("splits", {})):
        info = summary["splits"][split]
        n_err = report.n_errors_for(split)
        status = "OK" if n_err == 0 else "FAIL ({} error(s))".format(n_err)
        print("  {:<26} {:>8} {:>6}  {:<34} {}".format(
            split, info["rows"], len(info["nodes"]), info["manifest_md5"], status))

    for entry in compares:
        print("")
        print("  compared with {}".format(entry["dir"]))
        print("    identical splits : {}".format(len(entry["identical_splits"])))
        if entry["differing_splits"]:
            print("    DIFFERING splits : {}".format(", ".join(entry["differing_splits"])))

    if hashes:
        print("")
        print_hashes(hashes)

    for level, items in (("error", report.errors), ("warning", report.warnings)):
        if not items:
            continue
        print("")
        print("  {}S ({}):".format(level.upper(), len(items)))
        for issue in items:
            where = "[{}] ".format(issue["split"]) if issue["split"] else ""
            print("    {} {}{}: {}".format(level, where, issue["check"], issue["message"]))

    print("")
    print("=" * 78)
    if report.ok:
        print("  PASS -- {} split(s) verified, {} warning(s)".format(
            len(summary.get("splits", {})), len(report.warnings)))
    else:
        print("  FAIL -- {} error(s), {} warning(s)".format(
            len(report.errors), len(report.warnings)))
    print("=" * 78)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a data/processed split tree and compare it across nodes.")
    parser.add_argument("processed_dir", nargs="?", default="data/processed",
                        help="split tree to check (default: data/processed)")
    parser.add_argument("--compare", nargs="+", default=None, metavar="DIR",
                        help="further data/processed copies (e.g. fetched from the other "
                             "nodes) whose manifests must be byte-identical")
    parser.add_argument("--splits", nargs="+", default=None, metavar="NAME",
                        help="only check these split directories (default: all)")
    parser.add_argument("--no_disk", action="store_true",
                        help="skip the manifest-vs-filesystem check (manifest-only copies)")
    parser.add_argument("--hashes_only", action="store_true",
                        help="only print one md5 per manifest.csv and exit; run this on each "
                             "node and compare the tables instead of copying the trees")
    parser.add_argument("--group_file", default=None,
                        help="groups.json/groups.csv from scripts/analyze_flame_leakage.py; "
                             "re-checks node/split leakage with the file's group ids")
    parser.add_argument("--expect_subsample", action="store_true",
                        help="check that every <split>_sub<f> holds ~f x the base train count")
    parser.add_argument("--subsample_rtol", type=float, default=0.25,
                        help="relative tolerance of the subsample check (default: 0.25)")
    parser.add_argument("--json", dest="json_out", default=None, metavar="PATH",
                        help="also write the machine-readable summary here")
    return parser


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = Report()

    if args.hashes_only:
        splits = args.splits or discover_splits(args.processed_dir)
        if not splits:
            print("no split directory with a {} found under {}".format(
                MANIFEST_NAME, args.processed_dir))
            return 1
        payload: Dict[str, Any] = {"hashes": [split_hashes(args.processed_dir, splits)]}
        print_hashes(payload["hashes"][0])
        for other in (args.compare or []):
            entry = split_hashes(other, args.splits or discover_splits(other))
            payload["hashes"].append(entry)
            print("")
            print_hashes(entry)
        if args.json_out:
            _write_json(args.json_out, payload)
        return 0

    group_map = None
    if args.group_file:
        # absolute paths in the group file are resolved against the --data_dir
        # the splitter recorded in split_stats.json["_meta"]
        meta = (load_split_stats(args.processed_dir) or {}).get("_meta") or {}
        group_map = load_group_file(args.group_file, meta.get("data_dir"))
        print("group file: {} ({} grouped paths)".format(args.group_file, len(group_map)))

    summary = verify_dir(
        args.processed_dir, splits=args.splits, check_files=not args.no_disk,
        group_map=group_map, expect_subsample=args.expect_subsample,
        subsample_rtol=args.subsample_rtol, report=report)

    base_splits = sorted(summary.get("splits", {}))
    compares = [compare_dirs(args.processed_dir, base_splits, other, report)
                for other in (args.compare or [])]
    hashes = split_hashes(args.processed_dir, base_splits) if base_splits else None

    print_report(summary, report, hashes, compares)

    summary["ok"] = report.ok
    summary["errors"] = report.errors
    summary["warnings"] = report.warnings
    summary["compare"] = compares
    summary["hashes"] = hashes
    if args.json_out:
        _write_json(args.json_out, summary)
        print("  summary written to {}".format(args.json_out))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
