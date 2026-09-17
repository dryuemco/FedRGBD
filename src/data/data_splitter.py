"""FedRGBD Data Splitter -- IID / Non-IID / Dirichlet partitioning for 2 or 3 FL nodes.

The splitter walks ``--data_dir`` for ``Fire`` / ``No_Fire`` images, partitions
them across ``--nodes`` federated clients and writes a
``<output_dir>/<split>/<node>/{train,val,test}/{Fire,No_Fire}`` tree of links
(or copies) plus ``<output_dir>/split_stats.json`` and a per-split
``manifest.csv``.

Four features were added on top of the original image-level splitter.  All of
them are opt-in: **with no new flag the produced partition is bit-identical to
the original implementation for the same ``--seed``** (same sequence of
``random`` calls, same count arithmetic).

1. ``--group_file PATH`` -- leakage-safe grouping
   ------------------------------------------------------------------
   FLAME frames are extracted from video, so consecutive frames are
   near-duplicates.  ``scripts/analyze_flame_leakage.py`` clusters them and
   writes ``groups.json`` / ``groups.csv``; feeding that file here makes the
   splitter assign whole *groups* instead of individual images, so no
   near-duplicate group is ever split across two nodes or across
   train/val/test.  Applies to iid, non_iid_label, dirichlet and subsample.

   This is implemented with a **unit** abstraction: a unit is a list of image
   paths that must stay together (size 1 when no group file is given).  Units
   are built *per class* for bookkeeping, so a group that spans both labels
   becomes one Fire unit and one No_Fire unit (counted as
   ``cross_label_groups``) -- but in group mode the two are **bundled** and
   always land on the same node and in the same one of train/val/test.  On
   FLAME this matters: 22 cross-label groups hold 20,006 of the 47,992 images
   (a video in which fire appears and disappears is one near-duplicate group
   with frames of both labels), and splitting them by label leaked up to 72 %
   of a node's test images into another node's train split.  Because
   ``random.shuffle`` on a list of N size-1 units draws exactly the same
   numbers as ``random.shuffle`` on a list of N paths, and because
   :func:`take_units` reduces to plain slicing when every unit has size 1, the
   default (no group file) code path is unchanged.

   With a real group file the units are *large* (on FLAME a whole video
   sequence is one unit -- 47,863 of 47,992 images fall into 265 groups, the
   largest holding 4,341 images of one class).  Cutting a shuffled unit list at
   a cumulative-count boundary then leaves some buckets empty (a node with no
   validation images, a Dirichlet node with 12 images).  Therefore, **whenever
   at least one unit has size > 1**, every assignment (nodes, train/val/test,
   Dirichlet proportions) uses the *largest-first greedy* rule of the authors'
   own grouped splitter: units are processed in descending size (ties broken by
   a seeded shuffle) and each one goes to the bucket with the largest remaining
   deficit ``quota - assigned`` (LPT scheduling).  The image quotas are the very
   same numbers the image-level arithmetic produces, so the partition *design*
   (equal nodes, 80/50/20 label skew, Dirichlet proportions, 70/15/15) is
   unchanged; only the rounding to whole groups differs.  The achieved counts
   are reported in ``split_stats.json`` and must be quoted in the paper.

2. ``--dirichlet_alpha A [A ...]`` -- label-skew partitions
   ------------------------------------------------------------------
   For every alpha a split ``dirichlet_{alpha:g}`` is produced.  For each class
   the per-node proportions are drawn as ``p ~ Dirichlet(alpha * 1_nodes)``
   from ``numpy.random.RandomState(seed)`` and the class' units are handed to
   the nodes at the cumulative-image-count boundaries ``cumsum(p) * n_images``.
   Draws are repeated (same RNG, so still deterministic) until every node holds
   at least ``--dirichlet_min_size`` images.  Small alpha -> extreme skew.

3. ``--subsample_frac F [F ...]`` -- data-scarcity ablations
   ------------------------------------------------------------------
   Applied **per node, after** the train/val/test split.  Every base split
   (iid, non_iid_label and each dirichlet split) is additionally emitted as
   ``<split>_sub{frac:g}`` in which the splits named by ``--subsample_splits``
   (default: ``train`` only, so val/test stay full and results remain
   comparable with the full-data runs) are reduced to the given fraction.
   Sampling is stratified per class and seeded.  Without a group file it is
   done at unit (= image) level and keeps at least one image of a non-empty
   class.  With a group file the units are whole sequences, so unit-level
   sampling cannot produce a 5 % or 1 % training set (one unit may be 30 % of a
   node); the subsample is therefore drawn at *image* level **inside** the
   node's own units.  Dropped images go nowhere, so no near-duplicate group
   ever spans two nodes or two of train/val/test -- the leakage guarantee is
   untouched -- but the low-data regime reduces the number of *frames* per
   client, not the number of sequences (state this in the paper).

4. ``--export_manifests DIR`` / ``--from_manifest DIR`` -- ship the partition itself

   The partition depends on the order in which :func:`find_images` walks the
   dataset (``os.walk``), which is **not** guaranteed to match across machines
   or filesystems.  Regenerating the split independently on every node is
   therefore unsafe: two nodes could end up with different partitions, which
   destroys both the federated protocol and the leakage guarantee.  Instead the
   authoritative partition is exported once and replayed everywhere:

       # once, on the machine that produced data/processed
       python src/data/data_splitter.py --export_manifests data/splits

       # on every node, after the raw dataset is in place
       python src/data/data_splitter.py --from_manifest data/splits \
           --data_dir data/raw/flame_dataset --output_dir data/processed \
           --link_mode hardlink --clean --verify

   ``--export_manifests`` writes one gzipped ``manifest.csv`` per split (about
   1.1 MB for all 15 FLAME splits, small enough to track in git) plus
   ``split_stats.json``.  ``--from_manifest`` places exactly the listed files,
   uses no random number generator at all and reproduces the tree byte for byte,
   so the manifest MD5s of ``analysis/leakage/P0_SUMMARY.md`` match on every node.

5. ``--verify`` -- post-write self-check
   ------------------------------------------------------------------
   After every split has been written, the ``manifest.csv`` files are read
   back and checked: no ``(group_id, label)`` unit may span two nodes or two
   of train/val/test, and the manifest counts must equal ``split_stats.json``.
   The check is delegated to ``scripts/verify_splits.py`` when that module can
   be imported (so it also covers duplicated rows, images shared by two nodes
   and basename collisions) and falls back to a local minimal check otherwise.
   It prints ``VERIFY PASS`` / ``VERIFY FAIL`` and the CLI exits with status 1
   on FAIL.  ``scripts/verify_splits.py`` remains the full stand-alone tool
   (cross-node comparison, on-disk checks, md5 tables).

Usage
-----
    python3 src/data/data_splitter.py --data_dir data/raw/flame_dataset \\
        --output_dir data/processed --nodes 3 --seed 42

    python3 src/data/data_splitter.py --data_dir data/raw/flame_dataset \\
        --output_dir data/processed --nodes 3 \\
        --group_file analysis/leakage/groups.json \\
        --dirichlet_alpha 0.1 0.5 1.0 --subsample_frac 0.05 0.01 --clean --verify

``--clean`` is REQUIRED whenever ``--output_dir`` already holds a partition.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import random
import shutil
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
NODE_NAMES = ['node_a', 'node_b', 'node_c']
CLASS_DIRS = {'fire': 'Fire', 'no_fire': 'No_Fire', 'nofire': 'No_Fire'}

# A "unit" is a list of image paths that must be kept together.
Unit = List[str]


# --------------------------------------------------------------------------- #
# group-file loading (shared with scripts/analyze_flame_leakage.py)
# --------------------------------------------------------------------------- #
#: metadata keys that may sit next to the path->gid entries of a flat group file
_GROUP_META_KEYS = frozenset({
    'n_images', 'n_groups', 'threshold', 'phash_threshold', 'thresholds',
    'hash_size', 'method', 'data_dir', 'settings', 'timestamp', 'dataset',
})


def _normalise_group_path(raw: str, data_dir: Optional[str] = None) -> str:
    """Local copy of ``analyze_flame_leakage.normalise_group_path``."""
    path = str(raw).replace('\\', '/')
    while path.startswith('./'):
        path = path[2:]
    if not os.path.isabs(path):
        return path
    if data_dir:
        for root in {os.path.abspath(data_dir), os.path.realpath(data_dir)}:
            try:
                rel = os.path.relpath(os.path.realpath(path), root).replace(os.sep, '/')
            except (OSError, ValueError):
                continue
            if not rel.startswith('..'):
                return rel
    parts = [x for x in path.split('/') if x]
    return '/'.join(parts[-2:]) if len(parts) >= 2 else path


def _local_load_group_file(path: str, data_dir: Optional[str] = None) -> Dict[str, int]:
    """Standalone fallback copy of ``analyze_flame_leakage.load_group_file``.

    Reads ``groups.json`` in either layout -- ``{"groups": {"gid": ["Fire/a.jpg",
    ...]}}`` (this repo) or ``{"groups": {"<path>": gid}}`` / a bare
    ``{"<path>": gid}`` (the authors' script) -- or ``groups.csv``
    (``path,group_id`` columns), into ``{relative_path: group_id}``.
    """
    mapping: Dict[str, int] = {}
    if path.lower().endswith('.json'):
        with open(path) as f:
            data = json.load(f)
        embedded = data.get('data_dir') if isinstance(data.get('data_dir'), str) else None
        if isinstance(data.get('groups'), dict):
            groups = data['groups']
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
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                mapping[_normalise_group_path(row['path'], data_dir)] = int(row['group_id'])
    return mapping


def load_group_file(path: str, data_dir: Optional[str] = None) -> Dict[str, int]:
    """Use the canonical parser from ``scripts/``; fall back to the local copy.

    The fallback keeps the splitter runnable standalone (e.g. on the Jetson,
    where ``scripts/`` or its numpy/PIL imports may not be available).
    ``data_dir`` lets absolute paths (as written by the authors' own leakage
    script) be resolved back to ``--data_dir``-relative keys.
    """
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from scripts.analyze_flame_leakage import load_group_file as _impl  # noqa: WPS433
        return _impl(path, data_dir)
    except Exception:  # pragma: no cover - environment dependent
        return _local_load_group_file(path, data_dir)


# --------------------------------------------------------------------------- #
# image discovery  (unchanged behaviour)
# --------------------------------------------------------------------------- #
def find_images(data_dir):
    fire, nofire = [], []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                full = os.path.join(root, f)
                parent = os.path.basename(root).lower()
                if parent == 'fire':
                    fire.append(full)
                elif parent in ('no_fire', 'nofire'):
                    nofire.append(full)
    return fire, nofire


# --------------------------------------------------------------------------- #
# units
# --------------------------------------------------------------------------- #
def n_images(units: Sequence[Unit]) -> int:
    return sum(len(u) for u in units)


def flatten(units: Sequence[Unit]) -> List[str]:
    return [p for u in units for p in u]


def take_units(units: Sequence[Unit], target: int) -> Tuple[List[Unit], List[Unit]]:
    """Take units from the front until the cumulative image count reaches ``target``.

    Returns ``(taken, rest)``.  When every unit has size 1 this is exactly
    ``units[:target], units[target:]`` -- which is what keeps the default code
    path identical to the original image-level slicing.
    """
    cum, i = 0, 0
    while i < len(units) and cum < target:
        cum += len(units[i])
        i += 1
    return list(units[:i]), list(units[i:])


def split_units_into(units: Sequence[Unit], n: int) -> List[List[Unit]]:
    """Split units into ``n`` parts of (nearly) equal image count.

    Identical to the original ``split_into(lst, n)`` when all units have size 1.
    """
    total = n_images(units)
    k, m = divmod(total, n)
    bounds = [i * k + min(i, m) for i in range(n + 1)]
    parts, rest, taken = [], list(units), 0
    for i in range(n):
        part, rest = take_units(rest, bounds[i + 1] - taken)
        taken += n_images(part)
        parts.append(part)
    if rest:  # only possible with oversized units at the tail
        parts[-1].extend(rest)
    return parts


def is_grouped(*unit_lists: Sequence[Unit]) -> bool:
    """True when at least one unit holds more than one image (group mode)."""
    return any(len(u) > 1 for units in unit_lists for u in units)


def assign_units_greedy(units: Sequence[Unit], quotas: Sequence[float],
                        seed: int) -> List[List[Unit]]:
    """Largest-first greedy assignment of whole units to ``len(quotas)`` buckets.

    Units are visited in descending image count (ties broken by a shuffle from
    a private ``random.Random(seed)``) and each goes to the bucket with the
    largest remaining deficit ``quota - assigned`` (LPT rule; the authors'
    grouped splitter uses the same deficit rule in shuffled order).  Every unit
    is placed, so the bucket sums add up to the input; each bucket deviates
    from its quota by at most the largest unit it received.  Deterministic.
    """
    n_buckets = len(quotas)
    if n_buckets == 0:
        return []
    order = list(range(len(units)))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda i: -len(units[i]))  # stable: shuffled order breaks ties
    parts: List[List[Unit]] = [[] for _ in range(n_buckets)]
    filled = [0] * n_buckets
    for i in order:
        deficits = [q - f for q, f in zip(quotas, filled)]
        b = max(range(n_buckets), key=lambda k: (deficits[k], -k))
        parts[b].append(units[i])
        filled[b] += len(units[i])
    return parts


#: a bundle is the per-class view of ONE near-duplicate group: ``(fire_unit, nofire_unit)``
#: where either side may be empty.  Pure groups have one side, cross-label groups both.
Bundle = Tuple[Unit, Unit]


def bundle_units(fire_units: Sequence[Unit], nofire_units: Sequence[Unit],
                 gid_of: Optional[Dict[str, str]]) -> List[Bundle]:
    """Pair the Fire and No_Fire units that belong to the same group id.

    Without ``gid_of`` (or for singleton ids) every unit is its own bundle.
    The order is fire units first, then the nofire units that were not paired.
    """
    def gid(unit: Unit) -> Optional[str]:
        if not gid_of or not unit:
            return None
        g = gid_of.get(unit[0])
        return None if g is None or str(g).startswith('singleton') else str(g)

    bundles: List[Bundle] = []
    index: Dict[str, int] = {}
    for u in fire_units:
        g = gid(u)
        if g is not None:
            index[g] = len(bundles)
        bundles.append((list(u), []))
    for u in nofire_units:
        g = gid(u)
        pos = index.get(g) if g is not None else None
        if pos is None:
            bundles.append(([], list(u)))
        else:
            fire_part, nofire_part = bundles[pos]
            bundles[pos] = (fire_part, list(nofire_part) + list(u))
    return bundles


def assign_bundles_greedy(bundles: Sequence[Bundle], quotas_fire: Sequence[float],
                          quotas_nofire: Sequence[float], seed: int
                          ) -> List[Tuple[List[Unit], List[Unit]]]:
    """Largest-first greedy assignment of whole *bundles* to buckets with per-class quotas.

    Bundles are visited in descending total image count (ties broken by a
    seeded shuffle).  A bundle goes to the bucket with the largest
    composition-weighted remaining deficit
    ``w_f * (qf - filled_f) + w_n * (qn - filled_n)`` with ``w = share of the
    bundle's images in that class``; for a pure bundle this is exactly the
    per-class deficit rule of :func:`assign_units_greedy`.  Returns, per bucket,
    ``(fire_units, nofire_units)``.
    """
    n_buckets = len(quotas_fire)
    assert n_buckets == len(quotas_nofire)
    if n_buckets == 0:
        return []
    order = list(range(len(bundles)))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda i: -(len(bundles[i][0]) + len(bundles[i][1])))
    out: List[Tuple[List[Unit], List[Unit]]] = [([], []) for _ in range(n_buckets)]
    filled_f = [0] * n_buckets
    filled_n = [0] * n_buckets
    for i in order:
        fire_part, nofire_part = bundles[i]
        nf, nn = len(fire_part), len(nofire_part)
        tot = max(nf + nn, 1)
        wf, wn = nf / tot, nn / tot
        scores = [wf * (quotas_fire[b] - filled_f[b]) + wn * (quotas_nofire[b] - filled_n[b])
                  for b in range(n_buckets)]
        b = max(range(n_buckets), key=lambda k: (scores[k], -k))
        if fire_part:
            out[b][0].append(list(fire_part))
            filled_f[b] += nf
        if nofire_part:
            out[b][1].append(list(nofire_part))
            filled_n[b] += nn
    return out


def _tvt_quotas(n: int, train: float = 0.7, val: float = 0.15) -> List[int]:
    """The original ``int(n*0.7)`` / ``int(n*0.85)`` arithmetic as three quotas."""
    t, v = int(n * train), int(n * (train + val))
    return [t, v - t, n - v]


def split_node_bundled(fire_units: Sequence[Unit], nofire_units: Sequence[Unit],
                       gid_of: Optional[Dict[str, str]], seed: int,
                       train: float = 0.7, val: float = 0.15) -> Dict[str, Dict[str, List[Unit]]]:
    """Group-mode 70/15/15 split of one node: whole bundles, per-class quotas."""
    bundles = bundle_units(fire_units, nofire_units, gid_of)
    qf = _tvt_quotas(n_images(fire_units), train, val)
    qn = _tvt_quotas(n_images(nofire_units), train, val)
    parts = assign_bundles_greedy(bundles, qf, qn, seed)
    names = ('train', 'val', 'test')
    return {'fire': {nm: parts[i][0] for i, nm in enumerate(names)},
            'nofire': {nm: parts[i][1] for i, nm in enumerate(names)}}


def split_into(lst, n):
    """Original image-level helper, kept for backwards compatibility."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def _group_lookup(group_map: Dict[str, int]) -> Dict[str, Tuple[int, str]]:
    """Secondary index ``<ClassDir>/<basename> -> (gid, original key)``.

    Used as a match fallback when the group file stores longer paths than the
    ones discovered under ``--data_dir``; the original key is kept so the caller
    can report how much of the group file was actually consumed.
    """
    by_name: Dict[str, Tuple[int, str]] = {}
    for key, gid in group_map.items():
        parts = key.split('/')
        short = '/'.join(parts[-2:]) if len(parts) >= 2 else key
        by_name.setdefault(short, (gid, key))
    return by_name


def build_units(fire: Sequence[str], nofire: Sequence[str], data_dir: str,
                group_map: Optional[Dict[str, int]]) -> Tuple[List[Unit], List[Unit], Dict[str, str], dict]:
    """Group image paths into per-class units.

    Returns ``(fire_units, nofire_units, gid_of_path, info)``.  Without a group
    map every unit is a single image and the order is preserved exactly, so the
    downstream shuffles reproduce the original partition.  A group present in
    both classes yields one unit per class (counted in ``cross_label_groups``).
    """
    gid_of: Dict[str, str] = {}
    info = {'n_matched': 0, 'n_groups_used': 0, 'cross_label_groups': 0,
            'largest_unit': 1, 'n_units': len(fire) + len(nofire),
            'group_paths_total': len(group_map or {}), 'group_paths_matched': 0}
    if not group_map:
        for p in list(fire) + list(nofire):
            gid_of[p] = 'singleton'
        return [[p] for p in fire], [[p] for p in nofire], gid_of, info

    by_name = _group_lookup(group_map)
    root = os.path.abspath(data_dir)
    out: List[List[Unit]] = []
    classes_of_gid: Dict[int, set] = {}
    n_matched = 0
    n_singleton = 0
    matched_keys = set()

    for paths, cls in ((fire, 'Fire'), (nofire, 'No_Fire')):
        units: List[Unit] = []
        index: Dict[int, int] = {}  # gid -> position in `units`
        for p in paths:
            try:
                rel = os.path.relpath(os.path.abspath(p), root).replace(os.sep, '/')
            except ValueError:  # different drive on Windows
                rel = os.path.basename(p)
            gid = group_map.get(rel)
            if gid is not None:
                matched_keys.add(rel)
            else:
                hit = by_name.get('{}/{}'.format(cls, os.path.basename(p)))
                if hit is not None:
                    gid, source_key = hit
                    matched_keys.add(source_key)
            if gid is None:
                gid_of[p] = 'singleton_{}'.format(n_singleton)
                n_singleton += 1
                units.append([p])
                continue
            n_matched += 1
            gid_of[p] = str(gid)
            classes_of_gid.setdefault(gid, set()).add(cls)
            pos = index.get(gid)
            if pos is None:
                index[gid] = len(units)
                units.append([p])
            else:
                units[pos].append(p)
        out.append(units)

    fire_units, nofire_units = out
    info['n_matched'] = n_matched
    info['n_groups_used'] = len(classes_of_gid)
    info['cross_label_groups'] = sum(1 for c in classes_of_gid.values() if len(c) > 1)
    info['n_units'] = len(fire_units) + len(nofire_units)
    info['largest_unit'] = max([len(u) for u in fire_units + nofire_units] or [0])
    info['group_paths_matched'] = len(matched_keys)
    return fire_units, nofire_units, gid_of, info


def warn_group_coverage(info: dict, group_file: Optional[str] = None,
                        min_coverage: float = 0.5) -> bool:
    """Shout when barely any path of ``--group_file`` matched a discovered image.

    A group file written for a different directory layout matches nothing, the
    splitter then silently treats every image as a singleton and the resulting
    partition is NOT leakage-safe.  Returns True when the warning fired.
    """
    total = int(info.get('group_paths_total') or 0)
    matched = int(info.get('group_paths_matched') or 0)
    if total <= 0:
        return False
    coverage = matched / float(total)
    if coverage >= min_coverage:
        return False
    print("!" * 72)
    print("  WARNING: only {}/{} ({:.1%}) of the paths in {} matched an image".format(
        matched, total, coverage, group_file or 'the group file'))
    print("  under --data_dir.  Unmatched images are treated as SINGLETONS, so the")
    print("  partition is NOT group-safe and near-duplicate frames can still be")
    print("  split across nodes and across train/val/test.")
    print("  Check that the group file was built from this --data_dir (its paths must")
    print("  be relative to it, e.g. 'Fire/frame_00001.jpg').")
    print("!" * 72)
    return True


# --------------------------------------------------------------------------- #
# train / val / test
# --------------------------------------------------------------------------- #
def split_units(units: Sequence[Unit], train=0.7, val=0.15, seed=42,
                grouped: bool = False) -> Dict[str, List[Unit]]:
    """70/15/15 split of a list of units -- reseeds ``random`` exactly like the original.

    In group mode (``grouped=True``) the same 70/15/15 image quotas are filled
    with :func:`assign_units_greedy` so that oversized units cannot empty a
    bucket; the global ``random`` state is still advanced exactly as before.
    """
    random.seed(seed)
    s = list(units)
    random.shuffle(s)
    n = n_images(s)
    t, v = int(n * train), int(n * (train + val))
    if grouped:
        tr, va, te = assign_units_greedy(s, [t, v - t, n - v], seed)
        return {'train': tr, 'val': va, 'test': te}
    tr, rest = take_units(s, t)
    va, te = take_units(rest, v - n_images(tr))
    return {'train': tr, 'val': va, 'test': te}


def split_list(items, train=0.7, val=0.15, seed=42):
    """Original image-level 70/15/15 split (kept for backwards compatibility)."""
    out = split_units([[i] for i in items], train, val, seed)
    return {k: flatten(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# materialisation (symlink / hardlink / copy)
# --------------------------------------------------------------------------- #
_LINK_STATE = {'mode': 'symlink', 'warned': False}


def set_link_mode(mode: str) -> None:
    _LINK_STATE['mode'] = mode
    _LINK_STATE['warned'] = False


def _place(src: str, link: Path) -> None:
    mode = _LINK_STATE['mode']
    src = os.path.abspath(src)
    if mode == 'symlink':
        try:
            os.symlink(src, str(link))
            return
        except OSError as exc:
            if not _LINK_STATE['warned']:
                print("  WARNING: symlink failed ({}); falling back to --link_mode copy".format(exc))
                _LINK_STATE['warned'] = True
            _LINK_STATE['mode'] = 'copy'
            mode = 'copy'
    if mode == 'hardlink':
        try:
            os.link(src, str(link))
            return
        except OSError as exc:
            if not _LINK_STATE['warned']:
                print("  WARNING: hardlink failed ({}); falling back to --link_mode copy".format(exc))
                _LINK_STATE['warned'] = True
            _LINK_STATE['mode'] = 'copy'
    shutil.copy2(src, str(link))


def link_files(file_list, dest_dir, class_name):
    dest = Path(dest_dir) / class_name
    dest.mkdir(parents=True, exist_ok=True)
    for src in file_list:
        link = dest / os.path.basename(src)
        if link.is_symlink() or link.exists():
            link.unlink()
        _place(src, link)


def prepare_split_dir(output_dir: str, split_name: str, clean: bool = False) -> bool:
    """Delete (``--clean``) or loudly warn about a pre-existing split directory.

    ``link_files`` only replaces the *same* basename, so images that move to a
    different node or a different train/val/test bucket in a re-run are left
    behind by the previous partition.  The tree then holds both partitions at
    once, which silently puts training images into val/test.  Returns True when
    the directory was removed.
    """
    target = os.path.join(output_dir, split_name)
    if not os.path.isdir(target):
        return False
    with os.scandir(target) as it:
        if next(it, None) is None:
            return False
    if clean:
        print("  --clean: removing the stale {} tree".format(target))
        shutil.rmtree(target)
        return True
    print("!" * 72)
    print("  WARNING: {} already exists and is not empty.".format(target))
    print("  Files written by a PREVIOUS partition are NOT removed: any image that")
    print("  moves to another node or to another train/val/test bucket will exist")
    print("  twice, which silently leaks training images into val/test.")
    print("  Re-run with --clean (or delete the directory first) when re-splitting.")
    print("!" * 72)
    return False


# --------------------------------------------------------------------------- #
# stats / manifest
# --------------------------------------------------------------------------- #
def _counts(nf: int, nn: int) -> dict:
    return {'fire': nf, 'nofire': nn, 'total': nf + nn,
            'fire_ratio': round(nf / max(nf + nn, 1), 3)}


def class_counts_table(tvt: Dict[str, Dict[str, Dict[str, List[Unit]]]]) -> dict:
    """``{node: {"Fire": n, "No_Fire": n, "total": n, "fire_ratio": r}}`` over all splits."""
    table = {}
    for node, per_class in tvt.items():
        nf = sum(n_images(per_class['fire'][sp]) for sp in ('train', 'val', 'test'))
        nn = sum(n_images(per_class['nofire'][sp]) for sp in ('train', 'val', 'test'))
        table[node] = {'Fire': nf, 'No_Fire': nn, 'total': nf + nn,
                       'fire_ratio': round(nf / max(nf + nn, 1), 3)}
    return table


def write_manifest(output_dir: str, split_name: str,
                   tvt: Dict[str, Dict[str, Dict[str, List[Unit]]]],
                   data_dir: str, gid_of: Dict[str, str]) -> str:
    """``node,split,label,path,group_id`` for every placed image."""
    path = os.path.join(output_dir, split_name, 'manifest.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    root = os.path.abspath(data_dir)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['node', 'split', 'label', 'path', 'group_id'])
        for node, per_class in tvt.items():
            for sp in ('train', 'val', 'test'):
                for key, label in (('fire', 'Fire'), ('nofire', 'No_Fire')):
                    for p in flatten(per_class[key][sp]):
                        try:
                            rel = os.path.relpath(os.path.abspath(p), root).replace(os.sep, '/')
                        except ValueError:
                            rel = os.path.basename(p)
                        w.writerow([node, sp, label, rel, gid_of.get(p, 'singleton')])
    return path


# --------------------------------------------------------------------------- #
# split creation
# --------------------------------------------------------------------------- #
def partition_to_tvt(node_units: Dict[str, Dict[str, List[Unit]]], seed: int,
                     grouped: bool = False, gid_of: Optional[Dict[str, str]] = None):
    """Per-node 70/15/15 split.

    The ``random`` calls happen in exactly the original order (node by node,
    fire then nofire) so the global RNG state evolves as it did before.  In
    group mode each node is split with :func:`split_node_bundled` instead, so a
    cross-label group stays in one of train/val/test.
    """
    if grouped:
        return {node: split_node_bundled(nd['fire'], nd['nofire'], gid_of, seed)
                for node, nd in node_units.items()}
    return {node: {'fire': split_units(nd['fire'], seed=seed),
                   'nofire': split_units(nd['nofire'], seed=seed)}
            for node, nd in node_units.items()}


def materialize(output_dir: str, split_name: str, tvt, data_dir: str,
                gid_of: Dict[str, str], extra: Optional[dict] = None, quiet: bool = False,
                clean: bool = False) -> dict:
    """Place the files, write ``manifest.csv`` and return the stats block."""
    prepare_split_dir(output_dir, split_name, clean)
    stats = {}
    for node, per_class in tvt.items():
        node_stats = {}
        for sp in ('train', 'val', 'test'):
            d = os.path.join(output_dir, split_name, node, sp)
            fire_paths = flatten(per_class['fire'][sp])
            nofire_paths = flatten(per_class['nofire'][sp])
            link_files(fire_paths, d, 'Fire')
            link_files(nofire_paths, d, 'No_Fire')
            node_stats[sp] = _counts(len(fire_paths), len(nofire_paths))
        stats[node] = node_stats
        total = sum(s['total'] for s in node_stats.values())
        tf = sum(s['fire'] for s in node_stats.values())
        if not quiet:
            print("  {}: {} imgs (Fire:{}, NoFire:{}, ratio:{:.1%})  "
                  "train/val/test={}/{}/{}".format(
                      node, total, tf, total - tf, tf / max(total, 1),
                      node_stats['train']['total'], node_stats['val']['total'],
                      node_stats['test']['total']))
    stats['class_counts'] = class_counts_table(tvt)
    if extra:
        stats.update(extra)
    write_manifest(output_dir, split_name, tvt, data_dir, gid_of)
    return stats


def create_split(output_dir, split_name, node_data, seed, data_dir='.', gid_of=None,
                 extra=None, quiet=False, clean=False, grouped=False):
    """Split each node 70/15/15 and materialise it (original entry point)."""
    tvt = partition_to_tvt(node_data, seed, grouped=grouped, gid_of=gid_of)
    return materialize(output_dir, split_name, tvt, data_dir, gid_of or {}, extra, quiet,
                       clean), tvt


# --------------------------------------------------------------------------- #
# partitioners
# --------------------------------------------------------------------------- #
def _equal_quotas(total: int, n: int) -> List[int]:
    k, m = divmod(total, n)
    return [k + (1 if i < m else 0) for i in range(n)]


def _from_bundle_parts(parts, node_names):
    return {name: {'fire': parts[i][0], 'nofire': parts[i][1]} for i, name in enumerate(node_names)}


def partition_iid(fire_units, nofire_units, node_names, grouped: bool = False, seed: int = 0,
                  gid_of: Optional[Dict[str, str]] = None):
    """Equal random split -- caller must have shuffled the unit lists already.

    Group mode fills the same equal per-class quotas greedily (largest bundle
    first; a cross-label group is one bundle).
    """
    n = len(node_names)
    if grouped:
        parts = assign_bundles_greedy(bundle_units(fire_units, nofire_units, gid_of),
                                      _equal_quotas(n_images(fire_units), n),
                                      _equal_quotas(n_images(nofire_units), n), seed)
        return _from_bundle_parts(parts, node_names)
    else:
        fp = split_units_into(fire_units, n)
        np_ = split_units_into(nofire_units, n)
    return {name: {'fire': f, 'nofire': n_} for name, f, n_ in zip(node_names, fp, np_)}


def label_skew_quotas(n_fire: int, n_nofire: int, n_nodes: int) -> Tuple[List[int], List[int]]:
    """Per-node (fire, nofire) image quotas of the original label-skew heuristic.

    2 nodes: A is 70% fire.  3 nodes: A=80% fire, B absorbs the remainder
    (~88.5% fire on FLAME), C=80% no-fire.  Exactly the original arithmetic.
    """
    total = n_fire + n_nofire
    per_node = total // n_nodes
    if n_nodes == 2:
        a_fire = min(int(per_node * 0.7), n_fire)
        a_nofire = per_node - a_fire
        return [a_fire, n_fire - a_fire], [a_nofire, n_nofire - a_nofire]
    a_fire = min(int(per_node * 0.8), n_fire)
    a_nofire = per_node - a_fire
    c_nofire = min(int(per_node * 0.8), n_nofire - a_nofire)
    c_fire = per_node - c_nofire
    b_fire = n_fire - a_fire - c_fire
    b_nofire = n_nofire - a_nofire - c_nofire
    return [a_fire, b_fire, c_fire], [a_nofire, b_nofire, c_nofire]


def partition_non_iid_label(fire_units, nofire_units, node_names, grouped: bool = False,
                            seed: int = 0, gid_of: Optional[Dict[str, str]] = None):
    """Original label-skew heuristic, expressed with unit-wise takes.

    2 nodes: A is 70% fire.  3 nodes: A=80% fire, B balanced, C=80% no-fire.
    The arithmetic is exactly the original one; ``take_units`` replaces the
    image-index slicing and is equivalent when units have size 1.  In group
    mode the same quotas are filled greedily (largest unit first).
    """
    n_nodes = len(node_names)
    n_fire, n_nofire = n_images(fire_units), n_images(nofire_units)
    if grouped:
        qf, qn = label_skew_quotas(n_fire, n_nofire, n_nodes)
        parts = assign_bundles_greedy(bundle_units(fire_units, nofire_units, gid_of), qf, qn, seed)
        return _from_bundle_parts(parts, node_names)
    total = n_fire + n_nofire
    per_node = total // n_nodes

    if n_nodes == 2:
        a_fire = min(int(per_node * 0.7), n_fire)
        a_nofire = per_node - a_fire
        af, bf = take_units(fire_units, a_fire)
        an, bn = take_units(nofire_units, a_nofire)
        return {'node_a': {'fire': af, 'nofire': an},
                'node_b': {'fire': bf, 'nofire': bn}}

    a_fire = min(int(per_node * 0.8), n_fire)
    a_nofire = per_node - a_fire
    c_nofire = min(int(per_node * 0.8), n_nofire - a_nofire)
    c_fire = per_node - c_nofire
    b_fire = n_fire - a_fire - c_fire
    b_nofire = n_nofire - a_nofire - c_nofire
    af, rest_f = take_units(fire_units, a_fire)
    bf, cf = take_units(rest_f, b_fire)
    an, rest_n = take_units(nofire_units, a_nofire)
    bn, cn = take_units(rest_n, b_nofire)
    return {'node_a': {'fire': af, 'nofire': an},
            'node_b': {'fire': bf, 'nofire': bn},
            'node_c': {'fire': cf, 'nofire': cn}}


def _assign_by_proportions(units: Sequence[Unit], p: np.ndarray, grouped: bool = False,
                           seed: int = 0) -> List[List[Unit]]:
    """Hand units to nodes at the cumulative-image-count boundaries ``cumsum(p)*N``.

    Group mode: the quotas ``p * N`` are filled greedily (largest unit first).
    """
    total = n_images(units)
    if grouped:
        quotas = [float(x) * total for x in np.asarray(p, dtype=float)]
        return assign_units_greedy(units, quotas, seed)
    bounds = np.cumsum(np.asarray(p, dtype=float)) * total
    parts, rest, taken = [], list(units), 0
    n_nodes = len(p)
    for i in range(n_nodes):
        target = total if i == n_nodes - 1 else int(round(float(bounds[i])))
        part, rest = take_units(rest, target - taken)
        taken += n_images(part)
        parts.append(part)
    if rest:
        parts[-1].extend(rest)
    return parts


def partition_dirichlet(fire_units, nofire_units, node_names, alpha, seed,
                        min_size=10, max_tries=1000, grouped: bool = False,
                        gid_of: Optional[Dict[str, str]] = None):
    """Label-skew partition: ``p_c ~ Dirichlet(alpha * 1_nodes)`` per class.

    Redraws (advancing the same ``RandomState``) until every node holds at least
    ``min_size`` images.  Returns ``(node_units, proportions)``.
    """
    n_nodes = len(node_names)
    total = n_images(fire_units) + n_images(nofire_units)
    if min_size * n_nodes > total:
        raise ValueError(
            "--dirichlet_min_size {} x {} nodes = {} images required but the dataset only "
            "has {}".format(min_size, n_nodes, min_size * n_nodes, total))

    rng = np.random.RandomState(seed)
    # A fixed, seed-dependent order avoids handing nodes filename-contiguous
    # blocks; it uses a private Random instance so the global RNG is untouched.
    shuffler = random.Random(seed)
    fu, nu = list(fire_units), list(nofire_units)
    shuffler.shuffle(fu)
    shuffler.shuffle(nu)

    classes = (('fire', fu), ('nofire', nu))
    bundles = bundle_units(fu, nu, gid_of) if grouped else None
    for _ in range(max_tries):
        props = {}
        parts = {}
        for key, units in classes:
            p = rng.dirichlet(np.repeat(float(alpha), n_nodes))
            props[key] = p
            if not grouped:
                parts[key] = _assign_by_proportions(units, p)
        if grouped:
            # whole bundles (a cross-label group stays on one node), per-class quotas p*N
            qf = [float(x) * n_images(fu) for x in props['fire']]
            qn = [float(x) * n_images(nu) for x in props['nofire']]
            bparts = assign_bundles_greedy(bundles, qf, qn, seed)
            parts = {'fire': [bp[0] for bp in bparts], 'nofire': [bp[1] for bp in bparts]}
        sizes = [n_images(parts['fire'][i]) + n_images(parts['nofire'][i]) for i in range(n_nodes)]
        if min(sizes) >= min_size:
            node_units = {name: {'fire': parts['fire'][i], 'nofire': parts['nofire'][i]}
                          for i, name in enumerate(node_names)}
            proportions = {'Fire': [round(float(x), 6) for x in props['fire']],
                           'No_Fire': [round(float(x), 6) for x in props['nofire']]}
            return node_units, proportions
    raise ValueError(
        "Could not find a Dirichlet partition with >= {} images per node after {} draws "
        "(alpha={}, nodes={}, images={}). Lower --dirichlet_min_size or raise "
        "--dirichlet_alpha.".format(min_size, max_tries, alpha, n_nodes, total))


# --------------------------------------------------------------------------- #
# subsampling (per node, after the train/val/test split)
# --------------------------------------------------------------------------- #
def _stable_seed(seed: int, name: str) -> int:
    """Deterministic across processes (``hash()`` on str is salted, crc32 is not)."""
    return (seed + zlib.crc32(name.encode('utf-8'))) % (2 ** 31)


def _subsample_images_within_units(units: Sequence[Unit], target: int,
                                   rng: random.Random) -> List[Unit]:
    """Keep ``target`` images (>= 1) drawn uniformly from ``units``; the survivors
    stay in their original unit (group) so ``group_id`` stays meaningful."""
    flat = [(ui, p) for ui, u in enumerate(units) for p in u]
    rng.shuffle(flat)
    keep = flat[:max(target, 1)]
    kept: Dict[int, List[str]] = {}
    for ui, p in sorted(keep):
        kept.setdefault(ui, []).append(p)
    return [kept[ui] for ui in sorted(kept)]


def subsample_tvt(tvt, frac: float, which: Sequence[str], seed: int, split_name: str,
                  grouped: bool = False):
    """Reduce the named splits to ``frac`` of their images, stratified per class.

    Unit level (identical to the original behaviour) without a group file;
    image level *inside* the node's own units in group mode -- see the module
    docstring for why.
    """
    rng = random.Random(_stable_seed(seed, '{}|{:g}'.format(split_name, frac)))
    out = {}
    for node, per_class in sorted(tvt.items()):
        node_out = {'fire': {}, 'nofire': {}}
        for key in ('fire', 'nofire'):
            for sp in ('train', 'val', 'test'):
                units = list(per_class[key][sp])
                if sp not in which or not units:
                    node_out[key][sp] = units
                    continue
                target = int(round(frac * n_images(units)))
                if grouped:
                    node_out[key][sp] = _subsample_images_within_units(units, target, rng)
                    continue
                rng.shuffle(units)
                taken, _ = take_units(units, target)
                if not taken:
                    taken = units[:1]  # keep >= 1 unit of a non-empty class
                node_out[key][sp] = taken
        out[node] = node_out
    # restore the original node ordering
    return {node: out[node] for node in tvt}




# --------------------------------------------------------------------------- #
# --verify: re-read what was just written and check it
# --------------------------------------------------------------------------- #
def _read_manifest_rows(output_dir: str, split_name: str) -> List[Dict[str, str]]:
    path = os.path.join(output_dir, split_name, 'manifest.csv')
    with open(path, newline='') as f:
        return [dict(row) for row in csv.DictReader(f)]


def _local_verify_split(split_name: str, rows: Sequence[Dict[str, str]],
                        stats: Optional[dict]) -> List[str]:
    """Minimal standalone check (used when ``scripts/verify_splits.py`` is absent).

    (1) no ``(group_id, label)`` unit spans two nodes or two of train/val/test,
    (1b) no ``group_id`` does so either, regardless of label (a cross-label
    group split by label is exactly what the leakage audit flags),
    (2) the manifest counts equal the per-node counts in ``split_stats.json``.
    """
    problems: List[str] = []
    if not rows:
        return ['{}: manifest.csv has no rows'.format(split_name)]

    nodes_of_unit: Dict[Tuple[str, str], set] = {}
    tvt_of_unit: Dict[Tuple[str, str], set] = {}
    nodes_of_group: Dict[str, set] = {}
    tvt_of_group: Dict[str, set] = {}
    counts: Dict[Tuple[str, str, str], int] = {}
    label_key = {'Fire': 'fire', 'No_Fire': 'nofire'}
    for row in rows:
        gid = row.get('group_id') or ''
        key = (row['node'], row['split'], label_key.get(row['label'], row['label']))
        counts[key] = counts.get(key, 0) + 1
        if gid and not str(gid).startswith('singleton'):
            unit = (gid, row['label'])
            nodes_of_unit.setdefault(unit, set()).add(row['node'])
            tvt_of_unit.setdefault(unit, set()).add(row['split'])
            nodes_of_group.setdefault(str(gid), set()).add(row['node'])
            tvt_of_group.setdefault(str(gid), set()).add(row['split'])

    for gid, nodes in sorted(nodes_of_group.items()):
        if len(nodes) > 1:
            problems.append('{}: group {} spans nodes {} (across labels)'.format(
                split_name, gid, '+'.join(sorted(nodes))))
    for gid, parts in sorted(tvt_of_group.items()):
        if len(parts) > 1:
            problems.append('{}: group {} spans {} (across labels)'.format(
                split_name, gid, '+'.join(sorted(parts))))

    for unit, nodes in sorted(nodes_of_unit.items()):
        if len(nodes) > 1:
            problems.append('{}: group {} (label {}) spans nodes {}'.format(
                split_name, unit[0], unit[1], '+'.join(sorted(nodes))))
    order = ('train', 'val', 'test')
    for unit, parts in sorted(tvt_of_unit.items()):
        if len(parts) > 1:
            ordered = sorted(parts, key=lambda v: (order.index(v) if v in order else len(order), v))
            problems.append('{}: group {} (label {}) spans {}'.format(
                split_name, unit[0], unit[1], '+'.join(ordered)))

    block = (stats or {}).get(split_name)
    if not isinstance(block, dict):
        problems.append('{}: no split_stats.json entry'.format(split_name))
        return problems
    for node, node_block in sorted(block.items()):
        if not isinstance(node_block, dict) or not all(k in node_block for k in ('train', 'val', 'test')):
            continue
        for sp in ('train', 'val', 'test'):
            entry = node_block.get(sp) or {}
            for key in ('fire', 'nofire'):
                expected = int(entry.get(key, 0) or 0)
                actual = counts.get((node, sp, key), 0)
                if expected != actual:
                    problems.append('{}: {}/{}/{}: manifest has {}, split_stats says {}'.format(
                        split_name, node, sp, key, actual, expected))
    return problems


def verify_written_splits(output_dir: str, split_names: Sequence[str],
                          quiet: bool = False) -> bool:
    """Re-read the manifests written by this run and assert they are leak-free.

    Reuses ``scripts/verify_splits.py`` (``check_manifest`` + ``check_stats``,
    which also cover duplicated rows, images shared by two nodes and basename
    collisions) when it can be imported, and falls back to the local minimal
    check otherwise.  Returns True when every split passes.
    """
    problems: List[str] = []
    verify_splits = None
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from scripts import verify_splits as _vs  # noqa: WPS433
        verify_splits = _vs
    except Exception:  # pragma: no cover - environment dependent
        verify_splits = None

    if verify_splits is not None:
        report = verify_splits.Report()
        stats = verify_splits.load_split_stats(output_dir)
        for split_name in split_names:
            path = verify_splits.manifest_path(output_dir, split_name)
            if not os.path.isfile(path):
                report.error('manifest', 'manifest.csv is missing', split_name)
                continue
            rows = verify_splits.read_manifest(path)
            verify_splits.check_manifest(split_name, rows, report)
            verify_splits.check_stats(split_name, rows, stats, report)
        problems = ['{}: [{}] {}'.format(i['split'] or '-', i['check'], i['message'])
                    for i in report.errors]
    else:
        stats_path = os.path.join(output_dir, 'split_stats.json')
        stats = None
        if os.path.isfile(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
        for split_name in split_names:
            if not os.path.isfile(os.path.join(output_dir, split_name, 'manifest.csv')):
                problems.append('{}: manifest.csv is missing'.format(split_name))
                continue
            problems.extend(_local_verify_split(
                split_name, _read_manifest_rows(output_dir, split_name), stats))

    if not quiet:
        print("\n--- verify ({} split(s)) ---".format(len(split_names)))
        for message in problems[:20]:
            print("  {}".format(message))
        if len(problems) > 20:
            print("  ... and {} more".format(len(problems) - 20))
        print("  VERIFY {}: no unit spans nodes or train/val/test and the manifest "
              "counts match split_stats.json".format('PASS' if not problems else 'FAIL'))
    return not problems


# --------------------------------------------------------------------------- #
# --export_manifests / --from_manifest: ship the partition, do not re-derive it
# --------------------------------------------------------------------------- #
MANIFEST_HEADER = ['node', 'split', 'label', 'path', 'group_id']


def _manifest_archive_files(manifest_dir: str) -> Dict[str, str]:
    """``{split_name: path}`` for every ``<split>.csv.gz`` / ``<split>.csv`` in ``manifest_dir``."""
    out: Dict[str, str] = {}
    if not os.path.isdir(manifest_dir):
        return out
    for name in sorted(os.listdir(manifest_dir)):
        if name.endswith('.csv.gz'):
            out.setdefault(name[:-len('.csv.gz')], os.path.join(manifest_dir, name))
        elif name.endswith('.csv'):
            out.setdefault(name[:-len('.csv')], os.path.join(manifest_dir, name))
    return out


def _read_manifest_text(path: str) -> str:
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', newline='') as f:
            return f.read()
    with open(path, 'rt', newline='') as f:
        return f.read()


def export_manifests(output_dir: str, manifest_dir: str, quiet: bool = False) -> List[str]:
    """Copy every ``<output_dir>/<split>/manifest.csv`` into ``manifest_dir`` as
    ``<split>.csv.gz``, together with ``split_stats.json``.

    This is the archival record of the partition used in the paper: it is small
    enough to track in git and replaying it needs no RNG.
    """
    os.makedirs(manifest_dir, exist_ok=True)
    written: List[str] = []
    for entry in sorted(os.listdir(output_dir)):
        src = os.path.join(output_dir, entry, 'manifest.csv')
        if not os.path.isfile(src):
            continue
        dst = os.path.join(manifest_dir, entry + '.csv.gz')
        with open(src, 'rb') as fin:
            payload = fin.read()
        # mtime=0 keeps the archive byte-stable across runs (no timestamp in the header)
        with gzip.GzipFile(dst, 'wb', compresslevel=9, mtime=0) as fout:
            fout.write(payload)
        written.append(dst)
    stats_src = os.path.join(output_dir, 'split_stats.json')
    if os.path.isfile(stats_src):
        shutil.copy2(stats_src, os.path.join(manifest_dir, 'split_stats.json'))
        written.append(os.path.join(manifest_dir, 'split_stats.json'))
    if not quiet:
        print("Exported {} manifest(s) + split_stats.json to {}".format(
            len(written) - 1, manifest_dir))
    return written


def replay_manifests(manifest_dir: str, data_dir: str, output_dir: str,
                     clean: bool = False, quiet: bool = False) -> dict:
    """Rebuild ``output_dir`` exactly as recorded in ``manifest_dir``.

    No shuffling, no seeds, no ``os.walk`` order: every image named by the
    manifest is placed where the manifest says.  Raises ``FileNotFoundError``
    when source images are missing (the dataset must be in place first).
    """
    archives = _manifest_archive_files(manifest_dir)
    if not archives:
        raise FileNotFoundError(
            "no <split>.csv[.gz] manifests in {} -- run --export_manifests first".format(
                manifest_dir))
    root_dir = os.path.abspath(data_dir)
    all_stats: Dict[str, dict] = {}
    committed_stats: Dict[str, dict] = {}
    stats_path = os.path.join(manifest_dir, 'split_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            committed_stats = json.load(f)

    missing: List[str] = []
    for split_name, archive in archives.items():
        rows = list(csv.DictReader(io.StringIO(_read_manifest_text(archive))))
        if not rows:
            raise ValueError("{}: manifest is empty".format(archive))
        prepare_split_dir(output_dir, split_name, clean)
        # (node, tvt, label) -> [absolute source paths]
        buckets: Dict[Tuple[str, str, str], List[str]] = {}
        for row in rows:
            src = os.path.join(root_dir, row['path'].replace('/', os.sep))
            if not os.path.isfile(src):
                if len(missing) < 10:
                    missing.append(src)
                continue
            buckets.setdefault((row['node'], row['split'], row['label']), []).append(src)
        if missing:
            continue
        stats: Dict[str, dict] = {}
        nodes = sorted({row['node'] for row in rows})
        for node in nodes:
            node_stats = {}
            for sp in ('train', 'val', 'test'):
                dest = os.path.join(output_dir, split_name, node, sp)
                fire_paths = buckets.get((node, sp, 'Fire'), [])
                nofire_paths = buckets.get((node, sp, 'No_Fire'), [])
                link_files(fire_paths, dest, 'Fire')
                link_files(nofire_paths, dest, 'No_Fire')
                node_stats[sp] = _counts(len(fire_paths), len(nofire_paths))
            stats[node] = node_stats
            total = sum(v['total'] for v in node_stats.values())
            tf = sum(v['fire'] for v in node_stats.values())
            if not quiet:
                print("  {} {}: {} imgs (Fire:{}, NoFire:{}, ratio:{:.1%})  "
                      "train/val/test={}/{}/{}".format(
                          split_name, node, total, tf, total - tf, tf / max(total, 1),
                          node_stats['train']['total'], node_stats['val']['total'],
                          node_stats['test']['total']))
        # the manifest is copied verbatim: it IS the record
        dst_manifest = os.path.join(output_dir, split_name, 'manifest.csv')
        os.makedirs(os.path.dirname(dst_manifest), exist_ok=True)
        with open(dst_manifest, 'w', newline='') as f:
            f.write(_read_manifest_text(archive))
        block = dict(committed_stats.get(split_name) or {})
        block.update(stats)
        all_stats[split_name] = block

    if missing:
        raise FileNotFoundError(
            "{} source image(s) named by the manifests are missing, e.g.\n  {}\n"
            "Put the dataset in {} first (see data/README.md).".format(
                len(missing), "\n  ".join(missing[:10]), root_dir))

    meta = dict(committed_stats.get('_meta') or {})
    meta.update({
        'data_dir': root_dir,
        'link_mode': _LINK_STATE['mode'],
        'clean': bool(clean),
        'replayed_from': os.path.abspath(manifest_dir),
        'verify_ok': None,
    })
    all_stats['_meta'] = meta
    return all_stats


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    """Run the splitter.  ``args`` is an argparse Namespace (or anything with the
    same attributes); a plain dict is accepted too."""
    if isinstance(args, dict):
        args = argparse.Namespace(**dict(_defaults(), **args))

    set_link_mode(getattr(args, 'link_mode', 'symlink'))
    node_names = NODE_NAMES[:args.nodes]

    export_dir = getattr(args, 'export_manifests', None)
    if export_dir:
        export_manifests(args.output_dir, export_dir)
        return {'_meta': {'exported_to': os.path.abspath(export_dir)}}

    manifest_dir = getattr(args, 'from_manifest', None)
    if manifest_dir:
        print("=" * 50)
        print("  FedRGBD Data Splitter -- replaying {}".format(manifest_dir))
        print("=" * 50)
        all_stats = replay_manifests(manifest_dir, args.data_dir, args.output_dir,
                                     clean=bool(getattr(args, 'clean', False)))
        os.makedirs(args.output_dir, exist_ok=True)
        stats_path = os.path.join(args.output_dir, 'split_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        if getattr(args, 'verify', False):
            ok = verify_written_splits(
                args.output_dir, [k for k in all_stats if k != '_meta'])
            all_stats['_meta']['verify_ok'] = ok
            with open(stats_path, 'w') as f:
                json.dump(all_stats, f, indent=2)
        print("\nStats saved to {}/split_stats.json".format(args.output_dir))
        return all_stats

    print("=" * 50)
    print("  FedRGBD Data Splitter ({} nodes)".format(args.nodes))
    print("=" * 50)

    fire, nofire = find_images(args.data_dir)
    print("\nFire: {}, NoFire: {}, Total: {}".format(len(fire), len(nofire), len(fire) + len(nofire)))

    if not fire or not nofire:
        print("ERROR: No images found!")
        return {}

    group_map = None
    if getattr(args, 'group_file', None):
        group_map = load_group_file(args.group_file, args.data_dir)
        print("Group file: {} ({} grouped paths)".format(args.group_file, len(group_map)))

    fire_units, nofire_units, gid_of, ginfo = build_units(fire, nofire, args.data_dir, group_map)
    if group_map is not None:
        print("Matched {}/{} images to {} groups ({} units, largest {} images, "
              "{} cross-label groups)".format(
                  ginfo['n_matched'], len(fire) + len(nofire), ginfo['n_groups_used'],
                  ginfo['n_units'], ginfo['largest_unit'], ginfo['cross_label_groups']))
        warn_group_coverage(ginfo, args.group_file)

    grouped = is_grouped(fire_units, nofire_units)
    if grouped:
        print("Group mode: oversized units -> largest-first greedy assignment of whole "
              "groups (cross-label groups bundled) to the image quotas (nodes, "
              "train/val/test, Dirichlet); subsampling at image level inside units")

    random.seed(args.seed)
    all_stats: Dict[str, dict] = {}
    base_tvt: Dict[str, dict] = {}

    clean = bool(getattr(args, 'clean', False))

    if not getattr(args, 'skip_base_splits', False):
        # --- IID: equal random split ------------------------------------- #
        print("\n--- IID Split ({} nodes) ---".format(args.nodes))
        rf, rn = list(fire_units), list(nofire_units)
        random.shuffle(rf)
        random.shuffle(rn)
        iid_nodes = partition_iid(rf, rn, node_names, grouped=grouped, seed=args.seed,
                                  gid_of=gid_of)
        all_stats['iid'], base_tvt['iid'] = create_split(
            args.output_dir, 'iid', iid_nodes, args.seed, args.data_dir, gid_of, clean=clean,
            grouped=grouped)

        # --- Non-IID label skew ------------------------------------------ #
        print("\n--- Non-IID Label Skew ({} nodes) ---".format(args.nodes))
        random.shuffle(rf)
        random.shuffle(rn)
        noniid_nodes = partition_non_iid_label(rf, rn, node_names, grouped=grouped,
                                               seed=args.seed, gid_of=gid_of)
        all_stats['non_iid_label'], base_tvt['non_iid_label'] = create_split(
            args.output_dir, 'non_iid_label', noniid_nodes, args.seed, args.data_dir, gid_of,
            clean=clean, grouped=grouped)

    # --- Dirichlet label skew -------------------------------------------- #
    for alpha in (getattr(args, 'dirichlet_alpha', None) or []):
        name = 'dirichlet_{:g}'.format(alpha)
        print("\n--- Dirichlet Split alpha={:g} ({} nodes) ---".format(alpha, args.nodes))
        nodes_units, proportions = partition_dirichlet(
            fire_units, nofire_units, node_names, alpha, args.seed,
            min_size=getattr(args, 'dirichlet_min_size', 10), grouped=grouped, gid_of=gid_of)
        extra = {'dirichlet_alpha': float(alpha),
                 'dirichlet_proportions': proportions,
                 'seed': int(args.seed)}
        all_stats[name], base_tvt[name] = create_split(
            args.output_dir, name, nodes_units, args.seed, args.data_dir, gid_of, extra,
            clean=clean, grouped=grouped)

    # --- Subsampling ------------------------------------------------------ #
    fracs = getattr(args, 'subsample_frac', None) or []
    which = list(getattr(args, 'subsample_splits', None) or ['train'])
    for base_name in list(base_tvt):
        for frac in fracs:
            name = '{}_sub{:g}'.format(base_name, frac)
            print("\n--- Subsample {} (frac={:g}, splits={}) ---".format(
                name, frac, ','.join(which)))
            sub = subsample_tvt(base_tvt[base_name], frac, which, args.seed, base_name,
                                grouped=grouped)
            extra = {'subsample_frac': float(frac), 'subsample_splits': which,
                     'source_split': base_name, 'seed': int(args.seed)}
            if 'dirichlet_alpha' in all_stats.get(base_name, {}):
                extra['dirichlet_alpha'] = all_stats[base_name]['dirichlet_alpha']
                extra['dirichlet_proportions'] = all_stats[base_name]['dirichlet_proportions']
            all_stats[name] = materialize(
                args.output_dir, name, sub, args.data_dir, gid_of, extra, clean=clean)

    def _write_stats(stats: dict) -> str:
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, 'split_stats.json')
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)
        return path

    verify_ok = None
    if getattr(args, 'verify', False):
        # The verifier compares the manifests against split_stats.json ON DISK, so the
        # fresh stats must be written first -- otherwise a stale file from a previous
        # partition (the normal --clean re-split situation) is what gets compared.
        _write_stats(dict(all_stats, _meta={'verify_ok': None, 'partial': True}))
        verify_ok = verify_written_splits(args.output_dir, [k for k in all_stats])

    all_stats['_meta'] = {
        'seed': int(args.seed),
        'nodes': int(args.nodes),
        'node_names': node_names,
        'data_dir': os.path.abspath(args.data_dir),
        'n_fire': len(fire),
        'n_nofire': len(nofire),
        'link_mode': _LINK_STATE['mode'],
        'group_file': getattr(args, 'group_file', None),
        'grouped_images_matched': ginfo['n_matched'],
        'n_groups_used': ginfo['n_groups_used'],
        'cross_label_groups': ginfo['cross_label_groups'],
        'largest_unit': ginfo['largest_unit'],
        'n_units': ginfo['n_units'],
        'group_paths_total': ginfo['group_paths_total'],
        'group_paths_matched': ginfo['group_paths_matched'],
        'clean': clean,
        'grouped': grouped,
        'assignment': 'greedy_largest_first' if grouped else 'sequential_cut',
        'subsample_level': 'image_within_units' if grouped else 'unit',
        'verify_ok': verify_ok,
    }

    _write_stats(all_stats)
    print("\nStats saved to {}/split_stats.json".format(args.output_dir))
    return all_stats


def _defaults() -> dict:  # noqa: D401
    return {'data_dir': 'data/raw/flame_dataset', 'output_dir': 'data/processed',
            'seed': 42, 'nodes': 3, 'group_file': None, 'dirichlet_alpha': None,
            'dirichlet_min_size': 10, 'skip_base_splits': False,
            'subsample_frac': None, 'subsample_splits': ['train'], 'link_mode': 'symlink',
            'clean': False, 'verify': False,
            'export_manifests': None, 'from_manifest': None}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data/raw/flame_dataset")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nodes", type=int, default=3, choices=[2, 3])
    parser.add_argument("--group_file", default=None,
                        help="groups.json/groups.csv from scripts/analyze_flame_leakage.py; "
                             "keeps near-duplicate groups on one node and in one split")
    parser.add_argument("--dirichlet_alpha", type=float, nargs='+', default=None,
                        help="one or more Dirichlet concentrations -> splits dirichlet_<alpha>")
    parser.add_argument("--dirichlet_min_size", type=int, default=10,
                        help="minimum images per node for a Dirichlet draw to be accepted")
    parser.add_argument("--skip_base_splits", action="store_true",
                        help="do not regenerate iid / non_iid_label")
    parser.add_argument("--subsample_frac", type=float, nargs='+', default=None,
                        help="one or more fractions -> <split>_sub<frac> variants")
    parser.add_argument("--subsample_splits", nargs='+', default=['train'],
                        choices=['train', 'val', 'test'],
                        help="which splits the subsampling reduces (default: train only)")
    parser.add_argument("--link_mode", default="symlink", choices=["symlink", "hardlink", "copy"],
                        help="how images are placed; symlink falls back to copy if unsupported")
    parser.add_argument("--clean", action="store_true",
                        help="delete each <output_dir>/<split> tree before rewriting it; "
                             "REQUIRED when re-partitioning into an existing data/processed "
                             "(otherwise files from the previous partition stay behind and "
                             "leak train images into val/test)")
    parser.add_argument("--export_manifests", default=None, metavar="DIR",
                        help="write <split>.csv.gz + split_stats.json of an existing "
                             "--output_dir tree into DIR (the archival record of the "
                             "partition; ~1.1 MB for the 15 FLAME splits) and exit")
    parser.add_argument("--from_manifest", default=None, metavar="DIR",
                        help="rebuild --output_dir exactly as recorded in DIR instead of "
                             "re-deriving the partition; uses no RNG and does not depend on "
                             "os.walk order, so every node gets a bit-identical tree")
    parser.add_argument("--verify", action="store_true",
                        help="after writing, re-read the manifests and assert that no "
                             "(group_id,label) unit spans nodes or train/val/test and that the "
                             "manifest counts equal split_stats.json; prints PASS/FAIL and "
                             "exits non-zero on FAIL (see scripts/verify_splits.py for the "
                             "full stand-alone check)")
    return parser


def main(argv: Optional[Sequence[str]] = None):
    stats = run(build_parser().parse_args(argv))
    if isinstance(stats, dict) and stats.get('_meta', {}).get('verify_ok') is False:
        raise SystemExit(1)
    return stats


if __name__ == "__main__":
    main()
