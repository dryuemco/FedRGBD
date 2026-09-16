"""FedRGBD Data Splitter -- IID / Non-IID / Dirichlet partitioning for 2 or 3 FL nodes.

The splitter walks ``--data_dir`` for ``Fire`` / ``No_Fire`` images, partitions
them across ``--nodes`` federated clients and writes a
``<output_dir>/<split>/<node>/{train,val,test}/{Fire,No_Fire}`` tree of links
(or copies) plus ``<output_dir>/split_stats.json`` and a per-split
``manifest.csv``.

Three features were added on top of the original image-level splitter.  All of
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
   are built *per class*, so a group that spans both labels becomes one Fire
   unit and one No_Fire unit (counted as ``cross_label_groups``).  Because
   ``random.shuffle`` on a list of N size-1 units draws exactly the same
   numbers as ``random.shuffle`` on a list of N paths, and because
   :func:`take_units` reduces to plain slicing when every unit has size 1, the
   default (no group file) code path is unchanged.

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
   Sampling is stratified per class, done at unit level, seeded, and always
   keeps at least one unit of a non-empty class.

Usage
-----
    python3 src/data/data_splitter.py --data_dir data/raw/flame_dataset \\
        --output_dir data/processed --nodes 3 --seed 42

    python3 src/data/data_splitter.py --data_dir data/raw/flame_dataset \\
        --output_dir data/processed --nodes 3 \\
        --group_file analysis/leakage/groups.json \\
        --dirichlet_alpha 0.1 0.5 1.0 --subsample_frac 0.05 0.01
"""

from __future__ import annotations

import argparse
import csv
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
def _local_load_group_file(path: str) -> Dict[str, int]:
    """Standalone fallback copy of ``analyze_flame_leakage.load_group_file``.

    Reads ``groups.json`` (``{"groups": {"gid": ["Fire/a.jpg", ...]}}``) or
    ``groups.csv`` (``path,group_id`` columns) into ``{relative_path: group_id}``.
    """
    mapping: Dict[str, int] = {}
    if path.lower().endswith('.json'):
        with open(path) as f:
            data = json.load(f)
        groups = data['groups'] if 'groups' in data else data
        for gid, members in groups.items():
            for rel in members:
                mapping[rel.replace('\\', '/')] = int(gid)
    else:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                mapping[row['path'].replace('\\', '/')] = int(row['group_id'])
    return mapping


def load_group_file(path: str) -> Dict[str, int]:
    """Use the canonical parser from ``scripts/``; fall back to the local copy.

    The fallback keeps the splitter runnable standalone (e.g. on the Jetson,
    where ``scripts/`` or its numpy/PIL imports may not be available).
    """
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from scripts.analyze_flame_leakage import load_group_file as _impl  # noqa: WPS433
        return _impl(path)
    except Exception:  # pragma: no cover - environment dependent
        return _local_load_group_file(path)


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
def split_units(units: Sequence[Unit], train=0.7, val=0.15, seed=42) -> Dict[str, List[Unit]]:
    """70/15/15 split of a list of units -- reseeds ``random`` exactly like the original."""
    random.seed(seed)
    s = list(units)
    random.shuffle(s)
    n = n_images(s)
    t, v = int(n * train), int(n * (train + val))
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
def partition_to_tvt(node_units: Dict[str, Dict[str, List[Unit]]], seed: int):
    """Per-node 70/15/15 split.

    The ``random`` calls happen in exactly the original order (node by node,
    fire then nofire) so the global RNG state evolves as it did before.
    """
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
            print("  {}: {} imgs (Fire:{}, NoFire:{}, ratio:{:.1%})".format(
                node, total, tf, total - tf, tf / max(total, 1)))
    stats['class_counts'] = class_counts_table(tvt)
    if extra:
        stats.update(extra)
    write_manifest(output_dir, split_name, tvt, data_dir, gid_of)
    return stats


def create_split(output_dir, split_name, node_data, seed, data_dir='.', gid_of=None,
                 extra=None, quiet=False, clean=False):
    """Split each node 70/15/15 and materialise it (original entry point)."""
    tvt = partition_to_tvt(node_data, seed)
    return materialize(output_dir, split_name, tvt, data_dir, gid_of or {}, extra, quiet,
                       clean), tvt


# --------------------------------------------------------------------------- #
# partitioners
# --------------------------------------------------------------------------- #
def partition_iid(fire_units, nofire_units, node_names):
    """Equal random split -- caller must have shuffled the unit lists already."""
    fp = split_units_into(fire_units, len(node_names))
    np_ = split_units_into(nofire_units, len(node_names))
    return {name: {'fire': f, 'nofire': n} for name, f, n in zip(node_names, fp, np_)}


def partition_non_iid_label(fire_units, nofire_units, node_names):
    """Original label-skew heuristic, expressed with unit-wise takes.

    2 nodes: A is 70% fire.  3 nodes: A=80% fire, B balanced, C=80% no-fire.
    The arithmetic is exactly the original one; ``take_units`` replaces the
    image-index slicing and is equivalent when units have size 1.
    """
    n_nodes = len(node_names)
    n_fire, n_nofire = n_images(fire_units), n_images(nofire_units)
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


def _assign_by_proportions(units: Sequence[Unit], p: np.ndarray) -> List[List[Unit]]:
    """Hand units to nodes at the cumulative-image-count boundaries ``cumsum(p)*N``."""
    total = n_images(units)
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
                        min_size=10, max_tries=1000):
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
    for _ in range(max_tries):
        props = {}
        parts = {}
        for key, units in classes:
            p = rng.dirichlet(np.repeat(float(alpha), n_nodes))
            props[key] = p
            parts[key] = _assign_by_proportions(units, p)
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


def subsample_tvt(tvt, frac: float, which: Sequence[str], seed: int, split_name: str):
    """Reduce the named splits to ``frac`` of their units, stratified per class."""
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
                rng.shuffle(units)
                target = int(round(frac * n_images(units)))
                taken, _ = take_units(units, target)
                if not taken:
                    taken = units[:1]  # keep >= 1 unit of a non-empty class
                node_out[key][sp] = taken
        out[node] = node_out
    # restore the original node ordering
    return {node: out[node] for node in tvt}


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
        group_map = load_group_file(args.group_file)
        print("Group file: {} ({} grouped paths)".format(args.group_file, len(group_map)))

    fire_units, nofire_units, gid_of, ginfo = build_units(fire, nofire, args.data_dir, group_map)
    if group_map is not None:
        print("Matched {}/{} images to {} groups ({} units, largest {} images, "
              "{} cross-label groups)".format(
                  ginfo['n_matched'], len(fire) + len(nofire), ginfo['n_groups_used'],
                  ginfo['n_units'], ginfo['largest_unit'], ginfo['cross_label_groups']))
        warn_group_coverage(ginfo, args.group_file)

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
        iid_nodes = partition_iid(rf, rn, node_names)
        all_stats['iid'], base_tvt['iid'] = create_split(
            args.output_dir, 'iid', iid_nodes, args.seed, args.data_dir, gid_of, clean=clean)

        # --- Non-IID label skew ------------------------------------------ #
        print("\n--- Non-IID Label Skew ({} nodes) ---".format(args.nodes))
        random.shuffle(rf)
        random.shuffle(rn)
        noniid_nodes = partition_non_iid_label(rf, rn, node_names)
        all_stats['non_iid_label'], base_tvt['non_iid_label'] = create_split(
            args.output_dir, 'non_iid_label', noniid_nodes, args.seed, args.data_dir, gid_of,
            clean=clean)

    # --- Dirichlet label skew -------------------------------------------- #
    for alpha in (getattr(args, 'dirichlet_alpha', None) or []):
        name = 'dirichlet_{:g}'.format(alpha)
        print("\n--- Dirichlet Split alpha={:g} ({} nodes) ---".format(alpha, args.nodes))
        nodes_units, proportions = partition_dirichlet(
            fire_units, nofire_units, node_names, alpha, args.seed,
            min_size=getattr(args, 'dirichlet_min_size', 10))
        extra = {'dirichlet_alpha': float(alpha),
                 'dirichlet_proportions': proportions,
                 'seed': int(args.seed)}
        all_stats[name], base_tvt[name] = create_split(
            args.output_dir, name, nodes_units, args.seed, args.data_dir, gid_of, extra,
            clean=clean)

    # --- Subsampling ------------------------------------------------------ #
    fracs = getattr(args, 'subsample_frac', None) or []
    which = list(getattr(args, 'subsample_splits', None) or ['train'])
    for base_name in list(base_tvt):
        for frac in fracs:
            name = '{}_sub{:g}'.format(base_name, frac)
            print("\n--- Subsample {} (frac={:g}, splits={}) ---".format(
                name, frac, ','.join(which)))
            sub = subsample_tvt(base_tvt[base_name], frac, which, args.seed, base_name)
            extra = {'subsample_frac': float(frac), 'subsample_splits': which,
                     'source_split': base_name, 'seed': int(args.seed)}
            if 'dirichlet_alpha' in all_stats.get(base_name, {}):
                extra['dirichlet_alpha'] = all_stats[base_name]['dirichlet_alpha']
                extra['dirichlet_proportions'] = all_stats[base_name]['dirichlet_proportions']
            all_stats[name] = materialize(
                args.output_dir, name, sub, args.data_dir, gid_of, extra, clean=clean)

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
    }

    os.makedirs(args.output_dir, exist_ok=True)
    stats_path = os.path.join(args.output_dir, 'split_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print("\nStats saved to {}/split_stats.json".format(args.output_dir))
    return all_stats


def _defaults() -> dict:
    return {'data_dir': 'data/raw/flame_dataset', 'output_dir': 'data/processed',
            'seed': 42, 'nodes': 3, 'group_file': None, 'dirichlet_alpha': None,
            'dirichlet_min_size': 10, 'skip_base_splits': False,
            'subsample_frac': None, 'subsample_splits': ['train'], 'link_mode': 'symlink',
            'clean': False}


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
    return parser


def main(argv: Optional[Sequence[str]] = None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
