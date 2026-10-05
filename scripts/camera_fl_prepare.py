#!/usr/bin/env python3
"""FedRGBD -- camera experiment, question (b): materialise the five federated scene folds.

``docs/CAMERA_EXPERIMENT_PREREG.md`` section 6.  For fold f in 0..4:

    data/processed/camera_fold<f>/node_{a,b,c}/{train,val,test}/{Fire,No_Fire}/<id>.png

* test scenes = fold f, validation scenes = fold (f + 1) mod 5, training scenes = the
  other three folds -- the same scenes on every client (``data/splits_camera/fl_folds.csv``);
* each node gets ONLY its own sensor's valid frames (node_a D435if, node_b D435i,
  node_c ZED 2i; ``valid == 1`` in ``data/raw/camera/labels.csv``);
* every image is the RGB file preprocessed ONCE on the desktop
  (``scripts/camera_preprocess_frames.py``: ``camera_preprocess.rgb_224``, shorter side 224,
  centre crop), copied byte for byte and md5-checked against
  ``data/splits_camera/preprocessed_manifest.csv`` (prereg Amendment 2): no node decodes or
  resizes a native frame, so the nodes and the desktop baselines train on identical files.
  ``FlameDataset``'s own 224 x 224 resize of a 224 x 224 image is a no-op.  A node needs
  only its own camera's preprocessed files (``data/processed/camera_224/<node>/``).

Everything is derived from ``labels.csv`` and ``fl_folds.csv`` alone, sorted by
(node, split, class, id) -- never from directory-walk order (CLAUDE.md rule 2).  Before
anything is written, ``fl_folds.csv`` is recomputed from the kept scene ids with the
pre-registered generator (``scripts/camera_manifests.py``); a difference is refused.

Outputs besides the images:

* ``data/processed/camera_fold<f>/manifest.csv`` -- ``node,split,class,id,scene``, one row
  per image of all three nodes, LF line endings: identical bytes on every machine that
  has the same labels / folds, whichever nodes it materialises (``--nodes``), so its md5
  can be compared across the testbed like the FLAME split manifests;
* ``data/splits_camera/fl_materialised_manifest.csv`` -- per fold / node / split / class:
  image and scene counts, and the md5 of the fold's manifest.csv.

Idempotent: existing images are kept; a fold directory holding a file that is not in
its manifest is refused (stale partition, it would leak) unless ``--clean`` rebuilds it.
``--verify`` recomputes everything and checks manifest bytes, counts and the sorted file
list on disk, and that every image is byte for byte (md5) the file preprocessed on the
desktop; it writes nothing.

    python scripts/camera_fl_prepare.py --clean            # build all five folds
    python scripts/camera_fl_prepare.py --verify           # check, exit 1 on any problem
    python scripts/camera_fl_prepare.py --nodes node_b     # a node holding only its own frames
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import shutil
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.data.camera_preprocess import PREPROCESSED_DIR, PREPROCESSED_MANIFEST  # noqa: E402

NODES = ("node_a", "node_b", "node_c")
SENSORS = {"node_a": "D435if", "node_b": "D435i", "node_c": "ZED 2i"}
SPLITS = ("train", "val", "test")
CLASS_DIRS = {"fire": "Fire", "no_fire": "No_Fire"}
N_FOLDS = 5
IMG_SIZE = 224

DEFAULT_LABELS = os.path.join("data", "raw", "camera", "labels.csv")
DEFAULT_FOLDS = os.path.join("data", "splits_camera", "fl_folds.csv")
DEFAULT_OUT = os.path.join("data", "processed")
DEFAULT_COUNTS = os.path.join("data", "splits_camera", "fl_materialised_manifest.csv")

MANIFEST_COLUMNS = ("node", "split", "class", "id", "scene")
COUNT_COLUMNS = ("fold", "node", "split", "class", "n_images", "n_scenes", "fold_manifest_md5")


# --------------------------------------------------------------------------- inputs
def fold_dir_name(fold: int) -> str:
    return "camera_fold%d" % int(fold)


def read_folds(path: str) -> Dict[str, int]:
    if not os.path.isfile(path):
        raise SystemExit("%s is missing -- write it with scripts/camera_manifests.py first" % path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    folds = {}
    for r in rows:
        scene, fold = r["scene"].strip(), int(r["fold"])
        if scene in folds:
            raise ValueError("%s: scene %s listed twice" % (path, scene))
        if not 0 <= fold < N_FOLDS:
            raise ValueError("%s: scene %s in fold %d (folds are 0-%d)"
                             % (path, scene, fold, N_FOLDS - 1))
        folds[scene] = fold
    if not folds:
        raise ValueError("%s lists no scene" % path)
    return folds


def read_labels(path: str) -> List[Dict[str, object]]:
    if not os.path.isfile(path):
        raise SystemExit("%s is missing -- write it with scripts/camera_labels.py first" % path)
    from scripts.camera_labels import read_labels as _read
    return _read(path)


def check_folds_are_the_preregistered_ones(labels: Sequence[Dict[str, object]],
                                           folds: Dict[str, int]) -> None:
    """fl_folds.csv must be exactly what the pre-registered generator gives for the kept
    scene ids of labels.csv (prereg sec. 6: sorted ids, rng 20260928, round-robin)."""
    from scripts.camera_labels import kept_scenes
    from scripts.camera_manifests import fl_folds

    kept = kept_scenes(labels)
    want = fl_folds(kept)
    if want != folds:
        extra = sorted(set(folds) - set(want))
        missing = sorted(set(want) - set(folds))
        moved = sorted(s for s in set(want) & set(folds) if want[s] != folds[s])
        raise SystemExit("fl_folds.csv is not the pre-registered fold assignment of the kept "
                         "scenes of labels.csv (not kept: %s; missing: %s; other fold: %s) -- "
                         "regenerate it with scripts/camera_manifests.py"
                         % (extra or "-", missing or "-", moved or "-"))


def split_of_scene(scene_fold: int, fold: int) -> str:
    if scene_fold == fold:
        return "test"
    if scene_fold == (fold + 1) % N_FOLDS:
        return "val"
    return "train"


# --------------------------------------------------------------------------- plan
def plan_fold(labels: Sequence[Dict[str, object]], folds: Dict[str, int],
              fold: int) -> List[Dict[str, str]]:
    """Every image of fold ``fold``, all nodes: [{node, split, class, id, scene}], sorted.

    Only valid frames; each frame goes to its own node (the node column of labels.csv is
    the camera that recorded it).  Raises on anything the exclusion rules should have
    removed: a valid frame of a scene without a fold, or a scene missing a class on a
    node (prereg sec. 2 drops such a scene from the study).
    """
    entries = []
    seen = set()
    per_scene = defaultdict(set)                      # scene -> {(node, label)}
    for r in labels:
        if not int(r["valid"]):
            continue
        node, scene, label, fid = str(r["node"]), str(r["scene"]), str(r["label"]), str(r["id"])
        if node not in NODES:
            raise ValueError("labels.csv: frame %s on unknown node %r" % (fid, node))
        if label not in CLASS_DIRS:
            raise ValueError("labels.csv: frame %s has label %r (fire|no_fire)" % (fid, label))
        if scene not in folds:
            raise ValueError("labels.csv: valid frame %s/%s of scene %s, which has no fold in "
                             "fl_folds.csv" % (node, fid, scene))
        if (node, fid) in seen:
            raise ValueError("labels.csv: frame %s listed twice on %s" % (fid, node))
        seen.add((node, fid))
        per_scene[scene].add((node, label))
        entries.append({"node": node, "split": split_of_scene(folds[scene], fold),
                        "class": CLASS_DIRS[label], "id": fid, "scene": scene})
    for scene in sorted(per_scene):
        lacking = [(n, c) for n in NODES for c in CLASS_DIRS if (n, c) not in per_scene[scene]]
        if lacking:
            raise ValueError("scene %s has no valid %s frame on %s; the pre-registered exclusion "
                             "rule drops such a scene -- labels.csv is inconsistent"
                             % (scene, lacking[0][1], lacking[0][0]))
    order = {s: i for i, s in enumerate(SPLITS)}
    entries.sort(key=lambda e: (e["node"], order[e["split"]], e["class"], e["id"]))
    return entries


def manifest_bytes(entries: Sequence[Dict[str, str]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(MANIFEST_COLUMNS)
    for e in entries:
        w.writerow([e[c] for c in MANIFEST_COLUMNS])
    return buf.getvalue().encode("utf-8")


def count_rows(fold: int, entries: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    md5 = hashlib.md5(manifest_bytes(entries)).hexdigest()
    n = defaultdict(int)
    scenes = defaultdict(set)
    for e in entries:
        key = (e["node"], e["split"], e["class"])
        n[key] += 1
        scenes[key].add(e["scene"])
    rows = []
    for node in NODES:
        for split in SPLITS:
            for cls in ("Fire", "No_Fire"):
                key = (node, split, cls)
                rows.append({"fold": fold, "node": node, "split": split, "class": cls,
                             "n_images": n[key], "n_scenes": len(scenes[key]),
                             "fold_manifest_md5": md5})
    return rows


def counts_bytes(rows: Sequence[Dict[str, object]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COUNT_COLUMNS)
    for r in rows:
        w.writerow([r[c] for c in COUNT_COLUMNS])
    return buf.getvalue().encode("utf-8")


def rel_path(e: Dict[str, str]) -> str:
    """Path of an image below the fold directory, forward slashes."""
    return "%s/%s/%s/%s.png" % (e["node"], e["split"], e["class"], e["id"])


def files_on_disk(fold_dir: str, node: str) -> List[str]:
    """Sorted relative paths of every file below ``fold_dir/node`` (forward slashes)."""
    out = []
    root = os.path.join(fold_dir, node)
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            out.append(os.path.relpath(os.path.join(dirpath, name), fold_dir).replace("\\", "/"))
    return sorted(out)


# --------------------------------------------------------------------------- write
def _write_bytes(path: str, blob: bytes) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, path)


def copy_preprocessed(store, e: Dict[str, str], dst: str) -> None:
    """The frame's preprocessed RGB file, byte for byte; md5-checked before and after."""
    from src.data.camera_preprocess import md5_of

    blob = store.read_bytes(e["node"], e["id"], "rgb")          # raises on an md5 mismatch
    _write_bytes(dst, blob)
    if md5_of(dst) != store.rows[(e["node"], e["id"], "rgb")]["md5"]:
        raise SystemExit("%s: written copy differs from the preprocessed file" % dst)


def materialise_fold(fold: int, entries: Sequence[Dict[str, str]], store,
                     out_root: str, nodes: Sequence[str], clean: bool) -> Dict[str, int]:
    """Write one fold's images of ``nodes`` and its manifest.csv -> {node: images written}."""
    fold_dir = os.path.join(out_root, fold_dir_name(fold))
    wanted = {n: {rel_path(e): e for e in entries if e["node"] == n} for n in nodes}
    if clean:
        for n in nodes:
            shutil.rmtree(os.path.join(fold_dir, n), ignore_errors=True)
    else:
        for n in nodes:
            stale = [p for p in files_on_disk(fold_dir, n) if p not in wanted[n]]
            if stale:
                raise SystemExit("%s holds %d file(s) not in the fold's manifest (first: %s) -- "
                                 "a stale partition would leak; rerun with --clean"
                                 % (fold_dir, len(stale), stale[0]))
    missing_src = []
    for n in nodes:
        for rel, e in wanted[n].items():
            if not store.has(n, e["id"], "rgb"):
                missing_src.append("%s/%s (not in %s)" % (n, e["id"], store.manifest_path))
            elif not os.path.isfile(store.path(n, e["id"], "rgb")):
                missing_src.append(store.path(n, e["id"], "rgb"))
    if missing_src:
        raise SystemExit("%d preprocessed image(s) missing (first: %s)"
                         % (len(missing_src), missing_src[0]))
    written = {}
    for n in nodes:
        k = 0
        for rel, e in sorted(wanted[n].items()):
            dst = os.path.join(fold_dir, *rel.split("/"))
            if os.path.isfile(dst):
                continue
            copy_preprocessed(store, e, dst)
            k += 1
        written[n] = k
    _write_bytes(os.path.join(fold_dir, "manifest.csv"), manifest_bytes(entries))
    return written


# --------------------------------------------------------------------------- verify
def verify_fold(fold: int, entries: Sequence[Dict[str, str]], out_root: str,
                nodes: Sequence[str], store=None) -> List[str]:
    """Problems of one materialised fold (empty list = OK)."""
    problems = []
    fold_dir = os.path.join(out_root, fold_dir_name(fold))
    path = os.path.join(fold_dir, "manifest.csv")
    if not os.path.isfile(path):
        return ["%s missing" % path]
    with open(path, "rb") as f:
        if f.read() != manifest_bytes(entries):
            problems.append("%s differs from the recomputed manifest" % path)
    for n in nodes:
        want = sorted(rel_path(e) for e in entries if e["node"] == n)
        got = files_on_disk(fold_dir, n)
        if got != want:
            extra, missing = sorted(set(got) - set(want)), sorted(set(want) - set(got))
            problems.append("fold %d %s: %d file(s) missing, %d not in the manifest (first: %s)"
                            % (fold, n, len(missing), len(extra), (missing or extra)[0]))
            continue
        if store is not None:
            # every image must be, byte for byte, the file preprocessed on the desktop
            from src.data.camera_preprocess import md5_of
            by_rel = {rel_path(e): e for e in entries if e["node"] == n}
            for rel in got:
                e = by_rel[rel]
                row = store.rows.get((n, e["id"], "rgb"))
                if row is None:
                    problems.append("%s: %s/%s not in %s" % (rel, n, e["id"], store.manifest_path))
                elif md5_of(os.path.join(fold_dir, *rel.split("/"))) != row["md5"]:
                    problems.append("%s: not the file preprocessed on the desktop (md5)" % rel)
    return problems


# --------------------------------------------------------------------------- main
def plan_all(labels_csv: str, folds_csv: str, folds_to_do: Sequence[int]
             ) -> Tuple[Dict[int, List[Dict[str, str]]], List[Dict[str, object]]]:
    labels = read_labels(labels_csv)
    folds = read_folds(folds_csv)
    check_folds_are_the_preregistered_ones(labels, folds)
    plans = {f: plan_fold(labels, folds, f) for f in range(N_FOLDS)}
    counts = [row for f in range(N_FOLDS) for row in count_rows(f, plans[f])]
    return {f: plans[f] for f in folds_to_do}, counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels_csv", default=DEFAULT_LABELS)
    ap.add_argument("--folds_csv", default=DEFAULT_FOLDS)
    ap.add_argument("--preprocessed_dir", default=PREPROCESSED_DIR,
                    help="<dir>/<node>/<id>_rgb.png, written once on the desktop by "
                         "scripts/camera_preprocess_frames.py")
    ap.add_argument("--preprocessed_manifest", default=PREPROCESSED_MANIFEST)
    ap.add_argument("--output_root", default=DEFAULT_OUT,
                    help="folds go to <output_root>/camera_fold<f>/")
    ap.add_argument("--counts_csv", default=DEFAULT_COUNTS)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    ap.add_argument("--nodes", nargs="+", default=list(NODES), choices=NODES,
                    help="nodes whose images to write / check (a Jetson holds only its own)")
    ap.add_argument("--clean", action="store_true",
                    help="delete the nodes' fold directories first (rebuild)")
    ap.add_argument("--verify", action="store_true",
                    help="recompute and check manifests, counts and files; write nothing")
    args = ap.parse_args(argv)
    from scripts.camera_labels import refuse_pilot
    refuse_pilot(args.labels_csv, args.folds_csv, args.preprocessed_dir, args.preprocessed_manifest)
    if args.clean and args.verify:
        ap.error("--clean and --verify are exclusive")
    bad = [f for f in args.folds if not 0 <= f < N_FOLDS]
    if bad:
        ap.error("folds are 0-%d" % (N_FOLDS - 1))

    plans, counts = plan_all(args.labels_csv, args.folds_csv, sorted(set(args.folds)))
    nodes = [n for n in NODES if n in args.nodes]
    from src.data.camera_preprocess import PreprocessedStore
    try:
        store = PreprocessedStore(args.preprocessed_dir, args.preprocessed_manifest)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    if store.img_size not in (None, IMG_SIZE):
        raise SystemExit("%s holds %d px files; the folds are %d px"
                         % (args.preprocessed_manifest, store.img_size, IMG_SIZE))

    if args.verify:
        problems = []
        if not os.path.isfile(args.counts_csv):
            problems.append("%s missing" % args.counts_csv)
        else:
            with open(args.counts_csv, "rb") as f:
                if f.read() != counts_bytes(counts):
                    problems.append("%s differs from the recomputed counts" % args.counts_csv)
        for f, entries in plans.items():
            problems.extend(verify_fold(f, entries, args.output_root, nodes, store))
        for p in problems:
            print("PROBLEM:", p)
        print("camera folds %s, nodes %s: %s" % (" ".join(map(str, plans)), " ".join(nodes),
                                                 "identical" if not problems else
                                                 "%d problem(s)" % len(problems)))
        return 0 if not problems else 1

    for f, entries in plans.items():
        written = materialise_fold(f, entries, store, args.output_root, nodes, args.clean)
        md5 = hashlib.md5(manifest_bytes(entries)).hexdigest()
        print("fold %d: %d images (all nodes), manifest md5 %s; written now: %s"
              % (f, len(entries), md5, ", ".join("%s %d" % kv for kv in written.items())))
    _write_bytes(args.counts_csv, counts_bytes(counts))
    print("wrote %s" % args.counts_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
