#!/usr/bin/env python3
"""FedRGBD -- camera experiment: the scene manifests (LOSO folds and federated folds).

``docs/CAMERA_EXPERIMENT_PREREG.md`` sections 5 and 6.  **Both manifests are generated
from the kept scene ids alone** -- the sorted list of scene ids with at least one valid
frame in ``labels.csv`` (after the pre-registered exclusions of ``camera_labels.py``).
Nothing else enters: no image, no file order, no directory-walk order (CLAUDE.md rule 2),
so any machine that has the same kept scene ids writes byte-identical files.

* ``loso_folds.csv`` -- ``fold,test_scene,val_scene``: one fold per scene in sorted order;
  the validation scene is the next scene in that order (the first after the last); the
  training scenes are the remaining S - 2 (not listed: they are "all others").
* ``fl_folds.csv`` -- ``scene,fold``: the sorted ids permuted with
  ``numpy.random.default_rng(20260928).permutation``, dealt round-robin to folds 0-4
  (position i of the permutation -> fold i mod 5); rows listed in sorted scene order.
  In federated fold f the test scenes are fold f, validation fold (f + 1) mod 5.
* ``MANIFEST_SHA256.txt`` -- SHA-256 of both files and the scene list they came from.

    python scripts/camera_manifests.py                 # write
    python scripts/camera_manifests.py --check         # recompute, report identical/different
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from scripts.camera_labels import kept_scenes, read_labels  # noqa: E402

FL_SEED = 20260928
N_FL_FOLDS = 5
LOSO_FILE = "loso_folds.csv"
FL_FILE = "fl_folds.csv"
SHA_FILE = "MANIFEST_SHA256.txt"


def loso_folds(scenes: Sequence[str]) -> List[Dict[str, object]]:
    s = sorted(set(scenes))
    if len(s) < 3:
        raise ValueError("leave-one-scene-out with a validation scene needs >= 3 scenes, got %d"
                         % len(s))
    return [{"fold": i, "test_scene": s[i], "val_scene": s[(i + 1) % len(s)]}
            for i in range(len(s))]


def fl_folds(scenes: Sequence[str], seed: int = FL_SEED,
             n_folds: int = N_FL_FOLDS) -> Dict[str, int]:
    s = np.array(sorted(set(scenes)), dtype=str)
    perm = np.random.default_rng(seed).permutation(s)
    return {str(scene): i % n_folds for i, scene in enumerate(perm)}


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[object]]) -> bytes:
    buf = io.StringIO()
    buf.write(",".join(header) + "\n")
    for r in rows:
        buf.write(",".join(str(v) for v in r) + "\n")
    return buf.getvalue().encode("utf-8")


def render(scenes: Sequence[str]) -> Dict[str, bytes]:
    """The three files' exact bytes for a scene list."""
    s = sorted(set(scenes))
    loso = _csv_bytes(("fold", "test_scene", "val_scene"),
                      [(f["fold"], f["test_scene"], f["val_scene"]) for f in loso_folds(s)])
    fl = fl_folds(s)
    flb = _csv_bytes(("scene", "fold"), [(k, fl[k]) for k in s])
    sha = ("# generated from the kept scene ids alone (scripts/camera_manifests.py)\n"
           "# scenes (%d): %s\n" % (len(s), " ".join(s))
           + "%s  %s\n" % (hashlib.sha256(loso).hexdigest(), LOSO_FILE)
           + "%s  %s\n" % (hashlib.sha256(flb).hexdigest(), FL_FILE)).encode("utf-8")
    return {LOSO_FILE: loso, FL_FILE: flb, SHA_FILE: sha}


def write(scenes: Sequence[str], splits_dir: str) -> Dict[str, bytes]:
    files = render(scenes)
    os.makedirs(splits_dir, exist_ok=True)
    for name, blob in files.items():
        with open(os.path.join(splits_dir, name), "wb") as f:
            f.write(blob)
    return files


def check(scenes: Sequence[str], splits_dir: str) -> Dict[str, str]:
    """{file: "identical" | "different" | "missing"} against a fresh recomputation."""
    out = {}
    for name, blob in render(scenes).items():
        path = os.path.join(splits_dir, name)
        if not os.path.isfile(path):
            out[name] = "missing"
            continue
        with open(path, "rb") as f:
            out[name] = "identical" if f.read() == blob else "different"
    return out


def read_loso_folds(path: str) -> List[Dict[str, object]]:
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return [{"fold": int(r["fold"]), "test_scene": r["test_scene"],
                 "val_scene": r["val_scene"]} for r in csv.DictReader(f)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels_csv", default=os.path.join("data", "raw", "camera", "labels.csv"))
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="kept scene ids (default: the scenes with a valid frame in labels.csv)")
    ap.add_argument("--splits_dir", default=os.path.join("data", "splits_camera"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    from scripts.camera_labels import refuse_pilot
    refuse_pilot(args.labels_csv, args.splits_dir)

    scenes = sorted(set(args.scenes)) if args.scenes else kept_scenes(read_labels(args.labels_csv))
    if args.check:
        res = check(scenes, args.splits_dir)
        for name, status in res.items():
            print("%-20s %s" % (name, status))
        ok = all(v == "identical" for v in res.values())
        print("manifests: %s" % ("identical" if ok else "different"))
        return 0 if ok else 1
    files = write(scenes, args.splits_dir)
    print("camera manifests from %d kept scene ids: %s" % (len(scenes), " ".join(scenes)))
    for name in (LOSO_FILE, FL_FILE):
        print("  %s  %s" % (hashlib.sha256(files[name]).hexdigest(),
                            os.path.join(args.splits_dir, name)))
    if len(scenes) < N_FL_FOLDS:
        print("  WARNING: fewer scenes than federated folds; some folds are empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
