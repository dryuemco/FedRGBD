#!/usr/bin/env python3
"""FedRGBD -- camera experiment: preprocess every valid frame ONCE, on one machine.

``docs/CAMERA_EXPERIMENT_PREREG.md`` section 3 (the preprocessing) and Amendment 2
(2026-10-05, before any footage): decoding, resizing and the depth conversion run once,
on the desktop, and every consumer reads the resulting files, md5-verified against the
committed manifest.  Run this after ``camera_labels.py`` on the desktop copy of all three
nodes' frames:

    data/processed/camera_224/<node>/<id>_rgb.png     shorter side 224 + centre crop, uint8
    data/processed/camera_224/<node>/<id>_ir.png      same geometry, uint8 (RealSense only)
    data/processed/camera_224/<node>/<id>_depth.npy   float32 [0, 1] (0.3-10 m, invalid 0)
    data/splits_camera/preprocessed_manifest.csv      node,id,stream,file,img_size,bytes,md5
    data/splits_camera/preprocessed_info.json         machine, library versions, commit

Every valid frame of ``labels.csv`` (``valid == 1``) gets its RGB stream, plus depth and IR
where the capture has them.  The pixels come from the functions the dataset used before
(``custom_dataset._rgb_camera_u8`` / ``_ir_camera_u8`` / ``load_depth_camera``, i.e.
``camera_preprocess.rgb_224`` / ``depth_224``).

Consumers: ``camera_fl_prepare.py`` copies a node's RGB files byte for byte into its
federated folds (each node receives only its own camera's files), the desktop baselines
train on those folds, and ``CustomRGBDDataset(preprocess="camera")`` (LOSO, RGB-D, IR)
reads the files through ``camera_preprocess.PreprocessedStore``.

Pilot: with a pilot ``--raw_dir`` every output must also lie inside the pilot tree
(prereg section 10), and study inputs never write into it.

    python scripts/camera_preprocess_frames.py                 # build (rewrites the output)
    python scripts/camera_preprocess_frames.py --verify        # md5 of every file; writes nothing
    python scripts/camera_preprocess_frames.py --verify --nodes node_b   # a node's own files
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.data.camera_preprocess import (  # noqa: E402
    PREPROCESSED_DIR, PREPROCESSED_MANIFEST, SIZE, encode_stream, manifest_bytes, md5_of,
    preprocessed_file, read_manifest,
)

NODES = ("node_a", "node_b", "node_c")
DEFAULT_RAW = os.path.join("data", "raw", "camera")


def info_path(manifest: str) -> str:
    return os.path.splitext(manifest)[0].replace("_manifest", "_info") + ".json"


def check_paths(raw_dir: str, labels_csv: str, output_dir: str, manifest: str) -> None:
    from scripts.camera_labels import is_pilot_path, refuse_pilot
    if is_pilot_path(raw_dir) or is_pilot_path(labels_csv):
        outside = [p for p in (labels_csv, output_dir, manifest) if not is_pilot_path(p)]
        if outside:
            raise SystemExit("%s: pilot footage is preprocessed only into the pilot tree "
                             "(prereg section 10)" % outside[0])
    else:
        refuse_pilot(output_dir, manifest)


def frames_to_do(labels_csv: str, nodes: Sequence[str]) -> List[Dict[str, object]]:
    from scripts.camera_labels import read_labels
    if not os.path.isfile(labels_csv):
        raise SystemExit("%s is missing -- write it with scripts/camera_labels.py first" % labels_csv)
    rows = [r for r in read_labels(labels_csv) if int(r["valid"]) and str(r["node"]) in nodes]
    seen = set()
    for r in rows:
        key = (str(r["node"]), str(r["id"]))
        if key in seen:
            raise ValueError("%s: frame %s listed twice" % (labels_csv, "/".join(key)))
        seen.add(key)
    return sorted(rows, key=lambda r: (str(r["node"]), str(r["id"])))


def build(raw_dir: str, labels_csv: str, output_dir: str, manifest: str,
          img_size: int = SIZE) -> List[Dict[str, object]]:
    from src.data.custom_dataset import _ir_camera_u8, _rgb_camera_u8, load_depth_camera
    loaders = {"rgb": _rgb_camera_u8, "ir": _ir_camera_u8, "depth": load_depth_camera}
    suffix_raw = {"rgb": "_rgb.png", "ir": "_ir.png", "depth": "_depth.png"}
    frames = frames_to_do(labels_csv, NODES)
    if not frames:
        raise SystemExit("%s has no valid frame" % labels_csv)
    for n in NODES:
        shutil.rmtree(os.path.join(output_dir, n), ignore_errors=True)
    rows = []
    for r in frames:
        node, fid = str(r["node"]), str(r["id"])
        for stream in ("rgb", "depth", "ir"):
            src = os.path.join(raw_dir, node, fid + suffix_raw[stream])
            if not os.path.isfile(src):
                if stream == "rgb":
                    raise SystemExit("valid frame %s/%s has no %s" % (node, fid, src))
                continue
            blob = encode_stream(stream, loaders[stream](src, img_size))
            rel = preprocessed_file(node, fid, stream)
            dst = os.path.join(output_dir, *rel.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(blob)
            rows.append({"node": node, "id": fid, "stream": stream, "file": rel,
                         "img_size": img_size, "bytes": len(blob), "md5": md5_of(dst)})
    os.makedirs(os.path.dirname(os.path.abspath(manifest)), exist_ok=True)
    with open(manifest, "wb") as f:
        f.write(manifest_bytes(rows))
    with open(info_path(manifest), "w", encoding="utf-8", newline="\n") as f:
        json.dump(machine_info(), f, indent=2, sort_keys=True)
        f.write("\n")
    return rows


def machine_info() -> Dict[str, object]:
    import numpy
    import PIL
    try:
        commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    return {"host": platform.node(), "platform": platform.platform(),
            "python": platform.python_version(), "pillow": PIL.__version__,
            "numpy": numpy.__version__, "commit": commit}


def verify(output_dir: str, manifest: str, nodes: Sequence[str],
           labels_csv: Optional[str] = None) -> List[str]:
    """Problems (empty = OK): every manifest file of ``nodes`` present with its md5 and
    size, no other file below those node directories, every valid frame covered."""
    problems = []
    rows = {k: r for k, r in read_manifest(manifest).items() if k[0] in nodes}
    for (node, fid, stream), r in sorted(rows.items()):
        path = os.path.join(output_dir, *str(r["file"]).split("/"))
        if not os.path.isfile(path):
            problems.append("%s missing" % path)
        elif os.path.getsize(path) != r["bytes"] or md5_of(path) != r["md5"]:
            problems.append("%s: differs from the manifest" % path)
    wanted = {str(r["file"]) for r in rows.values()}
    for n in nodes:
        root = os.path.join(output_dir, n)
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                rel = os.path.relpath(os.path.join(dirpath, name), output_dir).replace("\\", "/")
                if rel not in wanted:
                    problems.append("%s: not in the manifest" % rel)
    if labels_csv and os.path.isfile(labels_csv):
        for r in frames_to_do(labels_csv, nodes):
            if (str(r["node"]), str(r["id"]), "rgb") not in rows:
                problems.append("valid frame %s/%s has no preprocessed RGB" % (r["node"], r["id"]))
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", default=DEFAULT_RAW, help="<raw_dir>/<node>/<id>_{rgb,depth,ir}.png")
    ap.add_argument("--labels_csv", default=None, help="default: <raw_dir>/labels.csv")
    ap.add_argument("--output_dir", default=PREPROCESSED_DIR)
    ap.add_argument("--manifest", default=PREPROCESSED_MANIFEST)
    ap.add_argument("--img_size", type=int, default=SIZE, help="224 = prereg; smaller only for tests")
    ap.add_argument("--nodes", nargs="+", default=list(NODES), choices=NODES,
                    help="--verify only: the nodes whose files to check (a Jetson holds its own)")
    ap.add_argument("--verify", action="store_true", help="check every file's md5; write nothing")
    args = ap.parse_args(argv)
    labels_csv = args.labels_csv or os.path.join(args.raw_dir, "labels.csv")
    check_paths(args.raw_dir, labels_csv, args.output_dir, args.manifest)
    if args.verify:
        problems = verify(args.output_dir, args.manifest, [n for n in NODES if n in args.nodes],
                          labels_csv)
        for p in problems:
            print("PROBLEM:", p)
        print("preprocessed camera files, nodes %s: %s" % (" ".join(args.nodes),
              "identical to the manifest" if not problems else "%d problem(s)" % len(problems)))
        return 0 if not problems else 1
    if args.nodes != list(NODES):
        ap.error("--nodes is for --verify; the build always covers all three nodes")
    rows = build(args.raw_dir, labels_csv, args.output_dir, args.manifest, args.img_size)
    print("preprocessed %d frame(s), %d file(s) -> %s; manifest %s (md5 %s)"
          % (len({(r["node"], r["id"]) for r in rows}), len(rows), args.output_dir, args.manifest,
             md5_of(args.manifest)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
