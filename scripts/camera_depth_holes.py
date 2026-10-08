#!/usr/bin/env python3
"""FedRGBD -- invalid-depth fraction in the e5 wall region (depth-hole diagnosis).

``docs/POST_5B_CHECKLIST.md`` e5: are the dotted holes in node_a's D435if depth caused by
the second RealSense's projector?  The fraction of depth == 0 pixels is computed only
inside the rectangle of ``configs/e5_depth_hole_region.json`` (normalized coordinates,
fixed from the pilot frames before any s06-s08 capture), per frame and per capture.

    python scripts/camera_depth_holes.py --root data/raw/camera_pilot \\
        --captures s06_no_fire_d200 s07_no_fire_d200 s08_no_fire_d200

A technical image property only.  Reads pilot / e5 footage only; the study root
``data/raw/camera`` is refused.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY_ROOT = os.path.join("data", "raw", "camera")
DEFAULT_ROOT = os.path.join("data", "raw", "camera_pilot")
DEFAULT_REGION = os.path.join(_REPO, "configs", "e5_depth_hole_region.json")


def _is_study_root(root: str) -> bool:
    a = os.path.normcase(os.path.abspath(root))
    return a in (os.path.normcase(os.path.abspath(STUDY_ROOT)),
                 os.path.normcase(os.path.abspath(os.path.join(_REPO, STUDY_ROOT))))


def load_region(path: str = DEFAULT_REGION) -> Dict:
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    if not (0 <= r["x0"] < r["x1"] <= 1 and 0 <= r["y0"] < r["y1"] <= 1):
        raise SystemExit("bad region in %s" % path)
    return r


def region_box(width: int, height: int, r: Dict) -> Tuple[int, int, int, int]:
    """Pixel bounds (left, top, right, bottom), half-open, of the normalized rectangle."""
    return (int(round(r["x0"] * width)), int(round(r["y0"] * height)),
            int(round(r["x1"] * width)), int(round(r["y1"] * height)))


def invalid_fraction(depth: np.ndarray, r: Dict) -> float:
    left, top, right, bottom = region_box(depth.shape[1], depth.shape[0], r)
    return float((depth[top:bottom, left:right] == 0).mean())


def measure(root: str, node: str, capture_id: str, r: Dict) -> Tuple[List[Tuple[str, float]], Dict]:
    files = sorted(glob.glob(os.path.join(root, node, capture_id + "_[0-9][0-9][0-9][0-9]_depth.png")))
    if not files:
        raise SystemExit("%s/%s: no depth frames for %s" % (root, node, capture_id))
    per = []
    for p in files:
        with Image.open(p) as im:
            per.append((os.path.basename(p)[:-len("_depth.png")], invalid_fraction(np.asarray(im), r)))
    v = np.array([f for _, f in per])
    with Image.open(files[0]) as im:
        w, h = im.size
    summary = {"node": node, "capture": capture_id, "n_frames": len(per),
               "invalid_mean": round(float(v.mean()), 5), "invalid_median": round(float(np.median(v)), 5),
               "invalid_min": round(float(v.min()), 5), "invalid_max": round(float(v.max()), 5),
               "region_px": list(region_box(w, h, r)), "image_wh": [w, h]}
    return per, summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--node", default=None, help="default: the region file's node")
    p.add_argument("--captures", nargs="+", required=True, help="capture ids, e.g. s06_no_fire_d200")
    args = p.parse_args(argv)
    if _is_study_root(args.root):
        print("refused: %s is the study footage; the e5 diagnosis reads pilot / e5 footage only"
              % args.root, file=sys.stderr)
        return 2
    r = load_region(args.region)
    node = args.node or r["node"]
    print("region (normalized) x [%g, %g) y [%g, %g) from %s" % (r["x0"], r["x1"], r["y0"], r["y1"],
                                                                  os.path.relpath(args.region, _REPO)))
    for cid in args.captures:
        per, summary = measure(args.root, node, cid, r)
        for fid, f in per:
            print("%s %s invalid=%.5f" % (node, fid, f))
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
