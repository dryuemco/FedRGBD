#!/usr/bin/env python3
"""FedRGBD -- flame height in pixels at the classifier's input vs the ruler (e5).

The flame-source acceptance criterion of the camera prereg (Amendment 4, section 13.2,
committed ce85a0d before e5): for every camera x distance, the median vertical flame
height in the 224 input image over the 40 frames of the fire capture must be at least
N = 5 px; a distance that fails on any camera leaves the study's distance set.  This tool
measures how tall the flame actually is in the 224 x 224 input of section 3
(``camera_preprocess.rgb_224``, the very function the study uses), prints PASS / FAIL per
camera x distance and the decision for the distance, and compares the height with what
the ruler predicts from the recorded intrinsics:

    expected_px = flame_height_cm / 100 * fy_224 / distance_m
    fy_224      = fy * (height after the shorter side is resized to 224) / height

Measurement, per node, for one scene x distance: the no-fire capture (same fixed
exposure) gives a per-pixel median reference at 224.  In every fire frame a pixel is a
*flame pixel* if some channel is at least ``--threshold`` brighter than the reference
AND the pixel itself looks like flame: either a near-saturated core (max channel >=
:data:`FLAME_CORE_V`, any hue) or bright and warm (max channel >= :data:`FLAME_V_MIN`,
R >= G >= B, saturation >= :data:`FLAME_WARM_S_MIN`).  Diffuse, low-saturation glow on
walls and floor fails both.  8-connected components of flame pixels that touch the edge
of the 224 crop are dropped; the flame is the remaining component that contains the
brightest flame pixel (ties: larger brightness increase, then the topmost, i.e. the
flame above its floor reflection).  Its height is its row span,
and a frame without flame pixels counts as 0 px (no visible flame).

Tool fix (2026-10-08, after e5 data had been seen; prereg 13.2): the first version took
the *largest* region of any brightness increase, which in s03 included the flame's
glow on the walls and the floor.  The pixel rule and the component choice above replace
it.  Its constants were fixed before the fixed tool was run on any fire frame, are
the same for all three cameras, and are never tuned to a distance result.  The tool is
validated per distance by (b) of prereg 13.2: ratio_measured_expected within
:data:`VALIDATION_BAND` (1 m wider: the candles stand one behind the other and the lit
candle body can join the flame).  The median over all fire frames is the criterion's statistic,
reported with min and max (flicker).  A capture with fewer or more than 40 fire frames is
INCOMPLETE, never PASS.  The flame height (cm) and measured distance come from the
fire capture's parsed ``--notes`` unless given on the command line.

    python scripts/camera_flame_height.py --root data/raw/camera_pilot --scene s01 \\
        --distance_m 3 --overlay flame_s01_d300.png

A technical image property only: no model, no accuracy, no prediction.  It reads only
pilot / acceptance-test footage; the study root ``data/raw/camera`` is refused (the
acceptance decision is taken before any study footage exists).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.data.camera_capture_common import (  # noqa: E402
    CAPTURES_DIR, DEFAULT_FRAMES, NODE_NAMES, make_capture_id, make_frame_id, parse_notes,
)
from src.data.camera_preprocess import SIZE, _crop_box, rgb_224  # noqa: E402

STUDY_ROOT = os.path.join("data", "raw", "camera")
DEFAULT_ROOT = os.path.join("data", "raw", "camera_pilot")
DEFAULT_THRESHOLD = 40
#: the acceptance criterion (prereg Amendment 4, sec. 13.2, ce85a0d): the median vertical
#: flame height at 224 over the CRITERION_FRAMES frames of a camera x distance >= CRITERION_PX
CRITERION_PX = 5
CRITERION_FRAMES = DEFAULT_FRAMES
COLUMNS = ("node", "scene", "distance_m", "distance_measured_m", "flame_height_cm",
           "fy_native", "fy_224", "expected_px", "measured_px_median",
           "measured_px_min", "measured_px_max", "measured_cm_median", "ratio_measured_expected",
           "n_fire_frames", "n_frames_detected", "touches_crop_border", "threshold",
           "validation_band", "tool_validated", "criterion_px", "criterion")


def _is_study_root(root: str) -> bool:
    a = os.path.normcase(os.path.abspath(root))
    return a in (os.path.normcase(os.path.abspath(STUDY_ROOT)),
                 os.path.normcase(os.path.abspath(os.path.join(_REPO, STUDY_ROOT))))


def fy_224(fy: float, width: int, height: int, size: int = SIZE) -> float:
    """Focal length in pixels of the 224 crop (resizing scales it; cropping does not)."""
    (new_w, new_h), _ = _crop_box(int(width), int(height), size)
    return float(fy) * new_h / float(height)


def expected_px(flame_cm: float, distance_m: float, fy224: float) -> float:
    return float(flame_cm) / 100.0 * fy224 / float(distance_m)


def _label(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """8-connected components (scipy if present, else a small flood fill)."""
    try:
        from scipy import ndimage
        return ndimage.label(mask, structure=np.ones((3, 3), int))
    except ImportError:  # pragma: no cover - scipy is on the desktop
        lab = np.zeros(mask.shape, np.int32)
        n = 0
        for y, x in zip(*np.nonzero(mask)):
            if lab[y, x]:
                continue
            n += 1
            stack = [(y, x)]
            lab[y, x] = n
            while stack:
                cy, cx = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                                and mask[ny, nx] and not lab[ny, nx]):
                            lab[ny, nx] = n
                            stack.append((ny, nx))
        return lab, n


#: flame-pixel rule (tool fix 2026-10-08, see the module docstring)
FLAME_CORE_V = 245        # near-saturated flame core: any hue
FLAME_V_MIN = 200         # otherwise bright ...
FLAME_WARM_S_MIN = 0.25   # ... and warm (R >= G >= B) with at least this saturation
#: (b) of prereg 13.2: the tool counts as validated at a distance if the measured/expected
#: ratio lies in this band; outside it, no criterion decision for that distance
VALIDATION_BAND = {1.0: (0.5, 4.0), 2.0: (0.5, 2.0), 3.0: (0.5, 2.0)}


def flame_mask(fire: np.ndarray, reference: np.ndarray, threshold: int
               ) -> Tuple[np.ndarray, np.ndarray]:
    """(flame-pixel mask, brightness increase) of one 224 frame against the reference."""
    f = fire.astype(np.int16)
    diff = (f - reference.astype(np.int16)).max(axis=2)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    v = f.max(axis=2)
    s = (v - f.min(axis=2)) / np.maximum(v, 1).astype(np.float64)
    core = v >= FLAME_CORE_V
    warm = (v >= FLAME_V_MIN) & (r >= g) & (g >= b) & (s >= FLAME_WARM_S_MIN)
    return (diff >= int(threshold)) & (core | warm), diff


def flame_region(fire: np.ndarray, reference: np.ndarray, threshold: int
                 ) -> Optional[Tuple[int, int, int, int]]:
    """(top, left, bottom, right), inclusive, of the flame in one 224 frame: the
    component of flame pixels, not touching the crop edge, holding the brightest one;
    None if there is none."""
    mask, diff = flame_mask(fire, reference, threshold)
    if not mask.any():
        return None
    lab, _ = _label(mask)
    # components touching the edge of the crop are not the flame
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    mask = mask & ~np.isin(lab, edge[edge > 0])
    if not mask.any():
        return None
    v = fire.max(axis=2).astype(np.int32)
    ys, xs = np.nonzero(mask)
    # brightest flame pixel; ties -> larger increase, then topmost, then leftmost
    order = np.lexsort((xs, ys, -diff[ys, xs], -v[ys, xs]))
    seed = lab[ys[order[0]], xs[order[0]]]
    ys, xs = np.nonzero(lab == seed)
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _record(root: str, node: str, capture_id: str) -> Dict:
    path = os.path.join(root, node, CAPTURES_DIR, capture_id + ".json")
    if not os.path.isfile(path):
        raise SystemExit("missing capture record %s" % path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _frames_224(root: str, node: str, rec: Dict) -> List[np.ndarray]:
    out = []
    for i in range(int(rec.get("n_frames_written") or 0)):
        p = os.path.join(root, node, make_frame_id(rec["capture_id"], i) + "_rgb.png")
        if os.path.isfile(p):
            with Image.open(p) as im:
                out.append(np.asarray(rgb_224(im), dtype=np.uint8))
    return out


def measure_node(root: str, node: str, scene: str, distance_m: float,
                 threshold: int = DEFAULT_THRESHOLD, flame_cm: Optional[float] = None,
                 measured_m: Optional[float] = None) -> Tuple[Dict, Optional[np.ndarray],
                                                             Optional[Tuple[int, int, int, int]]]:
    """-> (CSV row, median fire frame at 224, region in it)."""
    fire_rec = _record(root, node, make_capture_id(scene, "fire", distance_m))
    nofire_rec = _record(root, node, make_capture_id(scene, "no_fire", distance_m))
    notes = fire_rec.get("notes_parsed") or parse_notes(fire_rec.get("notes"))
    if flame_cm is None:
        flame_cm = float(notes["flame_height_cm"]) if notes.get("flame_height_cm") else None
    if measured_m is None:
        measured_m = (float(notes["distance_measured_m"]) if notes.get("distance_measured_m")
                      else float(distance_m))
    rgb = (fire_rec.get("intrinsics") or {}).get("rgb") or {}
    f224 = fy_224(rgb["fy"], rgb["width"], rgb["height"])
    fire = _frames_224(root, node, fire_rec)
    ref_frames = _frames_224(root, node, nofire_rec)
    if not fire or not ref_frames:
        raise SystemExit("%s: no frames (fire %d, no-fire %d)" % (node, len(fire), len(ref_frames)))
    reference = np.median(np.stack(ref_frames), axis=0).astype(np.uint8)
    heights, detected, border = [], 0, False
    for fr in fire:
        reg = flame_region(fr, reference, threshold)
        if reg is None:
            heights.append(0)                  # no visible flame in this frame
            continue
        detected += 1
        top, left, bottom, right = reg
        heights.append(bottom - top + 1)
        border = border or top == 0 or left == 0 or bottom == SIZE - 1 or right == SIZE - 1
    med_frame = np.median(np.stack(fire), axis=0).astype(np.uint8)
    med_region = flame_region(med_frame, reference, threshold)
    exp = expected_px(flame_cm, measured_m, f224) if flame_cm else None
    med = float(np.median(heights))
    row = {
        "node": node, "scene": scene, "distance_m": distance_m,
        "distance_measured_m": measured_m, "flame_height_cm": flame_cm,
        "fy_native": float(rgb["fy"]), "fy_224": round(f224, 2),
        "expected_px": None if exp is None else round(exp, 2),
        "measured_px_median": med,
        "measured_px_min": min(heights),
        "measured_px_max": max(heights),
        "measured_cm_median": round(med * 100.0 * measured_m / f224, 1),
        "ratio_measured_expected": None if not exp else round(med / exp, 2),
        "n_fire_frames": len(fire), "n_frames_detected": detected,
        "touches_crop_border": border, "threshold": threshold,
        "criterion_px": CRITERION_PX, "criterion": criterion(med, len(fire)),
    }
    band = VALIDATION_BAND.get(round(float(distance_m), 1))
    ratio = row["ratio_measured_expected"]
    row["validation_band"] = None if band is None else "%g-%g" % band
    row["tool_validated"] = bool(band and ratio is not None and band[0] <= ratio <= band[1])
    if not row["tool_validated"] and row["criterion"] != "INCOMPLETE":
        # (b): no criterion decision where the tool is not validated
        row["criterion"] = "NOT VALIDATED"
    return row, med_frame, med_region


def criterion(median_px: float, n_frames: int) -> str:
    """PASS / FAIL of one camera x distance; INCOMPLETE unless exactly CRITERION_FRAMES."""
    if n_frames != CRITERION_FRAMES:
        return "INCOMPLETE"
    return "PASS" if median_px >= CRITERION_PX else "FAIL"


def distance_decision(rows: Sequence[Dict]) -> str:
    """The distance stays in the study set only if every one of the three cameras PASSes."""
    verdicts = {r["node"]: r["criterion"] for r in rows}
    d = rows[0]["distance_m"]
    detail = ", ".join("%s %s" % (n, verdicts[n]) for n in sorted(verdicts))
    if set(verdicts) != set(NODE_NAMES):
        return "distance %g m: NOT DECIDED -- only %d of %d cameras measured (%s)" % (
            d, len(verdicts), len(NODE_NAMES), detail)
    if all(v == "PASS" for v in verdicts.values()):
        return "distance %g m: PASS on all three cameras (%s)" % (d, detail)
    if any(v == "INCOMPLETE" for v in verdicts.values()):
        return "distance %g m: NOT DECIDED -- a capture does not have %d frames (%s)" % (
            d, CRITERION_FRAMES, detail)
    if any(v == "NOT VALIDATED" for v in verdicts.values()):
        return ("distance %g m: NOT DECIDED -- the flame tool is not validated at this "
                "distance on every camera (prereg 13.2 (b)) (%s)" % (d, detail))
    return "distance %g m: FAIL -- removed from the study's distance set (%s)" % (d, detail)


def overlay(frames: Sequence[Tuple[str, np.ndarray, Optional[Tuple[int, int, int, int]]]],
            path: str, scale: int = 2) -> None:
    """Median fire frame per node at 224 (upscaled), the measured region boxed."""
    tiles = []
    for node, frame, reg in frames:
        im = Image.fromarray(frame).resize((SIZE * scale, SIZE * scale), Image.NEAREST)
        d = ImageDraw.Draw(im)
        if reg is not None:
            top, left, bottom, right = reg
            d.rectangle([left * scale, top * scale, (right + 1) * scale - 1,
                         (bottom + 1) * scale - 1], outline=(0, 255, 0), width=1)
        d.text((4, 4), node, fill=(255, 255, 0))
        tiles.append(im)
    out = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width
    out.save(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--scene", required=True)
    p.add_argument("--distance_m", type=float, required=True)
    p.add_argument("--nodes", nargs="+", default=list(NODE_NAMES), choices=NODE_NAMES)
    p.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                   help="per-channel brightness increase over the no-fire median (0-255)")
    p.add_argument("--flame_height_cm", type=float, default=None,
                   help="ruler reading; default: the fire capture's --notes")
    p.add_argument("--distance_measured_m", type=float, default=None,
                   help="default: the fire capture's --notes")
    p.add_argument("--out_csv", default=None)
    p.add_argument("--overlay", default=None, help="PNG with the measured region per node")
    args = p.parse_args(argv)
    if _is_study_root(args.root):
        print("refused: %s is the study footage; the acceptance test reads pilot / e5 "
              "footage only" % args.root, file=sys.stderr)
        return 2
    rows, tiles = [], []
    for node in args.nodes:
        row, frame, reg = measure_node(args.root, node, args.scene, args.distance_m,
                                       args.threshold, args.flame_height_cm,
                                       args.distance_measured_m)
        rows.append(row)
        tiles.append((node, frame, reg))
    w = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    for r in rows:
        print("%s %g m: %s (median %.1f px over %d frames, criterion >= %d px over %d)" % (
            r["node"], r["distance_m"], r["criterion"], r["measured_px_median"],
            r["n_fire_frames"], CRITERION_PX, CRITERION_FRAMES), file=sys.stderr)
    print(distance_decision(rows), file=sys.stderr)
    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            cw = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
            cw.writeheader()
            cw.writerows(rows)
    if args.overlay:
        overlay(tiles, args.overlay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
