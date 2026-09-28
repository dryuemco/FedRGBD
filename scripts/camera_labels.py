#!/usr/bin/env python3
"""FedRGBD -- camera experiment: frame table and the pre-registered exclusions.

Implements ``docs/CAMERA_EXPERIMENT_PREREG.md`` section 2, "Exclusions", exactly:

1. a frame is invalid if its RGB image is missing, unreadable, or nearly constant
   (every channel's standard deviation < 2 grey levels);
2. a capture with fewer than 30 valid frames on ANY camera is dropped on ALL three
   cameras (the pairing is kept);
3. a scene that loses a whole class on any camera is dropped from the study.

Input (the capture contract): ``data/raw/camera/<node>/{id}_rgb.png``, ``_depth.png``,
``_ir.png`` (RealSense only), ``_meta.json`` with frame_id, capture_id, scene, label, ...;
``id = f"{capture_id}_{frame_index:04d}"``, ``capture_id = f"{scene}_{label}_d{cm}"``;
capture records in ``data/raw/camera/<node>/_captures/<capture_id>.json``.

Output:
* ``data/raw/camera/labels.csv`` -- ``id,node,scene,label,distance_m,capture_id,
  frame_index,valid,exclusion_reason``, one row per frame found on disk (valid or not),
  sorted by ``(node, id)`` -- never directory-walk order.  Readable by
  ``src/data/custom_dataset.py`` (``id,scene,label[,node]``).  The label is the operator's
  declaration (meta.json / capture id), never read off the images; "fire" -> Fire (1),
  "no_fire" -> No_Fire (0).
* ``data/splits_camera/exclusions.csv`` -- ``level,key,reason`` for every excluded frame,
  capture and scene.

    python scripts/camera_labels.py --data_dir data/raw/camera --splits_dir data/splits_camera
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

NODES = ("node_a", "node_b", "node_c")
SENSORS = {"node_a": "D435if", "node_b": "D435i", "node_c": "ZED 2i"}
LABELS = ("fire", "no_fire")
LABEL_INDEX = {"no_fire": 0, "fire": 1}          # repo convention: No_Fire = 0, Fire = 1

MIN_VALID_FRAMES = 30                            # per capture and camera
CONSTANT_STD = 2.0                               # grey levels, every channel below -> invalid

LABEL_COLUMNS = ("id", "node", "scene", "label", "distance_m", "capture_id", "frame_index",
                 "valid", "exclusion_reason")
EXCLUSION_COLUMNS = ("level", "key", "reason")

_ID_RE = re.compile(r"^(?P<capture>(?P<scene>s\d+)_(?P<label>fire|no_fire)_d(?P<cm>\d+))"
                    r"_(?P<idx>\d{4,})$")
_SUFFIXES = ("_rgb.png", "_depth.png", "_ir.png", "_meta.json")


# --------------------------------------------------------------------------- frames
def parse_frame_id(frame_id: str) -> Optional[Dict[str, object]]:
    """``s01_no_fire_d200_0007`` -> scene, label, capture_id, distance_m, frame_index."""
    m = _ID_RE.match(frame_id)
    if not m:
        return None
    return {"scene": m.group("scene"), "label": m.group("label"),
            "capture_id": m.group("capture"), "distance_m": int(m.group("cm")) / 100.0,
            "frame_index": int(m.group("idx"))}


def rgb_status(path: str) -> str:
    """"" if the RGB image is valid, else the exclusion reason."""
    if not os.path.isfile(path):
        return "rgb_missing"
    try:
        with Image.open(path) as im:
            im.load()
            arr = np.asarray(im.convert("RGB"), dtype=np.float64)
    except Exception:  # noqa: BLE001 -- any decoding failure means unreadable
        return "rgb_unreadable"
    if arr.size == 0:
        return "rgb_unreadable"
    std = arr.reshape(-1, 3).std(axis=0)
    if bool(np.all(std < CONSTANT_STD)):
        return "rgb_nearly_constant"
    return ""


def _read_json(path: str) -> Dict[str, object]:
    try:
        with open(path, encoding="utf-8") as f:
            out = json.load(f)
        return out if isinstance(out, dict) else {}
    except (OSError, ValueError):
        return {}


def scan_node(data_dir: str, node: str) -> List[Dict[str, object]]:
    """One record per frame id with any file on disk (sorted by id)."""
    node_dir = os.path.join(data_dir, node)
    if not os.path.isdir(node_dir):
        return []
    ids = set()
    for name in os.listdir(node_dir):
        for suf in _SUFFIXES:
            if name.endswith(suf):
                ids.add(name[: -len(suf)])
    rows = []
    for fid in sorted(ids):
        parsed = parse_frame_id(fid)
        meta = _read_json(os.path.join(node_dir, fid + "_meta.json"))
        if parsed is None:
            raise ValueError("%s/%s: frame id does not follow <scene>_<label>_d<cm>_<nnnn>"
                             % (node, fid))
        for k in ("scene", "label", "capture_id"):
            if meta.get(k) is not None and str(meta[k]) != str(parsed[k]):
                raise ValueError("%s/%s: meta.json %s=%r contradicts the frame id (%r)"
                                 % (node, fid, k, meta[k], parsed[k]))
        if meta.get("frame_index") is not None and int(meta["frame_index"]) != parsed["frame_index"]:
            raise ValueError("%s/%s: meta.json frame_index contradicts the frame id" % (node, fid))
        distance = meta.get("distance_m")
        distance = float(distance) if distance is not None else float(parsed["distance_m"])
        reason = rgb_status(os.path.join(node_dir, fid + "_rgb.png"))
        rows.append({"id": fid, "node": node, "scene": parsed["scene"],
                     "label": parsed["label"], "distance_m": distance,
                     "capture_id": parsed["capture_id"],
                     "frame_index": parsed["frame_index"],
                     "valid": 0 if reason else 1, "exclusion_reason": reason})
    return rows


def recorded_captures(data_dir: str, node: str) -> List[str]:
    """Capture ids with a record in ``<node>/_captures`` (a capture may have no frames)."""
    cap_dir = os.path.join(data_dir, node, "_captures")
    if not os.path.isdir(cap_dir):
        return []
    return sorted(n[:-5] for n in os.listdir(cap_dir) if n.endswith(".json"))


# --------------------------------------------------------------------------- rules
def apply_exclusions(rows: List[Dict[str, object]], captures: Sequence[str],
                     nodes: Sequence[str] = NODES,
                     ) -> Tuple[List[Dict[str, object]], List[Dict[str, str]], Dict[str, object]]:
    """Apply the three pre-registered rules in order; -> (rows, exclusions, summary)."""
    rows = sorted(rows, key=lambda r: (r["node"], r["id"]))
    exclusions: List[Dict[str, str]] = []
    for r in rows:
        if r["exclusion_reason"]:
            exclusions.append({"level": "frame", "key": "%s/%s" % (r["node"], r["id"]),
                               "reason": str(r["exclusion_reason"])})

    all_captures = sorted(set(captures) | {str(r["capture_id"]) for r in rows})
    valid_count: Dict[Tuple[str, str], int] = {}
    for r in rows:
        if r["valid"]:
            key = (str(r["capture_id"]), str(r["node"]))
            valid_count[key] = valid_count.get(key, 0) + 1

    # rule 2: < 30 valid frames on ANY camera -> the capture is dropped on all cameras
    dropped_captures: Dict[str, str] = {}
    for cap in all_captures:
        short = ["%s (%d)" % (n, valid_count.get((cap, n), 0)) for n in nodes
                 if valid_count.get((cap, n), 0) < MIN_VALID_FRAMES]
        if short:
            dropped_captures[cap] = ("fewer than %d valid frames on %s"
                                     % (MIN_VALID_FRAMES, ", ".join(short)))
            exclusions.append({"level": "capture", "key": cap, "reason": dropped_captures[cap]})

    # rule 3: a scene without a kept capture of each class on some camera is dropped
    kept_classes: Dict[Tuple[str, str], set] = {}
    scenes = set()
    for cap in all_captures:
        parsed = parse_frame_id(cap + "_0000")
        if parsed is None:
            raise ValueError("capture id %r does not follow <scene>_<label>_d<cm>" % cap)
        scenes.add(parsed["scene"])
    for r in rows:
        if r["valid"] and r["capture_id"] not in dropped_captures:
            kept_classes.setdefault((str(r["scene"]), str(r["node"])), set()).add(str(r["label"]))
    dropped_scenes: Dict[str, str] = {}
    for scene in sorted(scenes):
        lost = []
        for n in nodes:
            missing = [c for c in LABELS if c not in kept_classes.get((scene, n), set())]
            if missing:
                lost.append("%s on %s" % ("+".join(missing), n))
        if lost:
            dropped_scenes[scene] = "lost a whole class: " + "; ".join(lost)
            exclusions.append({"level": "scene", "key": scene, "reason": dropped_scenes[scene]})

    for r in rows:
        if not r["valid"]:
            continue
        if r["capture_id"] in dropped_captures:
            r["valid"], r["exclusion_reason"] = 0, "capture_dropped"
        elif r["scene"] in dropped_scenes:
            r["valid"], r["exclusion_reason"] = 0, "scene_dropped"

    kept = sorted(scenes - set(dropped_scenes))
    frames = {n: sum(1 for r in rows if r["node"] == n and r["valid"]) for n in nodes}
    summary = {"scenes_found": sorted(scenes), "scenes_kept": kept,
               "scenes_dropped": dropped_scenes, "captures_found": len(all_captures),
               "captures_dropped": dropped_captures, "valid_frames_per_node": frames,
               "frames_found": len(rows)}
    return rows, exclusions, summary


# --------------------------------------------------------------------------- io
def _fmt(value) -> str:
    if isinstance(value, float):
        return "%.3f" % value
    return str(value)


def write_csv(path: str, columns: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    """LF line endings on every OS, so the bytes (and hashes) are machine-independent."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow([_fmt(r[c]) for c in columns])


def read_labels(path: str) -> List[Dict[str, object]]:
    """labels.csv rows with typed ``valid`` / ``frame_index`` / ``distance_m``."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["valid"] = int(r["valid"])
        r["frame_index"] = int(r["frame_index"])
        r["distance_m"] = float(r["distance_m"])
    return rows


def kept_scenes(labels_rows: Sequence[Dict[str, object]]) -> List[str]:
    """Scene ids with at least one valid frame, sorted (the study's scene set)."""
    return sorted({str(r["scene"]) for r in labels_rows if int(r["valid"])})


def build(data_dir: str, nodes: Sequence[str] = NODES):
    rows: List[Dict[str, object]] = []
    captures: set = set()
    for n in nodes:
        rows.extend(scan_node(data_dir, n))
        captures.update(recorded_captures(data_dir, n))
    if not rows and not captures:
        raise FileNotFoundError("no frames or capture records under %s" % data_dir)
    return apply_exclusions(rows, sorted(captures), nodes)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_dir", default=os.path.join("data", "raw", "camera"))
    ap.add_argument("--labels_csv", default=None,
                    help="default: <data_dir>/labels.csv")
    ap.add_argument("--splits_dir", default=os.path.join("data", "splits_camera"))
    args = ap.parse_args(argv)

    rows, exclusions, summary = build(args.data_dir)
    labels_csv = args.labels_csv or os.path.join(args.data_dir, "labels.csv")
    write_csv(labels_csv, LABEL_COLUMNS, rows)
    write_csv(os.path.join(args.splits_dir, "exclusions.csv"), EXCLUSION_COLUMNS, exclusions)

    print("camera labels: %d frames found, %d captures" % (summary["frames_found"],
                                                           summary["captures_found"]))
    print("  scenes kept    (%d): %s" % (len(summary["scenes_kept"]),
                                         " ".join(summary["scenes_kept"]) or "-"))
    for s, why in sorted(summary["scenes_dropped"].items()):
        print("  scene dropped  %s: %s" % (s, why))
    for c, why in sorted(summary["captures_dropped"].items()):
        print("  capture dropped %s: %s" % (c, why))
    for n, k in summary["valid_frames_per_node"].items():
        print("  %s (%s): %d valid frames" % (n, SENSORS.get(n, n), k))
    if len(summary["scenes_kept"]) < 15:
        print("  NOTE: fewer than 15 complete scenes; the analysis still runs and the paper "
              "states the number (prereg sec. 2)")
    print("  wrote %s and %s" % (labels_csv, os.path.join(args.splits_dir, "exclusions.csv")))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise SystemExit(main())
