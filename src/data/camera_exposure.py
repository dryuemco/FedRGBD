#!/usr/bin/env python3
"""FedRGBD -- one fixed (exposure, gain, white balance) per camera (prereg 13.1, DRAFT).

Every study capture of a camera uses the same exposure, gain and white balance, in every
scene, at every distance and for fire and no-fire alike, with auto exposure and auto
white balance off.  The values are set once per camera by :func:`calibrate_camera`, on a
**flame-free** calibration scene (the capture setup with the source in place and unlit;
``--notes`` must parse as a no-fire capture and name the lamps that are on):

1. **gain**: the camera's default (RealSense: the option range's ``default``; ZED: 0,
   the SDK reports no default);
2. **white balance**: with gain at its default and exposure at the camera's default, the
   value on the white balance grid that minimises |mean R - mean B| inside the camera's
   neutral wall rectangle (``configs/camera_wb_region.json``, normalized coordinates).
   Bisection on the sign of R - B (a higher white balance setting makes the image
   warmer), then the better of the two bracketing grid values;
3. **exposure**: gain and white balance fixed, the first exposure met by a bisection on
   a log scale (start: the camera's default) whose 224 input image (section 3 geometry)
   has a mean 8-bit luma in :data:`LUMA_TARGET`.  If the exposure cap is reached and the
   luma is still below the target, the exposure stays at the cap and the **gain** is
   raised by the same bisection (between the default gain and the gain maximum);
4. **verification**: a last set of frames with the final values; its 224 luma and the
   rectangle's mean R, G, B are reported and stored.

No calibration frame is written, only the measured values, into
``<root>/<node>/_exposure/exposure.json`` with every step of both searches.  A capture
is refused when the camera has no such file, when the file belongs to another camera,
or when its ``lamps_on`` differ from the capture's (a lamp change needs a new
calibration).  No frame with a flame is used to choose the values.

    python3 src/data/realsense_capture.py --calibrate_exposure --node node_b \\
        --root data/raw/camera_pilot --notes "source=none; distance_measured_m=3.00; \\
        flame_height_cm=0; lamps_on=tavan lambasi"
    python3 src/data/zed_capture.py --calibrate_exposure --node node_c --root ... --notes ...
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#: the 224-input mean luma (8-bit) the exposure search aims for (author, 2026-10-08)
LUMA_TARGET = (100.0, 130.0)
SETTLE_FRAMES = 15      # frames grabbed and discarded after each setting change
MEASURE_FRAMES = 10     # frames measured per step
MAX_STEPS = 24
EXPOSURE_DIR = "_exposure"
EXPOSURE_FILE = "exposure.json"
WB_REGION_FILE = os.path.join(_REPO, "configs", "camera_wb_region.json")


class ExposureError(RuntimeError):
    """No fixed exposure available, or the calibration could not complete."""


def exposure_path(root: str, node: str) -> str:
    """``<root>/<node>/_exposure/exposure.json``: the one setting of this camera."""
    return os.path.join(root, node, EXPOSURE_DIR, EXPOSURE_FILE)


def load_wb_region(node: str, path: str = WB_REGION_FILE) -> Dict[str, float]:
    with open(path, encoding="utf-8") as f:
        r = (json.load(f).get("regions") or {}).get(node)
    if not r:
        raise ExposureError("no neutral white-balance rectangle for %s in %s: define it "
                            "from a framing frame of this camera first" % (node, path))
    if not (0 <= r["x0"] < r["x1"] <= 1 and 0 <= r["y0"] < r["y1"] <= 1):
        raise ExposureError("bad rectangle for %s in %s" % (node, path))
    return r


def region_box(width: int, height: int, r: Dict[str, float]) -> Tuple[int, int, int, int]:
    """Pixel bounds (left, top, right, bottom), half-open, of a normalized rectangle."""
    return (int(round(r["x0"] * width)), int(round(r["y0"] * height)),
            int(round(r["x1"] * width)), int(round(r["y1"] * height)))


def mean_luma_224(rgb: np.ndarray) -> float:
    """Mean 8-bit luma (Pillow ``L``) of the section-3 224 input image of an RGB frame."""
    from PIL import Image
    from src.data.camera_preprocess import rgb_224
    img = rgb_224(Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), "RGB"))
    return float(np.asarray(img.convert("L"), dtype=np.float64).mean())


def region_rgb(rgb: np.ndarray, r: Dict[str, float]) -> Tuple[float, float, float]:
    left, top, right, bottom = region_box(rgb.shape[1], rgb.shape[0], r)
    patch = np.asarray(rgb[top:bottom, left:right], dtype=np.float64)
    return tuple(float(patch[..., c].mean()) for c in range(3))


def _snap(value: float, lo: float, hi: float, step: float) -> float:
    step = float(step) or 1.0
    v = lo + round((float(value) - lo) / step) * step
    return float(min(max(v, lo), hi))


def measure(backend, setting: Dict[str, float], region: Dict[str, float],
            settle: int = SETTLE_FRAMES, n: int = MEASURE_FRAMES) -> Dict[str, Any]:
    """Apply ``setting`` (AE/AWB off), discard ``settle`` frames, measure ``n``:
    the median 224 mean luma and the mean R, G, B in the neutral rectangle."""
    applied = dict(backend.apply_lock(setting))
    for _ in range(int(settle)):
        backend.skip()
    lum, rgbm = [], []
    for _ in range(int(n)):
        rgb = backend.grab().rgb
        lum.append(mean_luma_224(rgb))
        rgbm.append(region_rgb(rgb, region))
    m = np.mean(np.array(rgbm), axis=0)
    return {"setting": dict(setting), "applied": applied,
            "luma_224_median": round(float(np.median(lum)), 3),
            "region_rgb_mean": [round(float(v), 3) for v in m],
            "r_minus_b": round(float(m[0] - m[2]), 3)}


def search_white_balance(backend, gain, exposure, wb_range, region, max_steps=MAX_STEPS,
                         log=print) -> Tuple[float, List[Dict[str, Any]]]:
    """The grid value minimising |R - B| in ``region`` (bisection on the sign of R - B)."""
    lo, hi = float(wb_range["min"]), float(wb_range["max"])
    step = float(wb_range.get("step") or 1.0)
    trace: List[Dict[str, Any]] = []
    seen: Dict[float, Dict[str, Any]] = {}

    def at(wb):
        if wb not in seen:
            seen[wb] = measure(backend, {"exposure": exposure, "gain": gain,
                                         "white_balance": wb}, region)
            trace.append(dict(seen[wb], step="white_balance"))
            log("calibration WB %g: R-B %.2f (R,G,B %s)"
                % (wb, seen[wb]["r_minus_b"], seen[wb]["region_rgb_mean"]))
        return seen[wb]["r_minus_b"]

    d_lo, d_hi = at(lo), at(hi)
    if d_lo >= 0 or d_hi <= 0:          # no sign change: the better end
        return (lo if abs(d_lo) <= abs(d_hi) else hi), trace
    for _ in range(int(max_steps)):
        if hi - lo <= step:
            break
        mid = _snap((lo + hi) / 2.0, lo, hi, step)
        if mid in (lo, hi):
            break
        if at(mid) < 0:
            lo = mid
        else:
            hi = mid
    best = lo if abs(seen[lo]["r_minus_b"]) <= abs(seen[hi]["r_minus_b"]) else hi
    return best, trace


def search_gain(backend, exposure, wb, gain0, gain_range, region, target=LUMA_TARGET,
                max_steps=MAX_STEPS, log=print) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    """Exposure at its cap and still too dark: the first gain of the same log-scale
    bisection, between ``gain0`` and the gain maximum, whose 224 mean luma is in target."""
    g_max = float(gain_range["max"])
    step = float(gain_range.get("step") or 1.0)
    lo, hi = float(gain0), g_max
    trace: List[Dict[str, Any]] = []
    g = _snap(math.sqrt(max(lo, step) * hi), float(gain_range["min"]), g_max, step)
    for _ in range(int(max_steps)):
        m = measure(backend, {"exposure": exposure, "gain": g, "white_balance": wb}, region)
        trace.append(dict(m, step="gain"))
        log("calibration gain %g (exposure at cap %g): 224 mean luma %.2f"
            % (g, exposure, m["luma_224_median"]))
        if target[0] <= m["luma_224_median"] <= target[1]:
            return g, trace
        if m["luma_224_median"] < target[0]:
            lo = g
        else:
            hi = g
        nxt = _snap(math.sqrt(max(lo, step) * hi), float(gain_range["min"]), g_max, step)
        if nxt == g:
            break
        g = nxt
    return None, trace


def search_exposure(backend, gain, wb, exp_range, region, target=LUMA_TARGET,
                    max_steps=MAX_STEPS, log=print) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    """The first exposure of a log-scale bisection whose 224 mean luma lies in ``target``."""
    e_min, e_max = float(exp_range["min"]), float(exp_range["max"])
    step = float(exp_range.get("step") or 1.0)
    lo, hi = e_min, e_max
    e = _snap(exp_range.get("default") or math.sqrt(max(e_min, step) * e_max),
              e_min, e_max, step)
    trace: List[Dict[str, Any]] = []
    for _ in range(int(max_steps)):
        m = measure(backend, {"exposure": e, "gain": gain, "white_balance": wb}, region)
        trace.append(dict(m, step="exposure"))
        log("calibration exposure %g: 224 mean luma %.2f" % (e, m["luma_224_median"]))
        if target[0] <= m["luma_224_median"] <= target[1]:
            return e, trace
        if m["luma_224_median"] < target[0]:
            lo = e
        else:
            hi = e
        nxt = _snap(math.sqrt(max(lo, step) * hi), e_min, e_max, step)
        if nxt == e:
            break
        e = nxt
    return None, trace


def calibrate_camera(backend, root: str, node: str, defaults: Dict[str, Any],
                     notes: str = "", gain: Optional[float] = None,
                     region: Optional[Dict[str, float]] = None, recalibrate: bool = False,
                     target=LUMA_TARGET, clock: Callable[[], float] = time.time,
                     log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Gain (default), then white balance, then exposure, then a verification; write and
    return ``<root>/<node>/_exposure/exposure.json``.  ``backend`` is open."""
    from src.data.camera_capture_common import notes_problems, parse_notes, write_json
    bad = notes_problems(notes, "no_fire")
    if bad:
        raise ExposureError("calibration scene must be flame-free and its --notes complete: "
                            + "; ".join(bad))
    lamps = parse_notes(notes).get("lamps_on")
    if not lamps:
        raise ExposureError("calibration --notes must name the lamps that are on "
                            "(lamps_on=...)")
    path = exposure_path(root, node)
    if os.path.isfile(path) and not recalibrate:
        raise ExposureError("%s exists: one setting per camera; --recalibrate moves it to "
                            "_superseded/ (and is disclosed)" % path)
    region = region or load_wb_region(node)
    g = float(gain if gain is not None else defaults["gain"])
    exp_range, wb_range = defaults["exposure_range"], defaults["white_balance_range"]
    e_min, e_max = float(exp_range["min"]), float(exp_range["max"])
    e0 = _snap(exp_range.get("default") or math.sqrt(max(e_min, 1.0) * e_max),
               e_min, e_max, float(exp_range.get("step") or 1.0))
    wb, wb_trace = search_white_balance(backend, g, e0, wb_range, region, log=log)
    e, e_trace = search_exposure(backend, g, wb, exp_range, region, target=target, log=log)
    if e is None:
        # too dark even at the exposure cap: exposure stays at the cap, gain is raised
        cap = float(exp_range["max"])
        at_cap = [t for t in e_trace if t["setting"]["exposure"] == cap]
        if not at_cap:
            m = measure(backend, {"exposure": cap, "gain": g, "white_balance": wb}, region)
            at_cap = [dict(m, step="exposure")]
            e_trace.append(at_cap[0])
            log("calibration exposure %g (cap): 224 mean luma %.2f"
                % (cap, m["luma_224_median"]))
        if at_cap[-1]["luma_224_median"] < target[0]:
            g_new, g_trace = search_gain(backend, cap, wb, g, defaults["gain_range"], region,
                                         target=target, log=log)
            e_trace += g_trace
            if g_new is not None:
                e, g = cap, g_new
    if e is None:
        raise ExposureError("no exposure (and gain at the exposure cap) gives a 224 mean "
                            "luma in [%g, %g]; trace: %s"
                            % (target[0], target[1],
                               [(t["setting"]["exposure"], t["setting"]["gain"],
                                 t["luma_224_median"]) for t in e_trace]))
    check = measure(backend, {"exposure": e, "gain": g, "white_balance": wb}, region)
    log("verification: exposure %g, gain %g, white balance %g -> 224 mean luma %.2f, "
        "rectangle R,G,B %s, R-B %.2f" % (e, g, wb, check["luma_224_median"],
                                          check["region_rgb_mean"], check["r_minus_b"]))
    if not target[0] <= check["luma_224_median"] <= target[1]:
        raise ExposureError("verification frames left the target: 224 mean luma %.2f"
                            % check["luma_224_median"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        old = os.path.join(os.path.dirname(path), "_superseded")
        os.makedirs(old, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(clock()))
        shutil.move(path, os.path.join(old, "%s_%s" % (stamp, EXPOSURE_FILE)))
    rec = {"policy": "one fixed (exposure, gain, white balance) per camera, AE and AWB off "
                     "(prereg 13.1 draft)",
           "node": node, "serial": (backend.info() or {}).get("serial"),
           "exposure": e, "gain": g, "white_balance": wb, "lamps_on": lamps,
           "luma_target_224": list(target), "wb_region": region,
           "verification": check, "luma_224_median": check["luma_224_median"],
           "exposure_range": exp_range, "white_balance_range": wb_range,
           "gain_range": defaults.get("gain_range"), "gain_default": defaults["gain"],
           "settle_frames": SETTLE_FRAMES, "measure_frames": MEASURE_FRAMES,
           "calibration_notes": notes, "trace": wb_trace + e_trace,
           "determined_unix": clock(), "flame_frames_used": False}
    write_json(path, rec, exclusive=True)
    return rec


def fixed_exposure(root: str, node: str, info: Optional[Dict[str, Any]] = None,
                   notes: Optional[str] = None) -> Dict[str, Any]:
    """The camera's fixed setting for a capture (fire and no-fire alike).

    Raises :class:`ExposureError` when the camera was not calibrated, when the file
    belongs to another camera (serial), or when the capture's ``lamps_on`` differ from
    the calibration's (a lamp change needs a new calibration)."""
    path = exposure_path(root, node)
    if not os.path.isfile(path):
        raise ExposureError("no fixed exposure %s: run --calibrate_exposure on the "
                            "flame-free calibration scene first" % path)
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    serial = (info or {}).get("serial")
    if serial is not None and rec.get("serial") is not None \
            and str(rec["serial"]) != str(serial):
        raise ExposureError("%s was calibrated on camera %s, this is %s"
                            % (path, rec["serial"], serial))
    if notes is not None and rec.get("lamps_on") is not None:
        from src.data.camera_capture_common import parse_notes
        lamps = parse_notes(notes).get("lamps_on")
        if lamps != rec["lamps_on"]:
            raise ExposureError("lamps_on %r differs from the calibration's %r: the lamp "
                                "condition changed, recalibrate" % (lamps, rec["lamps_on"]))
    return {"exposure": rec["exposure"], "gain": rec["gain"],
            "white_balance": rec["white_balance"], "how": "fixed_camera",
            "source": "calibrated %s, 224 mean luma %s" % (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(rec.get("determined_unix", 0))),
                rec.get("luma_224_median")),
            "lock_file": path}
