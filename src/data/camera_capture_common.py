"""FedRGBD -- shared capture machinery of the pre-registered camera experiment.

``docs/CAMERA_EXPERIMENT_PREREG.md`` (sec. 2) fixes what one capture is: one
(scene, label, distance, camera), 40 frames at 5 frames per second, started on all three
nodes at a common scheduled wall-clock time, with per-frame metadata.  The camera
back-ends (``realsense_capture.py`` on node_a/node_b, ``zed_capture.py`` on node_c)
only implement :class:`CameraBackend`; everything that has to be identical on the three
nodes -- ids, validation, scheduling, sampling to 5 fps, the file layout, the metadata
keys, the capture record and the no-overwrite / retake rule -- lives here.

Layout (the data contract; ``src/data/custom_dataset.load_frame_index`` reads it)::

    data/raw/camera/<node>/{frame_id}_rgb.png      8-bit RGB, native resolution
    data/raw/camera/<node>/{frame_id}_depth.png    uint16 mm, aligned to the colour image
    data/raw/camera/<node>/{frame_id}_ir.png       8-bit IR (RealSense only)
    data/raw/camera/<node>/{frame_id}_meta.json    per-frame metadata (has scene, label)
    data/raw/camera/<node>/_captures/<capture_id>.json   capture record, written once
    data/raw/camera/<node>/_retakes/<timestamp>/   superseded captures (never deleted)

    capture_id = f"{scene}_{label}_d{int(round(distance_m * 100))}"   e.g. s03_no_fire_d200
    frame_id   = f"{capture_id}_{frame_index:04d}"

Python 3.10 / numpy 1.26 (Jetson).  No camera SDK is imported here.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import socket
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# the contract's constants
# --------------------------------------------------------------------------- #
DEFAULT_ROOT = os.path.join("data", "raw", "camera")
NODE_SENSORS: Dict[str, str] = {
    "node_a": "RealSense D435if",
    "node_b": "RealSense D435i",
    "node_c": "ZED 2i",
}
NODE_NAMES = tuple(NODE_SENSORS)
LABELS = ("fire", "no_fire")
DISTANCES_M = (1.0, 2.0, 3.0)
SCENE_RE = re.compile(r"^s\d{2}$")
#: pre-registration sec. 2: 40 frames per (scene, class, distance, camera) at 5 fps
DEFAULT_FRAMES = 40
DEFAULT_FPS = 5.0
#: verification: every node must start within this many seconds of the schedule
START_TOLERANCE_S = 1.0
CAPTURES_DIR = "_captures"
RETAKES_DIR = "_retakes"

#: keys every {frame_id}_meta.json carries (tests pin this list)
FRAME_META_KEYS = (
    "frame_id", "capture_id", "scene", "label", "distance_m", "frame_index", "node",
    "camera_model", "serial", "firmware", "sdk_version", "timestamp_unix",
    "timestamp_device", "exposure", "gain", "white_balance", "auto_exposure",
    "depth_mode", "rgb_resolution", "fps",
)
#: keys every _captures/<capture_id>.json carries
CAPTURE_RECORD_KEYS = (
    "capture_id", "scene", "label", "distance_m", "node", "camera_model", "serial",
    "firmware", "sdk_version", "depth_mode", "fps", "n_frames_requested",
    "n_frames_written", "start_scheduled_unix", "start_actual_unix", "end_unix", "host",
    "notes",
)


class CaptureError(RuntimeError):
    """A capture cannot (or must not) be written."""


class CaptureExistsError(CaptureError):
    """The capture already exists; a retake must be requested explicitly."""


# --------------------------------------------------------------------------- #
# ids and validation
# --------------------------------------------------------------------------- #
def validate_scene(scene: str) -> str:
    if not isinstance(scene, str) or not SCENE_RE.match(scene):
        raise ValueError("scene must match ^s\\d{2}$ (e.g. s01), got %r" % (scene,))
    return scene


def validate_label(label: str) -> str:
    if label not in LABELS:
        raise ValueError("label must be exactly one of %s, got %r" % (LABELS, label))
    return label


def validate_distance(distance_m) -> float:
    try:
        d = float(distance_m)
    except (TypeError, ValueError):
        raise ValueError("distance_m must be one of %s, got %r" % (DISTANCES_M, distance_m))
    for allowed in DISTANCES_M:
        if abs(d - allowed) < 1e-9:
            return allowed
    raise ValueError("distance_m must be one of %s, got %r" % (DISTANCES_M, distance_m))


def validate_node(node: str) -> str:
    if node not in NODE_SENSORS:
        raise ValueError("node must be one of %s, got %r" % (NODE_NAMES, node))
    return node


def make_capture_id(scene: str, label: str, distance_m) -> str:
    """``s03_no_fire_d200`` from (s03, no_fire, 2.0); validates all three parts."""
    scene = validate_scene(scene)
    label = validate_label(label)
    d = validate_distance(distance_m)
    return "%s_%s_d%d" % (scene, label, int(round(d * 100)))


def make_frame_id(capture_id: str, frame_index: int) -> str:
    return "%s_%04d" % (capture_id, int(frame_index))


def _frame_file_re(capture_id: str):
    return re.compile(r"^%s_\d{4}_(rgb|depth|ir)\.png$|^%s_\d{4}_meta\.json$"
                      % (re.escape(capture_id), re.escape(capture_id)))


# --------------------------------------------------------------------------- #
# scheduling and sampling
# --------------------------------------------------------------------------- #
def wait_until(start_at_unix: Optional[float],
               clock: Callable[[], float] = time.time,
               sleep: Callable[[float], None] = time.sleep,
               idle: Optional[Callable[[], None]] = None,
               idle_margin_s: float = 0.25,
               max_sleep_s: float = 0.5) -> float:
    """Block until the host wall clock reaches ``start_at_unix``; -> lateness in s.

    ``idle`` (e.g. grab-and-discard a frame, so the camera keeps streaming and no stale
    frame is queued at the start) is called repeatedly while more than
    ``idle_margin_s`` remain; the last stretch is slept.  Returns ``clock() - start``
    at the moment the wait ends (>= 0 when on time, larger when the command arrived
    or the camera opened late).  ``None`` means "start now" and returns 0.0.
    """
    if start_at_unix is None:
        return 0.0
    start = float(start_at_unix)
    while True:
        remaining = start - clock()
        if remaining <= 0:
            break
        if idle is not None and remaining > idle_margin_s:
            idle()
            continue
        sleep(min(remaining, max_sleep_s))
    return clock() - start


class FrameSampler:
    """Down-sample a camera stream to ``fps`` by time.

    Frame k is due at ``t0 + k / fps`` (t0 = timestamp of the first accepted frame); a
    frame is accepted when its timestamp has reached the next due time less a jitter
    tolerance: half a period of the camera stream when ``stream_fps`` is known (so the
    stream frame nearest the due time is kept, never the one before it), otherwise a
    quarter of the target period.  A 5 fps stream therefore keeps every frame; a 15 or
    30 fps stream keeps every 3rd / 6th.
    """

    def __init__(self, fps: float, stream_fps: Optional[float] = None):
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.period_ms = 1000.0 / float(fps)
        if stream_fps:
            self.tol_ms = 0.5 * min(self.period_ms, 1000.0 / float(stream_fps))
        else:
            self.tol_ms = 0.25 * self.period_ms
        self.t0: Optional[float] = None
        self.k = 0

    def accept(self, ts_ms: float) -> bool:
        ts_ms = float(ts_ms)
        if self.t0 is None:
            self.t0 = ts_ms
            self.k = 1
            return True
        due = self.t0 + self.k * self.period_ms
        if ts_ms >= due - self.tol_ms:
            # skip due slots a dropped frame missed, so the schedule stays anchored
            while ts_ms >= self.t0 + self.k * self.period_ms - self.tol_ms:
                self.k += 1
            return True
        return False


# --------------------------------------------------------------------------- #
# camera back-end interface
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    """One grabbed frame.  ``rgb`` HxWx3 uint8 RGB; ``depth_mm`` HxW uint16 aligned to
    ``rgb``; ``ir`` HxW uint8 or None; ``device_ts_ms`` the device timestamp in ms;
    ``meta`` per-frame camera settings (exposure, gain, white_balance, auto_exposure,
    plus any back-end specific extras)."""

    rgb: np.ndarray
    depth_mm: np.ndarray
    ir: Optional[np.ndarray]
    device_ts_ms: Optional[float]
    meta: Dict[str, Any] = field(default_factory=dict)


class CameraBackend:
    """What a camera back-end implements.  Tests inject a fake one.

    ``info()`` (valid after ``open()``) returns at least ``camera_model, serial,
    firmware, sdk_version, depth_mode, rgb_resolution [w, h], stream_fps`` and the
    calibration of prereg Amendment 3 (:func:`calibration_problems`): ``intrinsics``
    (``rgb`` and ``depth``: width, height, fx, fy, ppx, ppy, distortion model and
    coefficients), ``stereo_baseline_mm``, ``depth_scale_mm`` (millimetres per raw depth
    unit) and, where the SDK sets one, ``depth_range_mm``.
    """

    def open(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def grab(self) -> Frame:  # pragma: no cover - interface
        raise NotImplementedError

    def skip(self) -> None:
        """Grab and discard one frame (warm-up and while waiting for the start)."""
        self.grab()

    def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def settle_auto(self, n_frames: int) -> Dict[str, Any]:  # pragma: no cover - interface
        """Auto exposure and auto white balance on, ``n_frames`` grabbed and discarded,
        then -> {exposure, gain, white_balance, source}: the values to lock."""
        raise NotImplementedError

    def apply_lock(self, lock: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        """Auto exposure and auto white balance off, the locked values set; -> the values
        the camera reports afterwards (read back)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def depth_to_mm_uint16(depth, scale_to_mm: float = 1.0) -> np.ndarray:
    """Float/int depth -> uint16 millimetres; NaN, +-inf and negatives -> 0."""
    d = np.asarray(depth, dtype=np.float64) * float(scale_to_mm)
    d = np.where(np.isfinite(d) & (d > 0), d, 0.0)
    return np.clip(np.rint(d), 0, 65535).astype(np.uint16)


def to_jsonable(value):
    """numpy scalars/arrays and odd SDK values -> plain JSON types."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def write_json(path: str, obj, exclusive: bool = False) -> None:
    """Write JSON atomically (tmp + rename); ``exclusive`` refuses an existing file."""
    text = json.dumps(to_jsonable(obj), indent=2, sort_keys=True)
    if exclusive:
        with open(path, "x", encoding="utf-8") as f:
            f.write(text)
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_png(path: str, arr: np.ndarray) -> None:
    """uint8 HxWx3 / uint8 HxW / uint16 HxW array -> PNG (16 bit kept for depth)."""
    from PIL import Image

    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.uint16 and arr.ndim == 2:
        img = Image.fromarray(arr)  # mode I;16
    elif arr.dtype == np.uint8 and arr.ndim in (2, 3):
        img = Image.fromarray(arr)
    else:
        raise ValueError("unsupported image %s %s for %s" % (arr.dtype, arr.shape, path))
    img.save(path, format="PNG", compress_level=1)


def write_frame(node_dir: str, frame_id: str, frame: Frame, meta: Dict[str, Any]) -> None:
    """The contract's per-frame files: _rgb.png, _depth.png, [_ir.png], _meta.json."""
    rgb = np.asarray(frame.rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 uint8, got %s %s" % (rgb.dtype, rgb.shape))
    depth = np.asarray(frame.depth_mm)
    if depth.dtype != np.uint16:
        depth = depth_to_mm_uint16(depth)
    if depth.shape != rgb.shape[:2]:
        raise ValueError("depth %s is not aligned to rgb %s" % (depth.shape, rgb.shape[:2]))
    write_png(os.path.join(node_dir, frame_id + "_rgb.png"), rgb)
    write_png(os.path.join(node_dir, frame_id + "_depth.png"), depth)
    if frame.ir is not None:
        write_png(os.path.join(node_dir, frame_id + "_ir.png"), np.asarray(frame.ir, np.uint8))
    write_json(os.path.join(node_dir, frame_id + "_meta.json"), meta)


# --------------------------------------------------------------------------- #
# no-overwrite / retake
# --------------------------------------------------------------------------- #
def capture_paths(root: str, node: str, capture_id: str) -> Dict[str, str]:
    node_dir = os.path.join(root, node)
    return {
        "node_dir": node_dir,
        "captures_dir": os.path.join(node_dir, CAPTURES_DIR),
        "record": os.path.join(node_dir, CAPTURES_DIR, capture_id + ".json"),
        "retakes_dir": os.path.join(node_dir, RETAKES_DIR),
    }


def existing_capture_files(node_dir: str, capture_id: str) -> List[str]:
    """Frame files of ``capture_id`` present in ``node_dir`` (sorted names)."""
    if not os.path.isdir(node_dir):
        return []
    pat = _frame_file_re(capture_id)
    return sorted(f for f in os.listdir(node_dir) if pat.match(f))


def move_to_retakes(root: str, node: str, capture_id: str,
                    clock: Callable[[], float] = time.time) -> str:
    """Move the frames and record of ``capture_id`` to ``_retakes/<timestamp>/``.

    Never deletes.  Raises CaptureError when there is nothing to move (a retake of a
    capture that does not exist is almost certainly a typo in the id).
    """
    p = capture_paths(root, node, capture_id)
    frames = existing_capture_files(p["node_dir"], capture_id)
    has_record = os.path.isfile(p["record"])
    if not frames and not has_record:
        raise CaptureError("--retake given but %s/%s has no frames and no record"
                           % (node, capture_id))
    stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(clock()))
    dest = os.path.join(p["retakes_dir"], stamp)
    n = 1
    while os.path.exists(dest):
        n += 1
        dest = os.path.join(p["retakes_dir"], "%s_%d" % (stamp, n))
    os.makedirs(dest)
    for name in frames:
        shutil.move(os.path.join(p["node_dir"], name), os.path.join(dest, name))
    if has_record:
        shutil.move(p["record"], os.path.join(dest, capture_id + ".json"))
    return dest


def prepare_capture(root: str, node: str, capture_id: str, retake: bool = False,
                    clock: Callable[[], float] = time.time) -> Dict[str, Optional[str]]:
    """Refuse to overwrite; with ``retake`` move the old capture aside first.

    ``retake`` with nothing to move is allowed (a session re-run with --retake after a
    node failed to open its camera must not fail on that node); the record then has
    ``retake_moved_to`` None.
    """
    p = capture_paths(root, node, capture_id)
    frames = existing_capture_files(p["node_dir"], capture_id)
    exists = os.path.isfile(p["record"]) or bool(frames)
    moved = None
    if exists and not retake:
        raise CaptureExistsError(
            "%s/%s already exists (%s, %d frame file(s)); a capture is written once. "
            "Pass --retake to move it to %s/<timestamp>/ and capture again."
            % (node, capture_id, "record present" if os.path.isfile(p["record"])
               else "no record", len(frames), os.path.join(node, RETAKES_DIR)))
    if retake and exists:
        moved = move_to_retakes(root, node, capture_id, clock=clock)
    os.makedirs(p["captures_dir"], exist_ok=True)
    return {"record": p["record"], "node_dir": p["node_dir"], "retake_moved_to": moved}


# --------------------------------------------------------------------------- #
# one capture
# --------------------------------------------------------------------------- #
#: intrinsics fields every stream must carry (prereg Amendment 3)
INTRINSIC_FIELDS = ("width", "height", "fx", "fy", "ppx", "ppy", "model", "coeffs")
#: a stereo baseline outside this range is a unit error, not a camera (D435: 50, ZED 2i: 120)
BASELINE_PLAUSIBLE_MM = (20.0, 500.0)


def calibration_problems(info: Dict[str, Any]) -> List[str]:
    """What the capture would fail to record of prereg Amendment 3 ([] = complete):
    rgb and depth intrinsics with distortion, a plausible stereo baseline in mm, the depth
    scale, and the SDK and firmware versions."""
    problems = []
    intr = info.get("intrinsics") or {}
    for stream in ("rgb", "depth"):
        d = intr.get(stream) or {}
        missing = [k for k in INTRINSIC_FIELDS if d.get(k) is None]
        if missing:
            problems.append("%s intrinsics missing %s" % (stream, ", ".join(missing)))
    b = info.get("stereo_baseline_mm")
    lo, hi = BASELINE_PLAUSIBLE_MM
    if not isinstance(b, (int, float)) or not math.isfinite(b) or not lo <= b <= hi:
        problems.append("stereo_baseline_mm %r not recorded or outside %g-%g mm" % (b, lo, hi))
    sc = info.get("depth_scale_mm")
    if not isinstance(sc, (int, float)) or not math.isfinite(sc) or sc <= 0:
        problems.append("depth_scale_mm %r not recorded" % (sc,))
    for k in ("sdk_version", "firmware"):
        if not info.get(k):
            problems.append("%s not recorded" % k)
    return problems


# --------------------------------------------------------------------------- #
# study-capture gates (prereg sec. 13 draft, Amendment 4; e5 of POST_5B_CHECKLIST)
# --------------------------------------------------------------------------- #
#: fields every study capture's --notes must carry ("key=value; key=value; ...")
NOTES_REQUIRED = ("source", "distance_measured_m", "flame_height_cm")
NOTES_FORMAT = "source=<id|none>; distance_measured_m=<x.xx>; flame_height_cm=<x>"
#: measured distances outside this range are a typo, not a rig
MEASURED_DISTANCE_RANGE_M = (0.3, 10.0)
#: a ZED on a USB 3.x link runs at >= 5000 Mb/s (sysfs ``speed``)
USB3_MIN_SPEED_MBPS = 5000
ZED_USB_VENDOR = "2b03"
LOCKS_DIR = "_locks"


def parse_notes(notes: Optional[str]) -> Dict[str, str]:
    """``"a=1; b=x y"`` -> {"a": "1", "b": "x y"}; fragments without ``=`` go to "text"."""
    out: Dict[str, str] = {}
    text = []
    for part in (notes or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            text.append(part)
    if text:
        out["text"] = "; ".join(text)
    return out


def notes_problems(notes: Optional[str], label: str) -> List[str]:
    """What a study capture's --notes lacks ([] = complete).  Fire: one lit source and a
    measured flame height > 0; no-fire: ``source=none`` and ``flame_height_cm=0``."""
    f = parse_notes(notes)
    problems = ["--notes lacks %s" % k for k in NOTES_REQUIRED if not f.get(k)]
    if problems:
        return problems + ["--notes format: %s" % NOTES_FORMAT]
    try:
        d = float(f["distance_measured_m"])
        lo, hi = MEASURED_DISTANCE_RANGE_M
        if not lo <= d <= hi:
            problems.append("distance_measured_m %s outside %g-%g m" % (f["distance_measured_m"], lo, hi))
    except ValueError:
        problems.append("distance_measured_m %r is not a number" % f["distance_measured_m"])
    try:
        h = float(f["flame_height_cm"])
    except ValueError:
        h = None
        problems.append("flame_height_cm %r is not a number" % f["flame_height_cm"])
    source = f["source"]
    if label == "fire":
        if source == "none":
            problems.append("fire capture with source=none")
        if h is not None and not h > 0:
            problems.append("fire capture needs flame_height_cm > 0, got %s" % f["flame_height_cm"])
    elif label == "no_fire":
        if source != "none":
            problems.append("no-fire capture needs source=none, got %r" % source)
        if h is not None and h != 0:
            problems.append("no-fire capture needs flame_height_cm=0, got %s" % f["flame_height_cm"])
    return problems


def _usb_major(usb_type) -> Optional[int]:
    m = re.match(r"\s*(\d+)", str(usb_type)) if usb_type is not None else None
    return int(m.group(1)) if m else None


def usb_problems(info: Dict[str, Any]) -> List[str]:
    """USB 3.x gate ([] = pass): a RealSense must report ``usb_type`` 3.x, the ZED a link
    speed of at least 5000 Mb/s (``usb_speed_mbps``).  Not recorded = fail."""
    speed = info.get("usb_speed_mbps")
    if speed is not None:
        if float(speed) < USB3_MIN_SPEED_MBPS:
            return ["USB link %s Mb/s, need >= %d (USB 3.x)" % (speed, USB3_MIN_SPEED_MBPS)]
        return []
    usb_type = info.get("usb_type")
    major = _usb_major(usb_type)
    if major is None:
        return ["USB type not recorded (usb_type %r, usb_speed_mbps None)" % (usb_type,)]
    if major < 3:
        return ["USB type %s, need 3.x" % usb_type]
    return []


def zed_usb_devices(sys_root: str = "/sys/bus/usb/devices") -> List[Dict[str, Any]]:
    """Every Stereolabs USB device in sysfs: id, product id, product name, speed (Mb/s).
    The ZED 2i shows up as more than one device (camera, sensors); all are recorded."""
    out = []
    if not os.path.isdir(sys_root):
        return out
    for name in sorted(os.listdir(sys_root)):
        d = os.path.join(sys_root, name)

        def read(key, d=d):
            try:
                with open(os.path.join(d, key), encoding="utf-8") as f:
                    return f.read().strip()
            except OSError:
                return None
        if read("idVendor") != ZED_USB_VENDOR:
            continue
        speed = read("speed")
        try:
            speed_v = float(speed) if speed is not None else None
        except ValueError:
            speed_v = None
        out.append({"device": name, "id_product": read("idProduct"), "product": read("product"),
                    "speed_mbps": speed_v})
    return out


def zed_usb_speed_mbps(devices: List[Dict[str, Any]]) -> Optional[float]:
    """The fastest link among the ZED's USB devices (the camera's video interface; the
    sensor interface may sit at a lower speed behind the camera's own hub)."""
    speeds = [d["speed_mbps"] for d in devices if d.get("speed_mbps") is not None]
    return max(speeds) if speeds else None


LOCK_KEYS = ("exposure", "gain", "white_balance")


def lock_path(root: str, node: str, scene: str, distance_m) -> str:
    """``<root>/<node>/_locks/<scene>_d<cm>.json``: one exposure lock per scene x distance."""
    d = validate_distance(distance_m)
    return os.path.join(root, node, LOCKS_DIR, "%s_d%d.json" % (validate_scene(scene),
                                                               int(round(d * 100))))


def resolve_lock(backend: "CameraBackend", root: str, node: str, scene: str, label: str,
                 distance_m, settle_frames: int,
                 clock: Callable[[], float] = time.time) -> Dict[str, Any]:
    """The exposure lock of this capture.

    No-fire: auto exposure and auto white balance settle on the no-fire setup
    (``backend.settle_auto``), their values become the lock and are written to
    :func:`lock_path`.  If the fire capture of the same scene x distance already exists,
    the stored lock is reused instead, so the pair keeps identical values.  Fire: the
    stored lock is required and reused unchanged.  Returns
    {exposure, gain, white_balance, source, how, lock_file}.
    """
    path = lock_path(root, node, scene, distance_m)
    fire_rec = capture_paths(root, node, make_capture_id(scene, "fire", distance_m))["record"]
    if label == "fire" or os.path.isfile(fire_rec):
        if not os.path.isfile(path):
            raise CaptureError("no exposure lock %s: capture the no-fire setup of this scene "
                               "and distance first (its lock is reused for fire)" % path)
        with open(path, encoding="utf-8") as f:
            lock = json.load(f)
        lock["how"] = "reused"
    else:
        lock = dict(backend.settle_auto(int(settle_frames)))
        missing = [k for k in LOCK_KEYS if lock.get(k) is None]
        if missing:
            raise CaptureError("auto exposure did not report %s; no lock" % ", ".join(missing))
        lock.update(scene=scene, distance_m=validate_distance(distance_m), node=node,
                    determined_unix=clock())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isfile(path):  # a no-fire retake before any fire capture: keep the old one
            stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(clock()))
            old = os.path.join(os.path.dirname(path), "_superseded")
            os.makedirs(old, exist_ok=True)
            shutil.move(path, os.path.join(old, "%s_%s" % (stamp, os.path.basename(path))))
        write_json(path, lock, exclusive=True)
        lock["how"] = "determined"
    lock["lock_file"] = path
    return lock


def lock_mismatch(lock: Dict[str, Any], applied: Dict[str, Any],
                  rel_tol: float = 0.02) -> List[str]:
    """Locked values the camera did not take (read back after applying)."""
    out = []
    for k in LOCK_KEYS:
        want, got = lock.get(k), applied.get(k)
        if got is None or want is None or abs(float(got) - float(want)) > rel_tol * max(1.0, abs(float(want))):
            out.append("%s: set %s, camera reports %s" % (k, want, got))
    return out


def run_capture(backend: CameraBackend, scene: str, label: str, distance_m, node: str,
                root: str = DEFAULT_ROOT, frames: int = DEFAULT_FRAMES,
                fps: float = DEFAULT_FPS, start_at: Optional[float] = None,
                retake: bool = False, notes: str = "", warmup_frames: int = 15,
                timeout_s: Optional[float] = None,
                clock: Callable[[], float] = time.time,
                sleep: Callable[[float], None] = time.sleep,
                log: Callable[[str], None] = print,
                study_gates: bool = False, exposure_lock: bool = False,
                lock_settle_frames: int = 45) -> Dict[str, Any]:
    """Open the camera, wait for ``start_at``, keep ``frames`` frames at ``fps``.

    Writes every frame and, once the camera has opened, the capture record -- also on
    failure, with the frames actually written and the error in ``notes``, so that an
    incomplete capture is visible and can only be replaced with ``--retake``.
    Returns the record.

    ``study_gates`` (every study capture; prereg sec. 13 draft): the --notes must parse
    (:func:`notes_problems`, checked before the camera opens) and the camera must be on a
    USB 3.x link (:func:`usb_problems`).  ``exposure_lock``: exposure, gain and white
    balance fixed per scene x distance (:func:`resolve_lock`), and the capture FAILs if
    any colour frame reports auto exposure on (or does not report it).
    """
    node = validate_node(node)
    capture_id = make_capture_id(scene, label, distance_m)
    distance_m = validate_distance(distance_m)
    frames = int(frames)
    fps = float(fps)
    if frames < 1 or fps <= 0:
        raise ValueError("frames must be >= 1 and fps > 0")
    if study_gates:
        bad = notes_problems(notes, label)
        if bad:
            raise CaptureError("capture refused before it started: %s" % "; ".join(bad))
    prep = prepare_capture(root, node, capture_id, retake=retake, clock=clock)
    node_dir = prep["node_dir"]

    backend.open()
    info: Dict[str, Any] = {}
    record: Dict[str, Any] = {}
    written = 0
    start_actual = None
    lateness = None
    errors: List[str] = []
    futures: list = []
    lock: Optional[Dict[str, Any]] = None
    ae = {"on": 0, "unknown": 0}
    try:
        info = dict(backend.info())
        missing = calibration_problems(info)
        if missing:
            raise CaptureError("calibration not recorded, capture refused (prereg "
                               "Amendment 3): %s" % "; ".join(missing))
        if study_gates:
            bad = usb_problems(info)
            if bad:
                raise CaptureError("USB 3.x gate, capture refused: %s" % "; ".join(bad))
        if exposure_lock:
            lock = resolve_lock(backend, root, node, scene, label, distance_m,
                                lock_settle_frames, clock=clock)
            applied = dict(backend.apply_lock({k: lock[k] for k in LOCK_KEYS}))
            lock["applied"] = applied
            bad = lock_mismatch(lock, applied)
            if bad:
                raise CaptureError("exposure lock not applied: %s" % "; ".join(bad))
            log("exposure lock (%s): exposure %s, gain %s, white balance %s"
                % (lock["how"], lock["exposure"], lock["gain"], lock["white_balance"]))
        for _ in range(max(0, int(warmup_frames))):
            backend.skip()
        lateness = wait_until(start_at, clock=clock, sleep=sleep, idle=backend.skip)
        if start_at is not None and lateness > START_TOLERANCE_S:
            log("WARNING: started %.3f s after the scheduled time" % lateness)
        sampler = FrameSampler(fps, info.get("stream_fps"))
        limit = timeout_s if timeout_s is not None else max(10.0, 3.0 * frames / fps + 5.0)
        loop_start = clock()
        with ThreadPoolExecutor(max_workers=2) as pool:
            index = 0
            while index < frames:
                if clock() - loop_start > limit:
                    errors.append("timeout: %d of %d frames after %.1f s"
                                  % (index, frames, limit))
                    break
                fr = backend.grab()
                t_host = clock()
                ts = fr.device_ts_ms if fr.device_ts_ms is not None else t_host * 1000.0
                if not sampler.accept(ts):
                    continue
                if start_actual is None:
                    start_actual = t_host
                fid = make_frame_id(capture_id, index)
                meta = frame_meta(fid, capture_id, scene, label, distance_m, index, node,
                                  info, fr, t_host, fps)
                if lock is not None:
                    meta["exposure_source"] = "manual_lock"
                    meta["exposure_lock"] = {k: lock[k] for k in LOCK_KEYS}
                    flag = (fr.meta or {}).get("auto_exposure")
                    if flag is None:
                        ae["unknown"] += 1
                    elif flag:
                        ae["on"] += 1
                futures.append(pool.submit(write_frame, node_dir, fid, fr, meta))
                index += 1
        if lock is not None and (ae["on"] or ae["unknown"]):
            errors.append("exposure lock gate: auto exposure on in %d and unreported in %d "
                          "of %d colour frames" % (ae["on"], ae["unknown"], index))
    except Exception as e:  # noqa: BLE001 - recorded in the capture record, re-raised
        errors.append("%s: %s" % (type(e).__name__, e))
        raise
    finally:
        # the pool has joined here (also on an exception): count what reached disk
        for fut in futures:
            exc = fut.exception()
            if exc is None:
                written += 1
            else:
                errors.append("write failed: %s" % exc)
        try:
            backend.close()
        except Exception as e:  # noqa: BLE001
            errors.append("close failed: %s" % e)
        record = capture_record(capture_id, scene, label, distance_m, node, info, fps,
                                frames, written, start_at, start_actual, clock(),
                                notes, errors, lateness, prep.get("retake_moved_to"))
        record["notes_parsed"] = parse_notes(notes)
        record["study_gates"] = bool(study_gates)
        record["usb"] = {"usb_type": info.get("usb_type"),
                         "usb_speed_mbps": info.get("usb_speed_mbps"),
                         "usb_devices": info.get("usb_devices"),
                         "problems": usb_problems(info) if info else None}
        record["exposure_lock"] = lock
        record["ae_on_frames"] = ae["on"] if lock is not None else None
        record["ae_unknown_frames"] = ae["unknown"] if lock is not None else None
        write_json(prep["record"], record, exclusive=True)
    return record


def frame_meta(frame_id, capture_id, scene, label, distance_m, frame_index, node, info,
               frame: Frame, t_host, fps) -> Dict[str, Any]:
    m = frame.meta or {}
    meta = {
        "frame_id": frame_id,
        "capture_id": capture_id,
        "scene": scene,
        "label": label,
        "distance_m": distance_m,
        "frame_index": frame_index,
        "node": node,
        "sensor": NODE_SENSORS[node],
        "camera_model": info.get("camera_model"),
        "serial": info.get("serial"),
        "firmware": info.get("firmware"),
        "sdk_version": info.get("sdk_version"),
        "timestamp_unix": t_host,
        "timestamp_device": frame.device_ts_ms,
        "exposure": m.get("exposure"),
        "gain": m.get("gain"),
        "white_balance": m.get("white_balance"),
        "auto_exposure": m.get("auto_exposure"),
        "depth_mode": info.get("depth_mode"),
        "rgb_resolution": [int(frame.rgb.shape[1]), int(frame.rgb.shape[0])],
        "fps": fps,
        "stream_fps": info.get("stream_fps"),
    }
    for k, v in m.items():  # back-end extras (timestamp domain, exposure source, ...)
        meta.setdefault(k, v)
    return meta


def capture_record(capture_id, scene, label, distance_m, node, info, fps, n_requested,
                   n_written, start_scheduled, start_actual, end_unix, notes, errors,
                   lateness, retake_moved_to) -> Dict[str, Any]:
    note = notes or ""
    if errors:
        note = (note + " | " if note else "") + "; ".join(errors)
    rec = {
        "capture_id": capture_id,
        "scene": scene,
        "label": label,
        "distance_m": distance_m,
        "node": node,
        "sensor": NODE_SENSORS[node],
        "camera_model": info.get("camera_model"),
        "serial": info.get("serial"),
        "firmware": info.get("firmware"),
        "sdk_version": info.get("sdk_version"),
        "depth_mode": info.get("depth_mode"),
        "rgb_resolution": info.get("rgb_resolution"),
        "stream_fps": info.get("stream_fps"),
        "intrinsics": info.get("intrinsics"),
        "stereo_baseline_mm": info.get("stereo_baseline_mm"),
        "depth_scale_mm": info.get("depth_scale_mm"),
        "depth_range_mm": info.get("depth_range_mm"),
        "extrinsics_depth_to_color": info.get("extrinsics_depth_to_color"),
        "fps": fps,
        "n_frames_requested": int(n_requested),
        "n_frames_written": int(n_written),
        "start_scheduled_unix": start_scheduled,
        "start_actual_unix": start_actual,
        "start_wait_lateness_s": lateness,
        "end_unix": end_unix,
        "host": socket.gethostname(),
        "notes": note,
        "status": "complete" if (n_written == n_requested and not errors) else "incomplete",
        "retake_moved_to": retake_moved_to,
        "backend_info": info,
    }
    return rec


def smoke_test(backend: CameraBackend, node: str, frames: int = 5,
               fps: float = DEFAULT_FPS, log: Callable[[str], None] = print) -> bool:
    """``--test``: a few frames through the real capture path into a temp dir."""
    tmp = tempfile.mkdtemp(prefix="fedrgbd_camtest_")
    rec = run_capture(backend, "s00", "no_fire", 1.0, node, root=tmp, frames=frames,
                      fps=fps, warmup_frames=5, notes="smoke test", log=log)
    log("Wrote %d/%d frames to %s" % (rec["n_frames_written"], frames,
                                       os.path.join(tmp, node)))
    for k in ("camera_model", "serial", "firmware", "sdk_version", "depth_mode",
              "rgb_resolution", "stream_fps", "stereo_baseline_mm", "depth_scale_mm",
              "depth_range_mm"):
        log("  %-18s %s" % (k, rec.get(k)))
    for stream, d in sorted((rec.get("intrinsics") or {}).items()):
        log("  intrinsics %-7s %sx%s fx=%s fy=%s ppx=%s ppy=%s %s %s"
            % (stream, d.get("width"), d.get("height"), d.get("fx"), d.get("fy"),
               d.get("ppx"), d.get("ppy"), d.get("model"), d.get("coeffs")))
    first = os.path.join(tmp, node, make_frame_id(rec["capture_id"], 0) + "_meta.json")
    if os.path.isfile(first):
        with open(first, encoding="utf-8") as f:
            m = json.load(f)
        log("  frame 0 exposure=%s gain=%s white_balance=%s auto_exposure=%s"
            % (m.get("exposure"), m.get("gain"), m.get("white_balance"),
               m.get("auto_exposure")))
    ok = rec["status"] == "complete"
    log("Test %s" % ("PASSED" if ok else "FAILED: " + rec["notes"]))
    return ok


# --------------------------------------------------------------------------- #
# shared CLI
# --------------------------------------------------------------------------- #
def add_capture_args(parser, default_node: str, frames_default=DEFAULT_FRAMES,
                     fps_default=DEFAULT_FPS) -> None:
    """The capture flags both camera CLIs share."""
    parser.add_argument("--scene", help="scene id, s01, s02, ...")
    parser.add_argument("--label", choices=LABELS, help="declared class of the capture")
    parser.add_argument("--distance_m", type=float, help="rig-to-flame distance, 1/2/3 m")
    parser.add_argument("--frames", type=int, default=frames_default,
                        help="frames to keep (default %(default)s)")
    parser.add_argument("--fps", type=float, default=fps_default,
                        help="frames per second to keep (default %(default)s)")
    parser.add_argument("--start_at", type=float, default=None,
                        help="scheduled start, unix seconds (host wall clock)")
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="capture root (default %(default)s)")
    parser.add_argument("--node", default=default_node, choices=NODE_NAMES)
    parser.add_argument("--retake", action="store_true",
                        help="move an existing capture to _retakes/<timestamp>/ first")
    parser.add_argument("--notes", default="", help="required, parsed: " + NOTES_FORMAT)
    parser.add_argument("--no_lock", action="store_true",
                        help="TEST ONLY (e5): no exposure lock; never a study capture")
    parser.add_argument("--lock_settle_frames", type=int, default=45,
                        help="frames auto exposure runs before the no-fire lock is read")


def probe_sdks() -> Dict[str, Any]:
    """Which camera SDK imports here and how many cameras it sees (``--probe``)."""
    out: Dict[str, Any] = {"python": sys.version.split()[0], "host": socket.gethostname()}
    try:
        import pyrealsense2 as rs  # noqa: F401
        devs = rs.context().query_devices()
        cams = []
        for d in devs:
            try:
                usb_type = d.get_info(rs.camera_info.usb_type_descriptor)
            except Exception:  # noqa: BLE001
                usb_type = None
            cams.append({"name": d.get_info(rs.camera_info.name),
                         "serial": d.get_info(rs.camera_info.serial_number),
                         "usb_type": usb_type})
        out["pyrealsense2"] = {"import": True, "cameras": cams}
    except Exception as e:  # noqa: BLE001
        out["pyrealsense2"] = {"import": False, "error": "%s: %s" % (type(e).__name__, e)}
    try:
        import pyzed.sl as sl
        devs = sl.Camera.get_device_list()
        usb = zed_usb_devices()
        out["pyzed"] = {"import": True, "sdk_version": str(sl.Camera.get_sdk_version()),
                        "cameras": [{"model": str(d.camera_model),
                                     "serial": int(d.serial_number),
                                     "usb_speed_mbps": zed_usb_speed_mbps(usb)} for d in devs],
                        "usb_devices": usb}
    except Exception as e:  # noqa: BLE001
        out["pyzed"] = {"import": False, "error": "%s: %s" % (type(e).__name__, e)}
    out["time_unix"] = time.time()
    return out


if __name__ == "__main__":  # pragma: no cover - run on the nodes by the orchestrator
    if "--probe" in sys.argv[1:]:
        print(json.dumps(probe_sdks()))
    else:
        print("usage: python src/data/camera_capture_common.py --probe")
