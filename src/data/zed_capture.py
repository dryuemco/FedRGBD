"""FedRGBD -- ZED 2i capture for the pre-registered camera experiment (node_c).

Writes the data contract of ``src/data/camera_capture_common.py``: per frame the left
camera's RGB (the colour image), the depth of the left camera in millimetres (so it is
aligned to the RGB by construction; NaN / inf -> 0) and the metadata.  The ZED 2i has
no IR stream.  Depth mode NEURAL and HD1080 by default, both recorded; depth range set to
0.3-20 m as in v1, and intrinsics, distortion, stereo baseline, depth scale, SDK and
firmware recorded at every capture (prereg Amendment 3).

Usage (inside the node's venv, from the repo root)::

    python src/data/zed_capture.py --list
    python src/data/zed_capture.py --test
    python src/data/zed_capture.py --scene s01 --label fire --distance_m 2 \\
        --start_at 1790000000.0            # normally started by camera_capture_session.py

pyzed (ZED SDK 5.2.3) is imported lazily, so the module imports without it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.data.camera_capture_common import (  # noqa: E402
    CameraBackend, CaptureError, Frame, add_capture_args, depth_to_mm_uint16,
    run_capture, smoke_test, zed_usb_devices, zed_usb_speed_mbps,
)

DEFAULT_DEPTH_MODE = "NEURAL"
DEFAULT_RESOLUTION = "HD1080"
#: depth range set explicitly, as the v1 captures did (0.3-20 m; prereg Amendment 3), in
#: the capture's coordinate units (MILLIMETER) -- never the SDK default
DEPTH_RANGE_MM = (300.0, 20000.0)
#: ZED 2i at HD1080 streams at 15 or 30 fps; the capture keeps 5 fps by time
DEFAULT_STREAM_FPS = 15


def _import_sl():
    try:
        import pyzed.sl as sl
    except ImportError as e:
        raise ImportError("pyzed not found: the ZED SDK Python API is installed on node_c "
                          "only (ZED SDK 5.2.3); run inside that node's venv") from e
    return sl


#: where the ZED SDK keeps the factory calibration file SN<serial>.conf
ZED_SETTINGS_DIRS = ("/usr/local/zed/settings",)


def factory_conf(serial, dirs=None) -> Optional[Dict[str, Any]]:
    """The factory calibration file ``SN<serial>.conf`` (prereg sec. 13.1 draft): its
    source path, md5 and exact text, or None if it is not on this machine."""
    import hashlib
    if serial is None:
        return None
    for d in (ZED_SETTINGS_DIRS if dirs is None else dirs):
        path = os.path.join(d, "SN%s.conf" % serial)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                blob = f.read()
            return {"source": path, "md5": hashlib.md5(blob).hexdigest(),
                    "bytes": len(blob), "text": blob.decode("latin-1")}
    return None


def _transform(raw) -> Optional[Dict[str, Any]]:
    """The raw (unrectified) stereo transform, left -> right: rotation vector and
    translation (in the capture's units, mm)."""
    t = getattr(raw, "stereo_transform", None)
    if t is None:
        return None
    try:
        return {"rotation_vector": [float(x) for x in t.get_rotation_vector()],
                "translation_mm": [float(x) for x in t.get_translation().get()]}
    except Exception:  # noqa: BLE001 - recorded as missing; the capture then refuses
        return None


def _setting(zed, sl, name) -> Optional[int]:
    """get_camera_settings -> int or None.  SDK >= 4 returns (ERROR_CODE, value)."""
    key = getattr(sl.VIDEO_SETTINGS, name, None)
    if key is None:
        return None
    try:
        res = zed.get_camera_settings(key)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(res, tuple):
        err, value = res[0], res[-1]
        if err != sl.ERROR_CODE.SUCCESS:
            return None
        return int(value)
    return int(res) if res is not None and int(res) >= 0 else None


class ZedBackend(CameraBackend):
    def __init__(self, depth_mode: str = DEFAULT_DEPTH_MODE,
                 resolution: str = DEFAULT_RESOLUTION, stream_fps: int = DEFAULT_STREAM_FPS,
                 serial: Optional[int] = None):
        self.depth_mode = depth_mode.upper()
        self.resolution = resolution.upper()
        self.stream_fps = int(stream_fps)
        self.serial = serial
        self.sl = None
        self.zed = None
        self._info: Dict[str, Any] = {}

    def open(self) -> None:
        sl = self.sl = _import_sl()
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, self.resolution)
        init.camera_fps = self.stream_fps
        init.depth_mode = getattr(sl.DEPTH_MODE, self.depth_mode)
        init.coordinate_units = sl.UNIT.MILLIMETER
        init.depth_minimum_distance = DEPTH_RANGE_MM[0]
        init.depth_maximum_distance = DEPTH_RANGE_MM[1]
        if self.serial is not None:
            init.set_from_serial_number(int(self.serial))
        self.zed = sl.Camera()
        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            self.zed = None
            raise CaptureError("ZED open failed: %s" % err)
        self.runtime = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.depth = sl.Mat()
        cam = self.zed.get_camera_information()
        conf = getattr(cam, "camera_configuration", None)
        firmware = getattr(conf, "firmware_version", None)
        sensors = getattr(cam, "sensors_configuration", None)
        res = getattr(conf, "resolution", None) or getattr(cam, "camera_resolution", None)
        self._info = {
            "camera_model": str(cam.camera_model),
            "serial": int(cam.serial_number),
            "firmware": None if firmware is None else str(firmware),
            "sensors_firmware": None if sensors is None
            else str(getattr(sensors, "firmware_version", None)),
            "sdk_version": str(sl.Camera.get_sdk_version()),
            "depth_mode": self.depth_mode,
            "resolution_setting": self.resolution,
            "rgb_resolution": [int(res.width), int(res.height)] if res is not None else None,
            "stream_fps": float(getattr(conf, "fps", self.stream_fps) or self.stream_fps),
            "depth_units": "MILLIMETER",
            "requires_factory_calibration": True,
        }
        self._info.update(self._calibration(conf, res))
        usb = zed_usb_devices()
        self._info["usb_devices"] = usb
        self._info["usb_speed_mbps"] = zed_usb_speed_mbps(usb)

    def _calibration(self, conf, res) -> Dict[str, Any]:
        """Intrinsics with distortion (left = RGB and depth frame, right), the stereo
        baseline, the depth scale and the depth range actually applied (Amendment 3)."""
        calib = getattr(conf, "calibration_parameters", None)
        w = int(res.width) if res is not None else None
        h = int(res.height) if res is not None else None

        def cam(c):
            return {"width": w, "height": h, "fx": float(c.fx), "fy": float(c.fy),
                    "ppx": float(c.cx), "ppy": float(c.cy),
                    "model": "ZED (k1, k2, p1, p2, k3, ...)",
                    "coeffs": [float(x) for x in c.disto]}

        out: Dict[str, Any] = {"intrinsics": None, "stereo_baseline_mm": None,
                               "depth_scale_mm": 1.0, "depth_range_mm": None}
        if calib is not None:
            left = cam(calib.left_cam)
            out["intrinsics"] = {"rgb": left, "depth": dict(left), "right": cam(calib.right_cam)}
            # in the InitParameters coordinate units, set to MILLIMETER above
            out["stereo_baseline_mm"] = float(calib.get_camera_baseline())
        raw = getattr(conf, "calibration_parameters_raw", None)
        out["calibration_raw"] = None
        if raw is not None:
            out["calibration_raw"] = {"left": cam(raw.left_cam), "right": cam(raw.right_cam),
                                      "stereo_transform": _transform(raw)}
        out["factory_conf"] = factory_conf(self._info.get("serial"))
        applied = self.zed.get_init_parameters()
        rng = (float(applied.depth_minimum_distance), float(applied.depth_maximum_distance))
        out["depth_range_mm"] = list(rng)
        if rng != DEPTH_RANGE_MM:
            raise CaptureError("ZED depth range %s mm applied, %s requested (prereg Amendment 3)"
                               % (rng, DEPTH_RANGE_MM))
        return out

    def info(self) -> Dict[str, Any]:
        return dict(self._info)

    def _grab_ok(self):
        sl = self.sl
        err = self.zed.grab(self.runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            raise CaptureError("ZED grab failed: %s" % err)

    def skip(self) -> None:
        self._grab_ok()

    def grab(self) -> Frame:
        sl = self.sl
        self._grab_ok()
        ts = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
        bgra = self.image.get_data()
        rgb = np.ascontiguousarray(bgra[:, :, 2::-1])  # BGRA -> RGB, copied out of the Mat
        depth = depth_to_mm_uint16(np.array(self.depth.get_data(), copy=True))
        wb_auto = _setting(self.zed, sl, "WHITEBALANCE_AUTO")
        aec = _setting(self.zed, sl, "AEC_AGC")
        meta = {
            "exposure": _setting(self.zed, sl, "EXPOSURE"),
            "gain": _setting(self.zed, sl, "GAIN"),
            "white_balance": _setting(self.zed, sl, "WHITEBALANCE_TEMPERATURE"),
            "white_balance_auto": None if wb_auto is None else bool(wb_auto),
            "auto_exposure": None if aec is None else bool(aec),
            "exposure_units": "percent of frame period (ZED VIDEO_SETTINGS.EXPOSURE)",
            "timestamp_reference": "TIME_REFERENCE.IMAGE",
        }
        return Frame(rgb=rgb, depth_mm=depth, ir=None, device_ts_ms=float(ts), meta=meta)

    # -- exposure lock (prereg sec. 13 draft) ------------------------------------
    def _set(self, name, value) -> None:
        sl = self.sl
        err = self.zed.set_camera_settings(getattr(sl.VIDEO_SETTINGS, name), int(value))
        if err is not None and err != sl.ERROR_CODE.SUCCESS:
            raise CaptureError("ZED set_camera_settings(%s, %s) failed: %s" % (name, value, err))

    def settle_auto(self, n_frames) -> Dict[str, Any]:
        """AEC/AGC and auto white balance run for ``n_frames``; -> the values they applied
        (the ZED reports applied values while auto is on).  White balance on the SDK's
        100 K grid."""
        sl = self.sl
        self._set("AEC_AGC", 1)
        self._set("WHITEBALANCE_AUTO", 1)
        for _ in range(max(1, int(n_frames))):
            self._grab_ok()
        wb = _setting(self.zed, sl, "WHITEBALANCE_TEMPERATURE")
        return {"exposure": _setting(self.zed, sl, "EXPOSURE"),
                "gain": _setting(self.zed, sl, "GAIN"),
                "white_balance": None if wb is None else int(round(wb / 100.0) * 100),
                "source": "get_camera_settings with AEC_AGC and auto white balance on, "
                          "after %d frames" % n_frames}

    def apply_lock(self, lock) -> Dict[str, Any]:
        sl = self.sl
        self._set("AEC_AGC", 0)
        self._set("WHITEBALANCE_AUTO", 0)
        self._set("EXPOSURE", lock["exposure"])
        self._set("GAIN", lock["gain"])
        self._set("WHITEBALANCE_TEMPERATURE", lock["white_balance"])
        return {"exposure": _setting(self.zed, sl, "EXPOSURE"),
                "gain": _setting(self.zed, sl, "GAIN"),
                "white_balance": _setting(self.zed, sl, "WHITEBALANCE_TEMPERATURE"),
                "auto_exposure": _setting(self.zed, sl, "AEC_AGC"),
                "auto_white_balance": _setting(self.zed, sl, "WHITEBALANCE_AUTO")}

    def close(self) -> None:
        if self.zed is not None:
            self.zed.close()
            self.zed = None


def list_cameras():
    sl = _import_sl()
    devs = sl.Camera.get_device_list()
    if not devs:
        print("No ZED camera detected.")
    for d in devs:
        print("  Camera: %s  Serial: %s  State: %s  Id: %s"
              % (d.camera_model, d.serial_number, d.camera_state, d.id))
    print("  ZED SDK %s" % sl.Camera.get_sdk_version())
    return devs


def build_parser():
    p = argparse.ArgumentParser(description="FedRGBD ZED 2i capture (camera experiment)")
    add_capture_args(p, default_node="node_c")
    p.add_argument("--depth_mode", default=DEFAULT_DEPTH_MODE,
                   help="sl.DEPTH_MODE name (default %(default)s)")
    p.add_argument("--resolution", default=DEFAULT_RESOLUTION,
                   help="sl.RESOLUTION name (default %(default)s)")
    p.add_argument("--stream_fps", type=int, default=DEFAULT_STREAM_FPS,
                   help="camera stream rate, sampled down to --fps (default %(default)s)")
    p.add_argument("--serial", type=int, default=None)
    p.add_argument("--test", action="store_true",
                   help="grab a few frames to a temp dir and print the camera info")
    p.add_argument("--list", action="store_true", help="list connected ZED cameras")
    return p


def main(argv=None, backend_factory=None) -> int:
    args = build_parser().parse_args(argv)
    factory = backend_factory or (lambda: ZedBackend(args.depth_mode, args.resolution,
                                                     args.stream_fps, args.serial))
    if args.list:
        list_cameras()
        return 0
    if args.test:
        return 0 if smoke_test(factory(), args.node) else 1
    if not (args.scene and args.label and args.distance_m is not None):
        print("--scene, --label and --distance_m are required for a capture", file=sys.stderr)
        return 2
    try:
        rec = run_capture(factory(), args.scene, args.label, args.distance_m, args.node,
                          root=args.root, frames=args.frames, fps=args.fps,
                          start_at=args.start_at, retake=args.retake, notes=args.notes,
                          study_gates=True, exposure_lock=not args.no_lock,
                          lock_settle_frames=args.lock_settle_frames)
    except (CaptureError, ValueError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 1
    print(json.dumps({k: rec[k] for k in ("capture_id", "node", "n_frames_written",
                                          "n_frames_requested", "start_scheduled_unix",
                                          "start_actual_unix", "status")}))
    return 0 if rec["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
