"""
FedRGBD — RealSense Camera Capture Pipeline
=============================================
Captures synchronized RGB, Depth, and IR frames from Intel RealSense cameras.
Supports D435i and D455 models.

Usage:
    # Quick test (displays 5 frames info)
    python3 src/data/realsense_capture.py --test

    # Capture N frames and save to disk
    python3 src/data/realsense_capture.py --output data/raw/custom/node_a --frames 500

    # List connected cameras
    python3 src/data/realsense_capture.py --list

    # Camera experiment (docs/CAMERA_EXPERIMENT_PREREG.md): one capture in the data
    # contract of src/data/camera_capture_common.py -- 40 frames at 5 fps, depth aligned
    # to colour, left IR, per-frame metadata; normally started by
    # scripts/camera_capture_session.py with a common --start_at on all three nodes
    python3 src/data/realsense_capture.py --scene s01 --label fire --distance_m 2 \\
        --node node_a [--start_at <unix s>] [--retake]

    # Camera experiment smoke test (a few frames through the capture path, temp dir)
    python3 src/data/realsense_capture.py --test --node node_a

Without --scene the legacy behaviour (--output/--frames/--serial/--fps, default
500 frames at 30 fps into data/raw/custom/capture) is unchanged.  pyrealsense2 is
imported lazily, so the module imports on a machine without it.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.data.camera_capture_common import (  # noqa: E402
    DEFAULT_FPS, DEFAULT_FRAMES, DEFAULT_ROOT, LABELS, NOTES_FORMAT, CameraBackend,
    CaptureError, Frame, depth_to_mm_uint16, run_capture, run_exposure_calibration,
    run_session_check, smoke_test,
)
from src.data.camera_exposure import ExposureError  # noqa: E402

#: pyrealsense2, imported lazily (only node_a / node_b have it; tests run without it)
rs = None


def _import_rs(exit_on_fail=True):
    """Import pyrealsense2 on first use.  The legacy CLI exits as it always did."""
    global rs
    if rs is None:
        try:
            import pyrealsense2 as _rs
        except ImportError:
            if not exit_on_fail:
                raise
            print("ERROR: pyrealsense2 not found.")
            print("If built from source, try: export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.10/site-packages")
            sys.exit(1)
        rs = _rs
    return rs


def list_cameras():
    """List all connected RealSense cameras."""
    _import_rs()
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("No RealSense cameras detected.")
        print("Check USB3 connection and run: rs-enumerate-devices")
        return []

    cameras = []
    for dev in devices:
        info = {
            "name": dev.get_info(rs.camera_info.name),
            "serial": dev.get_info(rs.camera_info.serial_number),
            "firmware": dev.get_info(rs.camera_info.firmware_version),
            "usb_type": dev.get_info(rs.camera_info.usb_type_descriptor),
        }
        cameras.append(info)
        print(f"  Camera: {info['name']}")
        print(f"  Serial: {info['serial']}")
        print(f"  Firmware: {info['firmware']}")
        print(f"  USB Type: {info['usb_type']}")
        print()

    return cameras


def configure_pipeline(serial=None, width_rgb=1920, height_rgb=1080,
                       width_depth=1280, height_depth=720, fps=30):
    """Configure RealSense pipeline for RGB + Depth + IR capture."""
    _import_rs()
    pipeline = rs.pipeline()
    config = rs.config()

    if serial:
        config.enable_device(serial)

    # Enable streams
    config.enable_stream(rs.stream.color, width_rgb, height_rgb, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width_depth, height_depth, rs.format.z16, fps)
    config.enable_stream(rs.stream.infrared, 1, width_depth, height_depth, rs.format.y8, fps)

    return pipeline, config


def get_camera_intrinsics(profile):
    """Extract camera intrinsics from the pipeline profile."""
    _import_rs()
    intrinsics = {}

    for stream_type, name in [(rs.stream.color, "rgb"),
                               (rs.stream.depth, "depth"),
                               (rs.stream.infrared, "ir")]:
        try:
            stream_profile = profile.get_stream(stream_type)
            intr = stream_profile.as_video_stream_profile().get_intrinsics()
            intrinsics[name] = {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy,
                "model": str(intr.model),
                "coeffs": list(intr.coeffs),
            }
        except Exception as e:
            print(f"  Warning: Could not get {name} intrinsics: {e}")

    return intrinsics


def capture_test(num_frames=5):
    """Quick test: capture a few frames and display info."""
    _import_rs()
    print("=" * 60)
    print("RealSense Camera Test")
    print("=" * 60)
    print()

    cameras = list_cameras()
    if not cameras:
        return False

    pipeline, config = configure_pipeline()

    try:
        profile = pipeline.start(config)

        # Get device info
        device = profile.get_device()
        camera_name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)

        print(f"Started pipeline for: {camera_name} (S/N: {serial})")
        print(f"Capturing {num_frames} test frames...")
        print()

        # Get intrinsics
        intrinsics = get_camera_intrinsics(profile)
        for stream_name, intr in intrinsics.items():
            print(f"  {stream_name}: {intr['width']}x{intr['height']}, "
                  f"fx={intr['fx']:.1f}, fy={intr['fy']:.1f}")

        # Warm up (skip first few frames)
        for _ in range(10):
            pipeline.wait_for_frames(timeout_ms=5000)

        # Capture test frames
        for i in range(num_frames):
            frames = pipeline.wait_for_frames(timeout_ms=5000)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            ir_frame = frames.get_infrared_frame(1)

            if color_frame and depth_frame and ir_frame:
                color_data = np.asanyarray(color_frame.get_data())
                depth_data = np.asanyarray(depth_frame.get_data())
                ir_data = np.asanyarray(ir_frame.get_data())

                depth_min = depth_data[depth_data > 0].min() if (depth_data > 0).any() else 0
                depth_max = depth_data.max()

                print(f"  Frame {i+1}: RGB {color_data.shape}, "
                      f"Depth {depth_data.shape} [{depth_min}-{depth_max}mm], "
                      f"IR {ir_data.shape} [{ir_data.min()}-{ir_data.max()}]")
            else:
                print(f"  Frame {i+1}: INCOMPLETE (missing streams)")

        print()
        print("Test PASSED — camera is working correctly.")
        return True

    except rs.error as e:
        print(f"RealSense error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        pipeline.stop()


def capture_frames(output_dir, num_frames=500, serial=None,
                   width_rgb=1920, height_rgb=1080,
                   width_depth=1280, height_depth=720, fps=30):
    """Capture and save synchronized RGB + Depth + IR frames."""
    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV not found. Install: pip install opencv-python-headless")
        sys.exit(1)

    _import_rs()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline, config = configure_pipeline(
        serial=serial,
        width_rgb=width_rgb, height_rgb=height_rgb,
        width_depth=width_depth, height_depth=height_depth,
        fps=fps
    )

    try:
        profile = pipeline.start(config)

        device = profile.get_device()
        camera_name = device.get_info(rs.camera_info.name)
        camera_serial = device.get_info(rs.camera_info.serial_number)
        firmware = device.get_info(rs.camera_info.firmware_version)

        print(f"Camera: {camera_name} (S/N: {camera_serial})")
        print(f"Firmware: {firmware}")
        print(f"Output: {output_path}")
        print(f"Target frames: {num_frames}")
        print()

        # Get intrinsics
        intrinsics = get_camera_intrinsics(profile)

        # Save camera metadata
        meta = {
            "camera_name": camera_name,
            "serial_number": camera_serial,
            "firmware_version": firmware,
            "resolution_rgb": [width_rgb, height_rgb],
            "resolution_depth": [width_depth, height_depth],
            "fps": fps,
            "intrinsics": intrinsics,
            "capture_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Align depth to color (optional — useful for early fusion)
        align = rs.align(rs.stream.color)

        # Warm up
        print("Warming up camera (skipping first 30 frames)...")
        for _ in range(30):
            pipeline.wait_for_frames(timeout_ms=5000)

        # Capture loop
        captured = 0
        start_time = time.time()

        print(f"Capturing {num_frames} frames...")
        while captured < num_frames:
            frames = pipeline.wait_for_frames(timeout_ms=5000)

            # Align depth to color frame
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            ir_frame = frames.get_infrared_frame(1)  # IR is not aligned

            if not (color_frame and depth_frame and ir_frame):
                continue

            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            ir_image = np.asanyarray(ir_frame.get_data())

            # Get timestamp
            timestamp_ms = frames.get_timestamp()

            # Save frame ID with zero-padding
            frame_id = f"{captured:05d}"

            # Save RGB (BGR format from RealSense, save as-is for OpenCV)
            cv2.imwrite(str(output_path / f"{frame_id}_rgb.png"), color_image)

            # Save depth as 16-bit PNG (preserves mm precision)
            cv2.imwrite(str(output_path / f"{frame_id}_depth.png"), depth_image)

            # Save IR as 8-bit PNG
            cv2.imwrite(str(output_path / f"{frame_id}_ir.png"), ir_image)

            # Save per-frame metadata
            frame_meta = {
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
                "depth_min_mm": int(depth_image[depth_image > 0].min()) if (depth_image > 0).any() else 0,
                "depth_max_mm": int(depth_image.max()),
            }

            with open(output_path / f"{frame_id}_meta.json", "w") as f:
                json.dump(frame_meta, f)

            captured += 1

            if captured % 50 == 0:
                elapsed = time.time() - start_time
                fps_actual = captured / elapsed
                print(f"  Captured {captured}/{num_frames} frames "
                      f"({fps_actual:.1f} fps, {elapsed:.1f}s elapsed)")

        elapsed = time.time() - start_time
        meta["capture_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
        meta["total_frames"] = captured
        meta["capture_duration_s"] = round(elapsed, 2)
        meta["average_fps"] = round(captured / elapsed, 2)

        # Save session metadata
        with open(output_path / "capture_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print()
        print(f"Capture complete: {captured} frames in {elapsed:.1f}s "
              f"({captured/elapsed:.1f} fps)")
        print(f"Saved to: {output_path}")

    except rs.error as e:
        print(f"RealSense error: {e}")
        sys.exit(1)
    finally:
        pipeline.stop()


# --------------------------------------------------------------------------- #
# camera experiment back-end (docs/CAMERA_EXPERIMENT_PREREG.md)
# --------------------------------------------------------------------------- #
#: D435 colour is 1920x1080 max, depth/IR 1280x720 max; there is no 5 fps mode, so the
#: stream runs at 30 fps and the capture keeps 5 fps by device timestamp
EXP_RGB_RES = (1920, 1080)
EXP_DEPTH_RES = (1280, 720)
EXP_STREAM_FPS = 30


def realsense_sdk_version(rs_mod=None):
    """librealsense version string, or None.

    pyrealsense2 built from source on the nodes has no ``__version__``; fall back to the
    installed distribution's metadata, then to ``pkg-config --modversion realsense2``.
    """
    rs_mod = rs_mod if rs_mod is not None else rs
    for attr in ("__version__", "__full_version__"):
        v = getattr(rs_mod, attr, None) if rs_mod is not None else None
        if v:
            return str(v)
    try:
        from importlib import metadata
        for dist in ("pyrealsense2", "pyrealsense2-aarch64"):
            try:
                return metadata.version(dist)
            except metadata.PackageNotFoundError:
                continue
    except Exception:  # noqa: BLE001
        pass
    try:
        import subprocess
        out = subprocess.run(["pkg-config", "--modversion", "realsense2"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return "librealsense " + out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def _metadata(frame, key):
    """Frame metadata value, or None when the platform does not report it."""
    attr = getattr(rs.frame_metadata_value, key, None)
    if attr is None:
        return None
    try:
        if frame.supports_frame_metadata(attr):
            return int(frame.get_frame_metadata(attr))
    except Exception:  # noqa: BLE001
        pass
    return None


def _option(sensor, key):
    attr = getattr(rs.option, key, None)
    if sensor is None or attr is None:
        return None
    try:
        if sensor.supports(attr):
            return float(sensor.get_option(attr))
    except Exception:  # noqa: BLE001
        pass
    return None


def stereo_baseline_mm(depth_sensor):
    """Distance between the two IR imagers in mm (prereg Amendment 3): the D400 option
    ``stereo_baseline`` (mm), else the translation between the left and right infrared
    stream profiles (m).  None if neither is available."""
    v = _option(depth_sensor, "stereo_baseline")
    if v:
        return float(v)
    try:
        profiles = [p for p in depth_sensor.get_stream_profiles()
                    if p.stream_type() == rs.stream.infrared]
        left = next(p for p in profiles if p.stream_index() == 1)
        right = next(p for p in profiles if p.stream_index() == 2)
        return abs(float(left.get_extrinsics_to(right).translation[0])) * 1000.0
    except Exception:  # noqa: BLE001 - recorded as missing; the capture then refuses
        return None


def extrinsics_depth_to_color(profile):
    """Rotation (row-major 3x3) and translation (m) from the depth to the colour stream."""
    try:
        e = profile.get_stream(rs.stream.depth).get_extrinsics_to(profile.get_stream(rs.stream.color))
        return {"rotation": [float(x) for x in e.rotation],
                "translation_m": [float(x) for x in e.translation]}
    except Exception:  # noqa: BLE001
        return None


def _safe_info(device, key):
    try:
        return device.get_info(getattr(rs.camera_info, key))
    except Exception:  # noqa: BLE001
        return None


class RealSenseBackend(CameraBackend):
    """RGB (rgb8, 1920x1080), depth z16 aligned to colour with rs.align, left IR (y8)."""

    def __init__(self, serial=None, rgb_res=EXP_RGB_RES, depth_res=EXP_DEPTH_RES,
                 stream_fps=EXP_STREAM_FPS):
        self.serial = serial
        self.rgb_res = tuple(rgb_res)
        self.depth_res = tuple(depth_res)
        self.stream_fps = int(stream_fps)
        self.pipeline = None
        self._info = {}

    def open(self):
        _import_rs(exit_on_fail=False)
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(str(self.serial))
        w, h = self.rgb_res
        dw, dh = self.depth_res
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, self.stream_fps)
        config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, self.stream_fps)
        config.enable_stream(rs.stream.infrared, 1, dw, dh, rs.format.y8, self.stream_fps)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        device = profile.get_device()
        self.depth_sensor = device.first_depth_sensor()
        self.color_sensor = None
        try:
            self.color_sensor = device.first_color_sensor()
        except Exception:  # noqa: BLE001 - older librealsense: find it by name
            for sensor in device.query_sensors():
                if _safe_info(sensor, "name") == "RGB Camera":
                    self.color_sensor = sensor
        scale_m = float(self.depth_sensor.get_depth_scale())
        self.depth_scale_to_mm = scale_m * 1000.0
        emitter = _option(self.depth_sensor, "emitter_enabled")
        self._info = {
            "camera_model": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "sdk_version": realsense_sdk_version(rs),
            "depth_mode": None,
            "rgb_resolution": [int(w), int(h)],
            "depth_resolution": [int(dw), int(dh)],
            "stream_fps": float(self.stream_fps),
            "depth_scale_m": scale_m,
            "depth_scale_mm": scale_m * 1000.0,
            "depth_range_mm": None,          # not a RealSense setting; z16 is recorded as is
            "stereo_baseline_mm": stereo_baseline_mm(self.depth_sensor),
            "extrinsics_depth_to_color": extrinsics_depth_to_color(profile),
            "emitter_enabled": None if emitter is None else bool(emitter),
            "usb_type": _safe_info(device, "usb_type_descriptor"),
            "intrinsics": get_camera_intrinsics(profile),
            "depth_aligned_to": "color (rs.align)",
            "ir_stream": "infrared 1 (left imager), not aligned",
        }

    def info(self):
        return dict(self._info)

    def skip(self):
        self.pipeline.wait_for_frames(timeout_ms=5000)

    def grab(self):
        while True:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned = self.align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            ir = frames.get_infrared_frame(1)
            if color and depth and ir:
                break
        rgb = np.array(np.asanyarray(color.get_data()), copy=True)
        depth_raw = np.asanyarray(depth.get_data())
        if abs(self.depth_scale_to_mm - 1.0) < 1e-9:
            depth_mm = np.array(depth_raw, dtype=np.uint16, copy=True)
        else:
            depth_mm = depth_to_mm_uint16(depth_raw, self.depth_scale_to_mm)
        ir_img = np.array(np.asanyarray(ir.get_data()), dtype=np.uint8, copy=True)

        exposure = _metadata(color, "actual_exposure")
        source = "frame_metadata"
        if exposure is None:
            exposure = _option(self.color_sensor, "exposure")
            source = "sensor_option" if exposure is not None else None
        gain = _metadata(color, "gain_level")
        if gain is None:
            gain = _option(self.color_sensor, "gain")
        wb = _metadata(color, "white_balance")
        if wb is None:
            wb = _option(self.color_sensor, "white_balance")
        ae = _metadata(color, "auto_exposure")
        if ae is None:
            ae = _option(self.color_sensor, "enable_auto_exposure")
        try:
            domain = str(frames.get_frame_timestamp_domain())
        except Exception:  # noqa: BLE001
            domain = None
        meta = {
            "exposure": exposure,
            "exposure_source": source,
            "gain": gain,
            "white_balance": wb,
            "auto_exposure": None if ae is None else bool(ae),
            "ir_exposure": _metadata(ir, "actual_exposure"),
            "ir_gain": _metadata(ir, "gain_level"),
            "timestamp_domain": domain,
            "frame_number_color": int(color.get_frame_number()),
        }
        return Frame(rgb=rgb, depth_mm=depth_mm, ir=ir_img,
                     device_ts_ms=float(frames.get_timestamp()), meta=meta)

    # -- exposure lock (prereg sec. 13 draft) ------------------------------------
    def _set(self, key, value):
        self.color_sensor.set_option(getattr(rs.option, key), float(value))

    def _snap(self, key, value):
        """``value`` on the option's grid (white balance moves in steps of 10 K)."""
        try:
            r = self.color_sensor.get_option_range(getattr(rs.option, key))
            step = float(r.step) or 1.0
            v = float(r.min) + round((float(value) - float(r.min)) / step) * step
            return min(max(v, float(r.min)), float(r.max))
        except Exception:  # noqa: BLE001
            return float(round(float(value)))

    def exposure_defaults(self):
        """Default gain (option range), the white balance range, and the exposure range
        in the sensor's unit (UVC, 100 us), capped at one frame period of the stream so
        the frame rate does not drop."""
        r = {k: self.color_sensor.get_option_range(getattr(rs.option, k))
             for k in ("exposure", "gain", "white_balance")}
        cap = float(int(1e4 / float(self.stream_fps)))
        wb, gn = r["white_balance"], r["gain"]
        return {"gain": float(gn.default),
                "gain_range": {"min": float(gn.min), "max": float(gn.max),
                               "step": float(gn.step), "default": float(gn.default)},
                "white_balance_range": {"min": float(wb.min), "max": float(wb.max),
                                        "step": float(wb.step), "default": float(wb.default)},
                "exposure_range": {"min": float(r["exposure"].min),
                                   "max": min(float(r["exposure"].max), cap),
                                   "step": float(r["exposure"].step),
                                   "default": float(r["exposure"].default)}}

    def apply_lock(self, lock):
        self._set("enable_auto_exposure", 0)
        self._set("enable_auto_white_balance", 0)
        for k in ("exposure", "gain", "white_balance"):
            self._set(k, lock[k])
        return {"exposure": _option(self.color_sensor, "exposure"),
                "gain": _option(self.color_sensor, "gain"),
                "white_balance": _option(self.color_sensor, "white_balance"),
                "auto_exposure": _option(self.color_sensor, "enable_auto_exposure"),
                "auto_white_balance": _option(self.color_sensor, "enable_auto_white_balance")}

    def close(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            finally:
                self.pipeline = None


def build_parser():
    parser = argparse.ArgumentParser(description="FedRGBD RealSense Capture")
    parser.add_argument("--test", action="store_true",
                        help="Quick test: capture 5 frames and display info "
                             "(with --node: camera-experiment smoke test into a temp dir)")
    parser.add_argument("--list", action="store_true",
                        help="List connected RealSense cameras")
    parser.add_argument("--output", type=str, default="data/raw/custom/capture",
                        help="[legacy] Output directory for captured frames")
    parser.add_argument("--frames", type=int, default=None,
                        help="Number of frames to capture (legacy default 500; "
                             "camera experiment default %d)" % DEFAULT_FRAMES)
    parser.add_argument("--serial", type=str, default=None,
                        help="Camera serial number (auto-detect if not specified)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Legacy: stream frame rate (default 30). Camera experiment: "
                             "frames per second kept (default %g)" % DEFAULT_FPS)
    # camera experiment: the flags of camera_capture_common.add_capture_args, minus
    # --frames/--fps whose legacy defaults differ (resolved in main)
    parser.add_argument("--scene", help="[camera experiment] scene id, s01, s02, ...")
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--distance_m", type=float)
    parser.add_argument("--start_at", type=float, default=None,
                        help="scheduled start, unix seconds (host wall clock)")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--node", default=None, choices=("node_a", "node_b"))
    parser.add_argument("--retake", action="store_true",
                        help="move an existing capture to _retakes/<timestamp>/ first")
    parser.add_argument("--notes", default="", help="required, parsed: " + NOTES_FORMAT)
    parser.add_argument("--no_lock", action="store_true",
                        help="TEST ONLY (e5): no exposure lock; never a study capture")
    parser.add_argument("--calibrate_exposure", action="store_true",
                        help="set this camera's one fixed exposure on a flame-free scene "
                             "(prereg 13.1 draft); no capture")
    parser.add_argument("--recalibrate", action="store_true",
                        help="with --calibrate_exposure: replace an existing exposure file")
    parser.add_argument("--session_check", action="store_true",
                        help="check at the start of a capture session (prereg 13.1)")
    parser.add_argument("--stream_fps", type=int, default=EXP_STREAM_FPS,
                        help="[camera experiment] stream rate, sampled down to --fps")
    return parser


def main(argv=None, backend_factory=None):
    args = build_parser().parse_args(argv)
    experiment = (args.scene is not None or args.calibrate_exposure or args.session_check
                  or (args.test and args.node is not None))
    if not experiment:  # the original CLI, unchanged
        if args.list:
            list_cameras()
        elif args.test:
            success = capture_test()
            sys.exit(0 if success else 1)
        else:
            capture_frames(
                output_dir=args.output,
                num_frames=args.frames if args.frames is not None else 500,
                serial=args.serial,
                fps=int(args.fps) if args.fps is not None else 30
            )
        return 0

    frames = args.frames if args.frames is not None else DEFAULT_FRAMES
    fps = args.fps if args.fps is not None else DEFAULT_FPS
    factory = backend_factory or (lambda: RealSenseBackend(serial=args.serial,
                                                           stream_fps=args.stream_fps))
    if args.test:
        return 0 if smoke_test(factory(), args.node) else 1
    if args.calibrate_exposure:
        if args.node is None:
            print("--calibrate_exposure needs --node (node_a or node_b)", file=sys.stderr)
            return 2
        try:
            run_exposure_calibration(factory(), args.root, args.node, args.notes,
                                     recalibrate=args.recalibrate)
        except (CaptureError, ValueError, ExposureError) as e:
            print("ERROR: %s" % e, file=sys.stderr)
            return 1
        return 0
    if args.session_check:
        if args.node is None:
            print("--session_check needs --node (node_a or node_b)", file=sys.stderr)
            return 2
        try:
            rec = run_session_check(factory(), args.root, args.node, args.notes)
        except (CaptureError, ValueError, ExposureError) as e:
            print("ERROR: %s" % e, file=sys.stderr)
            return 1
        return 0 if rec["session_check"] == "PASS" else 1
    if args.node is None or args.label is None or args.distance_m is None:
        print("--scene needs --label, --distance_m and --node (node_a or node_b)",
              file=sys.stderr)
        return 2
    try:
        rec = run_capture(factory(), args.scene, args.label, args.distance_m, args.node,
                          root=args.root, frames=frames, fps=fps, start_at=args.start_at,
                          retake=args.retake, notes=args.notes, study_gates=True,
                          exposure_lock=not args.no_lock)
    except (CaptureError, ValueError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 1
    print(json.dumps({k: rec[k] for k in ("capture_id", "node", "n_frames_written",
                                          "n_frames_requested", "start_scheduled_unix",
                                          "start_actual_unix", "status")}))
    return 0 if rec["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
