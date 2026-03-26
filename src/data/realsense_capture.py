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
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 not found.")
    print("If built from source, try: export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.10/site-packages")
    sys.exit(1)


def list_cameras():
    """List all connected RealSense cameras."""
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


def main():
    parser = argparse.ArgumentParser(description="FedRGBD RealSense Capture")
    parser.add_argument("--test", action="store_true",
                        help="Quick test: capture 5 frames and display info")
    parser.add_argument("--list", action="store_true",
                        help="List connected RealSense cameras")
    parser.add_argument("--output", type=str, default="data/raw/custom/capture",
                        help="Output directory for captured frames")
    parser.add_argument("--frames", type=int, default=500,
                        help="Number of frames to capture")
    parser.add_argument("--serial", type=str, default=None,
                        help="Camera serial number (auto-detect if not specified)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Capture frame rate")

    args = parser.parse_args()

    if args.list:
        list_cameras()
    elif args.test:
        success = capture_test()
        sys.exit(0 if success else 1)
    else:
        capture_frames(
            output_dir=args.output,
            num_frames=args.frames,
            serial=args.serial,
            fps=args.fps
        )


if __name__ == "__main__":
    main()
