#!/usr/bin/env python3
"""
FedRGBD — Post-Setup Verification Script
==========================================
Run this after setup_jetson.sh to verify all components are working.

Usage:
    python3 scripts/verify_setup.py
"""

import subprocess
import sys
import os

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"


def check(name, condition, note=""):
    status = PASS if condition else FAIL
    suffix = f"  ({note})" if note else ""
    print(f"  {status} {name}{suffix}")
    return condition


def main():
    print("=" * 60)
    print("  FedRGBD — Setup Verification")
    print("=" * 60)
    print()

    all_pass = True

    # --- Python ---
    print("Python Environment:")
    ver = sys.version_info
    all_pass &= check("Python 3.10+", ver.major == 3 and ver.minor >= 10,
                       f"{ver.major}.{ver.minor}.{ver.micro}")

    # --- PyTorch + CUDA ---
    print("\nPyTorch & CUDA:")
    try:
        import torch
        all_pass &= check("PyTorch imported", True, torch.__version__)
        cuda_ok = torch.cuda.is_available()
        all_pass &= check("CUDA available", cuda_ok)
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            check("GPU detected", True, f"{gpu_name}, {gpu_mem:.1f} GB")

            # Quick CUDA test
            x = torch.randn(100, 100, device='cuda')
            y = torch.matmul(x, x)
            check("CUDA computation", True, "matmul test passed")
    except ImportError:
        all_pass &= check("PyTorch imported", False, "not installed")
    except Exception as e:
        all_pass &= check("PyTorch CUDA", False, str(e))

    # --- torchvision ---
    print("\ntorchvision:")
    try:
        import torchvision
        all_pass &= check("torchvision imported", True, torchvision.__version__)

        # Check MobileNetV3 availability
        model = torchvision.models.mobilenet_v3_small(weights=None, num_classes=2)
        param_count = sum(p.numel() for p in model.parameters())
        check("MobileNetV3-Small", True, f"{param_count/1e6:.1f}M params")
    except ImportError:
        all_pass &= check("torchvision imported", False, "not installed")
    except Exception as e:
        all_pass &= check("torchvision", False, str(e))

    # --- Flower ---
    print("\nFlower (FL Framework):")
    try:
        import flwr
        all_pass &= check("Flower imported", True, flwr.__version__)
    except ImportError:
        all_pass &= check("Flower imported", False, "pip install flwr")

    # --- pyrealsense2 ---
    print("\npyrealsense2:")
    try:
        import pyrealsense2 as rs
        all_pass &= check("pyrealsense2 imported", True)

        ctx = rs.context()
        devices = ctx.query_devices()
        num_cameras = len(devices)
        if num_cameras > 0:
            for dev in devices:
                name = dev.get_info(rs.camera_info.name)
                serial = dev.get_info(rs.camera_info.serial_number)
                check("Camera detected", True, f"{name} (S/N: {serial})")
        else:
            check("Camera detected", False, "Connect camera via USB3 and retry")
    except ImportError:
        msg = "Try: export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.10/site-packages"
        all_pass &= check("pyrealsense2 imported", False, msg)

    # --- Other Python packages ---
    print("\nPython Packages:")
    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("pingouin", "pingouin"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("yaml", "pyyaml"),
        ("cv2", "opencv"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
    ]
    for import_name, display_name in packages:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, '__version__', 'ok')
            check(display_name, True, ver)
        except ImportError:
            all_pass &= check(display_name, False, "not installed")

    # --- System tools ---
    print("\nSystem Tools:")

    # tegrastats
    tg_ok = os.path.exists("/usr/bin/tegrastats")
    all_pass &= check("tegrastats", tg_ok,
                       "/usr/bin/tegrastats" if tg_ok else "not found")

    # jtop
    jtop_ok = subprocess.run(["which", "jtop"],
                             capture_output=True).returncode == 0
    check("jtop", jtop_ok, "may need reboot" if not jtop_ok else "available")

    # Network
    print("\nNetwork:")
    try:
        result = subprocess.run(
            ["ip", "-4", "addr", "show"],
            capture_output=True, text=True
        )
        eth_lines = [l.strip() for l in result.stdout.split('\n')
                     if 'inet ' in l and '127.0.0.1' not in l]
        for line in eth_lines:
            check("Network interface", True, line.split()[1])
    except Exception:
        check("Network check", False, "could not query interfaces")

    # --- Memory check ---
    print("\nMemory:")
    try:
        import torch
        if torch.cuda.is_available():
            # Test if batch_size=16 with MobileNetV3 fits
            model = torchvision.models.mobilenet_v3_small(
                weights=None, num_classes=2
            ).cuda()
            # Simulate 5-channel input (RGB+D+IR early fusion)
            # Modify first conv layer
            old_conv = model.features[0][0]
            model.features[0][0] = torch.nn.Conv2d(
                5, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            ).cuda()

            x = torch.randn(16, 5, 224, 224, device='cuda')
            with torch.no_grad():
                out = model(x)

            mem_used = torch.cuda.max_memory_allocated() / 1e6
            check("Batch=16, 5ch MobileNetV3 (inference)", True,
                  f"peak GPU mem: {mem_used:.0f} MB")

            # Test training forward+backward
            torch.cuda.reset_peak_memory_stats()
            model.train()
            x = torch.randn(16, 5, 224, 224, device='cuda')
            out = model(x)
            loss = out.sum()
            loss.backward()

            mem_used = torch.cuda.max_memory_allocated() / 1e6
            check("Batch=16, 5ch MobileNetV3 (training)", True,
                  f"peak GPU mem: {mem_used:.0f} MB")

            del model, x, out, loss
            torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        check("Batch=16 memory test", False,
              "OOM! Reduce batch_size to 8 in configs/fl_config.yaml")
    except Exception as e:
        check("Memory test", False, str(e))

    # --- Summary ---
    print()
    print("=" * 60)
    if all_pass:
        print(f"  {PASS} All critical checks passed!")
    else:
        print(f"  {FAIL} Some checks failed — review above and fix before proceeding.")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
