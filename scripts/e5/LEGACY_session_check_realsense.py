# Session-start check (prereg 13.1, a626d06): fixed setting applied, flame-free
# verification measurement; 224 mean luma must lie within +-10 % of the calibration value.
# Run from ~/FedRGBD on the node:  python3 - <node> <serial> < session_check.py
import json, os, sys, time
sys.path.insert(0, os.getcwd())
from src.data.camera_exposure import exposure_path, load_wb_region, measure
from src.data.realsense_capture import RealSenseBackend

node, serial = "node_b", "405622076256"
root = "data/raw/camera_pilot"
cal = json.load(open(exposure_path(root, node)))
b = RealSenseBackend(serial=serial)
b.open()
try:
    m = measure(b, {"exposure": cal["exposure"], "gain": cal["gain"],
                    "white_balance": cal["white_balance"]}, cal.get("wb_region") or load_wb_region(node))
finally:
    b.close()
ref = float(cal["luma_224_median"])
lo, hi = 0.9 * ref, 1.1 * ref
ok = lo <= m["luma_224_median"] <= hi
print(json.dumps({"session_check": "PASS" if ok else "FAIL -> recalibrate", "node": node,
                  "time": time.strftime("%Y-%m-%d %H:%M:%S"), "setting": m["setting"],
                  "applied": m["applied"], "luma_224_median": m["luma_224_median"],
                  "calibration_luma_224": ref, "band": [round(lo, 3), round(hi, 3)],
                  "region_rgb_mean": m["region_rgb_mean"], "r_minus_b": m["r_minus_b"]}))
