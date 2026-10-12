# Test frame only (NOT e5 data): one D435if colour frame through the capture backend
# (1920x1080 rgb8, depth + IR streams on as in a capture), with the committed fixed
# setting of node_a applied (auto exposure and auto white balance off).
# Writes ~/e5_framing/, never data/raw/camera_pilot.  Run from ~/FedRGBD on node_a.
import os, sys, time, json, hashlib
sys.path.insert(0, os.getcwd())
from PIL import Image, ImageDraw
from src.data.realsense_capture import RealSenseBackend
from src.data.camera_exposure import fixed_exposure
from src.data.camera_preprocess import _crop_box, rgb_224, SIZE

SERIAL = "239722070442"
out = os.path.expanduser("~/e5_framing")
os.makedirs(out, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
b = RealSenseBackend(serial=SERIAL)
b.open()
try:
    info = b.info()
    fx = fixed_exposure("data/raw/camera_pilot", "node_a", info)
    applied = b.apply_lock({k: fx[k] for k in ("exposure", "gain", "white_balance")})
    for _ in range(30):
        b.skip()
    fr = b.grab()
finally:
    b.close()
img = Image.fromarray(fr.rgb, "RGB")
w, h = img.size
(nw, nh), (l, t, r, bt) = _crop_box(w, h, SIZE)
k = w / float(nw)
rect = (int(round(l * k)), int(round(t * h / float(nh))), int(round(r * k)) - 1,
        int(round(bt * h / float(nh))) - 1)
full = img.copy()
d = ImageDraw.Draw(full)
for i in range(4):
    d.rectangle((rect[0] - i, rect[1] + i, rect[2] + i, rect[3] - i), outline=(255, 0, 0))
p1 = os.path.join(out, "framing_%s_d435if_full_%dx%d_crop224rect.png" % (stamp, w, h))
full.save(p1)
p2 = os.path.join(out, "framing_%s_d435if_crop224_x3.png" % stamp)
rgb_224(img).resize((3 * SIZE, 3 * SIZE), Image.NEAREST).save(p2)
md5 = {p: hashlib.md5(open(p, "rb").read()).hexdigest() for p in (p1, p2)}
print(json.dumps({"framing_only_not_e5_data": True, "fixed_setting_applied": applied,
                  "serial": info.get("serial"), "firmware": info.get("firmware"),
                  "resolution": [w, h], "auto_exposure_flag": fr.meta.get("auto_exposure"),
                  "exposure": fr.meta.get("exposure"), "gain": fr.meta.get("gain"),
                  "white_balance": fr.meta.get("white_balance"),
                  "crop_rect_native_xyxy": rect, "files": md5}))
