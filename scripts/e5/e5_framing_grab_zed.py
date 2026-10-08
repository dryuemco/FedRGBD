# Framing check only (NOT e5 data): one ZED 2i left colour frame through the capture
# backend (HD1080, NEURAL, 15 fps), camera settings as they are, no lock.
# Writes ~/e5_framing/, never data/raw/camera_pilot.  Run from ~/FedRGBD on node_c.
import os, sys, time, json, hashlib
sys.path.insert(0, os.getcwd())
from PIL import Image, ImageDraw
from src.data.zed_capture import ZedBackend
from src.data.camera_preprocess import _crop_box, rgb_224, SIZE

out = os.path.expanduser("~/e5_framing")
os.makedirs(out, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
b = ZedBackend()
b.open()
try:
    info = b.info()
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
p1 = os.path.join(out, "framing_%s_zed2i_full_%dx%d_crop224rect.png" % (stamp, w, h))
full.save(p1)
p2 = os.path.join(out, "framing_%s_zed2i_crop224_x3.png" % stamp)
rgb_224(img).resize((3 * SIZE, 3 * SIZE), Image.NEAREST).save(p2)
md5 = {p: hashlib.md5(open(p, "rb").read()).hexdigest() for p in (p1, p2)}
print(json.dumps({"framing_only_not_e5_data": True, "serial": info.get("serial"),
                  "resolution": [w, h], "auto_exposure_flag": fr.meta.get("auto_exposure"),
                  "exposure": fr.meta.get("exposure"), "gain": fr.meta.get("gain"),
                  "white_balance": fr.meta.get("white_balance"),
                  "crop_rect_native_xyxy": rect, "files": md5}))
