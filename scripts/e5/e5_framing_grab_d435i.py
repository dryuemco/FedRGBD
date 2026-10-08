# Framing check only (NOT e5 data): one D435i colour frame, auto exposure on, no lock.
# Writes ~/e5_framing/, never data/raw/camera_pilot.  Run from ~/FedRGBD on node_b.
import os, sys, time, json, hashlib
import numpy as np
import pyrealsense2 as rs
from PIL import Image, ImageDraw
sys.path.insert(0, os.getcwd())
from src.data.camera_preprocess import _crop_box, rgb_224, SIZE

SERIAL = "405622076256"
out = os.path.expanduser("~/e5_framing")
os.makedirs(out, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")

pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_device(SERIAL)
cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
prof = pipe.start(cfg)
try:
    for _ in range(60):                      # let auto exposure settle
        fs = pipe.wait_for_frames(5000)
    c = fs.get_color_frame()
    ae = c.get_frame_metadata(rs.frame_metadata_value.auto_exposure) \
        if c.supports_frame_metadata(rs.frame_metadata_value.auto_exposure) else None
    img = Image.fromarray(np.asanyarray(c.get_data()).copy(), "RGB")
finally:
    pipe.stop()

w, h = img.size
(nw, nh), (l, t, r, b) = _crop_box(w, h, SIZE)
k = w / float(nw)                             # resized -> native pixels
rect = (int(round(l * k)), int(round(t * h / float(nh))), int(round(r * k)) - 1,
        int(round(b * h / float(nh))) - 1)
full = img.copy()
d = ImageDraw.Draw(full)
for i in range(4):
    d.rectangle((rect[0] - i, rect[1] + i, rect[2] + i, rect[3] - i), outline=(255, 0, 0))
p1 = os.path.join(out, "framing_%s_d435i_full_%dx%d_crop224rect.png" % (stamp, w, h))
full.save(p1)
crop = rgb_224(img).resize((3 * SIZE, 3 * SIZE), Image.NEAREST)
p2 = os.path.join(out, "framing_%s_d435i_crop224_x3.png" % stamp)
crop.save(p2)
md5 = {p: hashlib.md5(open(p, "rb").read()).hexdigest() for p in (p1, p2)}
print(json.dumps({"framing_only_not_e5_data": True, "serial": SERIAL, "resolution": [w, h],
                  "auto_exposure_flag": ae, "crop_rect_native_xyxy": rect,
                  "files": md5}))
