"""scripts/camera_depth_holes.py: invalid-depth fraction only inside the e5 wall region."""

import json
import os

import numpy as np
from PIL import Image

from scripts import camera_depth_holes as cdh


def _write(root, node, cid, frames):
    d = os.path.join(root, node)
    os.makedirs(d, exist_ok=True)
    for i, a in enumerate(frames):
        Image.fromarray(a.astype(np.uint16)).save(os.path.join(d, "%s_%04d_depth.png" % (cid, i)))


def test_committed_region_is_the_declared_rectangle():
    r = cdh.load_region()
    assert (r["node"], r["x0"], r["x1"], r["y0"], r["y1"]) == ("node_a", 0.66, 0.95, 0.18, 0.60)


def test_holes_outside_the_region_do_not_count(tmp_path, capsys):
    r = cdh.load_region()
    h, w = 1080, 1920
    left, top, right, bottom = cdh.region_box(w, h, r)
    assert (left, top, right, bottom) == (1267, 194, 1824, 648)
    clean = np.full((h, w), 1500)
    clean[:, :left] = 0                      # holes everywhere left of the region
    clean[bottom:, :] = 0                    # and below it (the floor)
    holed = clean.copy()
    holed[top:top + (bottom - top) // 4, left:right] = 0   # a quarter of the region
    root = str(tmp_path / "camera_pilot")
    _write(root, "node_a", "s06_no_fire_d200", [holed, holed])
    _write(root, "node_a", "s07_no_fire_d200", [clean])
    per, s = cdh.measure(root, "node_a", "s06_no_fire_d200", r)
    assert len(per) == 2 and abs(s["invalid_mean"] - 0.25) < 0.005
    assert cdh.measure(root, "node_a", "s07_no_fire_d200", r)[1]["invalid_mean"] == 0.0
    assert cdh.main(["--root", root, "--captures", "s06_no_fire_d200", "s07_no_fire_d200"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("region (normalized) x [0.66, 0.95) y [0.18, 0.6)")
    assert json.loads(out[-1])["capture"] == "s07_no_fire_d200"


def test_the_study_footage_is_refused():
    assert cdh.main(["--root", os.path.join("data", "raw", "camera"), "--captures", "s06_no_fire_d200"]) == 2
