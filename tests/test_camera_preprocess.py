"""The camera experiment's fixed preprocessing (docs/CAMERA_EXPERIMENT_PREREG.md, sec. 3)."""

import numpy as np
from PIL import Image

from src.data.camera_preprocess import DEPTH_MAX_MM, DEPTH_MIN_MM, depth_224, rgb_224


def test_every_sensor_resolution_becomes_224_square():
    for w, h in ((1920, 1080), (2208, 1242), (1280, 720), (640, 480), (224, 224)):
        assert rgb_224(Image.new("RGB", (w, h))).size == (224, 224)
        assert depth_224(np.zeros((h, w), np.uint16)).shape == (224, 224)


def test_centre_crop_keeps_the_centre_not_the_edges():
    arr = np.zeros((1080, 1920, 3), np.uint8)
    arr[:, :240] = 255                        # left edge white: must be cropped away
    arr[:, 840:1080] = 128                    # centre band grey: must survive
    out = np.asarray(rgb_224(Image.fromarray(arr)))
    assert out[:, 0].max() < 50
    assert abs(int(out[112, 112, 0]) - 128) <= 2


def test_depth_scaling_and_invalid_pixels():
    d = np.full((480, 640), 5000, np.uint16)
    d[:, :320] = 0                            # invalid (no depth)
    d[0:10, 400:410] = 20000                  # beyond range: invalid
    out = depth_224(d)
    assert out.dtype == np.float32 and out.min() >= 0.0 and out.max() <= 1.0
    want = (5000 - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM)
    assert np.isclose(out[112, 200], want)
    assert out[112, 20] == 0.0
