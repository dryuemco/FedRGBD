"""FedRGBD -- the camera experiment's fixed image preprocessing.

``docs/CAMERA_EXPERIMENT_PREREG.md`` section 3: every frame is resized so that its
shorter side is 224 px and centre-cropped to 224 x 224; the sensors' different
resolutions and fields of view are part of the sensor shift and are not corrected
further.  Depth (RGB-D, secondary) is aligned to the colour image at capture time,
clipped to 0.3-10 m, scaled to [0, 1], invalid pixels 0.  Both questions of the
pre-registration use exactly these functions, so the leave-one-scene-out runs and the
federated runs see the same pixels.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

SIZE = 224
DEPTH_MIN_MM = 300.0
DEPTH_MAX_MM = 10000.0


def _crop_box(width: int, height: int, size: int = SIZE):
    """Scale factor and centre-crop box after resizing the shorter side to ``size``."""
    scale = size / float(min(width, height))
    new_w, new_h = max(size, int(round(width * scale))), max(size, int(round(height * scale)))
    left, top = (new_w - size) // 2, (new_h - size) // 2
    return (new_w, new_h), (left, top, left + size, top + size)


def rgb_224(img: Image.Image, size: int = SIZE) -> Image.Image:
    """Shorter side to ``size`` (bilinear), then the centred ``size`` x ``size`` crop."""
    img = img.convert("RGB")
    new_size, box = _crop_box(img.width, img.height, size)
    return img.resize(new_size, Image.BILINEAR).crop(box)


def depth_224(depth_mm: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Aligned 16-bit depth in mm -> float32 [size, size] in [0, 1]; invalid pixels 0.

    Nearest-neighbour resizing, so no depth value is invented between a surface and an
    invalid (zero) pixel.  Valid means DEPTH_MIN_MM <= d <= DEPTH_MAX_MM.
    """
    d = np.asarray(depth_mm, dtype=np.float32)
    h, w = d.shape
    new_size, box = _crop_box(w, h, size)
    d = np.asarray(Image.fromarray(d, mode="F").resize(new_size, Image.NEAREST).crop(box),
                   dtype=np.float32)
    valid = (d >= DEPTH_MIN_MM) & (d <= DEPTH_MAX_MM)
    out = np.zeros_like(d)
    out[valid] = (d[valid] - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM)
    return out
