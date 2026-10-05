"""FedRGBD -- the camera experiment's fixed image preprocessing.

``docs/CAMERA_EXPERIMENT_PREREG.md`` section 3: every frame is resized so that its
shorter side is 224 px and centre-cropped to 224 x 224; the sensors' different
resolutions and fields of view are part of the sensor shift and are not corrected
further.  Depth (RGB-D, secondary) is aligned to the colour image at capture time,
clipped to 0.3-10 m, scaled to [0, 1], invalid pixels 0.  Both questions of the
pre-registration use exactly these functions, so the leave-one-scene-out runs and the
federated runs see the same pixels.

Run once, on one machine (prereg Amendment 2, 2026-10-05): the same functions in two
Pillow versions (the Jetsons pin 9.0.1, the desktop has 12.x) are not guaranteed to give
the same pixels, so ``scripts/camera_preprocess_frames.py`` writes every valid frame's
preprocessed streams once, on the desktop, with an md5 manifest
(``data/splits_camera/preprocessed_manifest.csv``, committed).  Every consumer -- the
federated folds on the nodes (``camera_fl_prepare.py``), the desktop baselines that train
on those folds, and the desktop LOSO / RGB-D runs (``CustomRGBDDataset(preprocess=
"camera")``) -- reads those files through :class:`PreprocessedStore`, which refuses any
file whose md5 differs from the manifest.  Nothing downstream re-decodes or resizes a
native-resolution frame.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image

SIZE = 224
DEPTH_MIN_MM = 300.0
DEPTH_MAX_MM = 10000.0

#: where the preprocessed streams are written (gitignored) and their committed manifest
PREPROCESSED_DIR = os.path.join("data", "processed", "camera_224")
PREPROCESSED_MANIFEST = os.path.join("data", "splits_camera", "preprocessed_manifest.csv")
#: stream -> file suffix: RGB/IR as uint8 PNG (lossless), depth as float32 .npy (exact)
STREAM_SUFFIX = {"rgb": "_rgb.png", "ir": "_ir.png", "depth": "_depth.npy"}
MANIFEST_COLUMNS = ("node", "id", "stream", "file", "img_size", "bytes", "md5")


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


# --------------------------------------------------------------------------- #
# preprocessed files: written once, read everywhere, md5-verified (Amendment 2)
# --------------------------------------------------------------------------- #
def preprocessed_file(node: str, frame_id: str, stream: str) -> str:
    """Path of a preprocessed stream below the preprocessed root, forward slashes."""
    return "%s/%s%s" % (node, frame_id, STREAM_SUFFIX[stream])


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def encode_stream(stream: str, arr: np.ndarray) -> bytes:
    """The file bytes of one preprocessed stream (PNG for RGB/IR, .npy for depth)."""
    buf = io.BytesIO()
    if stream == "depth":
        np.save(buf, np.ascontiguousarray(arr, dtype=np.float32), allow_pickle=False)
    else:
        mode = "RGB" if stream == "rgb" else "L"
        Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8), mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def decode_stream(stream: str, blob: bytes) -> np.ndarray:
    """RGB [H, W, 3] uint8, IR [H, W] uint8, depth [H, W] float32 in [0, 1]."""
    if stream == "depth":
        return np.load(io.BytesIO(blob), allow_pickle=False)
    with Image.open(io.BytesIO(blob)) as im:
        return np.asarray(im.convert("RGB" if stream == "rgb" else "L"), dtype=np.uint8)


def manifest_bytes(rows: Iterable[Dict[str, object]]) -> bytes:
    """The manifest, sorted by (node, id, stream), LF line endings."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(MANIFEST_COLUMNS)
    for r in sorted(rows, key=lambda r: (str(r["node"]), str(r["id"]), str(r["stream"]))):
        w.writerow([r[c] for c in MANIFEST_COLUMNS])
    return buf.getvalue().encode("utf-8")


def read_manifest(path: str) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    """{(node, id, stream): row} of a preprocessed manifest."""
    if not os.path.isfile(path):
        raise FileNotFoundError("%s is missing -- write it with scripts/camera_preprocess_frames.py "
                                "(prereg Amendment 2: camera frames are preprocessed once)" % path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        key = (r["node"], r["id"], r["stream"])
        if key in out:
            raise ValueError("%s: %s listed twice" % (path, "/".join(key)))
        r["img_size"], r["bytes"] = int(r["img_size"]), int(r["bytes"])
        out[key] = r
    return out


class PreprocessedStore:
    """Read-only access to the preprocessed streams; every read is md5-checked."""

    def __init__(self, root: str = PREPROCESSED_DIR, manifest: str = PREPROCESSED_MANIFEST):
        self.root = root
        self.manifest_path = manifest
        self.rows = read_manifest(manifest)
        sizes = {r["img_size"] for r in self.rows.values()}
        if len(sizes) > 1:
            raise ValueError("%s mixes image sizes %s" % (manifest, sorted(sizes)))
        self.img_size: Optional[int] = sizes.pop() if sizes else None

    def has(self, node: str, frame_id: str, stream: str) -> bool:
        return (node, frame_id, stream) in self.rows

    def path(self, node: str, frame_id: str, stream: str) -> str:
        row = self.rows.get((node, frame_id, stream))
        if row is None:
            raise KeyError("%s: no preprocessed %s stream for %s/%s"
                           % (self.manifest_path, stream, node, frame_id))
        return os.path.join(self.root, *str(row["file"]).split("/"))

    def read_bytes(self, node: str, frame_id: str, stream: str) -> bytes:
        """The file's bytes; raises if its md5 is not the manifest's."""
        path = self.path(node, frame_id, stream)
        with open(path, "rb") as f:
            blob = f.read()
        got, want = hashlib.md5(blob).hexdigest(), self.rows[(node, frame_id, stream)]["md5"]
        if got != want:
            raise ValueError("%s: md5 %s, the manifest says %s -- not the file preprocessed "
                             "on the desktop (prereg Amendment 2)" % (path, got, want))
        return blob

    def load(self, node: str, frame_id: str, stream: str) -> np.ndarray:
        return decode_stream(stream, self.read_bytes(node, frame_id, stream))
