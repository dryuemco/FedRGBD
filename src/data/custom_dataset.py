"""FedRGBD — Dataset for the custom multi-camera RGB-D captures (Phase B).

The custom captures live under ``data/raw/custom/node_{a,b,c}/`` and contain,
per synchronized frame ``{id}``:

    {id}_rgb.png     8-bit RGB
    {id}_depth.png   16-bit depth in millimetres
    {id}_ir.png      8-bit IR (RealSense nodes only; the ZED node has none)
    {id}_meta.json   timestamp / camera / intrinsics (see realsense_capture.py)

Two things this module provides:

* :func:`load_frame_index` — scans the capture tree and returns one dict per
  frame with the ``scene`` and ``label`` annotation resolved from an
  **explicit** source (meta.json → labels.csv → filename pattern).  If no
  source can annotate every frame it raises with a report of what was found;
  it never guesses silently.
* :class:`CustomRGBDDataset` — a ``torch.utils.data.Dataset`` that stacks the
  requested modalities into a single ``[C, H, W]`` tensor for the early-fusion
  MobileNetV3 (``src.models.mobilenetv3_multimodal.create_model``).

Scene-independent (leave-one-scene-out) evaluation is driven by the ``scene``
field; see ``scripts/cross_sensor_loso.py``.
"""

from __future__ import annotations

import csv
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# modality / normalisation configuration
# --------------------------------------------------------------------------- #

#: channel count per modality (mirrors configs/model_config.yaml input_channels)
MODALITY_CHANNELS: Dict[str, int] = {
    "rgb": 3,
    "depth": 1,
    "ir": 1,
    "rgb_d": 4,
    "rgb_d_ir": 5,
}

MODALITIES: List[str] = list(MODALITY_CHANNELS)

#: which raw streams each modality needs
MODALITY_STREAMS: Dict[str, List[str]] = {
    "rgb": ["rgb"],
    "depth": ["depth"],
    "ir": ["ir"],
    "rgb_d": ["rgb", "depth"],
    "rgb_d_ir": ["rgb", "depth", "ir"],
}

# configs/model_config.yaml → training.normalize_*
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEPTH_MEAN = 0.5
DEPTH_STD = 0.25
IR_MEAN = 0.5
IR_STD = 0.25

#: default clipping range for depth, in metres (16-bit mm PNG → metres)
DEFAULT_MAX_DEPTH_M = 10.0

#: canonical binary label names → class index (same convention as FlameDataset)
LABEL_ALIASES: Dict[str, int] = {
    "no_fire": 0, "nofire": 0, "no-fire": 0, "normal": 0, "negative": 0, "0": 0,
    "fire": 1, "anomaly": 1, "abnormal": 1, "positive": 1, "1": 1,
}

#: ``<scene>_<label>_<n>`` fallback, e.g. ``lab_fire_00017`` or
#: ``kitchen_no_fire_004`` (label may itself contain underscores).
_FILENAME_RE = re.compile(r"^(?P<scene>[^_]+)_(?P<label>.+)_(?P<n>\d+)$")


# --------------------------------------------------------------------------- #
# frame index
# --------------------------------------------------------------------------- #
def _read_meta(path: str) -> Dict[str, object]:
    try:
        with open(path) as f:
            meta = json.load(f)
        return meta if isinstance(meta, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _read_labels_csv(labels_csv: str) -> Dict[tuple, Dict[str, str]]:
    """Read ``id,scene,label`` (optional ``node``) into ``{(node, id): row}``.

    Rows without a ``node`` column are stored under ``(None, id)`` and match a
    frame of any node with that id.
    """
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        cols = [c.strip() for c in (reader.fieldnames or [])]
        missing = [c for c in ("id", "scene", "label") if c not in cols]
        if missing:
            raise ValueError(
                f"labels_csv {labels_csv!r} is missing required column(s) "
                f"{missing}; found columns {cols}. Expected: id,scene,label[,node]"
            )
        table: Dict[tuple, Dict[str, str]] = {}
        for row in reader:
            row = {(k.strip() if k else k): (v.strip() if isinstance(v, str) else v)
                   for k, v in row.items()}
            fid = row.get("id")
            if not fid:
                continue
            node = row.get("node") or None
            table[(node, fid)] = row
    return table


def _from_filename(frame_id: str) -> Optional[Dict[str, str]]:
    m = _FILENAME_RE.match(frame_id)
    if not m:
        return None
    return {"scene": m.group("scene"), "label": m.group("label")}


def _discover_nodes(root: str, nodes: Optional[Sequence[str]] = None) -> List[str]:
    """Node sub-directory names under ``root`` (or ``["."]`` for a flat dir)."""
    if nodes:
        return list(nodes)
    entries = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    )
    if entries:
        return entries
    return ["."]


def build_label_map(label_names: Iterable[str]) -> Dict[str, int]:
    """Map label *names* to integer class indices.

    Known binary aliases (``fire``/``no_fire``/...) keep the repository-wide
    convention ``No_Fire = 0``, ``Fire = 1``.  Any other label vocabulary is
    mapped in sorted order, which is deterministic across runs and machines.
    """
    names = sorted({str(n) for n in label_names})
    keys = [n.strip().lower().replace(" ", "_") for n in names]
    if keys and all(k in LABEL_ALIASES for k in keys):
        return {n: LABEL_ALIASES[k] for n, k in zip(names, keys)}
    return {n: i for i, n in enumerate(names)}


def load_frame_index(
    root: str,
    labels_csv: Optional[str] = None,
    nodes: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    """Scan a custom-capture tree and return one annotated dict per frame.

    Args:
        root: capture root, e.g. ``data/raw/custom`` (containing ``node_*/``)
            or a single node directory.
        labels_csv: optional CSV with columns ``id,scene,label`` (plus an
            optional ``node`` column to disambiguate ids repeated per node).
        nodes: restrict the scan to these node sub-directories.

    Returns:
        A list of dicts with keys ``uid, id, node, scene, label, label_name,
        label_source, has_depth, has_ir, path_rgb, path_depth, path_ir,
        path_meta``, sorted by ``(node, id)``.  ``label`` is the integer class
        index, ``label_name`` the raw annotation string.

    Raises:
        FileNotFoundError: ``root`` does not exist or contains no frames.
        ValueError: at least one frame could not be annotated; the message
            reports exactly which sources were tried and what they contained.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"capture root not found: {root}")

    csv_table: Dict[tuple, Dict[str, str]] = {}
    if labels_csv:
        if not os.path.isfile(labels_csv):
            raise FileNotFoundError(f"labels_csv not found: {labels_csv}")
        csv_table = _read_labels_csv(labels_csv)

    node_names = _discover_nodes(root, nodes)
    records: List[Dict[str, object]] = []
    unresolved: List[str] = []
    meta_keys_seen: set = set()
    source_counts = {"meta_json": 0, "labels_csv": 0, "filename": 0}

    for node in node_names:
        node_dir = root if node == "." else os.path.join(root, node)
        node_name = os.path.basename(os.path.normpath(root)) if node == "." else node
        if not os.path.isdir(node_dir):
            raise FileNotFoundError(f"node directory not found: {node_dir}")
        rgb_files = sorted(f for f in os.listdir(node_dir) if f.endswith("_rgb.png"))
        for fname in rgb_files:
            frame_id = fname[: -len("_rgb.png")]
            path_rgb = os.path.join(node_dir, fname)
            path_depth = os.path.join(node_dir, f"{frame_id}_depth.png")
            path_ir = os.path.join(node_dir, f"{frame_id}_ir.png")
            path_meta = os.path.join(node_dir, f"{frame_id}_meta.json")

            meta = _read_meta(path_meta) if os.path.isfile(path_meta) else {}
            meta_keys_seen.update(meta.keys())

            scene = label_name = None
            source = None
            if meta.get("scene") is not None and meta.get("label") is not None:
                scene, label_name, source = meta["scene"], meta["label"], "meta_json"
            elif csv_table:
                row = csv_table.get((node_name, frame_id)) or csv_table.get((None, frame_id))
                if row:
                    scene, label_name, source = row["scene"], row["label"], "labels_csv"
            if scene is None:
                parsed = _from_filename(frame_id)
                if parsed:
                    scene, label_name, source = parsed["scene"], parsed["label"], "filename"

            if scene is None or label_name is None:
                unresolved.append(f"{node_name}/{frame_id}")
                continue

            source_counts[source] += 1
            records.append({
                "uid": f"{node_name}/{frame_id}",
                "id": frame_id,
                "node": node_name,
                "scene": str(scene),
                "label_name": str(label_name),
                "label": None,  # filled in below, once the vocabulary is known
                "label_source": source,
                "has_depth": os.path.isfile(path_depth),
                "has_ir": os.path.isfile(path_ir),
                "path_rgb": path_rgb,
                "path_depth": path_depth if os.path.isfile(path_depth) else None,
                "path_ir": path_ir if os.path.isfile(path_ir) else None,
                "path_meta": path_meta if os.path.isfile(path_meta) else None,
            })

    n_frames = len(records) + len(unresolved)
    if n_frames == 0:
        raise FileNotFoundError(
            f"no '*_rgb.png' frames found under {root!r} "
            f"(scanned node directories: {node_names}). "
            "Expected data/raw/custom/node_x/{id}_rgb.png — see data/README.md."
        )

    if unresolved:
        raise ValueError(_annotation_error(
            root, node_names, n_frames, unresolved, source_counts,
            meta_keys_seen, labels_csv, csv_table,
        ))

    label_map = build_label_map(r["label_name"] for r in records)
    for r in records:
        r["label"] = label_map[r["label_name"]]

    records.sort(key=lambda r: (r["node"], r["id"]))
    return records


def _annotation_error(root, node_names, n_frames, unresolved, source_counts,
                      meta_keys_seen, labels_csv, csv_table) -> str:
    sample = ", ".join(unresolved[:10]) + (" ..." if len(unresolved) > 10 else "")
    csv_desc = (
        f"{labels_csv!r} with {len(csv_table)} row(s)" if labels_csv
        else "not provided (pass --labels_csv / labels_csv=...)"
    )
    return (
        f"Could not determine (scene, label) for {len(unresolved)} of {n_frames} "
        f"frame(s) under {root!r}.\n"
        f"  node directories scanned : {node_names}\n"
        f"  unresolved frames        : {sample}\n"
        f"  resolved by meta.json    : {source_counts['meta_json']}\n"
        f"  resolved by labels.csv   : {source_counts['labels_csv']}\n"
        f"  resolved by filename     : {source_counts['filename']}\n"
        f"  keys seen in meta.json   : {sorted(meta_keys_seen) or '(no meta.json found)'}\n"
        f"  labels_csv               : {csv_desc}\n"
        "Annotate the frames with ONE of:\n"
        "  1. \"scene\" and \"label\" keys inside each {id}_meta.json, or\n"
        "  2. a CSV with columns id,scene,label (optional node column) passed "
        "via --labels_csv, or\n"
        "  3. frame ids following the <scene>_<label>_<n> pattern "
        "(e.g. lab_fire_00017, kitchen_no_fire_004).\n"
        "Scene/label are never inferred implicitly — a scene-independent "
        "(leave-one-scene-out) evaluation is only meaningful with an explicit "
        "scene annotation."
    )


# --------------------------------------------------------------------------- #
# index helpers
# --------------------------------------------------------------------------- #
def required_streams(modality: str) -> List[str]:
    """Raw streams needed by ``modality`` (``rgb`` / ``depth`` / ``ir``)."""
    if modality not in MODALITY_STREAMS:
        raise ValueError(f"unknown modality {modality!r}; choose from {MODALITIES}")
    return list(MODALITY_STREAMS[modality])


def index_supports_modality(record: Dict[str, object], modality: str) -> bool:
    """True if this frame has every file the modality needs."""
    streams = required_streams(modality)
    if "depth" in streams and not record.get("has_depth"):
        return False
    if "ir" in streams and not record.get("has_ir"):
        return False
    return os.path.isfile(str(record["path_rgb"])) if "rgb" in streams else True


def filter_index(
    index: Sequence[Dict[str, object]],
    nodes: Optional[Sequence[str]] = None,
    scenes: Optional[Sequence[str]] = None,
    modality: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Subset a frame index by node, scene and modality availability."""
    node_set = set(nodes) if nodes else None
    scene_set = set(scenes) if scenes else None
    out = []
    for r in index:
        if node_set is not None and r["node"] not in node_set:
            continue
        if scene_set is not None and r["scene"] not in scene_set:
            continue
        if modality is not None and not index_supports_modality(r, modality):
            continue
        out.append(r)
    return out


def scenes_in(index: Sequence[Dict[str, object]]) -> List[str]:
    """Sorted unique scene names present in ``index``."""
    return sorted({str(r["scene"]) for r in index})


def class_distribution(index: Sequence[Dict[str, object]]) -> Dict[str, int]:
    """``{label_name: count}`` for a frame index."""
    counts: Dict[str, int] = {}
    for r in index:
        counts[str(r["label_name"])] = counts.get(str(r["label_name"]), 0) + 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------- #
# image loading
# --------------------------------------------------------------------------- #
def _resize_nearest(arr: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize that works for any numpy dtype (incl. uint16)."""
    h, w = arr.shape[:2]
    if (h, w) == (size, size):
        return arr
    yi = np.minimum((np.arange(size) * h) // size, h - 1)
    xi = np.minimum((np.arange(size) * w) // size, w - 1)
    return arr[yi][:, xi]


def load_rgb(path: str, img_size: int) -> np.ndarray:
    """RGB image as ``[H, W, 3]`` float32 in ``[0, 1]``."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def load_depth_m(path: str, img_size: int, max_depth_m: float = DEFAULT_MAX_DEPTH_M) -> np.ndarray:
    """16-bit millimetre depth PNG as ``[H, W]`` float32 metres, clipped."""
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:  # some writers store depth as a multi-channel PNG
        arr = arr[..., 0]
    arr = _resize_nearest(arr, img_size).astype(np.float32) / 1000.0
    return np.clip(arr, 0.0, float(max_depth_m))


def load_ir(path: str, img_size: int) -> np.ndarray:
    """8-bit IR image as ``[H, W]`` float32 in ``[0, 1]``."""
    with Image.open(path) as im:
        im = im.convert("L").resize((img_size, img_size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class CustomRGBDDataset(Dataset):
    """Synchronized multi-modal frames from the custom RGB-D captures.

    Args:
        root: capture root (``data/raw/custom``) or a single node directory.
        ids: frame identifiers to include, either ``"<node>/<id>"`` (preferred,
            unambiguous) or a bare ``"<id>"`` which selects that frame in every
            node.  ``None`` selects every frame of the index.
        modality: one of ``rgb, depth, ir, rgb_d, rgb_d_ir``.
        img_size: square input resolution (224 = paper setting).
        train: enables light augmentation (horizontal flip applied jointly to
            all modalities + RGB-only brightness/contrast jitter).
        index: a pre-built index from :func:`load_frame_index` (avoids
            re-scanning the tree for every fold); built on demand otherwise.
        labels_csv: forwarded to :func:`load_frame_index` when ``index`` is None.
        class_to_idx: fixed ``{label_name: class index}`` mapping, so every
            fold of a LOSO run shares one label vocabulary.
        max_depth_m: depth clipping range in metres (default 10).
        seed: base seed for the augmentation RNG.

    Each item is ``(tensor[C, H, W] float32, label int)`` where ``C`` is
    ``MODALITY_CHANNELS[modality]``.  RGB channels use ImageNet statistics;
    depth is converted mm → metres, clipped at ``max_depth_m``, scaled to
    ``[0, 1]`` and normalised with ``configs/model_config.yaml``'s
    ``normalize_depth``; IR is scaled to ``[0, 1]`` and normalised with
    ``normalize_ir``.
    """

    def __init__(
        self,
        root: str,
        ids: Optional[Sequence[str]] = None,
        modality: str = "rgb",
        img_size: int = 224,
        train: bool = False,
        index: Optional[Sequence[Dict[str, object]]] = None,
        labels_csv: Optional[str] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        max_depth_m: float = DEFAULT_MAX_DEPTH_M,
        seed: int = 42,
    ):
        if modality not in MODALITY_CHANNELS:
            raise ValueError(f"unknown modality {modality!r}; choose from {MODALITIES}")

        self.root = root
        self.modality = modality
        self.in_channels = MODALITY_CHANNELS[modality]
        self.img_size = int(img_size)
        self.train = bool(train)
        self.max_depth_m = float(max_depth_m)
        self.seed = int(seed)
        self.streams = required_streams(modality)

        full_index = list(index) if index is not None else load_frame_index(root, labels_csv)
        self.class_to_idx = dict(class_to_idx) if class_to_idx else build_label_map(
            r["label_name"] for r in full_index
        )

        if ids is None:
            selected = list(full_index)
        else:
            by_uid = {str(r["uid"]): r for r in full_index}
            by_id: Dict[str, List[Dict[str, object]]] = {}
            for r in full_index:
                by_id.setdefault(str(r["id"]), []).append(r)
            selected, missing = [], []
            for key in ids:
                key = str(key)
                if key in by_uid:
                    selected.append(by_uid[key])
                elif key in by_id:
                    selected.extend(by_id[key])
                else:
                    missing.append(key)
            if missing:
                raise KeyError(
                    f"{len(missing)} requested id(s) are not in the frame index "
                    f"of {root!r}, e.g. {missing[:5]}"
                )

        unsupported = [r for r in selected if not index_supports_modality(r, modality)]
        if unsupported:
            raise FileNotFoundError(
                f"{len(unsupported)} frame(s) lack the files required by modality "
                f"{modality!r} (needs {self.streams}), e.g. "
                f"{[r['uid'] for r in unsupported[:5]]}. The ZED node has no IR "
                "stream — filter the index with filter_index(..., modality=...) "
                "before building the dataset."
            )

        self.records = selected
        self.ids = [str(r["uid"]) for r in selected]

    # -- introspection ----------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.records)

    def get_class_distribution(self) -> Dict[str, int]:
        return class_distribution(self.records)

    def get_scenes(self) -> List[str]:
        return scenes_in(self.records)

    def get_nodes(self) -> List[str]:
        return sorted({str(r["node"]) for r in self.records})

    # -- loading ----------------------------------------------------------- #
    def _augment(self, rgb, depth, ir, idx):
        rng = np.random.RandomState((self.seed * 1_000_003 + idx) % (2 ** 31 - 1))
        if rng.rand() < 0.5:  # joint horizontal flip (geometry stays aligned)
            rgb = rgb[:, ::-1] if rgb is not None else None
            depth = depth[:, ::-1] if depth is not None else None
            ir = ir[:, ::-1] if ir is not None else None
        if rgb is not None:  # photometric jitter on RGB only
            brightness = 1.0 + rng.uniform(-0.2, 0.2)
            contrast = 1.0 + rng.uniform(-0.2, 0.2)
            mean = float(rgb.mean())
            rgb = np.clip((rgb * brightness - mean) * contrast + mean, 0.0, 1.0)
        return rgb, depth, ir

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        rgb = depth = ir = None
        if "rgb" in self.streams:
            rgb = load_rgb(str(rec["path_rgb"]), self.img_size)
        if "depth" in self.streams:
            depth = load_depth_m(str(rec["path_depth"]), self.img_size, self.max_depth_m)
        if "ir" in self.streams:
            ir = load_ir(str(rec["path_ir"]), self.img_size)

        if self.train:
            rgb, depth, ir = self._augment(rgb, depth, ir, idx)

        channels: List[np.ndarray] = []
        if rgb is not None:
            for c in range(3):
                channels.append((rgb[..., c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c])
        if depth is not None:
            channels.append((depth / self.max_depth_m - DEPTH_MEAN) / DEPTH_STD)
        if ir is not None:
            channels.append((ir - IR_MEAN) / IR_STD)

        tensor = torch.from_numpy(
            np.ascontiguousarray(np.stack(channels, axis=0), dtype=np.float32)
        )
        label = self.class_to_idx[str(rec["label_name"])]
        return tensor, int(label)


__all__ = [
    "CustomRGBDDataset",
    "DEFAULT_MAX_DEPTH_M",
    "MODALITIES",
    "MODALITY_CHANNELS",
    "MODALITY_STREAMS",
    "build_label_map",
    "class_distribution",
    "filter_index",
    "index_supports_modality",
    "load_depth_m",
    "load_frame_index",
    "load_ir",
    "load_rgb",
    "required_streams",
    "scenes_in",
]


if __name__ == "__main__":  # pragma: no cover — manual inspection helper
    import argparse

    p = argparse.ArgumentParser(description="Inspect a custom RGB-D capture tree")
    p.add_argument("--data_dir", default="data/raw/custom")
    p.add_argument("--labels_csv", default=None)
    p.add_argument("--modality", default="rgb_d", choices=MODALITIES)
    args = p.parse_args()

    idx = load_frame_index(args.data_dir, args.labels_csv)
    print(f"{len(idx)} frames, scenes={scenes_in(idx)}, "
          f"classes={class_distribution(idx)}")
    for node in sorted({r['node'] for r in idx}):
        sub = filter_index(idx, nodes=[node])
        usable = filter_index(sub, modality=args.modality)
        print(f"  {node}: {len(sub)} frames ({len(usable)} usable for {args.modality}), "
              f"scenes={scenes_in(sub)}, classes={class_distribution(sub)}")
