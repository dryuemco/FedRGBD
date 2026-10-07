#!/usr/bin/env python3
"""FedRGBD -- SYNTHETIC end-to-end dry run of the camera experiment's desktop pipeline.

Random frames at the real study size (S scenes x 2 classes x 3 distances x 40 frames per
camera, native resolutions: RGB 1920x1080 on all three cameras, depth aligned to it,
IR 1280x720 on the two RealSense cameras), pushed through every desktop step with the
pre-registered settings, each step timed:

  labels -> manifests -> preprocessing (224) -> LOSO (rgb with the random split, rgb_d,
  ir; 15 epochs, seeds 42/123/456, ImageNet weights) -> analysis (a) with B = 10,000 ->
  tab:loso -> federated folds -> desktop baselines (centralized, local-only; 50 epochs,
  5 folds x 3 seeds) -> their per-image predictions.

Everything is written under ``--work`` (outside the repository) and marked SYNTHETIC; no
real or pilot frame is read, nothing goes to results/, analysis/ or paper/.  The frames
are random: any number this produces is meaningless and is never reported as a result --
only "runs end to end" and the wall-clock times are.

    python scripts/camera_synthetic_dryrun.py generate --work D:/synth --scenes 20
    python scripts/camera_synthetic_dryrun.py subset --work D:/synth --from_scenes 20 --scenes 15
    python scripts/camera_synthetic_dryrun.py run --work D:/synth --scenes 20
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = ("node_a", "node_b", "node_c")
LABELS = ("fire", "no_fire")
DISTANCES_CM = (100, 200, 300)
FRAMES = 40
RGB_RES = (1920, 1080)
IR_RES = (1280, 720)
SEEDS = ("42", "123", "456")
MARK = "SYNTHETIC -- random frames, no real or pilot data; numbers are meaningless\n"


def tree_dir(work: str, scenes: int) -> str:
    return os.path.join(work, "S%d" % scenes, "camera")


#: --overlap: per-camera factor on the probability that a fire frame shows the signal
OVERLAP_CAMERA_FACTOR = {"node_a": 1.0, "node_b": 0.9, "node_c": 0.6}


def overlap_signal_prob(node: str, scene: int) -> float:
    """--overlap: probability that a fire frame of this scene and camera shows the flame
    signal (per scene 0.3-0.9, times the camera factor); the other fire frames are drawn
    exactly like no-fire frames, so the classes overlap."""
    q = 0.3 + 0.6 * float(np.random.default_rng([99, scene]).random())
    return q * OVERLAP_CAMERA_FACTOR[node]


def _frame(node: str, scene: int, label: str, cm: int, idx: int, out_dir: str,
           overlap: bool = False) -> None:
    from PIL import Image
    rng = np.random.default_rng([NODES.index(node), scene, LABELS.index(label), cm, idx])
    w, h = RGB_RES
    # low-frequency random content (compresses, non-constant), upscaled to native size
    small = rng.integers(0, 256, size=(18, 32, 3), dtype=np.uint8)
    if label == "fire" and not overlap:
        small[..., 0] = np.maximum(small[..., 0], 180)
    elif label == "fire" and rng.random() < overlap_signal_prob(node, scene):
        cells = rng.random((18, 32)) < 0.15          # weak signal: 15 % of the cells
        small[..., 0][cells] = np.maximum(small[..., 0][cells], 180)
    rgb = Image.fromarray(small, "RGB").resize((w, h), Image.NEAREST)
    cap = "s%02d_%s_d%d" % (scene, label, cm)
    fid = "%s_%04d" % (cap, idx)
    rgb.save(os.path.join(out_dir, fid + "_rgb.png"), compress_level=1)
    depth = (cm * 10 + rng.integers(0, 400, size=(18, 32))).astype(np.uint16)
    Image.fromarray(depth).resize((w, h), Image.NEAREST).save(
        os.path.join(out_dir, fid + "_depth.png"), compress_level=1)
    if node != "node_c":
        ir = rng.integers(0, 256, size=(18, 32), dtype=np.uint8)
        Image.fromarray(ir, "L").resize(IR_RES, Image.NEAREST).save(
            os.path.join(out_dir, fid + "_ir.png"), compress_level=1)
    meta = {"frame_id": fid, "capture_id": cap, "scene": "s%02d" % scene, "label": label,
            "distance_m": cm / 100.0, "frame_index": idx, "node": node,
            "serial": "SYNTHETIC-" + node, "timestamp_unix": 1.79e9 + idx * 0.2,
            "synthetic": True, "synthetic_overlap": bool(overlap)}
    with open(os.path.join(out_dir, fid + "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def _capture(args) -> None:
    node, scene, label, cm, out_dir, overlap = args
    for i in range(FRAMES):
        _frame(node, scene, label, cm, i, out_dir, overlap)
    cap = "s%02d_%s_d%d" % (scene, label, cm)
    with open(os.path.join(out_dir, "_captures", cap + ".json"), "w", encoding="utf-8") as f:
        json.dump({"capture_id": cap, "n_frames_written": FRAMES, "synthetic": True}, f)


def generate(work: str, scenes: int, workers: int = 12, overlap: bool = False) -> None:
    """``overlap``: classes overlap (weak flame signal in only part of the fire frames, per
    scene and camera, see ``overlap_signal_prob``), so balanced accuracy is well below 100 %
    and differs between scenes and cameras; otherwise every fire frame carries the signal."""
    root = tree_dir(work, scenes)
    jobs = []
    for node in NODES:
        d = os.path.join(root, node)
        os.makedirs(os.path.join(d, "_captures"), exist_ok=True)
        for s in range(1, scenes + 1):
            for label in LABELS:
                for cm in DISTANCES_CM:
                    jobs.append((node, s, label, cm, d, overlap))
    with open(os.path.join(work, "SYNTHETIC.txt"), "w", encoding="utf-8") as f:
        f.write(MARK)
        if overlap:
            f.write("overlapping classes (--overlap): weak flame signal in part of the fire "
                    "frames\n")
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_capture, jobs, chunksize=4))


def subset(work: str, from_scenes: int, scenes: int) -> None:
    """The first ``scenes`` scenes of a generated tree, as hard links (no new pixels)."""
    src, dst = tree_dir(work, from_scenes), tree_dir(work, scenes)
    keep = tuple("s%02d_" % s for s in range(1, scenes + 1))
    for node in NODES:
        for sub in ("", "_captures"):
            os.makedirs(os.path.join(dst, node, sub), exist_ok=True)
            for name in os.listdir(os.path.join(src, node, sub)):
                p = os.path.join(src, node, sub, name)
                if name.startswith(keep) and os.path.isfile(p):
                    q = os.path.join(dst, node, sub, name)
                    if not os.path.exists(q):
                        os.link(p, q)


# --------------------------------------------------------------------------- run
class Timer:
    def __init__(self, path: str):
        self.path = path
        self.steps: List[Dict[str, object]] = []
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                self.steps = json.load(f)["steps"]

    def done(self, name: str) -> bool:
        return any(s["step"] == name and s["rc"] == 0 for s in self.steps)

    def run(self, name: str, argv: Sequence[str]) -> int:
        if self.done(name):
            print("[skip] %s (already done)" % name, flush=True)
            return 0
        print("[%s] %s: %s" % (time.strftime("%H:%M:%S"), name, " ".join(argv)), flush=True)
        t0 = time.perf_counter()
        rc = subprocess.run(list(argv), cwd=REPO).returncode
        dt = time.perf_counter() - t0
        self.steps = [s for s in self.steps if s["step"] != name]
        self.steps.append({"step": name, "rc": rc, "seconds": round(dt, 1),
                           "ended": time.strftime("%Y-%m-%d %H:%M:%S")})
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"note": MARK.strip(), "steps": self.steps,
                       "total_seconds": round(sum(s["seconds"] for s in self.steps), 1)},
                      f, indent=2)
        print("    rc=%d  %.1f s" % (rc, dt), flush=True)
        if rc != 0:
            raise SystemExit("step %s failed (rc=%d)" % (name, rc))
        return rc


def run(work: str, scenes: int, python: str, B: int = 10000) -> None:
    base = os.path.join(work, "S%d" % scenes)
    root = tree_dir(work, scenes)
    splits = os.path.join(base, "splits_camera")
    pre = os.path.join(base, "camera_224")
    manifest = os.path.join(splits, "preprocessed_manifest.csv")
    loso_out = os.path.join(base, "SYNTHETIC_results", "loso")
    ana = os.path.join(base, "SYNTHETIC_analysis", "a")
    folds_out = os.path.join(base, "processed")
    base_out = os.path.join(base, "SYNTHETIC_results", "desktop")
    labels = os.path.join(root, "labels.csv")
    t = Timer(os.path.join(base, "SYNTHETIC_timings.json"))
    py = [python]
    t.run("labels", py + ["scripts/camera_labels.py", "--data_dir", root, "--splits_dir", splits])
    t.run("manifests", py + ["scripts/camera_manifests.py", "--labels_csv", labels,
                             "--splits_dir", splits])
    t.run("preprocess", py + ["scripts/camera_preprocess_frames.py", "--raw_dir", root,
                              "--output_dir", pre, "--manifest", manifest])
    common = ["--data_dir", root, "--splits_dir", splits, "--preprocessed_dir", pre,
              "--preprocessed_manifest", manifest, "--output_root", loso_out,
              "--seeds"] + list(SEEDS)
    t.run("loso_rgb", py + ["scripts/camera_loso.py", "--modalities", "rgb",
                            "--protocols", "loso", "random"] + common)
    t.run("loso_rgb_d", py + ["scripts/camera_loso.py", "--modalities", "rgb_d"] + common)
    t.run("loso_ir", py + ["scripts/camera_loso.py", "--modalities", "ir",
                           "--sources", "node_a", "node_b"] + common)
    t.run("analysis_a", py + ["scripts/camera_analysis_a.py", "--results_root", loso_out,
                              "--labels_csv", labels, "--output_dir", ana, "--B", str(B)])
    t.run("tab_loso", py + ["scripts/camera_export_loso_table.py", "--analysis_dir", ana,
                            "--output", os.path.join(base, "SYNTHETIC_tables",
                                                     "loso_tabular.tex"), "--synthetic"])
    t.run("fl_prepare", py + ["scripts/camera_fl_prepare.py", "--labels_csv", labels,
                              "--folds_csv", os.path.join(splits, "fl_folds.csv"),
                              "--preprocessed_dir", pre, "--preprocessed_manifest", manifest,
                              "--output_root", folds_out,
                              "--counts_csv", os.path.join(splits, "fl_materialised_manifest.csv"),
                              "--clean"])
    for fold in range(5):
        dirs = [os.path.join(folds_out, "camera_fold%d" % fold, n) for n in NODES]
        for seed in SEEDS:
            c = os.path.join(base_out, "rev_camera_fold%d_centralized_r10_seed%s" % (fold, seed))
            t.run("central_f%d_s%s" % (fold, seed),
                  py + ["scripts/train_centralized.py", "--data_dirs"] + dirs
                  + ["--epochs", "50", "--batch_size", "8", "--lr", "0.001", "--seed", seed,
                     "--output_dir", c])
            t.run("central_pred_f%d_s%s" % (fold, seed),
                  py + ["scripts/predict_from_checkpoint.py", c])
            loc = os.path.join(base_out, "rev_camera_fold%d_local_r10_seed%s" % (fold, seed))
            t.run("local_f%d_s%s" % (fold, seed),
                  py + ["scripts/train_local.py", "--batch", "--data_dirs"] + dirs
                  + ["--epochs", "50", "--batch_size", "8", "--lr", "0.001", "--seed", seed,
                     "--output_dir", loc])
            t.run("local_pred_f%d_s%s" % (fold, seed),
                  py + ["scripts/predict_from_checkpoint.py", loc])


def smoke(work: str, scenes: int, python: str, B: int = 10000, source: str = "node_a",
          seed: str = "42", epochs: int = 1, baselines: bool = False, fold: int = 0) -> None:
    """End-to-end smoke: one timed source x seed (rgb, LOSO + random) at ``epochs`` epochs.

    The two other sources are run as well, at the same settings and untimed as a reference,
    only because the contrast K is defined over all three sources; folds, epoch selection,
    pooled BA, the scene bootstrap with ``B`` resamples, Holm, K and tab:loso are all
    exercised.  ``baselines``: also the federated folds and, for one fold and ``seed``, the
    centralized and local-only baselines at ``epochs`` epochs (otherwise the settings of
    ``run``) with their per-image predictions.  Shares labels/manifests/preprocessing with
    ``run``; writes under ``S<scenes>/SMOKE_*``."""
    base = os.path.join(work, "S%d" % scenes)
    root = tree_dir(work, scenes)
    splits = os.path.join(base, "splits_camera")
    pre = os.path.join(base, "camera_224")
    manifest = os.path.join(splits, "preprocessed_manifest.csv")
    labels = os.path.join(root, "labels.csv")
    loso_out = os.path.join(base, "SMOKE_results", "loso")
    ana = os.path.join(base, "SMOKE_analysis", "a")
    t = Timer(os.path.join(base, "SMOKE_timings.json"))
    py = [python]
    t.run("labels", py + ["scripts/camera_labels.py", "--data_dir", root, "--splits_dir", splits])
    t.run("manifests", py + ["scripts/camera_manifests.py", "--labels_csv", labels,
                             "--splits_dir", splits])
    if not os.path.isfile(manifest):
        t.run("preprocess", py + ["scripts/camera_preprocess_frames.py", "--raw_dir", root,
                                  "--output_dir", pre, "--manifest", manifest])
    common = ["--data_dir", root, "--splits_dir", splits, "--preprocessed_dir", pre,
              "--preprocessed_manifest", manifest, "--output_root", loso_out,
              "--seeds", seed, "--epochs", str(epochs), "--modalities", "rgb",
              "--protocols", "loso", "random"]
    t.run("loso_rgb_%s_s%s" % (source, seed),
          py + ["scripts/camera_loso.py", "--sources", source] + common)
    others = [n for n in NODES if n != source]
    t.run("loso_rgb_other_sources", py + ["scripts/camera_loso.py", "--sources"] + others + common)
    t.run("analysis_a", py + ["scripts/camera_analysis_a.py", "--results_root", loso_out,
                              "--labels_csv", labels, "--output_dir", ana, "--B", str(B)])
    t.run("tab_loso", py + ["scripts/camera_export_loso_table.py", "--analysis_dir", ana,
                            "--output", os.path.join(base, "SMOKE_tables", "loso_tabular.tex"),
                            "--synthetic"])
    if not baselines:
        return
    folds_out = os.path.join(base, "SMOKE_processed")
    base_out = os.path.join(base, "SMOKE_results", "desktop")
    t.run("fl_prepare", py + ["scripts/camera_fl_prepare.py", "--labels_csv", labels,
                              "--folds_csv", os.path.join(splits, "fl_folds.csv"),
                              "--preprocessed_dir", pre, "--preprocessed_manifest", manifest,
                              "--output_root", folds_out,
                              "--counts_csv", os.path.join(base, "SMOKE_fl_materialised.csv"),
                              "--folds", str(fold), "--clean"])
    dirs = [os.path.join(folds_out, "camera_fold%d" % fold, n) for n in NODES]
    c = os.path.join(base_out, "rev_camera_fold%d_centralized_r10_seed%s" % (fold, seed))
    t.run("central_f%d_s%s" % (fold, seed),
          py + ["scripts/train_centralized.py", "--data_dirs"] + dirs
          + ["--epochs", str(epochs), "--batch_size", "8", "--lr", "0.001", "--seed", seed,
             "--output_dir", c])
    t.run("central_pred_f%d_s%s" % (fold, seed), py + ["scripts/predict_from_checkpoint.py", c])
    loc = os.path.join(base_out, "rev_camera_fold%d_local_r10_seed%s" % (fold, seed))
    t.run("local_f%d_s%s" % (fold, seed),
          py + ["scripts/train_local.py", "--batch", "--data_dirs"] + dirs
          + ["--epochs", str(epochs), "--batch_size", "8", "--lr", "0.001", "--seed", seed,
             "--output_dir", loc])
    t.run("local_pred_f%d_s%s" % (fold, seed), py + ["scripts/predict_from_checkpoint.py", loc])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--work", required=True)
    g.add_argument("--scenes", type=int, required=True)
    g.add_argument("--workers", type=int, default=12)
    g.add_argument("--overlap", action="store_true",
                   help="overlapping classes (weak flame signal in part of the fire frames)")
    s = sub.add_parser("subset")
    s.add_argument("--work", required=True)
    s.add_argument("--from_scenes", type=int, required=True)
    s.add_argument("--scenes", type=int, required=True)
    r = sub.add_parser("run")
    r.add_argument("--work", required=True)
    r.add_argument("--scenes", type=int, required=True)
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--B", type=int, default=10000)
    k = sub.add_parser("smoke")
    k.add_argument("--work", required=True)
    k.add_argument("--scenes", type=int, required=True)
    k.add_argument("--python", default=sys.executable)
    k.add_argument("--B", type=int, default=10000)
    k.add_argument("--source", default="node_a", choices=list(NODES))
    k.add_argument("--seed", default="42")
    k.add_argument("--epochs", type=int, default=1)
    k.add_argument("--baselines", action="store_true",
                   help="also the federated folds and one fold of the desktop baselines")
    k.add_argument("--fold", type=int, default=0)
    args = ap.parse_args(argv)
    work = os.path.abspath(args.work)
    if os.path.commonpath([work, REPO]) == REPO:
        raise SystemExit("--work must lie outside the repository")
    if args.cmd == "generate":
        generate(work, args.scenes, args.workers, args.overlap)
    elif args.cmd == "subset":
        subset(work, args.from_scenes, args.scenes)
    elif args.cmd == "smoke":
        smoke(work, args.scenes, args.python, args.B, args.source, args.seed, args.epochs,
              args.baselines, args.fold)
    else:
        run(work, args.scenes, args.python, args.B)
    return 0


if __name__ == "__main__":
    sys.exit(main())
