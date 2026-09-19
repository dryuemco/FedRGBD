#!/usr/bin/env python3
"""FedRGBD -- regenerate per-image predictions of a centralized / local-only baseline run.

The baselines (results schema 3) save ``model_selected.pt``, the model of the epoch
picked by the declared selection rule.  This script re-evaluates it on the val and
test split of every node the run used and writes, in the format of
``src/evaluation/predictions.py``:

    <run_dir>/predictions/selected_<node>_<split>.npz     (+ README.md)

It checks that the recomputed metrics reproduce the run's logged selected-epoch
metrics (``--atol``), and uses the run's own batch size so the evaluation is the
one that produced the logged numbers.

    python scripts/predict_from_checkpoint.py results/rev_baselines_sel/rev_iid_local_seed42
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.data.dataset import FlameDataset  # noqa: E402
from src.evaluation.metrics import MetricAccumulator  # noqa: E402
from src.evaluation.predictions import (README_TEXT, metrics_from_predictions,  # noqa: E402
                                        pack, unpack)
from src.models.mobilenetv3_multimodal import create_model  # noqa: E402

CHECK = ("accuracy", "balanced_accuracy", "mcc", "roc_auc")


def models_of(run_dir):
    """-> [(node, data_dirs_for_this_model, checkpoint, results_json)]."""
    if os.path.isfile(os.path.join(run_dir, "summary.json")):              # local-only
        out = []
        for res_path in sorted(glob.glob(os.path.join(run_dir, "node_*", "results.json"))):
            res = json.load(open(res_path))
            node_dir = os.path.dirname(res_path)
            out.append((os.path.basename(node_dir), [res["data_dir"]],
                        os.path.join(node_dir, "model_selected.pt"), res))
        return out
    res = json.load(open(os.path.join(run_dir, "results.json")))            # centralized
    return [("pooled", res["data_dirs"], os.path.join(run_dir, "model_selected.pt"), res)]


def predict(model, data_dir, split, batch_size, device, img_size=224):
    ds = FlameDataset(data_dir, split=split, img_size=img_size)
    acc = MetricAccumulator(num_classes=2, positive_class=1)
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
            acc.update(model(x.to(device)), y)
    logits, labels = acc.outputs()
    return pack([p for p, _ in ds.samples], labels, logits)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--img_size", type=int, default=224,
                    help="input resolution the run was trained at (224 = paper setting)")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.run_dir, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(README_TEXT)
    ok = True
    for name, data_dirs, ckpt, res in models_of(args.run_dir):
        if res.get("selected_epoch") is None or not os.path.isfile(ckpt):
            raise SystemExit("%s: no selected-epoch checkpoint (schema-3 run required)" % ckpt)
        model = create_model(num_classes=2, in_channels=3, pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.to(device).eval()
        batch = int(res.get("batch_size", 8))
        pooled = {"label": [], "margin": []}
        for data_dir in data_dirs:
            node = os.path.basename(os.path.normpath(data_dir))
            for split in ("val", "test"):
                blob = predict(model, data_dir, split, batch, device, args.img_size)
                with open(os.path.join(out_dir, "selected_%s_%s.npz" % (node, split)), "wb") as f:
                    f.write(blob)
                if split == "test":
                    d = unpack(blob)
                    pooled["label"].append(d["label"])
                    pooled["margin"].append(d["logit_margin"])
        got = metrics_from_predictions(np.concatenate(pooled["label"]),
                                       np.concatenate(pooled["margin"]))
        want = res["selected_test_metrics"]
        for key in CHECK:
            if abs(float(got[key]) - float(want[key])) > max(args.atol, 1e-6):
                ok = False
                print("MISMATCH %s %s: recomputed %.6f vs logged %.6f" % (name, key, got[key], want[key]))
        print("%s: selected epoch %s, test accuracy recomputed %.6f (logged %.6f)"
              % (name, res["selected_epoch"], got["accuracy"], want["accuracy"]))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
