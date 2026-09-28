#!/usr/bin/env python3
"""FedRGBD -- global evaluation: every model on the union of the three nodes' test splits.

Pre-registered in ``docs/GLOBAL_EVALUATION.md`` (definition, this code and the
interpretation rule, committed and pushed together before any local-only global value
was aggregated or compared -- the file's *Provenance* says exactly what existed before).
Read that file first; this script implements it and nothing else.

Two stages:

``predict``  runs each local-only node model (``model_selected.pt`` of the selection-rule
             baselines) on the test split of every node of its partition and writes
             ``<run>/predictions/global_<model node>_on_<data node>_test.npz``.  Before
             anything is kept it checks that the model reproduces its own-node predictions
             (``selected_<node>_test.npz``) and the cross-node metrics its training run
             logged (``cross_eval``); a failure stops the stage.

``analyze``  forms the global-evaluation tables in ``analysis/global_eval/``: every
             model's balanced accuracy on the union (full set and clean subset), the
             per-configuration means with their cluster-bootstrap intervals, and the
             seed-paired differences FL - local-only with interval and p-value, and
             the pre-registered verdicts, unadjusted and Holm-adjusted.

    python scripts/global_evaluation.py predict --results_dir results
    python scripts/global_evaluation.py analyze --results_dir results --output_dir analysis/global_eval
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.evaluation.bootstrap import (Unit, check_same_set, seed_paired_diff_ci,  # noqa: E402
                                      shared_set_ci)
from src.evaluation.predictions import load_npz, metrics_from_predictions  # noqa: E402

NODES = ("node_a", "node_b", "node_c")
METRIC = "balanced_accuracy"
B = 10000

#: the comparison families of docs/GLOBAL_EVALUATION.md -- never pooled with each other
FAMILIES = {
    "heterogeneous": {"power_config": "heterogeneous", "num_rounds": 3, "local_epochs": 5,
                      "lr": 1e-3,
                      "partitions": ("iid", "non_iid_label", "dirichlet_0.1", "dirichlet_0.5",
                                     "dirichlet_1", "iid_sub0.05", "iid_sub0.01",
                                     "non_iid_label_sub0.05", "non_iid_label_sub0.01")},
    "maxn": {"power_config": "maxn", "num_rounds": 10, "local_epochs": 5, "lr": 1e-3,
             "partitions": ("iid", "non_iid_label")},
}
STRATEGIES = (("fedavg", "FedAvg"), ("fedprox_0.01", "FedProx(0.01)"))
SUBSETS = ("full", "clean")
#: the reproduction checks of the predict stage
MARGIN_ATOL = 1e-4
LOGGED_ATOL = 1e-6

#: the interpretation rule (docs/GLOBAL_EVALUATION.md, section 2), fixed before any
#: local-only global value was aggregated or compared
ALPHA = 0.05
IMPROVES = "federation improves generalisation beyond the client's own distribution"
NO_DIFFERENCE = "no detectable difference"
WORSE = "federation worse"


def verdict_from_ci(ci_low: float, ci_high: float) -> str:
    """Unadjusted verdict: the 95 % interval of FL - local-only."""
    if ci_low > 0:
        return IMPROVES
    if ci_high < 0:
        return WORSE
    return NO_DIFFERENCE


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1), in input order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def verdict_from_holm(diff: float, p_holm: float) -> str:
    """Family-wise verdict: Holm-adjusted p < ALPHA, direction of the point estimate."""
    if p_holm < ALPHA:
        return IMPROVES if diff > 0 else WORSE
    return NO_DIFFERENCE


def apply_rule(comps: pd.DataFrame) -> pd.DataFrame:
    """Add the verdicts; Holm runs within each (family, subset) -- every partition x
    strategy comparison of a family is one correction family, the clean subset its own."""
    comps = comps.copy()
    if comps.empty:
        return comps
    comps["verdict"] = [verdict_from_ci(lo, hi) for lo, hi in zip(comps.ci_low, comps.ci_high)]
    comps["holm_m"] = 0
    comps["p_holm"] = np.nan
    for _, idx in comps.groupby(["family", "subset"]).groups.items():
        comps.loc[idx, "p_holm"] = holm(comps.loc[idx, "p_boot"].to_numpy())
        comps.loc[idx, "holm_m"] = len(idx)
    comps["verdict_holm"] = [verdict_from_holm(d, p) for d, p in zip(comps["diff"], comps.p_holm)]
    return comps


def comparisons_markdown(comps: pd.DataFrame) -> str:
    """Every comparison, unadjusted and Holm-adjusted."""
    lines = ["# Global evaluation: FL - local-only, balanced accuracy on the union", "",
             "Pre-registered in docs/GLOBAL_EVALUATION.md. Difference in percentage points, "
             "seed-paired, 95 %% cluster-bootstrap interval (B = %d). Verdict: interval rule, "
             "unadjusted; Holm: adjusted p < %.2f within the family." % (B, ALPHA), ""]
    for (family, subset), g in comps.groupby(["family", "subset"], sort=False):
        lines += ["## %s, %s set (Holm m = %d)" % (family, subset, len(g)), "",
                  "| partition | strategy | seeds | diff | 95 % CI | p | p Holm | verdict | verdict (Holm) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            lines.append("| %s | %s | %s | %+.1f | [%+.1f, %+.1f] | %.4f | %.4f | %s | %s |"
                         % (r.partition, r.strategy, r.paired_seeds, 100 * r.diff,
                            100 * r.ci_low, 100 * r.ci_high, r.p_boot, r.p_holm,
                            r.verdict, r.verdict_holm))
        lines.append("")
    return "\n".join(lines)


def global_file(pred_dir: str, model_node: str, data_node: str) -> str:
    return os.path.join(pred_dir, "global_%s_on_%s_test.npz" % (model_node, data_node))


def local_run_dirs(results_dir: str) -> List[str]:
    """Selection-rule local-only baselines (the ones the declared rule reports)."""
    return sorted(os.path.dirname(p) for p in
                  glob.glob(os.path.join(results_dir, "rev_baselines_sel", "*_local_seed*",
                                         "summary.json")))


# --------------------------------------------------------------------------- predict
def predict_run(run_dir: str, device, force: bool = False) -> List[str]:
    """Write the cross-node prediction files of one local-only run; -> problems found."""
    import torch

    from scripts.predict_from_checkpoint import models_of, predict
    from src.models.mobilenetv3_multimodal import create_model

    pred_dir = os.path.join(run_dir, "predictions")
    problems: List[str] = []
    for model_node, data_dirs, ckpt, res in models_of(run_dir):
        if res.get("selected_epoch") is None or not os.path.isfile(ckpt):
            return ["%s: no selected-epoch checkpoint" % ckpt]
        targets = [global_file(pred_dir, model_node, n) for n in NODES]
        if not force and all(os.path.isfile(t) for t in targets):
            continue
        model = create_model(num_classes=2, in_channels=3, pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.to(device).eval()
        partition_dir = os.path.dirname(os.path.normpath(data_dirs[0]))
        blobs = {}
        for data_node in NODES:
            blob = predict(model, os.path.join(partition_dir, data_node), "test",
                           int(res.get("batch_size", 8)), device)
            got = _unpack(blob)
            where = "%s: %s model on %s" % (os.path.basename(run_dir), model_node, data_node)
            if data_node == model_node:
                own = load_npz(os.path.join(pred_dir, "selected_%s_test.npz" % model_node))
                if not (np.array_equal(own["path"], got["path"])
                        and np.array_equal(own["label"], got["label"])):
                    problems.append(where + ": images differ from selected_%s_test.npz" % model_node)
                elif not np.array_equal(own["logit_margin"] > 0, got["logit_margin"] > 0) or \
                        np.max(np.abs(own["logit_margin"] - got["logit_margin"])) > MARGIN_ATOL:
                    problems.append(where + ": does not reproduce selected_%s_test.npz" % model_node)
            else:
                logged = (res.get("cross_eval") or {}).get(data_node, {}).get("test_metrics")
                if not logged:
                    problems.append(where + ": no logged cross_eval to check against")
                else:
                    m = metrics_from_predictions(got["label"], got["logit_margin"])
                    for key in ("accuracy", "balanced_accuracy"):
                        if abs(float(m[key]) - float(logged[key])) > LOGGED_ATOL:
                            problems.append(where + ": %s does not reproduce the logged "
                                            "cross_eval" % key)
            blobs[data_node] = blob
        if problems:
            return problems                     # keep nothing from a model that failed
        for data_node, blob in blobs.items():
            with open(global_file(pred_dir, model_node, data_node), "wb") as f:
                f.write(blob)
        print("  %s %s: 3 test splits written, reproduction checks passed"
              % (os.path.basename(run_dir), model_node))
    return problems


def _unpack(blob: bytes) -> Dict[str, np.ndarray]:
    from src.evaluation.predictions import unpack
    return unpack(blob)


def cmd_predict(args) -> int:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = local_run_dirs(args.results_dir)
    if not runs:
        raise SystemExit("no local-only selection-rule runs under %s" % args.results_dir)
    print("predict: %d local-only runs, device %s" % (len(runs), device))
    for run_dir in runs:
        problems = predict_run(run_dir, device, force=args.force)
        if problems:
            print("REPRODUCTION CHECK FAILED -- stopping, nothing kept for this model:")
            for p in problems:
                print("  -", p)
            return 1
    return 0


# --------------------------------------------------------------------------- analyze
def _unit(parts: Sequence[Dict[str, Any]]) -> Unit:
    return Unit(np.concatenate([p["label"] for p in parts]),
                np.concatenate([p["margin"] for p in parts]),
                np.concatenate([np.asarray(p["group"]).astype(str) for p in parts]))


def _node_parts(files: Sequence[str], groups: Dict[str, str], excluded: set,
                subset: str) -> List[Dict[str, Any]]:
    parts = []
    for path in files:
        d = load_npz(path)
        keep = np.ones(len(d["path"]), bool) if subset == "full" else \
            np.array([p not in excluded for p in d["path"]])
        parts.append({"label": d["label"][keep], "margin": d["logit_margin"][keep],
                      "group": np.array([groups[p] for p in d["path"][keep]]),
                      "paths": tuple(sorted(str(p) for p in d["path"]))})
    return parts


def local_models(record: Dict[str, Any], groups, excluded, subset: str,
                 union_paths: Sequence[tuple]) -> List[Unit]:
    """The three node models of one local-only run, each on the whole union."""
    pred_dir = os.path.join(record["run_dir"], "predictions")
    models = []
    for model_node in NODES:
        files = [global_file(pred_dir, model_node, n) for n in NODES]
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            raise SystemExit("missing %s -- run the predict stage first" % missing[0])
        parts = _node_parts(files, groups, excluded, subset)
        if [p["paths"] for p in parts] != list(union_paths):
            raise ValueError("%s %s: union differs from the partition's held-out set"
                             % (record["run_name"], model_node))
        models.append(_unit(parts))
    return models


def global_model(record: Dict[str, Any], subset: str) -> Unit:
    """FL global model / centralized model on the union: its pooled selected predictions."""
    return _unit(record["pred_units"] if subset == "full" else record["pred_units_clean"])


def _pick(runs, **want) -> List[Dict[str, Any]]:
    out = []
    for r in runs:
        ok = True
        for k, v in want.items():
            got = r.get(k)
            if isinstance(v, float):
                ok = ok and got is not None and abs(float(got) - v) < 1e-12
            else:
                ok = ok and got == v
        if ok:
            out.append(r)
    return sorted(out, key=lambda r: int(r["seed"]))


def family_configs(runs, family: str, partition: str) -> Dict[str, List[Dict[str, Any]]]:
    f = FAMILIES[family]
    base = [r for r in runs if r.get("protocol") == "group" and r.get("distribution") == partition
            and r.get("pred_units")]
    out = {}
    for strategy, label in STRATEGIES:
        out[label] = _pick(base, kind="fl", strategy=strategy, power_config=f["power_config"],
                           num_rounds=f["num_rounds"], local_epochs=float(f["local_epochs"]),
                           lr=f["lr"], n_nodes=3)
    out["Centralized"] = _pick(base, kind="centralized")
    out["Local-only"] = _pick(base, kind="local")
    return out


def analyze(runs: List[Dict[str, Any]], B: int = B):
    """-> (per-model rows, per-configuration rows, comparison rows) as DataFrames."""
    from scripts.analyze_results import partition_tables

    model_rows, config_rows, comp_rows = [], [], []
    for family, spec in FAMILIES.items():
        for partition in spec["partitions"]:
            configs = family_configs(runs, family, partition)
            if not any(configs[label] for _, label in STRATEGIES):
                print("  %s / %s: no federated runs (yet)" % (family, partition))
                continue
            groups, excluded = partition_tables(partition)
            ref = configs["Centralized"] or configs["Local-only"]
            union_paths = ref[0]["pred_paths"]
            for subset in SUBSETS:
                models: Dict[str, Dict[int, List[Unit]]] = {}
                for label, records in configs.items():
                    models[label] = {}
                    for r in records:
                        if label == "Local-only":
                            ms = local_models(r, groups, excluded, subset, union_paths)
                        else:
                            if r["pred_paths"] != union_paths:
                                raise ValueError("%s: union differs" % r["run_name"])
                            ms = [global_model(r, subset)]
                        models[label][int(r["seed"])] = ms
                        ones = np.ones((1, ms[0].G))
                        names = list(NODES) if label == "Local-only" else ["global"]
                        for name, u in zip(names, ms):
                            tp, fp, fn, tn = (ones @ u.counts)[0]
                            ba = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
                            model_rows.append({"family": family, "partition": partition,
                                               "method": label, "seed": int(r["seed"]),
                                               "model": name, "subset": subset,
                                               "n_images": int(u.n.sum()), METRIC: ba})
                    if models[label]:
                        runs_list = [models[label][s] for s in sorted(models[label])]
                        ci = shared_set_ci(runs_list, "global|%s|%s|%s|%s"
                                           % (family, partition, label, subset), B=B)
                        config_rows.append({"family": family, "partition": partition,
                                            "method": label, "subset": subset,
                                            "seeds": " ".join(map(str, sorted(models[label]))),
                                            "n_seeds": ci["n_runs"], "mean": ci["mean"],
                                            "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                                            "se": ci["se"], "B": ci["B"]})
                local = models["Local-only"]
                for _, label in STRATEGIES:
                    fl = models[label]
                    seeds = sorted(set(fl) & set(local))
                    if not seeds:
                        continue
                    res = seed_paired_diff_ci([fl[s] for s in seeds], [local[s] for s in seeds],
                                              "global|%s|%s|%s-local|%s"
                                              % (family, partition, label, subset), B=B)
                    comp_rows.append({"family": family, "partition": partition,
                                      "strategy": label, "subset": subset,
                                      "paired_seeds": " ".join(map(str, seeds)),
                                      "n_pairs": res["n_pairs"], "diff": res["diff"],
                                      "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                                      "p_boot": res["p_boot"], "B": res["B"]})
    return pd.DataFrame(model_rows), pd.DataFrame(config_rows), pd.DataFrame(comp_rows)


def cmd_analyze(args) -> int:
    from scripts.analyze_results import (attach_prediction_metrics, check_prediction_unions,
                                         collect_runs)

    runs = collect_runs(args.results_dir, warn=False)
    attach_prediction_metrics(runs, warn=False)
    check_prediction_unions(runs)
    models, configs, comps = analyze(runs, B=args.B)
    comps = apply_rule(comps)
    os.makedirs(args.output_dir, exist_ok=True)
    models.to_csv(os.path.join(args.output_dir, "global_eval_models.csv"), index=False)
    configs.to_csv(os.path.join(args.output_dir, "global_eval_configs.csv"), index=False)
    comps.to_csv(os.path.join(args.output_dir, "global_eval_comparisons.csv"), index=False)
    with open(os.path.join(args.output_dir, "global_eval_comparisons.md"), "w",
              encoding="utf-8") as f:
        f.write(comparisons_markdown(comps))
    print("wrote %d model rows, %d configuration rows, %d comparisons to %s"
          % (len(models), len(configs), len(comps), args.output_dir))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--force", action="store_true", help="recompute existing prediction files")
    a = sub.add_parser("analyze")
    a.add_argument("--results_dir", default="results")
    a.add_argument("--output_dir", default=os.path.join("analysis", "global_eval"))
    a.add_argument("--B", type=int, default=B)
    args = ap.parse_args(argv)
    return cmd_predict(args) if args.cmd == "predict" else cmd_analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
