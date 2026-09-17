"""FedRGBD — Unified results analysis: tables, statistics and IEEE-style plots.

Reads every experiment run under ``results/`` (federated, centralized and
local-only), in both the *old* ``results.json`` schema and the *new*
``results_schema_version: 2`` schema, and produces:

    runs.csv                            one row per run
    summary_table.csv / .md             mean +/- std [95% CI] per config & metric
    per_round_table.csv                 mean/std/CI per config & round
    pairwise_tests.csv / .md            paired strategy comparisons per distribution
    friedman.csv                        Friedman omnibus test per distribution
    <metric>_vs_round_<dist>.png/.pdf   convergence curves
    <metric>_vs_time_<dist>.png/.pdf    wall-clock efficiency
    <metric>_vs_communication_*.png     communication efficiency
    final_metrics_<dist>.png/.pdf       all-metrics bar chart (new-format runs)

Usage:
    python3 scripts/analyze_results.py --results_dir results --output_dir analysis

Everything is written to ``--output_dir`` (never inside ``results/``).
The module is import-safe: every step is a pure function so that tests can
call ``load_run``, ``collect_runs``, ``summarize``, ``pairwise_tests``,
``friedman_tests``, ``make_plots`` and ``main`` directly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
from scipy import stats

try:  # optional, only used for cross-checking
    import pingouin as pg
except Exception:  # pragma: no cover - environment dependent
    pg = None


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
FALLBACK_PAYLOAD_BYTES = 6.1e6
DEFAULT_LOCAL_EPOCHS_EQUIV = 5

#: metric ordering used for tables / bar charts (subset actually present is used)
METRIC_ORDER: List[str] = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "macro_specificity",
    "mcc",
    "roc_auc",
    "loss",
]

#: IEEE Sensors style, kept in sync with scripts/generate_plots.py
IEEE_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
}

PALETTE = [
    "#1565C0", "#C62828", "#2E7D32", "#E65100", "#6A1B9A",
    "#00838F", "#AD1457", "#558B2F", "#4527A0", "#EF6C00",
]
MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
BASELINE_COLORS = {"centralized": "#424242", "local": "#8D6E63"}

CONFIG_KEY_FIELDS = (
    "protocol",
    "kind", "strategy", "distribution", "num_rounds", "local_epochs", "lr", "n_nodes",
)

RECORD_FIELDS = [
    "run_dir", "run_name", "protocol", "kind", "strategy", "strategy_display", "mu",
    "distribution", "seed", "seed_label", "num_rounds", "local_epochs", "lr",
    "n_nodes", "schema_version", "config_id", "label", "timestamp",
    "time_estimated", "comm_estimated", "n_curve_points",
]

_PAYLOAD_CACHE: Dict[str, float] = {}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def estimate_payload_bytes(verbose: bool = True) -> float:
    """Bytes of one model transfer, from the real model when torch is available."""
    if "value" in _PAYLOAD_CACHE:
        return _PAYLOAD_CACHE["value"]
    value = FALLBACK_PAYLOAD_BYTES
    try:
        import torch  # noqa: F401  (import test only)

        from src.models.mobilenetv3_multimodal import create_model

        model = create_model(pretrained=False)
        total = 0
        for tensor in model.state_dict().values():
            try:
                total += int(tensor.numel()) * int(tensor.element_size())
            except AttributeError:
                total += int(np.asarray(tensor).nbytes)
        if total > 0:
            value = float(total)
    except Exception as exc:  # pragma: no cover - depends on environment
        if verbose:
            print(
                "[warn] could not build the model to size the payload "
                "({}); falling back to {:.3g} bytes".format(exc, FALLBACK_PAYLOAD_BYTES)
            )
    _PAYLOAD_CACHE["value"] = value
    return value


def ci95_half_width(values: Sequence[float]) -> float:
    """95% CI half-width of the mean; NaN when fewer than two samples."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = arr.size
    if n < 2:
        return float("nan")
    std = float(np.std(arr, ddof=1))
    return float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n))


def describe(values: Sequence[float]) -> Dict[str, float]:
    """n / mean / std(ddof=1) / 95% CI half-width / min / max of ``values``."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {
            "n": 0, "mean": float("nan"), "std": float("nan"), "ci95": float("nan"),
            "ci_low": float("nan"), "ci_high": float("nan"),
            "min": float("nan"), "max": float("nan"),
        }
    mean = float(arr.mean())
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
    ci = ci95_half_width(arr)
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "ci95": ci,
        "ci_low": mean - ci if np.isfinite(ci) else float("nan"),
        "ci_high": mean + ci if np.isfinite(ci) else float("nan"),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# --------------------------------------------------------------------------- #
# label / metadata derivation
# --------------------------------------------------------------------------- #
def parse_mu(strategy: Optional[str], proximal_mu: Any = None) -> Optional[float]:
    """Proximal term weight from the JSON value, else from the strategy name."""
    direct = _as_float(proximal_mu)
    if direct is not None:
        return direct
    if not strategy:
        return None
    match = re.search(r"fedprox[_\-]?(\d+(?:\.\d+)?)", str(strategy), re.IGNORECASE)
    if not match:
        return None
    token = match.group(1)
    if "." not in token and token.startswith("0") and len(token) > 1:
        token = "0." + token[1:]  # "001" -> 0.01, "01" -> 0.1
    return _as_float(token)


def _dist_from_token(token: Optional[str]) -> Optional[str]:
    """Distribution label from one token (tag / path segment / dir name)."""
    if not token:
        return None
    text = str(token).lower()
    match = re.search(r"dirichlet[_\-]?(\d+(?:\.\d+)?)", text)
    if match:
        return "dirichlet_" + match.group(1)
    if re.search(r"non[_\-]?iid", text):
        return "non_iid_label"
    if re.search(r"(?<![a-z])iid(?![a-z])", text) or re.search(r"[_\-]iid", text):
        return "iid"
    return None


def _sub_suffix(token: Optional[str]) -> str:
    if not token:
        return ""
    match = re.search(r"sub(\d+(?:\.\d+)?)", str(token).lower())
    return "_sub" + match.group(1) if match else ""


def _path_segments(path: Any) -> List[str]:
    if not isinstance(path, str):
        return []
    return [seg for seg in re.split(r"[\\/]+", path) if seg]


def parse_distribution(
    data: Dict[str, Any], run_name: str, client_config: Optional[Dict[str, Any]] = None
) -> str:
    """tags -> client_config data_dir segments -> directory name -> 'unknown'."""
    tags = data.get("tags")
    if isinstance(tags, (list, tuple)):
        for tag in tags:
            dist = _dist_from_token(tag)
            if dist:
                return dist + _sub_suffix(tag)

    if isinstance(client_config, dict):
        for node_cfg in client_config.values():
            if not isinstance(node_cfg, dict):
                continue
            for segment in _path_segments(node_cfg.get("data_dir")):
                dist = _dist_from_token(segment)
                if dist:
                    return dist + _sub_suffix(segment)

    for key in ("data_dirs", "data_dir"):
        value = data.get(key)
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            for segment in _path_segments(item):
                dist = _dist_from_token(segment)
                if dist:
                    return dist + _sub_suffix(segment)

    dist = _dist_from_token(run_name)
    if dist:
        return dist + _sub_suffix(run_name)
    return "unknown"


def parse_seed(data: Dict[str, Any], run_name: str, missing_seed: Optional[int] = None):
    """JSON ``seed`` -> ``seed(\\d+)`` in the dir name -> ``missing_seed`` -> None."""
    seed = data.get("seed")
    if seed is not None:
        try:
            return int(seed)
        except (TypeError, ValueError):
            pass
    match = re.search(r"seed[_\-]?(\d+)", run_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    if missing_seed is not None:
        return int(missing_seed)
    return None


def parse_n_nodes(data: Dict[str, Any], run_name: str, client_config=None) -> Optional[int]:
    value = data.get("min_clients")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    if isinstance(client_config, dict) and client_config:
        return len(client_config)
    nodes = data.get("nodes")
    if isinstance(nodes, dict) and nodes:
        return len(nodes)
    data_dirs = data.get("data_dirs")
    if isinstance(data_dirs, (list, tuple)) and data_dirs:
        return len(data_dirs)
    match = re.search(r"(?:^|[_\-])(\d+)\s*(?:node|nodes|n)(?:[_\-]|$)", run_name.lower())
    if match:
        return int(match.group(1))
    return None


def strategy_display(strategy: Optional[str], mu: Optional[float], kind: str) -> str:
    if kind == "centralized":
        return "Centralized"
    if kind == "local":
        return "Local-only"
    name = (strategy or "unknown").lower()
    if name.startswith("fedprox"):
        mu_text = ("%g" % mu) if mu is not None else "?"
        return "FedProx(mu={})".format(mu_text)
    if name == "fedavg":
        return "FedAvg"
    if name == "fedbn":
        return "FedBN"
    if name == "fedopt":
        return "FedOpt"
    return (strategy or "unknown").replace("_", " ")


def make_config_key(record: Dict[str, Any]) -> Tuple:
    return tuple(record.get(field) for field in CONFIG_KEY_FIELDS)


def config_id_of(record: Dict[str, Any]) -> str:
    return "|".join("" if v is None else str(v) for v in make_config_key(record))


PROTOCOL_GROUP = "group"    # leakage-safe, sequence/group-level split (revision, results/rev_*)
PROTOCOL_IMAGE = "image"    # random image-level split (paper v1)


def parse_protocol(data: Dict[str, Any], run_name: str) -> str:
    """Which partitioning protocol produced a run.

    The two protocols use different partitions, so their runs must never be
    pooled into one configuration (a v1 ``centralized_iid_seed42`` and a
    revision ``rev_iid_centralized_seed42`` even share the seed).  Explicit
    ``protocol`` / ``split_protocol`` keys or a ``group``/``image`` tag in the
    results file win; otherwise every ``results/rev_*`` run is group-level and
    everything else is the image-level v1 protocol.
    """
    for key in ("protocol", "split_protocol", "partition_protocol"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            v = val.strip().lower()
            return PROTOCOL_GROUP if v.startswith(("group", "seq")) else PROTOCOL_IMAGE
    tags = [str(t).lower() for t in (data.get("tags") or [])]
    if any(t in ("group", "group_level", "sequence_level") for t in tags):
        return PROTOCOL_GROUP
    if any(t in ("image", "image_level") for t in tags):
        return PROTOCOL_IMAGE
    return PROTOCOL_GROUP if str(run_name).startswith("rev_") else PROTOCOL_IMAGE


def make_label(record: Dict[str, Any]) -> str:
    parts = [record.get("strategy_display") or "", record.get("distribution") or ""]
    label = " ".join(p for p in parts if p).strip()
    n_nodes = record.get("n_nodes")
    if record.get("kind") == "fl" and n_nodes:
        label = "{} [{}N]".format(label, n_nodes)
    protocol = record.get("protocol")
    if protocol:
        label = "{} {{{}}}".format(label, protocol)
    return label or "run"


# --------------------------------------------------------------------------- #
# per-round metric extraction
# --------------------------------------------------------------------------- #
def _round_value_pairs(entries: Any, value_key: str) -> Dict[int, float]:
    """Parse ``[{round, value}]`` (or ``[[round, value]]``) into {round: value}."""
    out: Dict[int, float] = {}
    if not isinstance(entries, (list, tuple)):
        return out
    for i, entry in enumerate(entries):
        if isinstance(entry, dict):
            rnd = entry.get("round", entry.get("server_round", i + 1))
            val = entry.get(value_key, entry.get("value", entry.get("loss")))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            rnd, val = entry[0], entry[1]
        else:
            continue
        try:
            rnd_i = int(rnd)
        except (TypeError, ValueError):
            rnd_i = i + 1
        fval = _as_float(val)
        if fval is not None:
            out[rnd_i] = fval
    return out


def _metrics_distributed_by_round(data: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    """{round: {metric: value}} from ``metrics_distributed`` + ``losses_distributed``."""
    by_round: Dict[int, Dict[str, float]] = {}
    metrics = data.get("metrics_distributed")
    if isinstance(metrics, dict):
        for name, entries in metrics.items():
            if name.endswith("confusion_matrix_json") or name.startswith("cm_"):
                continue
            for rnd, val in _round_value_pairs(entries, "value").items():
                by_round.setdefault(rnd, {})[str(name)] = val
    for rnd, val in _round_value_pairs(data.get("losses_distributed"), "loss").items():
        by_round.setdefault(rnd, {}).setdefault("loss", val)
    return by_round


def _pooled_metrics_by_round(data: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    rounds = data.get("rounds")
    if not isinstance(rounds, (list, tuple)):
        return out
    for i, entry in enumerate(rounds):
        if not isinstance(entry, dict):
            continue
        rnd = int(entry.get("round", i + 1))
        evaluate = entry.get("evaluate") or {}
        aggregate = evaluate.get("aggregate") if isinstance(evaluate, dict) else None
        pooled = aggregate.get("pooled") if isinstance(aggregate, dict) else None
        if isinstance(pooled, dict):
            for name, val in pooled.items():
                fval = _as_float(val)
                if fval is not None:
                    out.setdefault(rnd, {})["pooled_" + str(name)] = fval
        if isinstance(aggregate, dict):
            for name, val in aggregate.items():
                if name == "pooled":
                    continue
                fval = _as_float(val)
                if fval is not None:
                    out.setdefault(rnd, {}).setdefault(str(name), fval)
    return out


def _round_timing_and_comm(data: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    """{round: {elapsed_s, cumulative_mb}} from the new-schema ``rounds`` list."""
    out: Dict[int, Dict[str, float]] = {}
    rounds = data.get("rounds")
    if not isinstance(rounds, (list, tuple)):
        return out
    for i, entry in enumerate(rounds):
        if not isinstance(entry, dict):
            continue
        rnd = int(entry.get("round", i + 1))
        info: Dict[str, float] = {}
        evaluate = entry.get("evaluate") if isinstance(entry.get("evaluate"), dict) else {}
        fit = entry.get("fit") if isinstance(entry.get("fit"), dict) else {}
        elapsed = _first_not_none(
            _as_float(evaluate.get("elapsed_s")),
            _as_float(fit.get("elapsed_s")),
            _as_float(entry.get("elapsed_s")),
        )
        if elapsed is not None:
            info["elapsed_s"] = elapsed
        comm = _as_float(entry.get("cumulative_communication_bytes"))
        if comm is not None:
            info["cumulative_mb"] = comm / 1e6
        if info:
            out[rnd] = info
    return out


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def _new_record(
    run_dir: str, data: Dict[str, Any], kind: str, strategy: str,
    client_config: Optional[Dict[str, Any]], missing_seed: Optional[int],
    warn: bool = True,
) -> Dict[str, Any]:
    run_name = os.path.basename(os.path.normpath(run_dir))
    mu = parse_mu(strategy, data.get("proximal_mu"))
    seed = parse_seed(data, run_name, missing_seed)
    if seed is None and warn:
        print("[warn] run '{}' has no seed (JSON or dir name); labelled seed=NA".format(run_name))

    local_epochs = None
    lr = None
    if isinstance(client_config, dict) and client_config:
        for node_cfg in client_config.values():
            if isinstance(node_cfg, dict):
                local_epochs = _first_not_none(local_epochs, node_cfg.get("local_epochs"))
                lr = _first_not_none(lr, _as_float(node_cfg.get("lr")))
    if local_epochs is None:
        local_epochs = _first_not_none(data.get("local_epochs"), data.get("epochs"))
    if lr is None:
        lr = _as_float(data.get("lr"))
    try:
        local_epochs = int(local_epochs) if local_epochs is not None else None
    except (TypeError, ValueError):
        local_epochs = None

    record: Dict[str, Any] = {
        "run_dir": os.path.abspath(run_dir),
        "run_name": run_name,
        "protocol": parse_protocol(data, run_name),
        "kind": kind,
        "strategy": strategy,
        "mu": mu,
        "distribution": parse_distribution(data, run_name, client_config),
        "seed": seed,
        "seed_label": "NA" if seed is None else str(seed),
        "num_rounds": None,
        "local_epochs": local_epochs,
        "lr": lr,
        "n_nodes": parse_n_nodes(data, run_name, client_config),
        "schema_version": int(data.get("results_schema_version", 1) or 1),
        "timestamp": data.get("timestamp"),
        "total_time_s": _as_float(data.get("total_time_s")),
        "tags": list(data.get("tags") or []),
        "curve": [],
        "metrics_final": {},
        "raw": data,
    }
    record["strategy_display"] = strategy_display(strategy, mu, kind)
    return record


def _finalize(record: Dict[str, Any]) -> Dict[str, Any]:
    """Fill derived fields (config id, label, final/best metric, estimate flags)."""
    curve = record.get("curve") or []
    curve.sort(key=lambda row: row["round"])
    record["curve"] = curve
    if record.get("num_rounds") is None and curve:
        record["num_rounds"] = int(curve[-1]["round"])
    record["n_curve_points"] = len(curve)
    record["time_estimated"] = any(bool(r.get("time_estimated")) for r in curve)
    record["comm_estimated"] = any(bool(r.get("comm_estimated")) for r in curve)

    if curve:
        final = curve[-1]
        metrics_final = {
            k: v for k, v in final.items()
            if k not in ("round", "time_estimated", "comm_estimated")
            and isinstance(v, (int, float)) and v is not None
        }
        record.setdefault("metrics_final", {})
        for key, val in metrics_final.items():
            record["metrics_final"].setdefault(key, val)
        if record.get("final_accuracy") is None:
            record["final_accuracy"] = _as_float(final.get("accuracy"))
        record["final_loss"] = _first_not_none(
            _as_float(record.get("final_loss")), _as_float(final.get("loss"))
        )
        accs = [_as_float(r.get("accuracy")) for r in curve]
        accs = [a for a in accs if a is not None]
        record["best_accuracy"] = max(accs) if accs else None
        record["round1_accuracy"] = _as_float(curve[0].get("accuracy"))
        record["final_elapsed_s"] = _as_float(final.get("elapsed_s"))
        record["final_cumulative_mb"] = _as_float(final.get("cumulative_mb"))
    else:
        record.setdefault("final_accuracy", None)
        record.setdefault("final_loss", None)
        record["best_accuracy"] = record.get("final_accuracy")
        record["round1_accuracy"] = None
        record["final_elapsed_s"] = None
        record["final_cumulative_mb"] = None

    if record.get("best_accuracy") is None:
        record["best_accuracy"] = record.get("final_accuracy")

    record["config_key"] = make_config_key(record)
    record["config_id"] = config_id_of(record)
    record["label"] = make_label(record)
    return record


def load_fl_run(
    run_dir: str, data: Dict[str, Any], payload_bytes: Optional[float] = None,
    missing_seed: Optional[int] = None, warn: bool = True,
) -> Dict[str, Any]:
    """Old (v1) or new (v2) federated ``results.json`` -> run record."""
    client_config = data.get("client_config")
    client_config = client_config if isinstance(client_config, dict) else None
    strategy = str(data.get("strategy") or "unknown")
    record = _new_record(run_dir, data, "fl", strategy, client_config, missing_seed, warn)
    record["num_rounds"] = _first_not_none(
        int(data["num_rounds"]) if data.get("num_rounds") is not None else None, None
    )
    record["client_config"] = client_config
    record["model_payload_bytes"] = _as_float(data.get("model_payload_bytes"))

    by_round = _metrics_distributed_by_round(data)
    for rnd, extra in _pooled_metrics_by_round(data).items():
        for key, val in extra.items():
            by_round.setdefault(rnd, {}).setdefault(key, val)

    timing = _round_timing_and_comm(data)
    total_time = record.get("total_time_s")
    num_rounds = record.get("num_rounds") or (max(by_round) if by_round else 0)

    if payload_bytes is None:
        payload_bytes = record["model_payload_bytes"]
    if payload_bytes is None:
        payload_bytes = estimate_payload_bytes()

    curve: List[Dict[str, Any]] = []
    for rnd in sorted(by_round):
        row: Dict[str, Any] = {"round": int(rnd)}
        row.update(by_round[rnd])
        info = timing.get(rnd, {})

        if "elapsed_s" in info:
            row["elapsed_s"] = info["elapsed_s"]
            row["time_estimated"] = False
        elif total_time is not None and num_rounds:
            row["elapsed_s"] = total_time * rnd / float(num_rounds)
            row["time_estimated"] = True
        else:
            row["elapsed_s"] = float("nan")
            row["time_estimated"] = True

        if "cumulative_mb" in info:
            row["cumulative_mb"] = info["cumulative_mb"]
            row["comm_estimated"] = False
        else:
            n_nodes = record.get("n_nodes")
            if n_nodes:
                row["cumulative_mb"] = payload_bytes * 2 * n_nodes * rnd / 1e6
            else:
                row["cumulative_mb"] = float("nan")
            row["comm_estimated"] = True
        curve.append(row)

    record["curve"] = curve
    return _finalize(record)


def load_centralized_run(
    run_dir: str, data: Dict[str, Any], local_epochs_equiv: int = DEFAULT_LOCAL_EPOCHS_EQUIV,
    missing_seed: Optional[int] = None, warn: bool = True,
) -> Dict[str, Any]:
    """Centralized ``results.json`` -> run record (epochs mapped to FL rounds)."""
    record = _new_record(run_dir, data, "centralized", "centralized", None, missing_seed, warn)
    history = data.get("history") if isinstance(data.get("history"), list) else []

    cumulative: Dict[int, float] = {}
    running = 0.0
    by_epoch: Dict[int, Dict[str, Any]] = {}
    for i, entry in enumerate(history):
        if not isinstance(entry, dict):
            continue
        epoch = int(entry.get("epoch", i + 1))
        running += _as_float(entry.get("epoch_time_s")) or 0.0
        cumulative[epoch] = running
        by_epoch[epoch] = entry

    step = max(int(local_epochs_equiv or DEFAULT_LOCAL_EPOCHS_EQUIV), 1)
    equivalents = data.get("fl_round_equivalents")
    mapping: List[Tuple[int, int]] = []  # (round, epoch)
    if isinstance(equivalents, dict) and equivalents:
        for key, entry in equivalents.items():
            match = re.search(r"(\d+)", str(key))
            if not match:
                continue
            rnd = int(match.group(1))
            epoch = int((entry or {}).get("epoch", rnd * step)) if isinstance(entry, dict) else rnd * step
            mapping.append((rnd, epoch))
    else:
        for epoch in sorted(by_epoch):
            if epoch % step == 0:
                mapping.append((epoch // step, epoch))
    mapping.sort()

    curve: List[Dict[str, Any]] = []
    for rnd, epoch in mapping:
        entry = by_epoch.get(epoch, {})
        row: Dict[str, Any] = {
            "round": int(rnd),
            "accuracy": _as_float(entry.get("val_accuracy")),
            "loss": _as_float(entry.get("val_loss")),
            "elapsed_s": cumulative.get(epoch, float("nan")),
            "cumulative_mb": 0.0,
            "time_estimated": epoch not in cumulative,
            "comm_estimated": False,
        }
        val_metrics = entry.get("val_metrics")
        if isinstance(val_metrics, dict):
            for name, val in val_metrics.items():
                fval = _as_float(val)
                if fval is not None:
                    row.setdefault(str(name), fval)
        curve.append(row)

    record["num_rounds"] = int(mapping[-1][0]) if mapping else None
    record["curve"] = curve
    record["final_accuracy"] = _as_float(data.get("final_test_accuracy"))
    record["final_loss"] = _as_float(data.get("final_test_loss"))
    final_metrics = data.get("final_test_metrics")
    if isinstance(final_metrics, dict):
        record["metrics_final"] = {
            str(k): _as_float(v) for k, v in final_metrics.items() if _as_float(v) is not None
        }
    if record["final_accuracy"] is not None:
        record["metrics_final"].setdefault("accuracy", record["final_accuracy"])
    if record["final_loss"] is not None:
        record["metrics_final"].setdefault("loss", record["final_loss"])
    record["epochs"] = data.get("epochs")
    return _finalize(record)


def load_local_run(
    run_dir: str, data: Dict[str, Any], local_epochs_equiv: int = DEFAULT_LOCAL_EPOCHS_EQUIV,
    missing_seed: Optional[int] = None, warn: bool = True,
) -> Dict[str, Any]:
    """Local-only run (``summary.json`` + ``node_*/results.json``) -> one record.

    The accuracy-vs-round curve is the mean over nodes of ``val_accuracy`` at
    epochs ``local_epochs_equiv, 2*local_epochs_equiv, ...``.
    """
    record = _new_record(run_dir, data, "local", "local_only", None, missing_seed, warn)

    node_files: List[str] = []
    if os.path.isdir(run_dir):
        for entry in sorted(os.listdir(run_dir)):
            candidate = os.path.join(run_dir, entry, "results.json")
            if os.path.isfile(candidate):
                node_files.append(candidate)

    node_data: Dict[str, Dict[str, Any]] = {}
    if node_files:
        for path in node_files:
            try:
                node_json = _read_json(path)
            except (OSError, ValueError):
                continue
            name = str(node_json.get("node_name") or os.path.basename(os.path.dirname(path)))
            node_data[name] = node_json
    elif str(data.get("experiment")) == "local_only":
        node_data[str(data.get("node_name") or "node")] = data

    step = max(int(local_epochs_equiv or DEFAULT_LOCAL_EPOCHS_EQUIV), 1)
    per_round_acc: Dict[int, List[float]] = {}
    per_round_loss: Dict[int, List[float]] = {}
    per_round_time: Dict[int, List[float]] = {}
    for node_json in node_data.values():
        history = node_json.get("history") if isinstance(node_json.get("history"), list) else []
        running = 0.0
        for i, entry in enumerate(history):
            if not isinstance(entry, dict):
                continue
            epoch = int(entry.get("epoch", i + 1))
            running += _as_float(entry.get("epoch_time_s")) or 0.0
            if epoch % step:
                continue
            rnd = epoch // step
            acc = _as_float(entry.get("val_accuracy"))
            loss = _as_float(entry.get("val_loss"))
            if acc is not None:
                per_round_acc.setdefault(rnd, []).append(acc)
            if loss is not None:
                per_round_loss.setdefault(rnd, []).append(loss)
            per_round_time.setdefault(rnd, []).append(running)

    curve: List[Dict[str, Any]] = []
    for rnd in sorted(set(per_round_acc) | set(per_round_loss) | set(per_round_time)):
        accs = per_round_acc.get(rnd, [])
        losses = per_round_loss.get(rnd, [])
        times = per_round_time.get(rnd, [])
        curve.append({
            "round": int(rnd),
            "accuracy": float(np.mean(accs)) if accs else None,
            "loss": float(np.mean(losses)) if losses else None,
            "elapsed_s": float(np.mean(times)) if times else float("nan"),
            "cumulative_mb": 0.0,
            "time_estimated": not times,
            "comm_estimated": False,
        })
    record["curve"] = curve

    nodes = data.get("nodes") if isinstance(data.get("nodes"), dict) else {}
    final_accs = [
        _as_float((cfg or {}).get("final_test_accuracy"))
        for cfg in nodes.values() if isinstance(cfg, dict)
    ]
    final_accs = [a for a in final_accs if a is not None]
    if not final_accs:
        final_accs = [
            _as_float(nj.get("final_test_accuracy")) for nj in node_data.values()
        ]
        final_accs = [a for a in final_accs if a is not None]
    final_losses = [
        _as_float((cfg or {}).get("final_test_loss"))
        for cfg in nodes.values() if isinstance(cfg, dict)
    ]
    final_losses = [x for x in final_losses if x is not None]

    record["final_accuracy"] = _first_not_none(
        _as_float(data.get("mean_test_accuracy")),
        float(np.mean(final_accs)) if final_accs else None,
    )
    record["final_loss"] = float(np.mean(final_losses)) if final_losses else None
    if record["final_accuracy"] is not None:
        record["metrics_final"]["accuracy"] = record["final_accuracy"]
    if record["final_loss"] is not None:
        record["metrics_final"]["loss"] = record["final_loss"]
    if record.get("n_nodes") is None and node_data:
        record["n_nodes"] = len(node_data)
    record["node_accuracies"] = {
        name: _as_float((cfg or {}).get("final_test_accuracy"))
        for name, cfg in nodes.items() if isinstance(cfg, dict)
    }
    record["epochs"] = data.get("epochs")
    return _finalize(record)


def load_run(
    path: str,
    payload_bytes: Optional[float] = None,
    local_epochs_equiv: int = DEFAULT_LOCAL_EPOCHS_EQUIV,
    missing_seed: Optional[int] = None,
    warn: bool = True,
) -> Dict[str, Any]:
    """Load one run from a run directory or a ``results.json``/``summary.json`` path."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        run_dir = path
        json_path = None
        for candidate in ("results.json", "summary.json"):
            candidate_path = os.path.join(path, candidate)
            if os.path.isfile(candidate_path):
                json_path = candidate_path
                break
        if json_path is None:
            raise FileNotFoundError("no results.json / summary.json in {}".format(path))
    else:
        json_path = path
        run_dir = os.path.dirname(path)

    data = _read_json(json_path)
    experiment = str(data.get("experiment") or "").lower()

    if experiment == "centralized":
        return load_centralized_run(run_dir, data, local_epochs_equiv, missing_seed, warn)
    if experiment in ("local_only_batch", "local_only"):
        return load_local_run(run_dir, data, local_epochs_equiv, missing_seed, warn)
    return load_fl_run(run_dir, data, payload_bytes, missing_seed, warn)


def iter_run_dirs(results_dir: str, include_test_runs: bool = False) -> Iterator[str]:
    """Yield every run directory below ``results_dir`` (does not descend into runs)."""
    if not os.path.isdir(results_dir):
        return
    for entry in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, entry)
        if not os.path.isdir(path):
            continue
        lowered = entry.lower()
        if not include_test_runs and (lowered.startswith("test_") or lowered == "test"):
            continue
        if any(os.path.isfile(os.path.join(path, n)) for n in ("results.json", "summary.json")):
            yield path
        else:
            for nested in iter_run_dirs(path, include_test_runs):
                yield nested


def collect_runs(
    results_dir: str,
    payload_bytes: Optional[float] = None,
    local_epochs_equiv: int = DEFAULT_LOCAL_EPOCHS_EQUIV,
    missing_seed: Optional[int] = None,
    include_test_runs: bool = False,
    warn: bool = True,
) -> List[Dict[str, Any]]:
    """Load every run under ``results_dir`` into a list of records."""
    runs: List[Dict[str, Any]] = []
    for run_dir in iter_run_dirs(results_dir, include_test_runs):
        try:
            runs.append(
                load_run(run_dir, payload_bytes, local_epochs_equiv, missing_seed, warn)
            )
        except Exception as exc:  # keep going on a single bad run
            if warn:
                print("[warn] skipping {}: {}".format(run_dir, exc))
    return runs


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def runs_dataframe(runs: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in runs:
        row = {field: record.get(field) for field in RECORD_FIELDS}
        row["tags"] = ",".join(record.get("tags") or [])
        row["final_accuracy"] = record.get("final_accuracy")
        row["best_accuracy"] = record.get("best_accuracy")
        row["round1_accuracy"] = record.get("round1_accuracy")
        row["final_loss"] = record.get("final_loss")
        row["total_time_s"] = record.get("total_time_s")
        row["final_elapsed_s"] = record.get("final_elapsed_s")
        row["final_cumulative_mb"] = record.get("final_cumulative_mb")
        rows.append(row)
    columns = RECORD_FIELDS + [
        "tags", "final_accuracy", "best_accuracy", "round1_accuracy", "final_loss",
        "total_time_s", "final_elapsed_s", "final_cumulative_mb",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values(["kind", "distribution", "strategy", "seed_label"]).reset_index(drop=True)
    return df


def _group_by_config(runs: Sequence[Dict[str, Any]]) -> "Dict[str, List[Dict[str, Any]]]":
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in runs:
        groups.setdefault(record["config_id"], []).append(record)
    return groups


def _run_metric_values(record: Dict[str, Any]) -> Dict[str, float]:
    """Scalar metrics of one run used by the summary table."""
    values: Dict[str, float] = {}
    for name, val in (record.get("metrics_final") or {}).items():
        fval = _as_float(val)
        if fval is not None:
            values["final_" + str(name)] = fval
    for key, name in (
        ("final_accuracy", "final_accuracy"),
        ("final_loss", "final_loss"),
        ("round1_accuracy", "round1_accuracy"),
        ("best_accuracy", "best_accuracy"),
        ("total_time_s", "total_time_s"),
        ("final_cumulative_mb", "final_cumulative_mb"),
    ):
        fval = _as_float(record.get(key))
        if fval is not None:
            values[name] = fval
    return values


def _metric_sort_key(name: str) -> Tuple[int, str]:
    core = name[len("final_"):] if name.startswith("final_") else name
    if name == "final_accuracy":
        return (-2, name)
    if name == "round1_accuracy":
        return (-1, name)
    if core in METRIC_ORDER:
        return (METRIC_ORDER.index(core), name)
    return (len(METRIC_ORDER) + 1, name)


def summary_table(runs: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Per config key and per available metric: n_seeds, mean, std, CI, min, max."""
    rows = []
    for config_id, group in _group_by_config(runs).items():
        first = group[0]
        per_metric: Dict[str, List[float]] = {}
        for record in group:
            for name, val in _run_metric_values(record).items():
                per_metric.setdefault(name, []).append(val)
        seeds = sorted({r["seed_label"] for r in group})
        for name in sorted(per_metric, key=_metric_sort_key):
            stats_dict = describe(per_metric[name])
            rows.append({
                "config_id": config_id,
                "label": first["label"],
                "kind": first["kind"],
                "strategy": first["strategy"],
                "mu": first.get("mu"),
                "distribution": first["distribution"],
                "n_nodes": first.get("n_nodes"),
                "num_rounds": first.get("num_rounds"),
                "metric": name,
                "n_seeds": stats_dict["n"],
                "seeds": ",".join(seeds),
                "mean": stats_dict["mean"],
                "std": stats_dict["std"],
                "ci95": stats_dict["ci95"],
                "ci_low": stats_dict["ci_low"],
                "ci_high": stats_dict["ci_high"],
                "min": stats_dict["min"],
                "max": stats_dict["max"],
            })
    df = pd.DataFrame(rows, columns=[
        "config_id", "label", "kind", "strategy", "mu", "distribution", "n_nodes",
        "num_rounds", "metric", "n_seeds", "seeds", "mean", "std", "ci95",
        "ci_low", "ci_high", "min", "max",
    ])
    if not df.empty:
        df = df.sort_values(["distribution", "kind", "label", "metric"]).reset_index(drop=True)
    return df


def per_round_table(runs: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Per config key and round: mean/std/CI of accuracy and loss."""
    rows = []
    for config_id, group in _group_by_config(runs).items():
        first = group[0]
        by_round: Dict[int, Dict[str, List[float]]] = {}
        for record in group:
            for point in record.get("curve") or []:
                bucket = by_round.setdefault(int(point["round"]), {"accuracy": [], "loss": [],
                                                                   "elapsed_s": [], "cumulative_mb": []})
                for key in ("accuracy", "loss", "elapsed_s", "cumulative_mb"):
                    val = _as_float(point.get(key))
                    if val is not None:
                        bucket[key].append(val)
        for rnd in sorted(by_round):
            bucket = by_round[rnd]
            acc = describe(bucket["accuracy"])
            loss = describe(bucket["loss"])
            rows.append({
                "config_id": config_id,
                "label": first["label"],
                "kind": first["kind"],
                "strategy": first["strategy"],
                "distribution": first["distribution"],
                "round": rnd,
                "n_seeds": max(acc["n"], loss["n"]),
                "accuracy_mean": acc["mean"], "accuracy_std": acc["std"],
                "accuracy_ci95": acc["ci95"], "accuracy_ci_low": acc["ci_low"],
                "accuracy_ci_high": acc["ci_high"],
                "loss_mean": loss["mean"], "loss_std": loss["std"],
                "loss_ci95": loss["ci95"], "loss_ci_low": loss["ci_low"],
                "loss_ci_high": loss["ci_high"],
                "elapsed_s_mean": describe(bucket["elapsed_s"])["mean"],
                "cumulative_mb_mean": describe(bucket["cumulative_mb"])["mean"],
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["distribution", "label", "round"]).reset_index(drop=True)
    return df


def summarize(runs: Sequence[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """All three tables at once: ``runs``, ``summary`` and ``per_round``."""
    return {
        "runs": runs_dataframe(runs),
        "summary": summary_table(runs),
        "per_round": per_round_table(runs),
    }


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _final_metric(record: Dict[str, Any], metric: str) -> Optional[float]:
    metrics_final = record.get("metrics_final") or {}
    candidates = [
        metrics_final.get(metric),
        record.get("final_" + metric),
        record.get("final_accuracy") if metric == "accuracy" else None,
        record.get("final_loss") if metric == "loss" else None,
    ]
    curve = record.get("curve") or []
    if curve:
        candidates.append(curve[-1].get(metric))
    for candidate in candidates:
        val = _as_float(candidate)
        if val is not None:
            return val
    return None


def _seed_values_by_strategy(
    runs: Sequence[Dict[str, Any]], metric: str
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """{distribution: {strategy: {seed: metric}}} — seeds without a value are dropped."""
    out: Dict[str, Dict[str, Dict[int, List[float]]]] = {}
    for record in runs:
        seed = record.get("seed")
        if seed is None:
            continue
        value = _final_metric(record, metric)
        if value is None:
            continue
        dist = record.get("distribution") or "unknown"
        strategy = record.get("strategy_display") or record.get("strategy") or "unknown"
        out.setdefault(dist, {}).setdefault(strategy, {}).setdefault(int(seed), []).append(value)
    collapsed: Dict[str, Dict[str, Dict[int, float]]] = {}
    for dist, strategies in out.items():
        for strategy, seeds in strategies.items():
            for seed, values in seeds.items():
                collapsed.setdefault(dist, {}).setdefault(strategy, {})[seed] = float(np.mean(values))
    return collapsed


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diffs.size < 2:
        return float("nan")
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return float("nan")
    return float(np.mean(diffs) / sd)


def cohens_d_unpaired(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    pooled = math.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    )
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def pairwise_tests(runs: Sequence[Dict[str, Any]], metric: str = "accuracy") -> pd.DataFrame:
    """Paired strategy comparisons within each distribution (>=2 common seeds)."""
    rows = []
    by_dist = _seed_values_by_strategy(runs, metric)
    for dist in sorted(by_dist):
        strategies = sorted(by_dist[dist])
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                name_a, name_b = strategies[i], strategies[j]
                map_a, map_b = by_dist[dist][name_a], by_dist[dist][name_b]
                seeds = sorted(set(map_a) & set(map_b))
                if len(seeds) < 2:
                    continue
                a = np.array([map_a[s] for s in seeds], dtype=float)
                b = np.array([map_b[s] for s in seeds], dtype=float)
                diffs = a - b
                notes: List[str] = []

                w_stat, w_p = float("nan"), float("nan")
                if np.allclose(diffs, 0.0):
                    notes.append("wilcoxon: all differences are zero")
                else:
                    try:
                        result = stats.wilcoxon(a, b)
                        w_stat, w_p = float(result[0]), float(result[1])
                    except Exception as exc:
                        notes.append("wilcoxon unavailable ({})".format(exc))

                try:
                    t_stat, t_p = stats.ttest_rel(a, b)
                    t_stat, t_p = float(t_stat), float(t_p)
                except Exception as exc:
                    t_stat, t_p = float("nan"), float("nan")
                    notes.append("paired t-test unavailable ({})".format(exc))

                pg_d, pg_w_p = float("nan"), float("nan")
                if pg is not None:
                    try:
                        pg_d = float(pg.compute_effsize(a, b, paired=True, eftype="cohen"))
                    except Exception:
                        pg_d = float("nan")
                    if not np.allclose(diffs, 0.0):
                        try:
                            pg_res = pg.wilcoxon(a, b)
                            pg_w_p = float(np.asarray(pg_res["p-val"])[0])
                        except Exception:
                            pg_w_p = float("nan")
                else:
                    notes.append("pingouin not installed")

                # pingouin's paired "cohen" is d_av (mean diff / average SD), our
                # cohen_d_paired is d_z (mean diff / SD of differences) - both reported.
                d_paired = cohens_d_paired(a, b)

                rows.append({
                    "distribution": dist,
                    "metric": metric,
                    "strategy_a": name_a,
                    "strategy_b": name_b,
                    "n_seeds": len(seeds),
                    "seeds": ",".join(str(s) for s in seeds),
                    "mean_a": float(a.mean()),
                    "mean_b": float(b.mean()),
                    "mean_diff": float(diffs.mean()),
                    "cohen_d_paired": d_paired,
                    "cohen_d_unpaired": cohens_d_unpaired(a, b),
                    "wilcoxon_stat": w_stat,
                    "wilcoxon_p": w_p,
                    "ttest_t": t_stat,
                    "ttest_p": t_p,
                    "pg_cohen_d_av": pg_d,
                    "pg_wilcoxon_p": pg_w_p,
                    "note": "; ".join(notes),
                })
    return pd.DataFrame(rows, columns=[
        "distribution", "metric", "strategy_a", "strategy_b", "n_seeds", "seeds",
        "mean_a", "mean_b", "mean_diff", "cohen_d_paired", "cohen_d_unpaired",
        "wilcoxon_stat", "wilcoxon_p", "ttest_t", "ttest_p", "pg_cohen_d_av",
        "pg_wilcoxon_p", "note",
    ])


def friedman_tests(runs: Sequence[Dict[str, Any]], metric: str = "accuracy") -> pd.DataFrame:
    """Friedman omnibus test per distribution (>=3 strategies sharing >=2 seeds)."""
    rows = []
    by_dist = _seed_values_by_strategy(runs, metric)
    for dist in sorted(by_dist):
        strategies = sorted(by_dist[dist])
        # drop the strategy with the fewest seeds until the shared seed set is usable
        while len(strategies) >= 3:
            shared = set.intersection(*[set(by_dist[dist][s]) for s in strategies])
            if len(shared) >= 2:
                break
            strategies = sorted(strategies, key=lambda s: len(by_dist[dist][s]))[1:]
            strategies = sorted(strategies)
        else:
            continue
        shared_seeds = sorted(set.intersection(*[set(by_dist[dist][s]) for s in strategies]))
        samples = [
            [by_dist[dist][s][seed] for seed in shared_seeds] for s in strategies
        ]
        chi2, p_value, note = float("nan"), float("nan"), ""
        try:
            chi2, p_value = stats.friedmanchisquare(*samples)
            chi2, p_value = float(chi2), float(p_value)
        except Exception as exc:
            note = "scipy friedman unavailable ({})".format(exc)
        pg_chi2, pg_p = float("nan"), float("nan")
        if pg is not None:
            try:
                long_df = pd.DataFrame({
                    "seed": [str(seed) for _ in strategies for seed in shared_seeds],
                    "strategy": [s for s in strategies for _ in shared_seeds],
                    "value": [v for sample in samples for v in sample],
                })
                pg_res = pg.friedman(data=long_df, dv="value", within="strategy", subject="seed")
                pg_chi2 = float(np.asarray(pg_res.iloc[:, -2])[0])
                pg_p = float(np.asarray(pg_res["p-unc"])[0])
            except Exception:
                pg_chi2, pg_p = float("nan"), float("nan")
        else:
            note = (note + "; " if note else "") + "pingouin not installed"
        rows.append({
            "distribution": dist,
            "metric": metric,
            "n_strategies": len(strategies),
            "strategies": ",".join(strategies),
            "n_seeds": len(shared_seeds),
            "seeds": ",".join(str(s) for s in shared_seeds),
            "chi_square": chi2,
            "p_value": p_value,
            "pg_chi_square": pg_chi2,
            "pg_p_value": pg_p,
            "note": note,
        })
    return pd.DataFrame(rows, columns=[
        "distribution", "metric", "n_strategies", "strategies", "n_seeds", "seeds",
        "chi_square", "p_value", "pg_chi_square", "pg_p_value", "note",
    ])


# --------------------------------------------------------------------------- #
# markdown rendering
# --------------------------------------------------------------------------- #
def _fmt(value: Any, digits: int = 4) -> str:
    fval = _as_float(value)
    if fval is None:
        return "n/a"
    if abs(fval) >= 1000:
        return "{:.1f}".format(fval)
    return "{:.{d}f}".format(fval, d=digits)


def _markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def summary_markdown(df: pd.DataFrame) -> str:
    out = ["# FedRGBD summary table", "",
           "`mean ± std [CI_low, CI_high]` — 95% CI of the mean over seeds "
           "(`±`/CI shown as `n/a` when only one seed is available).", ""]
    if df.empty:
        out.append("_No runs found._\n")
        return "\n".join(out)
    for dist in sorted(df["distribution"].unique()):
        out.append("## Distribution: `{}`".format(dist))
        out.append("")
        sub = df[df["distribution"] == dist]
        rows = []
        for _, row in sub.iterrows():
            value = "{} ± {} [{}, {}]".format(
                _fmt(row["mean"]), _fmt(row["std"]), _fmt(row["ci_low"]), _fmt(row["ci_high"])
            )
            rows.append([
                row["label"], row["metric"], int(row["n_seeds"]), value,
                _fmt(row["min"]), _fmt(row["max"]),
            ])
        out.append(_markdown_table(
            ["config", "metric", "n_seeds", "mean ± std [95% CI]", "min", "max"], rows))
    return "\n".join(out)


def pairwise_markdown(df: pd.DataFrame, metric: str) -> str:
    out = ["# Pairwise strategy comparisons (final {})".format(metric), "",
           "Paired by seed within each data distribution. `d_paired` = mean(diff) / "
           "std(diff, ddof=1); `d_unpaired` uses the pooled standard deviation.", ""]
    if df.empty:
        out.append("_No strategy pair shares at least two seeds._\n")
        return "\n".join(out)
    for dist in sorted(df["distribution"].unique()):
        out.append("## Distribution: `{}`".format(dist))
        out.append("")
        sub = df[df["distribution"] == dist]
        rows = []
        for _, row in sub.iterrows():
            rows.append([
                "{} vs {}".format(row["strategy_a"], row["strategy_b"]),
                int(row["n_seeds"]),
                _fmt(row["mean_a"]), _fmt(row["mean_b"]), _fmt(row["mean_diff"]),
                _fmt(row["cohen_d_paired"], 3), _fmt(row["cohen_d_unpaired"], 3),
                _fmt(row["wilcoxon_p"], 4), _fmt(row["ttest_p"], 4),
                row["note"] or "",
            ])
        out.append(_markdown_table(
            ["comparison", "n", "mean A", "mean B", "diff", "d_paired", "d_unpaired",
             "Wilcoxon p", "t-test p", "notes"], rows))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "unknown"


def _curve_stats(group: Sequence[Dict[str, Any]], metric: str) -> Dict[str, np.ndarray]:
    """Aggregate a config group's curves into mean/band arrays indexed by round."""
    per_round: Dict[int, Dict[str, List[float]]] = {}
    for record in group:
        for point in record.get("curve") or []:
            bucket = per_round.setdefault(
                int(point["round"]), {"y": [], "t": [], "c": []})
            y = _as_float(point.get(metric))
            if y is not None:
                bucket["y"].append(y)
            t = _as_float(point.get("elapsed_s"))
            if t is not None:
                bucket["t"].append(t)
            c = _as_float(point.get("cumulative_mb"))
            if c is not None:
                bucket["c"].append(c)
    rounds = sorted(r for r in per_round if per_round[r]["y"])
    mean, band, times, comms, counts = [], [], [], [], []
    for rnd in rounds:
        values = per_round[rnd]["y"]
        stats_dict = describe(values)
        mean.append(stats_dict["mean"])
        if stats_dict["n"] >= 2 and np.isfinite(stats_dict["ci95"]):
            band.append(stats_dict["ci95"])
        elif stats_dict["n"] >= 2 and np.isfinite(stats_dict["std"]):
            band.append(stats_dict["std"])
        else:
            band.append(0.0)
        counts.append(stats_dict["n"])
        times.append(describe(per_round[rnd]["t"])["mean"])
        comms.append(describe(per_round[rnd]["c"])["mean"])
    return {
        "rounds": np.asarray(rounds, dtype=float),
        "mean": np.asarray(mean, dtype=float),
        "band": np.asarray(band, dtype=float),
        "time": np.asarray(times, dtype=float),
        "comm": np.asarray(comms, dtype=float),
        "n": np.asarray(counts, dtype=float),
    }


def _legend_outside(fig, ax, ncol: int = 2) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=min(ncol, max(1, len(handles))), frameon=True, framealpha=0.95,
              edgecolor="#cccccc", columnspacing=1.0, handletextpad=0.5)
    fig.subplots_adjust(bottom=0.34)


def _save(fig, output_dir: str, stem: str, written: List[str]) -> None:
    for ext in (".png", ".pdf"):
        path = os.path.join(output_dir, stem + ext)
        fig.savefig(path)
        written.append(path)


def _baseline_lines(ax, baselines: Sequence[Dict[str, Any]]) -> None:
    by_kind: Dict[str, List[float]] = {}
    for record in baselines:
        value = _as_float(record.get("final_accuracy"))
        if value is not None:
            by_kind.setdefault(record["kind"], []).append(value)
    for kind, values in sorted(by_kind.items()):
        mean = float(np.mean(values))
        label = "Centralized" if kind == "centralized" else "Local-only (mean)"
        ax.axhline(mean, color=BASELINE_COLORS.get(kind, "#555555"), linestyle=":",
                   linewidth=1.5, zorder=2,
                   label="{} = {:.4f} (n={})".format(label, mean, len(values)))


def _figure_note(fig, notes: Sequence[str]) -> None:
    notes = [n for n in notes if n]
    if not notes:
        return
    fig.text(0.01, 0.005, "; ".join(notes), fontsize=6.5, style="italic", color="#555555")


def make_plots(
    runs: Sequence[Dict[str, Any]],
    output_dir: str,
    metric: str = "accuracy",
    distributions: Optional[Iterable[str]] = None,
) -> List[str]:
    """Write the convergence / time / communication / bar figures. Returns paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(IEEE_STYLE)
    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []

    curve_runs = [r for r in runs if r.get("kind") == "fl" and r.get("curve")]
    baseline_runs = [r for r in runs if r.get("kind") in ("centralized", "local")]
    if distributions is None:
        dist_values = sorted({r.get("distribution") or "unknown" for r in runs})
    else:
        dist_values = list(distributions)

    targets: List[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]] = []
    for dist in dist_values:
        targets.append((
            dist,
            [r for r in curve_runs if r.get("distribution") == dist],
            [r for r in baseline_runs if r.get("distribution") == dist],
        ))
    if len(dist_values) > 1:
        targets.append(("all", list(curve_runs), list(baseline_runs)))

    for dist, subset, baselines in targets:
        if not subset and not baselines:
            continue
        groups = _group_by_config(subset)
        prepared = []
        for idx, (config_id, group) in enumerate(sorted(groups.items())):
            stats_dict = _curve_stats(group, metric)
            if stats_dict["rounds"].size == 0:
                continue
            label = group[0]["label"]
            if dist == "all":
                label = "{} ({})".format(group[0]["strategy_display"], group[0]["distribution"])
            prepared.append({
                "label": "{} [n={}]".format(label, len(group)),
                "color": PALETTE[idx % len(PALETTE)],
                "marker": MARKER_CYCLE[idx % len(MARKER_CYCLE)],
                "time_estimated": any(r.get("time_estimated") for r in group),
                "comm_estimated": any(r.get("comm_estimated") for r in group),
                "stats": stats_dict,
            })

        ylabel = metric.replace("_", " ").title()
        # accuracy-like metrics live in [0, 1]; clip the CI band there (loss is unbounded)
        lo_clip, hi_clip = (-np.inf, np.inf) if "loss" in metric else (0.0, 1.0)
        est_time = any(p["time_estimated"] for p in prepared)
        est_comm = any(p["comm_estimated"] for p in prepared)

        # --- metric vs round ---------------------------------------------- #
        if prepared:
            fig, ax = plt.subplots(figsize=(4.8, 3.3))
            for item in prepared:
                s = item["stats"]
                ax.plot(s["rounds"], s["mean"], color=item["color"], marker=item["marker"],
                        linestyle="-", label=item["label"], markerfacecolor="white",
                        markeredgewidth=1.8, zorder=3)
                ax.fill_between(s["rounds"], np.clip(s["mean"] - s["band"], lo_clip, hi_clip), np.clip(s["mean"] + s["band"], lo_clip, hi_clip),
                                color=item["color"], alpha=0.18, linewidth=0, zorder=1)
            _baseline_lines(ax, baselines)
            ax.set_xlabel("Communication Round")
            ax.set_ylabel(ylabel)
            all_rounds = sorted({int(r) for item in prepared for r in item["stats"]["rounds"]})
            if all_rounds and len(all_rounds) <= 12:
                ax.set_xticks(all_rounds)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)
            ax.set_title("{} — {}".format(ylabel, dist), fontweight="bold", pad=8)
            _legend_outside(fig, ax)
            _figure_note(fig, ["shaded band = 95% CI over seeds (±std when n<2)"])
            _save(fig, output_dir, "{}_vs_round_{}".format(_safe_name(metric), _safe_name(dist)),
                  written)
            plt.close(fig)

            # --- metric vs wall-clock time -------------------------------- #
            fig, ax = plt.subplots(figsize=(4.8, 3.3))
            for item in prepared:
                s = item["stats"]
                ax.plot(s["time"], s["mean"], color=item["color"], marker=item["marker"],
                        linestyle="--" if item["time_estimated"] else "-",
                        label=item["label"] + (" (est. time)" if item["time_estimated"] else ""),
                        markerfacecolor="white", markeredgewidth=1.8, zorder=3)
                ax.fill_between(s["time"], np.clip(s["mean"] - s["band"], lo_clip, hi_clip), np.clip(s["mean"] + s["band"], lo_clip, hi_clip),
                                color=item["color"], alpha=0.18, linewidth=0, zorder=1)
            _baseline_lines(ax, baselines)
            ax.set_xlabel("Elapsed Wall-Clock Time (s)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)
            ax.set_title("{} vs time — {}".format(ylabel, dist), fontweight="bold", pad=8)
            _legend_outside(fig, ax)
            _figure_note(fig, [
                "dashed = time interpolated from total_time_s (estimated)" if est_time else "",
                "shaded band = 95% CI over seeds",
            ])
            _save(fig, output_dir, "{}_vs_time_{}".format(_safe_name(metric), _safe_name(dist)),
                  written)
            plt.close(fig)

            # --- metric vs communication ---------------------------------- #
            fig, ax = plt.subplots(figsize=(4.8, 3.3))
            for item in prepared:
                s = item["stats"]
                if not np.any(np.isfinite(s["comm"])):
                    continue
                ax.plot(s["comm"], s["mean"], color=item["color"], marker=item["marker"],
                        linestyle="--" if item["comm_estimated"] else "-",
                        label=item["label"] + (" (est. comm)" if item["comm_estimated"] else ""),
                        markerfacecolor="white", markeredgewidth=1.8, zorder=3)
                ax.fill_between(s["comm"], np.clip(s["mean"] - s["band"], lo_clip, hi_clip), np.clip(s["mean"] + s["band"], lo_clip, hi_clip),
                                color=item["color"], alpha=0.18, linewidth=0, zorder=1)
            ax.set_xlabel("Cumulative Communication (MB)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)
            ax.set_title("{} vs communication — {}".format(ylabel, dist),
                         fontweight="bold", pad=8)
            _legend_outside(fig, ax)
            _figure_note(fig, [
                "dashed = payload size estimated from the model state_dict" if est_comm else "",
                "shaded band = 95% CI over seeds",
            ])
            _save(fig, output_dir,
                  "{}_vs_communication_{}".format(_safe_name(metric), _safe_name(dist)), written)
            plt.close(fig)

        # --- all-metrics bar chart (new-format runs only) ------------------ #
        bar_groups = []
        for config_id, group in sorted(_group_by_config(subset).items()):
            names = set()
            for record in group:
                names |= {
                    k for k, v in (record.get("metrics_final") or {}).items()
                    if k in METRIC_ORDER and k != "loss" and _as_float(v) is not None
                }
            if len(names) > 1:
                bar_groups.append((group, names))
        if bar_groups:
            metric_names = [m for m in METRIC_ORDER if m != "loss"
                            and any(m in names for _, names in bar_groups)]
            if len(metric_names) > 1:
                fig, ax = plt.subplots(figsize=(7.16, 3.4))
                x = np.arange(len(metric_names))
                width = 0.8 / max(len(bar_groups), 1)
                for idx, (group, _names) in enumerate(bar_groups):
                    means, errs = [], []
                    for name in metric_names:
                        values = [
                            _as_float((r.get("metrics_final") or {}).get(name)) for r in group
                        ]
                        stats_dict = describe([v for v in values if v is not None])
                        means.append(stats_dict["mean"])
                        err = stats_dict["ci95"] if np.isfinite(stats_dict["ci95"]) else 0.0
                        errs.append(err)
                    offset = (idx - (len(bar_groups) - 1) / 2.0) * width
                    ax.bar(x + offset, means, width, yerr=errs, capsize=2,
                           color=PALETTE[idx % len(PALETTE)], alpha=0.85,
                           edgecolor="white", linewidth=0.5,
                           label="{} [n={}]".format(group[0]["label"], len(group)))
                ax.set_xticks(x)
                ax.set_xticklabels([m.replace("_", " ") for m in metric_names],
                                   rotation=30, ha="right", fontsize=8)
                ax.set_ylabel("Score (final round)")
                ax.set_title("Final global metrics — {}".format(dist), fontweight="bold", pad=8)
                ax.grid(True, alpha=0.2, linestyle="--", axis="y")
                ax.set_axisbelow(True)
                _legend_outside(fig, ax)
                _figure_note(fig, ["error bars = 95% CI over seeds"])
                _save(fig, output_dir, "final_metrics_{}".format(_safe_name(dist)), written)
                plt.close(fig)

    return written


# --------------------------------------------------------------------------- #
# stdout summary
# --------------------------------------------------------------------------- #
def print_summary(
    runs: Sequence[Dict[str, Any]], tables: Dict[str, pd.DataFrame],
    pairs: pd.DataFrame, friedman: pd.DataFrame, metric: str,
) -> None:
    print("")
    print("=" * 78)
    print("  FedRGBD results analysis")
    print("=" * 78)
    print("  runs loaded          : {}".format(len(runs)))
    kinds: Dict[str, int] = {}
    for record in runs:
        kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1
    print("  by kind              : {}".format(
        ", ".join("{}={}".format(k, v) for k, v in sorted(kinds.items())) or "-"))
    dists = sorted({r["distribution"] for r in runs})
    print("  distributions        : {}".format(", ".join(dists) or "-"))
    print("  configurations       : {}".format(len(_group_by_config(runs))))
    missing = [r["run_name"] for r in runs if r.get("seed") is None]
    if missing:
        print("  runs without a seed  : {}".format(", ".join(missing)))
    est_t = sum(1 for r in runs if r.get("time_estimated"))
    est_c = sum(1 for r in runs if r.get("comm_estimated"))
    print("  estimated timing/comm: {} / {} runs".format(est_t, est_c))
    print("-" * 78)

    summary = tables["summary"]
    key = "final_" + metric
    rows = summary[summary["metric"].isin([key, "final_accuracy"])]
    rows = rows[rows["metric"] == (key if (summary["metric"] == key).any() else "final_accuracy")]
    if not rows.empty:
        print("  final {} per configuration (mean ± std, n seeds)".format(metric))
        for _, row in rows.sort_values(["distribution", "mean"], ascending=[True, False]).iterrows():
            print("    {:<34} {:<16} {} ± {}  (n={})".format(
                row["label"][:34], row["distribution"][:16],
                _fmt(row["mean"]), _fmt(row["std"]), int(row["n_seeds"])))
        print("-" * 78)

    if not pairs.empty:
        print("  significant pairs (p<0.05, paired t-test):")
        sig = pairs[pairs["ttest_p"] < 0.05]
        if sig.empty:
            print("    none")
        for _, row in sig.iterrows():
            print("    [{}] {} vs {}: diff={} d={} p={}".format(
                row["distribution"], row["strategy_a"], row["strategy_b"],
                _fmt(row["mean_diff"]), _fmt(row["cohen_d_paired"], 3), _fmt(row["ttest_p"], 4)))
        print("-" * 78)

    if not friedman.empty:
        print("  Friedman omnibus:")
        for _, row in friedman.iterrows():
            print("    [{}] k={} n={} chi2={} p={}".format(
                row["distribution"], int(row["n_strategies"]), int(row["n_seeds"]),
                _fmt(row["chi_square"], 3), _fmt(row["p_value"], 4)))
        print("-" * 78)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate FedRGBD experiment results into tables, statistics and figures.")
    parser.add_argument("--results_dir", default="results",
                        help="directory holding one sub-directory per run (default: results)")
    parser.add_argument("--output_dir", default="analysis",
                        help="where tables and figures are written (default: analysis)")
    parser.add_argument("--missing_seed", type=int, default=None,
                        help="seed to assume for runs that do not record one")
    parser.add_argument("--payload_bytes", type=float, default=None,
                        help="bytes of one model transfer (default: measured from the model)")
    parser.add_argument("--local_epochs_equiv", type=int, default=DEFAULT_LOCAL_EPOCHS_EQUIV,
                        help="local epochs per FL round for centralized/local curves (default: 5)")
    parser.add_argument("--include_test_runs", action="store_true",
                        help="also analyse directories named test_*")
    parser.add_argument("--metric", default="accuracy",
                        help="metric used for curves and statistical tests (default: accuracy)")
    parser.add_argument("--no_plots", action="store_true", help="skip figure generation")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    results_dir = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(args.output_dir)
    try:
        inside = os.path.commonpath([results_dir, output_dir]) == results_dir
    except ValueError:  # different drives on Windows
        inside = False
    if inside:
        raise SystemExit("--output_dir must not be inside --results_dir ({})".format(results_dir))
    os.makedirs(output_dir, exist_ok=True)

    runs = collect_runs(
        results_dir,
        payload_bytes=args.payload_bytes,
        local_epochs_equiv=args.local_epochs_equiv,
        missing_seed=args.missing_seed,
        include_test_runs=args.include_test_runs,
    )
    if not runs:
        print("[warn] no runs found under {}".format(results_dir))

    tables = summarize(runs)
    pairs = pairwise_tests(runs, metric=args.metric)
    friedman = friedman_tests(runs, metric=args.metric)

    written: List[str] = []

    def _write_csv(df: pd.DataFrame, name: str) -> None:
        path = os.path.join(output_dir, name)
        df.to_csv(path, index=False)
        written.append(path)

    def _write_text(text: str, name: str) -> None:
        path = os.path.join(output_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        written.append(path)

    _write_csv(tables["runs"], "runs.csv")
    _write_csv(tables["summary"], "summary_table.csv")
    _write_text(summary_markdown(tables["summary"]), "summary_table.md")
    _write_csv(tables["per_round"], "per_round_table.csv")
    _write_csv(pairs, "pairwise_tests.csv")
    _write_text(pairwise_markdown(pairs, args.metric), "pairwise_tests.md")
    _write_csv(friedman, "friedman.csv")

    if not args.no_plots:
        try:
            written.extend(make_plots(runs, output_dir, metric=args.metric))
        except Exception as exc:  # never let a figure failure lose the tables
            print("[warn] plotting failed: {}".format(exc))

    print_summary(runs, tables, pairs, friedman, args.metric)
    print("  wrote {} files to {}".format(len(written), output_dir))
    for path in written:
        print("    {}".format(os.path.basename(path)))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
