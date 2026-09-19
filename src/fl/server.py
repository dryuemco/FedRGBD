"""FedRGBD — Flower FL Server with FedAvg, FedProx, and FedBN support (v3).

v3 (NCAA revision): every round the server now persists, in ``results.json``,

* per-client evaluation metrics (accuracy, balanced accuracy, precision,
  recall/sensitivity, specificity, F1, MCC, ROC-AUC, confusion matrix, loss),
* the weighted-global aggregate of those metrics **and** the "pooled" metrics
  computed from the summed confusion matrix of all clients,
* per-client fit/eval wall-clock time and model payload bytes (up/down), and
  the cumulative communication volume of the run,
* the server-side elapsed time at the end of every fit / evaluate phase.

Schema 3 (model-selection protocol): clients evaluate every round on the
validation split (``val_*`` keys; the unprefixed keys stay = validation) and on
the test split (``test_*`` keys, report only).  The server persists both sets
per client and weighted-global (test metrics weighted by test-set size), the
round timing with the test pass removed (``rounds[i]["timing"]``,
``total_time_excl_test_s``), and ``model_selection`` -- the round chosen by the
declared rule in ``src/evaluation/model_selection.py`` from validation losses
only.

All keys written by the previous version (``strategy``, ``num_rounds``,
``min_clients``, ``seed``, ``total_time_s``, ``timestamp``,
``losses_distributed``, ``metrics_distributed``) are kept unchanged so the
existing analysis code and the old result files remain compatible.
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import flwr as fl
from flwr.common import Metrics
from flwr.server.strategy import FedAvg, FedProx

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# FedBN strategy (local import)
from fedbn_strategy import FedBN  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    METRIC_KEYS,
    confusion_matrix_from_flat,
    metrics_from_confusion_matrix,
)
from src.evaluation.predictions import PRED_PREFIX, README_TEXT  # noqa: E402
from src.evaluation.model_selection import (  # noqa: E402
    SELECTION_RULE,
    SELECTION_RULE_TEXT,
    select_round,
    weighted_val_loss,
)

RESULTS_SCHEMA_VERSION = 3

#: metric namespaces sent by schema-3 clients (unprefixed keys = validation, as in v2)
NAMESPACES = ("val_", "test_")
_CM_KEY_RE = re.compile(r"^(?:val_|test_)?(?:cm_\d+_\d+|confusion_matrix_json)$")

TIMING_DEFINITION = (
    "round_time_s = server round wall-clock (end of previous evaluate phase to end of this "
    "one) minus test_eval_overhead_s, where test_eval_overhead_s = max_k(eval_wall_s) - "
    "max_k(eval_wall_s - test_eval_time_s - pred_pack_time_s) over clients k, i.e. the part "
    "of the evaluate phase's critical path spent on the report-only test pass and on packing "
    "the per-image predictions. The server's writing of the prediction files is excluded as "
    "well. elapsed_s and total_time_s are raw wall-clock and include all of these."
)

# metric keys that are averaged with num_examples weights
WEIGHTED_KEYS = set(METRIC_KEYS) | {"loss", "train_loss"}
# keys that are summed over clients
SUMMED_PREFIXES = ("payload_bytes", "cm_", "support_")
SUMMED_KEYS = {"num_examples", "tp", "fp", "fn", "tn"}
# keys for which mean AND max are reported (wall-clock)
TIME_SUFFIXES = ("_time_s", "_wall_s", "train_time")
# keys never aggregated (identity / config echo)
IDENTITY_KEYS = {"hostname", "node_name", "data_dir", "server_round", "strategy", "eval_split",
                 "local_epochs", "lr", "batch_size", "proximal_mu", "confusion_matrix_json",
                 "n_examples"}  # n_examples duplicates num_examples (summed as num_examples_total)


def set_seed(seed):
    """Set all random seeds for reproducibility on server side."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def weighted_average(metrics):
    """Aggregate accuracy across clients (v2 behaviour, kept for reference/tests)."""
    accuracies = [num * m["accuracy"] for num, m in metrics]
    totals = [num for num, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(totals)}


def model_payload_bytes() -> Optional[int]:
    """Bytes of one full model transfer (all state_dict tensors, as sent by the client)."""
    try:
        from src.models.mobilenetv3_multimodal import create_model
        model = create_model(num_classes=2, in_channels=3, pretrained=False)
        return int(sum(v.numel() * v.element_size() for v in model.state_dict().values()))
    except Exception as exc:  # pragma: no cover - torchvision missing etc.
        print(f"  [server] could not compute model payload size: {exc}")
        return None


# --------------------------------------------------------------------------- #
# per-round recorder
# --------------------------------------------------------------------------- #
class RoundRecorder:
    """Stateful metric aggregation that also remembers every client's raw metrics.

    Flower calls ``fit_metrics_aggregation_fn`` / ``evaluate_metrics_aggregation_fn``
    once per round with ``[(num_examples, metrics), ...]``.  The client echoes
    ``server_round`` and ``node_name`` in its metrics, which lets us key rows
    without subclassing the strategy.  Falls back to a call counter / hostname
    for older clients.
    """

    def __init__(self, start_time: Optional[float] = None):
        self.start_time = start_time if start_time is not None else time.perf_counter()
        self.rounds: Dict[int, dict] = {}
        self.client_config: Dict[str, dict] = {}
        self.cumulative_bytes = 0
        self._fit_calls = 0
        self._eval_calls = 0
        # model selection and timing (schema 3)
        self.val_loss_by_round: Dict[int, Optional[float]] = {}
        self.total_test_overhead_s = 0.0
        self._last_eval_elapsed = 0.0
        self._elapsed_excl_test = 0.0
        # per-image predictions (pred_*_npz byte metrics) are written here when set
        self.pred_dir: Optional[str] = None

    # -- helpers ---------------------------------------------------------- #
    def elapsed(self) -> float:
        return round(time.perf_counter() - self.start_time, 3)

    @staticmethod
    def _client_key(m: Metrics, idx: int) -> str:
        return str(m.get("node_name") or m.get("hostname") or f"client_{idx}")

    @staticmethod
    def _round_of(metrics: List[Tuple[int, Metrics]], counter: int) -> int:
        rounds = [int(m["server_round"]) for _, m in metrics if int(m.get("server_round", 0)) > 0]
        return max(rounds) if rounds else counter

    @staticmethod
    def _split_ns(key: str) -> Tuple[str, str]:
        """``"test_accuracy"`` -> ``("test_", "accuracy")``; unprefixed -> ``("", key)``."""
        for ns in NAMESPACES:
            if key.startswith(ns):
                return ns, key[len(ns):]
        return "", key

    @staticmethod
    def _client_row(num_examples: int, m: Metrics) -> dict:
        """Readable per-client row: nested confusion matrices, no flattened duplicates."""
        row = {}
        for k, v in m.items():
            if _CM_KEY_RE.match(k):
                continue
            if isinstance(v, (np.generic,)):
                v = v.item()
            row[k] = v
        row["num_examples"] = int(num_examples)
        for ns in ("",) + NAMESPACES:
            cm = confusion_matrix_from_flat(m, prefix=ns)
            if cm is not None:
                row[ns + "confusion_matrix"] = cm.tolist()
        return row

    @staticmethod
    def _is_number(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @classmethod
    def test_eval_overhead(cls, metrics: List[Tuple[int, Metrics]]) -> float:
        """Critical-path time of the evaluate phase spent on the report-only test pass.

        Clients evaluate in parallel, so the phase lasts as long as the slowest
        client.  Removing the test pass shortens it from ``max_k(eval_wall_s)`` to
        ``max_k(eval_wall_s - test_eval_time_s)``.  0 for clients without timers.
        """
        walls, without_test = [], []
        for _, m in metrics:
            wall, test = m.get("eval_wall_s"), m.get("test_eval_time_s")
            pack_t = m.get("pred_pack_time_s", 0.0)
            if not cls._is_number(pack_t) or not math.isfinite(pack_t):
                pack_t = 0.0
            if cls._is_number(wall) and cls._is_number(test) and math.isfinite(wall) \
                    and math.isfinite(test):
                walls.append(float(wall))
                without_test.append(float(wall) - float(test) - float(pack_t))
        if not walls:
            return 0.0
        return max(0.0, max(walls) - max(without_test))

    @staticmethod
    def round_val_loss(metrics: List[Tuple[int, Metrics]]) -> Optional[float]:
        """Validation loss of the round, weighted by client validation-set size.

        Reads validation keys only (``val_loss`` / ``val_n_examples``, falling back
        to the v2 ``loss`` / ``num_examples``, which are validation as well).
        """
        return weighted_val_loss(
            (m.get("val_n_examples", n), m.get("val_loss", m.get("loss")))
            for n, m in metrics
        )

    @staticmethod
    def aggregate(metrics: List[Tuple[int, Metrics]]) -> Dict[str, float]:
        """Weighted-global aggregate + pooled confusion-matrix metrics (flat, Flower-safe)."""
        total = float(sum(n for n, _ in metrics)) or 1.0
        agg: Dict[str, float] = {}
        keys = set()
        for _, m in metrics:
            keys |= set(m.keys())
        for key in sorted(keys):
            ns, base = RoundRecorder._split_ns(key)
            vals = []
            for n, m in metrics:
                if key in m and RoundRecorder._is_number(m[key]):
                    # test_* metrics are weighted by the client's test-set size,
                    # everything else by the validation count Flower passes in
                    w = m.get("test_n_examples", n) if ns == "test_" else n
                    vals.append((w, m[key]))
            if not vals or (not ns and key in IDENTITY_KEYS):
                continue
            if ns and base == "n_examples":
                agg[key + "_total"] = int(sum(v for _, v in vals))
            elif base in SUMMED_KEYS or base.startswith(SUMMED_PREFIXES):
                total_v = int(sum(v for _, v in vals))
                if key == "num_examples":
                    agg["num_examples_total"] = total_v
                elif base.startswith("payload_bytes"):
                    agg[key + "_total"] = total_v
                else:
                    agg[key] = total_v
            elif key.endswith(TIME_SUFFIXES):
                xs = [v for _, v in vals]
                agg[key + "_mean"] = float(np.mean(xs))
                agg[key + "_max"] = float(np.max(xs))
            else:  # weighted mean (metrics, loss, per-class values, ...)
                w = float(sum(n for n, _ in vals)) or 1.0
                agg[key] = float(sum(n * v for n, v in vals) / w)

        # pooled metrics from the summed confusion matrix, per namespace
        for ns in ("",) + NAMESPACES:
            cms = [confusion_matrix_from_flat(m, prefix=ns) for _, m in metrics]
            cms = [c for c in cms if c is not None]
            if not cms:
                continue
            k = max(c.shape[0] for c in cms)
            pooled_cm = np.zeros((k, k), dtype=np.int64)
            for c in cms:
                pooled_cm[: c.shape[0], : c.shape[1]] += c
            pooled = metrics_from_confusion_matrix(pooled_cm)
            for key in METRIC_KEYS:
                if key in pooled and pooled[key] is not None:
                    agg[f"pooled_{ns}{key}"] = float(pooled[key])
        return agg

    # -- Flower callbacks ------------------------------------------------- #
    def fit_aggregation(self, metrics: List[Tuple[int, Metrics]]) -> Dict[str, float]:
        self._fit_calls += 1
        rnd = self._round_of(metrics, self._fit_calls)
        clients = {}
        round_bytes = 0
        for i, (n, m) in enumerate(metrics):
            key = self._client_key(m, i)
            clients[key] = self._client_row(n, m)
            cfg = {k: m[k] for k in ("local_epochs", "lr", "batch_size", "data_dir", "hostname") if k in m}
            if cfg:
                self.client_config.setdefault(key, {}).update(cfg)
            round_bytes += int(m.get("payload_bytes_up", 0)) + int(m.get("payload_bytes_down", 0))
        agg = self.aggregate(metrics)
        self.cumulative_bytes += round_bytes
        entry = self.rounds.setdefault(rnd, {"round": rnd})
        entry["fit"] = {"clients": clients, "aggregate": agg, "elapsed_s": self.elapsed()}
        entry["cumulative_communication_bytes"] = self.cumulative_bytes
        return agg

    @staticmethod
    def pop_predictions(metrics: List[Tuple[int, Metrics]]) -> List[Tuple[str, str, bytes]]:
        """Remove every ``pred_<split>_npz`` payload from the clients' metrics -> [(client, split, bytes)].

        Runs before anything else looks at the metrics, so byte payloads never reach the
        aggregation, the per-client rows or results.json.
        """
        out = []
        for i, (_, m) in enumerate(metrics):
            client = RoundRecorder._client_key(m, i)
            # only the payloads (pred_<split>_npz); pred_pack_time_s is a timer the server needs
            for key in [k for k in m if str(k).startswith(PRED_PREFIX) and str(k).endswith("_npz")]:
                value = m.pop(key)
                if isinstance(value, (bytes, bytearray)):
                    split = key[len(PRED_PREFIX):].split("_")[0]
                    out.append((client, split, bytes(value)))
        return out

    def write_predictions(self, rnd: int, preds: List[Tuple[str, str, bytes]]) -> Dict[str, dict]:
        """Write r<round>_<client>_<split>.npz into ``pred_dir`` -> index for results.json."""
        index: Dict[str, dict] = {}
        if not preds:
            return index
        if self.pred_dir:
            os.makedirs(self.pred_dir, exist_ok=True)
            readme = os.path.join(self.pred_dir, "README.md")
            if not os.path.isfile(readme):
                with open(readme, "w", encoding="utf-8") as f:
                    f.write(README_TEXT)
        for client, split, blob in preds:
            name = "r%03d_%s_%s.npz" % (rnd, client, split)
            if self.pred_dir:
                with open(os.path.join(self.pred_dir, name), "wb") as f:
                    f.write(blob)
            index.setdefault(client, {})[split] = {"file": name, "bytes": len(blob)}
        return index

    def evaluate_aggregation(self, metrics: List[Tuple[int, Metrics]]) -> Dict[str, float]:
        self._eval_calls += 1
        preds = self.pop_predictions(metrics)          # before any other use of the metrics
        rnd = self._round_of(metrics, self._eval_calls)
        clients = {}
        round_bytes = 0
        for i, (n, m) in enumerate(metrics):
            key = self._client_key(m, i)
            clients[key] = self._client_row(n, m)
            round_bytes += int(m.get("payload_bytes_down", 0))
        agg = self.aggregate(metrics)
        self.cumulative_bytes += round_bytes
        entry = self.rounds.setdefault(rnd, {"round": rnd})
        now = self.elapsed()
        entry["evaluate"] = {"clients": clients, "aggregate": agg, "elapsed_s": now}
        entry["cumulative_communication_bytes"] = self.cumulative_bytes

        # model selection input: validation loss only
        self.val_loss_by_round[rnd] = self.round_val_loss(metrics)
        entry["weighted_val_loss"] = self.val_loss_by_round[rnd]

        # reported round time excludes the report-only test pass
        overhead = self.test_eval_overhead(metrics)
        round_wall = max(0.0, now - self._last_eval_elapsed)
        round_time = max(0.0, round_wall - overhead)
        fit_elapsed = (entry.get("fit") or {}).get("elapsed_s")
        self._elapsed_excl_test += round_time
        self.total_test_overhead_s += overhead
        entry["timing"] = {
            "round_wall_s": round(round_wall, 3),
            "test_eval_overhead_s": round(overhead, 3),
            "round_time_s": round(round_time, 3),
            "fit_phase_s": (round(fit_elapsed - self._last_eval_elapsed, 3)
                            if fit_elapsed is not None else None),
            "eval_phase_s": round(now - fit_elapsed, 3) if fit_elapsed is not None else None,
            "elapsed_excl_test_s": round(self._elapsed_excl_test, 3),
        }
        entry["evaluate"]["elapsed_excl_test_s"] = round(self._elapsed_excl_test, 3)
        if preds:
            entry["predictions"] = self.write_predictions(rnd, preds)
        # the next round starts after the prediction files are written: writing them is in
        # neither round's reported time
        self._last_eval_elapsed = self.elapsed() if preds else now
        # Flower history requires "accuracy" (v2 key) — guarantee it exists
        if "accuracy" not in agg:
            agg["accuracy"] = weighted_average(metrics)["accuracy"] if all("accuracy" in m for _, m in metrics) else 0.0
        return agg

    def to_list(self) -> List[dict]:
        return [self.rounds[r] for r in sorted(self.rounds)]


# --------------------------------------------------------------------------- #
# strategy factory
# --------------------------------------------------------------------------- #
def make_fit_config_fn(strategy_name):
    """Create on_fit_config_fn that passes strategy info to clients."""
    def fit_config(server_round: int):
        config = {"server_round": server_round}
        if strategy_name == "fedbn":
            config["fedbn"] = True
        return config
    return fit_config


def make_evaluate_config_fn():
    def evaluate_config(server_round: int):
        return {"server_round": server_round}
    return evaluate_config


def parse_mu(name: str) -> float:
    if name.startswith("fedprox"):
        return float(name.split("_")[-1]) if "_" in name else 0.01
    return 0.0


def get_strategy(name, min_clients=3, recorder: Optional[RoundRecorder] = None, **kwargs):
    """Create FL strategy by name."""
    common = dict(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=recorder.evaluate_aggregation if recorder else weighted_average,
        fit_metrics_aggregation_fn=recorder.fit_aggregation if recorder else None,
        on_fit_config_fn=make_fit_config_fn(name),
        on_evaluate_config_fn=make_evaluate_config_fn(),
    )
    common.update(kwargs)

    if name == "fedavg":
        return FedAvg(**common)
    elif name.startswith("fedprox"):
        return FedProx(proximal_mu=parse_mu(name), **common)
    elif name == "fedbn":
        return FedBN(**common)
    else:
        return FedAvg(**common)


# --------------------------------------------------------------------------- #
# results serialisation
# --------------------------------------------------------------------------- #
def json_safe(obj):
    """Make a value strictly JSON-serialisable (RFC 8259).

    ``json.dump`` happily writes the bare tokens ``NaN`` / ``Infinity``, which
    every non-Python JSON reader rejects.  A diverging local round (large
    ``proximal_mu``, high lr) produces exactly that via ``train_loss``, so the
    whole run's results.json would be unreadable outside Python.  Non-finite
    floats become ``None``; numpy scalars/arrays become Python types.
    """
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, bytes):  # pragma: no cover - no client sends bytes today
        return obj.decode("utf-8", "replace")
    return obj


def model_selection_block(recorder: RoundRecorder) -> dict:
    """``results.json["model_selection"]``: the declared rule applied to val losses.

    Selection reads ``recorder.val_loss_by_round`` only.  The test metrics of the
    selected round are copied afterwards, for reporting.
    """
    selected = select_round(recorder.val_loss_by_round)
    block = {
        "rule": SELECTION_RULE,
        "description": SELECTION_RULE_TEXT,
        "val_loss_by_round": {str(r): v for r, v in sorted(recorder.val_loss_by_round.items())},
        "selected_round": selected,
        "selected_val_loss": recorder.val_loss_by_round.get(selected) if selected else None,
        "selected_round_test": {},
    }
    if selected is not None:
        agg = ((recorder.rounds.get(selected) or {}).get("evaluate") or {}).get("aggregate") or {}
        block["selected_round_test"] = {
            k[len("test_"):]: v for k, v in agg.items()
            if k.startswith("test_") and not k.endswith(("_mean", "_max"))
        }
    return block


def build_results(args, history, total_time: float, recorder: RoundRecorder,
                  payload_bytes: Optional[int]) -> dict:
    """Assemble results.json — v2 keys first (unchanged), v3 keys appended."""
    results = {
        "strategy": args.strategy,
        "num_rounds": args.rounds,
        "min_clients": args.min_clients,
        "seed": args.seed,
        "total_time_s": round(total_time, 2),
        "timestamp": datetime.now().isoformat(),
        "losses_distributed": [
            {"round": i + 1, "loss": loss}
            for i, (_, loss) in enumerate(history.losses_distributed)
        ] if history.losses_distributed else [],
        "metrics_distributed": {
            key: [{"round": i + 1, "value": val}
                  for i, (_, val) in enumerate(values)]
            for key, values in history.metrics_distributed.items()
        } if history.metrics_distributed else {},
    }
    # ---- v3 additions ----
    results.update({
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "proximal_mu": parse_mu(args.strategy),
        "tags": list(args.tag or []),
        "address": args.address,
        "client_config": recorder.client_config,
        "model_payload_bytes": payload_bytes,
        "total_communication_bytes": recorder.cumulative_bytes,
        "metrics_distributed_fit": {
            key: [{"round": i + 1, "value": val} for i, (_, val) in enumerate(values)]
            for key, values in getattr(history, "metrics_distributed_fit", {}).items()
        },
        "rounds": recorder.to_list(),
        # ---- schema 3: model selection and test-free timing ----
        "model_selection": model_selection_block(recorder),
        "total_test_eval_overhead_s": round(recorder.total_test_overhead_s, 3),
        "total_time_excl_test_s": round(max(0.0, total_time - recorder.total_test_overhead_s), 2),
        "timing_definition": TIMING_DEFINITION,
    })
    return json_safe(results)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="fedavg",
                        help="fedavg, fedprox_<mu> (e.g. fedprox_0.01), fedbn")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--address", default="0.0.0.0:8080")
    parser.add_argument("--output_dir", default="results/fl_run")
    parser.add_argument("--min_clients", type=int, default=3,
                        help="Minimum number of clients (2 or 3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", nargs="*", default=None,
                        help="Free-form tags stored in results.json, e.g. --tag dirichlet_0.1 sub0.05")
    args = parser.parse_args(argv)

    # Set seed
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    recorder = RoundRecorder()
    recorder.pred_dir = os.path.join(args.output_dir, "predictions")
    strategy = get_strategy(args.strategy, min_clients=args.min_clients, recorder=recorder)
    payload = model_payload_bytes()

    print("=" * 60)
    print(f"  FedRGBD FL Server")
    print(f"  Strategy: {args.strategy}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Address: {args.address}")
    print(f"  Output: {args.output_dir}")
    print(f"  Min clients: {args.min_clients}")
    print(f"  Seed: {args.seed}")
    if args.tag:
        print(f"  Tags: {args.tag}")
    if payload:
        print(f"  Model payload: {payload / 1e6:.2f} MB per transfer")
    print(f"  Waiting for {args.min_clients} clients...")
    print("=" * 60)

    start = time.perf_counter()
    recorder.start_time = start

    history = fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    total_time = time.perf_counter() - start

    results = build_results(args, history, total_time, recorder, payload)
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFL completed in {total_time:.1f}s")
    print(f"Total communication: {recorder.cumulative_bytes / 1e6:.1f} MB")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
