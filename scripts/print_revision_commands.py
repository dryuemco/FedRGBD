"""FedRGBD — Print concrete commands for the NCAA revision experiment matrix.

Reads the ``revision`` section of ``configs/experiment_matrix.yaml`` and
expands every block into concrete runs: for each run it prints the intended
``results/<run>`` output directory, the ``src/fl/server.py`` command and the
three ``src/fl/client.py`` commands (one per node), or the
``scripts/train_centralized.py`` / ``scripts/train_local.py`` command for the
``baselines_extension`` block.

Runs whose ``results/<run>/results.json`` already exists are skipped by
default (``--skip_existing``, on by default) so the matrix can be resumed
across many sessions without re-launching completed work.

Usage
-----
    python3 scripts/print_revision_commands.py
    python3 scripts/print_revision_commands.py --block mu_grid
    python3 scripts/print_revision_commands.py --format bash > run_mu_grid.sh
    python3 scripts/print_revision_commands.py --no_skip_existing
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "experiment_matrix.yaml")

NODE_IPS = {
    "node_a": "192.168.1.4",
    "node_b": "192.168.1.5",
    "node_c": "192.168.1.3",
}
SERVER_ADDRESS = "192.168.1.4:8080"

DIST_TAGS = {
    "iid": "iid",
    "non_iid_label": "noniid",
}

_DEFAULT_ROUNDS = 3
_DEFAULT_LOCAL_EPOCHS = 5
_DEFAULT_LR = 0.001


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fmt_num(x) -> str:
    """Format a float/int the way it should appear in a split name or dir name.

    1.0 -> "1", 0.1 -> "0.1", 5 -> "5".
    """
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    return str(x)


def dist_tag(dist: str) -> str:
    return DIST_TAGS.get(dist, dist)


def dirichlet_split(alpha) -> str:
    return f"dirichlet_{fmt_num(alpha)}"


def dirichlet_dist_tag(alpha) -> str:
    return f"dirichlet{fmt_num(alpha)}"


def subsample_split(dist: str, frac) -> str:
    return f"{dist}_sub{fmt_num(frac)}"


def subsample_dist_tag(dist: str, frac) -> str:
    return f"{dist_tag(dist)}_sub{fmt_num(frac)}"


def make_output_dir(dist: str, strategy: str, seed, rounds=_DEFAULT_ROUNDS,
                    local_epochs=_DEFAULT_LOCAL_EPOCHS, lr=_DEFAULT_LR) -> str:
    """``results/rev_<dist>_<strategy>[_ep<E>][_lr<LR>][_r<R>]_seed<S>``.

    The optional suffixes are only appended when the value differs from the
    block default, so default-valued cells of a sweep share a directory with
    the equivalent cell of another block (see configs/experiment_matrix.yaml
    comments on the `revision` section).
    """
    parts = [f"rev_{dist}_{strategy}"]
    if local_epochs != _DEFAULT_LOCAL_EPOCHS:
        parts.append(f"ep{local_epochs}")
    if lr != _DEFAULT_LR:
        parts.append(f"lr{fmt_num(lr)}")
    if rounds != _DEFAULT_ROUNDS:
        parts.append(f"r{rounds}")
    parts.append(f"seed{seed}")
    return "results/" + "_".join(parts)


@dataclass
class Run:
    block: str
    output_dir: str
    kind: str  # "fl" or "baseline"
    dist: str
    split: str
    strategy: Optional[str] = None
    baseline_type: Optional[str] = None
    seed: int = 42
    rounds: int = _DEFAULT_ROUNDS
    local_epochs: int = _DEFAULT_LOCAL_EPOCHS
    lr: float = _DEFAULT_LR
    batch_size: int = 8
    epochs: int = field(default=0)  # baselines only

    def results_json(self) -> str:
        return os.path.join(REPO_ROOT, self.output_dir, "results.json")

    def exists(self) -> bool:
        return os.path.exists(self.results_json())

    def server_command(self) -> str:
        return (
            f"python3 src/fl/server.py --strategy {self.strategy} --rounds {self.rounds} "
            f"--seed {self.seed} --min_clients 3 --output_dir {self.output_dir} --tag {self.dist}"
        )

    def client_commands(self) -> List[str]:
        cmds = []
        for node in ("node_a", "node_b", "node_c"):
            data_dir = f"data/processed/{self.split}/{node}"
            cmds.append(
                f"python3 src/fl/client.py --server {SERVER_ADDRESS} --data_dir {data_dir} "
                f"--batch_size {self.batch_size} --seed {self.seed} "
                f"--local_epochs {self.local_epochs} --lr {self.lr}"
            )
        return cmds

    def baseline_command(self) -> str:
        data_dirs = " ".join(f"data/processed/{self.split}/{node}" for node in ("node_a", "node_b", "node_c"))
        if self.baseline_type == "centralized":
            return (
                f"python3 scripts/train_centralized.py --data_dirs {data_dirs} "
                f"--epochs {self.epochs} --batch_size {self.batch_size} --lr {self.lr} "
                f"--seed {self.seed} --output_dir {self.output_dir}"
            )
        # local_only
        return (
            f"python3 scripts/train_local.py --batch --cross_eval --data_dirs {data_dirs} "
            f"--epochs {self.epochs} --batch_size {self.batch_size} --lr {self.lr} "
            f"--seed {self.seed} --output_dir {self.output_dir}"
        )


# --------------------------------------------------------------------------- #
# per-block expansion
# --------------------------------------------------------------------------- #
def _fl_run(block, dist, split, strategy, seed, rounds, local_epochs, lr, batch_size) -> Run:
    return Run(
        block=block,
        output_dir=make_output_dir(dist_tag(dist), strategy, seed, rounds, local_epochs, lr),
        kind="fl",
        dist=dist_tag(dist),
        split=split,
        strategy=strategy,
        seed=seed,
        rounds=rounds,
        local_epochs=local_epochs,
        lr=lr,
        batch_size=batch_size,
    )


def expand_seed_extension(cfg: dict) -> List[Run]:
    # By default only the seeds that do not exist yet are emitted.  Set
    # ``all_seeds: true`` in the block (or pass --all_seeds) to regenerate all
    # five seeds, which is REQUIRED when data/processed was re-split (e.g. with
    # --group_file): the old 3-seed runs used a different partition and are
    # not comparable with new ones.
    seeds = cfg["seeds"] if cfg.get("all_seeds") else cfg.get("new_seeds", cfg["seeds"])
    runs = []
    for strategy in cfg["strategies"]:
        for dist in cfg["data_distributions"]:
            for seed in seeds:
                runs.append(_fl_run("seed_extension", dist, dist, strategy, seed,
                                     cfg["rounds"], cfg["local_epochs"], cfg["lr"], cfg["batch_size"]))
    return runs


def expand_dirichlet_skew(cfg: dict) -> List[Run]:
    runs = []
    for alpha in cfg["alphas"]:
        split = dirichlet_split(alpha)
        dist = dirichlet_dist_tag(alpha)
        for strategy in cfg["strategies"]:
            for seed in cfg["seeds"]:
                runs.append(_fl_run("dirichlet_skew", dist, split, strategy, seed,
                                     cfg["rounds"], cfg["local_epochs"], cfg["lr"], cfg["batch_size"]))
    return runs


def expand_low_data(cfg: dict) -> List[Run]:
    runs = []
    for frac in cfg["fractions"]:
        for dist in cfg["data_distributions"]:
            split = subsample_split(dist, frac)
            dtag = subsample_dist_tag(dist, frac)
            for strategy in cfg["strategies"]:
                for seed in cfg["seeds"]:
                    runs.append(_fl_run("low_data", dtag, split, strategy, seed,
                                         cfg["rounds"], cfg["local_epochs"], cfg["lr"], cfg["batch_size"]))
    return runs


def expand_long_horizon_fedbn(cfg: dict) -> List[Run]:
    runs = []
    for strategy in cfg["strategies"]:
        for dist in cfg["data_distributions"]:
            for seed in cfg["seeds"]:
                runs.append(_fl_run("long_horizon_fedbn", dist, dist, strategy, seed,
                                     cfg["rounds"], cfg["local_epochs"], cfg["lr"], cfg["batch_size"]))
    return runs


def expand_mu_grid(cfg: dict) -> List[Run]:
    runs = []
    for mu in cfg["mus"]:
        strategy = f"fedprox_{fmt_num(mu)}"
        for dist in cfg["data_distributions"]:
            for seed in cfg["seeds"]:
                runs.append(_fl_run("mu_grid", dist, dist, strategy, seed,
                                     cfg["rounds"], cfg["local_epochs"], cfg["lr"], cfg["batch_size"]))
    return runs


def expand_local_epochs(cfg: dict) -> List[Run]:
    runs = []
    for e in cfg["local_epochs_values"]:
        for strategy in cfg["strategies"]:
            for dist in cfg["data_distributions"]:
                for seed in cfg["seeds"]:
                    runs.append(_fl_run("local_epochs", dist, dist, strategy, seed,
                                         cfg["rounds"], e, cfg["lr"], cfg["batch_size"]))
    return runs


def expand_learning_rate(cfg: dict) -> List[Run]:
    runs = []
    for lr in cfg["lrs"]:
        for strategy in cfg["strategies"]:
            for dist in cfg["data_distributions"]:
                for seed in cfg["seeds"]:
                    runs.append(_fl_run("learning_rate", dist, dist, strategy, seed,
                                         cfg["rounds"], cfg["local_epochs"], lr, cfg["batch_size"]))
    return runs


def _baseline_run(block, dist, split, baseline_type, seed, rounds, local_epochs, lr, batch_size) -> Run:
    strategy = "centralized" if baseline_type == "centralized" else "local"
    epochs = rounds * local_epochs
    return Run(
        block=block,
        output_dir=make_output_dir(dist_tag(dist), strategy, seed, rounds, local_epochs, lr),
        kind="baseline",
        dist=dist_tag(dist),
        split=split,
        baseline_type=baseline_type,
        seed=seed,
        rounds=rounds,
        local_epochs=local_epochs,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
    )


def expand_baselines_extension(cfg: dict) -> List[Run]:
    runs = []
    rounds, local_epochs, lr, batch_size = cfg["rounds"], cfg["local_epochs"], cfg["lr"], cfg["batch_size"]

    new_seeds_part = cfg["parts"]["new_seeds"]
    for baseline_type in cfg["baseline_types"]:
        for dist in new_seeds_part["data_distributions"]:
            for seed in new_seeds_part["seeds"]:
                runs.append(_baseline_run("baselines_extension", dist, dist, baseline_type, seed,
                                           rounds, local_epochs, lr, batch_size))

    dirichlet_part = cfg["parts"]["dirichlet"]
    for baseline_type in cfg["baseline_types"]:
        for alpha in dirichlet_part["alphas"]:
            split = dirichlet_split(alpha)
            dtag = dirichlet_dist_tag(alpha)
            for seed in dirichlet_part["seeds"]:
                runs.append(_baseline_run("baselines_extension", dtag, split, baseline_type, seed,
                                           rounds, local_epochs, lr, batch_size))

    subsample_part = cfg["parts"]["subsample"]
    for baseline_type in cfg["baseline_types"]:
        for dist in subsample_part["data_distributions"]:
            for frac in subsample_part["fractions"]:
                split = subsample_split(dist, frac)
                dtag = subsample_dist_tag(dist, frac)
                for seed in subsample_part["seeds"]:
                    runs.append(_baseline_run("baselines_extension", dtag, split, baseline_type, seed,
                                               rounds, local_epochs, lr, batch_size))
    return runs


BLOCK_EXPANDERS = {
    "seed_extension": expand_seed_extension,
    "dirichlet_skew": expand_dirichlet_skew,
    "low_data": expand_low_data,
    "long_horizon_fedbn": expand_long_horizon_fedbn,
    "mu_grid": expand_mu_grid,
    "local_epochs": expand_local_epochs,
    "learning_rate": expand_learning_rate,
    "baselines_extension": expand_baselines_extension,
}

# Order in which blocks are expanded/printed when no --block filter is given.
BLOCK_ORDER = [
    "seed_extension",
    "dirichlet_skew",
    "low_data",
    "long_horizon_fedbn",
    "mu_grid",
    "local_epochs",
    "learning_rate",
    "baselines_extension",
]


def expand_all(revision_cfg: dict, block: Optional[str] = None) -> Dict[str, List[Run]]:
    names = [block] if block else BLOCK_ORDER
    out = {}
    for name in names:
        if name not in BLOCK_EXPANDERS:
            raise SystemExit(f"Unknown block {name!r}; choices: {', '.join(BLOCK_ORDER)}")
        out[name] = BLOCK_EXPANDERS[name](revision_cfg[name])
    return out


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #
def print_text(runs_by_block: Dict[str, List[Run]], skip_existing: bool) -> int:
    printed = 0
    for block, runs in runs_by_block.items():
        print(f"\n{'=' * 70}")
        print(f"Block: {block}  ({len(runs)} runs)")
        print("=" * 70)
        for run in runs:
            if skip_existing and run.exists():
                print(f"[SKIP] {run.output_dir} (results.json exists)")
                continue
            printed += 1
            print(f"\n--- {run.output_dir} ---")
            if run.kind == "fl":
                print(f"  Node A (server, {NODE_IPS['node_a']}):")
                print(f"    {run.server_command()}")
                for node, cmd in zip(("node_a", "node_b", "node_c"), run.client_commands()):
                    print(f"  {node} ({NODE_IPS[node]}):")
                    print(f"    {cmd}")
            else:
                print(f"  {run.baseline_command()}")
    return printed


def print_bash(runs_by_block: Dict[str, List[Run]], skip_existing: bool) -> int:
    printed = 0
    print("#!/bin/bash")
    print("# Auto-generated by scripts/print_revision_commands.py — run on the SERVER node (Node A).")
    print("set -e")
    for block, runs in runs_by_block.items():
        print(f"\necho '=== Block: {block} ({len(runs)} runs) ==='")
        for run in runs:
            if skip_existing and run.exists():
                print(f"echo '[SKIP] {run.output_dir} already exists'")
                continue
            printed += 1
            print(f"\necho '--- {run.output_dir} ---'")
            if run.kind == "fl":
                for node, cmd in zip(("node_a", "node_b", "node_c"), run.client_commands()):
                    print(f"echo '>>> START ON {node} ({NODE_IPS[node]}):'")
                    print(f"echo '  {cmd}'")
                print("echo '>>> Starting server in 10 seconds... (start clients on other nodes NOW)'")
                print("sleep 10")
                print(f"{run.server_command()}")
                print(f"echo '[DONE] {run.output_dir}'")
                print("echo '>>> Waiting 30s before next run...'")
                print("sleep 30")
            else:
                print(run.baseline_command())
                print(f"echo '[DONE] {run.output_dir}'")
    return printed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG, help="path to experiment_matrix.yaml")
    p.add_argument("--block", default=None, choices=BLOCK_ORDER, help="only print this block")
    p.add_argument("--format", choices=["text", "bash"], default="text")
    p.add_argument("--skip_existing", dest="skip_existing", action="store_true", default=True,
                   help="skip runs whose results/<run>/results.json already exists (default: on)")
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false",
                   help="print every run regardless of whether results already exist")
    p.add_argument("--all_seeds", action="store_true",
                   help="seed_extension: emit all 5 seeds instead of only the new ones "
                        "(use after re-splitting the data, e.g. with --group_file)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    revision_cfg = cfg["revision"]
    if args.all_seeds:
        revision_cfg = dict(revision_cfg)
        revision_cfg["seed_extension"] = dict(revision_cfg["seed_extension"], all_seeds=True)
    runs_by_block = expand_all(revision_cfg, args.block)

    if args.format == "bash":
        n = print_bash(runs_by_block, args.skip_existing)
    else:
        n = print_text(runs_by_block, args.skip_existing)

    total = sum(len(v) for v in runs_by_block.values())
    print(f"\n# {n}/{total} run(s) printed"
          f"{' (skipping ones with existing results.json)' if args.skip_existing else ''}.",
          file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
