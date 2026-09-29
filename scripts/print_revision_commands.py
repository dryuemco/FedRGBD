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

Power configurations
--------------------
``--power_config`` names the Jetson power modes a block runs under.  The main
matrix ran under ``heterogeneous`` (node_a 15W, node_b MAXN_SUPER, node_c 7W:
never harmonised) and writes ``results/rev_*``.  Any other configuration writes
its federated runs to ``results/pc_<name>/rev_*``, so a MAXN rerun of a cell is
never skipped as "already done" because the heterogeneous run exists, and
``analyze_results.py`` never pools the two (it reads the configuration back from
the directory).  Centralized / local-only baselines run on the desktop GPU and
are not affected.

A block may *declare* its configuration (``power_config: maxn`` in
``configs/experiment_matrix.yaml``).  Its runs then always live in that
namespace, whatever ``--power_config`` says, and naming a different
configuration explicitly for that block is an error: a block written for MAXN can
never be emitted for the heterogeneous testbed.

Identity gates.  A block may list ``identity_gates``: rounds whose per-image
predictions must be bitwise identical to a reference run of the heterogeneous
matrix.  They are emitted into the bash script as ``>>> IDENTITY GATE:`` lines,
which ``scripts/run_matrix.py`` parses and enforces.

Camera experiment.  ``camera_sensor_skew`` (question (b) of
docs/CAMERA_EXPERIMENT_PREREG.md, declared ``power_config: maxn``), the two smoke runs
of its determinism gate ``camera_determinism_smoke`` and its desktop baselines
``camera_sensor_skew_baselines`` are not part of the FLAME matrix: they are not in the
default output, are emitted only with ``--block``, and write to results/camera/.

Usage
-----
    python3 scripts/print_revision_commands.py
    python3 scripts/print_revision_commands.py --all_seeds --block seed_extension --power_config maxn
    python3 scripts/print_revision_commands.py --block mu_grid
    python3 scripts/print_revision_commands.py --format bash > run_mu_grid.sh
    python3 scripts/print_revision_commands.py --no_skip_existing
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "experiment_matrix.yaml")

NODE_IPS = {
    "node_a": "192.168.1.10",
    "node_b": "192.168.1.7",
    "node_c": "192.168.1.6",
}
SERVER_ADDRESS = "192.168.1.10:8080"

DIST_TAGS = {
    "iid": "iid",
    "non_iid_label": "noniid",
}

#: Jetson power configurations.  ``heterogeneous`` is the main matrix (results/rev_*);
#: every other one lives in its own namespace, results/pc_<name>/rev_*.
POWER_CONFIGS = ("heterogeneous", "maxn")
DEFAULT_POWER_CONFIG = "heterogeneous"

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
    #: power configuration the block declares (None: follows --power_config)
    power_config: Optional[str] = None
    #: [(reference_run_dir, [rounds])] -- bitwise-identity gates (run_matrix.py)
    identity: List[Tuple[str, List[int]]] = field(default_factory=list)
    #: commands printed after a baseline's training command (camera baselines: the
    #: per-image predictions of the selected model, which the analysis reads)
    post_commands: List[str] = field(default_factory=list)

    def results_json(self) -> str:
        return os.path.join(REPO_ROOT, self.output_dir, "results.json")

    def exists(self) -> bool:
        """``results.json`` (FL, centralized) or ``summary.json`` (``train_local.py --batch``
        writes a per-node tree plus a summary) marks a finished run."""
        return (os.path.exists(self.results_json())
                or os.path.exists(os.path.join(REPO_ROOT, self.output_dir, "summary.json")))

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


def _identity_gates(cfg: dict, dist: str, strategy: str, seed) -> List[Tuple[str, List[int]]]:
    """The (reference run dir, rounds) gates of ``identity_gates`` that apply to a run.

    References are runs of the heterogeneous matrix (``results/rev_*``) with the same
    strategy, partition and seed and ``reference_rounds`` rounds.
    """
    gates = []
    for gate in cfg.get("identity_gates") or []:
        if strategy not in gate.get("strategies", [strategy]):
            continue
        if dist not in gate.get("data_distributions", [dist]):
            continue
        if seed not in gate.get("seeds", [seed]):
            continue
        ref = make_output_dir(dist_tag(dist), strategy, seed, rounds=int(gate["reference_rounds"]),
                              local_epochs=cfg["local_epochs"], lr=cfg["lr"])
        rounds = sorted(int(r) for r in gate["compare_rounds"])
        if not rounds or rounds[-1] > int(cfg["rounds"]) or rounds[-1] > int(gate["reference_rounds"]):
            raise SystemExit("identity gate %r compares rounds %s beyond a run's budget"
                             % (gate.get("name"), rounds))
        gates.append((ref, rounds))
    return gates


def expand_maxn_long_horizon(cfg: dict) -> List[Run]:
    runs = []
    for strategy in cfg["strategies"]:
        for dist in cfg["data_distributions"]:
            for seed in cfg["seeds"]:
                run = _fl_run("maxn_long_horizon", dist, dist, strategy, seed,
                              cfg["rounds"], cfg["local_epochs"], cfg["lr"], cfg["batch_size"])
                run.identity = _identity_gates(cfg, dist, strategy, seed)
                runs.append(run)
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
    # Under the leakage-safe re-split the old-seed centralized / local-only runs of the
    # iid and non_iid_label splits are not comparable either, so ``--all_seeds`` (or
    # ``all_seeds: true`` in the block) emits all five seeds here as well.
    baseline_seeds = (new_seeds_part.get("all_seeds", new_seeds_part["seeds"])
                      if cfg.get("all_seeds") else new_seeds_part["seeds"])
    for baseline_type in cfg["baseline_types"]:
        for dist in new_seeds_part["data_distributions"]:
            for seed in baseline_seeds:
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


# --------------------------------------------------------------------------- #
# camera experiment, question (b): sensor-skewed clients (docs/CAMERA_EXPERIMENT_PREREG.md
# section 6).  One yaml block, ``camera_sensor_skew``, expanded three times: its 45
# federated runs (testbed, MAXN_SUPER; the block declares ``power_config: maxn``); under
# ``camera_determinism_smoke`` the two smoke runs its determinism gate compares (testbed,
# same configuration, run before the block); and under ``camera_sensor_skew_baselines``
# its local-only / centralized desktop baselines -- a separate name so that the bash
# run_matrix.py parses never contains a ``scripts/train_*`` line (it refuses those: they
# run on the desktop GPU).  All camera runs live in results/camera/ (testbed:
# results/camera/<power_config>/, desktop: results/camera/desktop/), separate from FLAME by
# dataset.  None is in BLOCK_ORDER: the camera experiment is not part of the FLAME matrix,
# so the default (all-block) output and every FLAME block stay exactly as they were
# (tests/test_revision_commands_frozen.py); select them with ``--block``.
# --------------------------------------------------------------------------- #
CAMERA_BLOCK = "camera_sensor_skew"
CAMERA_BASELINE_BLOCK = "camera_sensor_skew_baselines"
CAMERA_SMOKE_BLOCK = "camera_determinism_smoke"
#: the camera experiment's own results tree, separate from FLAME by dataset (not only by
#: power configuration): testbed runs in results/camera/<power_config>/, desktop GPU runs
#: in results/camera/desktop/.  scripts/analyze_results.py never reads it.
CAMERA_RESULTS = "results/camera"
CAMERA_BASELINE_ROOT = CAMERA_RESULTS + "/desktop"
#: camera blocks that run on the testbed (namespace results/camera/<power_config>/)
CAMERA_TESTBED_BLOCKS = (CAMERA_BLOCK, CAMERA_SMOKE_BLOCK)


def camera_root(power_config: str) -> str:
    """Results root of camera testbed runs under ``power_config``."""
    if power_config not in POWER_CONFIGS:
        raise SystemExit(f"Unknown power configuration {power_config!r}; "
                         f"choices: {', '.join(POWER_CONFIGS)}")
    return f"{CAMERA_RESULTS}/{power_config}"


def camera_split(fold) -> str:
    """``data/processed/<split>/node_{a,b,c}`` of a federated camera fold."""
    return f"camera_fold{int(fold)}"


def expand_camera_sensor_skew(cfg: dict) -> List[Run]:
    """45 federated runs: strategies x folds x seeds, ``rev_camera_fold<f>_<strategy>_r10_seed<S>``
    (the declared power configuration puts them in results/camera/<power_config>/)."""
    runs = []
    for strategy in cfg["strategies"]:
        for fold in cfg["folds"]:
            split = camera_split(fold)
            for seed in cfg["seeds"]:
                runs.append(_fl_run(CAMERA_BLOCK, split, split, strategy, seed,
                                    cfg["rounds"], cfg["local_epochs"], cfg["lr"], cfg["batch_size"]))
    return runs


def expand_camera_determinism_smoke(cfg: dict) -> List[Run]:
    """The two smoke runs of the camera block's determinism gate: the same command twice
    (``determinism_gate.smoke``: fold, strategy, seed, rounds; local epochs, lr and batch
    size of the block), written to the two directories ``determinism_gate.runs`` names."""
    gate = cfg["determinism_gate"]
    smoke, names = gate["smoke"], list(gate["runs"])
    if len(names) != 2:
        raise SystemExit(f"{CAMERA_BLOCK}.determinism_gate must name two runs, got {names!r}")
    split = camera_split(smoke["fold"])
    runs = []
    for name in names:
        run = _fl_run(CAMERA_SMOKE_BLOCK, split, split, smoke["strategy"], smoke["seed"],
                      smoke["rounds"], cfg["local_epochs"], cfg["lr"], cfg["batch_size"])
        run.output_dir = "results/" + os.path.basename(name.replace("\\", "/").rstrip("/"))
        runs.append(run)
    return runs


def expand_camera_sensor_skew_baselines(cfg: dict) -> List[Run]:
    """Desktop GPU baselines of the camera block: local-only (``train_local.py --batch``,
    one model per node) and centralized, epochs = rounds x local_epochs (10 x 5 = 50),
    same optimiser, per fold and seed, under ``results/camera/desktop/``."""
    part = cfg["desktop_baselines"]
    root = part.get("output_root", CAMERA_BASELINE_ROOT).replace("\\", "/").rstrip("/")
    if not root.startswith(CAMERA_RESULTS + "/"):
        raise SystemExit(f"camera baselines must write under {CAMERA_RESULTS}/, not {root}")
    runs = []
    for baseline_type in part["baseline_types"]:
        for fold in cfg["folds"]:
            split = camera_split(fold)
            for seed in cfg["seeds"]:
                run = _baseline_run(CAMERA_BASELINE_BLOCK, split, split, baseline_type, seed,
                                    cfg["rounds"], cfg["local_epochs"], cfg["lr"], cfg["batch_size"])
                run.output_dir = root + "/" + run.output_dir[len("results/"):]
                run.post_commands = [f"python3 scripts/predict_from_checkpoint.py {run.output_dir}"]
                runs.append(run)
    return runs


def apply_all_seeds(revision_cfg: dict) -> dict:
    """The transformation ``--all_seeds`` applies to the config.

    Only ``seed_extension`` and ``baselines_extension`` are seed-limited by default
    (they otherwise emit only the *new* seeds); every other block already spans its
    full seed set.  ``scripts/run_matrix.py`` always passes ``--all_seeds``
    (CLAUDE.md hard rule 3), so anything that needs to know which runs the testbed
    will actually execute must apply this first -- see
    ``tests/test_block_report.py``, which pins the two to each other.
    """
    out = dict(revision_cfg)
    for block in ("seed_extension", "baselines_extension"):
        if block in out:
            out[block] = dict(out[block], all_seeds=True)
    return out


BLOCK_EXPANDERS = {
    "seed_extension": expand_seed_extension,
    "dirichlet_skew": expand_dirichlet_skew,
    "low_data": expand_low_data,
    "long_horizon_fedbn": expand_long_horizon_fedbn,
    "maxn_long_horizon": expand_maxn_long_horizon,
    "mu_grid": expand_mu_grid,
    "local_epochs": expand_local_epochs,
    "learning_rate": expand_learning_rate,
    "baselines_extension": expand_baselines_extension,
    CAMERA_BLOCK: expand_camera_sensor_skew,
    CAMERA_BASELINE_BLOCK: expand_camera_sensor_skew_baselines,
    CAMERA_SMOKE_BLOCK: expand_camera_determinism_smoke,
}

#: blocks expanded from another block's yaml entry (the camera baselines live in
#: ``camera_sensor_skew.desktop_baselines``, the gate's smoke runs in
#: ``camera_sensor_skew.determinism_gate``)
BLOCK_CONFIG_KEYS = {CAMERA_BASELINE_BLOCK: CAMERA_BLOCK, CAMERA_SMOKE_BLOCK: CAMERA_BLOCK}
#: derived blocks that inherit their parent's power configuration (the desktop baselines
#: declare none: they do not run on the testbed)
POWER_CONFIG_KEYS = {CAMERA_SMOKE_BLOCK: CAMERA_BLOCK}

# Order in which blocks are expanded/printed when no --block filter is given.
BLOCK_ORDER = [
    "seed_extension",
    "dirichlet_skew",
    "low_data",
    "long_horizon_fedbn",
    "mu_grid",
    "local_epochs",
    "learning_rate",
    "maxn_long_horizon",
    "baselines_extension",
]

#: a line printed under a block's header (text and bash)
BLOCK_NOTES = {
    CAMERA_BASELINE_BLOCK: "DESKTOP GPU BASELINES (camera, question b) -- run on the desktop, "
                           "never with scripts/run_matrix.py",
}

#: blocks outside the FLAME matrix: never part of the default (all-block) output, only
#: emitted with ``--block``
EXTRA_BLOCKS = [CAMERA_BLOCK, CAMERA_BASELINE_BLOCK, CAMERA_SMOKE_BLOCK]


def expand_all(revision_cfg: dict, block: Optional[str] = None) -> Dict[str, List[Run]]:
    names = [block] if block else BLOCK_ORDER
    out = {}
    for name in names:
        if name not in BLOCK_EXPANDERS:
            raise SystemExit(f"Unknown block {name!r}; choices: {', '.join(BLOCK_ORDER)}")
        runs = BLOCK_EXPANDERS[name](revision_cfg[BLOCK_CONFIG_KEYS.get(name, name)])
        declared = block_power_config(revision_cfg, name)
        if declared:
            root = camera_root(declared) if name in CAMERA_TESTBED_BLOCKS \
                else power_config_root(declared)
            for run in runs:
                run.power_config = declared
                if run.kind == "fl" and root != "results" and run.output_dir.startswith("results/"):
                    run.output_dir = root + "/" + run.output_dir[len("results/"):]
        out[name] = runs
    return out


def block_power_config(revision_cfg: dict, block: str) -> Optional[str]:
    """The power configuration a block declares, or None."""
    return (revision_cfg.get(POWER_CONFIG_KEYS.get(block, block)) or {}).get("power_config")


def rebase_baselines(runs_by_block: Dict[str, List[Run]], root: str) -> None:
    """Move every baseline run's output dir from ``results/<run>`` to ``<root>/<run>``."""
    root = root.replace("\\", "/").rstrip("/")
    for runs in runs_by_block.values():
        for run in runs:
            if run.output_dir.startswith(CAMERA_RESULTS + "/"):
                continue            # the camera tree is fixed (results/camera/desktop)
            if run.kind == "baseline" and run.output_dir.startswith("results/"):
                run.output_dir = root + "/" + run.output_dir[len("results/"):]


def power_config_root(power_config: str) -> str:
    """Results root of a power configuration: ``results`` or ``results/pc_<name>``."""
    if power_config not in POWER_CONFIGS:
        raise SystemExit(f"Unknown power configuration {power_config!r}; "
                         f"choices: {', '.join(POWER_CONFIGS)}")
    return "results" if power_config == DEFAULT_POWER_CONFIG else f"results/pc_{power_config}"


def rebase_power_config(runs_by_block: Dict[str, List[Run]], power_config: str) -> None:
    """Move every federated run to the namespace of ``power_config``.

    Baselines are left alone: they run on the desktop GPU, not on the Jetsons.
    """
    root = power_config_root(power_config)
    if root == "results":
        return
    for runs in runs_by_block.values():
        for run in runs:
            if run.power_config is not None:
                continue            # the block declares its own namespace
            if run.kind == "fl" and run.output_dir.startswith("results/"):
                run.output_dir = root + "/" + run.output_dir[len("results/"):]


def check_declared_namespaces(revision_cfg: dict, runs_by_block: Dict[str, List[Run]]) -> None:
    """A run moved into a power namespace must not land in a directory that belongs to a
    block declaring that configuration: it would be skipped there as existing and carry
    none of that block's identity gates (e.g. long_horizon_fedbn with --power_config
    maxn writes exactly the six gated maxn_long_horizon ten-round directories)."""
    declared_dirs = {}
    for name, block in revision_cfg.items():
        if isinstance(block, dict) and block.get("power_config") and name in BLOCK_EXPANDERS:
            for run in expand_all(revision_cfg, name)[name]:
                declared_dirs[run.output_dir] = name
    for name, runs in runs_by_block.items():
        for run in runs:
            owner = declared_dirs.get(run.output_dir)
            if owner and owner != name:
                raise SystemExit(
                    f"{run.output_dir} of block {name!r} belongs to block {owner!r}, which "
                    f"declares its power configuration; run {owner!r} instead")


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #
def emit_status(run: Run, skip_existing: bool, seen: set) -> str:
    """Return "dup" / "skip" / "run" for one cell and mark its output dir as seen.

    The mu / local-epoch / learning-rate sweeps deliberately share a results
    directory with the seed-extension block at their default value.
    ``--skip_existing`` only looks at the filesystem *while the list is
    generated*, so without this de-duplication a freshly generated script would
    launch those 9 identical configurations two to four times (1.5-3 h of
    testbed time each) and every repeat would overwrite the previous
    results.json.
    """
    if run.output_dir in seen:
        return "dup"
    seen.add(run.output_dir)
    if skip_existing and run.exists():
        return "skip"
    return "run"


def print_text(runs_by_block: Dict[str, List[Run]], skip_existing: bool) -> int:
    printed = 0
    seen = set()
    for block, runs in runs_by_block.items():
        print(f"\n{'=' * 70}")
        print(f"Block: {block}  ({len(runs)} runs)")
        print("=" * 70)
        if block in BLOCK_NOTES:
            print(f"# {BLOCK_NOTES[block]}")
        for run in runs:
            status = emit_status(run, skip_existing, seen)
            if status == "dup":
                print(f"[DUP]  {run.output_dir} (same config already listed in an earlier block)")
                continue
            if status == "skip":
                print(f"[SKIP] {run.output_dir} (results.json exists)")
                continue
            printed += 1
            print(f"\n--- {run.output_dir} ---")
            for ref, rounds in run.identity:
                print(f"  identity gate: rounds {','.join(map(str, rounds))} == {ref}")
            if run.kind == "fl":
                print(f"  Node A (server, {NODE_IPS['node_a']}):")
                print(f"    {run.server_command()}")
                for node, cmd in zip(("node_a", "node_b", "node_c"), run.client_commands()):
                    print(f"  {node} ({NODE_IPS[node]}):")
                    print(f"    {cmd}")
            else:
                print(f"  {run.baseline_command()}")
                for cmd in run.post_commands:
                    print(f"  {cmd}")
    return printed


def print_bash(runs_by_block: Dict[str, List[Run]], skip_existing: bool,
               power_config: str = DEFAULT_POWER_CONFIG) -> int:
    printed = 0
    seen = set()
    print("#!/bin/bash")
    print("# Auto-generated by scripts/print_revision_commands.py — run on the SERVER node (Node A).")
    print(f"# power_config: {power_config}")
    print("set -e")
    for block, runs in runs_by_block.items():
        print(f"\necho '=== Block: {block} ({len(runs)} runs) ==='")
        if block in BLOCK_NOTES:
            print(f"# {BLOCK_NOTES[block]}")
        for run in runs:
            status = emit_status(run, skip_existing, seen)
            if status == "dup":
                print(f"echo '[DUP]  {run.output_dir} already listed in an earlier block'")
                continue
            if status == "skip":
                print(f"echo '[SKIP] {run.output_dir} already exists'")
                continue
            printed += 1
            print(f"\necho '--- {run.output_dir} ---'")
            for ref, rounds in run.identity:
                print(f"echo '>>> IDENTITY GATE: {ref} rounds {','.join(map(str, rounds))}'")
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
                for cmd in run.post_commands:
                    print(cmd)
                print(f"echo '[DONE] {run.output_dir}'")
    return printed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG, help="path to experiment_matrix.yaml")
    p.add_argument("--block", default=None, choices=BLOCK_ORDER + EXTRA_BLOCKS,
                   help="only print this block (%s: only with --block)" % ", ".join(EXTRA_BLOCKS))
    p.add_argument("--format", choices=["text", "bash"], default="text")
    p.add_argument("--skip_existing", dest="skip_existing", action="store_true", default=True,
                   help="skip runs whose results/<run>/results.json already exists (default: on)")
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false",
                   help="print every run regardless of whether results already exist")
    p.add_argument("--all_seeds", action="store_true",
                   help="seed_extension: emit all 5 seeds instead of only the new ones "
                        "(use after re-splitting the data, e.g. with --group_file)")
    p.add_argument("--baseline_root", default=None, metavar="DIR",
                   help="write centralized / local-only runs under DIR instead of results/ "
                        "(e.g. results/rev_baselines_sel for the re-run under the selection "
                        "rule, so the earlier rev_* baselines are not overwritten)")
    p.add_argument("--power_config", choices=POWER_CONFIGS, default=None,
                   help="Jetson power modes of the block: federated runs of any configuration "
                        "other than '%s' go to results/pc_<name>/ (default: the block's "
                        "declared configuration, else %s)"
                        % (DEFAULT_POWER_CONFIG, DEFAULT_POWER_CONFIG))
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    revision_cfg = cfg["revision"]
    if args.all_seeds:
        revision_cfg = apply_all_seeds(revision_cfg)
    runs_by_block = expand_all(revision_cfg, args.block)
    if args.power_config is not None:
        for name in list(runs_by_block):
            declared = block_power_config(revision_cfg, name)
            if declared and declared != args.power_config:
                if args.block:
                    raise SystemExit(
                        f"block {name!r} declares power_config {declared!r}; it cannot be "
                        f"emitted for {args.power_config!r}")
                print(f"# skipping block {name!r}: it declares power_config {declared!r}",
                      file=__import__("sys").stderr)
                del runs_by_block[name]
    effective = (args.power_config
                 or (block_power_config(revision_cfg, args.block) if args.block else None)
                 or DEFAULT_POWER_CONFIG)
    if args.baseline_root:
        rebase_baselines(runs_by_block, args.baseline_root)
    rebase_power_config(runs_by_block, effective)
    check_declared_namespaces(revision_cfg, runs_by_block)

    if args.format == "bash":
        n = print_bash(runs_by_block, args.skip_existing, effective)
    else:
        n = print_text(runs_by_block, args.skip_existing)

    total = sum(len(v) for v in runs_by_block.values())
    print(f"\n# {n}/{total} run(s) printed"
          f"{' (skipping ones with existing results.json)' if args.skip_existing else ''}.",
          file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
