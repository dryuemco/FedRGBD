# A5 -- additional Dirichlet partition draws (pre-registration)

**Declared 2026-10-07, about 02:00, before any new partition is drawn and before any A5
run exists.** Approved by the author. This file is committed and pushed before the new
partitions are generated. Nothing below changes after the first A5 result exists; a
change before that is a dated amendment in this file.

**Before this file was written,** the only thing done was a reproducibility check. On
the desktop, `data_splitter.py --seed 42` reproduced the three committed Dirichlet
manifests byte for byte (md5 equal to `analysis/leakage/P0_SUMMARY.md`), in a scratch
directory. That check produces no new data.

## 1. Why

Each Dirichlet concentration of the main study is one draw: partition seed 42
(`data/splits/dirichlet_{0.1,0.5,1}`). Every Dirichlet result is therefore conditional on
that draw (paper, Limitations). A5 describes how much the results move between draws.

## 2. Partitions

- α ∈ {0.1, 0.5, 1.0}. Three draws per α: the existing one (partition seed 42,
  `dirichlet_<α>`, unchanged) and two new ones, partition seeds **123** and **456**.
- The new draws are generated **once, on the desktop**, with the command that made the
  existing ones. Only `--seed` and a name suffix change:

  ```bash
  python src/data/data_splitter.py --data_dir data/raw/flame_dataset --output_dir <dir> \
      --nodes 3 --seed <123|456> --group_file analysis/leakage/groups.json \
      --skip_base_splits --dirichlet_alpha 0.1 0.5 1 --dirichlet_min_size 200 \
      --dirichlet_name_suffix _ps<seed> --link_mode hardlink
  ```

  The splits are named `dirichlet_<α>_ps<seed>`. `--dirichlet_name_suffix` is a new flag
  whose default (empty) leaves every existing name and the pinned default code path
  unchanged (CLAUDE.md rule 6).
- These are the same group-level, leakage-safe partitions as the main study: whole
  near-duplicate groups go to one node and one of train/val/test. The partition seed
  sets both the node assignment and the train/val/test assignment, as it did for seed 42.
- Their manifests are exported, committed to `data/splits/` (new files only; existing
  manifests untouched) and replayed on the nodes with `--from_manifest`, never
  re-derived (rule 2). Their md5s are added to the digest table of `P0_SUMMARY.md`, and
  the run pre-flight checks them on all three nodes.

## 3. Runs

### 3a. Testbed (primary for any A5 statement)

- **α = 0.1 only.** The two new draws (ps123, ps456) × {FedAvg, FedProx(μ = 0.01)} =
  **4 runs**.
- Same configuration as block 5b (`maxn_long_horizon`): all three nodes at MAXN_SUPER
  (`--power_config maxn`, `results/pc_maxn/`), 10 rounds, 5 local epochs, lr 1e-3,
  batch 8, and its determinism gate. One fixed training seed, **42**.
- **Order.** FedAvg before FedProx; within a strategy, ps123 before ps456.
- **Scheduling.** A run is started only if it is expected, from the 5b round times, to
  end before the night's stop time (2026-10-07: 07:30). The rest wait for later nights.
  When the camera experiment needs the testbed, it has priority. Scheduling never
  depends on any result.
- About 3.2 h per FedAvg run and 5.2 h per FedProx run, so 1 run is expected tonight.
- The seed-42 α = 0.1 partition is not re-run on the testbed under A5. Its testbed
  results are the main study's, which ran in another configuration: heterogeneous
  power, 3 rounds, 3 seeds.

### 3b. Desktop simulation sensitivity analysis (secondary)

- All 3 α × 3 draws (ps42, ps123, ps456) × {FedAvg, FedProx(0.01)} = **18 cells**. Same
  code, same hyperparameters (10 rounds, 5 local epochs, lr 1e-3, batch 8) and the same
  training seed 42, run as a sequential FL simulation on the desktop RTX 5090.
- Labelled **"desktop simulation sensitivity analysis"** everywhere. It is not expected
  to be bitwise identical to the testbed. No main claim rests on it; the main claims
  come from the testbed.
- It runs only once a desktop FL simulation path exists in the repository. None existed
  when this file was written.

## 4. Reporting (identical for 3a and 3b, reported separately)

- **Descriptive only.** For each α × strategy, per draw:
  - balanced accuracy, pooled (primary) and client mean (secondary), each with the
    paper's sequence-level stratified bootstrap interval;
  - across the draws, the range and SD.
- **Measured label skew of each draw**, with the measures of `scripts/partition_skew.py`:
  node-size-weighted mean JSD, unweighted mean JSD, maximum JSD, TV, per-node training
  fire fraction and per-node training size.
- **No new test, no verdict.** A5 joins none of the existing Holm families: neither the
  global-evaluation families (m = 18 heterogeneous, m = 4 maxn,
  `docs/GLOBAL_EVALUATION.md`) nor the cross-configuration families
  (`docs/CROSS_CONFIG_COMPARISON.md`).
- **Model selection and timing:** the declared rules (CLAUDE.md, "Model selection" and
  rule 12).
- A5 results are never pooled with the main study's Dirichlet results, and testbed and
  desktop results are never pooled with each other.
