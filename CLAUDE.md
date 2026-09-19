# FedRGBD — project instructions

Federated learning on a 3-node Jetson Orin Nano 8 GB cluster with heterogeneous RGB-D cameras.
Manuscript **NCAA-D-26-02211**, *Neural Computing and Applications*, **major revision due
2026-11-13**. All revision work happens on the branch `revision-ncaa`.

## Read these first

| File | What it is |
|---|---|
| `docs/REVISION_PLAN_TR.md` | The working plan (Turkish). §5 status, §5.1/§5.2 findings, **§6 the Jetson handoff: run order, time estimates, which table each block fills** |
| `docs/REVISION_CHANGES.md` | File-by-file record of every change made for the revision (English) |
| `docs/RESPONSE_TO_REVIEWERS.md` | Per-reviewer-comment answers; says what is DONE and what is PENDING EXPERIMENT |
| `docs/DESKTOP_GPU_BASELINES.md` | Desktop GPU environment and the baseline block that already ran |

## Environment

- **Jetson**: `./setup_jetson.sh` (JetPack 6.x, NVIDIA PyTorch wheel, Flower 1.13.1). Python 3.10
  syntax, `numpy==1.26.4`, `batch_size=8` — these are hardware constraints, do not "modernise" them.
- **Desktop (Windows, RTX 5090)**: venv `~/venvs/fedrgbd` (CPU, tests) and `~/venvs/fedrgbd-gpu`
  (CUDA 12.8, baselines). System Python and WSL have no torch.
- **Testbed network** (wired Gigabit Ethernet, static): node_a = `192.168.1.10` (also the Flower
  server, `:8080`), node_b = `192.168.1.7`, node_c = `192.168.1.6`.
- **FLAME layout**: the Kaggle archive unpacks as `Training/Training/{Fire,No_Fire}` and
  `Test/Test/{Fire,No_Fire}`, but the `data/splits` manifests expect a flat `{Fire,No_Fire}`
  layout. On a (re)built node, move all files into `data/raw/flame_dataset/Fire` and
  `data/raw/flame_dataset/No_Fire` (30155 + 17837) before running `--from_manifest`, otherwise the
  replay aborts with missing sources.
- **GUI off on all three nodes** (`sudo systemctl set-default multi-user.target`): the desktop
  session costs ~2.5 GB of the 8 GB, and the nodes must be comparable because per-round wall-clock
  is a reported result. As of 2026-09-19 all three nodes run with GUI off and have ~6.8 GB of the
  8 GB available, so they are comparable for timing.
- **v1 ran over WiFi (802.11ac), the revision over wired GbE.** v1 and revision wall-clock numbers
  are not comparable; revision timings supersede the v1 ones.
- Tests: `python -m pytest tests -q -k "not end_to_end"` (~250 CPU tests, seconds).
  On the Jetson nodes the ROS 2 pytest plugins break collection; use
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q -k "not end_to_end" -p no:cacheprovider`.

## Hard rules

1. **Never commit `*.eml`.** The reviewer/editor correspondence is confidential and this repo is public.
2. **Never re-derive the data partition.** `find_images()` depends on `os.walk` order, so two
   machines can produce *different* splits. The authoritative partition is committed in
   `data/splits/` and is replayed:
   ```bash
   python src/data/data_splitter.py --from_manifest data/splits \
       --data_dir data/raw/flame_dataset --output_dir data/processed \
       --link_mode hardlink --clean --verify
   ```
   Then check the manifest MD5s against the table in `analysis/leakage/P0_SUMMARY.md`; they must
   be identical on all three nodes.
3. **Always pass `--all_seeds`** to `scripts/print_revision_commands.py`. Runs from before the
   leakage-safe re-split used a different partition and are not comparable.
4. **Never type a number into the paper by hand.** Run `scripts/analyze_results.py` then
   `scripts/export_latex_tables.py`; the tables come from `analysis/`.
5. **Do not touch `results/3node_*`, `results/centralized_*`, `results/local_*`** — those are the
   v1 (image-level split) results kept as the published reference. New runs go to `results/rev_*`.
6. The default (no-flag) code path of `data_splitter.py` is pinned bit-identical to the original
   submission by a regression test. Changes must stay behind a flag.

## Why the protocol changed

A near-duplicate audit of FLAME found that **99.6 % of validation/test images had a near-duplicate
in some node's training split** under the original random image-level split: the dataset is video
frames, and 265 groups cover 99.7 % of the 47,992 images. The revision therefore assigns whole
sequences (groups) to one node and one of train/val/test. 22 groups contain both fire and no-fire
frames (42 % of the data) and are kept together regardless of label. Under the audited protocol the
references drop from ~99.6 % to 90.9 % (centralized IID) and 78.3 % (local-only IID), which is what
makes the reviewers' question — when does federation help — measurable at all.

## Current state (2026-09-17)

Everything that does not need the testbed is done: leakage audit, re-split, all 62
centralized/local-only baseline runs, bibliography, paper text, response letter.
**Remaining: 98 federated runs on the Jetsons (~220 h, see §6.2 of the plan)**, plus three
author-only items (funding wording, testbed photo, scene labels for the cross-sensor experiment).
Paper placeholders awaiting those runs are red `\todo{...}` / `\PHs` macros in `paper/main.tex`.

## Running a block

```bash
python scripts/print_revision_commands.py --all_seeds --block seed_extension --format bash > run.sh
# server on node A, clients on all three; finished runs are skipped automatically
bash run.sh
python scripts/analyze_results.py --results_dir results --output_dir analysis
python scripts/export_latex_tables.py --analysis_dir analysis --output_dir paper/tables
```

Block order and cost: `docs/REVISION_PLAN_TR.md` §6.2. After each block, update the matching
table listed in §6.3 and commit.
