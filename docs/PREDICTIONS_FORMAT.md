# Per-image predictions — on-disk format (v1)

This is the committed, static copy of the README that
`src/evaluation/predictions.py` writes into every `predictions/` directory. It is here so the
format is documented in git independently of whether any run has produced predictions yet.

**The generated copy is authoritative.** It is written from `README_TEXT` in
`src/evaluation/predictions.py`, so if that constant changes, this file is stale until someone
updates it. `tests/test_predictions.py` asserts the two agree.

---

## Files

One file per round, client and split:

| pattern | written by | when |
|---|---|---|
| `r<round>_<node>_<split>.npz` | `src/fl/server.py` | every round of a federated run |
| `selected_<node>_<split>.npz` | `scripts/predict_from_checkpoint.py` | from a baseline's `model_selected.pt` |

They live in `<run_dir>/predictions/`. Federated runs write them after the test pass, so packing
them does not enter the reported round time (the server's `pred_pack_time_s` is excluded along
with the test pass itself).

## Arrays

| array | dtype | meaning |
|---|---|---|
| `path` | str | manifest key `<Class>/<file>`; joins `data/splits/*.csv.gz`, `analysis/leakage/groups.csv` and `analysis/leakage/clean_subset/` |
| `label` | uint8 | true class, 1 = Fire |
| `logit_margin` | float32 | `z_fire - z_nofire` of the evaluated model |
| `p_fire` | float32 | `sigmoid(logit_margin)` |

`path` is the join key to everything else: it is what links a prediction to its near-duplicate
group (and therefore to the sequence-level bootstrap) and to the pre-registered clean-subset
exclusion lists. It is a manifest key, not a filesystem path, so it is stable across nodes.

## Reading them

```python
from src.evaluation.predictions import load_npz, metrics_from_predictions

d = load_npz("results/rev_.../predictions/selected_node_a_test.npz")
m = metrics_from_predictions(d["label"], d["logit_margin"])
```

## Regenerating them

**Baselines** — at any time, from the committed checkpoint:

```bash
python scripts/predict_from_checkpoint.py results/rev_baselines_sel/rev_iid_local_seed42
```

The script re-evaluates `model_selected.pt` at the run's own batch size and **fails if the
recomputed metrics do not reproduce the logged selected-epoch metrics**, so a silent mismatch
between a checkpoint and its `results.json` cannot pass unnoticed.

**Federated runs** — not regenerable. They exist only as the run wrote them; reproducing them
means re-running the block on the testbed. Archive `predictions/` with the run.

## What is and is not in git

The `.npz` files are gitignored (~54 MB for the full matrix), as is the generated
`predictions/README.md` — this file is its committed counterpart. What *is* committed is
everything needed to regenerate the baseline predictions: the checkpoints' `results.json`, the
partition manifests in `data/splits/`, and the hashes in `analysis/leakage/`.

## Why they exist

Per-image predictions are what make three things possible that round-level logging cannot:

- **sequence-level (cluster) bootstrap CIs**, which resample held-out *sequences* rather than
  images, because a few flights dominate a node's held-out set;
- the **pre-registered clean-subset** metrics, which need per-image joins to the exclusion
  lists (`analysis/leakage/clean_subset/`);
- recomputing any metric after the fact without re-running anything.

`scripts/analyze_results.py` switches a configuration's CIs from the t-interval over seeds to
the cluster bootstrap automatically, but only when **every** run of that configuration has
predictions — so a partially-populated configuration silently keeps the weaker method. After
regenerating predictions, check the `ci_method` column of `analysis/summary_table.csv`.
