# Global evaluation -- pre-registered second perspective

Declared 2026-09-28 (decision 2 of that day). The declaration -- (a) this definition,
(b) the code, (c) the interpretation rule, and their tests -- is committed as one commit
and pushed before any local-only global value is aggregated or compared; the numbers
follow in a later commit. What the paper may say about timing is exactly this: the
per-node cross-evaluation values of the local-only models were logged automatically
during baseline training on 2026-09-19/20; the federated and centralized global values
were already known from the pooled analysis (rule 8); the declaration precedes any
aggregation or comparison of local-only global values (details under *Provenance*). It
does not say that the perspective was declared before its numbers existed. Nothing in
this file may be changed after the numbers are computed; a change has to be disclosed in
the paper and the response letter (as for CLAUDE.md rule 7).

## 1. Definition

**Two perspectives.** The pooled figure already declared as primary (CLAUDE.md rule 8)
evaluates every model on the held-out images of the clients it serves: the federated
global model on every client's test split, each local-only model on *its own node's* test
split only, the centralized model on the pooled test set. For local-only that is a
*personalised* evaluation -- each model is tested on data distributed like its own
training data, including its node's class skew. It stays primary.

The **global evaluation** adds a second perspective: every model on the **union of the
three nodes' test splits** of its partition, the same images for every method.

* **Union.** For partition P, U(P) = the test splits of node_a, node_b and node_c of P
  (`data/processed/<P>/node_*/test`, replayed from `data/splits/`). It is image for image
  the set the pooled figure already uses (`check_prediction_unions`); the code verifies
  this again for every model it evaluates.
* **Federated.** Each strategy's global model at its selected round (declared rule,
  `src/evaluation/model_selection.py`), on U(P). For FedAvg and FedProx this is exactly
  the pooled figure already computed: the server evaluates the global model on every
  client's test split, and those predictions are pooled.
* **Centralized.** The selected-epoch model on its pooled test set, which is U(P) -- as
  already computed.
* **Local-only.** Each node's selected-epoch model (`model_selected.pt` of the
  selection-rule baselines, `results/rev_baselines_sel/`) run on the whole of U(P). The
  local-only value of a seed is the **mean over the three node models** of their balanced
  accuracy on U(P); the three per-node values are reported alongside.
* **Metric.** Balanced accuracy (the declared primary metric), pooled over U(P). No other
  metric enters this perspective.
* **Uncertainty.** The stratified cluster bootstrap of `src/evaluation/bootstrap.py`:
  sequences are the resampling unit, resampled across the three nodes on U(P), stratified
  by class composition; B = 10,000; 95 % percentile intervals; generator keyed by the
  configuration, so every number is reproducible.
* **Robustness.** Everything is repeated on the clean subset of U(P) (the pre-registered
  lists of CLAUDE.md rule 7, `analysis/leakage/clean_subset/`). The full held-out set is
  the headline; the clean subset is the robustness check.

**Scope -- which comparisons exist.** Every partition that has a local-only reference:
IID, label skew, the three Dirichlet partitions and the four low-data partitions
(IID and label skew at rho = 0.05 and 0.01). Two families, never pooled, because power
configurations are never pooled (CLAUDE.md, "Running a block"):

| family | FL runs | partitions | strategies | comparisons |
|---|---|---|---|---|
| `heterogeneous` | `results/rev_*`, R = 3, E = 5, lr = 1e-3 | 9 | FedAvg, FedProx(mu = 0.01) | 18 |
| `maxn` | `results/pc_maxn/rev_*` (block `maxn_long_horizon`), R = 10, E = 5, lr = 1e-3 | IID, label skew | FedAvg, FedProx(mu = 0.01) | 4 |

The `maxn` family is declared now and computed when the block has finished and has been
committed; its local-only reference is the same selection-rule baseline (15 epochs,
desktop GPU) -- no ten-round-equivalent local-only baseline exists.

**Not in scope, and why.**

* **FedBN** (heterogeneous R = 10 on label skew, and the FedBN runs of `maxn`). FedBN has
  no global model: each client keeps its own normalisation parameters, and the server's
  per-client predictions are those of client k's own model on client k's split. Putting
  FedBN on U(P) would need each client's model, and the FL runs save no checkpoints. It is
  therefore not evaluated globally, and the paper says so.
* **The label-skew hyperparameter sweeps** (mu grid, E in {1, 2}, lr = 1e-4, and FedAvg at
  R = 10 heterogeneous). They are sensitivity analyses of one partition, not strategies;
  they are not computed in this perspective.

**Pairing.** The comparison FL - local-only is seed-paired: seed s of the FL
configuration against the local-only baseline trained with seed s. Only seeds present on
both sides enter the difference -- IID and label skew 5 (42, 123, 456, 789, 1011),
Dirichlet 3, low-data 3 (42, 123, 456: the low-data FL runs have seeds 789 and 1011 too,
the local-only baselines do not). Each configuration's own global value is reported over
all of its seeds.

**The paired difference and its interval.** For partition P, strategy f and paired seeds
s = 1..S:

    D = (1/S) sum_s [ BA(FL_f,s ; U) - (1/3) sum_k BA(local_k,s ; U) ]

One bootstrap replicate draws **one** stratified sequence resample of U(P), applied to
every model of both sides (the test set is shared, so resampling it separately per side
would double-count its variance), and **one** draw of S seed indices with replacement,
applied to both sides (the pairing is kept). The 95 % interval is the 2.5 / 97.5
percentile of the B replicate values of D. The two-sided bootstrap p-value is
p = min(1, 2 min(#{D* <= 0} + 1, #{D* >= 0} + 1) / (B + 1)).

**Provenance -- what existed before this declaration.**

* The federated and centralized values on U(P) are the pooled figures of rule 8, computed
  and reported since 2026-09-27; this perspective re-uses them unchanged.
* The local-only baselines were trained with `scripts/train_local.py --cross_eval`
  (2026-09-19/20), which evaluated each node's selected-epoch model on the other two
  nodes' test splits and logged the metrics and confusion matrices automatically in the
  committed `results/rev_baselines_sel/*_local_*/node_*/results.json` (`cross_eval`) and
  in the training logs (31 runs x 3 node models x 2 other nodes = 186 per-node values).
  From those confusion matrices each local model's balanced accuracy on U(P) is
  computable by arithmetic. They were never aggregated, tabulated, compared or read by
  any analysis: no script reads `cross_eval`.
* **Author statement:** the author does not recall inspecting the logged `cross_eval`
  values and has no record of doing so; they were never aggregated or compared before
  this declaration.
* **Assistant transcripts** (all Claude Code sessions of the revision, searched
  2026-09-28 for the key, for the training-log "Cross-node evaluation" lines, and for
  numeric values next to either): no report or message to the author ever contained a
  `cross_eval` value or a summary of one. Raw tool output did show 4 of the 186 per-node
  values while the assistant inspected the file format -- the `node_b` model of
  `rev_noniid_sub0.05_local_seed42` on `node_a` and `node_c` (full metric blocks,
  2026-09-28 04:16 +03:00), and the `node_a` model of `rev_dirichlet0.1_local_seed123` on
  `node_b` and `node_c` (2026-09-28 ~08:53 +03:00, the session that made this commit).
  None was aggregated, compared with anything, or quoted in a report. An earlier draft of
  this section said the transcripts contained no value; that was wrong and is corrected
  here.
* No global-evaluation statistic (a local-only mean on U(P), a difference, an interval,
  a p-value, a verdict) had been computed by anyone when this declaration was committed.

**Reproduction check.** The code recomputes each local model's cross-node predictions
from `model_selected.pt`; before any statistic is formed it checks that (i) its
own-node predictions reproduce `selected_<node>_test.npz` and (ii) its cross-node
metrics reproduce the logged `cross_eval` metrics (accuracy and balanced accuracy within
1e-6, as `scripts/predict_from_checkpoint.py` does). A failure stops the evaluation.

**Outputs.** `analysis/global_eval/` -- per-run values (every model, full and clean),
per-configuration means with intervals, and the comparison table.

## 2. Interpretation rule

Fixed before any local-only global value was aggregated or compared (see *Provenance*);
applied mechanically by
`scripts/global_evaluation.py` (`apply_rule`).

For every partition P and strategy f of a family, on the seed-paired difference
D = FL_f - local-only (section 1) with its 95 % cluster-bootstrap interval [L, U]:

| interval | verdict |
|---|---|
| L > 0 | "federation improves generalisation beyond the client's own distribution" |
| L <= 0 <= U | "no detectable difference" |
| U < 0 | "federation worse" |

**Multiplicity.** Holm correction across all partition x strategy comparisons of a family:
the `heterogeneous` family on the full held-out set is one correction family (m = 18);
the `maxn` family on the full set is another (m = 4); each family's clean-subset
comparisons form their own (m = 18 and 4), because the clean subset is a robustness
check of the same comparisons, not additional hypotheses. The p-value is the two-sided
bootstrap p of section 1. **Holm verdict:** adjusted p < 0.05 -> "improves" if D > 0,
"worse" if D < 0; otherwise "no detectable difference".

**Reporting.** Every comparison is reported -- difference, interval, unadjusted p, Holm
p, the interval verdict and the Holm verdict -- whatever it shows; none is dropped or
merged. The headline statement of a comparison is its Holm verdict on the full set; the
interval verdict and the clean subset are reported alongside, and a comparison whose
verdict differs between them is named as such. The verdict wording is fixed: the paper
uses the three phrases above and does not upgrade "no detectable difference" to
"equivalent" or "no effect" (the rule is not an equivalence test).
