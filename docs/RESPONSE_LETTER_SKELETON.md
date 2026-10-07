# Response letter -- skeleton (NCAA-D-26-02211)

Drafted 2026-09-28 while block 5b runs. One row per reviewer comment:
what was done -> where the manuscript answers it -> status. The full per-comment
answers are in `docs/RESPONSE_TO_REVIEWERS.md` (written before the federated matrix
ran; its "PENDING EXPERIMENT" labels are superseded by the status column here).
No number appears in this file; every number in the letter will be quoted from the
manuscript tables, which come from `analysis/` (CLAUDE.md rule 4).

**Status**

* **done** -- the work and the manuscript text are complete.
* **done, text pending** -- every experiment has run and the table is generated; the
  results-dependent sentences are still `\todo` for the narrative stage.
* **pending 5b** -- waits for block `maxn_long_horizon` (all nodes MAXN_SUPER, running
  since 2026-09-28).
* **pending camera experiment** -- waits for the leave-one-scene-out captures and the
  author's scene labels.

## Reviewer 1

| # | Comment (short) | What was done | Where answered | Status |
|---|---|---|---|---|
| R1.1 | References before Eqs. 1-5b | Sources cited for each objective and update rule | Sec. `sec:objectives`, Eqs. `eq:global`-`eq:fedbn_b` | done |
| R1.2 | Cite 2025-26 references | New related-work subsection on recent non-IID FL benchmarks | Sec. `sec:related` (2025-2026 benchmarking) | done |
| R1.3 | Future work: hybrid method with wavelets for GPU edge clusters | Stated as a concrete future direction | Sec. `sec:future` | done |

## Reviewer 3

| # | Comment (short) | What was done | Where answered | Status |
|---|---|---|---|---|
| R3.1 | Sensors not connected to the FL experiment | Two experiments separated explicitly (FL on FLAME RGB; cross-sensor study) | Sec. `sec:sensorlink` | done |
| R3.2 | Random split leaks near-duplicate frames; use a sequence-level split | Near-duplicate audit; sequence-level re-split of all 15 partitions; all baselines repeated; 144 federated runs under the group-level split (114 in the main matrix, 30 in the MAXN_SUPER block; counted from `analysis/runs.csv`, kind fl, protocol group); pre-registered clean subset; nearest-neighbour distances | Sec. `sec:leakage`, `sec:clean`, `sec:res_leakage`; Tabs. `tab:leakage`, `tab:group_counts`, `tab:protocol_effect`, `tab:nn_distance` | done, text pending |
| R3.3 | Cross-sensor split mixes scenes; use leave-one-scene-out | The v1 cross-camera result is removed from the paper, together with every other v1 camera finding (the intrinsics table and the scene-count statements), because its random frame-level split over shared scenes cannot rule out scene overlap. It is replaced by a new, pre-registered camera experiment: leave-one-scene-out with a within-camera reference, plus sensor-skewed federated training. The pre-registration, its amendments and an erratum (section 14: the v1 captures were still on the nodes when the pre-registration said they were not) are dated in the public repository, `docs/CAMERA_EXPERIMENT_PREREG.md`. The v1 scripts are kept unchanged in `legacy/v1_cross_sensor/`. | Sec. `sec:loso`, `sec:res_loso`; Tab. `tab:loso`; `docs/CAMERA_EXPERIMENT_PREREG.md` (committed sections only) | v1 removed (2026-10-07); pending camera experiment |
| R3.4 | Task near saturation; add harder conditions | Dirichlet label skew (three concentrations) and a low-data regime (two subsampling rates), five seeds for the federated low-data cells (three for their references) | Sec. `sec:noniid`, `sec:lowdata`, `sec:res_dirichlet`, `sec:res_lowdata`; Tabs. `tab:dirichlet`, `tab:lowdata`; Fig. `fig:dirichlet` | done, text pending |
| R3.5 | Balanced accuracy, macro-F1, sensitivity, specificity, per-client confusion matrices | Full metric set, per client and pooled; balanced accuracy declared primary; both aggregations; sequence-level stratified bootstrap intervals | Sec. `sec:metrics`, `sec:primary`, `sec:stats`, `sec:res_metrics`; Tabs. `tab:fullmetrics`, `tab:perclient_metrics`; Fig. `fig:confmat` | done, text pending (`tab:perclient_metrics` still has placeholders) |
| R3.6 | FedBN conclusions limited by three rounds | Untested claim deleted; ten-round FedBN with a matched FedAvg reference (main matrix); ten-round FedBN at MAXN_SUPER | Sec. `sec:res_fedbn`, `sec:powercfg`; Tab. `tab:fedbn10` | done (main matrix), pending 5b (MAXN_SUPER) |
| R3.7 | Compare FedProx by elapsed time and communication too | Per-round test-free wall-clock and cumulative volume recorded for every run; three-panel figure; power modes stated | Sec. `sec:timecomm`, `sec:testbed`, `sec:res_time`; Tab. `tab:time`; Fig. `fig:timecomm` | done, text pending; pending 5b (MAXN_SUPER timing) |
| R3.8 | Three seeds too few; large effect sizes; more seeds for FedAvg-FedProx | Five seeds for the central cells; "statistically indistinguishable" removed; effect sizes caveated; intervals of differences | Sec. `sec:stats`, `sec:res_stats`; Tab. `tab:stats` | done, text pending |

## Reviewer 5

| # | Comment (short) | What was done | Where answered | Status |
|---|---|---|---|---|
| R5.1 | Narrow scope; which conclusions generalise | Scope subsection; generalisation discussion; limitations | Sec. `sec:scope`, `sec:generalise`, `sec:limitations` | done, text pending (generalisation paragraph depends on results) |
| R5.2 | More runs and confidence intervals | Five seeds; balanced accuracy (primary) with the stratified sequence-level bootstrap interval; the other held-out metrics as mean ± SD over seeds | Sec. `sec:stats`; every results table | done, text pending |
| R5.3 | Precision, recall, F1, BA, MCC, ROC-AUC, globally and per client | Full metric set per client and pooled; second, pre-registered global evaluation perspective (every model on the union of the test splits) | Sec. `sec:metrics`, `sec:primary`, `sec:perspectives`; Tabs. `tab:fullmetrics`, `tab:perclient_metrics` | done, text pending (global perspective: main matrix computed 2026-09-28; MAXN_SUPER family pending 5b) |
| R5.4 | Check other degrees and types of non-IID | Dirichlet label skew at three concentrations besides the manual skew | Sec. `sec:noniid`, `sec:res_dirichlet`; Tab. `tab:dirichlet`; Fig. `fig:dirichlet` | done, text pending |
| R5.5 | Hyperparameter sensitivity | One-factor sweeps of mu, local epochs, learning rate, rounds | Sec. `sec:sensitivity`, `sec:res_sensitivity`; Tab. `tab:sensitivity`; Fig. `fig:sensitivity` | done, text pending (`tab:sensitivity` still has placeholders) |
| R5.6 | Acknowledge metaheuristic hyperparameter tuning | Related-work subsection | Sec. `sec:related_meta` | done |
| R5.7 | Change the title | Title changed to the suggested one | Title | done |
| R5.8 | Moderate FedBN conclusions | Conclusions restricted to the tested budgets; ten-round results added | Sec. `sec:res_fedbn`, `sec:discussion` | done (main matrix), pending 5b (MAXN_SUPER) |

## Changes not requested by a reviewer (to state in the letter)

| Change | Where | Status |
|---|---|---|
| Declared model-selection rule (lowest aggregated validation loss; test evaluated every round, never used for selection) | Sec. `sec:selection` | done |
| Balanced accuracy declared primary (fixed before any federated run); pooled vs client-mean aggregation (fixed after the runs; both reported) | Sec. `sec:primary` | done, text pending (flipping comparisons) |
| Wired Gigabit Ethernet instead of WiFi; v1 timings superseded | Sec. `sec:testbed` | done |
| Per-node power modes of the main matrix not harmonised; MAXN_SUPER block and declared cross-configuration analysis | Sec. `sec:testbed`, `sec:powercfg` | done (method), pending 5b (results) |
| Reproducibility: fixed aggregation order; bitwise reproducible within a power configuration, not across | Sec. `sec:stats` (Reproducibility) | done |

## Still to do outside the experiments

* Author-only: funding wording, testbed photo (Fig. 1), scene labels for the cross-sensor
  captures.
* Narrative stage: every `\todo` in `paper/main.tex`, including the per-node power-mode
  consequences and the aggregation flips (`analysis/aggregation_flips.md`).
* Replace section/table labels in this skeleton by the compiled numbers when the letter is
  written.
