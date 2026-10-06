# Scope audit, 2026-10-07 (read-only)

A read-only agent produced this audit on 2026-10-07, between about 01:50 and 02:20. It
checked `paper/main.tex` at commit `c114be0` or later, `docs/RESPONSE_LETTER_SKELETON.md`,
`results/`, `analysis/`, `paper/tables/` and `data/splits/`. It did not open any `*.eml`.
The findings are listed as the agent reported them. They have **not** been re-verified
row by row, and nothing has been changed because of them.

**Coverage.** Every reviewer row of the skeleton was checked: R1.1-R1.3 (3), R3.1-R3.8
(8) and R5.1-R5.8 (8), 19 rows in total. The numbering has no gaps.

**Run count.** All 144 revision FL runs are complete. They split into 114 in `results/rev_*`
(the planned main-matrix grid, duplicates removed) and 30 in `results/pc_maxn/rev_*`. The
"98 federated runs" in CLAUDE.md and in the skeleton dates from before low_data went to
5 seeds.

## 1. Reviewer table

**(a) Labels.** Every cited label exists. `tab:nn_distance` lives in
`paper/tables/nn_distance.tex` and is pulled in through `\IfFileExists`.

| Row | Finding |
|---|---|
| R1.1 | OK. Equations and citations present (main.tex:268-300). |
| R1.2 | OK (main.tex:124). |
| R1.3 | OK (main.tex:1013-1017). |
| R3.1 | MISMATCH. `sec:sensorlink` (main.tex:200-202) calls Experiment II "centralised". The prereg and R3.3 also include sensor-skewed *federated* training (question b), which main.tex never mentions. |
| R3.2 | MISMATCH. The federated cells of `tab:protocol_effect` are still `\PH` although the runs exist. "98 federated runs" is out of date. "All … runs repeated" is not true for v1 FedBN at 3 rounds or for v1 FedProx 0.1 IID. |
| R3.3 | OK. Placeholders, consistent with "pending camera experiment". |
| R3.4 | MISMATCH (status). Seed claims are correct. But the federated rows of `tab:dirichlet` and `tab:lowdata` are `\PHs`, and `fig:dirichlet` is only a `\todo`: more than "text pending". |
| R3.5 | MISMATCH. The federated rows of `tab:fullmetrics` are placeholders too, not only `tab:perclient_metrics`. `fig:confmat` is a `\todo` only: no figure file and no plotting code. |
| R3.6 | MISMATCH. `tab:fedbn10` is all `\PHs`. Its 3-round column cannot be filled, because no 3-round FedBN run exists in the revision. Its IID rows have no main-matrix 10-round run; IID 10-round runs exist only in pc_maxn, which is never pooled with the main matrix. "Pending 5b" is out of date (5b committed 7cb0f9a, analysed d80665f). |
| R3.7 | MISMATCH (partial). `tab:time` is filled. `figures/fig_timecomm.pdf` does not exist, so the figure renders the `\todo` box. There is no FedBN row, and none is possible. "Pending 5b" is out of date. |
| R3.8 | MISMATCH. `tab:stats` is still the v1 table (image-level split, n=3, no CI of the difference). `tables/pairwise_tests.tex` exists but is never input. |
| R5.1 | OK (main.tex:73, 981, 992). |
| R5.2 | MISMATCH. The row claims a bootstrap interval for every held-out metric. Only balanced accuracy carries one; MCC, Acc, Sens, Spec, macro-F1 and ROC-AUC are mean ± std over seeds. |
| R5.3 | MISMATCH. (1) Precision and binary F1 are claimed (main.tex:92) but appear in no paper table, although they exist in `analysis/`. (2) Per-client coverage is only `tab:perclient_metrics`: label skew only, 4 metrics, all placeholders. (3) The global-perspective results are in `analysis/global_eval` but not in the Results section. (4) "MAXN family pending 5b" is out of date. |
| R5.4 | MISMATCH (status). Same as R3.4. |
| R5.5 | MISMATCH. See section 2. |
| R5.6 | OK (main.tex:131). |
| R5.7 | OK (not checked in detail). |
| R5.8 | MISMATCH. "Ten-round results added" is not true in the paper: `tab:fedbn10` is all placeholders. "Pending 5b" is out of date. |
| Changes not requested: power configurations | MISMATCH (status). "Pending 5b (results)": the results exist and are committed. |

**Other paper-internal points.**
- main.tex:435 and 994 say the ten-round runs use three seeds. The MAXN ten-round block has 5.
- `fig:confmat`, `fig:dirichlet` and `fig:sensitivity` are never cited with `\ref`.

## 2. Hyperparameter sensitivity runs

| Item | Finding |
|---|---|
| μ grid {0.001, 0.01, 0.05, 0.1, 0.5} | All 15 cells complete. μ=0.01 has 5 seeds (shared with seed_extension), the others 3. |
| E ∈ {1, 2, 5} × {FedAvg, FedProx 0.01} | Complete. `client_config` confirms local_epochs 1 and 2 on every node. |
| η ∈ {1e-4, 1e-3} × {FedAvg, FedProx 0.01} | Complete. `client_config` confirms lr = 1e-4. |
| R ∈ {3, 10}, FedBN + matched FedAvg (main.tex:390) | MISSING. No 3-round FedBN run exists under the group split; it was never planned. FedBN exists only at R=10. |
| Extra or unplanned runs | None. |
| Analysis output | Every sweep config is in `analysis/summary_table.csv`. No sensitivity table or figure generator exists. |
| `tab:sensitivity` | All `\PHs`. One value column although E and η were swept for two strategies. No R row. |
| `fig:sensitivity`, `fig:mu_tradeoff` | `fig:sensitivity` is a `\todo` only. `fig:mu_tradeoff` is still the v1 figure. |
| Note | `tables/time.tex` already prints label-skew FedProx BA for all five μ values. |

## 3. MCC / ROC-AUC / confusion matrices

| Item | Finding |
|---|---|
| eq:mcc, eq:auc | Defined (main.tex:418-428). |
| Confusion matrices | Stored per round in results.json (`metrics_distributed`). The paper says they are "reported for the non-IID conditions"; they are not. |
| MCC / ROC-AUC, pooled and client mean | Present in `analysis/summary_table.csv` (`selected_test_mcc`, `selected_test_roc_auc`, `selected_test_clientmean_*`, clean variants). |
| Per-client values | `analysis/per_client_selected.csv` has them for every configuration. |
| `fig:confmat` | MISSING. No figure file and no code. Its `\todo` says "produced by scripts/analyze_results.py", which it is not. |
| `tab:fullmetrics` | Reference rows filled. FL rows are `\PHs` by design of `export_latex_tables.py`. FedBN rows cannot be filled (no 3-round FedBN). No precision or binary-F1 columns. |
| `tab:perclient_metrics` | All `\PHs`. The FedBN block has no 3-round runs to fill it from. |

## 4. Dirichlet draws

| Item | Finding |
|---|---|
| Partitions per α | One each (`data/splits/dirichlet_{0.1,0.5,1}`; the `_sub*` variants derive from the same draw). |
| Partition seed | 42 for all. The README states `--dirichlet_min_size 200`; `split_stats.json` does not record the min size. |
| Training seeds | 3 per configuration (42, 123, 456); no FedBN. Matches the paper. |
| "Single draw" wording | OK (main.tex:249, 774, 782). |
| Minor wording that suggests an α axis | main.tex:787 (first `\todo`), 1008 ("varied … by Dirichlet concentration"), 1010 ("stable across α"), 247 ("seed of the draw fixed per experiment": it is one global seed, 42). |

## Inconsistencies, most important first (as reported)

1. **FedBN at 3 rounds was never run under the revision protocol**, yet four tables have 3-round FedBN cells: `tab:fullmetrics`, `tab:protocol_effect`, `tab:perclient_metrics` and `tab:fedbn10`. main.tex:390 also claims R ∈ {3, 10}.
2. **`tab:fedbn10` IID rows** have no main-matrix ten-round run (MAXN only, never pooled). This conflicts with main.tex:994.
3. **Skeleton statuses say "done" where the cited tables or figures are still placeholders**, although the runs exist: R3.2, R3.4/R5.4, R3.5, R3.6/R5.8, R3.7, R3.8, R5.5.
4. **"Pending 5b" is out of date** in R3.6, R3.7, R5.3, R5.8 and the power-configuration row.
5. **Precision and F1 are claimed but not shown**; the global-perspective results are absent from Results.
6. **R5.2 claims bootstrap CIs for every metric**; only BA has them.
7. **R3.1 vs `sec:sensorlink`**: Experiment II is described as centralised only.
8. **"98 federated runs"** is out of date (114 + 30 = 144).
9. **Seed count for ten-round runs**: the paper says 3; the MAXN ten-round block has 5.
10. **`tab:sensitivity` structure** does not match the sweep, and no exporter exists.
11. **Minor Dirichlet wording** that implies an α axis.

## Addendum: second report of the same agent (after the scope extension)

The same agent reported a second time, after the instruction to cover every row. It
again found 19 rows with continuous numbering. It notes that it could not check the
rows against the decision letter itself (`*.eml` not opened). New or sharper points
relative to the report above:

- **R3.3.** The skeleton row cited "prereg sections 1-14", but section 13
  (Amendment 4) is an uncommitted draft. *Corrected the same night:* the row now cites
  the prereg's committed sections only.
- **R3.6 / R5.8.** main.tex:848 still argues from the v1 three-round FedBN number.
- **R3.8.** Generated `tables/pairwise_tests.tex` and `tables/friedman.tex` exist but are
  never `\input`.
- **R5.3.** The global-perspective results have no table and no `\todo` in Results.
- **R5.7.** Whether the title is the reviewer's suggestion needs the decision letter (not
  opened).
- **Power configurations row.** The `sec:powercfg` results are still a `\todo`
  (main.tex:379).
- **Unused generated tables.** Nine `paper/tables/full_metrics_*.tex` files are generated
  but not `\input`.
- **Missing figures.** `fig:confmat`, `fig:sensitivity`, `fig:dirichlet` and `fig:timecomm`
  have no files in `paper/figures/`. No generator exists for confmat or sensitivity.
- **μ = 0.01.** This cell has 5 seeds (shared with seed_extension); the other μ points
  have 3.
- **Dirichlet min size.** `split_stats.json` does not record it; only the README says 200.
