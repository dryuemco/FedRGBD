# Wall-clock ratio FedProx(0.01) / FedAvg

Declared in docs/STRATEGY_TIMING_RATIO.md (2026-10-13, after the per-strategy times had been seen). Descriptive only: no verdicts, no tests. Ratio = geometric mean over the paired seeds of r_s = T_FedProx,s / T_FedAvg,s; interval = seed bootstrap, 95 % percentile, **indicative** (n paired seeds give at most C(2n-1, n) distinct resamples: 126 for five). Per-seed ratios in seed order. The last quantity is derived (R x the total ratio).

## heterogeneous

| partition | R | quantity | ratio | 95 % CI (indicative) | seeds | per-seed ratios |
|---|---|---|---|---|---|---|
| iid | 3 | steady | 1.548 | [1.541, 1.556] | 42 123 456 789 1011 | 1.5594 1.5432 1.5428 1.5371 1.5578 |
| iid | 3 | round1 | 1.532 | [1.513, 1.548] | 42 123 456 789 1011 | 1.4982 1.5328 1.5384 1.5319 1.5590 |
| iid | 3 | total | 1.543 | [1.537, 1.551] | 42 123 456 789 1011 | 1.5383 1.5397 1.5413 1.5353 1.5582 |
| iid | 3 | prox_round1_vs_avg_steady | 1.559 | [1.550, 1.568] | 42 123 456 789 1011 | 1.5698 1.5514 1.5532 1.5468 1.5729 |
| non_iid_label | 3 | steady | 1.546 | [1.537, 1.556] | 42 123 456 789 1011 | 1.5632 1.5539 1.5398 1.5355 1.5376 |
| non_iid_label | 3 | round1 | 1.535 | [1.527, 1.543] | 42 123 456 789 1011 | 1.5215 1.5479 1.5375 1.5354 1.5314 |
| non_iid_label | 3 | total | 1.542 | [1.536, 1.548] | 42 123 456 789 1011 | 1.5490 1.5519 1.5390 1.5355 1.5355 |
| non_iid_label | 3 | prox_round1_vs_avg_steady | 1.555 | [1.546, 1.565] | 42 123 456 789 1011 | 1.5728 1.5627 1.5512 1.5433 1.5464 |
| iid | 3 | fedavg_rounds_in_fedprox_budget | 4.628 | [4.611, 4.652] | 42 123 456 789 1011 | -- |
| non_iid_label | 3 | fedavg_rounds_in_fedprox_budget | 4.626 | [4.609, 4.644] | 42 123 456 789 1011 | -- |

## maxn

| partition | R | quantity | ratio | 95 % CI (indicative) | seeds | per-seed ratios |
|---|---|---|---|---|---|---|
| iid | 10 | steady | 1.618 | [1.592, 1.641] | 42 123 456 789 1011 | 1.6170 1.5698 1.6103 1.6439 1.6501 |
| iid | 10 | round1 | 1.541 | [1.465, 1.621] | 42 123 456 789 1011 | 1.4385 1.5822 1.6070 1.4358 1.6560 |
| iid | 10 | total | 1.609 | [1.587, 1.631] | 42 123 456 789 1011 | 1.5966 1.5732 1.6087 1.6191 1.6465 |
| iid | 10 | prox_round1_vs_avg_steady | 1.644 | [1.619, 1.670] | 42 123 456 789 1011 | 1.6370 1.5982 1.6309 1.6697 1.6836 |
| non_iid_label | 10 | steady | 1.641 | [1.631, 1.655] | 42 123 456 789 1011 | 1.6364 1.6362 1.6684 1.6374 1.6266 |
| non_iid_label | 10 | round1 | 1.620 | [1.532, 1.694] | 42 123 456 789 1011 | 1.4633 1.6233 1.6597 1.7421 1.6263 |
| non_iid_label | 10 | total | 1.639 | [1.624, 1.655] | 42 123 456 789 1011 | 1.6153 1.6349 1.6686 1.6502 1.6269 |
| non_iid_label | 10 | prox_round1_vs_avg_steady | 1.679 | [1.648, 1.730] | 42 123 456 789 1011 | 1.6512 1.6449 1.6750 1.7791 1.6492 |
| iid | 10 | fedavg_rounds_in_fedprox_budget | 16.086 | [15.873, 16.309] | 42 123 456 789 1011 | -- |
| non_iid_label | 10 | fedavg_rounds_in_fedprox_budget | 16.391 | [16.238, 16.550] | 42 123 456 789 1011 | -- |

