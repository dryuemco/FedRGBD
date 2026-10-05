# Cross-configuration comparison: MAXN_SUPER - heterogeneous

Declared in docs/CROSS_CONFIG_COMPARISON.md. Balanced accuracy pooled over the union of the clients' test splits at the selected round (rounds_1_3: among rounds 1-3; ten_rounds: among rounds 1-10; lowest aggregated validation loss); seed-paired difference in percentage points, 95 % stratified cluster-bootstrap interval (B = 10000); Holm within each family.

## rounds_1_3

| partition | strategy | seeds | diff | 95 % CI | p | p Holm | verdict | verdict (Holm) |
|---|---|---|---|---|---|---|---|---|
| iid | FedAvg | 42 123 456 789 1011 | +2.4 | [-3.6, +8.2] | 0.3822 | 1.0000 | no detectable difference | no detectable difference |
| iid | FedProx(0.01) | 42 123 456 789 1011 | +0.7 | [-1.4, +2.9] | 0.4000 | 1.0000 | no detectable difference | no detectable difference |
| non_iid_label | FedAvg | 42 123 456 789 1011 | +2.0 | [-1.8, +8.8] | 0.3330 | 1.0000 | no detectable difference | no detectable difference |
| non_iid_label | FedProx(0.01) | 42 123 456 789 1011 | +0.7 | [-0.6, +2.9] | 0.4720 | 1.0000 | no detectable difference | no detectable difference |

## ten_rounds

| partition | strategy | seeds | diff | 95 % CI | p | p Holm | verdict | verdict (Holm) |
|---|---|---|---|---|---|---|---|---|
| non_iid_label | FedAvg | 42 123 456 | -1.4 | [-4.6, +0.2] | 0.0968 | 0.1936 | no detectable difference | no detectable difference |
| non_iid_label | FedBN | 42 123 456 | +0.8 | [-3.5, +6.0] | 0.5985 | 0.5985 | no detectable difference | no detectable difference |
