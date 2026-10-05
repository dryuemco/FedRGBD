# Global evaluation: FL - local-only, balanced accuracy on the union

Pre-registered in docs/GLOBAL_EVALUATION.md. Difference in percentage points, seed-paired, 95 % cluster-bootstrap interval (B = 10000). Verdict: interval rule, unadjusted; Holm: adjusted p < 0.05 within the family.

## heterogeneous, full set (Holm m = 18)

| partition | strategy | seeds | diff | 95 % CI | p | p Holm | verdict | verdict (Holm) |
|---|---|---|---|---|---|---|---|---|
| iid | FedAvg | 42 123 456 789 1011 | -0.9 | [-13.2, +12.4] | 0.9497 | 1.0000 | no detectable difference | no detectable difference |
| iid | FedProx(0.01) | 42 123 456 789 1011 | +5.6 | [+2.2, +12.7] | 0.0012 | 0.0216 | federation improves generalisation beyond the client's own distribution | federation improves generalisation beyond the client's own distribution |
| non_iid_label | FedAvg | 42 123 456 789 1011 | -0.6 | [-6.5, +3.7] | 0.9945 | 1.0000 | no detectable difference | no detectable difference |
| non_iid_label | FedProx(0.01) | 42 123 456 789 1011 | +2.8 | [+0.2, +6.8] | 0.0414 | 0.5795 | federation improves generalisation beyond the client's own distribution | no detectable difference |
| dirichlet_0.1 | FedAvg | 42 123 456 | +10.0 | [+0.0, +22.2] | 0.0502 | 0.6005 | federation improves generalisation beyond the client's own distribution | no detectable difference |
| dirichlet_0.1 | FedProx(0.01) | 42 123 456 | +9.9 | [-2.7, +24.5] | 0.1186 | 1.0000 | no detectable difference | no detectable difference |
| dirichlet_0.5 | FedAvg | 42 123 456 | +10.3 | [+0.2, +23.8] | 0.0462 | 0.6005 | federation improves generalisation beyond the client's own distribution | no detectable difference |
| dirichlet_0.5 | FedProx(0.01) | 42 123 456 | -3.6 | [-20.6, +22.4] | 0.9521 | 1.0000 | no detectable difference | no detectable difference |
| dirichlet_1 | FedAvg | 42 123 456 | -14.0 | [-23.7, -0.1] | 0.0488 | 0.6005 | federation worse | no detectable difference |
| dirichlet_1 | FedProx(0.01) | 42 123 456 | -11.9 | [-21.2, +1.4] | 0.1236 | 1.0000 | no detectable difference | no detectable difference |
| iid_sub0.05 | FedAvg | 42 123 456 | +8.0 | [+1.7, +17.0] | 0.0088 | 0.1408 | federation improves generalisation beyond the client's own distribution | no detectable difference |
| iid_sub0.05 | FedProx(0.01) | 42 123 456 | +5.8 | [-0.4, +14.0] | 0.0708 | 0.7079 | no detectable difference | no detectable difference |
| iid_sub0.01 | FedAvg | 42 123 456 | +4.9 | [-2.3, +13.0] | 0.2324 | 1.0000 | no detectable difference | no detectable difference |
| iid_sub0.01 | FedProx(0.01) | 42 123 456 | +3.7 | [-1.4, +8.8] | 0.1614 | 1.0000 | no detectable difference | no detectable difference |
| non_iid_label_sub0.05 | FedAvg | 42 123 456 | +3.8 | [+0.6, +8.8] | 0.0266 | 0.3990 | federation improves generalisation beyond the client's own distribution | no detectable difference |
| non_iid_label_sub0.05 | FedProx(0.01) | 42 123 456 | +2.1 | [-3.0, +8.0] | 0.3626 | 1.0000 | no detectable difference | no detectable difference |
| non_iid_label_sub0.01 | FedAvg | 42 123 456 | +12.6 | [-2.9, +27.9] | 0.1200 | 1.0000 | no detectable difference | no detectable difference |
| non_iid_label_sub0.01 | FedProx(0.01) | 42 123 456 | +16.8 | [+5.7, +25.7] | 0.0044 | 0.0748 | federation improves generalisation beyond the client's own distribution | no detectable difference |

## heterogeneous, clean set (Holm m = 18)

| partition | strategy | seeds | diff | 95 % CI | p | p Holm | verdict | verdict (Holm) |
|---|---|---|---|---|---|---|---|---|
| iid | FedAvg | 42 123 456 789 1011 | -1.1 | [-13.9, +12.6] | 0.9683 | 1.0000 | no detectable difference | no detectable difference |
| iid | FedProx(0.01) | 42 123 456 789 1011 | +5.8 | [+2.3, +13.0] | 0.0014 | 0.0252 | federation improves generalisation beyond the client's own distribution | federation improves generalisation beyond the client's own distribution |
| non_iid_label | FedAvg | 42 123 456 789 1011 | -0.7 | [-6.7, +4.0] | 0.9991 | 1.0000 | no detectable difference | no detectable difference |
| non_iid_label | FedProx(0.01) | 42 123 456 789 1011 | +2.8 | [+0.1, +7.0] | 0.0438 | 0.5255 | federation improves generalisation beyond the client's own distribution | no detectable difference |
| dirichlet_0.1 | FedAvg | 42 123 456 | +10.1 | [-0.4, +22.8] | 0.0584 | 0.6423 | no detectable difference | no detectable difference |
| dirichlet_0.1 | FedProx(0.01) | 42 123 456 | +10.3 | [-2.5, +24.1] | 0.1162 | 1.0000 | no detectable difference | no detectable difference |
| dirichlet_0.5 | FedAvg | 42 123 456 | +10.8 | [+0.7, +24.3] | 0.0346 | 0.4844 | federation improves generalisation beyond the client's own distribution | no detectable difference |
| dirichlet_0.5 | FedProx(0.01) | 42 123 456 | -3.2 | [-20.5, +23.6] | 0.9747 | 1.0000 | no detectable difference | no detectable difference |
| dirichlet_1 | FedAvg | 42 123 456 | -14.7 | [-24.3, -0.7] | 0.0354 | 0.4844 | federation worse | no detectable difference |
| dirichlet_1 | FedProx(0.01) | 42 123 456 | -12.5 | [-22.0, +1.3] | 0.1164 | 1.0000 | no detectable difference | no detectable difference |
| iid_sub0.05 | FedAvg | 42 123 456 | +7.9 | [+1.8, +16.6] | 0.0082 | 0.1312 | federation improves generalisation beyond the client's own distribution | no detectable difference |
| iid_sub0.05 | FedProx(0.01) | 42 123 456 | +5.7 | [-0.6, +14.0] | 0.0770 | 0.7699 | no detectable difference | no detectable difference |
| iid_sub0.01 | FedAvg | 42 123 456 | +4.9 | [-2.4, +12.7] | 0.2376 | 1.0000 | no detectable difference | no detectable difference |
| iid_sub0.01 | FedProx(0.01) | 42 123 456 | +3.7 | [-1.4, +8.6] | 0.1610 | 1.0000 | no detectable difference | no detectable difference |
| non_iid_label_sub0.05 | FedAvg | 42 123 456 | +3.8 | [+0.7, +8.7] | 0.0226 | 0.3390 | federation improves generalisation beyond the client's own distribution | no detectable difference |
| non_iid_label_sub0.05 | FedProx(0.01) | 42 123 456 | +2.1 | [-3.2, +7.7] | 0.3668 | 1.0000 | no detectable difference | no detectable difference |
| non_iid_label_sub0.01 | FedAvg | 42 123 456 | +12.6 | [-2.6, +27.9] | 0.1172 | 1.0000 | no detectable difference | no detectable difference |
| non_iid_label_sub0.01 | FedProx(0.01) | 42 123 456 | +16.8 | [+5.4, +25.5] | 0.0062 | 0.1054 | federation improves generalisation beyond the client's own distribution | no detectable difference |

## maxn, full set (Holm m = 4)

| partition | strategy | seeds | diff | 95 % CI | p | p Holm | verdict | verdict (Holm) |
|---|---|---|---|---|---|---|---|---|
| iid | FedAvg | 42 123 456 789 1011 | +6.9 | [+1.1, +17.7] | 0.0218 | 0.0436 | federation improves generalisation beyond the client's own distribution | federation improves generalisation beyond the client's own distribution |
| iid | FedProx(0.01) | 42 123 456 789 1011 | +8.4 | [+3.5, +18.3] | 0.0006 | 0.0018 | federation improves generalisation beyond the client's own distribution | federation improves generalisation beyond the client's own distribution |
| non_iid_label | FedAvg | 42 123 456 789 1011 | +3.6 | [-0.2, +8.3] | 0.0566 | 0.0566 | no detectable difference | no detectable difference |
| non_iid_label | FedProx(0.01) | 42 123 456 789 1011 | +3.8 | [+1.9, +7.7] | 0.0002 | 0.0008 | federation improves generalisation beyond the client's own distribution | federation improves generalisation beyond the client's own distribution |

## maxn, clean set (Holm m = 4)

| partition | strategy | seeds | diff | 95 % CI | p | p Holm | verdict | verdict (Holm) |
|---|---|---|---|---|---|---|---|---|
| iid | FedAvg | 42 123 456 789 1011 | +7.2 | [+1.0, +18.2] | 0.0224 | 0.0448 | federation improves generalisation beyond the client's own distribution | federation improves generalisation beyond the client's own distribution |
| iid | FedProx(0.01) | 42 123 456 789 1011 | +8.7 | [+3.8, +19.1] | 0.0004 | 0.0012 | federation improves generalisation beyond the client's own distribution | federation improves generalisation beyond the client's own distribution |
| non_iid_label | FedAvg | 42 123 456 789 1011 | +3.6 | [-0.4, +8.7] | 0.0700 | 0.0700 | no detectable difference | no detectable difference |
| non_iid_label | FedProx(0.01) | 42 123 456 789 1011 | +3.8 | [+1.9, +7.9] | 0.0002 | 0.0008 | federation improves generalisation beyond the client's own distribution | federation improves generalisation beyond the client's own distribution |
