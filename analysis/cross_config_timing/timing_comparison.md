# Timing comparison: MAXN_SUPER / heterogeneous

Declared in docs/CROSS_CONFIG_COMPARISON.md (c) and its amendment. Descriptive only: no verdicts, no tests. T_round = test-free round time (`rounds[].timing.round_time_s`); round 1 is the T_round of round 1, a steady-state value is the median T_round over the stated rounds. Ratio = geometric mean over the paired seeds of T_MAXN / T_het, shown with every per-seed ratio; interval = seed bootstrap, B = 10000, 95 % percentile. **The interval is indicative**: with n paired seeds the bootstrap has at most C(2n-1, n) distinct resamples (column 'distinct').

## rounds_1_3

| partition | strategy | quantity | MAXN rounds | het rounds | seeds | ratio | 95 % interval (indicative) | distinct | per-seed ratios |
|---|---|---|---|---|---|---|---|---|---|
| iid | FedAvg | primary | 2-3 | 2-3 | 42 123 456 789 1011 | 0.6726 | [0.6677, 0.6769] | 126 | 0.6772 0.6769 0.6760 0.6642 0.6687 |
| iid | FedAvg | secondary | 2-10 | 2-3 | 42 123 456 789 1011 | 0.6719 | [0.6667, 0.6764] | 126 | 0.6760 0.6772 0.6757 0.6630 0.6676 |
| iid | FedAvg | round1 | 1 | 1 | 42 123 456 789 1011 | 0.7042 | [0.6753, 0.7393] | 126 | 0.7342 0.6758 0.6792 0.7635 0.6728 |
| iid | FedProx(0.01) | primary | 2-3 | 2-3 | 42 123 456 789 1011 | 0.7024 | [0.6955, 0.7073] | 126 | 0.7012 0.6895 0.7067 0.7082 0.7065 |
| iid | FedProx(0.01) | secondary | 2-10 | 2-3 | 42 123 456 789 1011 | 0.7022 | [0.6953, 0.7075] | 126 | 0.7010 0.6889 0.7052 0.7090 0.7072 |
| iid | FedProx(0.01) | round1 | 1 | 1 | 42 123 456 789 1011 | 0.7084 | [0.7023, 0.7140] | 126 | 0.7049 0.6976 0.7095 0.7156 0.7146 |
| non_iid_label | FedAvg | primary | 2-3 | 2-3 | 42 123 456 789 1011 | 0.6669 | [0.6640, 0.6709] | 126 | 0.6746 0.6635 0.6651 0.6638 0.6674 |
| non_iid_label | FedAvg | secondary | 2-10 | 2-3 | 42 123 456 789 1011 | 0.6674 | [0.6651, 0.6707] | 126 | 0.6738 0.6651 0.6647 0.6660 0.6673 |
| non_iid_label | FedAvg | round1 | 1 | 1 | 42 123 456 789 1011 | 0.6825 | [0.6670, 0.7089] | 126 | 0.7355 0.6676 0.6649 0.6767 0.6701 |
| non_iid_label | FedProx(0.01) | primary | 2-3 | 2-3 | 42 123 456 789 1011 | 0.7092 | [0.7039, 0.7160] | 126 | 0.7073 0.7025 0.7219 0.7111 0.7035 |
| non_iid_label | FedProx(0.01) | secondary | 2-10 | 2-3 | 42 123 456 789 1011 | 0.7084 | [0.7033, 0.7144] | 126 | 0.7053 0.7003 0.7203 0.7102 0.7059 |
| non_iid_label | FedProx(0.01) | round1 | 1 | 1 | 42 123 456 789 1011 | 0.7205 | [0.7050, 0.7448] | 126 | 0.7074 0.7001 0.7178 0.7678 0.7116 |

## ten_rounds

| partition | strategy | quantity | MAXN rounds | het rounds | seeds | ratio | 95 % interval (indicative) | distinct | per-seed ratios |
|---|---|---|---|---|---|---|---|---|---|
| non_iid_label | FedAvg | steady | 2-10 | 2-10 | 42 123 456 | 0.6664 | [0.6649, 0.6677] | 10 | 0.6677 0.6665 0.6649 |
| non_iid_label | FedAvg | round1 | 1 | 1 | 42 123 456 | 0.6945 | [0.6670, 0.7480] | 10 | 0.7480 0.6714 0.6670 |
| non_iid_label | FedBN | steady | 2-10 | 2-10 | 42 123 456 | 0.6702 | [0.6649, 0.6733] | 10 | 0.6725 0.6649 0.6733 |
| non_iid_label | FedBN | round1 | 1 | 1 | 42 123 456 | 0.6696 | [0.6668, 0.6730] | 10 | 0.6689 0.6668 0.6730 |

## Straggler (client with the largest fit_wall_s per round)

Counts over the paired seeds x the listed rounds; median fit_wall_s (s) of each node over the same rounds. Reported, never tested.

| family | partition | strategy | configuration | rounds | n | node_a | node_b | node_c | median a | median b | median c |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rounds_1_3 | iid | FedAvg | heterogeneous | 1 | 5 | 0 | 0 | 5 | 917.2 | 903.7 | 1353.5 |
| rounds_1_3 | iid | FedAvg | heterogeneous | 2-3 | 10 | 0 | 0 | 10 | 909.6 | 903.0 | 1346.3 |
| rounds_1_3 | iid | FedAvg | maxn | 1 | 5 | 0 | 4 | 1 | 865.7 | 913.5 | 906.7 |
| rounds_1_3 | iid | FedAvg | maxn | 2-3 | 10 | 0 | 6 | 4 | 854.2 | 905.7 | 873.4 |
| rounds_1_3 | iid | FedAvg | maxn | 2-10 | 45 | 0 | 27 | 18 | 852.0 | 904.2 | 875.6 |
| rounds_1_3 | iid | FedProx(0.01) | heterogeneous | 1 | 5 | 0 | 0 | 5 | 1549.0 | 1498.7 | 2093.5 |
| rounds_1_3 | iid | FedProx(0.01) | heterogeneous | 2-3 | 10 | 0 | 0 | 10 | 1537.5 | 1502.2 | 2090.2 |
| rounds_1_3 | iid | FedProx(0.01) | maxn | 1 | 5 | 0 | 5 | 0 | 1406.6 | 1481.6 | 1432.7 |
| rounds_1_3 | iid | FedProx(0.01) | maxn | 2-3 | 10 | 0 | 10 | 0 | 1402.7 | 1473.4 | 1422.8 |
| rounds_1_3 | iid | FedProx(0.01) | maxn | 2-10 | 45 | 0 | 45 | 0 | 1402.3 | 1473.5 | 1423.9 |
| rounds_1_3 | non_iid_label | FedAvg | heterogeneous | 1 | 5 | 0 | 0 | 5 | 920.1 | 898.6 | 1348.0 |
| rounds_1_3 | non_iid_label | FedAvg | heterogeneous | 2-3 | 10 | 0 | 0 | 10 | 919.2 | 891.6 | 1343.0 |
| rounds_1_3 | non_iid_label | FedAvg | maxn | 1 | 5 | 0 | 1 | 4 | 874.1 | 874.7 | 898.5 |
| rounds_1_3 | non_iid_label | FedAvg | maxn | 2-3 | 10 | 0 | 0 | 10 | 866.2 | 874.2 | 893.4 |
| rounds_1_3 | non_iid_label | FedAvg | maxn | 2-10 | 45 | 0 | 0 | 45 | 866.6 | 874.6 | 893.6 |
| rounds_1_3 | non_iid_label | FedProx(0.01) | heterogeneous | 1 | 5 | 0 | 0 | 5 | 1571.1 | 1466.6 | 2087.8 |
| rounds_1_3 | non_iid_label | FedProx(0.01) | heterogeneous | 2-3 | 10 | 0 | 0 | 10 | 1552.0 | 1454.6 | 2081.1 |
| rounds_1_3 | non_iid_label | FedProx(0.01) | maxn | 1 | 5 | 0 | 5 | 0 | 1425.9 | 1482.3 | 1454.7 |
| rounds_1_3 | non_iid_label | FedProx(0.01) | maxn | 2-3 | 10 | 0 | 10 | 0 | 1421.0 | 1477.1 | 1441.0 |
| rounds_1_3 | non_iid_label | FedProx(0.01) | maxn | 2-10 | 45 | 0 | 44 | 1 | 1420.6 | 1475.7 | 1440.7 |
| ten_rounds | non_iid_label | FedAvg | heterogeneous | 1 | 3 | 0 | 0 | 3 | 921.8 | 897.0 | 1343.8 |
| ten_rounds | non_iid_label | FedAvg | heterogeneous | 2-10 | 27 | 0 | 0 | 27 | 917.9 | 890.3 | 1342.7 |
| ten_rounds | non_iid_label | FedAvg | maxn | 1 | 3 | 0 | 1 | 2 | 874.1 | 874.6 | 896.3 |
| ten_rounds | non_iid_label | FedAvg | maxn | 2-10 | 27 | 0 | 0 | 27 | 868.0 | 873.6 | 893.0 |
| ten_rounds | non_iid_label | FedBN | heterogeneous | 1 | 3 | 0 | 0 | 3 | 927.0 | 896.7 | 1339.0 |
| ten_rounds | non_iid_label | FedBN | heterogeneous | 2-10 | 27 | 0 | 0 | 27 | 918.2 | 893.1 | 1334.8 |
| ten_rounds | non_iid_label | FedBN | maxn | 1 | 3 | 0 | 3 | 0 | 864.7 | 892.8 | 857.4 |
| ten_rounds | non_iid_label | FedBN | maxn | 2-10 | 27 | 0 | 27 | 0 | 856.2 | 894.3 | 856.3 |
