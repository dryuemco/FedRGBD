# FedRGBD summary table

`mean ± std [CI_low, CI_high]` — 95% CI of the mean over seeds (`±`/CI shown as `n/a` when only one seed is available).

## Distribution: `iid`

| config | metric | n_seeds | mean ± std [95% CI] | min | max |
|---|---|---|---|---|---|
| Centralized iid | best_accuracy | 3 | 0.9977 ± 0.0002 [0.9971, 0.9983] | 0.9975 | 0.9979 |
| Centralized iid | final_accuracy | 3 | 0.9965 ± 0.0013 [0.9931, 0.9998] | 0.9952 | 0.9979 |
| Centralized iid | final_cumulative_mb | 3 | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 |
| Centralized iid | final_elapsed_s | 3 | 15276.2 ± 601.3857 [13782.3, 16770.2] | 14662.8 | 15864.8 |
| Centralized iid | final_loss | 3 | 0.0123 ± 0.0037 [0.0031, 0.0214] | 0.0088 | 0.0161 |
| Centralized iid | round1_accuracy | 3 | 0.9916 ± 0.0029 [0.9845, 0.9986] | 0.9883 | 0.9933 |
| Centralized iid | total_time_s | 3 | 15276.4 ± 601.4020 [13782.4, 16770.3] | 14662.9 | 15864.9 |
| FedAvg iid [3N] | best_accuracy | 2 | 0.9938 ± 0.0005 [0.9891, 0.9985] | 0.9934 | 0.9942 |
| FedAvg iid [3N] | final_accuracy | 2 | 0.9932 ± 0.0003 [0.9904, 0.9960] | 0.9930 | 0.9934 |
| FedAvg iid [3N] | final_cumulative_mb | 2 | 110.3102 ± 0.0000 [110.3102, 110.3102] | 110.3102 | 110.3102 |
| FedAvg iid [3N] | final_elapsed_s | 2 | 6107.2 ± 203.5619 [4278.3, 7936.2] | 5963.3 | 6251.2 |
| FedAvg iid [3N] | final_loss | 2 | 0.0520 ± 0.0086 [-0.0257, 0.1297] | 0.0459 | 0.0581 |
| FedAvg iid [3N] | round1_accuracy | 2 | 0.6823 ± 0.1693 [-0.8388, 2.2034] | 0.5626 | 0.8020 |
| FedAvg iid [3N] | total_time_s | 2 | 6107.2 ± 203.5619 [4278.3, 7936.2] | 5963.3 | 6251.2 |
| FedBN iid [3N] | best_accuracy | 3 | 0.9569 ± 0.0312 [0.8793, 1.0344] | 0.9295 | 0.9909 |
| FedBN iid [3N] | final_accuracy | 3 | 0.9569 ± 0.0312 [0.8793, 1.0344] | 0.9295 | 0.9909 |
| FedBN iid [3N] | final_cumulative_mb | 3 | 110.3102 ± 0.0000 [110.3102, 110.3102] | 110.3102 | 110.3102 |
| FedBN iid [3N] | final_elapsed_s | 3 | 6322.6 ± 91.0997 [6096.3, 6548.9] | 6237.6 | 6418.8 |
| FedBN iid [3N] | final_loss | 3 | 0.1707 ± 0.1033 [-0.0859, 0.4273] | 0.0597 | 0.2640 |
| FedBN iid [3N] | round1_accuracy | 3 | 0.5708 ± 0.0712 [0.3939, 0.7478] | 0.5203 | 0.6523 |
| FedBN iid [3N] | total_time_s | 3 | 6322.6 ± 91.0997 [6096.3, 6548.9] | 6237.6 | 6418.8 |
| FedProx(mu=0.01) iid [3N] | best_accuracy | 3 | 0.9918 ± 0.0012 [0.9890, 0.9947] | 0.9905 | 0.9927 |
| FedProx(mu=0.01) iid [3N] | final_accuracy | 3 | 0.9918 ± 0.0012 [0.9890, 0.9947] | 0.9905 | 0.9927 |
| FedProx(mu=0.01) iid [3N] | final_cumulative_mb | 3 | 110.3102 ± 0.0000 [110.3102, 110.3102] | 110.3102 | 110.3102 |
| FedProx(mu=0.01) iid [3N] | final_elapsed_s | 3 | 10355.3 ± 126.0639 [10042.1, 10668.4] | 10277.6 | 10500.7 |
| FedProx(mu=0.01) iid [3N] | final_loss | 3 | 0.0262 ± 0.0022 [0.0206, 0.0317] | 0.0236 | 0.0274 |
| FedProx(mu=0.01) iid [3N] | round1_accuracy | 3 | 0.9824 ± 0.0024 [0.9764, 0.9885] | 0.9803 | 0.9851 |
| FedProx(mu=0.01) iid [3N] | total_time_s | 3 | 10355.3 ± 126.0639 [10042.1, 10668.4] | 10277.6 | 10500.7 |
| FedProx(mu=0.1) iid [3N] | best_accuracy | 3 | 0.9848 ± 0.0016 [0.9808, 0.9888] | 0.9831 | 0.9863 |
| FedProx(mu=0.1) iid [3N] | final_accuracy | 3 | 0.9829 ± 0.0048 [0.9711, 0.9947] | 0.9775 | 0.9863 |
| FedProx(mu=0.1) iid [3N] | final_cumulative_mb | 3 | 110.3102 ± 0.0000 [110.3102, 110.3102] | 110.3102 | 110.3102 |
| FedProx(mu=0.1) iid [3N] | final_elapsed_s | 3 | 10547.0 ± 396.9983 [9560.8, 11533.2] | 10285.0 | 11003.8 |
| FedProx(mu=0.1) iid [3N] | final_loss | 3 | 0.0454 ± 0.0112 [0.0177, 0.0731] | 0.0366 | 0.0580 |
| FedProx(mu=0.1) iid [3N] | round1_accuracy | 3 | 0.9706 ± 0.0183 [0.9251, 1.0160] | 0.9496 | 0.9831 |
| FedProx(mu=0.1) iid [3N] | total_time_s | 3 | 10547.0 ± 396.9983 [9560.8, 11533.2] | 10285.0 | 11003.8 |
| Local-only iid | best_accuracy | 3 | 0.9950 ± 0.0008 [0.9931, 0.9969] | 0.9942 | 0.9958 |
| Local-only iid | final_accuracy | 3 | 0.9949 ± 0.0008 [0.9928, 0.9970] | 0.9939 | 0.9955 |
| Local-only iid | final_cumulative_mb | 3 | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 |
| Local-only iid | final_elapsed_s | 3 | 5163.9 ± 45.8762 [5050.0, 5277.9] | 5122.4 | 5213.2 |
| Local-only iid | final_loss | 3 | 0.0153 ± 0.0008 [0.0133, 0.0173] | 0.0146 | 0.0162 |
| Local-only iid | round1_accuracy | 3 | 0.9936 ± 0.0006 [0.9923, 0.9950] | 0.9931 | 0.9942 |
| Local-only iid | total_time_s | 3 | 15836.9 ± 162.3786 [15433.5, 16240.3] | 15698.7 | 16015.7 |

## Distribution: `non_iid_label`

| config | metric | n_seeds | mean ± std [95% CI] | min | max |
|---|---|---|---|---|---|
| Centralized non_iid_label | best_accuracy | 3 | 0.9982 ± 0.0003 [0.9975, 0.9990] | 0.9980 | 0.9986 |
| Centralized non_iid_label | final_accuracy | 3 | 0.9972 ± 0.0007 [0.9954, 0.9990] | 0.9964 | 0.9978 |
| Centralized non_iid_label | final_cumulative_mb | 3 | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 |
| Centralized non_iid_label | final_elapsed_s | 3 | 15572.2 ± 325.2254 [14764.3, 16380.1] | 15343.0 | 15944.4 |
| Centralized non_iid_label | final_loss | 3 | 0.0101 ± 0.0021 [0.0048, 0.0153] | 0.0076 | 0.0113 |
| Centralized non_iid_label | round1_accuracy | 3 | 0.9967 ± 0.0013 [0.9934, 0.9999] | 0.9954 | 0.9980 |
| Centralized non_iid_label | total_time_s | 3 | 15572.2 ± 325.2557 [14764.2, 16380.2] | 15343.0 | 15944.5 |
| FedAvg non_iid_label [3N] | best_accuracy | 3 | 0.9934 ± 0.0039 [0.9836, 1.0032] | 0.9908 | 0.9979 |
| FedAvg non_iid_label [3N] | final_accuracy | 3 | 0.9931 ± 0.0039 [0.9834, 1.0029] | 0.9908 | 0.9977 |
| FedAvg non_iid_label [3N] | final_cumulative_mb | 3 | 110.3102 ± 0.0000 [110.3102, 110.3102] | 110.3102 | 110.3102 |
| FedAvg non_iid_label [3N] | final_elapsed_s | 3 | 6146.9 ± 409.6401 [5129.3, 7164.5] | 5720.9 | 6538.0 |
| FedAvg non_iid_label [3N] | final_loss | 3 | 0.0513 ± 0.0413 [-0.0513, 0.1540] | 0.0155 | 0.0965 |
| FedAvg non_iid_label [3N] | round1_accuracy | 3 | 0.4979 ± 0.0980 [0.2546, 0.7413] | 0.3865 | 0.5707 |
| FedAvg non_iid_label [3N] | total_time_s | 3 | 6146.9 ± 409.6401 [5129.3, 7164.5] | 5720.9 | 6538.0 |
| FedBN non_iid_label [3N] | best_accuracy | 3 | 0.7684 ± 0.0573 [0.6261, 0.9106] | 0.7164 | 0.8297 |
| FedBN non_iid_label [3N] | final_accuracy | 3 | 0.7506 ± 0.0836 [0.5430, 0.9583] | 0.6632 | 0.8297 |
| FedBN non_iid_label [3N] | final_cumulative_mb | 3 | 110.3102 ± 0.0000 [110.3102, 110.3102] | 110.3102 | 110.3102 |
| FedBN non_iid_label [3N] | final_elapsed_s | 3 | 6162.8 ± 170.0314 [5740.4, 6585.1] | 6020.9 | 6351.2 |
| FedBN non_iid_label [3N] | final_loss | 3 | 0.5774 ± 0.1201 [0.2790, 0.8759] | 0.4623 | 0.7020 |
| FedBN non_iid_label [3N] | round1_accuracy | 3 | 0.3759 ± 0.0136 [0.3421, 0.4097] | 0.3671 | 0.3916 |
| FedBN non_iid_label [3N] | total_time_s | 3 | 6162.8 ± 170.0314 [5740.4, 6585.1] | 6020.9 | 6351.2 |
| FedProx(mu=0.01) non_iid_label [3N] | best_accuracy | 3 | 0.9925 ± 0.0023 [0.9868, 0.9982] | 0.9904 | 0.9949 |
| FedProx(mu=0.01) non_iid_label [3N] | final_accuracy | 3 | 0.9922 ± 0.0024 [0.9863, 0.9982] | 0.9904 | 0.9949 |
| FedProx(mu=0.01) non_iid_label [3N] | final_cumulative_mb | 3 | 110.3102 ± 0.0000 [110.3102, 110.3102] | 110.3102 | 110.3102 |
| FedProx(mu=0.01) non_iid_label [3N] | final_elapsed_s | 3 | 10499.9 ± 494.1405 [9272.4, 11727.4] | 10098.5 | 11051.8 |
| FedProx(mu=0.01) non_iid_label [3N] | final_loss | 3 | 0.0248 ± 0.0036 [0.0159, 0.0337] | 0.0207 | 0.0270 |
| FedProx(mu=0.01) non_iid_label [3N] | round1_accuracy | 3 | 0.9702 ± 0.0196 [0.9214, 1.0189] | 0.9479 | 0.9851 |
| FedProx(mu=0.01) non_iid_label [3N] | total_time_s | 3 | 10499.9 ± 494.1405 [9272.4, 11727.4] | 10098.5 | 11051.8 |
| FedProx(mu=0.1) non_iid_label [3N] | best_accuracy | 3 | 0.9835 ± 0.0016 [0.9795, 0.9876] | 0.9819 | 0.9851 |
| FedProx(mu=0.1) non_iid_label [3N] | final_accuracy | 3 | 0.9833 ± 0.0019 [0.9786, 0.9881] | 0.9813 | 0.9851 |
| FedProx(mu=0.1) non_iid_label [3N] | final_cumulative_mb | 3 | 110.3102 ± 0.0000 [110.3102, 110.3102] | 110.3102 | 110.3102 |
| FedProx(mu=0.1) non_iid_label [3N] | final_elapsed_s | 3 | 9850.4 ± 483.2760 [8649.9, 11050.9] | 9298.0 | 10195.1 |
| FedProx(mu=0.1) non_iid_label [3N] | final_loss | 3 | 0.0465 ± 0.0032 [0.0387, 0.0544] | 0.0433 | 0.0496 |
| FedProx(mu=0.1) non_iid_label [3N] | round1_accuracy | 3 | 0.9703 ± 0.0076 [0.9513, 0.9893] | 0.9633 | 0.9784 |
| FedProx(mu=0.1) non_iid_label [3N] | total_time_s | 3 | 9850.4 ± 483.2760 [8649.9, 11050.9] | 9298.0 | 10195.1 |
| Local-only non_iid_label | best_accuracy | 3 | 0.9960 ± 0.0006 [0.9946, 0.9974] | 0.9953 | 0.9964 |
| Local-only non_iid_label | final_accuracy | 3 | 0.9950 ± 0.0003 [0.9944, 0.9957] | 0.9949 | 0.9953 |
| Local-only non_iid_label | final_cumulative_mb | 3 | 0.0000 ± 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 |
| Local-only non_iid_label | final_elapsed_s | 3 | 5453.5 ± 100.9820 [5202.6, 5704.3] | 5383.1 | 5569.2 |
| Local-only non_iid_label | final_loss | 3 | 0.0158 ± 0.0046 [0.0044, 0.0272] | 0.0127 | 0.0211 |
| Local-only non_iid_label | round1_accuracy | 3 | 0.9938 ± 0.0012 [0.9909, 0.9967] | 0.9926 | 0.9950 |
| Local-only non_iid_label | total_time_s | 3 | 16710.3 ± 340.0093 [15865.7, 17555.0] | 16466.2 | 17098.7 |
