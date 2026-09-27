# Pairwise strategy comparisons (headline accuracy)

Paired by seed within each partitioning protocol and data distribution. Headline value per run: selected-round test metric (revision FL), final-epoch test metric (centralized / local-only), final-round validation metric (v1 FL). `d_paired` = mean(diff) / std(diff, ddof=1); `d_unpaired` uses the pooled standard deviation.

## Distribution: `dirichlet_0.1` — protocol `group`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9154 | 0.8184 | 0.0970 | 3.006 | 4.181 | 0.2500 | 0.0350 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9154 | 0.8208 | 0.0945 | 1.163 | 1.188 | 0.2500 | 0.1816 |  |
| Centralized vs Local-only | 3 | 0.9154 | 0.9232 | -0.0079 | -0.227 | -0.327 | 1.0000 | 0.7324 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.8184 | 0.8208 | -0.0025 | -0.024 | -0.032 | 1.0000 | 0.9711 |  |
| FedAvg vs Local-only | 3 | 0.8184 | 0.9232 | -0.1049 | -34.120 | -6.899 | 0.2500 | 0.0003 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.8208 | 0.9232 | -0.1024 | -0.979 | -1.320 | 0.2500 | 0.2322 |  |

## Distribution: `dirichlet_0.5` — protocol `group`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.8944 | 0.8717 | 0.0226 | 0.360 | 0.397 | 1.0000 | 0.5966 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.8944 | 0.6938 | 0.2005 | 4.553 | 3.496 | 0.2500 | 0.0157 |  |
| Centralized vs Local-only | 3 | 0.8944 | 0.9122 | -0.0178 | -0.285 | -0.314 | 0.7500 | 0.6701 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.8717 | 0.6938 | 0.1779 | 2.537 | 2.382 | 0.2500 | 0.0481 |  |
| FedAvg vs Local-only | 3 | 0.8717 | 0.9122 | -0.0404 | -81.953 | -0.545 | 0.2500 | 0.0000 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.6938 | 0.9122 | -0.2183 | -3.133 | -2.931 | 0.2500 | 0.0323 |  |

## Distribution: `dirichlet_1` — protocol `group`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.8698 | 0.6197 | 0.2501 | 2.054 | 3.893 | 0.2500 | 0.0707 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.8698 | 0.6458 | 0.2240 | 2.070 | 3.703 | 0.2500 | 0.0697 |  |
| Centralized vs Local-only | 3 | 0.8698 | 0.9387 | -0.0689 | -0.740 | -1.123 | 0.5000 | 0.3283 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.6197 | 0.6458 | -0.0261 | -1.877 | -0.751 | 0.2500 | 0.0830 |  |
| FedAvg vs Local-only | 3 | 0.6197 | 0.9387 | -0.3190 | -6.497 | -8.800 | 0.2500 | 0.0078 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.6458 | 0.9387 | -0.2929 | -7.697 | -10.076 | 0.2500 | 0.0056 |  |

## Distribution: `iid` — protocol `group`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 5 | 0.9339 | 0.8307 | 0.1032 | 0.807 | 1.180 | 0.1250 | 0.1454 |  |
| Centralized vs FedProx(mu=0.01) | 5 | 0.9339 | 0.9271 | 0.0067 | 0.521 | 0.469 | 0.3125 | 0.3091 |  |
| Centralized vs Local-only | 5 | 0.9339 | 0.8685 | 0.0654 | 4.200 | 5.089 | 0.0625 | 0.0007 |  |
| FedAvg vs FedProx(mu=0.01) | 5 | 0.8307 | 0.9271 | -0.0965 | -0.737 | -1.099 | 0.1875 | 0.1746 |  |
| FedAvg vs Local-only | 5 | 0.8307 | 0.8685 | -0.0379 | -0.330 | -0.432 | 0.8125 | 0.5015 |  |
| FedProx(mu=0.01) vs Local-only | 5 | 0.9271 | 0.8685 | 0.0586 | 2.359 | 3.905 | 0.0625 | 0.0062 |  |

## Distribution: `iid_sub0.01` — protocol `group`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9122 | 0.8761 | 0.0361 | 1.608 | 1.615 | 0.2500 | 0.1083 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9122 | 0.8695 | 0.0427 | 2.027 | 3.383 | 0.2500 | 0.0724 |  |
| Centralized vs Local-only | 3 | 0.9122 | 0.8192 | 0.0929 | 1.582 | 2.800 | 0.2500 | 0.1114 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.8761 | 0.8695 | 0.0065 | 0.253 | 0.339 | 1.0000 | 0.7045 |  |
| FedAvg vs Local-only | 3 | 0.8761 | 0.8192 | 0.0568 | 0.823 | 1.568 | 0.2500 | 0.2899 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.8695 | 0.8192 | 0.0503 | 1.164 | 1.612 | 0.2500 | 0.1813 |  |

## Distribution: `iid_sub0.05` — protocol `group`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9218 | 0.9127 | 0.0091 | 0.168 | 0.312 | 1.0000 | 0.7984 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9218 | 0.8941 | 0.0277 | 0.534 | 0.922 | 0.7500 | 0.4525 |  |
| Centralized vs Local-only | 3 | 0.9218 | 0.8200 | 0.1018 | 4.246 | 3.802 | 0.2500 | 0.0180 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.9127 | 0.8941 | 0.0186 | 2.020 | 0.594 | 0.2500 | 0.0729 |  |
| FedAvg vs Local-only | 3 | 0.9127 | 0.8200 | 0.0927 | 2.472 | 3.287 | 0.2500 | 0.0505 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.8941 | 0.8200 | 0.0741 | 2.298 | 2.544 | 0.2500 | 0.0577 |  |

## Distribution: `non_iid_label` — protocol `group`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 5 | 0.9428 | 0.9195 | 0.0233 | 0.449 | 0.725 | 0.6250 | 0.3722 |  |
| Centralized vs FedBN | 3 | 0.9448 | 0.9308 | 0.0140 | 1.656 | 0.700 | 0.2500 | 0.1031 |  |
| Centralized vs FedProx(mu=0.001) | 3 | 0.9448 | 0.9404 | 0.0044 | 0.386 | 0.383 | 1.0000 | 0.5730 |  |
| Centralized vs FedProx(mu=0.01) | 5 | 0.9428 | 0.9482 | -0.0053 | -0.346 | -0.344 | 0.6250 | 0.4827 |  |
| Centralized vs FedProx(mu=0.05) | 3 | 0.9448 | 0.9513 | -0.0065 | -0.295 | -0.485 | 0.7500 | 0.6601 |  |
| Centralized vs FedProx(mu=0.1) | 3 | 0.9448 | 0.9318 | 0.0130 | 0.544 | 0.632 | 0.5000 | 0.4457 |  |
| Centralized vs FedProx(mu=0.5) | 3 | 0.9448 | 0.9067 | 0.0381 | 2.859 | 3.272 | 0.2500 | 0.0385 |  |
| Centralized vs Local-only | 5 | 0.9428 | 0.9554 | -0.0126 | -0.612 | -0.978 | 0.3125 | 0.2430 |  |
| FedAvg vs FedBN | 3 | 0.9386 | 0.9308 | 0.0078 | 0.298 | 0.419 | 0.7500 | 0.6575 |  |
| FedAvg vs FedProx(mu=0.001) | 3 | 0.9386 | 0.9404 | -0.0018 | -0.174 | -0.210 | 1.0000 | 0.7918 |  |
| FedAvg vs FedProx(mu=0.01) | 5 | 0.9195 | 0.9482 | -0.0286 | -0.579 | -0.901 | 0.3125 | 0.2650 |  |
| FedAvg vs FedProx(mu=0.05) | 3 | 0.9386 | 0.9513 | -0.0127 | -3.928 | -1.147 | 0.2500 | 0.0209 |  |
| FedAvg vs FedProx(mu=0.1) | 3 | 0.9386 | 0.9318 | 0.0067 | 0.189 | 0.352 | 1.0000 | 0.7742 |  |
| FedAvg vs FedProx(mu=0.5) | 3 | 0.9386 | 0.9067 | 0.0318 | 4.050 | 3.567 | 0.2500 | 0.0197 |  |
| FedAvg vs Local-only | 5 | 0.9195 | 0.9554 | -0.0359 | -1.009 | -1.173 | 0.0625 | 0.0870 |  |
| FedBN vs FedProx(mu=0.001) | 3 | 0.9308 | 0.9404 | -0.0096 | -0.500 | -0.562 | 0.5000 | 0.4781 |  |
| FedBN vs FedProx(mu=0.01) | 3 | 0.9308 | 0.9466 | -0.0158 | -0.769 | -0.714 | 0.2500 | 0.3144 |  |
| FedBN vs FedProx(mu=0.05) | 3 | 0.9308 | 0.9513 | -0.0205 | -0.716 | -1.113 | 0.5000 | 0.3406 |  |
| FedBN vs FedProx(mu=0.1) | 3 | 0.9308 | 0.9318 | -0.0011 | -0.036 | -0.044 | 1.0000 | 0.9557 |  |
| FedBN vs FedProx(mu=0.5) | 3 | 0.9308 | 0.9067 | 0.0240 | 1.161 | 1.395 | 0.2500 | 0.1819 |  |
| FedBN vs Local-only | 3 | 0.9308 | 0.9585 | -0.0277 | -1.264 | -1.626 | 0.2500 | 0.1599 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.01) | 3 | 0.9404 | 0.9466 | -0.0062 | -0.311 | -0.416 | 0.7500 | 0.6436 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.05) | 3 | 0.9404 | 0.9513 | -0.0109 | -0.953 | -1.299 | 0.2500 | 0.2406 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.1) | 3 | 0.9404 | 0.9318 | 0.0086 | 0.336 | 0.485 | 0.7500 | 0.6197 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.5) | 3 | 0.9404 | 0.9067 | 0.0337 | 12.492 | 6.483 | 0.2500 | 0.0021 |  |
| FedProx(mu=0.001) vs Local-only | 3 | 0.9404 | 0.9585 | -0.0181 | -3.122 | -4.022 | 0.2500 | 0.0325 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.05) | 3 | 0.9466 | 0.9513 | -0.0047 | -0.151 | -0.285 | 1.0000 | 0.8178 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.1) | 3 | 0.9466 | 0.9318 | 0.0148 | 1.678 | 0.654 | 0.2500 | 0.1008 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.5) | 3 | 0.9466 | 0.9067 | 0.0399 | 1.763 | 2.655 | 0.2500 | 0.0926 |  |
| FedProx(mu=0.01) vs Local-only | 5 | 0.9482 | 0.9554 | -0.0072 | -0.461 | -0.604 | 0.4375 | 0.3611 |  |
| FedProx(mu=0.05) vs FedProx(mu=0.1) | 3 | 0.9513 | 0.9318 | 0.0194 | 0.551 | 1.027 | 0.7500 | 0.4407 |  |
| FedProx(mu=0.05) vs FedProx(mu=0.5) | 3 | 0.9513 | 0.9067 | 0.0445 | 5.007 | 5.199 | 0.2500 | 0.0130 |  |
| FedProx(mu=0.05) vs Local-only | 3 | 0.9513 | 0.9585 | -0.0072 | -0.486 | -0.879 | 0.5000 | 0.4888 |  |
| FedProx(mu=0.1) vs FedProx(mu=0.5) | 3 | 0.9318 | 0.9067 | 0.0251 | 0.894 | 1.413 | 0.5000 | 0.2617 |  |
| FedProx(mu=0.1) vs Local-only | 3 | 0.9318 | 0.9585 | -0.0266 | -1.290 | -1.515 | 0.2500 | 0.1550 |  |
| FedProx(mu=0.5) vs Local-only | 3 | 0.9067 | 0.9585 | -0.0517 | -6.585 | -10.689 | 0.2500 | 0.0076 |  |

## Distribution: `non_iid_label_sub0.01` — protocol `group`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9393 | 0.8721 | 0.0672 | 0.818 | 1.126 | 0.2500 | 0.2921 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9393 | 0.9057 | 0.0336 | 0.696 | 1.029 | 0.5000 | 0.3511 |  |
| Centralized vs Local-only | 3 | 0.9393 | 0.8635 | 0.0758 | 0.830 | 1.198 | 0.2500 | 0.2872 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.8721 | 0.9057 | -0.0336 | -0.258 | -0.494 | 1.0000 | 0.6988 |  |
| FedAvg vs Local-only | 3 | 0.8721 | 0.8635 | 0.0086 | 0.055 | 0.099 | 1.0000 | 0.9329 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.9057 | 0.8635 | 0.0422 | 0.546 | 0.592 | 0.5000 | 0.4443 |  |

## Distribution: `non_iid_label_sub0.05` — protocol `group`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9196 | 0.9505 | -0.0309 | -2.124 | -1.597 | 0.2500 | 0.0666 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9196 | 0.9266 | -0.0070 | -0.130 | -0.252 | 0.7500 | 0.8429 |  |
| Centralized vs Local-only | 3 | 0.9196 | 0.9481 | -0.0285 | -1.653 | -1.832 | 0.2500 | 0.1034 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.9505 | 0.9266 | 0.0239 | 0.537 | 0.909 | 0.7500 | 0.4502 |  |
| FedAvg vs Local-only | 3 | 0.9505 | 0.9481 | 0.0024 | 0.140 | 0.188 | 1.0000 | 0.8310 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.9266 | 0.9481 | -0.0215 | -0.563 | -0.910 | 0.5000 | 0.4321 |  |

## Distribution: `dirichlet_0.1` — protocol `group_final_epoch`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8936 | 0.9070 | -0.0134 | -0.222 | -0.288 | 1.0000 | 0.7376 |  |

## Distribution: `dirichlet_0.5` — protocol `group_final_epoch`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8704 | 0.7304 | 0.1400 | 0.516 | 0.878 | 0.5000 | 0.4656 |  |

## Distribution: `dirichlet_1` — protocol `group_final_epoch`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.7024 | 0.8378 | -0.1354 | -2.234 | -2.192 | 0.2500 | 0.0607 |  |

## Distribution: `iid` — protocol `group_final_epoch`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 5 | 0.9092 | 0.7830 | 0.1262 | 2.736 | 3.232 | 0.0625 | 0.0036 |  |

## Distribution: `iid_sub0.01` — protocol `group_final_epoch`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8831 | 0.8398 | 0.0433 | 1.207 | 1.633 | 0.2500 | 0.1716 |  |

## Distribution: `iid_sub0.05` — protocol `group_final_epoch`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8798 | 0.8406 | 0.0391 | 0.679 | 1.015 | 0.5000 | 0.3607 |  |

## Distribution: `non_iid_label` — protocol `group_final_epoch`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 5 | 0.9422 | 0.9415 | 0.0008 | 0.033 | 0.036 | 1.0000 | 0.9446 |  |

## Distribution: `non_iid_label_sub0.01` — protocol `group_final_epoch`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8936 | 0.8889 | 0.0047 | 0.143 | 0.102 | 0.7500 | 0.8277 |  |

## Distribution: `non_iid_label_sub0.05` — protocol `group_final_epoch`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.9129 | 0.9039 | 0.0090 | 0.284 | 0.135 | 0.7500 | 0.6711 |  |

## Distribution: `iid` — protocol `image`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 2 | 0.9971 | 0.9932 | 0.0039 | 2.852 | 5.022 | 0.5000 | 0.1547 |  |
| Centralized vs FedBN | 3 | 0.9965 | 0.9569 | 0.0396 | 1.218 | 1.793 | 0.2500 | 0.1693 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9965 | 0.9918 | 0.0047 | 1.879 | 3.709 | 0.2500 | 0.0828 |  |
| Centralized vs FedProx(mu=0.1) | 3 | 0.9965 | 0.9829 | 0.0135 | 2.242 | 3.880 | 0.2500 | 0.0604 |  |
| Centralized vs Local-only | 3 | 0.9965 | 0.9949 | 0.0016 | 0.758 | 1.421 | 0.5000 | 0.3195 |  |
| FedAvg vs FedBN | 2 | 0.9932 | 0.9398 | 0.0534 | 3.740 | 5.173 | 0.5000 | 0.1190 |  |
| FedAvg vs FedProx(mu=0.01) | 2 | 0.9932 | 0.9914 | 0.0018 | 1.886 | 1.940 | 0.5000 | 0.2284 |  |
| FedAvg vs FedProx(mu=0.1) | 2 | 0.9932 | 0.9812 | 0.0120 | 2.407 | 3.194 | 0.5000 | 0.1819 |  |
| FedAvg vs Local-only | 2 | 0.9932 | 0.9947 | -0.0015 | -1.921 | -1.854 | 0.5000 | 0.2246 |  |
| FedBN vs FedProx(mu=0.01) | 3 | 0.9569 | 0.9918 | -0.0350 | -1.157 | -1.583 | 0.2500 | 0.1831 |  |
| FedBN vs FedProx(mu=0.1) | 3 | 0.9569 | 0.9829 | -0.0261 | -0.954 | -1.168 | 0.5000 | 0.2403 |  |
| FedBN vs Local-only | 3 | 0.9569 | 0.9949 | -0.0380 | -1.241 | -1.722 | 0.2500 | 0.1647 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.1) | 3 | 0.9918 | 0.9829 | 0.0089 | 2.475 | 2.572 | 0.2500 | 0.0503 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.9918 | 0.9949 | -0.0031 | -7.298 | -3.017 | 0.2500 | 0.0062 |  |
| FedProx(mu=0.1) vs Local-only | 3 | 0.9829 | 0.9949 | -0.0119 | -3.031 | -3.503 | 0.2500 | 0.0344 |  |

## Distribution: `non_iid_label` — protocol `image`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 2 | 0.9969 | 0.9942 | 0.0026 | 0.623 | 0.755 | 1.0000 | 0.5403 |  |
| Centralized vs FedBN | 3 | 0.9972 | 0.7506 | 0.2466 | 2.968 | 4.172 | 0.2500 | 0.0358 |  |
| Centralized vs FedProx(mu=0.01) | 2 | 0.9969 | 0.9909 | 0.0060 | 4.523 | 9.046 | 0.5000 | 0.0987 |  |
| Centralized vs FedProx(mu=0.1) | 2 | 0.9969 | 0.9832 | 0.0136 | 4.053 | 6.930 | 0.5000 | 0.1100 |  |
| Centralized vs Local-only | 3 | 0.9972 | 0.9950 | 0.0022 | 2.841 | 3.934 | 0.2500 | 0.0389 |  |
| FedAvg vs FedBN | 2 | 0.9942 | 0.7464 | 0.2478 | 2.195 | 2.973 | 0.5000 | 0.1984 |  |
| FedAvg vs FedProx(mu=0.01) | 2 | 0.9942 | 0.9909 | 0.0034 | 0.603 | 0.960 | 1.0000 | 0.5503 |  |
| FedAvg vs FedProx(mu=0.1) | 2 | 0.9942 | 0.9832 | 0.0110 | 1.447 | 2.781 | 0.5000 | 0.2893 |  |
| FedAvg vs Local-only | 2 | 0.9942 | 0.9951 | -0.0009 | -0.194 | -0.257 | 1.0000 | 0.8294 |  |
| FedBN vs FedProx(mu=0.01) | 2 | 0.7464 | 0.9909 | -0.2444 | -2.064 | -2.935 | 0.5000 | 0.2101 |  |
| FedBN vs FedProx(mu=0.1) | 2 | 0.7464 | 0.9832 | -0.2368 | -1.966 | -2.843 | 0.5000 | 0.2198 |  |
| FedBN vs Local-only | 3 | 0.7506 | 0.9950 | -0.2444 | -2.931 | -4.135 | 0.2500 | 0.0367 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.1) | 2 | 0.9909 | 0.9832 | 0.0076 | 3.748 | 3.885 | 0.5000 | 0.1187 |  |
| FedProx(mu=0.01) vs Local-only | 2 | 0.9909 | 0.9951 | -0.0042 | -4.431 | -8.274 | 0.5000 | 0.1007 |  |
| FedProx(mu=0.1) vs Local-only | 2 | 0.9832 | 0.9951 | -0.0119 | -3.966 | -6.185 | 0.5000 | 0.1123 |  |
