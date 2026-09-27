# Pairwise strategy comparisons (headline accuracy)

Paired by seed within each partitioning protocol and data distribution. Headline value per run: selected-round test metric (revision FL), final-epoch test metric (centralized / local-only), final-round validation metric (v1 FL). `d_paired` = mean(diff) / std(diff, ddof=1); `d_unpaired` uses the pooled standard deviation.

## Distribution: `dirichlet_0.1` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9154 | 0.8184 | 0.0970 | 3.006 | 4.181 | 0.2500 | 0.0350 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9154 | 0.8208 | 0.0945 | 1.163 | 1.188 | 0.2500 | 0.1816 |  |
| Centralized vs Local-only | 3 | 0.9154 | 0.9232 | -0.0079 | -0.227 | -0.327 | 1.0000 | 0.7324 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.8184 | 0.8208 | -0.0025 | -0.024 | -0.032 | 1.0000 | 0.9711 |  |
| FedAvg vs Local-only | 3 | 0.8184 | 0.9232 | -0.1049 | -34.120 | -6.899 | 0.2500 | 0.0003 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.8208 | 0.9232 | -0.1024 | -0.979 | -1.320 | 0.2500 | 0.2322 |  |

## Distribution: `dirichlet_0.5` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.8944 | 0.8717 | 0.0226 | 0.360 | 0.397 | 1.0000 | 0.5966 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.8944 | 0.6938 | 0.2005 | 4.553 | 3.496 | 0.2500 | 0.0157 |  |
| Centralized vs Local-only | 3 | 0.8944 | 0.9122 | -0.0178 | -0.285 | -0.314 | 0.7500 | 0.6701 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.8717 | 0.6938 | 0.1779 | 2.537 | 2.382 | 0.2500 | 0.0481 |  |
| FedAvg vs Local-only | 3 | 0.8717 | 0.9122 | -0.0404 | -81.953 | -0.545 | 0.2500 | 0.0000 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.6938 | 0.9122 | -0.2183 | -3.133 | -2.931 | 0.2500 | 0.0323 |  |

## Distribution: `dirichlet_1` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.8698 | 0.6197 | 0.2501 | 2.054 | 3.893 | 0.2500 | 0.0707 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.8698 | 0.6458 | 0.2240 | 2.070 | 3.703 | 0.2500 | 0.0697 |  |
| Centralized vs Local-only | 3 | 0.8698 | 0.9387 | -0.0689 | -0.740 | -1.123 | 0.5000 | 0.3283 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.6197 | 0.6458 | -0.0261 | -1.877 | -0.751 | 0.2500 | 0.0830 |  |
| FedAvg vs Local-only | 3 | 0.6197 | 0.9387 | -0.3190 | -6.497 | -8.800 | 0.2500 | 0.0078 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.6458 | 0.9387 | -0.2929 | -7.697 | -10.076 | 0.2500 | 0.0056 |  |

## Distribution: `iid` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 5 | 0.9339 | 0.8307 | 0.1032 | 0.807 | 1.180 | 0.1250 | 0.1454 |  |
| Centralized vs FedProx(mu=0.01) | 5 | 0.9339 | 0.9271 | 0.0067 | 0.521 | 0.469 | 0.3125 | 0.3091 |  |
| Centralized vs Local-only | 5 | 0.9339 | 0.8685 | 0.0654 | 4.200 | 5.089 | 0.0625 | 0.0007 |  |
| FedAvg vs FedProx(mu=0.01) | 5 | 0.8307 | 0.9271 | -0.0965 | -0.737 | -1.099 | 0.1875 | 0.1746 |  |
| FedAvg vs Local-only | 5 | 0.8307 | 0.8685 | -0.0379 | -0.330 | -0.432 | 0.8125 | 0.5015 |  |
| FedProx(mu=0.01) vs Local-only | 5 | 0.9271 | 0.8685 | 0.0586 | 2.359 | 3.905 | 0.0625 | 0.0062 |  |

## Distribution: `iid_sub0.01` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9122 | 0.8761 | 0.0361 | 1.608 | 1.615 | 0.2500 | 0.1083 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9122 | 0.8695 | 0.0427 | 2.027 | 3.383 | 0.2500 | 0.0724 |  |
| Centralized vs Local-only | 3 | 0.9122 | 0.8192 | 0.0929 | 1.582 | 2.800 | 0.2500 | 0.1114 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.8761 | 0.8695 | 0.0065 | 0.253 | 0.339 | 1.0000 | 0.7045 |  |
| FedAvg vs Local-only | 3 | 0.8761 | 0.8192 | 0.0568 | 0.823 | 1.568 | 0.2500 | 0.2899 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.8695 | 0.8192 | 0.0503 | 1.164 | 1.612 | 0.2500 | 0.1813 |  |

## Distribution: `iid_sub0.05` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9218 | 0.9127 | 0.0091 | 0.168 | 0.312 | 1.0000 | 0.7984 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9218 | 0.8941 | 0.0277 | 0.534 | 0.922 | 0.7500 | 0.4525 |  |
| Centralized vs Local-only | 3 | 0.9218 | 0.8200 | 0.1018 | 4.246 | 3.802 | 0.2500 | 0.0180 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.9127 | 0.8941 | 0.0186 | 2.020 | 0.594 | 0.2500 | 0.0729 |  |
| FedAvg vs Local-only | 3 | 0.9127 | 0.8200 | 0.0927 | 2.472 | 3.287 | 0.2500 | 0.0505 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.8941 | 0.8200 | 0.0741 | 2.298 | 2.544 | 0.2500 | 0.0577 |  |

## Distribution: `non_iid_label` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 5 | 0.9428 | 0.9047 | 0.0381 | 0.743 | 1.245 | 0.3125 | 0.1722 |  |
| Centralized vs FedAvg (E=1) | 3 | 0.9448 | 0.9273 | 0.0175 | 1.434 | 0.800 | 0.2500 | 0.1309 |  |
| Centralized vs FedAvg (E=2) | 3 | 0.9448 | 0.9248 | 0.0200 | 0.379 | 0.700 | 1.0000 | 0.5793 |  |
| Centralized vs FedAvg (R=10) | 3 | 0.9448 | 0.9601 | -0.0153 | -0.455 | -0.540 | 0.7500 | 0.5136 |  |
| Centralized vs FedAvg (lr=0.0001) | 3 | 0.9448 | 0.9667 | -0.0219 | -1.088 | -1.914 | 0.5000 | 0.2002 |  |
| Centralized vs FedBN (R=10) | 3 | 0.9448 | 0.9308 | 0.0140 | 1.656 | 0.700 | 0.2500 | 0.1031 |  |
| Centralized vs FedProx(mu=0.001) | 3 | 0.9448 | 0.9404 | 0.0044 | 0.386 | 0.383 | 1.0000 | 0.5730 |  |
| Centralized vs FedProx(mu=0.01) | 5 | 0.9428 | 0.9475 | -0.0046 | -0.181 | -0.216 | 0.8125 | 0.7071 |  |
| Centralized vs FedProx(mu=0.01) (E=1) | 3 | 0.9448 | 0.9317 | 0.0131 | 0.417 | 0.398 | 0.7500 | 0.5450 |  |
| Centralized vs FedProx(mu=0.01) (E=2) | 3 | 0.9448 | 0.9479 | -0.0031 | -0.184 | -0.281 | 1.0000 | 0.7805 |  |
| Centralized vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9448 | 0.9613 | -0.0165 | -1.064 | -1.322 | 0.2500 | 0.2067 |  |
| Centralized vs FedProx(mu=0.05) | 3 | 0.9448 | 0.9513 | -0.0065 | -0.295 | -0.485 | 0.7500 | 0.6601 |  |
| Centralized vs FedProx(mu=0.1) | 3 | 0.9448 | 0.9318 | 0.0130 | 0.544 | 0.632 | 0.5000 | 0.4457 |  |
| Centralized vs FedProx(mu=0.5) | 3 | 0.9448 | 0.9067 | 0.0381 | 2.859 | 3.272 | 0.2500 | 0.0385 |  |
| Centralized vs Local-only | 5 | 0.9428 | 0.9554 | -0.0126 | -0.612 | -0.978 | 0.3125 | 0.2430 |  |
| FedAvg vs FedAvg (E=1) | 3 | 0.9139 | 0.9273 | -0.0134 | -0.292 | -0.488 | 0.7500 | 0.6629 |  |
| FedAvg vs FedAvg (E=2) | 3 | 0.9139 | 0.9248 | -0.0109 | -0.260 | -0.329 | 1.0000 | 0.6969 |  |
| FedAvg vs FedAvg (R=10) | 3 | 0.9139 | 0.9601 | -0.0462 | -2.205 | -1.407 | 0.2500 | 0.0622 |  |
| FedAvg vs FedAvg (lr=0.0001) | 3 | 0.9139 | 0.9667 | -0.0528 | -1.914 | -2.624 | 0.2500 | 0.0802 |  |
| FedAvg vs FedBN (R=10) | 3 | 0.9139 | 0.9308 | -0.0169 | -0.444 | -0.648 | 0.7500 | 0.5220 |  |
| FedAvg vs FedProx(mu=0.001) | 3 | 0.9139 | 0.9404 | -0.0265 | -0.981 | -1.315 | 0.5000 | 0.2315 |  |
| FedAvg vs FedProx(mu=0.01) | 5 | 0.9047 | 0.9475 | -0.0427 | -0.728 | -1.267 | 0.1875 | 0.1790 |  |
| FedAvg vs FedProx(mu=0.01) (E=1) | 3 | 0.9139 | 0.9317 | -0.0178 | -0.273 | -0.485 | 0.7500 | 0.6833 |  |
| FedAvg vs FedProx(mu=0.01) (E=2) | 3 | 0.9139 | 0.9479 | -0.0340 | -1.269 | -1.709 | 0.2500 | 0.1590 |  |
| FedAvg vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9139 | 0.9613 | -0.0474 | -1.298 | -2.286 | 0.2500 | 0.1535 |  |
| FedAvg vs FedProx(mu=0.05) | 3 | 0.9139 | 0.9513 | -0.0374 | -2.115 | -1.757 | 0.2500 | 0.0671 |  |
| FedAvg vs FedProx(mu=0.1) | 3 | 0.9139 | 0.9318 | -0.0179 | -0.342 | -0.680 | 0.7500 | 0.6133 |  |
| FedAvg vs FedProx(mu=0.5) | 3 | 0.9139 | 0.9067 | 0.0072 | 0.295 | 0.355 | 1.0000 | 0.6604 |  |
| FedAvg vs Local-only | 5 | 0.9047 | 0.9554 | -0.0507 | -1.435 | -1.749 | 0.0625 | 0.0326 |  |
| FedAvg (E=1) vs FedAvg (E=2) | 3 | 0.9273 | 0.9248 | 0.0025 | 0.039 | 0.077 | 1.0000 | 0.9520 |  |
| FedAvg (E=1) vs FedAvg (R=10) | 3 | 0.9273 | 0.9601 | -0.0328 | -0.795 | -1.016 | 0.2500 | 0.3025 |  |
| FedAvg (E=1) vs FedAvg (lr=0.0001) | 3 | 0.9273 | 0.9667 | -0.0394 | -1.259 | -2.052 | 0.2500 | 0.1610 |  |
| FedAvg (E=1) vs FedBN (R=10) | 3 | 0.9273 | 0.9308 | -0.0035 | -0.385 | -0.137 | 1.0000 | 0.5731 |  |
| FedAvg (E=1) vs FedProx(mu=0.001) | 3 | 0.9273 | 0.9404 | -0.0131 | -0.563 | -0.681 | 0.5000 | 0.4326 |  |
| FedAvg (E=1) vs FedProx(mu=0.01) | 3 | 0.9273 | 0.9454 | -0.0182 | -0.632 | -0.571 | 0.5000 | 0.3879 |  |
| FedAvg (E=1) vs FedProx(mu=0.01) (E=1) | 3 | 0.9273 | 0.9317 | -0.0044 | -0.226 | -0.123 | 0.7500 | 0.7336 |  |
| FedAvg (E=1) vs FedProx(mu=0.01) (E=2) | 3 | 0.9273 | 0.9479 | -0.0206 | -0.727 | -1.087 | 0.5000 | 0.3349 |  |
| FedAvg (E=1) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9273 | 0.9613 | -0.0340 | -1.428 | -1.714 | 0.2500 | 0.1319 |  |
| FedAvg (E=1) vs FedProx(mu=0.05) | 3 | 0.9273 | 0.9513 | -0.0240 | -0.702 | -1.175 | 0.5000 | 0.3480 |  |
| FedAvg (E=1) vs FedProx(mu=0.1) | 3 | 0.9273 | 0.9318 | -0.0045 | -0.189 | -0.177 | 1.0000 | 0.7746 |  |
| FedAvg (E=1) vs FedProx(mu=0.5) | 3 | 0.9273 | 0.9067 | 0.0206 | 0.809 | 1.064 | 0.5000 | 0.2962 |  |
| FedAvg (E=1) vs Local-only | 3 | 0.9273 | 0.9585 | -0.0312 | -1.312 | -1.628 | 0.2500 | 0.1510 |  |
| FedAvg (E=2) vs FedAvg (R=10) | 3 | 0.9248 | 0.9601 | -0.0353 | -0.572 | -0.951 | 0.5000 | 0.4265 |  |
| FedAvg (E=2) vs FedAvg (lr=0.0001) | 3 | 0.9248 | 0.9667 | -0.0419 | -1.282 | -1.576 | 0.2500 | 0.1566 |  |
| FedAvg (E=2) vs FedBN (R=10) | 3 | 0.9248 | 0.9308 | -0.0060 | -0.098 | -0.191 | 1.0000 | 0.8808 |  |
| FedAvg (E=2) vs FedProx(mu=0.001) | 3 | 0.9248 | 0.9404 | -0.0156 | -0.375 | -0.586 | 1.0000 | 0.5828 |  |
| FedAvg (E=2) vs FedProx(mu=0.01) | 3 | 0.9248 | 0.9454 | -0.0207 | -0.330 | -0.562 | 0.7500 | 0.6252 |  |
| FedAvg (E=2) vs FedProx(mu=0.01) (E=1) | 3 | 0.9248 | 0.9317 | -0.0069 | -0.088 | -0.171 | 1.0000 | 0.8924 |  |
| FedAvg (E=2) vs FedProx(mu=0.01) (E=2) | 3 | 0.9248 | 0.9479 | -0.0231 | -0.643 | -0.875 | 0.5000 | 0.3816 |  |
| FedAvg (E=2) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9248 | 0.9613 | -0.0365 | -0.898 | -1.350 | 0.2500 | 0.2602 |  |
| FedAvg (E=2) vs FedProx(mu=0.05) | 3 | 0.9248 | 0.9513 | -0.0265 | -0.770 | -0.964 | 0.5000 | 0.3139 |  |
| FedAvg (E=2) vs FedProx(mu=0.1) | 3 | 0.9248 | 0.9318 | -0.0070 | -0.135 | -0.223 | 1.0000 | 0.8365 |  |
| FedAvg (E=2) vs FedProx(mu=0.5) | 3 | 0.9248 | 0.9067 | 0.0181 | 0.446 | 0.677 | 0.5000 | 0.5203 |  |
| FedAvg (E=2) vs Local-only | 3 | 0.9248 | 0.9585 | -0.0337 | -0.841 | -1.268 | 0.2500 | 0.2826 |  |
| FedAvg (R=10) vs FedAvg (lr=0.0001) | 3 | 0.9601 | 0.9667 | -0.0066 | -0.168 | -0.249 | 1.0000 | 0.7981 |  |
| FedAvg (R=10) vs FedBN (R=10) | 3 | 0.9601 | 0.9308 | 0.0294 | 0.909 | 0.944 | 0.2500 | 0.2561 |  |
| FedAvg (R=10) vs FedProx(mu=0.001) | 3 | 0.9601 | 0.9404 | 0.0197 | 0.592 | 0.747 | 0.5000 | 0.4131 |  |
| FedAvg (R=10) vs FedProx(mu=0.01) | 3 | 0.9601 | 0.9454 | 0.0147 | 0.221 | 0.401 | 0.7500 | 0.7390 |  |
| FedAvg (R=10) vs FedProx(mu=0.01) (E=1) | 3 | 0.9601 | 0.9317 | 0.0284 | 0.473 | 0.701 | 1.0000 | 0.4990 |  |
| FedAvg (R=10) vs FedProx(mu=0.01) (E=2) | 3 | 0.9601 | 0.9479 | 0.0122 | 0.334 | 0.467 | 0.7500 | 0.6214 |  |
| FedAvg (R=10) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9601 | 0.9613 | -0.0012 | -0.027 | -0.045 | 1.0000 | 0.9664 |  |
| FedAvg (R=10) vs FedProx(mu=0.05) | 3 | 0.9601 | 0.9513 | 0.0088 | 0.289 | 0.324 | 0.7500 | 0.6661 |  |
| FedAvg (R=10) vs FedProx(mu=0.1) | 3 | 0.9601 | 0.9318 | 0.0283 | 0.497 | 0.901 | 0.5000 | 0.4803 |  |
| FedAvg (R=10) vs FedProx(mu=0.5) | 3 | 0.9601 | 0.9067 | 0.0534 | 1.690 | 2.018 | 0.2500 | 0.0996 |  |
| FedAvg (R=10) vs Local-only | 3 | 0.9601 | 0.9585 | 0.0017 | 0.043 | 0.064 | 1.0000 | 0.9477 |  |
| FedAvg (lr=0.0001) vs FedBN (R=10) | 3 | 0.9667 | 0.9308 | 0.0359 | 1.268 | 2.102 | 0.2500 | 0.1591 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.001) | 3 | 0.9667 | 0.9404 | 0.0263 | 2.859 | 5.540 | 0.2500 | 0.0385 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.01) | 3 | 0.9667 | 0.9454 | 0.0213 | 0.554 | 0.825 | 0.7500 | 0.4388 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.01) (E=1) | 3 | 0.9667 | 0.9317 | 0.0350 | 0.731 | 1.124 | 0.5000 | 0.3331 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.01) (E=2) | 3 | 0.9667 | 0.9479 | 0.0188 | 5.478 | 5.380 | 0.2500 | 0.0109 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9667 | 0.9613 | 0.0054 | 0.499 | 0.786 | 0.7500 | 0.4787 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.05) | 3 | 0.9667 | 0.9513 | 0.0154 | 1.551 | 1.856 | 0.2500 | 0.1151 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.1) | 3 | 0.9667 | 0.9318 | 0.0349 | 1.311 | 1.977 | 0.2500 | 0.1512 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.5) | 3 | 0.9667 | 0.9067 | 0.0600 | 6.761 | 11.804 | 0.2500 | 0.0072 |  |
| FedAvg (lr=0.0001) vs Local-only | 3 | 0.9667 | 0.9585 | 0.0082 | 1.052 | 1.893 | 0.5000 | 0.2099 |  |
| FedBN (R=10) vs FedProx(mu=0.001) | 3 | 0.9308 | 0.9404 | -0.0096 | -0.500 | -0.562 | 0.5000 | 0.4781 |  |
| FedBN (R=10) vs FedProx(mu=0.01) | 3 | 0.9308 | 0.9454 | -0.0147 | -0.405 | -0.480 | 0.7500 | 0.5558 |  |
| FedBN (R=10) vs FedProx(mu=0.01) (E=1) | 3 | 0.9308 | 0.9317 | -0.0010 | -0.034 | -0.028 | 1.0000 | 0.9579 |  |
| FedBN (R=10) vs FedProx(mu=0.01) (E=2) | 3 | 0.9308 | 0.9479 | -0.0171 | -0.686 | -1.018 | 0.5000 | 0.3569 |  |
| FedBN (R=10) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9308 | 0.9613 | -0.0306 | -1.282 | -1.715 | 0.2500 | 0.1565 |  |
| FedBN (R=10) vs FedProx(mu=0.05) | 3 | 0.9308 | 0.9513 | -0.0205 | -0.716 | -1.113 | 0.5000 | 0.3406 |  |
| FedBN (R=10) vs FedProx(mu=0.1) | 3 | 0.9308 | 0.9318 | -0.0011 | -0.036 | -0.044 | 1.0000 | 0.9557 |  |
| FedBN (R=10) vs FedProx(mu=0.5) | 3 | 0.9308 | 0.9067 | 0.0240 | 1.161 | 1.395 | 0.2500 | 0.1819 |  |
| FedBN (R=10) vs Local-only | 3 | 0.9308 | 0.9585 | -0.0277 | -1.264 | -1.626 | 0.2500 | 0.1599 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.01) | 3 | 0.9404 | 0.9454 | -0.0050 | -0.138 | -0.196 | 1.0000 | 0.8338 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.01) (E=1) | 3 | 0.9404 | 0.9317 | 0.0087 | 0.210 | 0.278 | 0.7500 | 0.7514 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.01) (E=2) | 3 | 0.9404 | 0.9479 | -0.0075 | -1.298 | -2.052 | 0.2500 | 0.1536 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9404 | 0.9613 | -0.0209 | -1.973 | -3.026 | 0.2500 | 0.0760 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.05) | 3 | 0.9404 | 0.9513 | -0.0109 | -0.953 | -1.299 | 0.2500 | 0.2406 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.1) | 3 | 0.9404 | 0.9318 | 0.0086 | 0.336 | 0.485 | 0.7500 | 0.6197 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.5) | 3 | 0.9404 | 0.9067 | 0.0337 | 12.492 | 6.483 | 0.2500 | 0.0021 |  |
| FedProx(mu=0.001) vs Local-only | 3 | 0.9404 | 0.9585 | -0.0181 | -3.122 | -4.022 | 0.2500 | 0.0325 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.01) (E=1) | 3 | 0.9454 | 0.9317 | 0.0137 | 0.547 | 0.342 | 0.5000 | 0.4437 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.01) (E=2) | 3 | 0.9454 | 0.9479 | -0.0025 | -0.065 | -0.096 | 1.0000 | 0.9205 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9454 | 0.9613 | -0.0159 | -0.571 | -0.605 | 0.7500 | 0.4266 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.05) | 3 | 0.9454 | 0.9513 | -0.0058 | -0.124 | -0.219 | 1.0000 | 0.8494 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.1) | 3 | 0.9454 | 0.9318 | 0.0136 | 1.154 | 0.441 | 0.2500 | 0.1837 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.5) | 3 | 0.9454 | 0.9067 | 0.0387 | 0.985 | 1.498 | 0.5000 | 0.2302 |  |
| FedProx(mu=0.01) vs Local-only | 5 | 0.9475 | 0.9554 | -0.0079 | -0.314 | -0.415 | 0.8125 | 0.5208 |  |
| FedProx(mu=0.01) (E=1) vs FedProx(mu=0.01) (E=2) | 3 | 0.9317 | 0.9479 | -0.0162 | -0.356 | -0.522 | 0.7500 | 0.6008 |  |
| FedProx(mu=0.01) (E=1) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9317 | 0.9613 | -0.0296 | -0.776 | -0.939 | 0.5000 | 0.3110 |  |
| FedProx(mu=0.01) (E=1) vs FedProx(mu=0.05) | 3 | 0.9317 | 0.9513 | -0.0195 | -0.371 | -0.613 | 0.7500 | 0.5864 |  |
| FedProx(mu=0.01) (E=1) vs FedProx(mu=0.1) | 3 | 0.9317 | 0.9318 | -0.0001 | -0.003 | -0.003 | 1.0000 | 0.9961 |  |
| FedProx(mu=0.01) (E=1) vs FedProx(mu=0.5) | 3 | 0.9317 | 0.9067 | 0.0250 | 0.571 | 0.802 | 0.5000 | 0.4268 |  |
| FedProx(mu=0.01) (E=1) vs Local-only | 3 | 0.9317 | 0.9585 | -0.0267 | -0.668 | -0.860 | 0.5000 | 0.3670 |  |
| FedProx(mu=0.01) (E=2) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9479 | 0.9613 | -0.0134 | -1.348 | -2.193 | 0.2500 | 0.1446 |  |
| FedProx(mu=0.01) (E=2) vs FedProx(mu=0.05) | 3 | 0.9479 | 0.9513 | -0.0034 | -0.363 | -0.437 | 1.0000 | 0.5936 |  |
| FedProx(mu=0.01) (E=2) vs FedProx(mu=0.1) | 3 | 0.9479 | 0.9318 | 0.0161 | 0.617 | 0.925 | 0.2500 | 0.3970 |  |
| FedProx(mu=0.01) (E=2) vs FedProx(mu=0.5) | 3 | 0.9479 | 0.9067 | 0.0412 | 7.369 | 10.095 | 0.2500 | 0.0061 |  |
| FedProx(mu=0.01) (E=2) vs Local-only | 3 | 0.9479 | 0.9585 | -0.0106 | -1.825 | -3.369 | 0.2500 | 0.0872 |  |
| FedProx(mu=0.01) (lr=0.0001) vs FedProx(mu=0.05) | 3 | 0.9613 | 0.9513 | 0.0100 | 0.522 | 1.035 | 0.5000 | 0.4614 |  |
| FedProx(mu=0.01) (lr=0.0001) vs FedProx(mu=0.1) | 3 | 0.9613 | 0.9318 | 0.0295 | 1.834 | 1.608 | 0.2500 | 0.0864 |  |
| FedProx(mu=0.01) (lr=0.0001) vs FedProx(mu=0.5) | 3 | 0.9613 | 0.9067 | 0.0546 | 4.269 | 7.637 | 0.2500 | 0.0178 |  |
| FedProx(mu=0.01) (lr=0.0001) vs Local-only | 3 | 0.9613 | 0.9585 | 0.0029 | 0.583 | 0.432 | 0.5000 | 0.4191 |  |
| FedProx(mu=0.05) vs FedProx(mu=0.1) | 3 | 0.9513 | 0.9318 | 0.0194 | 0.551 | 1.027 | 0.7500 | 0.4407 |  |
| FedProx(mu=0.05) vs FedProx(mu=0.5) | 3 | 0.9513 | 0.9067 | 0.0445 | 5.007 | 5.199 | 0.2500 | 0.0130 |  |
| FedProx(mu=0.05) vs Local-only | 3 | 0.9513 | 0.9585 | -0.0072 | -0.486 | -0.879 | 0.5000 | 0.4888 |  |
| FedProx(mu=0.1) vs FedProx(mu=0.5) | 3 | 0.9318 | 0.9067 | 0.0251 | 0.894 | 1.413 | 0.5000 | 0.2617 |  |
| FedProx(mu=0.1) vs Local-only | 3 | 0.9318 | 0.9585 | -0.0266 | -1.290 | -1.515 | 0.2500 | 0.1550 |  |
| FedProx(mu=0.5) vs Local-only | 3 | 0.9067 | 0.9585 | -0.0517 | -6.585 | -10.689 | 0.2500 | 0.0076 |  |

## Distribution: `non_iid_label_sub0.01` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9393 | 0.8721 | 0.0672 | 0.818 | 1.126 | 0.2500 | 0.2921 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9393 | 0.9057 | 0.0336 | 0.696 | 1.029 | 0.5000 | 0.3511 |  |
| Centralized vs Local-only | 3 | 0.9393 | 0.8635 | 0.0758 | 0.830 | 1.198 | 0.2500 | 0.2872 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.8721 | 0.9057 | -0.0336 | -0.258 | -0.494 | 1.0000 | 0.6988 |  |
| FedAvg vs Local-only | 3 | 0.8721 | 0.8635 | 0.0086 | 0.055 | 0.099 | 1.0000 | 0.9329 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.9057 | 0.8635 | 0.0422 | 0.546 | 0.592 | 0.5000 | 0.4443 |  |

## Distribution: `non_iid_label_sub0.05` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9196 | 0.9505 | -0.0309 | -2.124 | -1.597 | 0.2500 | 0.0666 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9196 | 0.9266 | -0.0070 | -0.130 | -0.252 | 0.7500 | 0.8429 |  |
| Centralized vs Local-only | 3 | 0.9196 | 0.9481 | -0.0285 | -1.653 | -1.832 | 0.2500 | 0.1034 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.9505 | 0.9266 | 0.0239 | 0.537 | 0.909 | 0.7500 | 0.4502 |  |
| FedAvg vs Local-only | 3 | 0.9505 | 0.9481 | 0.0024 | 0.140 | 0.188 | 1.0000 | 0.8310 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.9266 | 0.9481 | -0.0215 | -0.563 | -0.910 | 0.5000 | 0.4321 |  |

## Distribution: `dirichlet_0.1` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8936 | 0.9070 | -0.0134 | -0.222 | -0.288 | 1.0000 | 0.7376 |  |

## Distribution: `dirichlet_0.5` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8704 | 0.7304 | 0.1400 | 0.516 | 0.878 | 0.5000 | 0.4656 |  |

## Distribution: `dirichlet_1` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.7024 | 0.8378 | -0.1354 | -2.234 | -2.192 | 0.2500 | 0.0607 |  |

## Distribution: `iid` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 5 | 0.9092 | 0.7830 | 0.1262 | 2.736 | 3.232 | 0.0625 | 0.0036 |  |

## Distribution: `iid_sub0.01` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8831 | 0.8398 | 0.0433 | 1.207 | 1.633 | 0.2500 | 0.1716 |  |

## Distribution: `iid_sub0.05` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8798 | 0.8406 | 0.0391 | 0.679 | 1.015 | 0.5000 | 0.3607 |  |

## Distribution: `non_iid_label` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 5 | 0.9422 | 0.9415 | 0.0008 | 0.033 | 0.036 | 1.0000 | 0.9446 |  |

## Distribution: `non_iid_label_sub0.01` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8936 | 0.8889 | 0.0047 | 0.143 | 0.102 | 0.7500 | 0.8277 |  |

## Distribution: `non_iid_label_sub0.05` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.9129 | 0.9039 | 0.0090 | 0.284 | 0.135 | 0.7500 | 0.6711 |  |

## Distribution: `iid` — protocol `image` — power `unrecorded`

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

## Distribution: `non_iid_label` — protocol `image` — power `unrecorded`

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
