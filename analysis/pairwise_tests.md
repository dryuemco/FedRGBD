# Pairwise strategy comparisons (headline balanced_accuracy)

Paired by seed within each partitioning protocol and data distribution. Headline value per run: selected-round test metric (revision FL), final-epoch test metric (centralized / local-only), final-round validation metric (v1 FL). `d_paired` = mean(diff) / std(diff, ddof=1); `d_unpaired` uses the pooled standard deviation.

## `accuracy` — distribution `iid` — protocol `image` — power `unrecorded`

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

## `accuracy` — distribution `non_iid_label` — protocol `image` — power `unrecorded`

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

## `balanced_accuracy` — distribution `dirichlet_0.1` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9164 | 0.8567 | 0.0596 | 2.212 | 3.054 | 0.2500 | 0.0619 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9164 | 0.8556 | 0.0608 | 1.033 | 1.017 | 0.2500 | 0.2156 |  |
| Centralized vs Local-only | 3 | 0.9164 | 0.9261 | -0.0097 | -0.324 | -0.471 | 0.7500 | 0.6314 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.8567 | 0.8556 | 0.0012 | 0.016 | 0.020 | 1.0000 | 0.9810 |  |
| FedAvg vs Local-only | 3 | 0.8567 | 0.9261 | -0.0694 | -18.304 | -5.367 | 0.2500 | 0.0010 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.8556 | 0.9261 | -0.0705 | -0.921 | -1.217 | 0.2500 | 0.2517 |  |

## `balanced_accuracy` — distribution `dirichlet_0.5` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9085 | 0.8953 | 0.0132 | 0.246 | 0.296 | 1.0000 | 0.7114 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9085 | 0.7562 | 0.1523 | 3.507 | 3.490 | 0.2500 | 0.0260 |  |
| Centralized vs Local-only | 3 | 0.9085 | 0.8793 | 0.0291 | 0.273 | 0.361 | 1.0000 | 0.6825 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.8953 | 0.7562 | 0.1391 | 2.400 | 2.312 | 0.2500 | 0.0533 |  |
| FedAvg vs Local-only | 3 | 0.8953 | 0.8793 | 0.0160 | 0.301 | 0.176 | 1.0000 | 0.6543 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.7562 | 0.8793 | -0.1231 | -1.197 | -1.365 | 0.2500 | 0.1738 |  |

## `balanced_accuracy` — distribution `dirichlet_1` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.8903 | 0.6973 | 0.1931 | 2.088 | 3.987 | 0.2500 | 0.0687 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.8903 | 0.7181 | 0.1723 | 2.108 | 3.807 | 0.2500 | 0.0675 |  |
| Centralized vs Local-only | 3 | 0.8903 | 0.9570 | -0.0667 | -0.977 | -1.490 | 0.2500 | 0.2327 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.6973 | 0.7181 | -0.0208 | -1.877 | -0.753 | 0.2500 | 0.0830 |  |
| FedAvg vs Local-only | 3 | 0.6973 | 0.9570 | -0.2597 | -7.212 | -9.696 | 0.2500 | 0.0063 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.7181 | 0.9570 | -0.2389 | -9.021 | -11.648 | 0.2500 | 0.0041 |  |

## `balanced_accuracy` — distribution `iid` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 5 | 0.9374 | 0.8640 | 0.0734 | 0.701 | 1.049 | 0.1875 | 0.1921 |  |
| Centralized vs FedProx(mu=0.01) | 5 | 0.9374 | 0.9294 | 0.0080 | 0.572 | 0.488 | 0.4375 | 0.2702 |  |
| Centralized vs Local-only | 5 | 0.9374 | 0.8525 | 0.0849 | 5.304 | 6.109 | 0.0625 | 0.0003 |  |
| FedAvg vs FedProx(mu=0.01) | 5 | 0.8640 | 0.9294 | -0.0654 | -0.629 | -0.926 | 0.3125 | 0.2324 |  |
| FedAvg vs Local-only | 5 | 0.8640 | 0.8525 | 0.0115 | 0.125 | 0.164 | 0.8125 | 0.7944 |  |
| FedProx(mu=0.01) vs Local-only | 5 | 0.9294 | 0.8525 | 0.0769 | 3.117 | 4.608 | 0.0625 | 0.0022 |  |

## `balanced_accuracy` — distribution `iid_sub0.01` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9161 | 0.8542 | 0.0619 | 1.934 | 2.350 | 0.2500 | 0.0787 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9161 | 0.8426 | 0.0735 | 5.176 | 8.051 | 0.2500 | 0.0122 |  |
| Centralized vs Local-only | 3 | 0.9161 | 0.7926 | 0.1235 | 2.214 | 3.631 | 0.2500 | 0.0618 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.8542 | 0.8426 | 0.0116 | 0.323 | 0.469 | 1.0000 | 0.6319 |  |
| FedAvg vs Local-only | 3 | 0.8542 | 0.7926 | 0.0616 | 0.768 | 1.501 | 0.2500 | 0.3150 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.8426 | 0.7926 | 0.0500 | 1.109 | 1.525 | 0.2500 | 0.1948 |  |

## `balanced_accuracy` — distribution `iid_sub0.05` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9221 | 0.9058 | 0.0163 | 0.236 | 0.407 | 1.0000 | 0.7224 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9221 | 0.8844 | 0.0377 | 0.625 | 0.928 | 0.2500 | 0.3924 |  |
| Centralized vs Local-only | 3 | 0.9221 | 0.8061 | 0.1160 | 3.630 | 4.546 | 0.2500 | 0.0244 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.9058 | 0.8844 | 0.0214 | 1.078 | 0.462 | 0.2500 | 0.2029 |  |
| FedAvg vs Local-only | 3 | 0.9058 | 0.8061 | 0.0997 | 2.598 | 2.955 | 0.2500 | 0.0460 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.8844 | 0.8061 | 0.0783 | 2.244 | 2.280 | 0.2500 | 0.0603 |  |

## `balanced_accuracy` — distribution `non_iid_label` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 5 | 0.9471 | 0.9240 | 0.0231 | 0.621 | 0.983 | 0.3125 | 0.2372 |  |
| Centralized vs FedAvg (E=1) | 3 | 0.9488 | 0.9399 | 0.0089 | 1.015 | 0.622 | 0.5000 | 0.2209 |  |
| Centralized vs FedAvg (E=2) | 3 | 0.9488 | 0.9400 | 0.0088 | 0.225 | 0.400 | 1.0000 | 0.7349 |  |
| Centralized vs FedAvg (R=10) | 3 | 0.9488 | 0.9680 | -0.0191 | -0.710 | -0.876 | 0.5000 | 0.3440 |  |
| Centralized vs FedAvg (lr=0.0001) | 3 | 0.9488 | 0.9676 | -0.0188 | -1.447 | -2.598 | 0.2500 | 0.1290 |  |
| Centralized vs FedBN (R=10) | 3 | 0.9488 | 0.9422 | 0.0066 | 0.744 | 0.448 | 0.5000 | 0.3264 |  |
| Centralized vs FedProx(mu=0.001) | 3 | 0.9488 | 0.9524 | -0.0036 | -0.549 | -0.485 | 0.5000 | 0.4420 |  |
| Centralized vs FedProx(mu=0.01) | 5 | 0.9471 | 0.9580 | -0.0109 | -0.561 | -0.688 | 0.3125 | 0.2778 |  |
| Centralized vs FedProx(mu=0.01) (E=1) | 3 | 0.9488 | 0.9443 | 0.0045 | 0.171 | 0.178 | 0.7500 | 0.7945 |  |
| Centralized vs FedProx(mu=0.01) (E=2) | 3 | 0.9488 | 0.9580 | -0.0091 | -0.863 | -1.327 | 0.5000 | 0.2735 |  |
| Centralized vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9488 | 0.9634 | -0.0145 | -1.284 | -1.892 | 0.2500 | 0.1562 |  |
| Centralized vs FedProx(mu=0.05) | 3 | 0.9488 | 0.9563 | -0.0075 | -0.756 | -1.081 | 0.5000 | 0.3205 |  |
| Centralized vs FedProx(mu=0.1) | 3 | 0.9488 | 0.9390 | 0.0099 | 0.448 | 0.546 | 0.7500 | 0.5186 |  |
| Centralized vs FedProx(mu=0.5) | 3 | 0.9488 | 0.8848 | 0.0640 | 5.278 | 7.867 | 0.2500 | 0.0118 |  |
| Centralized vs Local-only | 5 | 0.9471 | 0.9528 | -0.0057 | -0.501 | -0.715 | 0.4375 | 0.3251 |  |
| FedAvg vs FedAvg (E=1) | 3 | 0.9313 | 0.9399 | -0.0086 | -0.253 | -0.425 | 0.7500 | 0.7035 |  |
| FedAvg vs FedAvg (E=2) | 3 | 0.9313 | 0.9400 | -0.0087 | -0.261 | -0.331 | 1.0000 | 0.6958 |  |
| FedAvg vs FedAvg (R=10) | 3 | 0.9313 | 0.9680 | -0.0367 | -2.212 | -1.405 | 0.2500 | 0.0619 |  |
| FedAvg vs FedAvg (lr=0.0001) | 3 | 0.9313 | 0.9676 | -0.0363 | -1.695 | -2.272 | 0.2500 | 0.0991 |  |
| FedAvg vs FedBN (R=10) | 3 | 0.9313 | 0.9422 | -0.0109 | -0.343 | -0.529 | 0.7500 | 0.6125 |  |
| FedAvg vs FedProx(mu=0.001) | 3 | 0.9313 | 0.9524 | -0.0211 | -0.981 | -1.315 | 0.5000 | 0.2315 |  |
| FedAvg vs FedProx(mu=0.01) | 5 | 0.9240 | 0.9580 | -0.0340 | -0.728 | -1.267 | 0.1875 | 0.1791 |  |
| FedAvg vs FedProx(mu=0.01) (E=1) | 3 | 0.9313 | 0.9443 | -0.0130 | -0.253 | -0.446 | 0.7500 | 0.7039 |  |
| FedAvg vs FedProx(mu=0.01) (E=2) | 3 | 0.9313 | 0.9580 | -0.0266 | -1.214 | -1.683 | 0.2500 | 0.1702 |  |
| FedAvg vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9313 | 0.9634 | -0.0320 | -1.180 | -1.980 | 0.2500 | 0.1776 |  |
| FedAvg vs FedProx(mu=0.05) | 3 | 0.9313 | 0.9563 | -0.0250 | -1.199 | -1.578 | 0.2500 | 0.1735 |  |
| FedAvg vs FedProx(mu=0.1) | 3 | 0.9313 | 0.9390 | -0.0076 | -0.166 | -0.331 | 1.0000 | 0.8004 |  |
| FedAvg vs FedProx(mu=0.5) | 3 | 0.9313 | 0.8848 | 0.0465 | 2.886 | 2.834 | 0.2500 | 0.0378 |  |
| FedAvg vs Local-only | 5 | 0.9240 | 0.9528 | -0.0287 | -0.987 | -1.248 | 0.1250 | 0.0920 |  |
| FedAvg (E=1) vs FedAvg (E=2) | 3 | 0.9399 | 0.9400 | -0.0001 | -0.002 | -0.004 | 1.0000 | 0.9975 |  |
| FedAvg (E=1) vs FedAvg (R=10) | 3 | 0.9399 | 0.9680 | -0.0281 | -0.884 | -1.155 | 0.2500 | 0.2653 |  |
| FedAvg (E=1) vs FedAvg (lr=0.0001) | 3 | 0.9399 | 0.9676 | -0.0277 | -1.308 | -2.159 | 0.2500 | 0.1517 |  |
| FedAvg (E=1) vs FedBN (R=10) | 3 | 0.9399 | 0.9422 | -0.0023 | -0.524 | -0.125 | 0.5000 | 0.4597 |  |
| FedAvg (E=1) vs FedProx(mu=0.001) | 3 | 0.9399 | 0.9524 | -0.0125 | -0.822 | -0.968 | 0.5000 | 0.2903 |  |
| FedAvg (E=1) vs FedProx(mu=0.01) | 3 | 0.9399 | 0.9564 | -0.0165 | -0.744 | -0.689 | 0.5000 | 0.3264 |  |
| FedAvg (E=1) vs FedProx(mu=0.01) (E=1) | 3 | 0.9399 | 0.9443 | -0.0044 | -0.249 | -0.160 | 0.7500 | 0.7080 |  |
| FedAvg (E=1) vs FedProx(mu=0.01) (E=2) | 3 | 0.9399 | 0.9580 | -0.0180 | -0.961 | -1.428 | 0.5000 | 0.2379 |  |
| FedAvg (E=1) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9399 | 0.9634 | -0.0235 | -1.316 | -1.792 | 0.2500 | 0.1503 |  |
| FedAvg (E=1) vs FedProx(mu=0.05) | 3 | 0.9399 | 0.9563 | -0.0164 | -0.893 | -1.296 | 0.5000 | 0.2620 |  |
| FedAvg (E=1) vs FedProx(mu=0.1) | 3 | 0.9399 | 0.9390 | 0.0010 | 0.049 | 0.046 | 1.0000 | 0.9398 |  |
| FedAvg (E=1) vs FedProx(mu=0.5) | 3 | 0.9399 | 0.8848 | 0.0551 | 2.633 | 4.122 | 0.2500 | 0.0449 |  |
| FedAvg (E=1) vs Local-only | 3 | 0.9399 | 0.9546 | -0.0146 | -1.104 | -1.115 | 0.5000 | 0.1961 |  |
| FedAvg (E=2) vs FedAvg (R=10) | 3 | 0.9400 | 0.9680 | -0.0280 | -0.570 | -0.948 | 0.5000 | 0.4274 |  |
| FedAvg (E=2) vs FedAvg (lr=0.0001) | 3 | 0.9400 | 0.9676 | -0.0276 | -1.048 | -1.308 | 0.2500 | 0.2113 |  |
| FedAvg (E=2) vs FedBN (R=10) | 3 | 0.9400 | 0.9422 | -0.0022 | -0.045 | -0.088 | 1.0000 | 0.9446 |  |
| FedAvg (E=2) vs FedProx(mu=0.001) | 3 | 0.9400 | 0.9524 | -0.0124 | -0.375 | -0.586 | 1.0000 | 0.5831 |  |
| FedAvg (E=2) vs FedProx(mu=0.01) | 3 | 0.9400 | 0.9564 | -0.0164 | -0.330 | -0.561 | 0.7500 | 0.6257 |  |
| FedAvg (E=2) vs FedProx(mu=0.01) (E=1) | 3 | 0.9400 | 0.9443 | -0.0043 | -0.069 | -0.134 | 1.0000 | 0.9161 |  |
| FedAvg (E=2) vs FedProx(mu=0.01) (E=2) | 3 | 0.9400 | 0.9580 | -0.0179 | -0.623 | -0.855 | 0.5000 | 0.3932 |  |
| FedAvg (E=2) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9400 | 0.9634 | -0.0234 | -0.787 | -1.098 | 0.2500 | 0.3059 |  |
| FedAvg (E=2) vs FedProx(mu=0.05) | 3 | 0.9400 | 0.9563 | -0.0163 | -0.552 | -0.776 | 0.7500 | 0.4403 |  |
| FedAvg (E=2) vs FedProx(mu=0.1) | 3 | 0.9400 | 0.9390 | 0.0011 | 0.024 | 0.039 | 1.0000 | 0.9710 |  |
| FedAvg (E=2) vs FedProx(mu=0.5) | 3 | 0.9400 | 0.8848 | 0.0552 | 1.876 | 2.574 | 0.2500 | 0.0831 |  |
| FedAvg (E=2) vs Local-only | 3 | 0.9400 | 0.9546 | -0.0145 | -0.428 | -0.683 | 1.0000 | 0.5356 |  |
| FedAvg (R=10) vs FedAvg (lr=0.0001) | 3 | 0.9680 | 0.9676 | 0.0004 | 0.012 | 0.017 | 1.0000 | 0.9855 |  |
| FedAvg (R=10) vs FedBN (R=10) | 3 | 0.9680 | 0.9422 | 0.0258 | 0.923 | 1.049 | 0.2500 | 0.2510 |  |
| FedAvg (R=10) vs FedProx(mu=0.001) | 3 | 0.9680 | 0.9524 | 0.0156 | 0.589 | 0.744 | 0.5000 | 0.4148 |  |
| FedAvg (R=10) vs FedProx(mu=0.01) | 3 | 0.9680 | 0.9564 | 0.0116 | 0.219 | 0.399 | 0.7500 | 0.7408 |  |
| FedAvg (R=10) vs FedProx(mu=0.01) (E=1) | 3 | 0.9680 | 0.9443 | 0.0237 | 0.507 | 0.737 | 1.0000 | 0.4722 |  |
| FedAvg (R=10) vs FedProx(mu=0.01) (E=2) | 3 | 0.9680 | 0.9580 | 0.0100 | 0.340 | 0.483 | 0.7500 | 0.6152 |  |
| FedAvg (R=10) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9680 | 0.9634 | 0.0046 | 0.135 | 0.219 | 1.0000 | 0.8368 |  |
| FedAvg (R=10) vs FedProx(mu=0.05) | 3 | 0.9680 | 0.9563 | 0.0117 | 0.416 | 0.561 | 0.5000 | 0.5459 |  |
| FedAvg (R=10) vs FedProx(mu=0.1) | 3 | 0.9680 | 0.9390 | 0.0290 | 0.592 | 1.087 | 0.5000 | 0.4128 |  |
| FedAvg (R=10) vs FedProx(mu=0.5) | 3 | 0.9680 | 0.8848 | 0.0832 | 3.437 | 3.917 | 0.2500 | 0.0271 |  |
| FedAvg (R=10) vs Local-only | 3 | 0.9680 | 0.9546 | 0.0134 | 0.433 | 0.637 | 0.7500 | 0.5315 |  |
| FedAvg (lr=0.0001) vs FedBN (R=10) | 3 | 0.9676 | 0.9422 | 0.0254 | 1.161 | 1.904 | 0.2500 | 0.1821 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.001) | 3 | 0.9676 | 0.9524 | 0.0152 | 2.225 | 4.199 | 0.2500 | 0.0612 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.01) | 3 | 0.9676 | 0.9564 | 0.0112 | 0.364 | 0.549 | 1.0000 | 0.5925 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.01) (E=1) | 3 | 0.9676 | 0.9443 | 0.0233 | 0.616 | 0.948 | 0.5000 | 0.3978 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.01) (E=2) | 3 | 0.9676 | 0.9580 | 0.0097 | 3.973 | 3.932 | 0.2500 | 0.0205 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9676 | 0.9634 | 0.0042 | 0.688 | 1.003 | 0.2500 | 0.3556 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.05) | 3 | 0.9676 | 0.9563 | 0.0113 | 3.400 | 4.349 | 0.2500 | 0.0276 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.1) | 3 | 0.9676 | 0.9390 | 0.0287 | 1.117 | 1.691 | 0.2500 | 0.1926 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.5) | 3 | 0.9676 | 0.8848 | 0.0828 | 13.402 | 16.501 | 0.2500 | 0.0019 |  |
| FedAvg (lr=0.0001) vs Local-only | 3 | 0.9676 | 0.9546 | 0.0131 | 1.579 | 3.003 | 0.2500 | 0.1118 |  |
| FedBN (R=10) vs FedProx(mu=0.001) | 3 | 0.9422 | 0.9524 | -0.0102 | -0.667 | -0.760 | 0.5000 | 0.3672 |  |
| FedBN (R=10) vs FedProx(mu=0.01) | 3 | 0.9422 | 0.9564 | -0.0142 | -0.536 | -0.587 | 0.7500 | 0.4512 |  |
| FedBN (R=10) vs FedProx(mu=0.01) (E=1) | 3 | 0.9422 | 0.9443 | -0.0021 | -0.107 | -0.076 | 1.0000 | 0.8695 |  |
| FedBN (R=10) vs FedProx(mu=0.01) (E=2) | 3 | 0.9422 | 0.9580 | -0.0157 | -0.809 | -1.197 | 0.5000 | 0.2964 |  |
| FedBN (R=10) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9422 | 0.9634 | -0.0212 | -1.076 | -1.556 | 0.5000 | 0.2035 |  |
| FedBN (R=10) vs FedProx(mu=0.05) | 3 | 0.9422 | 0.9563 | -0.0141 | -0.752 | -1.071 | 0.5000 | 0.3226 |  |
| FedBN (R=10) vs FedProx(mu=0.1) | 3 | 0.9422 | 0.9390 | 0.0032 | 0.136 | 0.152 | 1.0000 | 0.8359 |  |
| FedBN (R=10) vs FedProx(mu=0.5) | 3 | 0.9422 | 0.8848 | 0.0574 | 2.815 | 4.138 | 0.2500 | 0.0396 |  |
| FedBN (R=10) vs Local-only | 3 | 0.9422 | 0.9546 | -0.0124 | -0.835 | -0.906 | 0.5000 | 0.2850 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.01) | 3 | 0.9524 | 0.9564 | -0.0040 | -0.137 | -0.195 | 1.0000 | 0.8347 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.01) (E=1) | 3 | 0.9524 | 0.9443 | 0.0081 | 0.249 | 0.328 | 0.7500 | 0.7088 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.01) (E=2) | 3 | 0.9524 | 0.9580 | -0.0055 | -1.211 | -1.966 | 0.2500 | 0.1710 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9524 | 0.9634 | -0.0110 | -1.401 | -2.461 | 0.2500 | 0.1361 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.05) | 3 | 0.9524 | 0.9563 | -0.0039 | -1.102 | -1.329 | 0.5000 | 0.1965 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.1) | 3 | 0.9524 | 0.9390 | 0.0135 | 0.550 | 0.791 | 0.7500 | 0.4412 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.5) | 3 | 0.9524 | 0.8848 | 0.0676 | 11.148 | 12.992 | 0.2500 | 0.0027 |  |
| FedProx(mu=0.001) vs Local-only | 3 | 0.9524 | 0.9546 | -0.0021 | -0.428 | -0.470 | 0.5000 | 0.5361 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.01) (E=1) | 3 | 0.9564 | 0.9443 | 0.0121 | 0.570 | 0.380 | 0.5000 | 0.4275 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.01) (E=2) | 3 | 0.9564 | 0.9580 | -0.0015 | -0.053 | -0.076 | 1.0000 | 0.9357 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9564 | 0.9634 | -0.0070 | -0.283 | -0.338 | 1.0000 | 0.6725 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.05) | 3 | 0.9564 | 0.9563 | 0.0001 | 0.003 | 0.004 | 1.0000 | 0.9965 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.1) | 3 | 0.9564 | 0.9390 | 0.0174 | 3.319 | 0.662 | 0.2500 | 0.0289 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.5) | 3 | 0.9564 | 0.8848 | 0.0716 | 2.061 | 3.442 | 0.2500 | 0.0703 |  |
| FedProx(mu=0.01) vs Local-only | 5 | 0.9580 | 0.9528 | 0.0052 | 0.278 | 0.345 | 0.6250 | 0.5681 |  |
| FedProx(mu=0.01) (E=1) vs FedProx(mu=0.01) (E=2) | 3 | 0.9443 | 0.9580 | -0.0136 | -0.384 | -0.557 | 0.7500 | 0.5749 |  |
| FedProx(mu=0.01) (E=1) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9443 | 0.9634 | -0.0191 | -0.574 | -0.771 | 0.5000 | 0.4246 |  |
| FedProx(mu=0.01) (E=1) vs FedProx(mu=0.05) | 3 | 0.9443 | 0.9563 | -0.0120 | -0.339 | -0.490 | 0.7500 | 0.6169 |  |
| FedProx(mu=0.01) (E=1) vs FedProx(mu=0.1) | 3 | 0.9443 | 0.9390 | 0.0054 | 0.234 | 0.181 | 0.7500 | 0.7243 |  |
| FedProx(mu=0.01) (E=1) vs FedProx(mu=0.5) | 3 | 0.9443 | 0.8848 | 0.0595 | 1.547 | 2.392 | 0.2500 | 0.1156 |  |
| FedProx(mu=0.01) (E=1) vs Local-only | 3 | 0.9443 | 0.9546 | -0.0102 | -0.346 | -0.413 | 0.7500 | 0.6096 |  |
| FedProx(mu=0.01) (E=2) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9580 | 0.9634 | -0.0054 | -1.037 | -1.518 | 0.5000 | 0.2143 |  |
| FedProx(mu=0.01) (E=2) vs FedProx(mu=0.05) | 3 | 0.9580 | 0.9563 | 0.0016 | 1.124 | 1.303 | 0.2500 | 0.1908 |  |
| FedProx(mu=0.01) (E=2) vs FedProx(mu=0.1) | 3 | 0.9580 | 0.9390 | 0.0190 | 0.780 | 1.131 | 0.2500 | 0.3090 |  |
| FedProx(mu=0.01) (E=2) vs FedProx(mu=0.5) | 3 | 0.9580 | 0.8848 | 0.0731 | 12.330 | 16.357 | 0.2500 | 0.0022 |  |
| FedProx(mu=0.01) (E=2) vs Local-only | 3 | 0.9580 | 0.9546 | 0.0034 | 0.562 | 0.915 | 0.5000 | 0.4332 |  |
| FedProx(mu=0.01) (lr=0.0001) vs FedProx(mu=0.05) | 3 | 0.9634 | 0.9563 | 0.0071 | 1.093 | 1.922 | 0.2500 | 0.1989 |  |
| FedProx(mu=0.01) (lr=0.0001) vs FedProx(mu=0.1) | 3 | 0.9634 | 0.9390 | 0.0244 | 1.253 | 1.424 | 0.2500 | 0.1622 |  |
| FedProx(mu=0.01) (lr=0.0001) vs FedProx(mu=0.5) | 3 | 0.9634 | 0.8848 | 0.0785 | 7.044 | 13.907 | 0.2500 | 0.0067 |  |
| FedProx(mu=0.01) (lr=0.0001) vs Local-only | 3 | 0.9634 | 0.9546 | 0.0088 | 1.783 | 1.741 | 0.2500 | 0.0908 |  |
| FedProx(mu=0.05) vs FedProx(mu=0.1) | 3 | 0.9563 | 0.9390 | 0.0174 | 0.690 | 1.032 | 0.2500 | 0.3548 |  |
| FedProx(mu=0.05) vs FedProx(mu=0.5) | 3 | 0.9563 | 0.8848 | 0.0715 | 15.057 | 15.717 | 0.2500 | 0.0015 |  |
| FedProx(mu=0.05) vs Local-only | 3 | 0.9563 | 0.9546 | 0.0018 | 0.281 | 0.463 | 0.7500 | 0.6745 |  |
| FedProx(mu=0.1) vs FedProx(mu=0.5) | 3 | 0.9390 | 0.8848 | 0.0541 | 1.815 | 3.118 | 0.2500 | 0.0881 |  |
| FedProx(mu=0.1) vs Local-only | 3 | 0.9390 | 0.9546 | -0.0156 | -0.801 | -0.908 | 0.2500 | 0.2996 |  |
| FedProx(mu=0.5) vs Local-only | 3 | 0.8848 | 0.9546 | -0.0697 | -6.615 | -12.161 | 0.2500 | 0.0075 |  |

## `balanced_accuracy` — distribution `non_iid_label_sub0.01` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9421 | 0.8360 | 0.1061 | 0.886 | 1.265 | 0.2500 | 0.2647 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9421 | 0.8778 | 0.0643 | 1.052 | 1.461 | 0.5000 | 0.2101 |  |
| Centralized vs Local-only | 3 | 0.9421 | 0.8527 | 0.0894 | 0.928 | 1.273 | 0.2500 | 0.2491 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.8360 | 0.8778 | -0.0418 | -0.231 | -0.441 | 1.0000 | 0.7275 |  |
| FedAvg vs Local-only | 3 | 0.8360 | 0.8527 | -0.0166 | -0.085 | -0.152 | 0.7500 | 0.8962 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.8778 | 0.8527 | 0.0251 | 0.310 | 0.303 | 0.7500 | 0.6454 |  |

## `balanced_accuracy` — distribution `non_iid_label_sub0.05` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9281 | 0.9541 | -0.0260 | -1.020 | -1.408 | 0.2500 | 0.2193 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9281 | 0.9373 | -0.0091 | -0.183 | -0.366 | 0.7500 | 0.7811 |  |
| Centralized vs Local-only | 3 | 0.9281 | 0.9448 | -0.0167 | -0.782 | -0.952 | 0.5000 | 0.3083 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.9541 | 0.9373 | 0.0169 | 0.619 | 0.885 | 0.5000 | 0.3958 |  |
| FedAvg vs Local-only | 3 | 0.9541 | 0.9448 | 0.0093 | 0.838 | 1.341 | 0.5000 | 0.2836 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.9373 | 0.9448 | -0.0075 | -0.262 | -0.415 | 0.7500 | 0.6943 |  |

## `balanced_accuracy` — distribution `dirichlet_0.1` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.9046 | 0.5870 | 0.3176 | 6.493 | 8.656 | 0.2500 | 0.0078 |  |

## `balanced_accuracy` — distribution `dirichlet_0.5` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8879 | 0.8013 | 0.0865 | 0.738 | 1.180 | 0.5000 | 0.3292 |  |

## `balanced_accuracy` — distribution `dirichlet_1` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.7598 | 0.8767 | -0.1169 | -4.254 | -3.332 | 0.2500 | 0.0179 |  |

## `balanced_accuracy` — distribution `iid` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 5 | 0.9199 | 0.7655 | 0.1543 | 4.882 | 4.922 | 0.0625 | 0.0004 |  |

## `balanced_accuracy` — distribution `iid_sub0.01` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8912 | 0.8036 | 0.0876 | 3.423 | 4.688 | 0.2500 | 0.0273 |  |

## `balanced_accuracy` — distribution `iid_sub0.05` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8973 | 0.8096 | 0.0877 | 1.582 | 2.754 | 0.2500 | 0.1114 |  |

## `balanced_accuracy` — distribution `non_iid_label` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 5 | 0.9447 | 0.8790 | 0.0657 | 2.299 | 3.238 | 0.0625 | 0.0068 |  |

## `balanced_accuracy` — distribution `non_iid_label_sub0.01` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.9078 | 0.7980 | 0.1098 | 2.294 | 1.714 | 0.2500 | 0.0579 |  |

## `balanced_accuracy` — distribution `non_iid_label_sub0.05` — protocol `group_final_epoch` — power `desktop_gpu`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.9159 | 0.8532 | 0.0627 | 6.989 | 1.526 | 0.2500 | 0.0068 |  |

## `clientmean_balanced_accuracy` — distribution `dirichlet_0.1` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.7918 | 0.8813 | -0.0895 | -1.200 | -1.823 | 0.2500 | 0.1732 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.7918 | 0.8237 | -0.0319 | -0.326 | -0.454 | 0.7500 | 0.6292 |  |
| Centralized vs Local-only | 3 | 0.7918 | 0.5622 | 0.2296 | 2.370 | 3.393 | 0.2500 | 0.0545 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.8813 | 0.8237 | 0.0576 | 1.023 | 1.080 | 0.5000 | 0.2184 |  |
| FedAvg vs Local-only | 3 | 0.8813 | 0.5622 | 0.3191 | 6.245 | 6.378 | 0.2500 | 0.0084 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.8237 | 0.5622 | 0.2615 | 42.284 | 3.692 | 0.2500 | 0.0002 |  |

## `clientmean_balanced_accuracy` — distribution `dirichlet_0.5` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9540 | 0.9361 | 0.0180 | 2.540 | 1.022 | 0.2500 | 0.0480 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9540 | 0.8568 | 0.0972 | 3.228 | 3.044 | 0.2500 | 0.0305 |  |
| Centralized vs Local-only | 3 | 0.9540 | 0.8234 | 0.1306 | 3.735 | 4.721 | 0.2500 | 0.0231 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.9361 | 0.8568 | 0.0792 | 2.609 | 2.374 | 0.2500 | 0.0456 |  |
| FedAvg vs Local-only | 3 | 0.9361 | 0.8234 | 0.1126 | 3.659 | 3.841 | 0.2500 | 0.0240 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.8568 | 0.8234 | 0.0334 | 0.557 | 0.843 | 0.5000 | 0.4366 |  |

## `clientmean_balanced_accuracy` — distribution `dirichlet_1` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9222 | 0.7627 | 0.1595 | 2.453 | 4.693 | 0.2500 | 0.0512 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9222 | 0.7901 | 0.1321 | 2.798 | 5.328 | 0.2500 | 0.0400 |  |
| Centralized vs Local-only | 3 | 0.9222 | 0.9442 | -0.0221 | -0.502 | -0.918 | 0.7500 | 0.4766 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.7627 | 0.7901 | -0.0274 | -0.962 | -0.955 | 0.2500 | 0.2376 |  |
| FedAvg vs Local-only | 3 | 0.7627 | 0.9442 | -0.1816 | -5.557 | -6.467 | 0.2500 | 0.0106 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.7901 | 0.9442 | -0.1541 | -34.771 | -9.796 | 0.2500 | 0.0003 |  |

## `clientmean_balanced_accuracy` — distribution `iid` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 5 | 0.9368 | 0.8639 | 0.0729 | 0.690 | 1.041 | 0.3125 | 0.1978 |  |
| Centralized vs FedProx(mu=0.01) | 5 | 0.9368 | 0.9287 | 0.0082 | 0.572 | 0.479 | 0.4375 | 0.2697 |  |
| Centralized vs Local-only | 5 | 0.9368 | 0.8449 | 0.0920 | 5.300 | 6.146 | 0.0625 | 0.0003 |  |
| FedAvg vs FedProx(mu=0.01) | 5 | 0.8639 | 0.9287 | -0.0647 | -0.623 | -0.917 | 0.3125 | 0.2359 |  |
| FedAvg vs Local-only | 5 | 0.8639 | 0.8449 | 0.0191 | 0.207 | 0.272 | 0.6250 | 0.6671 |  |
| FedProx(mu=0.01) vs Local-only | 5 | 0.9287 | 0.8449 | 0.0838 | 3.328 | 4.825 | 0.0625 | 0.0017 |  |

## `clientmean_balanced_accuracy` — distribution `iid_sub0.01` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9154 | 0.8475 | 0.0679 | 2.054 | 2.473 | 0.2500 | 0.0707 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9154 | 0.8351 | 0.0803 | 5.603 | 8.597 | 0.2500 | 0.0105 |  |
| Centralized vs Local-only | 3 | 0.9154 | 0.7868 | 0.1286 | 2.395 | 3.993 | 0.2500 | 0.0535 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.8475 | 0.8351 | 0.0123 | 0.330 | 0.477 | 1.0000 | 0.6252 |  |
| FedAvg vs Local-only | 3 | 0.8475 | 0.7868 | 0.0607 | 0.768 | 1.508 | 0.2500 | 0.3150 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.8351 | 0.7868 | 0.0484 | 1.137 | 1.567 | 0.2500 | 0.1878 |  |

## `clientmean_balanced_accuracy` — distribution `iid_sub0.05` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9213 | 0.9031 | 0.0182 | 0.253 | 0.435 | 1.0000 | 0.7035 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9213 | 0.8803 | 0.0409 | 0.663 | 0.965 | 0.2500 | 0.3699 |  |
| Centralized vs Local-only | 3 | 0.9213 | 0.7980 | 0.1232 | 3.718 | 4.833 | 0.2500 | 0.0233 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.9031 | 0.8803 | 0.0228 | 1.012 | 0.463 | 0.2500 | 0.2218 |  |
| FedAvg vs Local-only | 3 | 0.9031 | 0.7980 | 0.1051 | 2.622 | 2.945 | 0.2500 | 0.0452 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.8803 | 0.7980 | 0.0823 | 2.210 | 2.261 | 0.2500 | 0.0620 |  |

## `clientmean_balanced_accuracy` — distribution `non_iid_label` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 5 | 0.9375 | 0.8940 | 0.0436 | 0.989 | 1.250 | 0.0625 | 0.0915 |  |
| Centralized vs FedAvg (E=1) | 3 | 0.9394 | 0.9007 | 0.0387 | 1.127 | 1.690 | 0.2500 | 0.1902 |  |
| Centralized vs FedAvg (E=2) | 3 | 0.9394 | 0.9103 | 0.0291 | 0.436 | 0.796 | 0.7500 | 0.5288 |  |
| Centralized vs FedAvg (R=10) | 3 | 0.9394 | 0.9548 | -0.0154 | -0.708 | -0.459 | 0.5000 | 0.3448 |  |
| Centralized vs FedAvg (lr=0.0001) | 3 | 0.9394 | 0.9206 | 0.0188 | 0.561 | 0.938 | 0.5000 | 0.4337 |  |
| Centralized vs FedBN (R=10) | 3 | 0.9394 | 0.8979 | 0.0415 | 0.631 | 0.949 | 0.5000 | 0.3886 |  |
| Centralized vs FedProx(mu=0.001) | 3 | 0.9394 | 0.9430 | -0.0036 | -0.190 | -0.213 | 0.7500 | 0.7729 |  |
| Centralized vs FedProx(mu=0.01) | 5 | 0.9375 | 0.9493 | -0.0117 | -0.536 | -0.785 | 0.3125 | 0.2970 |  |
| Centralized vs FedProx(mu=0.01) (E=1) | 3 | 0.9394 | 0.9179 | 0.0215 | 0.469 | 0.508 | 0.7500 | 0.5015 |  |
| Centralized vs FedProx(mu=0.01) (E=2) | 3 | 0.9394 | 0.9474 | -0.0079 | -0.303 | -0.466 | 0.7500 | 0.6519 |  |
| Centralized vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9394 | 0.9307 | 0.0087 | 0.325 | 0.518 | 0.7500 | 0.6302 |  |
| Centralized vs FedProx(mu=0.05) | 3 | 0.9394 | 0.9384 | 0.0010 | 0.067 | 0.054 | 1.0000 | 0.9188 |  |
| Centralized vs FedProx(mu=0.1) | 3 | 0.9394 | 0.9194 | 0.0200 | 0.398 | 0.722 | 0.5000 | 0.5622 |  |
| Centralized vs FedProx(mu=0.5) | 3 | 0.9394 | 0.8625 | 0.0769 | 5.840 | 3.253 | 0.2500 | 0.0096 |  |
| Centralized vs Local-only | 5 | 0.9375 | 0.8864 | 0.0511 | 2.060 | 2.963 | 0.0625 | 0.0100 |  |
| FedAvg vs FedAvg (E=1) | 3 | 0.9012 | 0.9007 | 0.0005 | 0.008 | 0.016 | 1.0000 | 0.9897 |  |
| FedAvg vs FedAvg (E=2) | 3 | 0.9012 | 0.9103 | -0.0091 | -0.172 | -0.218 | 1.0000 | 0.7938 |  |
| FedAvg vs FedAvg (R=10) | 3 | 0.9012 | 0.9548 | -0.0536 | -1.838 | -1.371 | 0.2500 | 0.0861 |  |
| FedAvg vs FedAvg (lr=0.0001) | 3 | 0.9012 | 0.9206 | -0.0194 | -0.671 | -0.683 | 0.5000 | 0.3648 |  |
| FedAvg vs FedBN (R=10) | 3 | 0.9012 | 0.8979 | 0.0033 | 0.035 | 0.069 | 1.0000 | 0.9566 |  |
| FedAvg vs FedProx(mu=0.001) | 3 | 0.9012 | 0.9430 | -0.0418 | -1.220 | -1.592 | 0.2500 | 0.1689 |  |
| FedAvg vs FedProx(mu=0.01) | 5 | 0.8940 | 0.9493 | -0.0553 | -1.019 | -1.628 | 0.1875 | 0.0849 |  |
| FedAvg vs FedProx(mu=0.01) (E=1) | 3 | 0.9012 | 0.9179 | -0.0167 | -0.207 | -0.357 | 0.7500 | 0.7542 |  |
| FedAvg vs FedProx(mu=0.01) (E=2) | 3 | 0.9012 | 0.9474 | -0.0462 | -1.382 | -1.752 | 0.2500 | 0.1390 |  |
| FedAvg vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9012 | 0.9307 | -0.0295 | -0.768 | -1.128 | 0.5000 | 0.3149 |  |
| FedAvg vs FedProx(mu=0.05) | 3 | 0.9012 | 0.9384 | -0.0372 | -1.253 | -1.373 | 0.2500 | 0.1622 |  |
| FedAvg vs FedProx(mu=0.1) | 3 | 0.9012 | 0.9194 | -0.0182 | -0.271 | -0.532 | 0.7500 | 0.6853 |  |
| FedAvg vs FedProx(mu=0.5) | 3 | 0.9012 | 0.8625 | 0.0387 | 1.767 | 1.248 | 0.2500 | 0.0922 |  |
| FedAvg vs Local-only | 5 | 0.8940 | 0.8864 | 0.0076 | 0.147 | 0.217 | 1.0000 | 0.7582 |  |
| FedAvg (E=1) vs FedAvg (E=2) | 3 | 0.9007 | 0.9103 | -0.0096 | -0.161 | -0.265 | 0.7500 | 0.8069 |  |
| FedAvg (E=1) vs FedAvg (R=10) | 3 | 0.9007 | 0.9548 | -0.0541 | -0.967 | -1.634 | 0.5000 | 0.2358 |  |
| FedAvg (E=1) vs FedAvg (lr=0.0001) | 3 | 0.9007 | 0.9206 | -0.0199 | -0.542 | -1.027 | 0.7500 | 0.4471 |  |
| FedAvg (E=1) vs FedBN (R=10) | 3 | 0.9007 | 0.8979 | 0.0028 | 0.081 | 0.065 | 1.0000 | 0.9018 |  |
| FedAvg (E=1) vs FedProx(mu=0.001) | 3 | 0.9007 | 0.9430 | -0.0423 | -1.738 | -2.626 | 0.2500 | 0.0949 |  |
| FedAvg (E=1) vs FedProx(mu=0.01) | 3 | 0.9007 | 0.9503 | -0.0496 | -11.212 | -2.450 | 0.2500 | 0.0026 |  |
| FedAvg (E=1) vs FedProx(mu=0.01) (E=1) | 3 | 0.9007 | 0.9179 | -0.0172 | -0.416 | -0.410 | 1.0000 | 0.5459 |  |
| FedAvg (E=1) vs FedProx(mu=0.01) (E=2) | 3 | 0.9007 | 0.9474 | -0.0466 | -1.720 | -2.874 | 0.2500 | 0.0966 |  |
| FedAvg (E=1) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9007 | 0.9307 | -0.0300 | -1.368 | -1.883 | 0.2500 | 0.1413 |  |
| FedAvg (E=1) vs FedProx(mu=0.05) | 3 | 0.9007 | 0.9384 | -0.0377 | -1.282 | -2.161 | 0.2500 | 0.1564 |  |
| FedAvg (E=1) vs FedProx(mu=0.1) | 3 | 0.9007 | 0.9194 | -0.0187 | -1.016 | -0.687 | 0.2500 | 0.2206 |  |
| FedAvg (E=1) vs FedProx(mu=0.5) | 3 | 0.9007 | 0.8625 | 0.0382 | 0.917 | 1.657 | 0.5000 | 0.2532 |  |
| FedAvg (E=1) vs Local-only | 3 | 0.9007 | 0.8870 | 0.0137 | 16.481 | 0.611 | 0.2500 | 0.0012 |  |
| FedAvg (E=2) vs FedAvg (R=10) | 3 | 0.9103 | 0.9548 | -0.0445 | -0.583 | -1.019 | 0.5000 | 0.4188 |  |
| FedAvg (E=2) vs FedAvg (lr=0.0001) | 3 | 0.9103 | 0.9206 | -0.0103 | -0.310 | -0.299 | 0.7500 | 0.6454 |  |
| FedAvg (E=2) vs FedBN (R=10) | 3 | 0.9103 | 0.8979 | 0.0124 | 0.143 | 0.239 | 0.7500 | 0.8274 |  |
| FedAvg (E=2) vs FedProx(mu=0.001) | 3 | 0.9103 | 0.9430 | -0.0327 | -0.663 | -1.000 | 0.5000 | 0.3694 |  |
| FedAvg (E=2) vs FedProx(mu=0.01) | 3 | 0.9103 | 0.9503 | -0.0400 | -0.701 | -1.144 | 0.5000 | 0.3488 |  |
| FedAvg (E=2) vs FedProx(mu=0.01) (E=1) | 3 | 0.9103 | 0.9179 | -0.0076 | -0.077 | -0.150 | 1.0000 | 0.9067 |  |
| FedAvg (E=2) vs FedProx(mu=0.01) (E=2) | 3 | 0.9103 | 0.9474 | -0.0371 | -0.894 | -1.130 | 0.2500 | 0.2617 |  |
| FedAvg (E=2) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9103 | 0.9307 | -0.0204 | -0.469 | -0.625 | 1.0000 | 0.5019 |  |
| FedAvg (E=2) vs FedProx(mu=0.05) | 3 | 0.9103 | 0.9384 | -0.0281 | -0.540 | -0.842 | 0.7500 | 0.4485 |  |
| FedAvg (E=2) vs FedProx(mu=0.1) | 3 | 0.9103 | 0.9194 | -0.0091 | -0.175 | -0.231 | 0.7500 | 0.7907 |  |
| FedAvg (E=2) vs FedProx(mu=0.5) | 3 | 0.9103 | 0.8625 | 0.0478 | 0.803 | 1.305 | 0.5000 | 0.2986 |  |
| FedAvg (E=2) vs Local-only | 3 | 0.9103 | 0.8870 | 0.0233 | 0.385 | 0.642 | 0.7500 | 0.5732 |  |
| FedAvg (R=10) vs FedAvg (lr=0.0001) | 3 | 0.9548 | 0.9206 | 0.0342 | 0.766 | 1.096 | 0.5000 | 0.3157 |  |
| FedAvg (R=10) vs FedBN (R=10) | 3 | 0.9548 | 0.8979 | 0.0569 | 0.652 | 1.141 | 0.5000 | 0.3762 |  |
| FedAvg (R=10) vs FedProx(mu=0.001) | 3 | 0.9548 | 0.9430 | 0.0118 | 0.322 | 0.402 | 0.7500 | 0.6331 |  |
| FedAvg (R=10) vs FedProx(mu=0.01) | 3 | 0.9548 | 0.9503 | 0.0045 | 0.087 | 0.142 | 1.0000 | 0.8946 |  |
| FedAvg (R=10) vs FedProx(mu=0.01) (E=1) | 3 | 0.9548 | 0.9179 | 0.0369 | 0.598 | 0.758 | 0.5000 | 0.4094 |  |
| FedAvg (R=10) vs FedProx(mu=0.01) (E=2) | 3 | 0.9548 | 0.9474 | 0.0074 | 0.179 | 0.253 | 1.0000 | 0.7864 |  |
| FedAvg (R=10) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9548 | 0.9307 | 0.0241 | 0.544 | 0.824 | 0.5000 | 0.4455 |  |
| FedAvg (R=10) vs FedProx(mu=0.05) | 3 | 0.9548 | 0.9384 | 0.0164 | 0.535 | 0.544 | 0.5000 | 0.4521 |  |
| FedAvg (R=10) vs FedProx(mu=0.1) | 3 | 0.9548 | 0.9194 | 0.0354 | 0.499 | 0.967 | 0.5000 | 0.4788 |  |
| FedAvg (R=10) vs FedProx(mu=0.5) | 3 | 0.9548 | 0.8625 | 0.0923 | 5.114 | 2.745 | 0.2500 | 0.0125 |  |
| FedAvg (R=10) vs Local-only | 3 | 0.9548 | 0.8870 | 0.0677 | 1.222 | 2.044 | 0.2500 | 0.1686 |  |
| FedAvg (lr=0.0001) vs FedBN (R=10) | 3 | 0.9206 | 0.8979 | 0.0227 | 0.321 | 0.540 | 1.0000 | 0.6341 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.001) | 3 | 0.9206 | 0.9430 | -0.0224 | -1.282 | -1.913 | 0.2500 | 0.1565 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.01) | 3 | 0.9206 | 0.9503 | -0.0297 | -0.909 | -1.751 | 0.5000 | 0.2559 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.01) (E=1) | 3 | 0.9206 | 0.9179 | 0.0027 | 0.038 | 0.066 | 1.0000 | 0.9538 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.01) (E=2) | 3 | 0.9206 | 0.9474 | -0.0268 | -2.565 | -2.252 | 0.2500 | 0.0471 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9206 | 0.9307 | -0.0101 | -0.675 | -0.882 | 0.5000 | 0.3628 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.05) | 3 | 0.9206 | 0.9384 | -0.0178 | -0.936 | -1.320 | 0.2500 | 0.2464 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.1) | 3 | 0.9206 | 0.9194 | 0.0012 | 0.030 | 0.048 | 1.0000 | 0.9639 |  |
| FedAvg (lr=0.0001) vs FedProx(mu=0.5) | 3 | 0.9206 | 0.8625 | 0.0581 | 2.156 | 2.872 | 0.2500 | 0.0648 |  |
| FedAvg (lr=0.0001) vs Local-only | 3 | 0.9206 | 0.8870 | 0.0335 | 0.907 | 1.727 | 0.2500 | 0.2569 |  |
| FedBN (R=10) vs FedProx(mu=0.001) | 3 | 0.8979 | 0.9430 | -0.0451 | -0.760 | -1.111 | 0.2500 | 0.3185 |  |
| FedBN (R=10) vs FedProx(mu=0.01) | 3 | 0.8979 | 0.9503 | -0.0524 | -1.327 | -1.234 | 0.2500 | 0.1483 |  |
| FedBN (R=10) vs FedProx(mu=0.01) (E=1) | 3 | 0.8979 | 0.9179 | -0.0200 | -0.457 | -0.357 | 0.5000 | 0.5114 |  |
| FedBN (R=10) vs FedProx(mu=0.01) (E=2) | 3 | 0.8979 | 0.9474 | -0.0495 | -0.800 | -1.216 | 0.2500 | 0.3001 |  |
| FedBN (R=10) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.8979 | 0.9307 | -0.0328 | -0.580 | -0.809 | 0.5000 | 0.4207 |  |
| FedBN (R=10) vs FedProx(mu=0.05) | 3 | 0.8979 | 0.9384 | -0.0406 | -0.632 | -0.984 | 0.5000 | 0.3878 |  |
| FedBN (R=10) vs FedProx(mu=0.1) | 3 | 0.8979 | 0.9194 | -0.0215 | -0.620 | -0.466 | 0.5000 | 0.3955 |  |
| FedBN (R=10) vs FedProx(mu=0.5) | 3 | 0.8979 | 0.8625 | 0.0354 | 0.468 | 0.807 | 0.5000 | 0.5024 |  |
| FedBN (R=10) vs Local-only | 3 | 0.8979 | 0.8870 | 0.0108 | 0.309 | 0.249 | 0.7500 | 0.6463 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.01) | 3 | 0.9430 | 0.9503 | -0.0072 | -0.364 | -0.553 | 0.5000 | 0.5930 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.01) (E=1) | 3 | 0.9430 | 0.9179 | 0.0251 | 0.469 | 0.642 | 0.5000 | 0.5018 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.01) (E=2) | 3 | 0.9430 | 0.9474 | -0.0043 | -0.553 | -0.851 | 0.5000 | 0.4395 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9430 | 0.9307 | 0.0123 | 1.549 | 3.037 | 0.2500 | 0.1154 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.05) | 3 | 0.9430 | 0.9384 | 0.0046 | 0.753 | 0.559 | 0.5000 | 0.3218 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.1) | 3 | 0.9430 | 0.9194 | 0.0236 | 0.668 | 1.053 | 0.5000 | 0.3669 |  |
| FedProx(mu=0.001) vs FedProx(mu=0.5) | 3 | 0.9430 | 0.8625 | 0.0805 | 4.151 | 4.697 | 0.2500 | 0.0188 |  |
| FedProx(mu=0.001) vs Local-only | 3 | 0.9430 | 0.8870 | 0.0560 | 2.304 | 3.458 | 0.2500 | 0.0574 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.01) (E=1) | 3 | 0.9503 | 0.9179 | 0.0323 | 0.752 | 0.790 | 0.5000 | 0.3225 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.01) (E=2) | 3 | 0.9503 | 0.9474 | 0.0029 | 0.127 | 0.220 | 1.0000 | 0.8458 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9503 | 0.9307 | 0.0196 | 1.103 | 1.518 | 0.5000 | 0.1963 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.05) | 3 | 0.9503 | 0.9384 | 0.0118 | 0.472 | 0.804 | 0.5000 | 0.4995 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.1) | 3 | 0.9503 | 0.9194 | 0.0309 | 1.526 | 1.208 | 0.2500 | 0.1183 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.5) | 3 | 0.9503 | 0.8625 | 0.0878 | 2.340 | 4.167 | 0.2500 | 0.0558 |  |
| FedProx(mu=0.01) vs Local-only | 5 | 0.9493 | 0.8864 | 0.0629 | 7.157 | 4.084 | 0.0625 | 0.0001 |  |
| FedProx(mu=0.01) (E=1) vs FedProx(mu=0.01) (E=2) | 3 | 0.9179 | 0.9474 | -0.0294 | -0.487 | -0.752 | 0.5000 | 0.4877 |  |
| FedProx(mu=0.01) (E=1) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9179 | 0.9307 | -0.0128 | -0.225 | -0.328 | 0.7500 | 0.7344 |  |
| FedProx(mu=0.01) (E=1) vs FedProx(mu=0.05) | 3 | 0.9179 | 0.9384 | -0.0205 | -0.377 | -0.517 | 0.7500 | 0.5804 |  |
| FedProx(mu=0.01) (E=1) vs FedProx(mu=0.1) | 3 | 0.9179 | 0.9194 | -0.0015 | -0.026 | -0.033 | 1.0000 | 0.9682 |  |
| FedProx(mu=0.01) (E=1) vs FedProx(mu=0.5) | 3 | 0.9179 | 0.8625 | 0.0554 | 0.941 | 1.307 | 0.5000 | 0.2447 |  |
| FedProx(mu=0.01) (E=1) vs Local-only | 3 | 0.9179 | 0.8870 | 0.0309 | 0.760 | 0.734 | 0.2500 | 0.3185 |  |
| FedProx(mu=0.01) (E=2) vs FedProx(mu=0.01) (lr=0.0001) | 3 | 0.9474 | 0.9307 | 0.0166 | 3.152 | 3.696 | 0.2500 | 0.0320 |  |
| FedProx(mu=0.01) (E=2) vs FedProx(mu=0.05) | 3 | 0.9474 | 0.9384 | 0.0089 | 0.756 | 1.059 | 0.5000 | 0.3204 |  |
| FedProx(mu=0.01) (E=2) vs FedProx(mu=0.1) | 3 | 0.9474 | 0.9194 | 0.0279 | 0.824 | 1.241 | 0.5000 | 0.2896 |  |
| FedProx(mu=0.01) (E=2) vs FedProx(mu=0.5) | 3 | 0.9474 | 0.8625 | 0.0849 | 3.590 | 4.918 | 0.2500 | 0.0249 |  |
| FedProx(mu=0.01) (E=2) vs Local-only | 3 | 0.9474 | 0.8870 | 0.0603 | 2.209 | 3.699 | 0.2500 | 0.0620 |  |
| FedProx(mu=0.01) (lr=0.0001) vs FedProx(mu=0.05) | 3 | 0.9307 | 0.9384 | -0.0077 | -0.564 | -0.984 | 0.5000 | 0.4314 |  |
| FedProx(mu=0.01) (lr=0.0001) vs FedProx(mu=0.1) | 3 | 0.9307 | 0.9194 | 0.0113 | 0.388 | 0.507 | 0.7500 | 0.5705 |  |
| FedProx(mu=0.01) (lr=0.0001) vs FedProx(mu=0.5) | 3 | 0.9307 | 0.8625 | 0.0682 | 2.555 | 4.018 | 0.2500 | 0.0474 |  |
| FedProx(mu=0.01) (lr=0.0001) vs Local-only | 3 | 0.9307 | 0.8870 | 0.0437 | 1.971 | 2.728 | 0.2500 | 0.0762 |  |
| FedProx(mu=0.05) vs FedProx(mu=0.1) | 3 | 0.9384 | 0.9194 | 0.0190 | 0.460 | 0.813 | 0.5000 | 0.5094 |  |
| FedProx(mu=0.05) vs FedProx(mu=0.5) | 3 | 0.9384 | 0.8625 | 0.0759 | 5.702 | 4.124 | 0.2500 | 0.0101 |  |
| FedProx(mu=0.05) vs Local-only | 3 | 0.9384 | 0.8870 | 0.0514 | 1.756 | 2.933 | 0.2500 | 0.0932 |  |
| FedProx(mu=0.1) vs FedProx(mu=0.5) | 3 | 0.9194 | 0.8625 | 0.0569 | 1.041 | 2.045 | 0.2500 | 0.2131 |  |
| FedProx(mu=0.1) vs Local-only | 3 | 0.9194 | 0.8870 | 0.0324 | 1.684 | 1.187 | 0.2500 | 0.1002 |  |
| FedProx(mu=0.5) vs Local-only | 3 | 0.8625 | 0.8870 | -0.0246 | -0.593 | -1.063 | 0.5000 | 0.4122 |  |

## `clientmean_balanced_accuracy` — distribution `non_iid_label_sub0.01` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.9171 | 0.7928 | 0.1242 | 1.220 | 2.112 | 0.2500 | 0.1690 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9171 | 0.8526 | 0.0644 | 3.664 | 1.985 | 0.2500 | 0.0239 |  |
| Centralized vs Local-only | 3 | 0.9171 | 0.7091 | 0.2079 | 2.296 | 2.866 | 0.2500 | 0.0578 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.7928 | 0.8526 | -0.0598 | -0.519 | -0.948 | 0.7500 | 0.4638 |  |
| FedAvg vs Local-only | 3 | 0.7928 | 0.7091 | 0.0837 | 0.493 | 0.925 | 0.7500 | 0.4834 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.8526 | 0.7091 | 0.1435 | 1.511 | 1.887 | 0.2500 | 0.1202 |  |

## `clientmean_balanced_accuracy` — distribution `non_iid_label_sub0.05` — protocol `group` — power `heterogeneous`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 3 | 0.8938 | 0.9236 | -0.0298 | -0.441 | -0.872 | 0.7500 | 0.5250 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.8938 | 0.8861 | 0.0078 | 0.108 | 0.215 | 1.0000 | 0.8687 |  |
| Centralized vs Local-only | 3 | 0.8938 | 0.8815 | 0.0123 | 0.297 | 0.450 | 0.7500 | 0.6580 |  |
| FedAvg vs FedProx(mu=0.01) | 3 | 0.9236 | 0.8861 | 0.0376 | 3.739 | 1.190 | 0.2500 | 0.0230 |  |
| FedAvg vs Local-only | 3 | 0.9236 | 0.8815 | 0.0421 | 1.553 | 2.012 | 0.2500 | 0.1149 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.8861 | 0.8815 | 0.0046 | 0.150 | 0.189 | 1.0000 | 0.8197 |  |
