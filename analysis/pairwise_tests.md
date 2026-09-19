# Pairwise strategy comparisons (headline accuracy)

Paired by seed within each partitioning protocol and data distribution. Headline value per run: selected-round test metric (revision FL), final-epoch test metric (centralized / local-only), final-round validation metric (v1 FL). `d_paired` = mean(diff) / std(diff, ddof=1); `d_unpaired` uses the pooled standard deviation.

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
