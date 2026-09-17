# Pairwise strategy comparisons (final accuracy)

Paired by seed within each data distribution. `d_paired` = mean(diff) / std(diff, ddof=1); `d_unpaired` uses the pooled standard deviation.

## Distribution: `dirichlet_0.1`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8936 | 0.9070 | -0.0134 | -0.222 | -0.288 | 1.0000 | 0.7376 |  |

## Distribution: `dirichlet_0.5`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8704 | 0.7304 | 0.1400 | 0.516 | 0.878 | 0.5000 | 0.4656 |  |

## Distribution: `dirichlet_1`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.7024 | 0.8378 | -0.1354 | -2.234 | -2.192 | 0.2500 | 0.0607 |  |

## Distribution: `iid`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 2 | 0.9578 | 0.9932 | -0.0354 | -1.805 | -2.595 | 0.5000 | 0.2377 |  |
| Centralized vs FedBN | 3 | 0.9579 | 0.9569 | 0.0010 | 0.027 | 0.042 | 1.0000 | 0.9674 |  |
| Centralized vs FedProx(mu=0.01) | 3 | 0.9579 | 0.9918 | -0.0340 | -2.336 | -3.512 | 0.2500 | 0.0560 |  |
| Centralized vs FedProx(mu=0.1) | 3 | 0.9579 | 0.9829 | -0.0251 | -1.424 | -2.457 | 0.2500 | 0.1324 |  |
| Centralized vs Local-only | 5 | 0.9323 | 0.8483 | 0.0840 | 2.824 | 1.736 | 0.0625 | 0.0032 |  |
| FedAvg vs FedBN | 2 | 0.9932 | 0.9398 | 0.0534 | 3.740 | 5.173 | 0.5000 | 0.1190 |  |
| FedAvg vs FedProx(mu=0.01) | 2 | 0.9932 | 0.9914 | 0.0018 | 1.886 | 1.940 | 0.5000 | 0.2284 |  |
| FedAvg vs FedProx(mu=0.1) | 2 | 0.9932 | 0.9812 | 0.0120 | 2.407 | 3.194 | 0.5000 | 0.1819 |  |
| FedAvg vs Local-only | 2 | 0.9932 | 0.8713 | 0.1219 | 3.842 | 5.488 | 0.5000 | 0.1159 |  |
| FedBN vs FedProx(mu=0.01) | 3 | 0.9569 | 0.9918 | -0.0350 | -1.157 | -1.583 | 0.2500 | 0.1831 |  |
| FedBN vs FedProx(mu=0.1) | 3 | 0.9569 | 0.9829 | -0.0261 | -0.954 | -1.168 | 0.5000 | 0.2403 |  |
| FedBN vs Local-only | 3 | 0.9569 | 0.8860 | 0.0709 | 2.163 | 2.178 | 0.2500 | 0.0644 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.1) | 3 | 0.9918 | 0.9829 | 0.0089 | 2.475 | 2.572 | 0.2500 | 0.0503 |  |
| FedProx(mu=0.01) vs Local-only | 3 | 0.9918 | 0.8860 | 0.1058 | 3.124 | 4.423 | 0.2500 | 0.0325 |  |
| FedProx(mu=0.1) vs Local-only | 3 | 0.9829 | 0.8860 | 0.0970 | 2.819 | 4.014 | 0.2500 | 0.0395 |  |

## Distribution: `iid_sub0.01`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8831 | 0.8398 | 0.0433 | 1.207 | 1.633 | 0.2500 | 0.1716 |  |

## Distribution: `iid_sub0.05`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8798 | 0.8406 | 0.0391 | 0.679 | 1.015 | 0.5000 | 0.3607 |  |

## Distribution: `non_iid_label`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs FedAvg | 2 | 0.9808 | 0.9942 | -0.0135 | -2.449 | -3.861 | 0.5000 | 0.1789 |  |
| Centralized vs FedBN | 3 | 0.9741 | 0.7506 | 0.2235 | 2.606 | 3.747 | 0.2500 | 0.0457 |  |
| Centralized vs FedProx(mu=0.01) | 2 | 0.9808 | 0.9909 | -0.0101 | -166.346 | -15.969 | 0.5000 | 0.0027 |  |
| Centralized vs FedProx(mu=0.1) | 2 | 0.9808 | 0.9832 | -0.0025 | -1.175 | -1.261 | 0.5000 | 0.3449 |  |
| Centralized vs Local-only | 5 | 0.9561 | 0.9564 | -0.0004 | -0.024 | -0.014 | 1.0000 | 0.9599 |  |
| FedAvg vs FedBN | 2 | 0.9942 | 0.7464 | 0.2478 | 2.195 | 2.973 | 0.5000 | 0.1984 |  |
| FedAvg vs FedProx(mu=0.01) | 2 | 0.9942 | 0.9909 | 0.0034 | 0.603 | 0.960 | 1.0000 | 0.5503 |  |
| FedAvg vs FedProx(mu=0.1) | 2 | 0.9942 | 0.9832 | 0.0110 | 1.447 | 2.781 | 0.5000 | 0.2893 |  |
| FedAvg vs Local-only | 2 | 0.9942 | 0.9700 | 0.0242 | 1.701 | 3.249 | 0.5000 | 0.2507 |  |
| FedBN vs FedProx(mu=0.01) | 2 | 0.7464 | 0.9909 | -0.2444 | -2.064 | -2.935 | 0.5000 | 0.2101 |  |
| FedBN vs FedProx(mu=0.1) | 2 | 0.7464 | 0.9832 | -0.2368 | -1.966 | -2.843 | 0.5000 | 0.2198 |  |
| FedBN vs Local-only | 3 | 0.7506 | 0.9701 | -0.2195 | -2.435 | -3.702 | 0.2500 | 0.0519 |  |
| FedProx(mu=0.01) vs FedProx(mu=0.1) | 2 | 0.9909 | 0.9832 | 0.0076 | 3.748 | 3.885 | 0.5000 | 0.1187 |  |
| FedProx(mu=0.01) vs Local-only | 2 | 0.9909 | 0.9700 | 0.0209 | 2.406 | 3.153 | 0.5000 | 0.1820 |  |
| FedProx(mu=0.1) vs Local-only | 2 | 0.9832 | 0.9700 | 0.0132 | 1.993 | 1.923 | 0.5000 | 0.2171 |  |

## Distribution: `non_iid_label_sub0.01`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.8936 | 0.8889 | 0.0047 | 0.143 | 0.102 | 0.7500 | 0.8277 |  |

## Distribution: `non_iid_label_sub0.05`

| comparison | n | mean A | mean B | diff | d_paired | d_unpaired | Wilcoxon p | t-test p | notes |
|---|---|---|---|---|---|---|---|---|---|
| Centralized vs Local-only | 3 | 0.9129 | 0.9039 | 0.0090 | 0.284 | 0.135 | 0.7500 | 0.6711 |  |
