# Response to Reviewers

**Manuscript:** NCAA-D-26-02211
**Journal:** Neural Computing and Applications
**Decision:** Major Revision
**Revised title:** *Empirical Evaluation of Federated Learning on Edge GPU Clusters with Heterogeneous RGB-D Sensors*

---

We thank the Editor and the three reviewers for a careful and constructive reading of our
manuscript. The reviews identified two classes of problem that we agree were serious: (i) an
evaluation protocol that was not strong enough to support the claims we made (random
image-level splitting of a frame-extracted dataset, scene-overlapping cross-sensor splits,
accuracy-only reporting, three seeds, a single hand-crafted skew, a three-round budget), and
(ii) claims stated more broadly than the evidence allowed. We have restructured the paper
around both points: the scope is now stated explicitly and early, every comparison is reported
with confidence intervals rather than point estimates, the phrase "statistically
indistinguishable" and all equivalence claims have been removed, and every conclusion is
bounded by the configuration in which it was obtained.

Alongside the manuscript revision we have implemented the full experimental infrastructure
required by the reviews (leakage audit and group-level splitting, Dirichlet partitions, a
low-data regime, the complete metric set with per-client confusion matrices, per-round
wall-clock and communication instrumentation, a five-seed matrix with hyperparameter sweeps
and a ten-round FedBN block, and a leave-one-scene-out cross-sensor protocol). The code is on
the `revision-ncaa` branch and is covered by unit tests.

### Status convention used below

- **DONE (manuscript)** — the text change is in the revised manuscript.
- **DONE (code)** — the required experimental capability is implemented and tested.
- **PENDING EXPERIMENT** — the experiment is specified, scripted and scheduled on the Jetson
  testbed; the manuscript currently carries a clearly marked placeholder
  (`\todo{...}`, `XX.X`, `X.XXX`) at the exact location where the number will go.

**Note on placeholders.** No placeholder in the submitted revision is a real or estimated
measurement. Every cell awaiting a pending run is typeset in red so that it cannot be mistaken
for a result. The final submission will contain no red text.

**Note on the v1 numbers.** The results of the original submission were obtained under a
random *image-level* split. Because the revised protocol changes the partition, the old and
new numbers are not comparable. We therefore retain the v1 results in a clearly labelled
subsection ("Results Under the Image-Level-Split Protocol") and present the audited protocol
separately, so that the effect of the protocol change is itself visible to the reader.

**Note on model selection.** The revision declares its model-selection rule before the
federated runs (new Section III, "Model Selection and Use of the Test Split"): the reported
model is the one from the round with the lowest validation loss, aggregated across clients
weighted by client validation-set size; ties are broken toward the earlier round. Test
metrics are computed every round for logging but never influence model selection, which
uses the aggregated validation loss only; the reported test metrics are those of the
selected round. No maximum over rounds is reported. The centralized and local-only
references are re-run under the same rule.

**Note on the leakage guarantee (raised by us).** The group-level partition is built so that
no validation or test image lies within Hamming distance 8 of any training image under the
64-bit dHash, and the audit confirms leak rate 0 at that threshold for all 15 partitions. We
want to be explicit that this guarantee is relative to that definition. Consecutive segments of
the same flight can fall just above the threshold. We therefore measured, for every held-out
image, the distance to its nearest training image under dHash and under an independent second
hash (pHash), and report it in the revised manuscript (new table `tab:nn_distance`):

| Partition | held-out images | dHash ≤ 8 | dHash 9–12 | pHash ≤ 8 | pHash ≤ 10 |
|---|---|---|---|---|---|
| IID | 14,311 | 0 | 735 (5.1 %) | 335 (2.3 %) | 567 (4.0 %) |
| Label skew | 14,397 | 0 | 712 (4.9 %) | 277 (1.9 %) | 480 (3.3 %) |
| Dirichlet α = 0.1 | 14,360 | 0 | 1,030 (7.2 %) | 359 (2.5 %) | 682 (4.7 %) |
| Dirichlet α = 0.5 | 14,400 | 0 | 635 (4.4 %) | 265 (1.8 %) | 535 (3.7 %) |
| Dirichlet α = 1.0 | 14,399 | 0 | 1,045 (7.3 %) | 293 (2.0 %) | 466 (3.2 %) |

For comparison, 99.6 % of the held-out images of the original image-level split had a training
near-duplicate within dHash distance 8. The residual cases are concentrated in a few pairs of
adjacent segments of the same flight. We evaluated coarser groupings (dHash ≤ 8 merged with
pHash ≤ 3, 4 or 6): they shrink the residual but never remove it, because each merge exposes
the next band of distances, and they worsen the balance of the partitions and the dominance of
single sequences in the held-out sets. We kept the partition and instead **pre-registered a
clean-subset analysis**: on 19 September 2026, before any clean-subset metric was computed
(at that time the full-set results of the centralized and local-only references and one
federated run existed), we fixed and published the rule *exclude every held-out image within
dHash ≤ 12 or pHash ≤ 10 of a training image of the same partition*, together with the
excluded image lists of all 15 partitions and their checksums (repository commit 873b389,
`analysis/leakage/clean_subset/`). The rule removes 950–1,466 held-out images (6.6–10.2 %)
from the five full partitions. Every headline metric is reported on all held-out images and on
the clean subset. The rule will not be changed after results are seen; any change would be
disclosed.

Confidence intervals of held-out metrics are now **sequence-level (cluster) bootstrap**
intervals over seeds and held-out sequences, because a few sequences can dominate a node's
held-out set (for example, 79 % of node C's IID validation Fire images and 69 % of its test
Fire images each come from a single sequence). The revised Limitations section states this
partition property explicitly.

**Note on the testbed network.** The testbed was disassembled and reassembled between the
original submission and the revision. The v1 experiments ran over WiFi (IEEE 802.11ac); all
revision experiments run on the same three Jetson nodes connected by wired Gigabit Ethernet
through a single switch, as now stated in Section III-A and Table I. Because both the network
and the data partition changed, the revision wall-clock timings supersede, rather than extend,
the v1 timings, and the two are not compared with each other.

---

## Reviewer 1

### R1.1 — "Please add related references before Eq. 1 to Eq. 5b."

**Response — DONE (manuscript).** The federated objectives are now collected in a new
methodology subsection, *III-H "Federated Optimisation Objectives"*, in which every equation is
introduced together with its originating reference. (Because the revision adds equations to
earlier subsections, the equations numbered (1)–(5b) in the original submission are numbered
(5)–(10) in the revision; the mapping is given below.)

| Equation | Content | Citation added |
|---|---|---|
| (5) | Global federated objective `F(w) = Σ (n_k/n) F_k(w)` | McMahan et al., AISTATS 2017 |
| (6) | Local empirical risk `F_k(w)` | McMahan et al., AISTATS 2017 |
| (7) | FedAvg aggregation | McMahan et al., AISTATS 2017 |
| (8) | FedProx proximally regularised local objective | Li et al., MLSys 2020 |
| (9) | FedBN aggregation of non-normalisation parameters | Li et al., ICLR 2021 |
| (10) | FedBN client-local normalisation parameters | Li et al., ICLR 2021 |

The remaining method equations introduced in the revision are likewise attributed at the point
of use: the near-duplicate/leakage criterion to Barz and Denzler (*J. Imaging*, 2020), the MCC
definition to Chicco and Jurman (*BMC Genomics*, 2020), and the Dirichlet partition to the
recent benchmarking literature cited in R1.2.

### R1.2 — "Please cite new 2025–26 references."

**Response — DONE (manuscript).** A new related-work subsection, *II-D "Recent Benchmarking of
Non-IID Federated Learning (2025–2026)"*, was added, and the edge-FL subsection was extended.
All added references were verified against the publisher record (DOI or arXiv identifier given
in the bibliography):

- Reis, *Applied Sciences* 15(12):6452, 2025 — Edge-FLGuard on Raspberry Pi / Jetson Nano.
  (This also replaces the incomplete placeholder citation of the original submission.)
- Banerjee, Chandrashekar, Eswar and Simmhan, Euro-Par 2025 (LNCS 15900, pp. 264–278) — federated learning under a global
  energy budget on heterogeneous edge accelerators.
- Zhang, Chen, Lin, Chen and Zhao, IEEE ICASSP 2025 (arXiv:2501.01850) — clustered FL for heterogeneous data.
- Seo, Catak and Rong, NIKT 2024 (arXiv:2502.00182, 2025) — experimental IID-to-non-IID study;
  cited in support of our decision to report intervals rather than single-seed comparisons.
- Domini, Aguzzi and Viroli, arXiv:2503.20618, 2025 — ProFed benchmark for non-IID partitioning.
- Borazjani, Abdisarabshali, Khosravan and Hosseinalipour, *IEEE Trans. Artificial Intelligence*,
  7(9):5045–5060, 2026 (arXiv:2503.14553) — embedding-based redefinition of non-IID for vision tasks.
- Mreish et al., *Sensors* 25(23):7314, 2025 — MFedBN, a FedBN follow-up; cited where we bound
  our own FedBN conclusions.
- Prashanthi, Kesanapalli and Simmhan, *Proc. ACM Meas. Anal. Comput. Syst.* 6(3), 2022 —
  Jetson training characterisation (centralised counterpart of our measurements).
- Ibrahim, Hussein, Guinovart and Qaraad, *Arch. Comput. Methods Eng.* 32(8), 2025 — see R5.6.
- Dede et al., *Engineering Reports* 7(2):e70027, 2025, and Yang et al., *PLOS ONE*
  20(5):e0323322, 2025 — wavelet-CNN hybrids; see R1.3.
- Shamsoshoara et al., *Computer Networks* 193:108001, 2021 — the primary FLAME dataset paper,
  which the original submission cited only through its Kaggle mirror.

### R1.3 — "Future work: a combined (hybrid) method with wavelet transformations for GPU edge clusters."

**Response — DONE (manuscript).** A dedicated future-work subsection, *VI-A "Future Work"*, now
opens with the paragraph **"Hybrid wavelet-transform feature extraction for edge GPU clusters."**
It argues the specific reason this is attractive in a *federated* edge setting: a wavelet front
end carries no learnable parameters, so it reduces the spatial resolution the backbone must
process and shrinks the payload exchanged every round without adding anything to the
aggregation cost, and its multi-resolution sub-bands give a natural granularity for
transmitting only changed coefficients. The paragraph cites Dede et al. (2025) for
wavelet-domain classification of high-resolution imagery and Yang et al. (2025) for a 2-D DWT
front end on a MobileNetV3 backbone (a 25 % parameter reduction on the same backbone family we
use), and states that we intend to evaluate the hybrid on the same edge GPU cluster, measured
on the time and communication axes introduced in Section III-M. A forward reference to this
paragraph is also placed in the metaheuristic related-work subsection.

---

## Reviewer 3

### R3.1 — "The heterogeneous RGB-D sensors are not connected to the main FL experiment (FL uses FLAME RGB partitions; depth is not used by the classifier). Make the distinction clear."

**Response — DONE (manuscript).** We accept this criticism; the original framing implied that
sensor heterogeneity was acting inside the federated runs, which it was not. The revision makes
the separation explicit in four places:

1. **Title** — changed to "Empirical Evaluation of Federated Learning on Edge GPU Clusters with
   Heterogeneous RGB-D Sensors", which describes the testbed rather than claiming multimodal FL.
2. **Abstract** — now states that "the federated experiments train a single MobileNetV3-Small
   backbone on RGB partitions of the FLAME dataset ... the heterogeneous RGB-D sensors are
   characterised separately ... and the depth modality is *not* an input of the federated
   classifier."
3. **Introduction, Section I-A "Scope of this study"** — the fifth scope bullet, "Role of the
   RGB-D sensors", states the same point and forward-references the two relevant sections.
4. **New methodology subsection III-B, "Relationship Between the Sensor Testbed and the
   Federated Experiment"** — defines Experiment I (federated optimisation; RGB only;
   heterogeneity is the heterogeneity of the *partition*) and Experiment II (cross-sensor
   generalisation; centralised; scene-controlled), and closes with: "Statements about sensor
   heterogeneity in this paper derive from Experiment II; statements about FL strategy behaviour
   derive from Experiment I. We do not attribute federated convergence effects to the cameras."

The claim that this is "the first multimodal FL study with real depth cameras" has been removed
from the contribution list. The empty v1 placeholder subsections for a modality ablation,
tegrastats resource profiling and network-constraint sensitivity have been removed; the revised
Limitations subsection states explicitly that these three measurements are out of scope and
they are listed as future work.

### R3.2 — "FLAME is randomly split: related frames from the same acquisition sequence may be in train and test. Use a sequence/source-level split."

**Response — DONE (code), DONE (manuscript), DONE (leakage audit), PENDING EXPERIMENT (accuracy
under the new protocol).** We agree, and this is the change with the largest potential effect on
our reported accuracies.

*Audit result (Table VII, Section IV-B).* We ran the near-duplicate audit on the full FLAME tree
(47,992 images, 64-bit difference hash, τ = 8 bits, chosen from a sweep over τ ∈ {4,…,12}).
The reviewer's suspicion is confirmed in the strongest form: 265 non-trivial near-duplicate
groups cover 99.73 % of the images, the largest group holds 4,924 frames of one sequence, and
99.06 % of consecutive frame numbers fall into the same group, i.e. the groups recover the
acquisition sequences. Under the image-level split of the original submission **99.6 % of all
validation and test images have a near-duplicate in the training split of some node**, and
235 (IID) / 219 (label skew) groups were spread over several nodes. We state in the revised
manuscript that the v1 accuracies therefore largely measure recognition of sequences seen in
training. A further finding required a protocol decision: 22 groups (20,006 images) contain
both fire and no-fire frames, because a FLAME sequence is a video in which the fire appears and
disappears in a fixed scene. We keep such groups whole regardless of label (a group is a
sequence), since splitting them by label re-creates the leakage. Under the group-level protocol
an independent audit of all 15 written partitions (IID, label skew, Dirichlet α ∈ {0.1, 0.5, 1},
and their 5 % / 1 % low-data variants) gives a leak rate of exactly 0 and no group spanning two
nodes. Because the units of assignment are whole sequences, the quotas of each design are filled
with a largest-first greedy rule; the achieved per-node counts are reported in a new table
(Table VIII in the revision) rather than the nominal ones.

*Manuscript.* A new methodology subsection III-D, **"Sequence-Level Data Splitting and Leakage
Audit"**, explains why a random image-level split of a frame-extracted dataset measures partly
memorisation (citing Barz and Denzler, 2020, for the same effect on standard benchmarks),
defines near-duplicate grouping by 64-bit perceptual hashing with a Hamming threshold τ
(Eq. 1), makes the near-duplicate *group* the atomic unit of assignment — a group never crosses
a node boundary or a train/val/test boundary — and defines the leakage rate (Eq. 2) that we
report per node and pooled.

*Code.* `scripts/analyze_flame_leakage.py` performs the audit and emits the group file;
`src/data/data_splitter.py --group_file` performs the group-safe partitioning (with the
no-flags path pinned bit-identical to the original splitter by a regression test).

*Results.* Section IV-B contains three tables: Table VII (leakage audit: dataset statistics,
threshold sweep, and leak rate L / groups spanning several nodes for both protocols — **filled
with the measured values**), Table VIII (achieved per-node counts of every group-level
partition — **filled**), and the protocol-effect table (final accuracy under both protocols).
**PENDING EXPERIMENT** — only the group-level accuracy column of the protocol-effect table is a
placeholder until the `rev_*` runs have been executed. All v1 results are labelled "image-level
split" so that the reader can see which protocol produced which number.

Because the partition changes, the previously collected seeds cannot be reused; all five seeds
of every configuration will be re-run under the group-level split.

### R3.3 — "The cross-sensor experiment splits frames of the same five scenes between train and test, so 100 % cross-camera accuracy may reflect scene similarity. Leave-one-scene-out is needed."

**Response — DONE (manuscript), PENDING EXPERIMENT.** We accept this entirely. The near-perfect
cross-camera accuracy of the original submission has been **removed** from the manuscript,
because it cannot be separated from scene overlap.

A new methodology subsection III-N, **"Leave-One-Scene-Out Cross-Sensor Protocol"**, states the
problem explicitly and defines the replacement protocol: for each held-out scene *s*, train on
the frames of the remaining *S*−1 scenes of the source camera and test on scene *s* of the
target camera, reporting the mean over the *S* folds with a confidence interval. The same folds
are used for every (source, target) camera pair, so the within-camera diagonal (same camera,
held-out scene) is a scene-shift-only reference, and the sensor effect is isolated as the
row-wise gap (Eq. 22). Results Section IV-J holds the corresponding 3×3 table with the diagonal
defined as the reference. A script (`scripts/cross_sensor_loso.py`) implements the folds.
**PENDING EXPERIMENT** — the table is a placeholder; the captures are on the Jetson nodes.

### R3.4 — "The task is near saturation (~99 %): add a more demanding condition (fewer local samples, stronger heterogeneity)."

**Response — DONE (code), DONE (manuscript), PENDING EXPERIMENT.** We agree that at ~99 % the
comparison has almost no resolution, and we say so explicitly in the scope statement
(Section I-A: "the task is close to saturation for a centralised learner, which limits the
resolution of accuracy-based comparisons"). Two harder conditions were added:

- **Low-data regime (new Section III-F, results Section IV-E).** Each node's *training* split is
  subsampled at group level, stratified per class, to ρ ∈ {1.00, 0.05, 0.01} of its size
  (Eq. 4), with validation and test untouched so that all regimes are evaluated on identical
  data. At ρ = 0.01 a node holds on the order of 10² training images, i.e. a regime where a
  federated model should be clearly better than a local one. The reported quantity is the
  federated-minus-local gap as a function of ρ.
- **Stronger heterogeneity (new Section III-E, results Section IV-D).** Dirichlet label skew
  with α ∈ {0.1, 0.5, 1.0}; α = 0.1 concentrates each class on few nodes. A node left with very
  few images at α = 0.1 is treated as the intended difficulty and is reported, not removed.

**PENDING EXPERIMENT** — Tables XI and XII are placeholders.

*Reference conditions already executed (desktop GPU, Jetson hyperparameters).* The centralized
and local-only baselines of every new condition have been run under the audited protocol (62
runs: five seeds for IID and label skew, three for each Dirichlet and low-data partition).
Removing the sequence leakage moves the centralized reference from 99.7 % to 93.4 ± 1.2 % (IID)
and 94.3 ± 1.6 % (label skew) test accuracy and the local-only reference to 86.9 ± 1.4 % and
95.5 ± 0.8 %, so the task is no longer saturated (Section IV-C, Table IX). Under label skew the
local-only accuracy is now *higher* than the centralized one (95.5 vs. 94.3 %) while balanced
accuracy (88.6 vs. 94.7 %) and MCC (0.81 vs. 0.88) go the other way — accuracy alone would rank
the wrong method first, which is exactly the metric issue the reviewer raised, now visible in
our own data. Under Dirichlet skew the local-only balanced accuracy falls to 56.2 % at α = 0.1
(two clients never see a no-fire frame), 82.3 % at α = 0.5 and 94.4 % at α = 1.0 (Table XI). In the
low-data regime the local-only reference loses little under IID and about eight points under
label skew when a node is reduced to ~112 training frames, because the frame-level subsample
still covers most of a node's sequences; we state this conservative reading in the manuscript.
The federated rows of these tables await the Jetson runs.

**Note on the size of the reference gap (raised by us).** Under the declared model-selection
rule the centralized-to-local-only gap under IID is **6.5 percentage points** (93.4 % vs.
86.9 %), not the 12.6 points that the same runs show when the final epoch is reported
(90.9 % vs. 78.3 %). We report the 6.5-point figure, because selecting the epoch by validation
loss is the protocol we declared in Section III-F and apply to every condition, federated and
reference alike. We draw attention to the difference because it is a finding about reporting
protocol: the effect is asymmetric, since the local-only models are the unstable ones across
epochs — on node C the validation accuracy of the local-only IID baseline moves between 0.96
and 0.42 in consecutive epochs while its training loss stays below 0.05 — so a final-epoch
snapshot penalises them far more often than it penalises the pooled model, and final-epoch
reporting therefore more than doubles the apparent advantage of pooling the data. Both
protocols are kept in the repository under separate labels (`group` and `group_final_epoch` in
`analysis/runs.csv`); the numbers above and in the manuscript are all produced by
`scripts/analyze_results.py` and `scripts/export_latex_tables.py`, and the values typed into
the manuscript tables are pinned to `analysis/summary_table.csv` by
`tests/test_paper_numbers.py`. The narrower gap does not weaken the case for the audited
protocol: it is the honest size of the band in which federation has to show its value, and it
is measured under a rule fixed before the federated runs.

### R3.5 — "Report balanced accuracy, macro-F1, sensitivity, specificity, and per-client confusion matrices."

**Response — DONE (code), DONE (manuscript), PENDING EXPERIMENT.** A new methodology subsection
III-K, **"Evaluation Metrics"**, states why accuracy alone is inadequate when client class
priors differ by construction, and defines every metric with an equation: accuracy (11),
sensitivity/recall (12), specificity (13), precision (14), balanced accuracy (15), F1 and
macro-F1 (16), MCC (17, with Chicco and Jurman cited for why it is preferred over accuracy and
F1 on imbalanced data) and ROC-AUC (18, Mann–Whitney form with ties at one half). Each metric is
computed at three levels — per client, num-examples-weighted global, and pooled from the summed
confusion matrix — and per-client confusion matrices are stored every round.

Results Section IV-C contains the global metric table (Table IX, full-width), the per-client
metric table (Table X) and a per-client confusion-matrix figure for rounds 1 and final under
label skew, which is the most direct evidence of *which* class each drifting client fails on.
We also now state in the Discussion that "aggregate accuracy is the least informative of the
metrics we report for this task". **PENDING EXPERIMENT** — the metric tables are placeholders;
the metric implementation (`src/evaluation/metrics.py`) is complete and is validated cell by
cell against scikit-learn in the unit tests.

### R3.6 — "Only three rounds: FedBN conclusions must stay limited; the claim that FedBN should be reserved for many more rounds is untested."

**Response — DONE (manuscript), PENDING EXPERIMENT.** The untested claim has been deleted. The
revision handles FedBN as follows:

- The scope statement (Section I-A) lists the three-round budget as a boundary of the study and
  says that the separate ten-round block exists "solely to probe whether the three-round ranking
  of FedBN persists; no claim is made about asymptotic behaviour beyond ten rounds".
- Section III-H explains *structurally* why a very small budget is hard for FedBN: Eq. (10) means
  FedBN produces K personalised models, each client is evaluated with its own normalisation
  parameters, and at least one local pass is needed before those statistics are meaningful.
- Section IV-G opens with an explicit non-claim: "We deliberately do *not* conclude from this
  that FedBN is unsuitable for this task, nor that it would necessarily become competitive given
  more rounds: both statements exceed the evidence of a three-round experiment."
- The ten-round FedBN block (with a matched FedAvg reference) is implemented in the experiment
  matrix. Its caption instructs that if the R10 difference has a CI containing zero, the
  conclusion is that ten rounds are insufficient to separate the methods — *not* that they are
  equivalent.
- The Conclusion states that FedBN behaviour "is reported for the three- and ten-round budgets
  that were run, and no claim is made about its behaviour at larger budgets", and the
  generalisability subsection lists the FedBN ranking as configuration-specific (tied to the
  normalisation structure of MobileNetV3-Small and to the executed budget).

**PENDING EXPERIMENT** — Table XIV (`long_horizon_fedbn` block).

### R3.7 — "FedProx has better first-round accuracy but much more wall-clock time: compare by elapsed time and communication cost too."

**Response — DONE (code), DONE (manuscript text), PENDING EXPERIMENT (timings).** A new methodology
subsection III-M, **"Time- and Communication-Normalised Comparison"**, states that comparing at
equal round counts hides the fact that a round costs different amounts under different
strategies, describes the per-round instrumentation (client fit/evaluation wall-clock time,
payload bytes up and down; server elapsed time and cumulative volume), and gives the
communication model B(R) = 2·R·K·|w| (Eq. 21), reduced for FedBN by the parameters withheld from
aggregation.

**PENDING EXPERIMENT** for the numbers. The original submission timed the v1 runs over WiFi and
under the image-level split. The testbed has since been reassembled on wired Gigabit Ethernet
and the partition is now group-level, so the v1 timings (including the v1 FedProx/FedAvg ratio)
are superseded rather than extended, and are no longer reported (see the note on the testbed
network above). Results Section IV-H keeps the argument the reviewer asks for — "an improvement
that is obtained per *round* is not necessarily an improvement per *second*" — and the identical
communication volume of FedAvg and FedProx (110.3 MB over three rounds with K = 3, from Eq. 21,
because FedProx changes the local objective, not the payload). The measured time ratio, and
whether FedAvg completes more than one round within FedProx's first-round time, are marked
placeholders to be filled from the revision runs. The practical-implications paragraph states
the FedProx recommendation conditionally on those timings ("when rounds are expensive
(scheduled, bandwidth-limited or manually supervised aggregation)"). Table XV is now generated
by `scripts/export_latex_tables.py` from the revision runs only, with a measured communication
column, and the time/accuracy figure becomes a three-panel figure (accuracy vs. round, vs.
elapsed seconds, vs. cumulative MB) drawn from the revision runs only; v1 runs are not placed on
the time or communication axes.

### R3.8 — "With three seeds, 'statistically indistinguishable' is unsupported; interpret very large effect sizes with care; add seeds for the central FedAvg–FedProx comparison."

**Response — DONE (manuscript), DONE (code), PENDING EXPERIMENT.** We agree and have removed
the offending wording entirely.

- **Wording.** The phrase "statistically indistinguishable" no longer appears anywhere in the
  manuscript, and Section III-L ("Statistical Analysis") states the policy: "we do not interpret
  a non-significant test as evidence of equivalence, and we do not use the phrase 'statistically
  indistinguishable'; we report the confidence interval of the difference and state explicitly
  when it contains zero."
- **Confidence intervals.** All aggregate numbers are now reported as mean ± std with a 95 %
  *t*-interval of the mean (Eq. 19) and the per-cell *n*. The v1 table (Table V) carries these
  intervals and states n = 3 (n = 2 for the 3-node FedAvg IID cell), which makes the imprecision
  of the old numbers visible; two intervals that exceed 100 % are marked as truncated.
- **Effect sizes.** Section III-L adds an explicit caution: "with n ≤ 5, s_Δ is itself estimated
  from few observations, so d has a wide sampling distribution and a large |d| can arise from an
  unusually small s_Δ rather than from a large true effect." Section IV-I applies it to the
  concrete case the reviewer would have found most suspicious: d = −7.30 for FedProx 0.01 vs.
  local-only under IID corresponds to a mean difference of only −0.31 percentage points, and is
  presented as an artefact of a small paired standard deviation, not as an important difference.
- **Multiplicity.** Section IV-I now reports that only three of fifteen pairs per distribution
  reach p < 0.05 uncorrected and that **none survives Bonferroni correction** (α = 0.0033), and
  that the Friedman omnibus is computed on only n = 2 seeds complete across all six methods and
  is therefore uninformative. The v1 statistics are thus presented as non-confirmatory.
- **More seeds.** The experiment matrix contains a five-seed block covering FedAvg and
  FedProx(μ = 0.01) under IID and label skew, plus the centralised and local-only references.
  **PENDING EXPERIMENT** — Table XVI will be regenerated at n = 5 under the group-level split,
  with the 95 % CI of each paired difference added.

---

## Reviewer 5

### R5.1 — "Scope is narrow (3 clients, 1 binary dataset, 1 backbone, 3 rounds). Justify and delimit which conclusions generalise."

**Response — DONE (manuscript).** Rather than defend the scope implicitly, we now declare it
before any result is presented and separate the findings by how far they travel.

- **New Section I-A, "Scope of this study"**, with five explicit bullets: cluster size (three
  clients, full participation — "nothing in this paper speaks to cross-device FL with hundreds
  or thousands of intermittently available clients"); task and dataset (one binary task, near
  saturation); backbone (one architecture, with the caveat that strategy rankings may depend on
  the amount and placement of normalisation layers); communication budget; and the role of the
  RGB-D sensors.
- **New Section V-B, "Which Conclusions Generalise"**, splits the findings into three groups:
  *expected to generalise* (a proximal constraint reduces first-round damage under strong label
  skew; per-round and per-second orderings can differ on accelerated edge devices; frame-extracted
  video datasets require group-level splitting — a property of the data, not of the method);
  *specific to this configuration* (the numerical size of the first-round penalty; the FedBN
  ranking; the size of the FedProx time overhead, which follows from the 8 GB memory constraint and our
  CPU-resident implementation of w^t; and all absolute accuracies); and *not addressed at all*
  (cross-device federations, partial participation, dropout, secure aggregation and DP overheads,
  non-vision modalities, multi-class and dense prediction).
- **Section V-C, "Limitations"**, was expanded from three sentences to eight numbered
  limitations, and the abstract and conclusion were rewritten to state the boundaries directly.
- The paper is now framed as "measurement and protocol rather than a new algorithm", which is
  what it is.

### R5.2 — "Three seeds is insufficient: more runs and confidence intervals."

**Response — see R3.8.** n = 5 seeds throughout the revised protocol, 95 % *t*-intervals on
every aggregate, paired effect sizes with an explicit caution about small n, Wilcoxon and paired
t-tests, a Friedman omnibus, and Bonferroni thresholds reported alongside raw p-values
(Section III-L). **PENDING EXPERIMENT** for the n = 5 numbers.

### R5.3 — "Accuracy alone is insufficient: precision, recall, F1, balanced accuracy, MCC and ROC-AUC, globally and per client."

**Response — see R3.5.** All requested metrics are defined with equations in Section III-K and
reported globally *and* per client, together with per-client confusion matrices. MCC and ROC-AUC
were specifically included, with Chicco and Jurman (2020) cited for the reason MCC is preferred
on imbalanced data. **PENDING EXPERIMENT** for the values.

### R5.4 — "Check the conclusions under different degrees and types of non-IID (not one manual label skew)."

**Response — see R3.4.** Section III-E now describes three partitioning regimes: IID, the
original hand-constructed label skew (retained for continuity with v1), and Dirichlet label skew
at α ∈ {0.1, 0.5, 1.0} (Eq. 3), with the realised per-node class counts to be reported from
`split_stats.json` so that the achieved skew is documented rather than asserted. Section IV-D
reports final balanced accuracy per α with a figure showing the strategy ranking as a function
of α, and the caption requires an explicit statement of whether the ranking is preserved. We
also cite the recent literature arguing that label-based skew is an incomplete model of
heterogeneity for vision tasks (Borazjani et al., 2026) and note embedding- and proximity-based
partitioning as a further direction. **PENDING EXPERIMENT.**

### R5.5 — "Hyperparameter sensitivity (batch 8, Adam 1e-3, 5 local epochs, μ only 0.01/0.1): a sensitivity analysis is needed."

**Response — DONE (code), DONE (manuscript), PENDING EXPERIMENT.** A new methodology subsection
III-J, **"Hyperparameter Sensitivity Protocol"**, sweeps one factor at a time about the default
point: μ ∈ {0.001, 0.01, 0.05, 0.1, 0.5} (two orders of magnitude, versus two values in v1),
local epochs E ∈ {1, 2, 5}, learning rate η ∈ {10⁻⁴, 10⁻³} and communication budget
R ∈ {3, 10}. Each cell uses the same seeds as the corresponding default cell, so differences are
paired.

Two honest caveats are stated in the text rather than hidden: batch size is fixed at 8 because it
is a *hardware* constraint of the 8 GB Jetson Orin Nano and not a free design choice (listed in
Limitations), and a one-factor-at-a-time grid cannot resolve interactions between μ, E and η
(also listed in Limitations, and connected to R5.6 below). Results Section IV-F presents the
sensitivity table and a three-panel figure, with both final and round-1 values, since μ and E
affect the early rounds far more than the final round. **PENDING EXPERIMENT.**

### R5.6 — "Related work should acknowledge metaheuristic hyperparameter tuning (e.g. the study 'Convolutional neural networks hyperparameters tuning')."

**Response — DONE (manuscript).** We thank the reviewer for the pointer. A new related-work
subsection II-E, **"Metaheuristic and Hybrid Hyperparameter Optimisation for CNNs"**, was added.
It cites the study named by the reviewer — E. Tuba, N. Bačanin, I. Strumberger and M. Tuba,
"Convolutional neural networks hyperparameters tuning," in *Artificial Intelligence: Theory and
Applications* (Studies in Computational Intelligence), Springer, 2021, pp. 65–84,
doi:10.1007/978-3-030-72711-6_4 — together with the recent comprehensive review by Ibrahim,
Hussein, Guinovart and Qaraad (*Archives of Computational Methods in Engineering* 32(8), 2025).

The paragraph explains why this literature matters specifically for our setting rather than
citing it in passing: on a constrained device a single configuration evaluation costs hours, so
the sample efficiency of the search dominates the total cost, and in FL the search space is
larger than in the centralised case because it also contains federation-level parameters (local
epochs per round, proximal strength, participation). We state plainly that our own sensitivity
analysis is a structured grid, not a metaheuristic search, and identify automated population-based
search over the joint local/federation space as a continuation of this work (referenced again in
the Future Work section, where it is combined with the wavelet hybrid of R1.3).

### R5.7 — "Change the title, e.g. 'Empirical Evaluation of Federated Learning on Edge GPU Clusters with Heterogeneous RGB-D Sensors'."

**Response — DONE (manuscript).** Adopted verbatim. The title is now *"Empirical Evaluation of
Federated Learning on Edge GPU Clusters with Heterogeneous RGB-D Sensors"*. The running head and
the keyword list were updated accordingly ("edge GPU cluster", "data leakage", "empirical
evaluation" added), and the abstract now describes the work as "an empirical, deliberately
narrow-scope evaluation", consistent with the new title.

### R5.8 — "Moderate the FedBN conclusions (three-round budget)."

**Response — see R3.6.** All FedBN claims are now bounded by the executed budget, the untested
"reserve FedBN for many-round scenarios" claim was removed, a ten-round block was added to test
it, and the FedBN ranking is listed in Section V-B among the findings that must not be
extrapolated.

---

## Summary of manuscript changes

| Section | Change |
|---|---|
| Title, running head, keywords | New title per R5.7 |
| Abstract | Scope declared; sensor/FL distinction stated; CI-based wording; FedBN claims bounded |
| I. Introduction | New Section I-A "Scope of this study" (5 bullets); contribution list rewritten around protocol and measurement; overreaching "first multimodal FL" claim removed |
| II-A Related work (edge FL) | Extended with verified 2025–26 references; placeholder citation replaced |
| II-D (new) | Recent benchmarking of non-IID FL (2025–2026), incl. evaluation-hygiene literature |
| II-E (new) | Metaheuristic and hybrid hyperparameter optimisation for CNNs (R5.6, R1.3) |
| III-B (new) | Relationship between the sensor testbed and the federated experiment (R3.1) |
| III-D (new) | Sequence-level data splitting and leakage audit, Eqs. (1)–(2) (R3.2) |
| III-E (new) | Non-IID partitions: manual skew + Dirichlet, Eq. (3) (R3.4, R5.4) |
| III-F (new) | Low-data regime, Eq. (4) (R3.4) |
| III-H (new) | Federated optimisation objectives, Eqs. (5)–(10) with original citations (R1.1) |
| III-J (new) | Hyperparameter sensitivity protocol (R5.5) |
| III-K (new) | Evaluation metrics, Eqs. (11)–(18) (R3.5, R5.3) |
| III-L (new) | Statistical analysis, Eqs. (19)–(20) (R3.8, R5.2) |
| III-M (new) | Time- and communication-normalised comparison, Eq. (21) (R3.7) |
| III-N (new) | Leave-one-scene-out cross-sensor protocol, Eq. (22) (R3.3) |
| IV-A | v1 results retained, relabelled "image-level split", CIs and per-cell n added |
| IV-B … IV-J (new) | Leakage audit; full metric set and per-client confusion matrices; Dirichlet; low-data; sensitivity; extended FedBN budget; time/communication; statistics; LOSO cross-sensor |
| V-A | Practical implications rewritten around the time-normalised finding |
| V-B (new) | Which conclusions generalise (R5.1) |
| V-C | Limitations expanded to eight items |
| VI | Conclusion rewritten; new VI-A Future Work (wavelet hybrid + scene-independent multi-site collection) |
| References | 15 new verified entries with DOI/arXiv identifiers |

## Experiments still to be executed

The following runs are scripted and resumable
(`scripts/print_revision_commands.py`); each corresponds to a placeholder in the manuscript.

| Block | Reviewer comment | Manuscript location |
|---|---|---|
| Seed extension (5 seeds, group-level split) | R3.8, R5.2, R3.5 | Tables V, VIII, IX, X, XVI |
| Dirichlet sweep (α = 0.1, 0.5, 1.0) | R3.4, R5.4 | Table XI, Fig. (Dirichlet) |
| Low-data (ρ = 0.05, 0.01) | R3.4 | Table XII |
| μ grid, local epochs, learning rate | R5.5 | Table XIII, Fig. (sensitivity) |
| 10-round FedBN + matched FedAvg | R3.6, R5.8 | Table XIV |
| Leave-one-scene-out cross-sensor | R3.3 | Table XVII |

*(Table numbers refer to the compiled revised manuscript and should be re-checked after the
final compile.)*

---

We believe the revision addresses every point raised. We are grateful to the reviewers: the
protocol problems they identified (near-duplicate leakage and scene overlap in particular) would
have undermined the paper's conclusions, and correcting them has made the study considerably
more defensible.
