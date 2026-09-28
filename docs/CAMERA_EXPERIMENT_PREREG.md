# Camera experiment -- pre-registration

Declared 2026-09-28, **before any footage of this experiment exists**: on that day no
camera capture of any kind was present on the three nodes or the desktop (checked
read-only; the only matching file on the nodes is a librealsense unit-test `.bag`). The
five-scene captures behind the v1 cross-camera result are no longer on the nodes and are
not used here. This file is committed and pushed before the first capture; the code that
implements it is committed before the capture ends. Nothing below may change after
footage exists; any change has to be disclosed in the paper and the response letter (as
for CLAUDE.md rules 7 and 9). It answers Reviewer 3's comment R3.3 and connects sensor
heterogeneity to the federated experiment.

## 1. Questions

**(a) Cross-sensor generalisation under leave-one-scene-out.** A classifier trained on
sensor X, tested on a scene it has never seen: how much does its balanced accuracy change
when that scene is recorded by sensor Y instead of X? The same-sensor, held-out-scene
result is the reference; the difference isolates the sensor shift from the scene shift.

**(b) Federated training with sensor-skewed clients.** Each client holds only its own
sensor's data -- node_a the RealSense D435if, node_b the RealSense D435i, node_c the ZED
2i. Does federated training (FedAvg, FedProx with mu = 0.01, FedBN) do better than each
client training alone (local-only), and how does it compare with pooling all three
sensors' data (centralized)?

## 2. Design

* **Scenes.** 15 to 20 scenes. A scene is one physical location with one fixed background
  and one fixed set of distractor objects; two scenes differ in location or background.
  Scene ids `s01`, `s02`, ... are assigned in capture order and never reused.
* **Paired classes.** Every scene is captured twice, fire and no-fire, with the rig, the
  background and the distractors unchanged; the only intended difference is the flame.
  The label of a capture is declared by the operator before recording starts (it is in
  the capture command), never read off the images.
* **Distractors.** Every scene contains at least one fire-like distractor (red or orange
  object, warm lamp, reflection or similar), present in the no-fire capture and left in
  place for the fire capture. Each distractor is recorded in the scene sheet.
* **Distances.** Each (scene, class) is captured at 2 or 3 rig-to-flame distances from
  {1.0, 2.0, 3.0} m (the RealSense depth range ends near 3 m); the distances of a scene are
  the same for both classes and are recorded.
* **Rig and synchronisation.** The three cameras are mounted side by side on one rig,
  pointing the same way, and record simultaneously: node_a starts every capture on all
  three nodes at a common scheduled wall-clock time (scenes are static; second-level
  synchronisation is enough; the nodes are NTP-synchronised, measured node-to-node offsets
  within about 10 ms on 2026-09-28). Frames are paired across cameras by capture, not by
  timestamp.
* **Frames.** 40 frames per (scene, class, distance, camera), at 5 frames per second
  (8 s). Every frame keeps per-frame metadata: camera serial, firmware, SDK version,
  exposure, gain, white balance where exposed, timestamp, and for the ZED 2i the depth
  mode.
* **Fire source.** A controlled flame; the source type (candle group, gas burner, tray)
  is recorded per scene and may repeat across scenes.
* **The scene is the unit of every split.** No frame of a held-out scene is ever used for
  training or model selection, in any question.

**Exclusions**, decided now and applied automatically before any model is trained: a frame
is dropped if its RGB image is missing, unreadable, or nearly constant (all channels'
standard deviation < 2 grey levels); a capture with fewer than 30 valid frames on any
camera is dropped **on all three cameras** (pairing is kept); a scene that loses a whole
class on any camera is dropped from the study. If fewer than 15 complete scenes remain,
the analysis still runs and the paper states the number. Every exclusion is logged with its
reason in the manifest.

## 3. Modalities

* **RGB -- primary.** All three sensors produce an RGB image with the same meaning, and
  the classifier of the main federated study is RGB-only, so an RGB result speaks directly
  to that study. Every frame is resized so that its shorter side is 224 px and centre-cropped
  to 224 x 224; the sensors' different resolutions and fields of view are part of the
  sensor shift and are not corrected further.
* **Depth -- secondary.** Depth is produced by different principles (active infrared stereo
  on the RealSense cameras, passive stereo with neural depth on the ZED 2i) with different
  range and invalid-pixel behaviour, so a depth difference mixes the sensor with its depth
  algorithm. It is analysed as RGB-D (depth aligned to the colour image, clipped to
  0.3-10 m, scaled to [0, 1], invalid pixels 0), in its own family, reported as secondary.
* **IR -- RealSense only, descriptive.** Only the two RealSense cameras expose an infrared
  stream (the ZED 2i does not), so IR can only be compared between node_a and node_b. It
  is reported descriptively, with intervals but no verdicts.

## 4. Common statistical machinery (as in the existing declarations)

* **Metric.** Balanced accuracy is primary (CLAUDE.md rule 8). MCC, accuracy and the other
  metrics are reported alongside, never used to rank.
* **Selection.** The declared rule (`src/evaluation/model_selection.py`): the round (or
  epoch) with the lowest validation loss, aggregated across clients weighted by
  validation-set size, earlier on ties; test metrics logged every round, never used to
  select; the reported test metrics are those of the selected round.
* **Bootstrap.** Sequence-level cluster bootstrap with the **scene** as the cluster;
  B = 10,000; 95 % percentile intervals. Every scene contains both classes by design, so
  the class-composition stratification of `src/evaluation/bootstrap.py` reduces to one
  stratum. Seeds are resampled jointly with the scenes as in `seed_paired_diff_ci`. The
  two-sided bootstrap p-value is
  p = min(1, 2 min(#{D* <= 0} + 1, #{D* >= 0} + 1) / (B + 1)).
* **Seeds.** 42, 123, 456 for every model of both questions.
* **Verdicts.** For a difference D with interval [L, U]: L > 0 -> the positive phrase,
  U < 0 -> the negative phrase, otherwise "no detectable difference". Holm correction on
  the bootstrap p-values within each family; the Holm verdict is the headline, the
  interval verdict is reported alongside. "No detectable difference" is never read as
  "equivalent" or "no effect".

## 5. Question (a): leave-one-scene-out

* **Folds.** One fold per scene s. Scenes sorted by id; the validation scene of fold s is
  the next scene in that order (the first after the last); the training scenes are the
  remaining S - 2.
* **Models.** For each source sensor X in {D435if, D435i, ZED 2i}, each fold and each seed:
  MobileNetV3-Small, ImageNet-pretrained, trained on X's frames of the training scenes,
  15 epochs, Adam 1e-3, batch 8, epoch selected on X's frames of the validation scene. The
  selected model is evaluated on scene s as recorded by **each** of the three sensors.
  Desktop GPU.
* **Out-of-fold predictions.** Every frame of every scene is predicted exactly once per
  (source sensor, seed): by the model of the fold in which its scene is held out.
* **Statistic.** For an ordered pair X -> Y (X != Y):
  D(X -> Y) = mean over seeds of [ BA(model X, sensor-Y frames) - BA(model X, sensor-X
  frames) ], each BA pooled over all scenes' out-of-fold predictions. Both sides cover the
  same scenes (the rig records every scene on every camera), so one scene resample is
  applied to both sides.
* **Family A-RGB (primary):** the six ordered pairs, Holm m = 6. Phrases: L > 0 "higher on
  the target sensor", U < 0 "lower on the target sensor", otherwise "no detectable
  difference".
* **Contrast A-hierarchy (primary, single test, no Holm).** The paper's sensor hierarchy
  predicts that a cross-technology shift costs more than a within-family one:
  K = mean(D over the four pairs involving the ZED 2i) - mean(D over D435if <-> D435i).
  Same bootstrap. Phrases: U < 0 "the cross-technology shift costs more than the
  within-family shift", L > 0 "the within-family shift costs more than the
  cross-technology shift", otherwise "no detectable difference".
* **Family A-RGBD (secondary):** the same six pairs and the same rules with RGB-D input,
  Holm m = 6.
* **IR (descriptive):** D435if <-> D435i with IR input; D and interval, no verdict.
* **Also reported, descriptive:** the full 3 x 3 matrix of pooled balanced accuracy with
  intervals, per modality; and the same data under a random frame-level split (the v1
  protocol that R3.3 criticised), to show how much scene overlap inflates the numbers.

**Interpretation rule (a).** The paper states that moving from sensor X to sensor Y costs
accuracy beyond the change of scene only for the ordered pairs whose Holm verdict in
A-RGB is "lower on the target sensor"; it reports the count of pairs under each verdict.
The sensor hierarchy is supported only if A-hierarchy reads "the cross-technology shift
costs more than the within-family shift". Depth (A-RGBD) is reported as secondary evidence
and cannot overturn an RGB verdict. A "no detectable difference" is reported as absence
of evidence, never as sensor invariance.

## 6. Question (b): federated training with sensor-skewed clients

* **Clients.** node_a holds only D435if frames, node_b only D435i, node_c only ZED 2i:
  the natural sensor skew, no relabelling or resampling.
* **Folds.** Five scene folds, fixed before data: scene ids sorted, permuted with
  `numpy.random.default_rng(20260928)`, dealt round-robin to folds 0-4. In fold f the test
  scenes are fold f, the validation scenes fold (f + 1) mod 5, the training scenes the
  other three folds -- the same scenes on every client. Each scene is a test scene in
  exactly one fold.
* **Methods.** FedAvg, FedProx (mu = 0.01), FedBN: 10 rounds, 5 local epochs, Adam 1e-3,
  batch 8, all three clients every round, on the testbed with all three nodes at
  MAXN_SUPER (its own power configuration; never pooled with any other). Local-only (each
  client alone) and centralized (the three clients' training data pooled): 50 epochs
  (= 10 x 5), same optimiser, desktop GPU. Every method, fold and seed uses the declared
  selection rule.
* **Evaluation (personalised pooled, as CLAUDE.md rule 8).** Per (method, seed) the test
  predictions of all five folds are concatenated, so every scene counts once: FedAvg and
  FedProx, the global model on every client's test split; FedBN, each client's own model
  on its own split; local-only, each client's own model on its own split; centralized, the
  model on all three clients' test splits. All methods are thus evaluated on the same
  images. Balanced accuracy pooled over them is primary; the unweighted mean over the
  three clients is secondary; both carry the scene bootstrap.
* **Family B-local (primary):** D = FL_f - local-only for f in {FedAvg, FedProx(0.01),
  FedBN}, seed-paired, Holm m = 3. Phrases: L > 0 "federation improves on training each
  sensor alone", U < 0 "federation worse than training each sensor alone", otherwise "no
  detectable difference".
* **Family B-central (primary):** D = FL_f - centralized for the same three, Holm m = 3.
  Phrases: L > 0 "higher than centralized", U < 0 "lower than centralized", otherwise "no
  detectable difference".
* RGB only; depth and IR are not used in question (b).

**Interpretation rule (b).** The paper states that federated training across the three
sensors helps a client beyond its own sensor's data only for a strategy whose Holm verdict
in B-local is "federation improves on training each sensor alone". It never describes a
strategy as matching centralized training: a B-central "no detectable difference" is
reported as such. Rankings among the three strategies are descriptive (no test between
strategies is declared).

## 7. Cost and schedule

* Capture: 15-20 scenes x 2 classes x 2-3 distances x 40 frames x 3 cameras.
* (a): 3 source sensors x S folds x 3 seeds x (RGB, RGB-D) + IR on two sensors, desktop
  GPU, small models: hours.
* (b): 5 folds x 3 seeds x 3 strategies = 45 federated runs on the testbed (each client
  holds roughly 3/5 of one sensor's frames; estimated well under an hour per run at
  MAXN_SUPER), plus 5 x 3 x 4 local-only/centralized trainings on the desktop.

## 8. Implementation (committed before the capture ends)

Capture (ZED 2i and RealSense), a node_a orchestrator that starts each capture on all three
nodes at a common time, per-frame metadata, `labels.csv`, the scene manifests (LOSO folds
and the five federated folds), the LOSO runner and its analysis, and the sensor-skew
federated configuration. The manifests are generated from the scene ids alone, never from
directory-walk order (CLAUDE.md rule 2), and are committed before any model is trained.
