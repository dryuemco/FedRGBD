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
  MAXN_SUPER (its own power configuration; never pooled with any other; power-mode lock,
  determinism gate and results location fixed by Amendment 1, section 9). Local-only (each
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

## 9. Amendment 1 (2026-09-29, before any footage exists)

Declared 2026-09-29 and committed and pushed before the first capture. On that day no
footage of this experiment existed: `data/raw/camera`, `data/processed/camera_fold*` and
`data/splits_camera` were absent on all three nodes (checked read-only), and the desktop
held only the blank scene-sheet template. So this amendment is part of the
pre-registration, not a change after data. It adds execution conditions and changes no
question, split, model, metric, family, verdict phrase or interpretation rule above.

* **Sensor to node.** Each camera is captured on its own node and its frames stay
  there for the federated runs: D435if on node_a, D435i on node_b, ZED 2i on node_c. The
  federated client on a node trains only on the frames that node's camera recorded
  (`camera_fl_prepare.py --nodes <node>`). No node receives another camera's frames for
  question (b).
* **Where each training runs.** The 45 federated runs of question (b) run on the testbed
  with all three nodes at MAXN_SUPER. Nothing else moves: question (a) (leave-one-scene-out)
  and the local-only and centralized baselines of question (b) stay on the desktop GPU as
  in sections 5 and 6, on copies of the same frames.
* **Power-mode lock.** As for block 5b (`maxn_long_horizon`): the camera block declares
  `power_config: maxn`. `scripts/run_matrix.py` refuses to start unless `nvpmodel -q`
  reports MAXN_SUPER on all three nodes. It reads the modes again after every run, and a
  run during which any mode changed is moved out of `results/`, never counted.
* **Determinism gate (hard).** As for block 5b: before the block starts, two smoke runs of
  one command must be bitwise identical -- every prediction file (same files, dtype, shape
  and bytes), every client's per-round training loss and every round's aggregated
  validation loss. The smoke command: camera fold 0, FedAvg, seed 42, 2 rounds, 5 local
  epochs, Adam 1e-3, batch 8, all three nodes at MAXN_SUPER
  (`--block camera_determinism_smoke`, runs `results/camera/maxn/diag_camera_smoke` and
  `..._r2`). They run after the capture and before the block. If they differ, the block does
  not start, and the failure is reported as it is, never explained away. The smoke runs are
  never analysed. 5b's gate (FLAME smoke runs) does not cover the camera block.
* **Separate from FLAME by dataset.** Camera testbed runs are written to
  `results/camera/maxn/`, desktop runs to `results/camera/desktop/`. The FLAME analysis
  (`scripts/analyze_results.py` and everything built on it) never reads `results/camera/`,
  so no camera run appears in any FLAME table. The camera analyses
  (`scripts/camera_analysis_a.py`, `scripts/camera_analysis_b.py`) read only
  `results/camera/`. The camera MAXN_SUPER runs are never pooled or compared with the FLAME
  MAXN_SUPER runs.
* **Capture clock check (implementation of section 2).** A capture is not scheduled while
  any node's clock differs from node_a's by more than 1 s or cannot be measured: the
  second-level synchronisation of section 2, enforced. The measured offsets are logged
  with every capture.

## 10. The pilot (declared 2026-09-29, committed and pushed before the pilot)

A pilot of one scene with all three cameras is run before the study captures begin.

* **Outside the study.** Pilot footage is stored only under `data/raw/camera_pilot/`.
  It is never used in any manifest, fold, training run or analysis of either question.
  The study scripts refuse paths under it: `camera_manifests.py`, `camera_fl_prepare.py`,
  `camera_loso.py` and `camera_analysis_{a,b}.py`. The pilot's frame table
  (`camera_labels.py`) may only be written inside the pilot tree
  (`scripts/camera_labels.refuse_pilot`, `tests/test_camera_separation.py`). If the
  pilot's physical setup is later used as a study scene, it is captured fresh, and no
  pilot frame enters the study.
* **What the pilot may check.** Only that the pipeline runs end to end without error:
  capture on all three nodes at the common start, the clock check, frame and metadata
  files, the frame table and exclusions, transfer, and preprocessing. Technical image
  properties may be looked at, for example exposure, framing, depth validity and
  synchronisation. No accuracy, loss or prediction quality on pilot data is computed or
  inspected.
* **Changes permitted after the pilot.** Technical changes only: capture settings,
  exposure handling, synchronisation, file handling and resolution. Each is recorded as a
  dated amendment to this file that states what the pilot revealed and what changed, and
  is committed before the first study capture.
* **Not permitted after the pilot.** Changes to the questions, metrics, splits, selection
  or interpretation rules, or anything else in sections 1-6 and 9 beyond the technical
  items above. Such a change would be a change after footage exists, to be disclosed
  in the paper and the response letter.
* From the first pilot capture on, footage exists in the sense of this file's preamble.
  The permitted technical amendments above are the only exception.

## 11. Amendment 2 (2026-10-05, before any footage exists)

Declared 2026-10-05 and committed and pushed before the pilot. On that day no footage of
this experiment existed: `data/raw/camera`, `data/raw/camera_pilot` and
`data/processed/camera_*` were absent on all three nodes (checked read-only), and the
desktop held only the blank scene-sheet template. So this amendment is part of the
pre-registration, not a change after data. It changes where the preprocessing of
section 3 runs, not what it computes, and no question, split, model, metric, family,
verdict phrase or interpretation rule.

* **Why.** The preprocessing of section 3 is a fixed set of functions
  (`src/data/camera_preprocess.py`). Until now each consumer ran them itself: every node
  on its own frames for the federated folds, with the nodes' Pillow 9.0.1, and the desktop
  for the leave-one-scene-out runs, the RGB-D and IR comparisons and the federated
  baselines, with Pillow 12.x. Two library versions are not guaranteed to give the same
  pixels. On FLAME the deterministic decode and resize happen to agree bitwise between
  the two, but the camera frames are larger and resized by a different path, and nothing
  checked them.
* **Preprocessed once, on one machine.** After the frame table (`camera_labels.py`),
  the desktop runs `scripts/camera_preprocess_frames.py` once on its copy of all three
  nodes' frames. It writes every valid frame's preprocessed streams:
  - RGB and IR as 224 x 224 uint8 PNG (lossless);
  - depth as a float32 `.npy` of the section-3 values (exact).

  It also writes their md5 manifest, `data/splits_camera/preprocessed_manifest.csv`, with
  the machine and library versions in `preprocessed_info.json`. Both are committed before
  any model is trained.
* **Every consumer reads those files, md5-verified.**
  - The federated folds (`camera_fl_prepare.py`) are byte-for-byte copies of the RGB
    files, checked against the manifest when written and by `--verify`. Each node
    receives only its own camera's preprocessed files, so section 9's "no node receives
    another camera's frames" still holds. No node decodes or resizes a native frame.
  - The desktop baselines of question (b) train on the same fold files.
  - Question (a) and the RGB-D/IR analyses read the files through
    `CustomRGBDDataset(preprocess="camera", preprocessed=...)`. That dataset refuses to
    run without the manifest and refuses any file whose md5 differs.
* **What stays as it was.** Training-time augmentation is still computed by the training
  code on each machine (federated: `FlameDataset`'s train transform on the nodes;
  desktop: `CustomRGBDDataset`'s numpy augmentation). It is part of training, not of the
  section-3 preprocessing.
* **The pilot** (section 10) exercises this path: the pilot frames are preprocessed with
  the same script into the pilot tree only (the script refuses any other destination for
  pilot frames, and any pilot destination for study frames).

## 12. Amendment 3 (2026-10-05, before any footage exists)

Declared 2026-10-05 and committed and pushed before the pilot. Checked read-only again
that day, after Amendment 2, no footage of this experiment existed:
- on all three nodes: no `data/raw/camera`, `data/raw/camera_pilot` or
  `data/processed/camera_*`, no capture record (`_captures/*.json`), and no `*_rgb.png`
  newer than 2026-09-28;
- on the desktop: only the blank scene-sheet template in `data/raw/camera`.

The v1 ZED scenes of April 2026 (`data/raw/captures/node_c/` on node_c) belong to the
original submission, not to this experiment. This amendment adds what every capture
records and fixes one capture setting. It changes no question, split, model, metric,
family, verdict phrase or interpretation rule, and not the preprocessing of section 3.

* **Why.** The original submission characterised the sensors' heterogeneity partly by
  their calibration: resolution, intrinsics and stereo baseline per camera. The
  revision's capture code recorded the RealSense intrinsics, but for the ZED 2i it
  recorded no intrinsics, distortion or baseline, and it left the ZED depth range at the
  SDK default. Without this amendment the revision would lose that evidence.
* **Recorded at every capture, for every camera**, in the capture record
  (`_captures/<capture_id>.json`):
  - intrinsics of the RGB and depth streams: width, height, fx, fy, principal point, and
    the distortion model and coefficients. The ZED also records its right camera; the
    RealSense also records its IR stream;
  - the stereo baseline in mm. RealSense: the `stereo_baseline` option, or else the
    left/right IR extrinsics. ZED: the calibration baseline;
  - the depth scale in mm per raw unit;
  - the depth-to-colour extrinsics (RealSense);
  - the depth range applied (ZED);
  - SDK and firmware versions.
* **Enforced.** A capture whose camera does not provide all of these is refused before
  any frame is kept, and its record says why (`calibration_problems` in
  `src/data/camera_capture_common.py`). A baseline outside 20-500 mm counts as missing,
  so a unit error cannot pass. A ZED depth range the SDK does not apply exactly is
  refused.
* **ZED depth range.** Set explicitly to 0.3-20 m (300-20 000 mm in the capture's
  millimetre units), as in the v1 captures, instead of the SDK default. The analysis
  clip of section 3 (0.3-10 m) is unchanged.
* **The pilot** checks it: every capture record carries the calibration, and the
  baselines are plausible (about 50 mm for the D435 models; 120 mm for the ZED 2i,
  serial 35201583). *Correction 2026-10-07: this line said "serial 32608934, as in v1".
  The pilot's ZED 2i reports 35201583. 32608934 is the serial of the ZED 2i in the v1
  captures (April 2026 capture metadata on the nodes, five scenes, and
  `/usr/local/zed/settings/SN32608934.conf` of 2026-03-27 on node_c); the SDK fetched
  `SN35201583.conf` on 2026-10-06, when the current unit was first connected. So the
  revision captures use a different ZED 2i unit from v1, of the same model; the 120 mm
  baseline holds for both. The two units differ in focal length: at the same 1920 x
  1080 the v1 unit records fx = 1951 px, about 52 deg horizontal field of view, and the
  current unit fx = 1051 px, about 85 deg. This is consistent with the 4 mm and 2.1 mm
  lens options of the ZED 2i, but the lens is not recorded. Commit `2dab803` called the
  old number a transcription error; that was wrong and is corrected here.*

## 13. Amendment 4 (in preparation; the flame-source criterion committed 2026-10-08, before e5)

> Amendment 4 is written after the pilot of section 10 (scene `s01`, four captures on
> 2026-10-07, all PASS) and before any study footage. The pilot frames were looked at
> for technical image properties only: exposure, framing, depth validity, flame size in
> pixels. No accuracy, loss or prediction quality was computed or inspected.
> Only the flame-source acceptance criterion is committed now, before any e5 capture, so
> that the e5 acceptance test is judged by a rule fixed beforehand. The other items of
> this amendment, including 13.1, are committed before the first study capture. Values
> marked `[e5]` are filled from the e5 acceptance test (`docs/POST_5B_CHECKLIST.md`).

### 13.2 Changes to section 2 (beyond section 10's technical list; to be disclosed)

These are changes after footage exists, because the pilot frames count as footage
(section 10). They are disclosed as such in the paper and the response letter. None of
them touches the questions, metrics, splits, selection or interpretation rules of
sections 1 and 4-6.

* **Flame-source acceptance criterion.** This is a *new acceptance criterion*; it was
  not part of the original pre-registration.
  - *When it was written.* In this pixel form on 2026-10-08: **after the pilot frames had
    been seen** (technical image properties only, as stated above), and **before any e5
    data existed**. The author set N and the statistic on 2026-10-08, and they were
    committed and pushed before the e5 acceptance test was captured.
  - *Earlier wording.* The draft of 2026-10-07 stated the criterion as "a visible flame
    height of at least about 10 cm at 3 m". That wording is replaced, and it was never
    applied to any data.
  - *Criterion:* the visible vertical height of the flame source in the 224 input image
    of section 3 is **at least N = 5 px at every distance used and on every camera**.
    - It is measured in e5 at every candidate distance (1, 2 and 3 m) on all three
      cameras: the D435if, the D435i and the ZED 2i.
    - *Statistic:* for each camera x distance, the **median** of the vertical flame
      height over the **40 frames** of that camera's capture at that distance, measured
      in the 224 input image. It passes if the median is >= 5 px.
  - *Consequence:*
    - A distance at which the source falls below N px on any camera is removed from the
      study's distance set, for every scene. Because the capture is simultaneous, it is
      removed for all three cameras.
    - If no distance passes, the source is not accepted, and another candidate is tested
      before any study capture.
  - *Rationale for a pixel threshold:* what the classifier sees is the flame's extent in
    the 224 crop. A source of a given physical size spans different numbers of pixels on
    each camera and at each distance. From the recorded pilot intrinsics, one pixel of
    the 224 crop covers 1.38 cm at 3 m on the ZED 2i (fy_224 = 218.0 px) and 1.06 cm on
    the RealSense cameras (282-284 px). Below some size, the flame is a few saturated
    pixels rather than a region, unlike the flames in FLAME.
  - *Source: not yet fixed.* Candidates, in this order:
    1. a torch, re-tested with its wick adjusted;
    2. as the alternative, four candles side by side, as one fixed unit.

    The source is the first candidate that passes the criterion at the distances
    needed.
  - *To be filled from e5:* the chosen source, the median flame height per camera x
    distance in 224-pixels, the height in cm against the ruler, and the resulting
    distance set. [e5]
  - The two pillar candles of the pilot are not used further.
  - Section 2 already allows the source type to be chosen per scene. What is new is the
    size criterion.

## 14. Erratum (2026-10-07)

**The preamble (2026-09-28) says:** "on that day no camera capture of any kind was
present on the three nodes or the desktop (checked read-only; the only matching file on
the nodes is a librealsense unit-test `.bag`). The five-scene captures behind the v1
cross-camera result are no longer on the nodes and are not used here."

**Correct:** The v1 five-scene captures were on all three nodes that day and still are,
in `~/FedRGBD/data/raw/captures/`:

| node | files | content |
|---|---|---|
| node_a | 2765 | its own scenes, plus copies of node_b's and node_c's (byte-identical) |
| node_b | 1005 | its own scenes |
| node_c | 755 | its own scenes |

The files' birth time is 2026-04-23, 12:35-13:47, and their ctime and mtime are the
same. Access times are not used as evidence. The read-only check of 2026-09-28 missed
these files; its search is not recorded. Amendment 3 (2026-10-05) named the node_c
scenes as belonging to the original submission. It did not correct the preamble and did
not mention the copies on node_a or the originals on node_b. The files are left as they
are. Their content as of 2026-10-07 is fixed by md5 manifests in `data/v1_captures_md5/`,
checked against all three nodes with `md5sum -c` (all OK).

**Audit (2026-10-07):**
- *Code in the repository.* No code in the repository reads `data/raw/captures`, now or
  in any commit since 2026-09-28 (`git log -G` on the path; the only hit is the prose of
  Amendment 3, `4802ebe`).
- *Write paths.* The camera experiment writes only to `data/raw/camera`,
  `data/raw/camera_pilot` and `data/processed/camera_*`. The legacy default
  `data/raw/custom` is also separate. None of these overlaps with `data/raw/captures`.
- *Manifests, folds, training and analysis.* None of these exists yet for this
  experiment. The only frame table and preprocessing so far are the pilot's, and they
  read only `data/raw/camera_pilot`. The study scripts read only the paths above.
- *The v1 pipeline on node_a, outside git.* node_a also holds the untracked v1
  cross-sensor pipeline that read these captures. All its files are dated 2026-04-23:
  - scripts `scripts/prepare_captures.py`, `scripts/cross_sensor_eval.py` and
    `scripts/cross_sensor_depth_eval.py`, never committed;
  - their outputs `data/processed/captures/` (903 files), `data/processed/captures_depth/`
    and `data/processed/captures.zip`;
  - `results/cross_sensor/` and `results/cross_sensor_depth/`.

  They are the original submission's cross-sensor analysis. They are derived camera
  data, so the preamble's "no camera capture of any kind" missed them too. No revision
  code reads them: the repository's `scripts/cross_sensor_loso.py` (2026-09-17) reads
  `data/raw/custom`. node_b and node_c hold no such derived files; their checkouts have
  no untracked files.

  The files on node_a are left as they are. Their content as of 2026-10-07 is recorded
  in two places:
  - `data/v1_captures_md5/node_a_v1_derived.md5` (commit `1409085`): 1812 files, verified
    with `md5sum -c`;
  - `legacy/v1_cross_sensor/` (commit `89eb15c`): the three scripts and the two
    `results.json` as byte copies, md5-identical to that manifest. They were not run,
    and no image data was copied.
- *Manual inspection.* node_a's `.bash_history` keeps no timestamps. It contains the v1
  capture, `scp` and pipeline commands above, plus `find` and `ls` commands on
  `data/raw/captures` (for example `find ~ -type d -name "captures"` and `ls` of
  `node_*/scene01_fire_candles_1.0m/`). When they were run is unknown. No code path of
  the revision read the files, but manual inspection from a shell, before or after
  2026-09-28, cannot be excluded entirely.

So the statement "not used here" holds, and the statement "no longer on the nodes" does
not. The v1 files also record the v1 ZED 2i unit (S/N 32608934, fx 1951 px at
1920 x 1080), which differs from the revision's (section 12 correction).
