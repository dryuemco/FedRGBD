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

## 13. Amendment 4 (DRAFT 2026-10-07 -- after the pilot, before any study footage)

> **DRAFT.** Values marked `[e5]` come from the e5 acceptance test
> (`docs/POST_5B_CHECKLIST.md`). Nothing here is in force until the draft is completed,
> committed and pushed, and that happens before the first study capture.
> **Exception:** the flame-source acceptance criterion of 13.2 (N = 5 px, median over 40
> frames per camera x distance) was committed and pushed on 2026-10-08, before any e5
> capture. So were the choice of source (four candles) and "One lit source per capture".
> These parts are in force as committed, and only their `[e5]` values are still to be
> filled.
> **Committed 2026-10-08, during e5, before the s03 acceptance test and before any
> calibration value existed:** the whole of section 13 as it stands then, including
> the exposure and white-balance policy of 13.1, sequential capture (13.2) and the
> cancelled e5 steps. The `[e5]` values are filled later, each as a dated change.

Written after the pilot of section 10 (scene `s01`, four captures on 2026-10-07, all
PASS), and after its frames were looked at for technical image properties only:
exposure, framing, depth validity, flame size in pixels. No accuracy, loss or prediction
quality was computed or inspected. No study footage exists (`data/raw/camera` is absent
on all three nodes). Each item says what the pilot revealed, what changes, and whether
the change is one of the technical changes that section 10 permits after the pilot, or
a change to section 2 that has to be disclosed in the paper and the response letter.

### 13.1 Technical changes (permitted by section 10)

* **Rig geometry for simultaneous capture.** *Pilot:* one tripod was available. Mounting
  the cameras one after another would have broken the simultaneous capture of section 2.
  A rig near the floor gave grazing-angle depth (valid depth 61-88 %, none in the bottom
  quarter of the ZED image); raised and tilted it gave 99 %. With that tilt, a flame at
  1 m fell below the RealSense colour field of view, which is narrower (vertical about
  42 deg against about 70 deg for the ZED). *Change:* the three cameras are on one rigid
  plate on one tripod, at one height, pointing the same way. Before every capture, one
  snapshot per camera shows the flame position inside all three colour images and inside
  the 224 centre crop (section 3). Rig height above the floor, plate tilt and the
  distances used are recorded per scene. Distances are measured on the floor tape from
  the plate's front edge. A distance at which the flame is not in all three colour
  images is not captured for that scene.
* **USB 3.x gate.** *Pilot:* node_a's D435if enumerated at USB 2.1 (480 Mb/s), where the
  colour and depth profile of section 2 does not resolve. The v1 records did not log the
  USB type. *Change:* a capture FAILs unless every RealSense reports
  `usb_type_descriptor` 3.x and the ZED's USB device runs at >= 5000 Mb/s (sysfs
  `speed`). Both values go into the capture record and are checked in
  `camera_capture_session.py --check`.
* **Lighting and exposure handling.** *Pilot:* in all 320 RealSense frames the logged
  exposure, gain and white balance are the sensor option values
  (`exposure_source: sensor_option`), not the values auto exposure applied. Both nodes
  build librealsense 2.55.1 with `FORCE_RSUSB_BACKEND=true` (CMakeCache). A probe on
  node_a (2026-10-07, no frames kept) showed why: under this backend the D435 colour
  frame delivers per-frame metadata (auto-exposure flag, timestamps) but not
  `actual_exposure`, `gain_level` or `white_balance`, and the depth frame does deliver
  them. The ZED logs its applied values. *Change:*
  - *Lighting.* Room lighting is fixed for the duration of a scene: the same lamps on,
    curtains closed, no daylight change between the fire and no-fire captures. The lamps
    are recorded in the scene sheet.
  - *Exposure and white balance: one fixed setting per camera (DRAFT 2026-10-08).* This
    replaces the earlier draft, in which the lock was "determined on the no-fire capture
    of each scene x distance": auto exposure and auto white balance settled on the
    no-fire setup, their values were fixed, and the fire capture at the same distance
    reused them. The new draft was written after the e5 geometry captures (s09), before
    the s03 acceptance test and before any calibration value existed. It is committed
    before the acceptance test.
    - *Policy.* Each camera has one exposure, one gain and one white balance. Auto
      exposure and auto white balance are off, and the same values are used in every
      scene, at every distance and under every condition, for fire and no-fire alike.
      Nothing is determined per capture. The ZED follows the same rule, set manually
      through `sl.VIDEO_SETTINGS` (`AEC_AGC` and `WHITEBALANCE_AUTO` off; `EXPOSURE`,
      `GAIN` and `WHITEBALANCE_TEMPERATURE` fixed).
    - *Calibration scene.* The current setup with the candles in place and unlit,
      `lamps_on=tavan lambasi`. The cameras are calibrated one after another from the
      same tripod position. The calibration `--notes` must parse as a no-fire capture
      (`source=none`, `flame_height_cm=0`) and must name the lamps that are on.
    - *Step 1, gain.* The camera's default gain: for a RealSense, the colour option
      range's `default`. The ZED SDK reports no default gain, so the ZED gain starts at
      0 (decided by the author on 2026-10-08). The gain changes only by the rule of
      step 3.
    - *Step 2, white balance.* Gain and exposure are at their defaults (ZED exposure: the
      geometric middle of 1-100 % of the frame period). The white balance is the grid
      value (RealSense step 10 K, ZED 100 K) that minimises |mean R - mean B| inside a
      neutral rectangle on the white wall. The rectangle is defined per camera in
      normalized coordinates (`configs/camera_wb_region.json`). It excludes the curtain,
      the floor, the skirting board, the objects and the flame position. The search
      bisects on the sign of R - B, assuming that a higher white balance setting makes
      the image warmer, and then takes the better of the two neighbouring grid values;
      a tie goes to the lower value. If R - B does not change sign over the range, the
      better end of the range is taken.
    - *Step 3, exposure.* Gain and white balance are fixed. The exposure is the first
      value reached by a bisection on a log scale, starting at the camera's default,
      whose section-3 224 input image has a mean 8-bit luma in [100, 130]. A frame that
      is too bright lowers the upper bound, and one that is too dark raises the lower
      bound. The RealSense exposure is capped at one frame period of the 30 fps stream
      (333 x 100 us); the ZED exposure is capped at 100 % of the frame period.
      *If the exposure reaches the cap and the 224 luma is still below 100* (decided by
      the author on 2026-10-08): the exposure stays at the cap, and the gain is raised by
      the same bisection, between the starting gain and the gain maximum (RealSense
      128, ZED 100), until the luma lies in [100, 130]. The same rule holds for the
      ZED: exposure first, up to the cap, then gain. If neither reaches the target, the
      calibration fails and writes nothing.
    - *Measurement at each step:* apply the setting, discard 15 frames, then measure 10
      frames: the median 224 mean luma, and the mean R, G and B over the rectangle.
    - *Step 4, verification.* A last measurement with the final values. It reports the
      224 mean luma and the rectangle's mean R, G and B (with R - B). If the luma falls
      outside [100, 130], the calibration fails.
    - *Record.* Only measured values are kept, no calibration frame. They go into
      `<root>/<node>/_exposure/exposure.json`, with every step of both searches, the
      verification, the camera serial, the rectangle, `lamps_on` and the calibration
      notes.
    - *Refusals.* A capture is refused if this file is missing, if it belongs to another
      serial, or if the capture's `lamps_on` differs from the calibration's. **If the
      lamp condition changes, the camera is recalibrated.** Recalibrating moves the old
      file to `_superseded/` and is disclosed.
    - *Check at the start of every capture session* (decided by the author on
      2026-10-08, committed before s03). With the camera's fixed setting applied, a
      flame-free verification measurement is taken: the same measurement as a
      calibration step, on the flame-free setup with `lamps_on` as calibrated. If its
      224 mean luma lies outside +-10 % of the calibration value (the `luma_224_median`
      of `exposure.json`), the camera is recalibrated before any capture of the session.
      Every check, passed or not, and any recalibration it causes, is recorded in the
      session log.
    - *No flame frame is used.* The values are chosen without looking at any frame with
      a flame. They are committed before the acceptance test.
    - The values go into every frame's metadata (`exposure_source: manual_lock`) and
      into the capture record.
    - Code (draft, not committed): `src/data/camera_exposure.py`,
      `configs/camera_wb_region.json`, and `--calibrate_exposure` in
      `realsense_capture.py` and `zed_capture.py`.
  - *Gate.* A locked capture FAILs if the auto-exposure flag is on in any colour frame:
    the RealSense colour frame's `auto_exposure` metadata, which this backend does
    deliver, and the ZED's `AEC_AGC` setting. e5's lock test was to check that the gate sees
    the flag change.
  - *What e5 showed (2026-10-08, node_b D435i, scene s09, one camera at a time).*
    - *Lock values.* After auto exposure was switched off, the colour sensor options
      read exposure 166, gain 64 and white balance 4600. These are the option range's
      defaults (`get_option_range(...).default`). The "lock" was therefore the sensor
      defaults, not values auto exposure had applied. Under this backend `get_option`
      does not return the applied exposure while auto exposure runs.
    - *Camera state.* Auto exposure and auto white balance stay off after a capture
      (`enable_auto_exposure` reads 0).
    - *Brightness, descriptive only.* The 224 mean luma of the locked
      `s09_no_fire_d100` frames was 65.38-65.40. Two framing-only frames taken with
      auto exposure on and the candles lit gave 122.4 and 123.3. Those two frames are
      not e5 data.
    - *AE gate triggered on real hardware.* In `s09_no_fire_d300`, auto exposure was on
      in 5 of 5 colour frames, while the settings read back off. The capture is kept as
      it is (status incomplete, not moved to `_retakes/`). The cause was not
      investigated.
    - [e5: the s03 acceptance test runs under the policy above once it is decided]
* **ZED raw calibration.** *Pilot:* the ZED record holds the calibration of the
  rectified images (all distortion coefficients 0, correct for the saved frames). The
  factory calibration is not kept. *Change:* every ZED capture record also stores
  `calibration_parameters_raw` (both cameras, with distortion, and the stereo
  transform), plus the factory file `SN35201583.conf` (from `/usr/local/zed/settings/`,
  byte copy, with its md5).
* **Mandatory capture notes.** *Pilot:* no `--notes` were given. The session log
  therefore does not say which source was lit or how large the flame was; the number of
  candles had to be read off the frames afterwards. *Change:* `--notes` is required and
  parsed: `source=<id>; distance_measured_m=<x.xx>; flame_height_cm=<x>`. For a
  no-fire capture, `source=none` and `flame_height_cm=0`. A capture with a missing or
  unparseable field FAILs before it starts. The lock values are not typed into the
  notes: each node writes them into its camera's exposure file
  (`_exposure/exposure.json`, draft policy above) and into every capture record and
  frame, so they cannot be mistyped.

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
  - *Source: four candles side by side, as one source* (source id `candles4`; see "One
    lit source per capture" below).
    - Decided by the author on 2026-10-08, before any e5 capture, by a physical
      comparison with a ruler. No capture was taken and no frame was looked at for it.
    - The torch, with its wick adjusted, gave a smaller flame than the four candles.
    - The torch is therefore not tested in e5 (its planned scene `s02` is cancelled).
      The acceptance test is scene `s03`, with the four candles and 40 frames per capture.
    - The reason is also written into every e5 fire capture's `--notes`.
  - *Flame-tool error fix made after the e5 data was seen* (2026-10-08, decided by the
    author; committed and pushed before the fixed tool was run on any fire frame).
    - *The error.* The first version of `scripts/camera_flame_height.py` (`3681d83`)
      took the **largest** region of pixels at least 40 (8-bit) brighter than the
      no-fire median reference. On the D435i's s03 captures it measured 35, 21.5 and
      45 px at 1, 2 and 3 m, against 9.9, 4.9 and 3.3 px expected from the ruler.
      - The author looked at the overlays (the median fire frame with the measured box).
        At 3 m the box (about 34 x 44 px at 224) covered not the flame core but the
        flame's glow on the two walls of the corner and on the floor. At 2 m it took in
        the floor glow, and at 1 m the lit candle body.
      - Run on the no-fire frames, presented as fire against their own reference, the
        first version found 0 px at 1 and 2 m. At 3 m it found a region in 17 of 40
        frames (up to 101 px, touching the crop edge), with median 0.
    - *The fix.* A pixel is a flame pixel only if it is at least 40 brighter than the
      reference **and** looks like flame itself:
      - either a near-saturated core (max channel >= 245, any hue);
      - or bright and warm (max channel >= 200, R >= G >= B, saturation >= 0.25).

      Components (8-connected) that touch the edge of the 224 crop are dropped. The
      flame is the remaining component that contains the brightest flame pixel (ties:
      the larger brightness increase, then the topmost). Its height is its row span, as
      before.
      - The constants 245 / 200 / 0.25 are the same for all three cameras. They were
        written down after the D435i's three median fire overlays had been seen and
        before the fixed tool was run on any fire frame. They are never tuned to a
        distance result.
    - *Validation, fixed before applying the tool to fire frames.*
      - (a) On the no-fire frames, presented as fire, the fixed tool must find 0 px. It
        does at 1, 2 and 3 m on the D435i.
      - (b) On the fire frames, `ratio_measured_expected` (measured px over the ruler's
        expected px) must lie in [0.5, 2.0] at 2 m and 3 m, and in [0.5, 4.0] at 1 m.
        The 1 m band is wider because the four candles stand one behind the other and
        the lit candle body can join the flame.
      - If (b) holds at a distance, the tool counts as validated there, and the 5 px
        criterion is applied with it.
      - If (b) fails at a distance, the tool is not validated for that distance. No
        criterion decision is taken there (the tool prints `NOT VALIDATED`, and the
        distance is `NOT DECIDED`), the result is reported, and the tool is not
        readjusted.
    - The tool is checked per camera x distance. The distance decision itself is taken
      only after all three cameras' s03 exist.
  - *Scene settling and a clean acceptance geometry* (decided by the author on
    2026-10-08, after the e5 D435i data was seen; committed and pushed before any
    further acceptance capture).
    - *What the data showed.* In the D435i's `s03_no_fire_d300` capture, frames 0-16
      (the first about 3.2 s) differ from the capture's median along a thin vertical
      strip at x of about 157 px (224 input), rows 0 to about 100. The author
      identified it as the curtain edge still moving after the scene was set up. From
      about 3 s on, the frames are stable. With the fixed tool the strip gives 0 px.
    - *Rule for every capture.* The curtain's bottom edge is fixed in place. After any
      movement in the scene (setting up, moving or lighting the source), wait at least
      10 s before a capture starts.
    - *Repeat of the 2 m and 3 m acceptance test in a clean geometry.* With the fixed
      tool (`76a7003`), the D435i's s03 at 2 m and 3 m was `NOT VALIDATED`
      (`ratio_measured_expected` 2.63 and 4.56, outside [0.5, 2.0]). In s03 the source
      stood near the corner, on the glossy floor, close to the walls.
      - The 2 m and 3 m acceptance test is repeated with the source on a matte,
        non-reflective base, at least 1 m from the nearest wall. The geometry is the
        same for every camera.
      - Same tool (`76a7003`), same validation band (b), same 5 px criterion.
      - If the tool is validated at a distance, the 5 px criterion is applied there. If
        it is not validated, that distance is removed from the study's distance set.
      - The existing s03 2 m and 3 m captures stay as they are. They are reported as
        "NOT VALIDATED (reflective geometry)". s03 1 m (validated) stands.
  - *Reference tripod position and clean geometry s10* (decided by the author on
    2026-10-08; committed and pushed before any capture at the new position).
    - The tripod was moved 1 m back from its e5 position, which had not been marked on
      the floor. The new position is the **reference position** and is now taped on the
      floor. All calibrations and session checks are done there.
    - *D435i recalibrated there* (reason: the tripod moved and the old position was not
      marked; no study data exists yet). The earlier D435i calibration (`eef07c7`) and
      the D435i's s03 and s09 captures stay on record unchanged. The old calibration file
      moves to `_exposure/_superseded/`.
    - *D435if and ZED* are calibrated at the reference position too.
    - *s10, the clean acceptance geometry.* The source stands on a matte, non-reflective
      base. The 1, 2 and 3 m marks are measured from the reference position, and at 3 m
      the source is at least 1 m from every wall. The source is now 1 m in front of the
      corner and 3 m from the camera. s10 repeats the acceptance test at 1, 2 and 3 m
      with the tool, validation band and 5 px criterion of the items above, the same for
      every camera.
  - *To be filled from e5:* the median flame height per camera x distance in 224-pixels,
    the height in cm against the ruler, and the resulting distance set. [e5]
  - The two pillar candles of the pilot are not used further.
  - Section 2 already allows the source type to be chosen per scene. What is new is the
    size criterion.
* **One lit source per capture.** Exactly one flame source is lit in a fire capture, and
  the same one in every fire capture of a scene. It is recorded through the notes of
  13.1. The four candles side by side count as one source with one source id
  (`candles4`): always the same four candles, in the same arrangement, all lit together.
* **Arrangement of `candles4`** (decided by the author on 2026-10-08, committed and pushed
  before any e5 capture). Before this decision, two framing-only frames were taken with
  the D435i, auto exposure on and the candles lit. They were stored outside
  `data/raw/camera_pilot` and are not e5 data.
  - The four candles stand in **one row along the camera's optical axis**, with **5 cm
    between neighbouring candles, edge to edge**. This replaces "side by side" above.
  - The capture distance is measured from the plate's front edge to the **front
    candle**, the one nearest the camera.
  - Every capture's `--notes` records the arrangement as
    `arrangement=inline_axis; spacing_cm=5; distance_ref=front_candle`.
* **Scene diversity.** Scenes differ in background *and* floor (surface material or
  colour), not only in the set of distractors. Each scene's background and floor are
  recorded in the scene sheet. This narrows section 2's definition ("two scenes differ
  in location or background"). The pilot's room, with its curtain and one laminate floor,
  could supply only a single scene under this rule.

* **Distance set (conditional on e5).** *Pilot:* with the plate tilted for valid
  near-field depth, a flame at 1 m fell below the RealSense colour field of view, so the
  pilot captured 2 and 3 m only. e5 tests whether some tilt keeps a 1 m flame inside
  all three colour images *and* inside the 224 crop, while also keeping the 3 m flame
  there and depth valid. If no tilt does, the study's distance set is {2, 3} m for every
  scene, and that narrowing of section 2's {1, 2, 3} m (2 or 3 per scene) is disclosed
  here. [e5: outcome]

* **Sequential capture: one tripod, one camera at a time** (decided by the author on
  2026-10-08, during e5, before the s03 acceptance test). This changes section 2 and
  13.1 ("Rig geometry for simultaneous capture"), and is disclosed in the paper and the
  response letter.
  - *What happened first.* At the start of e5 the author decided that e5 would capture
    one camera at a time, starting with the D435i on node_b. Each capture was started
    with the per-node script (`src/data/realsense_capture.py` / `zed_capture.py --node
    <node>`), not with the three-node orchestrator. The camera order and each capture's
    time went into its `--notes` and the e5 log.
  - *Decision.* The study captures are sequential as well. There is a single tripod at
    a fixed position, and the cameras are mounted on it and captured one after another.
    The three cameras therefore do not see the same flame instant.
  - *What this changes:*
    - Each camera has its own fixed (exposure, gain, white balance), by the 13.1
      policy. Its calibration runs on that camera, from the same tripod position.
    - The flame and its height can differ between the cameras of one scene x distance.
      The 13.2 criterion is still evaluated per camera x distance, and a distance that
      fails on any camera is still removed for all three cameras and every scene.
    - Section 2's paired cross-sensor design (questions (a) and (b)) pairs captures of
      one scene and distance taken one after another, not at the same instant. This is
      disclosed with the results.
  - *e5 steps cancelled because of it, with the reasons:*
    - **Depth-hole diagnosis s06-s08.** It asked whether the second RealSense's
      projector causes the dotted holes in the D435if depth. In a sequential capture no
      second camera streams while the D435if captures, so the question does not arise
      in the study. Instead, the D435if's own hole fraction is measured on its s03
      no-fire frames, inside the rectangle fixed before e5 (`baccca6`,
      `configs/e5_depth_hole_region.json`, `scripts/camera_depth_holes.py`). The
      firmware decision of 13.3 is taken from that measurement.
    - **The lock test (s04/s05) and its per-frame AE-flag check.** It tested the
      per-capture lock that the 13.1 policy replaces. Its place is taken by three
      things: the calibration, the verification frames of each camera, and the AE gate
      that already triggered on real hardware (`s09_no_fire_d300`, 13.1 "What e5
      showed").
    - **The 3 m pair of s09.** The geometry is judged from the framing frames and from
      s03. The failed capture `s09_no_fire_d300` stays as it is: incomplete, not moved
      to `_retakes/`, and not used.

### 13.3 Deferred

* **D435 firmware.** librealsense 2.55.1 recommends D4xx firmware 5.16.0.1. node_a's
  D435if runs 5.13.0.55 and node_b's D435i 5.17.0.10, the same as in v1. This is
  decided after e5, together with the dotted depth holes seen on the wall in the D435if
  depth (fewer in the D435i's). Since capture is sequential (13.2), the holes are
  measured on the D435if's own s03 no-fire frames, inside the `baccca6` rectangle; the
  test against the second camera's projector (s06-s08) is cancelled.

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
