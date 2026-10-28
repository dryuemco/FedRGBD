# Study capture plan -- DRAFT (2026-10-09, not committed)

> **DRAFT for review with the author.** Nothing here is decided or committed. Rules quoted
> from the pre-registration (`docs/CAMERA_EXPERIMENT_PREREG.md`) are marked with their
> section. Everything else is a proposal; open points are listed at the end.
> Times come from `logs/e5/e5_capture.log` (s10/s11, 2026-10-08/09); estimates built on
> them say which parts are measured and which are assumed.

## 1. Fixed by the pre-registration

| Rule | Source |
|---|---|
| 15-20 scenes; a scene = one location, one fixed background, one fixed set of distractors | 2 |
| Scenes differ in background **and** floor (material or colour); both recorded in the scene sheet | 13.2 "Scene diversity" |
| Every scene captured fire and no-fire, rig/background/distractors unchanged; label declared in the command | 2 |
| At least one fire-like distractor per scene, present in both classes, recorded | 2 |
| 2 or 3 distances per scene from {1, 2, 3} m, the same for both classes | 2 |
| 40 frames per (scene, class, distance, camera) at 5 fps | 2 |
| Sequential capture: one tripod, cameras mounted one at a time | 13.2 |
| One fixed (exposure, gain, WB) per camera, AE/AWB off, the same in every scene; recalibration if the lamp condition changes | 13.1 |
| Session-start check per camera, flame-free, +-10 % of the calibration luma; otherwise recalibrate | 13.1 |
| Curtain edge fixed; >= 10 s after any movement before a capture | 13.2 |
| One lit source per capture, the same in every fire capture of a scene; `candles4` (same four candles, inline along the optical axis, 5 cm gaps, distance to the front candle) | 13.2 |
| Scene ids assigned in capture order, never reused: the study starts at **s12** (s01-s11 are pilot/e5) | 2 |
| Every capture `--root data/raw/camera` (study root), never the pilot root | 10, 11 |

**Not fixed by the pre-registration:** the order of fire and no-fire captures within a
scene. Section 2 does not state one; the e5 order (fire 3-2-1, then no-fire 1-2-3) is
declared "e5 only", and 13.2 says study captures "keep the order of the
pre-registration", which does not exist as such. **Open item O1.**

## 2. Two options -- (B) DECIDED 2026-10-13 (prereg 13.4)

| | (A) 15 scenes x 3 distances | (B) 18 scenes x 2 distances |
|---|---|---|
| Distances per scene | 1, 2, 3 m | alternating pairs: {1,2}, {1,3}, {2,3}, each in 6 scenes |
| Captures per distance | 15 scenes each | 12 scenes each (balanced) |
| Captures per camera | 15 x 2 x 3 = 90 | 18 x 2 x 2 = 72 |
| Captures in total (3 cameras) | 270 | 216 |
| Frames in total | 10 800 | 8 640 |
| Scenes for LOSO folds | 15 | 18 |

(B)'s pair rotation, by scene: s12 {1,2}, s13 {1,3}, s14 {2,3}, s15 {1,2}, ... (repeat),
so each pair occurs 6 times and each distance 12 times.

## 3. Time estimate

**Measured (s11, 2026-10-09):**
- one 40-frame capture: 13-15 s (camera open, 15 warm-up frames, 40 frames at 5 fps);
- gap between consecutive captures of one camera (operator moves the source, reads the
  ruler, waits 10 s): 26-56 s, mean about 40 s;
- one camera's full block of 6 captures: 4.3 min (D435i), 4.3 min (D435if), 5.2 min (ZED);
- calibration: about 30-60 s of camera time per camera; session check: about 15 s.

**Assumed (not measured; to be timed on the first scene):**
- camera swap on the plate (unmount, mount, cable, 10 s settle): about 3 min;
- scene set-up (background, floor, distractors, scene sheet, tape marks): about 15 min;
- session start (each camera mounted at the reference position for its check, then back
  to the scene): about 4 min per camera, 12 min per session.

**Per scene, updated 2026-10-13 for O1** (fire first per camera, prereg 13.2). One camera
block = its captures and the gaps between them. The step from the last fire capture to
the first no-fire capture is now >= 120 s (the capture script enforces it) instead of a
~40 s gap, inside the block. Lighting the candles and waiting for steady flames is not
timed yet and not included.

| | (A) 3 distances | (B) 2 distances |
|---|---|---|
| captures per camera | 6 | 4 |
| camera block | 6 x 14 s + 4 x 40 s + 120 s = 6.1 min | 4 x 14 s + 2 x 40 s + 120 s = 4.3 min |
| + swap (3 cameras) | 3 x (6.1 + 3) = 27 min | 3 x (4.3 + 3) = 22 min |
| + scene set-up | 15 min | 15 min |
| **per scene** | **about 42 min** | **about 37 min** |
| all scenes | 15 x 42 = 10.5 h | 18 x 37 = 11.1 h |
| + session starts (one per day, 12 min) | about 11 h, two days | about 11.5 h, two days |

(Before O1, with no-fire first and the 2 minutes inside the camera swap: about 38 and 33
min per scene, 9.5 and 9.9 h.) Both options take about the same time; (B) gives 3 more
scenes for the LOSO folds, (A) gives every scene all three distances. Neither estimate
includes breaks, candle changes, lighting the candles or a recalibration.

## 4. Per scene (proposal)

1. **Scene set-up.** Change the background and the floor (section 5), place the
   distractors, tape the 1/2/3 m marks from the tripod's front edge, fill in the scene
   sheet: background, floor, distractors (each named, unbranded), lamps (`tavan lambasi`),
   source `candles4`, distances, tilt `17` deg (prereg 13.1, held fixed). Curtain edge fixed.
2. **Camera order rotates across scenes** (Latin square). **DECIDED 2026-10-13 with a different
   assignment** (prereg 13.4: order = floor((n-12)/3) mod 3, so that distance pair and order are
   crossed); the table below is the superseded proposal:

   | scene | 1st | 2nd | 3rd |
   |---|---|---|---|
   | s12, s15, s18, ... | D435i (node_b) | D435if (node_a) | ZED (node_c) |
   | s13, s16, s19, ... | D435if | ZED | D435i |
   | s14, s17, s20, ... | ZED | D435i | D435if |

3. **Per camera: fire first, then no-fire** (O1, decided 2026-10-13, prereg 13.2 "Study
   capture order within a scene").
   - light the candles, wait for steady flames;
   - fire 3 -> 2 -> 1 m (moving the lit base), each after >= 10 s still;
   - put the candles out: **>= 2 min and an ember check**; the capture script refuses a
     no-fire capture earlier than 120 s after the node's last fire capture of the scene;
   - no-fire 1 -> 2 -> 3 m (candles unlit, cold, no ember), each after >= 10 s still;
   - (B) the same with the scene's two distances.
   - The 2 min now falls inside every camera block (about +6 min per scene against the
     earlier proposal, not yet in the time estimate of section 3).
4. **Notes per capture:** `source`, `distance_measured_m`, `flame_height_cm`, scene sheet
   fields, `camera_order=...`, `capture_time`, `lighting=...`, `lamps_on=tavan lambasi`
   (exactly; a different value is refused by the code), `arrangement`, `spacing_cm`,
   `distance_ref`.
5. **No study capture with a camera whose session check of the day has not passed.**

## 5. Backgrounds and floors (proposal; each scene differs in both)

Keeping every scene in the **same room under the same ceiling lamp** keeps the lamp
condition of the calibration (13.1) -- another room or another lamp means a
recalibration of all three cameras (open item O2). Variation then comes from movable
backdrops and floor covers:

- **Backgrounds** (behind the source, filling the 224 crop): plain white wall; the dark
  curtain; a light-grey cloth; a beige cloth; a wooden board / cupboard door; a bookshelf
  with spines turned inward or covered; a patterned (non-text) fabric; a corkboard.
- **Floors** (under the source and the near field): the laminate; a grey felt/carpet
  tile; a beige rug; a dark rubber mat; a light wooden board; a terracotta-coloured mat;
  a concrete-look tile.
- Pair them so that no two scenes share both, and each background and each floor recurs
  at most 2-3 times. A list of 15/18 (background, floor) pairs is left for the review.

### 5a. DECIDED 2026-10-28 (prereg 13.4)

Position B (in front of the curtain, facing the wardrobe, 4.5 m), the session-check
set-up, the 18 (position, backdrop, floor, distractors) assignments, the placement rules
and the materials are recorded in prereg 13.4. A first proposal of the same day used the
same Graeco-Latin square at A and B, so that the two scenes sharing backdrop and floor
also shared distance pair and camera order; it was replaced before it was recorded
(reviewer comment R3.3: a held-out scene should have no near-copy among the training
scenes).

## 6. Distractors (unbranded; at least one fire-like per scene)

Fire-like, unbranded: a plain red/orange mug; a red or orange book with a blank cover; a
red toy car (logo covered); an orange ball; a red/orange cloth; a copper or brass pot
(reflections); a plain orange plastic box. Neutral ones can be added (a white mug, a
plant). **Not usable as distractors without a decision:** anything that emits light (a
warm lamp, an LED) -- it changes the lamp condition of the calibration (open item O3).
Anything with a visible brand or text is covered or left out (the red tube with a brand
seen in s11 is replaced).

**Author's inventory (2026-10-28).** A red book, a red toy car, a red PS controller, a red
cardboard sheet, a 5 l bottle with a red cap, and a teapot or a pot (reflective metal).
Conditions: the book's cover and spine text and the car's logo are covered with matte tape;
the controller is switched off, has no cable, its light bar is dark (lit, it would be a
light-emitting object, O3) and its PS logo is covered; the bottle's label is removed and
whether it is filled is recorded; a brand stamp on the teapot/pot is covered. The author
also offered the corridor with the kitchen as background. It would have needed a
recalibration of all three cameras and a change to 13.4 (lamp condition, daylight from the
kitchen window, light-emitting appliances, wall clearance at 3 m); the author agreed to
the room instead, so the "one room, one lamp" rule of 13.4 stands (a second tripod
position in the same room, B, was added on 2026-10-28).

## 7. Materials

- Candles: the same type as `candles4`, at least 3 complete spare sets of four (the
  flame height changes as the candles burn down; open item O4), plus a lighter and a
  snuffer (no blowing, to keep the smoke and the flame position predictable).
- The matte non-reflective base used in s10/s11 (and a spare of the same kind).
- The vertical ruler; floor tape for the marks; the tape measure.
- Backdrops (cloths, boards) and floor covers from section 5, with clamps/stands.
- The unbranded distractors from section 6.
- No inclinometer needed: the tilt of record is 17 deg (prereg 13.1); it is re-checked by
  the image method only if the plate is moved.
- The scene sheet (printed or a file), a timer for the 2 min / 10 s waits.

## 8. Open items for the review

- **O1. Fire/no-fire order in the study. DECIDED 2026-10-13:** per camera, fire
  far-to-near, then no-fire near-to-far (the e5 order), so that the per-node no-fire wait
  of the capture script covers every fire-to-no-fire step (prereg 13.2).
- **O2. One room, one lamp. DECIDED 2026-10-13:** every scene in the pilot/e5 room under
  the ceiling lamp only; disclosed as a limitation (prereg 13.4). Whether the tripod may
  face another wall in some scenes is still open.
- **O3. DECIDED 2026-10-13: no light-emitting object in view** (prereg 13.4). Original question:
  light-emitting distractors (warm lamps): excluded under the current
  calibration rule, or allowed with recalibration?
- **O4. Candle burn-down. DECIDED 2026-10-13:** only between scenes; the set is replaced
  if the ruler flame height at the scene start is outside 3.5 +- 1 cm or the shortest
  candle is at half its height when new; the study starts with a new set (prereg 13.4).
- **O5. The 2-minute rule in s11 (author, 2026-10-09).** The rule was not followed in
  s11. From the capture records (`end_unix` -> `start_actual_unix`) the gap was 51.9 s
  (D435i), 55.8 s (ZED 2i) and 56.2 s (D435if before its update); the log time stamps
  had suggested about 48-53 s. Check (a) (no-fire frames 0 px) passed on all three
  cameras. **Recorded in prereg 13.2 (a9aef35).**
- **O5a. A guard in the code for the study (author, 2026-10-09).** A no-fire capture is
  refused by the capture script if fewer than 120 s have passed since the end of the last
  fire capture of the same scene and distance. Code, test and section 13 before the first
  study capture. **DONE a9aef35**, as "same scene, any distance" (approved by the author
  2026-10-13), per node, `_retakes/` counted. (With the proposed order "no-fire first, then fire" per camera, the
  guard applies whenever a no-fire capture follows a fire capture of the same scene and
  distance, e.g. a retake.)
- **O6. Tilt. DECIDED 2026-10-13:** 17 deg, the s11 tilt from the D435if images (16.4 /
  16.7 deg), held fixed; the phone reading (22 deg) is not used (prereg 13.1).
- **O7. The D435if repeat. DONE 2026-10-12:** all three cameras PASS at 1, 2 and 3 m;
  distance set {1, 2, 3} m (prereg 13.2, a9aef35).
