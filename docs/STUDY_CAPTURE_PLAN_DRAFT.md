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

## 2. Two options

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

**Per scene** (one camera block = its captures + the gaps between them; the >= 2 min
after putting out the candles overlaps the next camera swap -- see section 4):

| | (A) 3 distances | (B) 2 distances |
|---|---|---|
| captures per camera | 6 | 4 |
| camera block | 6 x 14 s + 5 x 40 s = 4.7 min | 4 x 14 s + 3 x 40 s = 2.9 min |
| + swap (3 cameras) | 3 x (4.7 + 3) = 23 min | 3 x (2.9 + 3) = 18 min |
| + scene set-up | 15 min | 15 min |
| **per scene** | **about 38 min** | **about 33 min** |
| all scenes | 15 x 38 = 9.5 h | 18 x 33 = 9.9 h |
| + session starts (one per day, 12 min) | about 10 h, two days | about 10 h, two days |

Both options take about the same time; (B) gives 3 more scenes for the LOSO folds, (A)
gives every scene all three distances. Neither estimate includes breaks, candle changes or
a recalibration.

## 4. Per scene (proposal)

1. **Scene set-up.** Change the background and the floor (section 5), place the
   distractors, tape the 1/2/3 m marks from the tripod's front edge, fill in the scene
   sheet: background, floor, distractors (each named, unbranded), lamps (`tavan lambasi`),
   source `candles4`, distances, tilt (if measured). Curtain edge fixed.
2. **Camera order rotates across scenes** (Latin square), so no camera is always first:

   | scene | 1st | 2nd | 3rd |
   |---|---|---|---|
   | s12, s15, s18, ... | D435i (node_b) | D435if (node_a) | ZED (node_c) |
   | s13, s16, s19, ... | D435if | ZED | D435i |
   | s14, s17, s20, ... | ZED | D435i | D435if |

3. **Per camera, proposed order: all no-fire first, then all fire** (open item O1).
   - no-fire 1 -> 2 -> 3 m (candles unlit, cold, no ember), each after >= 10 s still;
   - light the candles, wait for steady flames;
   - fire 3 -> 2 -> 1 m (moving the lit base), each after >= 10 s still;
   - put the candles out: **>= 2 min and an ember check** before the next camera's
     no-fire. The camera swap (about 3 min) covers the wait if the candles go out before
     the swap starts; the 2 min is timed, not assumed.
   - (B) the same with the scene's two distances.
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

## 6. Distractors (unbranded; at least one fire-like per scene)

Fire-like, unbranded: a plain red/orange mug; a red or orange book with a blank cover; a
red toy car (logo covered); an orange ball; a red/orange cloth; a copper or brass pot
(reflections); a plain orange plastic box. Neutral ones can be added (a white mug, a
plant). **Not usable as distractors without a decision:** anything that emits light (a
warm lamp, an LED) -- it changes the lamp condition of the calibration (open item O3).
Anything with a visible brand or text is covered or left out (the red tube with a brand
seen in s11 is replaced).

## 7. Materials

- Candles: the same type as `candles4`, at least 3 complete spare sets of four (the
  flame height changes as the candles burn down; open item O4), plus a lighter and a
  snuffer (no blowing, to keep the smoke and the flame position predictable).
- The matte non-reflective base used in s10/s11 (and a spare of the same kind).
- The vertical ruler; floor tape for the marks; the tape measure.
- Backdrops (cloths, boards) and floor covers from section 5, with clamps/stands.
- The unbranded distractors from section 6.
- A phone inclinometer (tilt is still "unknown").
- The scene sheet (printed or a file), a timer for the 2 min / 10 s waits.

## 8. Open items for the review

- **O1. Fire/no-fire order in the study.** The pre-registration does not fix it.
  Proposal: per camera, no-fire 1-2-3, then fire 3-2-1. Needs a decision and a dated
  entry in 13 before the first study capture.
- **O2. One room, one lamp?** If scenes go to other rooms or lamps, every lamp condition
  needs its own calibration of all three cameras (13.1), and the session checks must
  follow it.
- **O3. Light-emitting distractors** (warm lamps): excluded under the current
  calibration rule, or allowed with recalibration?
- **O4. Candle burn-down.** When are the four candles replaced (a ruler height, or a
  number of fire captures)? The flame height in `--notes` records it either way.
- **O5. The 2-minute rule in s11 (author, 2026-10-09).** The rule was not followed in
  s11: the logged gap between the end of the 1 m fire capture and the start of the 1 m
  no-fire capture was about 48-53 s on all three cameras (node_b 00:30:47 -> 00:31:35,
  node_a 00:51:29 -> 00:52:22, node_c 01:35:04 -> 01:35:55). Check (a) (no-fire frames
  0 px) passed on all three cameras. It is written into section 13 as a protocol
  deviation.
- **O5a. A guard in the code for the study (author, 2026-10-09).** A no-fire capture is
  refused by the capture script if fewer than 120 s have passed since the end of the last
  fire capture of the same scene and distance. Code, test and section 13 before the first
  study capture. (With the proposed order "no-fire first, then fire" per camera, the
  guard applies whenever a no-fire capture follows a fire capture of the same scene and
  distance, e.g. a retake.)
- **O6. Tilt.** Still "unknown"; measure it once with the inclinometer before the study
  and record it.
- **O7. The D435if repeat** (firmware 5.17.0.10, recalibration, s11, holes) comes first
  tomorrow; the plan assumes it passes.
