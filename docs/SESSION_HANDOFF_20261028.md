# Session handoff -- 2026-10-28

Read this first in a new session; then `CLAUDE.md` and, for the camera experiment,
`docs/CAMERA_EXPERIMENT_PREREG.md` section 13.4. Branch `revision-ncaa`. The previous
handoff (`SESSION_HANDOFF_20261013.md`) still holds for everything not mentioned here
(its section 0a lists the author's other open answers: grant number, AI wording, reading
the narrative text, testbed photo, upload package).

## 0. Where we stopped (tell the author this first)

- **Section 13 of the camera prereg is complete and in force** (DRAFT lifted, committed
  and pushed 2026-10-28). 13.4 now records:
  - position B: second taped tripod position in front of the closed curtain, facing the
    white wardrobe (plate front edge to wardrobe 4.5 m; side walls >= 1 m from the 3 m
    source, author); A = reference position;
  - s12-s20 at A, s21-s29 at B; the tripod moves once, after s20; tilt check
    (D435if, 17 +- 1 deg) after every move to B, even though the author expects none;
  - session checks always at A on the calibration set-up (no rack, no kilim, no study
    distractor, candles unlit at 3 m), the scene is set up afterwards;
  - the 18-scene table (backdrop none/blanket/shirt on a drying rack x floor
    laminate/kilim 1/kilim 2 x two distractors), designed after R3.3: the two scenes
    sharing backdrop and floor differ in position, distance pair, camera order and both
    distractors;
  - placement rules (rack >= 0.5 m behind the 3 m mark, taped per position; kilim along
    the axis, far end at 3.5 m, never under a tripod leg; distractors >= 0.5 m from every
    flame position, small left / large right at A, swapped at B, visible in all three
    224 crops) and the materials.
- **The author is preparing the room** and will come back when it is ready. Nothing was
  captured. The corridor-with-kitchen option was considered and dropped (recalibration,
  daylight, appliance LEDs, wall clearance); recorded in 13.4.

## 1. Inventory (author, 2026-10-28)

- Candles: 16 new of the `candles4` type (four sets) and a snuffer. The e5 set is not
  used (height when new not recorded). Larger candles also bought: another source type,
  not used.
- Backdrops: drying rack with the blanket (white and orange) or a red football shirt
  hung inside out (print that shows through is taped).
- Floors: laminate; kilim 1 mustard and blue; kilim 2 blue and black; each about 3-4 m.
- Distractors: red book (cover/spine text taped), red toy car (logo taped), red PS
  controller (off, no cable, light bar dark, PS logo taped), red cardboard sheet, 5 l
  bottle with red cap (label removed, **half full** -- author, 2026-10-28; kept so in
  all its scenes), teapot or pot (the author decides at the first pot scene, s14; the
  same one in all six scenes, named in the scene sheet).

## 2. When the author is back, before s12

1. Nodes: power-on checks as in the previous handoff (MAXN_SUPER on all three, WiFi
   off, NTP synchronised, GUI off, no FL process, `logs/resume.log` "no restart"), and
   check that `data/raw/camera` does not exist on any node.
2. Offer the filled scene sheet for s12-s29 (columns of
   `data/raw/camera/scenes_template.csv`, plus position, backdrop, floor, rack distance,
   distractor positions, candle heights when new, ruler readings).
3. Session start at A: session checks of 13.1 on the calibration set-up, then the D435if
   tilt check, then set up s12 (A, {1,2} m, camera order 0, no backdrop, laminate,
   book + cardboard). New candle set: measure each candle's height into the scene sheet
   first, then the ruler check (3.5 +- 1 cm).
4. `--notes` of every study capture add `position=`, `backdrop=`, `floor=`,
   `distractors=` and `camera_order=` to the fields of 13.1/13.2.
