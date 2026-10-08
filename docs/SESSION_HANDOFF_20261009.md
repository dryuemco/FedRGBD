# Session handoff -- 2026-10-09 (night of 8 to 9 October)

Read this first in a new session; then `CLAUDE.md`, `docs/CAMERA_EXPERIMENT_PREREG.md`
section 13 (Amendment 4), and `logs/e5/e5_capture.log` (desktop only, gitignored; every
e5 command, time and raw output of the night is there). Branch `revision-ncaa`.

## 1. Where things stand

- **e5 (camera flame-source acceptance test) is almost done.** The current, valid round
  is **s11** (clean restart, prereg 13.2 "Clean restart of e5", commit 49d3537). Earlier
  rounds s03/s09/s10 are archived (moved, md5-checked) under
  `data/raw/camera_pilot/_archive_20261009_000714/` on node_b and the desktop, and are
  reported as described in 13.2 (s03 2/3 m "NOT VALIDATED (reflective geometry)").
- **s11 was captured on all three cameras**, one camera at a time from the taped
  reference tripod position, in the e5 order (fire 3-2-1 m, then no-fire 1-2-3 m).
  Results of the flame tool (`scripts/camera_flame_height.py`, rule (a)/(b)/(c) of 13.2,
  tool version 49d3537) -- do not retype numbers, read them from:
  - `logs/e5/flame/s11_all_d100.csv`, `s11_all_d200.csv`, `s11_all_d300.csv` (all three
    cameras; overlays `s11_all_d{100,200,300}.png`);
  - per camera: `logs/e5/flame/s11_node_{a,b}_d{100,200,300}.csv` and `.png`;
  - the raw console output: `logs/e5/e5_capture.log` ("s11 ALL THREE CAMERAS").
  - To regenerate: `python scripts/camera_flame_height.py --root data/raw/camera_pilot
    --scene s11 --distance_m <1|2|3> --out_csv ... --overlay ...` (desktop, after copying
    the nodes' pilot frames; see `scripts/e5/README.md`).
- **The distance decision is NOT written yet.** node_a's s11 is the D435if **before** its
  firmware update; the D435if repeat (section 5) comes first, then the decision.
- **Depth holes** (13.3, rule bbe9c2c; raw output in the e5 log, "DEPTH HOLES s11"):
  the D435if exceeded the firmware rule's threshold at all three distances; the firmware
  was then updated (section 3). After the repeat, the D435if's hole fraction is
  measured again by the same method, descriptive only (019922e).
- **s01 pilot** frames stay in place on all nodes (never moved).

## 2. Physical set-up (as left at about 01:40)

- Tripod at the **taped reference position** (13.2, e0a5593). Plate tilt = the s11 tilt,
  **not measured** (logged as `tilt_deg=unknown`); the D435i IMU gives no data
  (HARDWARE_SETUP.md).
- **ZED 2i is on the plate**, on node_c (USB 3, 5000 Mb/s). Its optical centre is 1.5 cm
  higher than the RealSense's (RealSense 56.5 cm, ZED 58 cm, author's measurement).
- **D435if is off the plate**, plugged into node_a (USB 3.2), firmware **5.17.0.10**
  since 01:26.
- **D435i** is off the plate, plugged into node_b, firmware 5.17.0.10.
- Candles: `candles4` (four in one row along the optical axis, 5 cm gaps, distance to
  the front candle) on the **matte base**, unlit, last at the 3 m mark.
- Lighting: curtains closed, curtain edge fixed, only the ceiling lamp:
  `lamps_on=tavan lambasi` (must match exactly; the code refuses another value).
- Nodes: all three at MAXN_SUPER, no FL process; repo at f9bd254 on the nodes (later
  commits are docs/data only) -- **pull first**.

## 3. Camera settings (one fixed setting per camera, 13.1)

Committed byte copies in `data/camera_exposure/camera_pilot/<node>/exposure.json`
(the live files are `data/raw/camera_pilot/<node>/_exposure/exposure.json` on each node):

| camera | commit | exposure / gain / WB | verification 224 luma | WB rectangle |
|---|---|---|---|---|
| D435i (node_b) | e727516 | 333 / 91 / 4540 | 112.542 | x 0.03-0.30, y 0.10-0.30 |
| D435if (node_a) | 1f6c174 | 333 / 91 / 4510 | 105.467 | x 0.03-0.28, y 0.12-0.30 |
| ZED 2i (node_c) | 57f5f3c | 100 / 10 / 5000 | 104.94 | x 0.05-0.35, y 0.08-0.28 |

(Values read from the committed files; rectangles in `configs/camera_wb_region.json`.)
Superseded, kept under `_superseded/`: node_b `20261008T234719` (= eef07c7, old tripod
position) and `20261009T002253` (= a83a84d, before the new tilt); node_c
`20261009T012839` (invalid: WB search direction bug, fixed in f9bd254). **The D435if's
calibration (1f6c174) predates its firmware update and is repeated in the morning.**
Session checks are appended to `_exposure/session_checks.jsonl` on each node
(`--session_check`, eba04ae).

## 4. Commits of 2026-10-08/09 (all pushed to origin/revision-ncaa)

```
57f5f3c e5 s11: ZED 2i calibrated with the fixed WB search, exposure.json; invalid first calibration in _superseded/
5f0c132 D435if firmware 5.13.0.55 -> 5.17.0.10 (rs-fw-update, rc 0): records after the update
a727863 D435if firmware update: records before the update and the firmware file note
f9bd254 Calibration WB search: direction from the R-B signs at both ends (ZED WB reversed); prereg 13.1
eba04ae Session-start check as a committed tool; WB rectangle node_c
019922e prereg 13.3: after the firmware update the hole fraction is re-measured, descriptive only
bbe9c2c prereg 13.3: depth-hole measurement in each RealSense's WB rectangle, firmware rule
1f6c174 e5 s11: D435if calibrated at the reference position, exposure.json
529fdf5 test_camera_exposure: missing-rectangle test uses its own region file (5c0e62f was committed with it failing)
5c0e62f WB rectangle node_a (D435if)
e727516 e5 s11: D435i recalibrated with the new tilt, exposure.json
2ebc36e WB rectangle node_b for s11 at the new tilt
49d3537 e5 clean restart: s11, decision rule (a)/(b)/(c), e5 order; prereg 13.2
a83a84d e5: D435i recalibrated at the reference tripod position
86001a6 WB rectangle node_b redefined at the reference position
4f4686c prereg 13.2: WB rectangle checked on a framing frame before each calibration
e0a5593 prereg 13.2: reference tripod position; s10 clean geometry
cb1fe85 prereg 13.2: curtain fixed + 10 s settling; 2/3 m repeated in a clean geometry
76a7003 Flame tool fix after e5 data was seen; prereg 13.2
a626d06 prereg 13.1: flame-free check at the start of every capture session
eef07c7 e5: D435i fixed exposure/WB calibration (old tripod position)
46a1320 prereg Amendment 4 (section 13): exposure/WB policy, sequential capture, cancelled e5 steps
58bff21 Camera exposure: one fixed setting per camera
56599a1 prereg Amendment 4: candles4 arrangement
69032ea A5 desktop simulation: 18 cells
baccca6 e5 depth-hole region (node_a, pilot view)
c310d4b prereg Amendment 4: four candles as one source
3681d83 camera_flame_height: PASS/FAIL per camera x distance
ce85a0d prereg Amendment 4: flame-source acceptance criterion
cfbd211 FetchResults: time limits and pop-ups
45150d1 Block A5: the 4 MAXN_SUPER testbed runs
```
(plus the commit that adds this note, `scripts/e5/` and `docs/e5/`.)

**Not committed (on purpose):**
- `docs/HARDWARE_SETUP.md` (two-ZED table, D435I name note, D435i IMU note) and
  `docs/POST_5B_CHECKLIST.md` (e1-e4 DONE notes) -- the author asked to commit them
  earlier; that batch was paused for e5 and is still open (section 6).
- `docs/STUDY_CAPTURE_PLAN_DRAFT.md` -- the study capture plan draft, to be reviewed.
- Everything under `data/raw/camera_pilot/` and `logs/` (gitignored: frames, e5 log,
  flame CSVs/overlays, framing frames).
- (`docs/firmware/D435IF_FIRMWARE_FILE.md` **is** committed, in a727863.)

## 5. Tomorrow morning, in this order

1. `git pull` on all three nodes; check MAXN_SUPER, no FL process.
2. **Session checks** (flame-free, candles unlit at 3 m on the matte base,
   `lamps_on=tavan lambasi`) for the cameras used that day:
   `python3 src/data/<realsense|zed>_capture.py --session_check --node <node> --root data/raw/camera_pilot --notes "source=none; distance_measured_m=3.00; flame_height_cm=0; lamps_on=tavan lambasi"`.
3. **D435if repeat** (firmware 5.17.0.10), D435if mounted on the plate at the reference
   position, plate tilt unchanged:
   framing frame (`scripts/e5/e5_framing_grab_d435if.py`, candles unlit) -> WB rectangle
   check on it (redefine and commit only if it no longer covers plain wall) ->
   `--calibrate_exposure --recalibrate` (moves 1f6c174's file to `_superseded/`) ->
   commit exposure.json -> `--session_check` -> 1 m test frame (candles lit,
   `scripts/e5/e5_flame_position.py`) -> s11 again for node_a only, e5 order (fire 3-2-1,
   then no-fire 1-2-3 after >= 2 min and an ember check; >= 10 s after every movement).
   The repeated node_a s11 replaces the pre-update one; how the old one is kept
   (archive) is to be decided before capturing (scene id stays s11 or a new id?).
4. Depth holes of the D435if again (13.3, descriptive):
   `python scripts/camera_depth_holes.py --root data/raw/camera_pilot --region <json of node_a's WB rectangle> --node node_a --captures s11_no_fire_d100 s11_no_fire_d200 s11_no_fire_d300`.
5. Flame tool on all three cameras' s11 -> **distance decision**, then fill section 13's
   `[e5]` values, record the s11 2-minute-rule deviation (section 6), and commit
   Amendment 4 complete.
5a. Implement the 120 s no-fire guard in the capture script (code + test + section 13),
   before any study capture.
6. Review `docs/STUDY_CAPTURE_PLAN_DRAFT.md` with the author; decide; record the
   decisions in section 13 before the first study capture (study root `data/raw/camera`,
   scene ids from s12).
7. Study capture.

## 6. Open decisions

- 2 or 3 distances per scene (plan options A/B).
- Sequential capture in the study: fire/no-fire order (the prereg fixes none), camera
  order rotation, one room / one lamp (otherwise recalibration), light-emitting
  distractors, candle replacement.
- ZED method notes (in 13.1, f9bd254): WB settings give R-B in coarse steps (4600-4900
  vs 5000-5100 identical); luma flat above exposure 75 (possible cause 15 fps frame
  period, not verified).
- D435if hole result after the firmware update (descriptive).
- **Protocol deviation in s11 (to be written into section 13 tomorrow):** the 2-minute
  rule after putting the candles out was not followed; the logged gap between the end of
  the 1 m fire capture and the start of the 1 m no-fire capture was about 48-53 s on all
  three cameras (e5 log timestamps). Check (a) of the decision rule (no-fire frames give
  0 px) passed on all three cameras (`n_nofire_detected` in
  `docs/e5/records/flame/s11_all_d{100,200,300}.csv`).
- **For the study capture (code + test + section 13, before the first capture):** a
  no-fire capture is refused by the capture script if fewer than 120 s have passed since
  the end of the last fire capture of the same scene and distance.
- **Paused since 2026-10-08 (the author's earlier request, before e5):** A5 analysis
  tables (testbed 4 runs + "desktop simulation sensitivity analysis" 18 cells, fill the
  A5 `\todo`), commit HARDWARE_SETUP.md + POST_5B_CHECKLIST.md e1-e4 notes, fix the
  `tab:global_eval` / `tab:time` overflows (format only).
- node_a holds an untracked `results/_gate_failed/20260928_pc_maxn_rev_iid_fedavg_r10_seed42/`
  that makes `test_main_on_real_repository_results` fail on node_a only; untouched.

## 7. Rules (reminder)

- `CLAUDE.md` hard rules; the pre-registration and its amendments; **rule before data**:
  every decision about a measurement is written into section 13 and committed/pushed
  before the data it applies to is captured or computed.
- No number is typed by hand into the paper or the notes: read from logs/CSVs/analysis.
- Never `grep -i` / `pgrep -i` / `rg -i` / Grep `-i` / `-imatch` / `-ilike`.
- Never commit `*.eml`, never re-derive splits, never touch `results/3node_*`,
  `results/centralized_*`, `results/local_*`.
- Push only with the author's approval; confirm before anything that changes state.
- Nodes never pull while a block runs. Sudo: only `sudo -n nvpmodel -m 2|3` on node_c,
  never while FL runs.
- Pilot/e5 data only under `data/raw/camera_pilot`; technical image properties only,
  never accuracy, loss or predictions. Raw output to the author, no interpretation.
- Before each physical step, tell the author what to do and wait for "ready".

## 8. Files moved out of the session scratchpad

- `docs/e5/records/` -- text copies of the e5 record (gitignored originals stay in
  `logs/e5/`): `e5_capture_20261008_09_log.txt` (the full e5 log up to the s11 flame
  tool run), `flame/*.csv` (all flame-tool CSVs: s03 old/fixed tool, s10, s11 per camera
  and all three), `flame_diag/*.csv` (no-fire-as-fire diagnostics). No PNGs, no frames.

- `docs/e5/E5_COMMAND_SHEET_20261008.md` -- the e5 command sheet (its step list is
  superseded by section 13: sequential capture, s11, cancelled s04-s08/s09 3 m).
- `scripts/e5/` -- framing/test-frame scripts for the D435i, D435if and ZED, the flame
  position helper, the archive script, the (failed) IMU tilt script, the legacy session
  check; usage in `scripts/e5/README.md`.
- `docs/e5/diffs/` -- reviewed drafts of the flame-tool fix and the exposure policy.
- The firmware zip/bin are in the scratchpad only (`fw_download_41896/`) and on node_a
  (`~/fw/`); their SHA256s are in `docs/firmware/D435IF_FIRMWARE_FILE.md`. The original
  zip is in `C:\Users\CORSAIR\Downloads\`.
