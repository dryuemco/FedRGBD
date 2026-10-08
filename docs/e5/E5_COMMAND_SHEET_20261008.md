# e5 command sheet (PREPARED, NOT RUN) -- 2026-10-08, final version

Repo commit on all three nodes and the desktop: `3681d83`.
- `ce85a0d`: the acceptance criterion (prereg 13.2), committed and pushed before e5.
- `3681d83`: `camera_flame_height.py` prints PASS/FAIL against it.

Nothing below has been run. e5 starts only after your approval.
Sources: `docs/POST_5B_CHECKLIST.md` e5, `docs/CAMERA_EXPERIMENT_PREREG.md` sec. 10 and
13 (13.2 criterion committed, the rest still a draft), `scripts/camera_capture_session.py`,
`src/data/{realsense,zed}_capture.py`, `src/data/camera_capture_common.py`,
`scripts/camera_flame_height.py`, `tests/test_camera_e5_gates.py`.

**Pilot rules that apply to every step (prereg sec. 10, checklist e5):**
- Every frame goes to `data/raw/camera_pilot/` only.
- `--root data/raw/camera_pilot` is mandatory on every capture command. The scripts'
  default root is the study tree `data/raw/camera`, which must not exist yet.
- Only technical image properties are checked: flame height in pixels, depth validity,
  luminance. Never accuracy, loss or prediction quality.

## 0. Common

**Shell for every capture command** (all captures are started from node_a):

```bash
# desktop -> node_a, through the Windows OpenSSH client (Git Bash ssh cannot see the key)
C:\Windows\System32\OpenSSH\ssh.exe jetson-a@192.168.1.10
cd ~/FedRGBD && . ~/fedrgbd_venv/bin/activate
```

The orchestrator `scripts/camera_capture_session.py` starts the same capture on all three
nodes over SSH:
- D435if on node_a, D435i on node_b, ZED 2i on node_c (prereg Amendment 1);
- hosts, users and venvs come from `configs/testbed.local.yaml`;
- default lead time 25 s, 40 frames at 5 fps.

**`--notes` is mandatory and parsed** on every capture, including `--no_lock` captures.
The nodes always apply `study_gates=True`.
- Format: `source=<id|none>; distance_measured_m=<x.xx>; flame_height_cm=<x>`
- no_fire: `source=none` and `flame_height_cm=0`.
- fire: `source` must not be `none`, and `flame_height_cm` must be > 0.
- **Lighting (author, 2026-10-08):** curtains closed, artificial light only. Every fire and
  no-fire capture carries `lighting=curtains closed, artificial light only` and
  `lamps_on=<the lamps that are on>` (extra keys are kept in `notes_parsed`; no `;` or `=`
  inside a value).
- `distance_measured_m` must be between 0.3 and 10.
- Fragments without `=` are allowed; they are stored as free text (`text`). I use them
  below to label each test.
- If a field is missing or does not parse, the capture is refused before anything starts.

**Fixed values:**
- `--distance_m` must be exactly 1, 2 or 3.
- Scene ids must match `s\d{2}`.
- Capture id = `<scene>_<label>_d<cm>`. A second capture with the same id is refused
  unless `--retake` is given, which moves the old capture to `_retakes/<stamp>/` and keeps
  it.

**PASS line of the orchestrator (`verify_records`), per node:**
- the record exists and belongs to this session;
- frames written equal frames requested;
- start within 1.0 s of the schedule;
- the expected S/N: node_a 239722070442, node_b 405622076256, node_c 35201583;
- rc 0, status complete;
- USB 3.x: RealSense `usb_type_descriptor` 3.x, ZED sysfs speed >= 5000 Mb/s;
- unless `--no_lock`: an exposure lock was recorded, the camera took it (within 2 %), and
  auto exposure is off in every colour frame (`ae_on_frames == 0` and
  `ae_unknown_frames == 0`).

Each session is appended to `data/raw/camera_pilot/session_log.jsonl` on node_a; node logs
go to `logs/camera/` on node_a.

**Scene ids (decided 2026-10-08).** All of them are captured with
`--root data/raw/camera_pilot` and nowhere else.

| id | use |
|---|---|
| s02 | cancelled (torch; the author's ruler comparison chose the 4 candles, `c310d4b`) |
| s03 | acceptance test, source `candles4` (4 candles side by side = one source), 40 frames |
| s04 | lock test: auto exposure ON (`--no_lock`) |
| s05 | lock test: locked |
| s06 | depth holes (i): all three cameras streaming |
| s07 | depth holes (ii): node_a alone, D435i unplugged |
| s08 | depth holes (iii, optional control): node_a alone, D435i plugged in but idle |
| s09 | 1 m geometry trials (`--retake` per tilt attempt) |

**Execution order:**
- P (prerequisites) -> 4 (1 m geometry) -> 1 (lock test) -> 2 (AE flag on real hardware, frame by frame, uses the
  frames of step 1) -> 3 (depth holes) -> 5 (acceptance) -> copy + analysis.
- Why this order: the geometry fixes the plate tilt that every later capture uses, and
  the acceptance pairs depend on the lock procedure that step 1 settles.

## P. Prerequisites

**[YOU]**
1. The rig is as in e1: the three cameras on one plate, on one tripod, pointing the same
   way. D435if to node_a, D435i to node_b, ZED 2i to node_c, each on a USB 3 port with a
   USB 3 cable. In e1, node_a's D435if came up at USB 2.1 until the cable was replaced.
2. Fix the room lighting for the whole session: the same lamps, curtains closed. Write the
   lamps down for the scene sheet (13.1).
3. Floor tape at 1, 2 and 3 m, measured from the plate's front edge (13.1).

**[CLAUDE]** Checks, read-only:
- no `run_matrix`, `server.py` or `client.py` process on any node;
- no other process holding a camera (in particular no ZED process on node_c);
- `nvpmodel -q` on all three nodes. MAXN_SUPER is not needed to capture, but it stays as
  it is.

Then, on node_a:

```bash
python3 scripts/camera_capture_session.py --check
```

PASS means, on every node:
- SSH answers;
- the SDK imports (pyrealsense2 on node_a/node_b, pyzed on node_c);
- exactly the expected S/N;
- USB 3.x;
- the clock offset to node_a is under 1 s.

FAIL names the problem. Fix it before any capture.

## 4. 1 m geometry (done first)

**Goal (checklist e5, prereg 13.2 "Distance set"):** a plate tilt that keeps a 1 m flame
inside all three colour images and inside the 224 centre crop, while the 3 m flame stays
there too and depth stays valid. Record the rig height and the tilt. If no tilt works,
the distance set becomes {2, 3} m.

**[YOU]**
1. Set a tilt. Measure the rig height above the floor at the plate (cm) and the plate
   tilt (deg; a phone inclinometer is enough).
2. Put the flame source at the 1 m tape and light it. Read the visible flame height off a
   vertical ruler next to the flame.
3. After the 1 m pair, move the source to the 3 m tape for the second pair.
4. If a tilt fails, change it and repeat the four captures with `--retake`. Note the
   height and tilt of every attempt.

**[CLAUDE]** Four short captures per tilt attempt, on node_a, without a lock:

```bash
# no-fire reference, then fire, at 1 m
python3 scripts/camera_capture_session.py --root data/raw/camera_pilot --scene s09 --label no_fire --distance_m 1 --frames 5 --no_lock \
  --notes "source=none; distance_measured_m=1.00; flame_height_cm=0; tilt_deg=<deg>; rig_height_cm=<cm>; e5 geometry; lighting=curtains closed, artificial light only; lamps_on=<lamps on>"
python3 scripts/camera_capture_session.py --root data/raw/camera_pilot --scene s09 --label fire --distance_m 1 --frames 5 --no_lock \
  --notes "source=candles4; distance_measured_m=1.00; flame_height_cm=<ruler cm>; tilt_deg=<deg>; rig_height_cm=<cm>; e5 geometry; source chosen by author ruler comparison: torch flame smaller than the 4 candles; lighting=curtains closed, artificial light only; lamps_on=<lamps on>"
# the same pair at 3 m (--distance_m 3, distance_measured_m=3.00)
# another tilt attempt: the same commands with --retake
```

**Check [CLAUDE]:**
1. Copy the frames to the desktop (section C).
2. Run the flame tool for each distance:

   ```bash
   python scripts/camera_flame_height.py --root data/raw/camera_pilot --scene s09 --distance_m 1 --overlay e5_geom_d100.png
   ```

3. The geometry passes when, on every node:
   - the flame is detected (`n_frames_detected > 0`). These are 5-frame captures, so the
     tool's `criterion` column reads INCOMPLETE here. That is expected: the acceptance
     verdict comes only from the 40-frame captures of step 5;
   - `touches_crop_border` is false at both 1 m and 3 m;
   - the depth of the s09 no-fire frames is valid in the scene region. There is no
     committed tool for this; I read the invalid fraction of the depth PNGs (0 = invalid)
     ad hoc.

**Open issues:**
- **No snapshot tool exists.** 13.1 requires "one snapshot per camera" before every
  capture, but nothing in the repo takes one. The `--test` smoke of each capture script
  writes to a temp dir outside the pilot tree, so I use these short pilot-tree captures as
  the snapshots instead.
- **Source:** `candles4` throughout (`c310d4b`).

## 1. Exposure-lock behaviour (13.1 "Open until e5")

**Question:** when auto exposure is switched off, does a RealSense keep the last auto
value, or fall back to the manual option value? And the same with `AEC_AGC` off on the ZED.

How the code fixes a lock:
- RealSense under RSUSB (`settle_auto`): auto exposure runs for 45 frames, is switched
  off, and then the sensor options are read. If the camera falls back to the option value,
  that value is not what auto exposure used.
- ZED: the values auto exposure applied are read while it is still on.

**[YOU]**
1. Rig at the tilt from step 4. No flame, scene static, lighting as fixed in P.
2. Nothing moves between the two captures; they run directly one after the other.

**[CLAUDE]** On node_a:

```bash
# (a) auto exposure ON in every frame (test only)
python3 scripts/camera_capture_session.py --root data/raw/camera_pilot --scene s04 --label no_fire --distance_m 2 --frames 5 --no_lock \
  --notes "source=none; distance_measured_m=2.00; flame_height_cm=0; e5 lock test AE on; lighting=curtains closed, artificial light only; lamps_on=<lamps on>"
# (b) immediately after: locked (auto exposure settles for 45 frames, the values are locked, then 5 frames)
python3 scripts/camera_capture_session.py --root data/raw/camera_pilot --scene s05 --label no_fire --distance_m 2 --frames 5 \
  --notes "source=none; distance_measured_m=2.00; flame_height_cm=0; e5 lock test locked; lighting=curtains closed, artificial light only; lamps_on=<lamps on>"
```

**Outputs:**
- `data/raw/camera_pilot/<node>/s04_no_fire_d200_*` and `s05_no_fire_d200_*`;
- `_captures/*.json` with `exposure_lock` (values, source string, `applied` read-back),
  `ae_on_frames` and `ae_unknown_frames`;
- `_locks/s05_d200.json` on every node.

**PASS / FAIL:**
- (a) PASS means the normal checks only; the lock check is skipped. Its log ends with
  "NOTE: --no_lock -- a test capture".
- (b) PASS requires the lock to be applied (within 2 %) and `ae_on_frames = 0`,
  `ae_unknown_frames = 0` on all three nodes.

**Analysis [CLAUDE, on the desktop after copying; no committed tool, ad hoc]:**
1. Mean luminance at 224 of the 5 frames of (a) against the 5 frames of (b), per camera.
2. The lock values in (b) against the per-frame exposure, gain and white balance in (a).
3. Result for 13.1:
   - luminance unchanged: the camera keeps the last auto value, and the procedure stays;
   - luminance jumps: it falls back to the option value. The lock procedure has to be
     changed, and that change is a dated technical amendment before any study capture.

**Gaps:**
- **Not the checklist's within-stream comparison.** The checklist asks for 5 frames
  before and 5 after the switch in one stream. The code cannot save frames from the
  settle phase, so (a) and (b) are two captures about 40 s apart. With fixed lighting
  that is equivalent, but it is not the literal procedure.
- **Depth-frame exposure.** The checklist also says to read the depth frame's
  `actual_exposure`. The code stores only the IR/stereo exposure, as `ir_exposure`, which
  belongs to the stereo module and not to the colour sensor. Nothing else is recorded.

## 2. AE gate on real hardware: the flag the gate reads, per frame

**What is tested on the cameras.** The gate's input is the per-frame AE flag:
- in the `auto_exposure` key of every `<frame_id>_meta.json`;
- RealSense: the colour frame's `auto_exposure` metadata;
- ZED: the `AEC_AGC` setting read back.

On real hardware it must read:
- **"on" (true) in every frame of the `--no_lock` capture s04**, which shows the gate
  really sees auto exposure on;
- **"off" (false) in every frame of the locked capture s05**.

Both are reported **frame by frame**, on all three cameras. There is no extra capture:
these are the frames of step 1.

**What stays tested on fake cameras only (stated as such in the Amendment 4 record).**
The FAIL branch itself: a *locked* capture whose flag is on in some frame, or is not
reported, FAILs.
- Tests: `test_auto_exposure_on_in_a_locked_capture_fails_it` and
  `test_unreported_auto_exposure_flag_fails_a_locked_capture` in
  `tests/test_camera_e5_gates.py`.
- No CLI path produces a locked capture with auto exposure on. `--no_lock` switches the
  lock check off altogether, so this branch cannot be triggered on real hardware without
  a test-only code hook, and none exists.
- The real-hardware test shows that the gate's input reads correctly in both states. The
  fake-camera tests show that the gate acts on that input.

**[CLAUDE]** Desktop, after the copy (section C), with the CPU venv:

```bash
~/venvs/fedrgbd/Scripts/python.exe - <<'EOF'
import glob, json, os
want = {"s04": True, "s05": False}            # s04 --no_lock: AE on; s05 locked: AE off
ok = True
for node in ("node_a", "node_b", "node_c"):
    for scene, expect in want.items():
        files = sorted(glob.glob(os.path.join("data/raw/camera_pilot", node,
                                              "%s_no_fire_d200_[0-9][0-9][0-9][0-9]_meta.json" % scene)))
        flags = [json.load(open(f, encoding="utf-8")).get("auto_exposure") for f in files]
        for f, v in zip(files, flags):
            print("%s %s %s auto_exposure=%s" % (node, scene, os.path.basename(f)[:-10],
                                                 {True: "on", False: "off", None: "NOT REPORTED"}[v]))
        good = len(flags) == 5 and all(v is expect for v in flags)
        ok &= good
        print("%s %s: %d frames, on=%d off=%d unreported=%d -> %s" % (
            node, scene, len(flags), flags.count(True), flags.count(False), flags.count(None),
            "PASS" if good else "FAIL"))
print("AE flag on real hardware:", "PASS" if ok else "FAIL")
EOF
```

- **PASS:** on every node, s04 has 5 frames and every one reads `on`, and s05 has
  5 frames and every one reads `off`.
- **FAIL otherwise.** A frame reading `NOT REPORTED` is a FAIL too.
- Also check that the s05 capture record has `ae_on_frames = 0` and
  `ae_unknown_frames = 0`. The orchestrator's PASS line already requires this.

**Gaps:**
- **The RealSense flag source is not recorded.** If the colour-frame `auto_exposure`
  metadata is missing, `realsense_capture.py` (l. 546-548) falls back to the sensor
  option `enable_auto_exposure`, which the code itself has just set. The frame meta does
  not record which source was used.
  - The pilot probe says the RSUSB colour frame does deliver `auto_exposure`, so the
    fallback should not trigger, but the saved data cannot prove it.
  - On the ZED, the flag is the `AEC_AGC` setting read back, not per-frame metadata.

## 3. Depth-hole diagnosis (D435i plugged in / unplugged)

The checklist (e5), quoted:
> "node_a only (`realsense_capture.py --frames 5` into the pilot tree), same plate and
> scene, twice: (i) all cameras as in e3; (ii) node_b's D435i emitter off -- simplest by
> unplugging the D435i for the shot, which removes its projector. Compare the
> invalid-depth fraction on the right-hand wall region between (i) and (ii)."

So "takılı/sökülü" means the D435i's USB cable on node_b: plugged in vs. unplugged.

**Inconsistency, important:**
- In e3, all three cameras streamed at the same time, so the D435i's projector was on.
- A node_a-only capture with the D435i plugged in but idle has no D435i projector either:
  the emitter only fires while the D435i streams depth. A literal (i) would therefore not
  reproduce e3.
- Fix: run (i) with the orchestrator (all three stream), and (ii) as a node_a-only
  capture with the D435i unplugged.
- The optional (iii), node_a-only with the D435i plugged in but idle, separates "unplugged"
  from "not streaming".

**[YOU]**
1. Same plate, tilt and scene as in step 4. No flame. The right-hand wall in view.
2. (i): all cameras plugged in.
3. Before (ii): unplug the D435i's USB cable at node_b, or at the camera.
4. After (ii): plug it back in, into a USB 3 port.

**[CLAUDE]** On node_a:

```bash
# (i) all three cameras stream, as in e3
python3 scripts/camera_capture_session.py --root data/raw/camera_pilot --scene s06 --label no_fire --distance_m 2 --frames 5 --no_lock \
  --notes "source=none; distance_measured_m=2.00; flame_height_cm=0; e5 depth holes (i) all cameras streaming; lighting=curtains closed, artificial light only; lamps_on=<lamps on>"
# (ii) after the D435i is unplugged: node_a alone
python src/data/realsense_capture.py --scene s07 --label no_fire --distance_m 2 --node node_a --root data/raw/camera_pilot --frames 5 --no_lock \
  --notes "source=none; distance_measured_m=2.00; flame_height_cm=0; e5 depth holes (ii) D435i unplugged; lighting=curtains closed, artificial light only; lamps_on=<lamps on>"
# (iii, optional) after the D435i is plugged back in, node_a alone again
python src/data/realsense_capture.py --scene s08 --label no_fire --distance_m 2 --node node_a --root data/raw/camera_pilot --frames 5 --no_lock \
  --notes "source=none; distance_measured_m=2.00; flame_height_cm=0; e5 depth holes (iii) D435i plugged idle; lighting=curtains closed, artificial light only; lamps_on=<lamps on>"
# after re-plugging: confirm node_b again (serial and USB 3.x)
python3 scripts/camera_capture_session.py --check
```

- The direct `realsense_capture.py` prints one JSON line with `status`, which must be
  `complete`. The USB 3.x gate applies; the lock does not (`--no_lock`).
- **Analysis [CLAUDE, ad hoc, no committed tool]:** the fraction of depth == 0 in the
  right-hand wall region of node_a's `*_depth.png`, (i) vs. (ii) (vs. (iii)).
  - Holes gone in (ii): they come from the second projector, and Amendment 4 decides
    what to do.
  - Holes still there: firmware is the next suspect; that decision comes after e5, in
    13.3.
- The wall region is not defined in any doc. I would fix it as a rectangle in pixels
  before looking at (ii), and record it.

## 5. Flame-source acceptance test

**Criterion (prereg 13.2, committed `ce85a0d` before e5):**
- For every camera x distance, the **median** vertical flame height in the 224 input,
  over the **40 frames** of the fire capture, must be **>= 5 px**.
- A frame with no visible flame counts as 0 px.
- A distance that fails on any camera is removed from the distance set **for every
  scene**.
- `camera_flame_height.py` (`3681d83`) prints this verdict itself, so nothing is compared
  by hand.

**Source (decided, `c310d4b`):** four candles side by side as one source (`candles4`): the
same four, in the same arrangement, lit together. The author's ruler comparison: the torch
gave a smaller flame than the 4 candles. s02 (torch) is cancelled; only s03 is captured.

**[YOU]** (s03, `candles4`):
1. Rig at the tilt from step 4, lighting fixed.
2. Put the source at the 1 m tape, unlit, with a ruler standing vertically beside the
   flame position, in frame.
3. Tell me "no-fire ready": I capture no_fire. This fixes the lock for scene x distance.
4. Light the source, wait until the flame is steady, and read the visible flame height
   on the ruler in cm. Tell me "fire ready, <cm>": I capture fire.
5. Repeat steps 2-4 at 2 m and 3 m. Skip 1 m if step 4 found no tilt that works.
6. Measure the real distance on the tape for each shot, to the cm.

**[CLAUDE]** On node_a, for d in 1, 2, 3; no-fire always before fire, because fire reuses
the no-fire lock and is refused without it:

```bash
python3 scripts/camera_capture_session.py --root data/raw/camera_pilot --scene s03 --label no_fire --distance_m <d> --frames 40 \
  --notes "source=none; distance_measured_m=<x.xx>; flame_height_cm=0; lighting=curtains closed, artificial light only; lamps_on=<lamps on>"
python3 scripts/camera_capture_session.py --root data/raw/camera_pilot --scene s03 --label fire --distance_m <d> --frames 40 \
  --notes "source=candles4; distance_measured_m=<x.xx>; flame_height_cm=<ruler cm>; source chosen by author ruler comparison: torch flame smaller than the 4 candles; lighting=curtains closed, artificial light only; lamps_on=<lamps on>"
```

- 40 frames, locked, full gates. PASS as in section 0.
- If a capture fails, repeat it with `--retake`. A no-fire retake before the fire capture
  re-locks and keeps the old lock in `_locks/_superseded/`. After the fire capture, a
  no-fire retake keeps the pair on one lock.

**Measurement [CLAUDE, desktop, after copying]:**

```bash
python scripts/camera_flame_height.py --root data/raw/camera_pilot --scene s03 --distance_m <d> \
  --out_csv e5_flame_s03_d<cm>.csv --overlay e5_flame_s03_d<cm>.png
```

- **Per node (CSV):**
  - `fy_224` and `expected_px` (from the ruler cm in the notes);
  - `measured_px_median`, `measured_px_min`, `measured_px_max`. Frames without a flame
    count as 0;
  - `ratio_measured_expected`, `n_fire_frames`, `n_frames_detected`,
    `touches_crop_border`;
  - `criterion_px` = 5 and `criterion` = PASS / FAIL / INCOMPLETE. INCOMPLETE means the
    capture does not have exactly 40 frames.
- **On stderr:** one line per camera, then the distance decision:
  - `distance <d> m: PASS on all three cameras (...)`
  - `... FAIL -- removed from the study's distance set (...)`
  - `... NOT DECIDED -- ...`: fewer than three cameras measured, or an INCOMPLETE capture.
- A study-root path is refused.
- These results fill the `[e5]` marks in 13.2: the chosen source, the median per camera
  x distance, the cm from the ruler, and the resulting distance set.

## C. Copying the e5 frames to the desktop [CLAUDE]

There is no committed copy script; e4 copied ad hoc with a per-file md5 check. The same
procedure, for s02-s09 only (s01 is already on the desktop and identical), from Git Bash
on the desktop:

```bash
SSH=/c/Windows/System32/OpenSSH/ssh.exe; cd ~/projects/FedRGBD
for hn in "jetson-a@192.168.1.10 /home/jetson-a/FedRGBD node_a" "jetson-b@192.168.1.7 /home/jetson-b/FedRGBD node_b" "yunus@192.168.1.6 /home/yunus/FedRGBD node_c"; do
  set -- $hn
  $SSH $1 "cd $2/data/raw/camera_pilot && tar cf - --exclude='$3/s01_*' --exclude='$3/_captures/s01_*' $3" | tar xf - -C data/raw/camera_pilot
  $SSH $1 "cd $2/data/raw/camera_pilot && find $3 -type f ! -name 's01_*' ! -path '*/_captures/s01_*' | LC_ALL=C sort | xargs md5sum" > /tmp/e5_$3.remote.md5
  (cd data/raw/camera_pilot && find $3 -type f ! -name 's01_*' ! -path '*/_captures/s01_*' | LC_ALL=C sort | xargs md5sum | sed -E 's/ [ *]/  /') > /tmp/e5_$3.local.md5
  diff /tmp/e5_$3.remote.md5 /tmp/e5_$3.local.md5 && echo "$3 identical"
done
# node_a's session_log.jsonl: the desktop copy must be a prefix of it; then replace it
```

- `data/raw/camera_pilot/` stays outside every manifest and training run. The flame tool
  and the ad hoc analyses read only that tree.
- Use the scratchpad instead of `/tmp` when this runs from Claude.

## Undetermined / to decide before e5 runs

**Decided:** N = 5 px with the median over 40 frames (`ce85a0d`); the tool's PASS/FAIL
(`3681d83`); scene ids s02-s09, all under `--root data/raw/camera_pilot`.

1. **The source:** decided, `candles4`, s03 only (`c310d4b`).
2. **The right-hand wall rectangle:** decided, `configs/e5_depth_hole_region.json`
   (`baccca6`), `scripts/camera_depth_holes.py`.
3. **Where e5 departs from the docs.** Each of these goes as a sentence into the
   Amendment 4 record:
   - the lock test compares two captures, not frames before and after the switch in one
     stream;
   - the AE gate: the flag it reads is tested on real hardware, frame by frame (on with
     `--no_lock`, off when locked). The FAIL branch is tested on fake cameras only;
   - the depth-hole (i) shot runs with all three cameras streaming.
4. **Firmware decision** (13.3) after e5. Any update is [YOU], with sudo.
