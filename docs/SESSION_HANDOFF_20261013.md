# Session handoff -- 2026-10-13 (evening of 12 October to the night of 13 October)

Read this first in a new session; then `CLAUDE.md` and, for the camera experiment,
`docs/CAMERA_EXPERIMENT_PREREG.md` section 13. Branch `revision-ncaa`, everything below is
committed and pushed; all three nodes are at the last commit of the session.
The previous handoff (`SESSION_HANDOFF_20261009.md`) is superseded.

## 0. Update, morning of 13 October (read this first)

- **Power cut** about 07:59 (desktop event log), nodes back at 08:12, desktop at 09:31.
  Nothing was running (the night session had ended at 01:25, everything pushed). Checked
  after the cut: all three nodes MAXN_SUPER, WiFi off, NTP synchronised, GUI off, no FL
  process; `resume_after_reboot` correctly logged "no restart" (its 07:52 time stamp was
  written before NTP sync). Nodes are still at `8dad714` (docs-only commits since).
- **Author's decisions of section 4 below, as applied:**
  1. Plate tilt **17 deg**, held fixed (prereg 13.1, `12f1e9a`).
  2. Low-data wording rewritten to the measured coverage (`d2438ad`): the retained
     sequences hold 97-99 % of a node's frames; the dropped ones are short.
  3. Timing ratio method approved, declared (`docs/STRATEGY_TIMING_RATIO.md`, `32f4e29`)
     and computed (`aa75664`): FedProx/FedAvg steady factor 1.55 (heterogeneous).
  4. A5 text: the author will read it.
  5. Study capture: decided by the author and recorded in prereg 13.4 (`905aa63`,
     `843c5ea`): 18 scenes x 2 distances (s12-s29, pair by scene id), one room / one
     lamp, candles changed only between scenes (ruler 3.5 +- 1 cm or half height; start
     with a new set), no light-emitting object in view, Latin-square camera order crossed
     with the distance pair, tripod may be turned (tilt re-checked after each turn).
     **Still open before the DRAFT of section 13 is lifted:** the 18 background/floor
     pairs and the materials (needs the author's inventory).
  6. Narrative: every FLAME `\todo` written (see `docs/REVISION_CHANGES.md`, "Narrative
     stage, FLAME"), including abstract and conclusion; **the author reads it next**.
     Remaining `\todo`s: camera experiment, Level-1 calibration table, testbed photo,
     funding/AI wording, final scene count.
  7. Author-only items: later, low priority.
  8. node_a `results/_gate_failed/`: left in place (decided; reasons in
     `docs/POST_5B_CHECKLIST.md` d2). On node_a the real-results test is expected to fail.

## 0a. Update, evening of 13 October -- "where did we stop" (tell the author this first)

- **Done** (`0ffa8aa`, pushed): affiliation = Department of Computer Engineering, Cukurova
  University; abstract shortened to about 190 words, every number kept (the camera result
  is to be merged into the camera sentence with at most ~12 words, so it stays ~200); new
  `Declarations` section: funding Cukurova University, no competing interests, ethics not
  applicable, data: FLAME public + manifests/results/analysis in the repo, **camera
  recordings on reasonable request** (author), code = the public repo. Acknowledgment
  "The author". 920 tests pass, 32 pages, no overfull box.
- **Waiting for the author (asked on 13 Oct evening):**
  1. A grant / BAP project number? (none -> remove the acknowledgment `\todo`)
  2. Keep the AI-assistance wording as it is?
  3. **Study-capture inventory** (blocks the 18 background/floor pairs, the materials list
     and lifting the DRAFT of prereg section 13). The author has: 4 candles, a red book, a
     red toy car, a red toothpaste tube. Asked for:
     - were these 4 candles burnt in e5? (then not usable: height when new unknown);
       buy **at least 4 new sets (16 candles) of the same type as `candles4`** (ask before
       buying another type: 13.4 requires the same type) and a snuffer;
     - the toothpaste's brand/text fully covered with matte tape (the branded red tube of
       s11 was to be replaced);
     - 3-5 more non-emitting distractors from home: orange/mandarin, red apple or pepper,
       an orange/red towel, a steel pot/tray or aluminium foil (reflection), an orange
       plastic box;
     - backgrounds/floors: colours and rough sizes of sheets, duvet covers, blankets,
       rugs, mats at home; for each direction the tripod could face, the free distance to
       the wall (needs about 3 m for {1,2} m scenes, about 4 m for scenes with 3 m).
       Proposal: 6 backgrounds x 6 floors, each used 3 times -> 18 distinct pairs.
       Coverage at 3.5 m on the ZED crop about 1.2 m high x 3.6 m wide (RealSense about
       0.85 x 2.8 m). Safety: candles on the matte base, textiles >= 0.5 m from flames,
       water / extinguisher at hand.
  4. Still to read: abstract, conclusion, A5 text and all narrative sections.
  5. Later: testbed photo (Fig. 1); journal limits (keywords: 9 now -- check the NCAA
     guidelines, the page needed a login); Editorial Manager upload: **editable source
     files only, no PDF** (editor's letter), the new title in the title field, the response
     letter.
- **Claude's own open items:** response letter (R3.4 old mixed-aggregation numbers, stale
  "PENDING EXPERIMENT" labels); a single-file `.tex` + figures package for the upload;
  after the study capture: camera analyses, Level-1 calibration table, camera `\todo`s.
- **Nodes:** checked 22:59, all three idle (no FL / capture process), at `8dad714`,
  tracked tree clean. The author asked to shut them down; a shutdown needs sudo, which
  CLAUDE.md leaves to the author, so the author shuts them down. After the next power-on,
  check as after the power cut: MAXN_SUPER on all three, WiFi off, NTP synchronised,
  GUI off, no FL process, `logs/resume.log` "no restart".

## 1. Where things stand

- **All FLAME federated runs are done and analysed**: the 98-run heterogeneous matrix, 5a,
  5b (MAXN, ten rounds) and A5 (Dirichlet draws). Paper tables come from `analysis/`; a
  full re-run of `analyze_results.py` reproduces the committed CSVs (checked 13 Oct).
- **e5 (camera flame-source acceptance) is closed.** The D435if was recalibrated after its
  firmware update and its s11 retaken (`--retake`, prereg 13.3). All nine camera x
  distance cells PASS, so the study's distance set is {1, 2, 3} m. Every `[e5]` value of
  Amendment 4 is filled. **Section 13 is still marked DRAFT**: it is lifted together with
  the study-capture decisions (section 4 below).
- **Capture script:** a no-fire capture is refused < 120 s after the node's last fire
  capture of the same scene (any distance). The study order is fixed: per camera fire
  far-to-near, then no-fire near-to-far (prereg 13.2, O1).
- **Paper:** every table builds without an overfull box. A5 is in the paper
  (`tab:a5_testbed`, supplementary `tab:a5_desktop_sim`, `tab:a5_skew`) with its methods
  paragraph. About 45 `\todo`s remain, most of them "Narrative" or camera results.

## 2. Physical set-up (as left at about 23:52 on 12 Oct)

- Tripod at the taped reference position, plate tilt unchanged since s11.
- **D435if on the plate**, on node_a (USB 3.2). ZED 2i off the plate, still plugged into
  node_c. D435i off the plate, not plugged in (node_b showed no RealSense on 12 Oct).
- `candles4` unlit on the matte base at the 3 m mark. Curtain closed, its left edge
  moved back to the room corner on 12 Oct (it had moved since 9 Oct) and fixed. Only
  the ceiling lamp (`lamps_on=tavan lambasi`).
- Nodes: MAXN_SUPER, no FL process, WiFi off, clocks synchronised.

## 3. Commits of 12/13 Oct

```
9de4273 prereg 13.3: D435if repeat keeps scene id s11, --retake (before any frame of the repeat)
cee5ee3 D435if recalibrated (333 / 91 / 4540), old calibration in _superseded/
fe2909e e5 repeat records as text; scripts/e5/e5_test_grab_d435if_fixed.py
8228d88 HARDWARE_SETUP + POST_5B_CHECKLIST e1-e4 notes (paused since 8 Oct)
a9aef35 Amendment 4 [e5] values; 120 s no-fire wait in the capture script + tests
a2bb847 prereg 13.2: study capture order (O1, fire first per camera)
93d0b0a tab:time two-line cells, tab:global_eval one float per Holm family, tab:time_maxn
c077cfb A5 analysis (scripts/a5_dirichlet_draws.py, analysis/a5/), tables, methods text
e67cd93 low-data sequence coverage (analysis/lowdata_sequence_coverage.csv) -- see 4.2
a8f95e9 tab:sensitivity fix; test_main_and_export_end_to_end repaired (broken since 2a27846)
04e1c7d resume_after_reboot dry run on node_a done (checklist c1)
97defeb REVISION_CHANGES entries for all of the above
6c79731 response skeleton R3.4 (A5)
475cc11 scripts/camera_calibration_table.py (Level-1 evidence tool, run after the study)
91be8cb last three overfull boxes removed (grouping_tradeoff table*, tab:leakage, a path)
4cae5b6 POST_5B_CHECKLIST e5 DONE note
8dad714 study plan draft: time estimate for fire-first
```

## 4. Decisions for the author (collected overnight)

1. **Plate tilt.** The phone level gave X 1.8 deg, Y 22 deg. From the D435if images
   (candle base row, camera height 56.5 cm, fy/cy from the record) the optical axis is
   tilted about 16.4 deg (3 m) and 16.7 deg (1 m) -- about 5.5 deg less, more than the
   assumptions explain (e5 log, 13 Oct). How and where was the phone placed? Then record
   the tilt (section 13 / scene sheet). The RealSense IMUs give no data (RSUSB backend);
   the ZED 2i IMU would work but needs the ZED on the plate.
2. **Low-data wording is not supported by the data.** The paper says the frame-level
   subsample "removes redundant near-duplicate frames before it removes sequences" and
   (methods) "reduces the number of frames per client rather than the number of scenes".
   The manifests show that at rho = 0.01 a node keeps frames from only 14-26 % of its
   training sequences (rho = 0.05: 25-44 %), because uniform frame sampling drops small
   sequences (`analysis/lowdata_sequence_coverage.csv`). Both sentences carry a `\todo`;
   they need rewriting. Also `docs/RESPONSE_TO_REVIEWERS.md` (R3, low-data item) says
   "subsampled at group level", which matches neither the paper nor the manifests.
3. **Timing todos** (`sec:res_time`, discussion): "time of FedAvg and FedProx 0.01 under
   label skew and their ratio with 95 % CIs", "FedProx first round vs FedAvg per round",
   "overhead factor". No ratio method is declared. Proposal: the seed-paired ratio per
   seed (same partition and seed), geometric mean over seeds with a seed bootstrap,
   per-seed ratios printed next to it, interval called indicative -- the method already
   declared for the MAXN/heterogeneous timing comparison (`docs/CROSS_CONFIG_COMPARISON.md`
   (c)). Needs the author's yes (and the wording "declared on <date>, after the times were
   seen").
4. **A5 text** (written overnight from the prereg; please read): the methods paragraph
   after the Dirichlet description in `sec:noniid`, the paragraph and `tab:a5_testbed` in
   `sec:res_dirichlet` (with a `\todo{Narrative.}`), the two supplementary tables.
5. **Study capture** (lift the DRAFT of section 13 once these are recorded there):
   (A) 15 scenes x 3 distances or (B) 18 x 2; O2 one room / one lamp; O3 light-emitting
   distractors; O4 when candles are replaced; camera order rotation across scenes;
   the background/floor pairs; materials (backdrops, floor covers, unbranded distractors,
   spare candles, snuffer). Plan: `docs/STUDY_CAPTURE_PLAN_DRAFT.md` (time estimate
   updated for fire-first: about 42 / 37 min per scene).
6. **Narrative session** for the ~25 `\todo{Narrative...}` items, the abstract and the
   conclusion; the aggregation-flip wording (`analysis/aggregation_flips.md`); the 7 W
   mechanism wording.
7. **Author-only:** funding / AI-assistance wording, testbed photo (Fig. 1).
8. node_a keeps an untracked `results/_gate_failed/20260928_pc_maxn_rev_iid_fedavg_r10_seed42/`
   that makes `test_main_on_real_repository_results` fail on node_a only (905 other tests
   pass there). Leave, or move it outside `results/`?

## 5. Working notes

- SSH from the desktop: Windows OpenSSH (`C:\Windows\System32\OpenSSH\ssh.exe`,
  `jetson-a@192.168.1.10`); Git Bash's ssh has no key access. From node_a use IPs
  (`jetson-b@192.168.1.7`, `yunus@192.168.1.6`); the names node_b/node_c do not resolve.
  Pipe scripts with `cmd //c "ssh ... bash -s < file"` (PowerShell pipes add a BOM).
- The Bash tool turns `\\` in an inline command into `\`: write Python with backslashes
  to a file with the Write tool (or use raw strings with single backslashes).
- Regenerating `paper/tables` on Windows rewrites unchanged files with CRLF; restore those
  with `git diff --quiet --ignore-cr-at-eol -- f && git checkout -- f` before committing.
- Standard tests: `python -m pytest tests -q -k "not end_to_end"` (911 pass); run
  `-k end_to_end` too after touching analysis or export code (8 pass).
- e5 log (desktop, gitignored): `logs/e5/e5_capture.log`; text copy of the 12/13 Oct part:
  `docs/e5/records/e5_capture_20261012_log.txt`.

## 6. Rules (reminder)

`CLAUDE.md` hard rules; rule before data (every measurement decision written into
section 13 and pushed before the data it applies to); no number typed by hand into the
paper; pilot/e5 data only under `data/raw/camera_pilot`, never accuracy or predictions;
before each physical step tell the author and wait for "ready"; sudo only
`sudo -n nvpmodel -m 2|3` on node_c, never while FL runs.
