# Post-5b checklist

The ordered steps for the day block 5b (`maxn_long_horizon`, 30 runs, `results/pc_maxn/`)
ends. Do them in order. A step starts only when the previous step's verification has
passed. Every step says who runs it:

* **[YOU]** needs you: any sudo beyond the assistant's scope (CLAUDE.md: only
  `sudo -n nvpmodel -m 2|3` on node_c, and never while an FL process runs), physical work
  with the cameras or the flame, and the decisions marked **DECISION**.
* **[CLAUDE]** the assistant runs it from the desktop (over SSH through the Windows
  OpenSSH client; Git Bash's ssh cannot see the key).
* **[CLAUDE, after your go]** the assistant runs it, but only after you say so explicitly
  in the session. This covers everything that changes the nodes.

Until step (d), the nodes stay on the commit 5b runs on. No `git pull` on any node, and
no commit of a partial block.

---

## Before 5b ends (can be done now)

**P1 -- DONE (2026-09-29, pushed).** The statistic of the timing comparison is declared
in `docs/CROSS_CONFIG_COMPARISON.md` (c):
- seed-paired ratio MAXN / heterogeneous (geometric mean over seeds, seed bootstrap),
  descriptive only, no verdicts;
- primary: rounds 2-3 in both configurations; secondary: MAXN over rounds 2-10; round 1
  as its own ratio;
- label skew FedAvg and FedBN, seeds 42/123/456: ten rounds against `long_horizon_fedbn`,
  rounds 2-10 in both;
- the straggler of each configuration is reported.

**P2 -- DONE (2026-09-29, pushed).** Pilot footage is declared outside the study in
`docs/CAMERA_EXPERIMENT_PREREG.md` section 10:
- it is stored only under `data/raw/camera_pilot/`, and the study scripts refuse it;
- the pilot only checks that the pipeline runs end to end without error. No accuracy,
  loss or prediction quality on pilot data is computed or looked at;
- afterwards, only technical changes are allowed, each as a dated amendment committed
  before the first study capture.

---

## (a) Close 5b: completeness, checksums, commit

**DONE (2026-10-05, `7cb0f9a`, pushed).**

**a1 [CLAUDE]** Confirm the block ended cleanly on node_a:
- `logs/run_matrix.log`: the last invocation ends with `block finished: N ok, 0 failed,
  0 not attempted`;
- `logs/resume.log` has no restart after that line;
- no `run_matrix.py` or FL process is alive on any node.

The block was resumed once (outage of 2026-09-28), so the last invocation counts only
the runs it had left (27). The count of 30 comes from a2.
*Verify:* the quoted lines themselves, and `pgrep -af "run_matrix|fl/server|fl/client"`
empty on all three nodes.

**a2 [CLAUDE]** All 30 runs present and valid:
- `python scripts/block_report.py --repo . --status_only` prints
  `maxn_long_horizon       30/30  COMPLETE`;
- each of the 30 `results/pc_maxn/rev_*_r10_seed*/results.json` has
  `model_selection.selected_round` and a `power` block with `power_config: maxn` and
  MAXN_SUPER measured before and after on all three nodes;
- no `IDENTITY_GATE_FAILED.json` exists under `results/pc_maxn/`.

The interrupted seed-789 attempt lives in `results/_interrupted/20260928_224412/` and is
neither counted nor committed.
*Verify:* the block report line, plus a script listing 30 dirs x
(selected_round, power_config, 3 x before/after mode), all filled.

**a3 [CLAUDE]** Checksums against node_a:
- `md5sum` of every file of the 30 run dirs, results.json and every
  `predictions/*.npz`, on node_a and on the desktop copy;
- strip the Windows `*` before diffing.

*Verify:* the two sorted lists are identical, and their file counts are equal and
non-zero.

**a4 [CLAUDE]** Commit only the 30 `results.json`:
- `predictions/*.npz` are gitignored (`git check-ignore` on one of them must print it);
- one commit, pushed. Nothing else goes into this commit.

*Verify:* `git show --stat HEAD` lists exactly 30 files, all
`results/pc_maxn/rev_*/results.json`.

---

## (b) Analyses -- each in its own commit, in this order

**DONE (2026-10-05: b1 `bc2dda3`, b2 `d80665f`, b3 `aa2fe8a`, b4 `c144397`, pushed).**

Each step reruns the tests (`python -m pytest tests -q -k "not end_to_end"`) and
`python scripts/clean_subset.py --check` (must print "identical") before committing.

**b1 [CLAUDE]** Regenerate `analysis/` from the full `results/`:
- `python scripts/analyze_results.py --results_dir results --output_dir analysis`;
- the heterogeneous tables: `python scripts/export_latex_tables.py --analysis_dir analysis
  --output_dir paper/tables`;
- the MAXN tables into their own directory, `--power_config maxn --output_dir
  paper/tables/maxn`. They must not overwrite the heterogeneous `time.tex`.

*Verify:*
- `summary_table.csv` has `power_config == maxn` rows for all 6 strategy x partition
  cells with `n_seeds == 5`;
- every heterogeneous row is unchanged except the two new timing metrics
  (`round1_time_s`, `steady_round_time_s`): compare with the previous commit, excluding
  those metrics;
- `analysis/aggregation_flips.md` regenerated (`scripts/aggregation_sensitivity.py`).

**b2 [CLAUDE]** The declared cross-configuration comparison, both Holm families (rounds
1-3; ten rounds: label skew x FedAvg/FedBN, seeds 42/123/456 vs `long_horizon_fedbn`):
`python scripts/cross_config_comparison.py --results_dir results --output_dir
analysis/cross_config`, with the code unchanged since `docs/CROSS_CONFIG_COMPARISON.md`.
Report every verdict whatever it shows. It is never a gate, and never "equivalent".
*Verify:*
- `git log` shows no change to `scripts/cross_config_comparison.py` after the
  declaration commit;
- both families are present with their declared m;
- a second run gives byte-identical output.

**b3 [CLAUDE]** Global evaluation, `maxn` family (`docs/GLOBAL_EVALUATION.md`, m = 4 on the
full set plus 4 on the clean subset):
`python scripts/global_evaluation.py analyze --results_dir results --output_dir
analysis/global_eval`, on the full `results/` now. The heterogeneous family was computed
on a view without `pc_maxn`.
*Verify:*
- the heterogeneous family's rows are byte-identical to `dce6f54`;
- the maxn family has 4 + 4 comparisons with the fixed verdict phrases.

**b4 [CLAUDE]** Timing comparison, exactly as declared in `docs/CROSS_CONFIG_COMPARISON.md`
(c), with its code written now and committed in the same commit as its output:
- both families;
- primary, secondary and round-1 ratios, each on its own;
- straggler counts per configuration.

Output goes to `analysis/cross_config_timing/`.
*Verify:*
- every per-run value equals an independent recomputation from `results.json`
  (`rounds[].timing.round_time_s`: round 1, and the median of rounds 2..R);
- no value is total / R.

---

## (c) Post-5b fixes (desktop code; the nodes get them in step d)

**DONE (2026-10-05: c1 `2c31a9c`, c2 `3cca463`, c3 `de924db`, pushed). Still open from
c1: the dry run on node_a after step d.**

**c1 [CLAUDE]** `scripts/resume_after_reboot.py` waits for synchronised clocks. Before the
pre-flight, it polls all three nodes until each reports
`timedatectl show -p NTPSynchronized --value` = `yes`, up to a timeout, and otherwise
logs and does not restart. It measures each node's offset against node_a (the best of 5
SSH round trips, as `camera_capture_session.clock_offset` does) and writes
`RESUME [...] clocks: node_a sync=yes, node_b +x ms, node_c +y ms` to `logs/resume.log`.
*Verify:*
- tests with a fake node that reports `no` for the first k polls: no restart before
  `yes`, and timeout -> no restart plus a log line;
- the offsets appear in the log line;
- a dry run on node_a (`--dry_run`) after step d.

**c2 [CLAUDE]** `scripts/run_matrix.py`: the per-run duration in the `%.1f min` log line is
measured with `time.monotonic()` instead of `time.time()`. The timestamps of the log lines
stay wall-clock.
*Verify:* a test that patches `time.time` to jump by 22 min mid-run and checks that the
logged duration does not.

**c3 [CLAUDE]** `scripts/fetch_results.ps1` also fetches `results/camera/<config>/` runs
(`rev_camera_*`, `diag_camera_*`), in both the remote `find` regex and
`$RunPathRegex`. Everything else stays unchanged, especially the alert scan: the
2026-09-28 bug (a message over 255 characters made msg.exe fail, and under
`$ErrorActionPreference = 'Stop'` that aborted the scan) silently disabled every alert.
*Verify:*
- `-DryRun` against node_a lists the FLAME runs exactly as before (diff the run list with
  and without the change) plus the camera paths of a fixture directory;
- a realistic long alert: feed `Invoke-AlertScan` (or `Show-Alert` directly) a
  `logs/resume.log` / `logs/run_matrix.log` line of about 600 characters with a path, a
  colon, quotes and Turkish characters (the nodes' locale prints `Sal`, `Eyl`);
- that alert must raise a pop-up (msg.exe, or the message-box fallback), leave
  `logs/fetch.log` with the full text and no error, and save the alert state (the same
  line is not re-alerted on the next pass);
- then one real hourly pass with no new lines raises nothing.

Commit c1-c3 (one commit each, tests passing) and push.

---

## (d) Pull on all three nodes, then MAXN pre-flight

**DONE (2026-10-05; all three nodes at `f97c60b`).**
- d1: on node_a, the 30 run files equalled their committed blobs; they were moved to
  `~/fedrgbd_reconcile_20261005/` and match the checkout by md5. On node_c, an untracked
  April `src/data/zed_capture.py` blocked the pull. It is the script that captured the v1
  ZED scenes, and it is kept in `~/fedrgbd_reconcile_20261005/` (md5 `ff23c646...`).
- d2: the crontab entry is present. The node test counts differ from the desktop for two
  environment-only reasons:
  - the depth-PNG dtype test fails under Pillow 9.0.1 (it reads back int32 with
    identical values);
  - node_a's real-results analysis test refuses, by design, because of
    `results/_gate_failed/`.
- d3: PRE-FLIGHT OK. All three nodes are MAXN_SUPER, with WiFi disabled, GUI off and NTP
  synchronised.
- The c1 dry run logged `clocks: node_a sync=yes, node_b +1134 ms, node_c +121 ms`. The
  NTP offsets are all under 4 ms; node_b's figure is an SSH-setup artifact (see below).

**Before the pilot (2026-10-05):**
- DONE (2026-10-05 21:54, by you) node_b sshd: `PermitEmptyPasswords yes` had made every
  key login wait ~2-4 s on a failed empty-password PAM attempt.
  - Now `PermitEmptyPasswords no`, PermitRootLogin and StrictModes at their defaults,
    `~/.ssh` 700 and `authorized_keys` 600.
  - Afterwards: key logins from node_a take 0.27-0.28 s (from ~4.2 s), with 0
    `pam_unix(sshd:auth)` failures in 14 logins. The resume script measures node_b
    +130 ms and node_c +116 ms (it was +1134 ms for node_b).
- DECISION: the FLAME train-transform difference between Pillow 9.0.1 (nodes) and 12.3
  (desktop baselines). It affects 7 of 300 images and 0.0024 % of pixels; the eval
  transform is bitwise identical. How it is disclosed is still to decide.
- Done: camera preprocessing once on the desktop (prereg Amendment 2, `e37e788`). The
  nodes get it with their next pull, before f1; the pilot (e) does not need it on the
  nodes.

**d1 [CLAUDE, after your go]** node_a first. Its 30 run dirs hold the untracked originals
of the files committed in a4, and those would block `git pull`. For each file:
- check it against the committed blob (`git hash-object` = the blob in `origin/revision-ncaa`);
- move it to `~/fedrgbd_reconcile_<date>/`;
- `git pull`;
- check that the checked-out file equals the backup (md5).

Then `git pull` on node_b and node_c.
*Verify:*
- `git rev-parse HEAD` identical on all three nodes and equal to the desktop's pushed
  HEAD;
- `git status --short` shows no modified tracked files;
- the reconcile backup exists and its md5s match.

**d2 [CLAUDE]** On each node:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q -k "not end_to_end" -p no:cacheprovider`;
- `crontab -l` still has the `@reboot` resume entry (node_a).

*Verify:* the test counts match the desktop, and the crontab line is present.

**d3 [CLAUDE]** `python3 scripts/run_matrix.py --power_config maxn --check_only` on node_a.
*Verify:*
- `pre-flight OK`;
- `nvpmodel -q` reports MAXN_SUPER on all three nodes;
- WiFi still disabled on all three;
- GUI off (`multi-user.target`);
- `timedatectl` reports synchronised clocks on all three.

If a mode is wrong: node_c you may ask the assistant to switch
(`sudo -n nvpmodel -m 2`, reboots node_c); node_a and node_b are **[YOU]**.

---

## (e) Camera pilot: one scene, all three cameras, end to end

Under `docs/CAMERA_EXPERIMENT_PREREG.md` section 10: pilot footage only under
`data/raw/camera_pilot/`. The pilot checks only that the pipeline runs without error; no
accuracy, loss or prediction quality is computed or looked at.

**e1 [YOU]**
- mount the rig: D435if on node_a, D435i on node_b, ZED 2i on node_c, side by side,
  pointing the same way;
- connect the cameras;
- install any missing camera SDK (needs sudo).

**e2 [CLAUDE]** `python3 scripts/camera_capture_session.py --check` on node_a.
*Verify:* PASS -- SSH, SDK and camera detected on every node, every clock offset under
1 s (the check now fails otherwise).

**e3 [YOU] + [CLAUDE]** You set the scene and the flame, and the assistant starts each
capture when you say "ready". Four captures:
- fire and no-fire at one distance (2 m), and then the same pair at a second distance;
- command:
  `python3 scripts/camera_capture_session.py --root data/raw/camera_pilot --scene s01 --label <fire|no_fire> --distance_m 2`
  (the pilot root, never `data/raw/camera`).

*Verify:*
- each capture prints PASS: 40 frames per camera, start within 1 s of schedule, the
  record belongs to this session;
- the clock offsets are logged in `session_log.jsonl`.

**e4 [CLAUDE]** Copy the pilot frames of all three nodes to the desktop under
`data/raw/camera_pilot/`, then:
- `python scripts/camera_labels.py --data_dir data/raw/camera_pilot --splits_dir data/raw/camera_pilot/splits`
  (the script refuses a splits directory outside the pilot tree);
- preprocess them once, into the pilot tree only (prereg Amendment 2):
  `python scripts/camera_preprocess_frames.py --raw_dir data/raw/camera_pilot --output_dir data/raw/camera_pilot/camera_224 --manifest data/raw/camera_pilot/preprocessed_manifest.csv`,
  then `--verify` with the same paths;
- load every valid frame through `CustomRGBDDataset(preprocess="camera", preprocessed=PreprocessedStore(<those paths>))`.

*Verify:*
- no exclusion fires, or each one is explained;
- metadata keys complete: serial, firmware, SDK version, exposure, gain, white balance
  where exposed, timestamp, ZED depth mode;
- depth aligned to RGB (same size);
- IR present on the RealSense nodes only;
- every preprocessed tensor 224 x 224 with finite values;
- a contact sheet of one frame per camera and class, for you to check technical image
  properties only: exposure, framing, depth validity. Any technical change that follows
  is a dated amendment (prereg section 10) committed before the first study capture.

**What one scene cannot test.** The leave-one-scene-out manifests need at least 3 scenes,
and the five federated folds are dealt from the complete set of kept scenes. So
`camera_manifests.py`, `camera_fl_prepare.py`, `camera_loso.py` and the FL runs cannot be
exercised on the pilot. They also refuse pilot paths by declaration. They are covered
by the synthetic end-to-end tests in `tests/test_camera_*`, and first meet real frames in
step f.

---

## (f) After the full capture: camera determinism smoke pair, then the camera block

Fold 0 is fixed only once all scenes are captured and the folds are generated, so the
smoke pair comes here, not after the pilot (prereg section 9: "after the capture and
before the block").

**f1 [CLAUDE]** Build the study data:
- `camera_labels.py`, then `camera_manifests.py`: commit `data/splits_camera/` and the
  manifests before any model is trained;
- on the desktop, preprocess every valid frame once (prereg Amendment 2):
  `camera_preprocess_frames.py`, then `--verify`; commit
  `data/splits_camera/preprocessed_manifest.csv` and `preprocessed_info.json` with the
  manifests, before any model is trained;
- copy each node ONLY its own camera's preprocessed files,
  `data/processed/camera_224/<node>/`, and `labels.csv`; on the node,
  `camera_preprocess_frames.py --verify --nodes <own node>` must report "identical";
- then, on each node, and on the desktop for the baselines,
  `camera_fl_prepare.py --clean --nodes <own node>` and `--verify --nodes <own node>`
  (`--verify` checks every fold image's md5 against the preprocessed manifest);
- commit `fl_materialised_manifest.csv`.

*Verify:*
- `--verify` passes on every node;
- `data/processed/camera_fold<f>/manifest.csv` has the same md5 on all three nodes,
  equal to `fold_manifest_md5`.

Once `fl_materialised_manifest.csv` is committed, `run_matrix.py --check_only`, which
checks every known split, also expects the five camera fold manifests on every node, so
prepare them on all three nodes before the next pre-flight.

**f2 [CLAUDE, after your go]**
`python3 scripts/run_matrix.py --block camera_determinism_smoke --power_config maxn` on
node_a.
*Verify:*
- both `results/camera/maxn/diag_camera_smoke{,_r2}` finished;
- `python3 scripts/run_matrix.py --block camera_sensor_skew --power_config maxn
  --dry_run` logs `determinism gate OK`. `--check_only` does not evaluate the gate.
  `--dry_run` stops right after the gate and the run list.

If it prints `PRE-FLIGHT FAIL -- determinism gate`, stop and report it as it is. The
block does not start.

**f3 [CLAUDE, after your go]**
`python3 scripts/run_matrix.py --block camera_sensor_skew --power_config maxn`.
