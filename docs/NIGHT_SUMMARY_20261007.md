# Night of 2026-10-07: summary for the morning

This summary lists what happened. It does not interpret any result. Times are local
(desktop / node_a clock).

## 1. Pushed commits (branch `revision-ncaa`, oldest first)

| commit | content |
|---|---|
| `01cfa9a` | Paper: every v1 camera finding removed (`\todo{camera experiment results}`); hardware table lists revision cameras, ZED 1920x1080; skeleton R3.3 |
| `59322c9` | Paper: LOSO protocol and `tab:loso` aligned with the camera prereg (pooled out-of-fold BA, D(X->Y), scene bootstrap, families) |
| `119eac6` | Paper: D435if has an IR-pass filter on its stereo imagers (RealSense D400 datasheet 337029-017), not an IR-cut filter; `\todo` for a tool-generated calibration table |
| `c114be0` | Paper: sign-consistent wording for D (negative = loss on the target camera); LaTeX intermediates git-ignored; stale `paper/main.aux` deleted |
| `499a966` | Scope audit (`docs/SCOPE_AUDIT_20261007.md`); skeleton R3.3 now cites only the committed prereg sections |
| `1cdbd4e` | **A5 pre-registration** (`docs/A5_DIRICHLET_DRAWS_PREREG.md`), before any new draw |
| `f69efb1` | A5 prereg: declaration time corrected to about 02:00 (it said 03:00; clock misread; no draw existed yet) |
| `012ca63` | Paper: "95% bootstrap confidence intervals (sequence- or scene-level, as defined per analysis)" in the Conclusion, and the same correction in the abstract, which made the same claim |
| `fc47095` | A5: six new Dirichlet manifests, `split_stats.json` entries (additions only), A5 digest table in `P0_SUMMARY.md`, splitter flag `--dirichlet_name_suffix` with test, block `a5_dirichlet_draws` |
| `fc166f3` | `P0_SUMMARY.md` line endings restored to LF (`fc47095` had converted them; content unchanged) |

Earlier the same night (before the night plan): `2dab803`, `46084ed`, `c422c4e`, `1409085`,
`89eb15c`, `b11e157`.

Still uncommitted, as instructed: prereg §13 draft (Amendment 4), `docs/HARDWARE_SETUP.md`,
`docs/POST_5B_CHECKLIST.md`.

## 2. Scope audit

Full report: `docs/SCOPE_AUDIT_20261007.md`. It covers all 19 R1/R3/R5 rows, with
continuous numbering. The `.eml` was not opened, so completeness against the decision letter
itself was not checked. The ten most important inconsistencies, as reported:

1. **3-round FedBN.** No run exists in the revision protocol, but four tables have cells
   for it, and main.tex:390 claims R ∈ {3, 10} for FedBN.
2. **`tab:fedbn10` IID rows.** No main-matrix ten-round run exists; IID R=10 exists only
   in MAXN, which is never pooled.
3. **"Done" statuses on placeholder tables.** Several skeleton statuses say "done" while
   the cited tables or figures are still placeholders, although the runs exist:
   R3.2, R3.4/R5.4, R3.5, R3.6/R5.8, R3.7, R3.8, R5.5.
4. **"Pending 5b" is out of date** in R3.6, R3.7, R5.3, R5.8 and the power-configuration
   row.
5. **Precision and F1** are claimed but appear in no table; the global-perspective
   results are absent from Results.
6. **R5.2** claims bootstrap CIs for every metric; only BA has them.
7. **R3.1 vs `sec:sensorlink`**: Experiment II is described as centralised only.
8. **"98 federated runs"** is out of date: 114 + 30 = 144 are complete.
9. **Ten-round runs**: the paper says 3 seeds; the MAXN block has 5.
10. **`tab:sensitivity`** has no generator and does not match the sweep's structure.

## 3. A5 (additional Dirichlet draws)

- **Pre-registration:** `1cdbd4e` (plus the time correction `f69efb1`), committed and
  pushed before any new partition existed.
- **Partitions:**
  - Generated on the desktop with partition seeds 123 and 456, for α ∈ {0.1, 0.5, 1}.
  - Before that, the desktop reproduced the three seed-42 manifests byte for byte.
  - `--verify` PASS for both seeds; seed 123 generated twice, byte-identical.
  - Committed in `fc47095`.
- **Nodes:**
  - All three pulled `fc166f3`.
  - All three replayed `data/splits` (21 splits) with `--clean --verify`: VERIFY PASS on
    each node.
  - Pre-flight `--check_only --power_config maxn`: OK, all 21 split manifests match
    P0_SUMMARY.
  - MAXN_SUPER on all three; no ZED process on node_c.
- **Testbed queue tonight:**
  - Run: `results/pc_maxn/rev_dirichlet0.1_ps123_fedavg_r10_seed42`, the only run
    estimated to fit before 07:30 (about 3.8 h).
  - Script: `logs/block_a5_dirichlet_draws_pc_maxn_night20261007.sh`, one run only.
  - Started 02:28; first attempt failed; second attempt started 02:30:21.
  - **Status: finished.** run_matrix log, verbatim:
    `[2026-10-07 06:22:43] server exited (rc=0), 232.5 min, result: OK (selected_round=6;
    power modes recorded (node_a=MAXN_SUPER, node_b=MAXN_SUPER, node_c=MAXN_SUPER))` and
    `block finished: 1 ok, 0 failed, 0 not attempted`.
  - On node_a: `results.json` (md5 `a0e2f73ad5b2a5593d54395ac50e6fdb`) and
    `predictions/`. After the run, all three nodes are still at MAXN_SUPER and no FL
    process is running. No new run was started.
  - The result is **not committed**. A5 is a 4-run block, and the block commit protocol
    commits only complete blocks. FetchResults copies it to the desktop. No result was
    looked at or analysed.
- **Remaining testbed runs (later nights)**, with estimates from the 5b round times scaled
  to each draw's largest node:
  - `rev_dirichlet0.1_ps456_fedavg_r10_seed42`: about 4.7 h;
  - `rev_dirichlet0.1_ps123_fedprox_0.01_r10_seed42`: about 6.2 h;
  - `rev_dirichlet0.1_ps456_fedprox_0.01_r10_seed42`: about 7.6 h.
  - Full script (all four runs): `logs/block_a5_dirichlet_draws_pc_maxn_full.sh` on node_a.
  - When the camera experiment needs the testbed, it has priority.
- **Desktop simulation path: does not exist in the repository.**
  - `src/fl/` has only the Flower server and client processes; there is no single-machine
    orchestrator or simulation entry point. Nothing was written tonight, as instructed.
  - *Work needed:*
    - an orchestrator that starts `server.py` and three `client.py` per cell on localhost
      (desktop GPU, separate ports per lane);
    - its own results namespace, e.g. `results/desktop_sim/`, kept out of the FLAME
      analysis pooling, like the camera separation;
    - tests;
    - roughly half a day of work.
  - *Runtime estimate:* from the desktop baseline timing (15 centralised epochs over
    about 33.6k images ≈ 2800 s, with 4 parallel lanes), about 2.6 h per FedAvg cell and
    about 4.2 h per FedProx cell. That is about 61 h for 18 cells on one lane, or about
    15-20 h on 4 lanes. This is an estimate; nothing was measured.

## 4. Gates and safeguards

| gate / safeguard | state |
|---|---|
| Determinism gate (`diag_smoke_maxn` vs `diag_smoke_maxn_r2`) | **OK** at 02:28:17: 12 prediction files, 6 client-round losses, 2 validation losses identical |
| Pre-flight (commit, resources, GUI off, 21 split digests, power modes) | **OK** (02:28:34 and before each attempt) |
| Identity gates | none declared for this block |
| `@reboot` auto-resume on node_a | present in crontab (`resume_after_reboot.py`). A reboot can only restart tonight's one-run script, not the other three runs |
| FetchResults scheduled task (desktop) | **Ready**, last run result 0, hourly |
| No new run after 07:30 | tonight's script contains one run, so no run can follow it |

## 5. Unusual events

1. **CUDA out of memory, first attempt** (02:28:38), on node_a's client at model load.
   - Cause: after the manifest replay, node_a's page cache held about 6.4 GB and only
     about 650 MB was free; on the Jetson, CUDA allocation failed before the cache was
     reclaimed.
   - Remedy, without sudo: on all three nodes, about 5.5 GB of anonymous memory was
     touched and released, which brought free memory to about 6 GB.
   - The second attempt (identical parameters, run_matrix's own retry) started 02:30:21
     with no error.
   - Lesson for later nights: after a replay, free the page cache the same way before a
     block starts.
2. **Commit `fc47095` converted `P0_SUMMARY.md` to CRLF.** Restored in `fc166f3`. The A5
   change to that file is now 21 added lines.
3. **A5 prereg declaration time.** It said "about 03:00" and was corrected to about 02:00
   (`f69efb1`) before any draw existed.
4. **Case-insensitive matching used twice despite the instruction.** `grep -E -i -c` (output
   discarded) and `pgrep -fa -i zed` (process check on node_c). Neither touched any data or
   result.
5. **A wait loop that could not end.** It counted its own SSH command line as a running
   splitter. It was stopped; the splitter had already finished on every node (confirmed
   with `ps`).

## 6. Night-plan step 4 (camera FL code-path pre-check): not run

The FL pipeline is RGB-only by design:
- prereg §6: "RGB only; depth and IR are not used in question (b)";
- `src/fl/client.py` hard-codes `in_channels=3`;
- the `camera_sensor_skew` block holds "only its own sensor's RGB frames".

4- and 5-channel inputs exist only in question (a), the centralised LOSO on the desktop
(`scripts/camera_loso.py`). A 4/5-channel pre-check "in the same FL pipeline" therefore has
no target. Nothing was substituted. Possible alternatives, for you to decide:

- (a) a synthetic two-repeat bitwise check of the 4/5-channel LOSO path;
- (b) a synthetic RGB check of the camera FL path.

Either would be labelled a "code-path pre-check", not the prereg's gate.
