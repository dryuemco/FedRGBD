# e5 helper scripts (2026-10-08/09)

Ad hoc helpers used during the e5 acceptance test, kept so a new session can repeat the
steps. None of them is a study tool; the study tools are `src/data/realsense_capture.py`,
`src/data/zed_capture.py` (`--calibrate_exposure`, `--session_check`, captures),
`scripts/camera_flame_height.py` and `scripts/camera_depth_holes.py`.

| script | what it does | how it was run |
|---|---|---|
| `e5_framing_grab_d435i.py` | one framing-only frame of the D435i (S/N 405622076256) to `~/e5_framing/` on node_b: full frame with the 224 crop rectangle + the 224 crop x3; prints JSON with md5s | `ssh node_b "cd ~/FedRGBD && . ~/fedrgbd_venv/bin/activate && python3 -" < e5_framing_grab_d435i.py` (via node_a) |
| `e5_framing_grab_d435if.py` | the same for the D435if (S/N 239722070442) on node_a | `ssh node_a "... python3 -" < ...` |
| `e5_framing_grab_zed.py` | the same for the ZED 2i on node_c, through `ZedBackend` (HD1080, NEURAL, 15 fps), camera settings as they are | via node_a to node_c |
| `e5_test_grab_zed_fixed.py` | ZED test frame with the committed fixed setting applied (`fixed_exposure` + `apply_lock`) | the same |
| `e5_flame_position.py` | flame-core position in the 224 crop of one test frame (absolute rule only) | on the desktop, on the copied `*_crop224_x3.png` |
| `e5_imu_tilt_d435i.py` | D435i accelerometer tilt; **gave no data under the RSUSB backend** (HARDWARE_SETUP.md) | not used for any value |
| `archive_e5.sh` | moves (never deletes) s03/s09/s10 of node_b under `data/raw/camera_pilot/_archive_<stamp>/` with an md5 manifest | `bash -s <pilot_root> <stamp>` on node_b and the desktop |
| `LEGACY_session_check_realsense.py` | the first, RealSense-only session check; **superseded** by `--session_check` (eba04ae) | not to be used |

Copying a node's pilot frames to the desktop with an md5 check (as in e5):
`ssh node_a "ssh <node> 'cd <repo>/data/raw/camera_pilot && tar cf - --exclude=<node>/s01_* --exclude=<node>/_captures/s01_* <node>'" | tar xf - -C data/raw/camera_pilot`,
then `find <node> -type f ! -name 's01_*' | LC_ALL=C sort | xargs md5sum` on both sides and `diff`.

`docs/e5/diffs/` holds the reviewed drafts of the flame-tool fix and the exposure policy
(the committed versions are in git history: 76a7003, 58bff21).
