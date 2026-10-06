# v1 cross-sensor pipeline (legacy, not used by the revision)

These are the scripts and result files that produced the original submission's (v1)
cross-sensor results. They date from 2026-04-23 and were never in git: they lived only
on node_a (`~/FedRGBD/`). On 2026-10-07 they were added here **unchanged**, as byte
copies, line endings included. `.gitattributes` marks this directory `-text`, so git
never converts them.

| file here | original on node_a | md5 |
|---|---|---|
| `scripts/prepare_captures.py` | `scripts/prepare_captures.py` | `319aad09991a427072095ee2f58ab517` |
| `scripts/cross_sensor_eval.py` | `scripts/cross_sensor_eval.py` | `35c10829e7a8b75e1014c2005f7b6049` |
| `scripts/cross_sensor_depth_eval.py` | `scripts/cross_sensor_depth_eval.py` | `d17746c5ce7a5f122a1352c1bacb92cf` |
| `results/cross_sensor/seed42/results.json` | same path | `cfec3e7e1aeb281f73cc316228025f33` |
| `results/cross_sensor_depth/seed42/results.json` | same path | `37ca1effb940c545275dde96bd40a0ba` |

Every md5 equals its entry in `data/v1_captures_md5/node_a_v1_derived.md5`.

**What they did (v1).**
- `prepare_captures.py` read the v1 captures in `data/raw/captures/` and wrote
  `data/processed/captures/`.
- `cross_sensor_eval.py` (RGB) and `cross_sensor_depth_eval.py` (depth) trained on
  one camera's scenes and tested on the others'. They wrote the two `results.json`
  files.

The processed image data (`data/processed/captures*`, about 4 GB) is not included. It
stays on node_a, fixed by the same manifest.

**Not used by the revision.** No revision code imports or runs these scripts. They
were not run when they were added here. The v1 camera results are removed from the
paper and replaced by the pre-registered camera experiment
(`docs/CAMERA_EXPERIMENT_PREREG.md`), whose leave-one-scene-out design addresses the
scene-overlap risk of the v1 evaluation. See the erratum in that file (section 14).
The scripts are kept so that the v1 result can be traced.
