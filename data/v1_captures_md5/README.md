# v1 camera captures: md5 manifests

`data/raw/captures/` on the three nodes holds the five-scene captures of the original
submission (v1), recorded on 2026-04-23 between 12:35 and 13:47 (file birth time;
ctime and mtime are the same). The files are not in git and are not touched by the
revision. These manifests fix their content as of 2026-10-07, so any later change can be
detected.

| file | node | files | content |
|---|---|---|---|
| `node_a.md5` | node_a | 2765 | `node_a/` (own scenes) plus copies of `node_b/` and `node_c/` |
| `node_b.md5` | node_b | 1005 | `node_b/` |
| `node_c.md5` | node_c | 755 | `node_c/` |

The copies on node_a are byte-identical to the originals on node_b (1005 of 1005 files)
and on node_c (755 of 755).

Paths are relative to `~/FedRGBD/data/raw/captures/`. They were produced read-only with:

```bash
cd ~/FedRGBD/data/raw/captures && find . -type f | LC_ALL=C sort | sed "s|^\./||" | xargs -d "\n" md5sum
```

To check later, run on the node: `cd ~/FedRGBD/data/raw/captures && md5sum -c --quiet <manifest>`.

## v1 derived files on node_a (`node_a_v1_derived.md5`)

node_a also holds the v1 cross-sensor pipeline that read these captures. None of it was
ever in git. All files are dated 2026-04-23, between 12:44 and 17:02 (mtime). The
manifest covers 1812 files; paths are relative to `~/FedRGBD/`:

| path | files |
|---|---|
| `scripts/prepare_captures.py`, `scripts/cross_sensor_eval.py`, `scripts/cross_sensor_depth_eval.py` | 3 |
| `data/processed/captures/` | 903 |
| `data/processed/captures_depth/` | 903 |
| `data/processed/captures.zip` | 1 |
| `results/cross_sensor/seed42/results.json`, `results/cross_sensor_depth/seed42/results.json` | 2 |

It was produced read-only, with the same `md5sum` command run from `~/FedRGBD` over
these paths, and verified with `md5sum -c` (all OK). The files are left as they are.

Context: the camera prereg's preamble (2026-09-28) said these captures were no longer on the nodes; an erratum is being prepared.
