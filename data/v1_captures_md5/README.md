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

Context: the camera prereg's preamble (2026-09-28) said these captures were no longer on the nodes; an erratum is being prepared.
