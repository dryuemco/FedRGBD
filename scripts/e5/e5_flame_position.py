#!/usr/bin/env python3
"""e5 helper (not a study tool): where is the flame core in the 224 crop of ONE test frame?

Reads a framing/test frame saved by e5_framing_grab_*.py as ``*_crop224_x3.png`` (the
section-3 224 crop, enlarged 3x with nearest neighbour), undoes the enlargement, and
lists the components of flame-core pixels by the absolute part of the flame tool's rule
(max channel >= FLAME_CORE_V and R >= G >= B; a single frame has no no-fire reference).
Prints each component's bbox in 224 and native (1920x1080, crop x 420-1499) coordinates
and whether it touches the crop edge; writes ``*_crop224_x3_flamecore.png``.

    python scripts/e5/e5_flame_position.py logs/e5/framing/<name>_crop224_x3.png
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.camera_flame_height import FLAME_CORE_V, _label  # noqa: E402


def main(src):
    c = np.asarray(Image.open(src).convert("RGB").resize((224, 224), Image.NEAREST)).astype(np.int16)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mask = (c.max(axis=2) >= FLAME_CORE_V) & (r >= g) & (g >= b)
    lab, n = _label(mask)
    print("flame-core pixels (max channel >= %d, R>=G>=B) in the 224 crop: %d, components: %d"
          % (FLAME_CORE_V, int(mask.sum()), int(n)))
    s = 1080 / 224.0
    comps = []
    for k in range(1, int(n) + 1):
        ys, xs = np.nonzero(lab == k)
        comps.append((ys.size, k, ys.min(), xs.min(), ys.max(), xs.max()))
    for size, k, t, l, bt, rt in sorted(comps, reverse=True)[:5]:
        edge = t == 0 or l == 0 or bt == 223 or rt == 223
        print("component %d: %d px, 224 bbox top=%d left=%d bottom=%d right=%d (h=%d w=%d), "
              "touches crop edge=%s, native bbox x=%d-%d y=%d-%d"
              % (k, size, t, l, bt, rt, bt - t + 1, rt - l + 1, edge, round(420 + l * s),
                 round(420 + (rt + 1) * s) - 1, round(t * s), round((bt + 1) * s) - 1))
    ov = Image.fromarray(c.astype(np.uint8)).resize((672, 672), Image.NEAREST)
    d = ImageDraw.Draw(ov)
    for size, k, t, l, bt, rt in sorted(comps, reverse=True)[:1]:
        d.rectangle([l * 3, t * 3, (rt + 1) * 3 - 1, (bt + 1) * 3 - 1], outline=(0, 255, 0))
    out = src.replace("_crop224_x3.png", "_crop224_x3_flamecore.png")
    ov.save(out)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
