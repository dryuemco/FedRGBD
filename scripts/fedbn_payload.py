#!/usr/bin/env python3
"""FedRGBD -- share of the model payload that FedBN keeps local.

In this implementation every client sends and receives the full ``state_dict``
(``src/fl/client.py``: ``get_parameters``) under every strategy.  Under FedBN the
client discards the received BatchNorm entries on load (``set_parameters``), so
FedBN communicates exactly as much as FedAvg.  This script measures what an
implementation that did not transmit the BatchNorm entries would save: their share
of one model transfer, using the client's own ``get_bn_indices`` so the entries
counted are exactly the ones FedBN keeps local (weight, bias, running mean,
running variance and the batch counter of every BatchNorm layer).

    python scripts/fedbn_payload.py --output analysis/fedbn_payload.json
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def bn_payload_share():
    from src.fl.client import get_bn_indices
    from src.models.mobilenetv3_multimodal import create_model

    model = create_model(num_classes=2, in_channels=3, pretrained=False)
    bn = get_bn_indices(model)
    total_bytes = bn_bytes = total_elems = bn_elems = 0
    for i, tensor in enumerate(model.state_dict().values()):
        nbytes = int(tensor.numel()) * int(tensor.element_size())
        total_bytes += nbytes
        total_elems += int(tensor.numel())
        if i in bn:
            bn_bytes += nbytes
            bn_elems += int(tensor.numel())
    return {
        "model": "MobileNetV3-Small, 2 classes, 3 input channels",
        "transfer": "full state_dict, every strategy (src/fl/client.py get_parameters)",
        "total_tensors": len(model.state_dict()),
        "bn_tensors": len(bn),
        "total_bytes": total_bytes,
        "bn_bytes": bn_bytes,
        "total_elements": total_elems,
        "bn_elements": bn_elems,
        "bn_share_of_bytes": bn_bytes / total_bytes,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", default=os.path.join("analysis", "fedbn_payload.json"))
    args = ap.parse_args(argv)
    out = bn_payload_share()
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print("BN entries: %d of %d tensors, %d of %d bytes (%.2f %% of one transfer)"
          % (out["bn_tensors"], out["total_tensors"], out["bn_bytes"], out["total_bytes"],
             100 * out["bn_share_of_bytes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
