#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Write the best node-retraining spring parameters back into the "
            "node-coarsened topology. Stage-2 BT must use this output, not the "
            "pre-retraining topology."
        )
    )
    p.add_argument("--topology", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = args.topology.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(source)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    z = np.load(source, allow_pickle=True)
    arrays = {k: z[k] for k in z.files}
    ckpt = torch.load(checkpoint, map_location="cpu")
    if "spring_Y" not in ckpt:
        raise KeyError(f"Checkpoint has no spring_Y: {checkpoint}")

    spring_y = ckpt["spring_Y"]
    if torch.is_tensor(spring_y):
        spring_y = spring_y.detach().cpu().numpy()
    spring_y = np.asarray(spring_y)

    if len(spring_y) != len(arrays["springs"]):
        raise RuntimeError(
            f"spring mismatch: checkpoint={len(spring_y)}, "
            f"topology={len(arrays['springs'])}"
        )
    if not np.isfinite(spring_y).all():
        raise FloatingPointError("Checkpoint spring_Y contains NaN/Inf")

    arrays["spring_Y"] = spring_y
    arrays["retrained_checkpoint"] = np.asarray(str(checkpoint))
    arrays["post_node_retrain"] = np.asarray(True)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)

    print("[DONE]", output)
    print("source topology      :", source)
    print("retrained checkpoint :", checkpoint)
    print("springs              :", len(spring_y))
    print("Y min/mean/max       :", float(spring_y.min()), float(spring_y.mean()), float(spring_y.max()))


if __name__ == "__main__":
    main()
