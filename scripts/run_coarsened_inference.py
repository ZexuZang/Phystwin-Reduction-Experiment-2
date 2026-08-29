#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phystwin_reduction.phystwin_runtime import (
    load_split,
    prepare_phystwin,
    resolve_scene_root,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run inference for a node-coarsened/final-budget PhysTwin graph "
            "using InvPhyTrainerWarpCoarsening."
        )
    )
    p.add_argument("--phystwin-root", required=True, type=Path)
    p.add_argument("--scene", required=True)
    p.add_argument("--base-path", type=Path)
    p.add_argument("--coarsened-data", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--train-frame", type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.phystwin_root.expanduser().resolve()
    scene_root = resolve_scene_root(root, args.scene, args.base_path)
    split = load_split(scene_root)
    train_frame = int(args.train_frame) if args.train_frame is not None else int(split["train"][1])

    coarsened_data = args.coarsened_data.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for p in [coarsened_data, checkpoint]:
        if not p.is_file():
            raise FileNotFoundError(p)

    prepare_phystwin(root, args.scene, scene_root, seed=args.seed)
    from qqtt.engine.trainer_warp_coarsening import InvPhyTrainerWarpCoarsening

    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "data_path": str(scene_root / "final_data.pkl"),
        "base_dir": str(output_dir),
        "coarsened_data": str(coarsened_data),
        "device": args.device,
        "train_frame": train_frame,
        "pure_inference_mode": True,
    }

    sig = inspect.signature(InvPhyTrainerWarpCoarsening.__init__)
    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not has_kwargs:
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    print("===== COARSENED INFERENCE =====")
    print("scene           :", args.scene)
    print("checkpoint      :", checkpoint)
    print("coarsened_data  :", coarsened_data)
    print("train_frame     :", train_frame)
    print("output_dir      :", output_dir)

    trainer = InvPhyTrainerWarpCoarsening(**kwargs)
    trainer.test(model_path=str(checkpoint))

    dense = output_dir / "inference.pkl"
    physical = output_dir / "inference_physical.pkl"
    if not dense.is_file():
        raise FileNotFoundError(f"Expected dense inference was not produced: {dense}")
    print("[DONE] dense    :", dense)
    print("[DONE] physical :", physical if physical.exists() else "not emitted")


if __name__ == "__main__":
    main()
