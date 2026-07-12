#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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
    parser = argparse.ArgumentParser(
        description="Run PhysTwin inference with an external topology NPZ."
    )
    parser.add_argument("--phystwin-root", required=True, type=Path)
    parser.add_argument("--scene", default="double_stretch_sloth")
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--train-frame", type=int)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--topology-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.phystwin_root.expanduser().resolve()
    scene_root = resolve_scene_root(root, args.scene, args.base_path)
    split = load_split(scene_root)
    train_frame = (
        int(args.train_frame)
        if args.train_frame is not None
        else int(split["train"][1])
    )
    model_path = args.model_path.expanduser().resolve()
    topology_path = args.topology_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in [model_path, topology_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    os.environ["EXTERNAL_TOPOLOGY_NPZ"] = str(topology_path)
    prepare_phystwin(root, args.scene, scene_root, seed=args.seed)
    from qqtt import InvPhyTrainerWarp
    from qqtt.utils import logger

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.set_log_file(path=str(output_dir), name="external_topology_inference")
    trainer = InvPhyTrainerWarp(
        data_path=str(scene_root / "final_data.pkl"),
        base_dir=str(output_dir),
        train_frame=train_frame,
    )
    inference_path = output_dir / "inference.pkl"
    try:
        trainer.test(model_path=str(model_path))
    except Exception:
        if inference_path.is_file():
            print(
                "[WARNING] PhysTwin raised an exception after saving inference.pkl; "
                "the saved trajectory is kept."
            )
        else:
            raise
    print("[DONE]", inference_path)


if __name__ == "__main__":
    main()
