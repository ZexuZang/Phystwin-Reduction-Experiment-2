#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
import types

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phystwin_reduction.phystwin_runtime import (
    load_split,
    prepare_phystwin,
    resolve_scene_root,
    stage_boundaries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first stage of PhysTwin.")
    parser.add_argument("--phystwin-root", required=True, type=Path)
    parser.add_argument("--scene", default="double_stretch_sloth")
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--stage1-ratio", type=float, default=0.5)
    parser.add_argument("--train-frame", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--enable-open3d-video",
        action="store_true",
        help="Do not replace trainer.visualize_sim with a no-op.",
    )
    return parser.parse_args()


def _dummy_visualize(self, *args, **kwargs):
    video_path = kwargs.get("video_path")
    if video_path:
        path = Path(video_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    return None


def main() -> None:
    args = parse_args()
    root = args.phystwin_root.expanduser().resolve()
    scene_root = resolve_scene_root(root, args.scene, args.base_path)
    split = load_split(scene_root)
    _, stage1_end, _, _, _ = stage_boundaries(split, args.stage1_ratio)
    train_frame = args.train_frame if args.train_frame is not None else stage1_end
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root
        / "results"
        / "online_adaptation"
        / args.scene
        / f"stage1_train_{int(round(args.stage1_ratio * 100))}"
    )

    cfg = prepare_phystwin(
        root, args.scene, scene_root, seed=args.seed
    )
    from qqtt import InvPhyTrainerWarp
    from qqtt.utils import logger

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.set_log_file(path=str(output_dir), name="stage1_train_log")
    trainer = InvPhyTrainerWarp(
        data_path=str(scene_root / "final_data.pkl"),
        base_dir=str(output_dir),
        train_frame=int(train_frame),
    )
    if not args.enable_open3d_video:
        trainer.visualize_sim = types.MethodType(_dummy_visualize, trainer)
    print("[SCENE]", args.scene)
    print("[TRAIN_FRAME]", train_frame)
    print("[OUTPUT]", output_dir)
    trainer.train()


if __name__ == "__main__":
    main()
