#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phystwin_reduction.phystwin_runtime import (
    load_split,
    prepare_phystwin,
    resolve_scene_root,
    stage_boundaries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PhysTwin checkpoint topology.")
    parser.add_argument("--phystwin-root", required=True, type=Path)
    parser.add_argument("--scene", default="double_stretch_sloth")
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--stage1-ratio", type=float, default=0.5)
    parser.add_argument("--train-frame", type=int)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> None:
    args = parse_args()
    root = args.phystwin_root.expanduser().resolve()
    scene_root = resolve_scene_root(root, args.scene, args.base_path)
    split = load_split(scene_root)
    _, stage1_end, _, _, _ = stage_boundaries(split, args.stage1_ratio)
    train_frame = args.train_frame if args.train_frame is not None else stage1_end
    model_path = args.model_path.expanduser().resolve()
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path
        else root
        / "results"
        / "online_adaptation"
        / args.scene
        / "topologies"
        / "stage1_official_topology.npz"
    )
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    cfg = prepare_phystwin(root, args.scene, scene_root, seed=args.seed)
    from qqtt import InvPhyTrainerWarp
    from qqtt.utils import logger

    tmp_dir = output_path.parent / "export_topology_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logger.set_log_file(path=str(tmp_dir), name="export_topology_log")
    trainer = InvPhyTrainerWarp(
        data_path=str(scene_root / "final_data.pkl"),
        base_dir=str(tmp_dir),
        train_frame=int(train_frame),
    )
    checkpoint = torch.load(model_path, map_location=cfg.device)

    points_full = to_numpy(trainer.init_vertices)
    springs = to_numpy(trainer.init_springs)
    rest_lengths = to_numpy(trainer.init_rest_lengths)
    masses = to_numpy(trainer.init_masses)
    spring_y = to_numpy(checkpoint["spring_Y"])
    num_object_springs = int(
        checkpoint.get("num_object_springs", trainer.num_object_springs)
    )
    if len(springs) != len(spring_y):
        raise RuntimeError(
            f"Topology mismatch: springs={len(springs)}, spring_Y={len(spring_y)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points_full=points_full,
        springs=springs,
        rest_lengths=rest_lengths,
        masses=masses,
        spring_Y=spring_y,
        num_object_springs=num_object_springs,
        case_name=args.scene,
        train_frame=int(train_frame),
        model_path=str(model_path),
    )
    print("[DONE]", output_path)


if __name__ == "__main__":
    main()
