#!/usr/bin/env python3
from __future__ import annotations

"""Run vanilla PhysTwin inference for a FULL topology only.

Node-coarsened / Stage-2 topologies must use ``run_coarsened_inference.py`` and
``InvPhyTrainerWarpCoarsening``.  This guard prevents the old assertion/indexing
failure caused by feeding a reduced graph to the vanilla trainer.
"""

import argparse
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phystwin_reduction.phystwin_runtime import load_split, prepare_phystwin, resolve_scene_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run vanilla PhysTwin inference with a full topology NPZ.")
    p.add_argument("--phystwin-root", required=True, type=Path)
    p.add_argument("--scene", required=True)
    p.add_argument("--base-path", type=Path)
    p.add_argument("--train-frame", type=int)
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--topology-path", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    root = a.phystwin_root.expanduser().resolve()
    scene_root = resolve_scene_root(root, a.scene, a.base_path)
    split = load_split(scene_root)
    train_frame = int(a.train_frame) if a.train_frame is not None else int(split["train"][1])
    model_path = a.model_path.expanduser().resolve()
    topology_path = a.topology_path.expanduser().resolve()
    output_dir = a.output_dir.expanduser().resolve()
    for path in [model_path, topology_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    z = np.load(topology_path, allow_pickle=True)
    if "reduced_object_points" in z.files or "mapping_indices" in z.files:
        raise RuntimeError(
            "Reduced/coarsened topology detected. Do NOT use the vanilla "
            "InvPhyTrainerWarp path. Use scripts/run_coarsened_inference.py."
        )

    os.environ["EXTERNAL_TOPOLOGY_NPZ"] = str(topology_path)
    prepare_phystwin(root, a.scene, scene_root, seed=a.seed)
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
            print("[WARNING] PhysTwin raised after saving inference.pkl; keeping saved trajectory.")
        else:
            raise
    print("[DONE]", inference_path)


if __name__ == "__main__":
    main()
