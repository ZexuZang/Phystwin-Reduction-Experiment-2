from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import random
import sys
from typing import Any

import numpy as np
import torch

from .pytorch3d_stub import install_pytorch3d_stub


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_scene_root(
    phystwin_root: Path,
    scene: str,
    base_path: Path | None = None,
) -> Path:
    if base_path is not None:
        return base_path.expanduser().resolve() / scene
    return phystwin_root.expanduser().resolve() / "data" / "different_types" / scene


def load_split(scene_root: Path) -> dict[str, Any]:
    path = scene_root / "split.json"
    if not path.is_file():
        raise FileNotFoundError(f"split.json not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stage_boundaries(
    split: dict[str, Any],
    stage1_ratio: float,
) -> tuple[int, int, int, int, int]:
    if not 0 < stage1_ratio <= 1:
        raise ValueError("stage1_ratio must be in (0, 1]")
    train_start, train_end = map(int, split["train"])
    test_start, test_end = map(int, split["test"])
    stage1_end = train_start + int((train_end - train_start) * stage1_ratio)
    return train_start, stage1_end, train_end, test_start, test_end


def prepare_phystwin(
    phystwin_root: Path,
    scene: str,
    scene_root: Path,
    *,
    seed: int = 42,
):
    root = phystwin_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PhysTwin root not found: {root}")
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("SKIP_OPEN3D_VIDEO", "1")

    install_pytorch3d_stub()
    set_all_seeds(seed)

    from qqtt.utils import cfg

    config_path = (
        "configs/cloth.yaml"
        if "cloth" in scene or "package" in scene
        else "configs/real.yaml"
    )
    cfg.load_from_yaml(config_path)

    optimal_path = root / "experiments_optimization" / scene / "optimal_params.pkl"
    if not optimal_path.is_file():
        raise FileNotFoundError(f"Optimal parameters not found: {optimal_path}")
    with optimal_path.open("rb") as handle:
        cfg.set_optimal_params(pickle.load(handle))

    calibrate_path = scene_root / "calibrate.pkl"
    metadata_path = scene_root / "metadata.json"
    final_data_path = scene_root / "final_data.pkl"
    for path in [calibrate_path, metadata_path, final_data_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    with calibrate_path.open("rb") as handle:
        c2ws = pickle.load(handle)
    cfg.c2ws = np.asarray(c2ws)
    cfg.w2cs = np.asarray([np.linalg.inv(c2w) for c2w in c2ws])

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    cfg.intrinsics = np.asarray(metadata["intrinsics"])
    cfg.WH = metadata["WH"]
    cfg.overlay_path = str(scene_root / "color")
    return cfg
