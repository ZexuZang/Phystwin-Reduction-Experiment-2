from __future__ import annotations

"""Generic utilities for Gaussian render refinement.

This module deliberately contains no scene name, Colab path, fixed spring
budget, or fixed NeuSpring candidate.  It is shared by the render-refinement
training and orchestration scripts.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
from typing import Iterable, Sequence

import numpy as np


RENDER_VARIANTS: dict[str, tuple[str, ...]] = {
    "rgb_only": ("features_dc",),
    "rgb_opacity": ("features_dc", "opacity"),
    "rgb_opa_scale": ("features_dc", "opacity", "scaling"),
}


def validate_variants(values: Iterable[str]) -> list[str]:
    variants = list(values)
    if not variants:
        raise ValueError("At least one render-refinement variant is required")
    unknown = [v for v in variants if v not in RENDER_VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown variants {unknown}; choose from {sorted(RENDER_VARIANTS)}"
        )
    return variants


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("_.-")
    return value or "run"


def load_split(scene_root: str | Path) -> dict:
    scene_root = Path(scene_root)
    split_path = scene_root / "split.json"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    for key in ("train", "test"):
        if key not in split or len(split[key]) != 2:
            raise ValueError(f"split.json must contain {key}=[start,end]")
    return split


def uniform_train_frames(
    split: dict,
    count: int,
    *,
    skip_rest_frame: bool = True,
) -> list[int]:
    start, end = map(int, split["train"])
    first = start + 1 if skip_rest_frame else start
    frames = list(range(first, end))
    if not frames:
        raise ValueError(f"Empty training interval [{first}, {end})")
    if count <= 0 or count >= len(frames):
        return frames
    ids = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return [frames[int(i)] for i in ids]


def resolve_gaussian_model(
    phystwin_root: str | Path,
    scene: str,
    explicit: str | Path | None = None,
) -> Path:
    root = Path(phystwin_root).expanduser().resolve()
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path

    base = root / "gaussian_output" / scene
    candidates = sorted(
        p for p in base.glob("init=*") if p.is_dir()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No Gaussian model found under {base}. "
            "Pass --gaussian-model explicitly."
        )
    raise RuntimeError(
        f"Multiple Gaussian models found under {base}: "
        f"{[p.name for p in candidates]}. "
        "Pass --gaussian-model explicitly."
    )


def resolve_gaussian_source(
    phystwin_root: str | Path,
    scene: str,
    explicit: str | Path | None = None,
) -> Path:
    root = Path(phystwin_root).expanduser().resolve()
    path = (
        Path(explicit).expanduser().resolve()
        if explicit is not None
        else root / "data" / "gaussian_data" / scene
    )
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def resolve_gt_root(
    phystwin_root: str | Path,
    scene: str,
    explicit: str | Path | None = None,
) -> Path:
    root = Path(phystwin_root).expanduser().resolve()
    path = (
        Path(explicit).expanduser().resolve()
        if explicit is not None
        else root / "data" / "different_types" / scene
    )
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def resolve_human_root(
    phystwin_root: str | Path,
    scene: str,
    explicit: str | Path | None = None,
) -> Path | None:
    root = Path(phystwin_root).expanduser().resolve()
    path = (
        Path(explicit).expanduser().resolve()
        if explicit is not None
        else root / "data" / "different_types_human_mask" / scene
    )
    return path if path.is_dir() else None


def resolve_inference_from_neuspring_summary(
    summary_path: str | Path,
    model: str,
) -> Path:
    """Resolve a physical trajectory from a generic NeuSpring target summary.

    Supported models:
      prior-adapt, prior-joint, winner-adapt, winner-joint
    """

    path = Path(summary_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))

    # Accept an overall multi-target summary only when it contains one target.
    if "targets" in summary:
        targets = summary["targets"]
        if len(targets) != 1:
            raise ValueError(
                "Overall NeuSpring summary contains multiple targets. "
                "Pass a target_*/summary.json instead."
            )
        summary = targets[0]

    selection = summary.get("selection", {})
    model = model.lower()

    if model == "winner-joint":
        candidate = summary.get("winner_joint_inference")
    elif model == "winner-adapt":
        winner_dir = selection.get("winner_directory")
        candidate = (
            str(Path(winner_dir) / "adapt" / "inference.pkl")
            if winner_dir
            else None
        )
    elif model == "prior-adapt":
        prior_dir = selection.get("prior_control_directory")
        candidate = (
            str(Path(prior_dir) / "adapt" / "inference.pkl")
            if prior_dir
            else None
        )
    elif model == "prior-joint":
        prior_dir = selection.get("prior_control_directory")
        candidate = (
            str(Path(prior_dir) / "joint_nsf" / "inference" / "inference.pkl")
            if prior_dir
            else None
        )
    else:
        raise ValueError(
            "--summary-model must be one of: "
            "prior-adapt, prior-joint, winner-adapt, winner-joint"
        )

    if not candidate:
        raise KeyError(
            f"Could not resolve {model!r} from NeuSpring summary {path}"
        )
    result = Path(candidate).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def infer_object_mask_id(gt_root: str | Path, view_id: int) -> int:
    """Infer the object mask id for one rendered view.

    Preferred source is mask_info_<view>.json.  A unique numeric directory
    under mask/<view> is used as a fallback.
    """

    gt_root = Path(gt_root)
    info_path = gt_root / "mask" / f"mask_info_{view_id}.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        ids = [
            int(key)
            for key, value in info.items()
            if str(value).strip().lower() != "hand"
        ]
        if len(ids) == 1:
            return ids[0]
        if len(ids) > 1:
            raise RuntimeError(
                f"View {view_id}: multiple non-hand mask ids {ids} in {info_path}"
            )

    view_root = gt_root / "mask" / str(view_id)
    if view_root.is_dir():
        ids = sorted(
            int(p.name)
            for p in view_root.iterdir()
            if p.is_dir() and p.name.isdigit()
        )
        if len(ids) == 1:
            return ids[0]

    raise RuntimeError(
        f"Cannot infer object mask id for view {view_id}. "
        f"Expected {info_path} or a unique numeric directory under {view_root}."
    )


def find_human_mask(
    human_root: str | Path | None,
    view_id: int,
    frame: int,
) -> Path | None:
    if human_root is None:
        return None
    root = Path(human_root)
    if not root.is_dir():
        return None

    candidates = [
        root / "mask" / str(view_id) / "0" / f"{frame}.png",
        root / "mask" / str(view_id) / f"{frame}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    base = root / "mask" / str(view_id)
    if base.is_dir():
        hits = list(base.rglob(f"{frame}.png"))
        if hits:
            return hits[0]
    return None


def detect_render_view_root(render_root: str | Path, scene: str) -> Path:
    root = Path(render_root).expanduser().resolve()
    with_scene = root / scene
    if with_scene.is_dir():
        return with_scene
    return root


@contextmanager
def temporarily_replace_file(target: Path, source: Path):
    target = target.expanduser().resolve()
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    backup = target.with_name(
        f"{target.stem}_backup_before_render_refinement{target.suffix}"
    )
    had_original = target.exists()
    if had_original:
        shutil.copy2(target, backup)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source != target:
            shutil.copy2(source, target)
        yield
    finally:
        if had_original and backup.is_file():
            shutil.copy2(backup, target)
            backup.unlink()
        elif not had_original and target.exists():
            target.unlink()


def build_subprocess_env(
    phystwin_root: str | Path,
    repo_root: str | Path,
) -> dict[str, str]:
    env = os.environ.copy()
    root = Path(phystwin_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()

    python_paths = [str(root), str(repo / "src")]
    old_pythonpath = env.get("PYTHONPATH", "")
    if old_pythonpath:
        python_paths.append(old_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_SILENT", "true")
    env.setdefault("SKIP_OPEN3D_VIDEO", "1")
    env.setdefault(
        "TORCH_EXTENSIONS_DIR",
        str(root / ".cache" / "torch_extensions_render_refinement"),
    )

    # Make the ninja executable visible to PyTorch JIT extensions when the
    # Python package is installed in the same environment.
    try:
        import ninja  # type: ignore

        env["PATH"] = os.pathsep.join(
            [str(ninja.BIN_DIR), env.get("PATH", "")]
        )
    except Exception:
        pass

    # Add the active PyTorch shared-library directory and use only the current
    # GPU architecture instead of compiling every visible architecture.
    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(torch_lib), env.get("LD_LIBRARY_PATH", "")]
        )
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            env.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    except Exception:
        pass

    return env
