from __future__ import annotations

from pathlib import Path


def resolve_phystwin_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PhysTwin root does not exist: {root}")
    return root


def default_topology_path(root: Path, scene: str) -> Path:
    return root / "results" / f"{scene}_phystwin_topology_open3d.npz"


def keep_tag(keep_ratio: float) -> str:
    return str(int(round(keep_ratio * 100)))


def bt_weight_tag(weight: float) -> str:
    return f"w{int(round(weight * 10)):02d}"
