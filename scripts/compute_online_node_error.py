#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phystwin_reduction.online_adaptation import compute_online_node_error
from phystwin_reduction.phystwin_runtime import load_split, resolve_scene_root, stage_boundaries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute online per-dense-node error.")
    p.add_argument("--phystwin-root", required=True, type=Path)
    p.add_argument("--scene", required=True)
    p.add_argument("--base-path", type=Path)
    p.add_argument("--stage1-ratio", type=float, default=0.5)
    p.add_argument("--online-start", type=int)
    p.add_argument("--online-end", type=int)
    p.add_argument("--inference-path", required=True, type=Path)
    p.add_argument("--topology-path", required=True, type=Path)
    p.add_argument("--output-path", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    root = a.phystwin_root.expanduser().resolve()
    scene_root = resolve_scene_root(root, a.scene, a.base_path)
    split = load_split(scene_root)
    _, stage1_end, train_end, _, _ = stage_boundaries(split, a.stage1_ratio)
    online_start = int(a.online_start) if a.online_start is not None else stage1_end
    online_end = int(a.online_end) if a.online_end is not None else train_end

    error, normalized, metadata = compute_online_node_error(
        a.inference_path,
        scene_root / "final_data.pkl",
        a.topology_path,
        online_start=online_start,
        online_end=online_end,
    )
    out = a.output_path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        node_error=error,
        node_error_normalized=normalized,
        **{k: np.asarray(v) for k, v in metadata.items()},
    )
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[DONE]", out)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
