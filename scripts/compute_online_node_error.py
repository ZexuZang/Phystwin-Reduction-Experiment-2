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
from phystwin_reduction.phystwin_runtime import (
    load_split,
    resolve_scene_root,
    stage_boundaries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute online per-node error.")
    parser.add_argument("--phystwin-root", required=True, type=Path)
    parser.add_argument("--scene", default="double_stretch_sloth")
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--stage1-ratio", type=float, default=0.5)
    parser.add_argument("--online-start", type=int)
    parser.add_argument("--online-end", type=int)
    parser.add_argument("--inference-path", required=True, type=Path)
    parser.add_argument("--topology-path", required=True, type=Path)
    parser.add_argument("--output-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.phystwin_root.expanduser().resolve()
    scene_root = resolve_scene_root(root, args.scene, args.base_path)
    split = load_split(scene_root)
    _, stage1_end, train_end, _, _ = stage_boundaries(
        split, args.stage1_ratio
    )
    online_start = (
        int(args.online_start)
        if args.online_start is not None
        else stage1_end
    )
    online_end = (
        int(args.online_end)
        if args.online_end is not None
        else train_end
    )
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path
        else root
        / "results"
        / "online_adaptation"
        / args.scene
        / "node_error_online.npz"
    )

    error, normalized, metadata = compute_online_node_error(
        args.inference_path,
        scene_root / "final_data.pkl",
        args.topology_path,
        online_start=online_start,
        online_end=online_end,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        node_error=error,
        node_error_normalized=normalized,
        **{key: np.asarray(value) for key, value in metadata.items()},
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print("[DONE]", output_path)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
