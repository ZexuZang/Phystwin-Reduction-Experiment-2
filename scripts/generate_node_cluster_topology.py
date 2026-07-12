#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phystwin_reduction.online_adaptation import (
    generate_node_cluster_topology,
)
from phystwin_reduction.phystwin_runtime import (
    load_split,
    resolve_scene_root,
    stage_boundaries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trajectory-neighbor node-cluster topology."
    )
    parser.add_argument("--phystwin-root", required=True, type=Path)
    parser.add_argument("--scene", default="double_stretch_sloth")
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--stage1-ratio", type=float, default=0.5)
    parser.add_argument("--online-start", type=int)
    parser.add_argument("--online-end", type=int)
    parser.add_argument("--topology-path", required=True, type=Path)
    parser.add_argument("--inference-path", required=True, type=Path)
    parser.add_argument("--node-error-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--node-keep-ratio", type=float, default=0.8)
    parser.add_argument("--max-cluster-size", type=int, default=8)
    parser.add_argument("--high-error-percentile", type=float, default=90.0)
    parser.add_argument("--no-protect-controller", action="store_true")
    parser.add_argument("--no-protect-high-error", action="store_true")
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
    keep_tag = int(round(args.node_keep_ratio * 100))
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path
        else root
        / "results"
        / "online_adaptation"
        / args.scene
        / "topologies_node_clustering"
        / f"online_cluster_node_keep_{keep_tag}.npz"
    )

    node_error = None
    if args.node_error_path:
        data = np.load(args.node_error_path.expanduser().resolve())
        key = (
            "node_error_normalized"
            if "node_error_normalized" in data.files
            else "node_error"
        )
        node_error = np.asarray(data[key], dtype=np.float64)

    path, metadata = generate_node_cluster_topology(
        args.topology_path,
        args.inference_path,
        output_path,
        online_start=online_start,
        online_end=online_end,
        node_keep_ratio=args.node_keep_ratio,
        max_cluster_size=args.max_cluster_size,
        node_error=node_error,
        protect_controller_attached=not args.no_protect_controller,
        protect_high_error=not args.no_protect_high_error,
        high_error_percentile=args.high_error_percentile,
    )
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print("[DONE]", path)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
