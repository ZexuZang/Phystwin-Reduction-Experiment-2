#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phystwin_reduction.bt_guided import compute_bt_node_scores
from phystwin_reduction.topology import (
    generate_connected_pruned_topology,
    load_topology,
    normalize_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate online-error-aware stiffness and BT-guided topologies."
    )
    parser.add_argument("--topology-path", required=True, type=Path)
    parser.add_argument("--node-error-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--keep-ratio", type=float, default=0.3)
    parser.add_argument("--online-error-weight", type=float, default=0.3)
    parser.add_argument("--bt-weight", type=float, default=0.7)
    parser.add_argument("--min-degree", type=int, default=1)
    parser.add_argument("--allow-bridge", action="store_true")
    parser.add_argument("--local-budget", type=int, default=300)
    parser.add_argument("--max-input-nodes", type=int, default=2)
    parser.add_argument("--reduced-order", type=int, default=20)
    parser.add_argument("--damping", type=float, default=1e-1)
    parser.add_argument("--anchor", type=float, default=1e-3)
    parser.add_argument("--feedthrough", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.online_error_weight <= 1:
        raise ValueError("--online-error-weight must be in [0, 1]")
    if not 0 <= args.bt_weight <= 1:
        raise ValueError("--bt-weight must be in [0, 1]")

    topology_path = args.topology_path.expanduser().resolve()
    node_error_path = args.node_error_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    topology = load_topology(topology_path)
    error_data = np.load(node_error_path)
    key = (
        "node_error_normalized"
        if "node_error_normalized" in error_data.files
        else "node_error"
    )
    node_error = normalize_score(np.asarray(error_data[key], dtype=np.float64))

    i = topology.object_springs[:, 0].astype(np.int64)
    j = topology.object_springs[:, 1].astype(np.int64)
    max_node = max(int(i.max(initial=0)), int(j.max(initial=0)))
    if len(node_error) <= max_node:
        raise ValueError(
            f"Node error length {len(node_error)} is too small for edge index {max_node}"
        )

    stiffness = normalize_score(
        topology.object_spring_Y
        / np.maximum(topology.object_rest_lengths, 1e-8)
    )
    online_edge_error = normalize_score(
        0.5 * (node_error[i] + node_error[j])
    )
    online_stiffness = normalize_score(
        (1.0 - args.online_error_weight) * stiffness
        + args.online_error_weight * online_edge_error
    )

    keep_tag = int(round(args.keep_ratio * 100))
    stiffness_path = (
        output_dir / f"online_stiffness_keep_{keep_tag}.npz"
    )
    _, stiffness_meta = generate_connected_pruned_topology(
        topology_path,
        stiffness_path,
        online_stiffness,
        keep_ratio=args.keep_ratio,
        min_degree=args.min_degree,
        allow_bridge=args.allow_bridge,
        method="online_stiffness",
        extra_metadata={
            "online_error_weight": args.online_error_weight,
        },
        extra_arrays={
            "stiffness_edge_score": stiffness,
            "online_edge_error": online_edge_error,
        },
    )

    bt_node_scores, bt_info = compute_bt_node_scores(
        topology,
        local_budget=args.local_budget,
        max_input_nodes=args.max_input_nodes,
        reduced_order=args.reduced_order,
        damping=args.damping,
        anchor=args.anchor,
        feedthrough=args.feedthrough,
    )
    bt_edge = normalize_score(
        0.5 * (bt_node_scores[i] + bt_node_scores[j])
    )
    base_bt = normalize_score(
        args.bt_weight * bt_edge + (1.0 - args.bt_weight) * stiffness
    )
    online_bt = normalize_score(
        (1.0 - args.online_error_weight) * base_bt
        + args.online_error_weight * online_edge_error
    )
    bt_tag = f"w{int(round(args.bt_weight * 10)):02d}"
    bt_path = (
        output_dir
        / f"online_bt_guided_keep_{keep_tag}_{bt_tag}.npz"
    )
    scalar_bt_info = {
        key: value
        for key, value in bt_info.items()
        if key not in {"local_nodes", "seed_nodes"}
    }
    _, bt_meta = generate_connected_pruned_topology(
        topology_path,
        bt_path,
        online_bt,
        keep_ratio=args.keep_ratio,
        min_degree=args.min_degree,
        allow_bridge=args.allow_bridge,
        method="online_bt_guided",
        extra_metadata={
            "online_error_weight": args.online_error_weight,
            "bt_weight": args.bt_weight,
            **scalar_bt_info,
        },
        extra_arrays={
            "global_node_scores": bt_node_scores,
            "bt_edge_score": bt_edge,
            "stiffness_edge_score": stiffness,
            "online_edge_error": online_edge_error,
            "local_nodes": bt_info["local_nodes"],
            "seed_nodes": bt_info["seed_nodes"],
        },
    )
    summary = {
        "online_stiffness": str(stiffness_path),
        "online_bt_guided": str(bt_path),
        "stiffness_metadata": stiffness_meta,
        "bt_metadata": bt_meta,
    }
    (output_dir / "online_topology_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print("[DONE]", stiffness_path)
    print("[DONE]", bt_path)


if __name__ == "__main__":
    main()
