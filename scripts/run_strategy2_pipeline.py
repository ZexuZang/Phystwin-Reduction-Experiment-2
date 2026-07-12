#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT / "src"))

from phystwin_reduction.phystwin_runtime import (
    load_split,
    resolve_scene_root,
    stage_boundaries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Strategy-2 workflow: Stage-1 training, online "
            "node error, node clustering, online spring pruning, and inference."
        )
    )
    parser.add_argument("--phystwin-root", required=True, type=Path)
    parser.add_argument("--scene", default="double_stretch_sloth")
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--stage1-ratio", type=float, default=0.5)
    parser.add_argument("--stage1-model-path", type=Path)
    parser.add_argument(
        "--skip-stage1-training",
        action="store_true",
        help="Use --stage1-model-path or an existing checkpoint.",
    )
    parser.add_argument("--node-keep-ratio", type=float, default=0.8)
    parser.add_argument("--max-cluster-size", type=int, default=8)
    parser.add_argument("--spring-keep-ratio", type=float, default=0.3)
    parser.add_argument("--online-error-weight", type=float, default=0.3)
    parser.add_argument("--bt-weight", type=float, default=0.7)
    parser.add_argument("--local-budget", type=int, default=300)
    parser.add_argument("--reduced-order", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n" + "=" * 100)
    print("$", " ".join(command))
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def latest_checkpoint(train_dir: Path) -> Path:
    candidates = list((train_dir / "train").glob("iter_*.pth"))
    candidates += list((train_dir / "train").glob("best_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under: {train_dir / 'train'}")

    def iteration(path: Path) -> int:
        numbers = re.findall(r"\d+", path.stem)
        return int(numbers[-1]) if numbers else -1

    return sorted(
        candidates,
        key=lambda path: (iteration(path), path.stat().st_mtime),
    )[-1]


def main() -> None:
    args = parse_args()
    root = args.phystwin_root.expanduser().resolve()
    scene_root = resolve_scene_root(root, args.scene, args.base_path)
    split = load_split(scene_root)
    train_start, stage1_end, train_end, test_start, test_end = stage_boundaries(
        split, args.stage1_ratio
    )

    online_root = root / "results" / "online_adaptation" / args.scene
    stage1_dir = online_root / f"stage1_train_{int(round(args.stage1_ratio * 100))}"
    topology_dir = online_root / "topologies"
    node_topology_dir = online_root / "topologies_node_clustering"
    runs_dir = online_root / "runs"
    for path in [online_root, stage1_dir, topology_dir, node_topology_dir, runs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    base_args = [
        "--phystwin-root", str(root),
        "--scene", args.scene,
        "--stage1-ratio", str(args.stage1_ratio),
    ]
    if args.base_path:
        base_args += ["--base-path", str(args.base_path.expanduser().resolve())]

    model_path = (
        args.stage1_model_path.expanduser().resolve()
        if args.stage1_model_path
        else None
    )
    if not args.skip_stage1_training and model_path is None:
        run(
            [
                sys.executable,
                str(SCRIPTS / "train_stage1.py"),
                *base_args,
                "--output-dir",
                str(stage1_dir),
                "--seed",
                str(args.seed),
            ]
        )
    if model_path is None:
        model_path = latest_checkpoint(stage1_dir)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    stage1_topology = topology_dir / "stage1_official_topology.npz"
    run(
        [
            sys.executable,
            str(SCRIPTS / "export_stage1_topology.py"),
            *base_args,
            "--model-path",
            str(model_path),
            "--output-path",
            str(stage1_topology),
            "--seed",
            str(args.seed),
        ]
    )

    stage1_rollout_dir = runs_dir / "stage1_full_rollout"
    stage1_inference = stage1_rollout_dir / "inference.pkl"
    run(
        [
            sys.executable,
            str(SCRIPTS / "run_external_topology_inference.py"),
            "--phystwin-root",
            str(root),
            "--scene",
            args.scene,
            "--train-frame",
            str(stage1_end),
            "--model-path",
            str(model_path),
            "--topology-path",
            str(stage1_topology),
            "--output-dir",
            str(stage1_rollout_dir),
            "--seed",
            str(args.seed),
            *(
                ["--base-path", str(args.base_path.expanduser().resolve())]
                if args.base_path
                else []
            ),
        ]
    )

    node_error_path = online_root / "node_error_online.npz"
    run(
        [
            sys.executable,
            str(SCRIPTS / "compute_online_node_error.py"),
            *base_args,
            "--inference-path",
            str(stage1_inference),
            "--topology-path",
            str(stage1_topology),
            "--output-path",
            str(node_error_path),
        ]
    )

    node_tag = int(round(args.node_keep_ratio * 100))
    node_topology = (
        node_topology_dir / f"online_cluster_node_keep_{node_tag}.npz"
    )
    run(
        [
            sys.executable,
            str(SCRIPTS / "generate_node_cluster_topology.py"),
            *base_args,
            "--topology-path",
            str(stage1_topology),
            "--inference-path",
            str(stage1_inference),
            "--node-error-path",
            str(node_error_path),
            "--output-path",
            str(node_topology),
            "--node-keep-ratio",
            str(args.node_keep_ratio),
            "--max-cluster-size",
            str(args.max_cluster_size),
        ]
    )

    node_run_dir = runs_dir / f"node_cluster_keep_{node_tag}"
    node_inference = node_run_dir / "inference.pkl"
    run(
        [
            sys.executable,
            str(SCRIPTS / "run_external_topology_inference.py"),
            "--phystwin-root",
            str(root),
            "--scene",
            args.scene,
            "--train-frame",
            str(train_end),
            "--model-path",
            str(model_path),
            "--topology-path",
            str(node_topology),
            "--output-dir",
            str(node_run_dir),
            "--seed",
            str(args.seed),
            *(
                ["--base-path", str(args.base_path.expanduser().resolve())]
                if args.base_path
                else []
            ),
        ]
    )

    reconstructed_dir = runs_dir / f"node_cluster_keep_{node_tag}_reconstructed"
    reconstructed = reconstructed_dir / "inference.pkl"
    run(
        [
            sys.executable,
            str(SCRIPTS / "reconstruct_node_cluster_trajectory.py"),
            "--reduced-inference-path",
            str(node_inference),
            "--topology-path",
            str(node_topology),
            "--output-path",
            str(reconstructed),
        ]
    )

    run(
        [
            sys.executable,
            str(SCRIPTS / "generate_online_spring_topologies.py"),
            "--topology-path",
            str(stage1_topology),
            "--node-error-path",
            str(node_error_path),
            "--output-dir",
            str(topology_dir),
            "--keep-ratio",
            str(args.spring_keep_ratio),
            "--online-error-weight",
            str(args.online_error_weight),
            "--bt-weight",
            str(args.bt_weight),
            "--local-budget",
            str(args.local_budget),
            "--reduced-order",
            str(args.reduced_order),
        ]
    )

    keep_tag = int(round(args.spring_keep_ratio * 100))
    bt_tag = f"w{int(round(args.bt_weight * 10)):02d}"
    stiffness_topology = topology_dir / f"online_stiffness_keep_{keep_tag}.npz"
    bt_topology = topology_dir / f"online_bt_guided_keep_{keep_tag}_{bt_tag}.npz"

    inference_outputs = {}
    for method, topology_path in [
        ("online_stiffness", stiffness_topology),
        ("online_bt_guided", bt_topology),
    ]:
        output_dir = runs_dir / method
        run(
            [
                sys.executable,
                str(SCRIPTS / "run_external_topology_inference.py"),
                "--phystwin-root",
                str(root),
                "--scene",
                args.scene,
                "--train-frame",
                str(train_end),
                "--model-path",
                str(model_path),
                "--topology-path",
                str(topology_path),
                "--output-dir",
                str(output_dir),
                "--seed",
                str(args.seed),
                *(
                    ["--base-path", str(args.base_path.expanduser().resolve())]
                    if args.base_path
                    else []
                ),
            ]
        )
        inference_outputs[method] = str(output_dir / "inference.pkl")

    summary = {
        "scene": args.scene,
        "train": [train_start, train_end],
        "stage1_end": stage1_end,
        "online": [stage1_end, train_end],
        "test": [test_start, test_end],
        "stage1_model": str(model_path),
        "stage1_topology": str(stage1_topology),
        "stage1_inference": str(stage1_inference),
        "node_error": str(node_error_path),
        "node_cluster_topology": str(node_topology),
        "node_cluster_inference": str(node_inference),
        "node_cluster_reconstructed_inference": str(reconstructed),
        "online_stiffness_topology": str(stiffness_topology),
        "online_bt_topology": str(bt_topology),
        "inference_outputs": inference_outputs,
    }
    summary_path = online_root / "strategy2_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[DONE]", summary_path)


if __name__ == "__main__":
    main()
