#!/usr/bin/env python3
from __future__ import annotations

"""Stable paper pipeline for PhysTwin hierarchical reduction.

FORMAL INVARIANT
----------------
Stage-2 BT/stiffness importance is NEVER computed on the full graph or on the
pre-retraining node topology.  The order is strictly:

    Full Stage-1 model
        -> node coarsening (Geometry / Trajectory / Krylov / SOAR)
        -> Stage-1 reduced-physics retraining
        -> best reduced checkpoint
        -> write best spring_Y back into topology_retrained.npz
        -> online residual on UPDATE frames
        -> project residual to coarse nodes
        -> recompute BT + stiffness on topology_retrained.npz
        -> connectivity-constrained pruning to FINAL spring budget(s)
        -> pure inference on TEST frames.

For a common split train=[0,134], test=[134,192] and --stage1-ratio 0.5:
    Stage-1 frames : [0, 67)
    UPDATE frames  : [67, 134)
    TEST frames    : [134, 192)

The Stage-2 budget is expressed relative to the FULL spring count, e.g.
``--final-spring-ratios 0.60 0.50``.  It is not "delete another 50%".
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def run(cmd: list[object], *, env: dict[str, str]) -> None:
    print("\n" + "=" * 100)
    print("$", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True, cwd=str(REPO), env=env)


def best_checkpoint(train_output: Path) -> Path:
    history = train_output / "train" / "coarsening_train_history.csv"
    if history.is_file():
        df = pd.read_csv(history)
        if "is_final_best" in df.columns and (df["is_final_best"] == 1).any():
            row = df.loc[df["is_final_best"] == 1].iloc[-1]
        else:
            row = df.loc[df["loss"].idxmin()]
        candidate = train_output / "train" / f"best_{int(row['epoch'])}.pth"
        if candidate.is_file():
            return candidate

    files = list((train_output / "train").glob("best_*.pth"))
    if not files:
        files = list(train_output.rglob("best_*.pth"))
    if not files:
        raise FileNotFoundError(f"No best_*.pth under {train_output}")

    def epoch(path: Path) -> int:
        m = re.search(r"best_(\d+)", path.stem)
        return int(m.group(1)) if m else -1

    return max(files, key=lambda p: (epoch(p), p.stat().st_mtime))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node coarsening -> retrain -> post-retrain Online-BT final-budget pipeline."
    )
    p.add_argument("--phystwin-root", required=True, type=Path)
    p.add_argument("--scene", required=True, help="PhysTwin case name, e.g. double_stretch_sloth")
    p.add_argument("--base-path", type=Path, help="Parent directory containing scene folders")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--python", dest="python_bin", default=sys.executable)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--stage1-ratio", type=float, default=0.5)
    p.add_argument("--stage1-checkpoint", type=Path)
    p.add_argument("--stage1-inference", type=Path)
    p.add_argument("--skip-stage1-training", action="store_true")

    p.add_argument(
    "--baseline-mode",
    choices=["original", "stage1"],
    default="original",
    help=(
        "Baseline source. 'original' uses the archived PhysTwin "
        "checkpoint and inference under experiments/<scene>. "
        "'stage1' reproduces the previous behavior and trains a "
        "new Stage-1 full model."
    ),
)

    p.add_argument(
        "--node-method",
        choices=["geometry", "trajectory", "krylov", "soar"],
        default="soar",
    )
    p.add_argument("--node-keep-ratio", type=float, default=0.75)
    p.add_argument("--node-rank", type=int)
    p.add_argument("--alpha-dyn", type=float, default=1.0)
    p.add_argument("--beta-geo", type=float, default=1.0)
    p.add_argument("--protect-top-pct", type=float, default=10.0)
    p.add_argument("--max-cluster-size", type=int, default=16)
    p.add_argument("--mapping-k", type=int, default=4)
    p.add_argument("--dashpot", type=float, default=100.0)
    p.add_argument("--drag", type=float, default=3.0)

    p.add_argument("--retrain-epochs", type=int, default=200)
    p.add_argument("--checkpoint-interval", type=int, default=20)
    p.add_argument("--dt-scale", type=float, default=1.0)

    p.add_argument(
        "--final-spring-ratios",
        type=float,
        nargs="+",
        default=[0.60],
        help="FINAL total spring ratios relative to Full, e.g. 0.60 0.55 0.50",
    )
    p.add_argument("--bt-weight", type=float, default=0.70)
    p.add_argument("--online-error-weight", type=float, default=0.30)
    p.add_argument("--min-degree", type=int, default=1)
    p.add_argument("--local-budget", type=int, default=300)
    p.add_argument("--reduced-order", type=int, default=20)

    p.add_argument("--resume", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--benchmark-fps", action="store_true")
    p.add_argument("--fps-repeats", type=int, default=5)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if not 0.0 < a.stage1_ratio < 1.0:
        raise ValueError("--stage1-ratio must be in (0,1) so UPDATE is non-empty")
    if not 0.0 < a.node_keep_ratio <= 1.0:
        raise ValueError("--node-keep-ratio must be in (0,1]")
    for ratio in a.final_spring_ratios:
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"Invalid final spring ratio: {ratio}")

    python_bin = str(Path(a.python_bin).expanduser())
    root = a.phystwin_root.expanduser().resolve()
    base_path = (
        a.base_path.expanduser().resolve()
        if a.base_path is not None
        else root / "data" / "different_types"
    )
    scene_root = base_path / a.scene
    split_path = scene_root / "split.json"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_start, train_end = map(int, split["train"])
    test_start, test_end = map(int, split["test"])
    stage1_end = train_start + int((train_end - train_start) * a.stage1_ratio)
    if not (train_start < stage1_end < train_end):
        raise ValueError(
            f"Need non-empty Stage-1 and UPDATE windows; train={split['train']}, "
            f"stage1_end={stage1_end}"
        )

    out = (
        a.output_root.expanduser().resolve()
        if a.output_root is not None
        else root
        / "results"
        / "hierarchical_v2"
        / a.scene
        / f"{a.node_method}_node{int(round(a.node_keep_ratio * 100))}"
    )
    stage1_dir = out / "stage1"
    stage1_rollout = out / "stage1_rollout"
    node_dir = out / "node"
    online_dir = out / "online"
    stage2_root = out / "stage2"
    for d in [stage1_dir, stage1_rollout, node_dir, online_dir, stage2_root]:
        d.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(REPO / "src"), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("SKIP_OPEN3D_VIDEO", "1")

    print("===== FORMAL SPLIT =====")
    print("scene        :", a.scene)
    print("Stage-1      :", [train_start, stage1_end])
    print("UPDATE       :", [stage1_end, train_end])
    print("TEST         :", [test_start, test_end])
    print("node method  :", a.node_method)
    print("node keep    :", a.node_keep_ratio)
    print("final budgets:", a.final_spring_ratios)
    print("baseline mode:", a.baseline_mode)


    # ------------------------------------------------------------------
    # Canonical Full baseline
    # ------------------------------------------------------------------

    canonical_original_ckpt = (
        root
        / "experiments"
        / a.scene
        / "train"
        / "best_199.pth"
    )

    canonical_original_inference = (
        root
        / "experiments"
        / a.scene
        / "inference.pkl"
    )

    # Explicit CLI checkpoint always has highest priority.
    stage1_ckpt = (
        a.stage1_checkpoint.expanduser().resolve()
        if a.stage1_checkpoint is not None
        else None
    )

    if stage1_ckpt is None:

        if a.baseline_mode == "original":

            # Canonical Original PhysTwin checkpoint.
            stage1_ckpt = canonical_original_ckpt.resolve()

            if not stage1_ckpt.is_file():
                raise FileNotFoundError(
                    "Canonical Original PhysTwin checkpoint not found: "
                    f"{stage1_ckpt}"
                )

            print("\n===== CANONICAL ORIGINAL BASELINE =====")
            print("checkpoint :", stage1_ckpt)
            print("inference  :", canonical_original_inference)

        else:

            # Legacy behavior: train a new Stage-1 full model.
            if a.skip_stage1_training:
                raise ValueError(
                    "--skip-stage1-training requires "
                    "--stage1-checkpoint when "
                    "--baseline-mode=stage1"
                )

            if not (
                a.resume
                and list(stage1_dir.rglob("best_*.pth"))
            ):
                run(
                    [
                        python_bin,
                        SCRIPTS / "train_stage1.py",
                        "--phystwin-root", root,
                        "--scene", a.scene,
                        "--base-path", base_path,
                        "--train-frame", stage1_end,
                        "--output-dir", stage1_dir,
                        "--seed", a.seed,
                    ],
                    env=env,
                )

            stage1_ckpt = best_checkpoint(stage1_dir)

    if not stage1_ckpt.is_file():
        raise FileNotFoundError(stage1_ckpt)

    full_topology = out / "full_stage1_topology.npz"
    if not (a.resume and full_topology.is_file()):
        run(
            [
                python_bin,
                SCRIPTS / "export_stage1_topology.py",
                "--phystwin-root", root,
                "--scene", a.scene,
                "--base-path", base_path,
                "--train-frame", stage1_end,
                "--model-path", stage1_ckpt,
                "--output-path", full_topology,
                "--seed", a.seed,
            ],
            env=env,
        )

    if a.stage1_inference is not None:

        full_inference = (
            a.stage1_inference
            .expanduser()
            .resolve()
        )

        if not full_inference.is_file():
            raise FileNotFoundError(full_inference)


    elif a.baseline_mode == "original":

        full_inference = (
            canonical_original_inference
            .resolve()
        )

        if not full_inference.is_file():
            raise FileNotFoundError(
                "Canonical Original PhysTwin inference not found: "
                f"{full_inference}"
            )

        print(
            "Using canonical Original inference:",
            full_inference,
        )


    else:

        full_inference = (
            stage1_rollout
            / "inference.pkl"
        )

        if not (
            a.resume
            and full_inference.is_file()
        ):

            run(
                [
                    python_bin,
                    SCRIPTS / "run_external_topology_inference.py",

                    "--phystwin-root", root,
                    "--scene", a.scene,
                    "--base-path", base_path,
                    "--train-frame", train_end,

                    "--model-path", stage1_ckpt,
                    "--topology-path", full_topology,
                    "--output-dir", stage1_rollout,

                    "--seed", a.seed,
                ],
                env=env,
            )

    # ------------------------------------------------------------------
    # Stage 1: node coarsening -> trainer schema -> formal retraining
    # ------------------------------------------------------------------
    node_topology = node_dir / "topology.npz"
    if not (a.resume and node_topology.is_file()):
        cmd: list[object] = [
            python_bin,
            SCRIPTS / "generate_hierarchical_node_topology.py",
            "--topology-path", full_topology,
            "--inference-path", full_inference,
            "--output-path", node_topology,
            "--method", a.node_method,
            "--frame-start", train_start,
            "--frame-end", stage1_end,
            "--keep-ratio", a.node_keep_ratio,
            "--alpha-dyn", a.alpha_dyn,
            "--beta-geo", a.beta_geo,
            "--protect-top-pct", a.protect_top_pct,
            "--max-cluster-size", a.max_cluster_size,
            "--mapping-k", a.mapping_k,
            "--dashpot", a.dashpot,
            "--drag", a.drag,
        ]
        if a.node_rank is not None:
            cmd += ["--rank", a.node_rank]
        run(cmd, env=env)

    node_trainer = node_dir / "trainer.npz"
    if not (a.resume and node_trainer.is_file()):
        run(
            [
                python_bin,
                SCRIPTS / "convert_hierarchical_to_trainer.py",
                "--phystwin-root", root,
                "--scene", a.scene,
                "--base-path", base_path,
                "--input", node_topology,
                "--output", node_trainer,
                "--mode", a.node_method,
            ],
            env=env,
        )

    retrain_dir = node_dir / "retrain"
    retrain_ready = (
        (retrain_dir / "inference.pkl").is_file()
        and list((retrain_dir / "train").glob("best_*.pth"))
    )
    if not (a.resume and retrain_ready):
        run(
            [
                python_bin,
                SCRIPTS / "run_coarsening_retrain.py",
                "--phystwin_root", root,
                "--base_path", base_path,
                "--case_name", a.scene,
                "--coarsened_data", node_trainer,
                "--parent_checkpoint",stage1_ckpt,
                "--out_dir", retrain_dir,
                "--device", a.device,
                "--train_frame", stage1_end,
                "--dt_scale", a.dt_scale,
                "--retrain_epochs", a.retrain_epochs,
                "--checkpoint_interval", a.checkpoint_interval,
                "--seed", a.seed,
            ],
            env=env,
        )
    node_best = best_checkpoint(retrain_dir)
    retrained_dense_inference = retrain_dir / "inference.pkl"
    if not retrained_dense_inference.is_file():
        raise FileNotFoundError(retrained_dense_inference)

    # This file is the hard Stage-2 boundary.  The best retrained spring_Y is
    # written back before any BT/stiffness importance is computed.
    retrained_topology = node_dir / "topology_retrained.npz"
    if not (a.resume and retrained_topology.is_file()):
        run(
            [
                python_bin,
                SCRIPTS / "apply_checkpoint_to_topology.py",
                "--topology", node_topology,
                "--checkpoint", node_best,
                "--output", retrained_topology,
            ],
            env=env,
        )

    # ------------------------------------------------------------------
    # UPDATE only: dense residual -> coarse residual
    # ------------------------------------------------------------------
    dense_error = online_dir / "dense_node_error.npz"
    if not (a.resume and dense_error.is_file()):
        run(
            [
                python_bin,
                SCRIPTS / "compute_online_node_error.py",
                "--phystwin-root", root,
                "--scene", a.scene,
                "--base-path", base_path,
                "--online-start", stage1_end,
                "--online-end", train_end,
                "--inference-path", retrained_dense_inference,
                "--topology-path", full_topology,
                "--output-path", dense_error,
            ],
            env=env,
        )

    coarse_error = online_dir / "coarse_node_error.npz"
    if not (a.resume and coarse_error.is_file()):
        run(
            [
                python_bin,
                SCRIPTS / "project_online_error_to_coarse.py",
                "--dense-node-error", dense_error,
                "--coarsened-topology", retrained_topology,
                "--output-path", coarse_error,
            ],
            env=env,
        )

    # ------------------------------------------------------------------
    # Stage 2: one or more FINAL budgets, all reusing the same post-retrain
    # BT/stiffness source graph and the same UPDATE residual.
    # ------------------------------------------------------------------
    final_runs: list[dict] = []
    for final_ratio in a.final_spring_ratios:
        ratio_tag = f"final{int(round(final_ratio * 100))}"
        stage2_dir = stage2_root / ratio_tag
        stage2_dir.mkdir(parents=True, exist_ok=True)

        stage2_ready = (
            (stage2_dir / "topology.npz").is_file()
            and (stage2_dir / "trainer.npz").is_file()
            and (stage2_dir / "initial.pth").is_file()
        )
        if not (a.resume and stage2_ready):
            run(
                [
                    python_bin,
                    SCRIPTS / "build_final_budget_stage2.py",
                    "--full-topology", full_topology,
                    "--retrained-topology", retrained_topology,
                    "--coarse-node-error", coarse_error,
                    "--base-trainer", node_trainer,
                    "--node-checkpoint", node_best,
                    "--output-dir", stage2_dir,
                    "--target-final-spring-ratio", final_ratio,
                    "--bt-weight", a.bt_weight,
                    "--online-error-weight", a.online_error_weight,
                    "--min-degree", a.min_degree,
                    "--local-budget", a.local_budget,
                    "--reduced-order", a.reduced_order,
                    "--label", f"{a.node_method}_{ratio_tag}",
                ],
                env=env,
            )

        final_inference_dir = stage2_dir / "inference"
        if not (a.resume and (final_inference_dir / "inference.pkl").is_file()):
            run(
                [
                    python_bin,
                    SCRIPTS / "run_coarsened_inference.py",
                    "--phystwin-root", root,
                    "--scene", a.scene,
                    "--base-path", base_path,
                    "--coarsened-data", stage2_dir / "trainer.npz",
                    "--checkpoint", stage2_dir / "initial.pth",
                    "--output-dir", final_inference_dir,
                    "--train-frame", train_end,
                    "--device", a.device,
                    "--seed", a.seed,
                ],
                env=env,
            )

        fps_json = stage2_dir / "fps.json"
        if a.benchmark_fps and not (a.resume and fps_json.is_file()):
            run(
                [
                    python_bin,
                    SCRIPTS / "benchmark_hierarchical_fps.py",
                    "--phystwin-root", root,
                    "--scene", a.scene,
                    "--mode", "reduced",
                    "--label", f"{a.node_method}-{ratio_tag}",
                    "--coarse-path", stage2_dir / "trainer.npz",
                    "--checkpoint", stage2_dir / "initial.pth",
                    "--measure-start", test_start,
                    "--measure-end", test_end,
                    "--repeats", a.fps_repeats,
                    "--output", fps_json,
                ],
                env=env,
            )

        stage2_meta = json.loads(
            (stage2_dir / "stage2_summary.json").read_text(encoding="utf-8")
        )
        final_runs.append(
            {
                "target_final_spring_ratio": final_ratio,
                "stage2_dir": str(stage2_dir),
                "stage2": stage2_meta,
                "dense_inference": str(final_inference_dir / "inference.pkl"),
                "fps_json": str(fps_json) if a.benchmark_fps else None,
            }
        )

    # One geometry table for the node baseline and every requested final budget.
    geometry_csv = out / "geometry.csv"
    if a.evaluate:
        cmd: list[object] = [
            python_bin,
            SCRIPTS / "evaluate_geometry.py",
            "--phystwin-root", root,
            "--scene", a.scene,
            "--run", f"Full-Stage1={full_inference}",
            "--run", f"{a.node_method}-node={retrained_dense_inference}",
        ]
        for item in final_runs:
            ratio = item["target_final_spring_ratio"]
            cmd += [
                "--run",
                f"{a.node_method}-final{int(round(ratio*100))}={item['dense_inference']}",
            ]
        cmd += ["--output-csv", geometry_csv]
        run(cmd, env=env)

    summary = {
        "protocol": "node_coarsening_retrain_then_post_retrain_online_bt_final_budget",
        "scene": a.scene,
        "train_split": [train_start, train_end],
        "stage1_frames": [train_start, stage1_end],
        "update_frames": [stage1_end, train_end],
        "test_frames": [test_start, test_end],
        "node_method": a.node_method,
        "node_keep_ratio": a.node_keep_ratio,
        "retrain_epochs": a.retrain_epochs,
        "bt_weight": a.bt_weight,
        "online_error_weight": a.online_error_weight,
        "stage1_checkpoint": str(stage1_ckpt),
        "full_topology": str(full_topology),
        "full_inference": str(full_inference),
        "node_topology_pre_retrain": str(node_topology),
        "node_trainer": str(node_trainer),
        "node_best_checkpoint": str(node_best),
        "node_topology_retrained": str(retrained_topology),
        "node_retrained_dense_inference": str(retrained_dense_inference),
        "dense_online_error": str(dense_error),
        "coarse_online_error": str(coarse_error),
        "final_runs": final_runs,
        "geometry_csv": str(geometry_csv) if a.evaluate else None,
    }
    (out / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 100)
    print("[DONE] Stable hierarchical V2 pipeline")
    print("scene             :", a.scene)
    print("node method       :", a.node_method)
    print("retrained topology:", retrained_topology)
    print("final budgets     :", a.final_spring_ratios)
    print("summary           :", out / "pipeline_summary.json")


if __name__ == "__main__":
    main()
