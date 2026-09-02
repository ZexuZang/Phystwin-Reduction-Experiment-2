#!/usr/bin/env python3
from __future__ import annotations

"""End-to-end Gaussian render-refinement pipeline.

This script can refine any fixed PhysTwin trajectory.  The trajectory may be
provided directly with --inference or resolved from a generic NeuSpring summary.
No spring budget, scene, candidate id, or Colab path is hard-coded.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(REPO / "src"))

from phystwin_reduction.render_refinement import (
    build_subprocess_env,
    resolve_gaussian_model,
    resolve_gaussian_source,
    resolve_gt_root,
    resolve_human_root,
    resolve_inference_from_neuspring_summary,
    slug,
    temporarily_replace_file,
    validate_variants,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed-physics Gaussian RGB/opacity/scale refinement, rendering, "
            "and multiview evaluation."
        )
    )
    parser.add_argument("--phystwin-root", required=True, type=Path)
    parser.add_argument("--scene", required=True)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inference", type=Path)
    source.add_argument(
        "--neuspring-summary",
        type=Path,
        help="target_*/summary.json from run_neuspring_pipeline.py",
    )
    parser.add_argument(
        "--summary-model",
        choices=[
            "prior-adapt",
            "prior-joint",
            "winner-adapt",
            "winner-joint",
        ],
        default="prior-adapt",
        help="Physical trajectory to resolve when --neuspring-summary is used.",
    )

    parser.add_argument("--run-name")
    parser.add_argument("--gaussian-model", type=Path)
    parser.add_argument("--gaussian-source", type=Path)
    parser.add_argument("--gt-root", type=Path)
    parser.add_argument("--human-root", type=Path)
    parser.add_argument("--output-root", type=Path)

    parser.add_argument(
        "--variants",
        nargs="+",
        default=["rgb_only", "rgb_opacity", "rgb_opa_scale"],
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-train-frames", type=int, default=24)
    parser.add_argument(
        "--camera-indices",
        type=int,
        nargs="+",
        default=[0, 50, 100],
    )
    parser.add_argument(
        "--gt-view-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )
    parser.add_argument("--lr-rgb", type=float, default=1e-3)
    parser.add_argument("--lr-opacity", type=float, default=5e-4)
    parser.add_argument("--lr-scale", type=float, default=2e-4)
    parser.add_argument("--lambda-ssim", type=float, default=0.20)
    parser.add_argument("--lambda-alpha", type=float, default=0.10)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument(
        "--backend",
        choices=["gsplat", "legacy"],
        default="gsplat",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the baseline and every fine-tuned Gaussian model.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run multiview PSNR/SSIM/LPIPS/IoU after rendering.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed variants/renders when their expected outputs exist.",
    )
    return parser.parse_args()


def run(cmd: list[object], env: dict[str, str]) -> None:
    print("\n" + "=" * 100)
    print("$", " ".join(str(x) for x in cmd))
    print("=" * 100)
    subprocess.run(
        [str(x) for x in cmd],
        check=True,
        cwd=str(REPO),
        env=env,
    )


def main() -> None:
    args = parse_args()
    root = args.phystwin_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    variants = validate_variants(args.variants)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.num_train_frames <= 0:
        raise ValueError("--num-train-frames must be positive")
    if len(args.camera_indices) != len(args.gt_view_ids):
        raise ValueError("--camera-indices and --gt-view-ids must have equal length")

    if args.inference is not None:
        inference = args.inference.expanduser().resolve()
        default_run_name = inference.parent.name
    else:
        inference = resolve_inference_from_neuspring_summary(
            args.neuspring_summary,
            args.summary_model,
        )
        default_run_name = args.summary_model

    if not inference.is_file():
        raise FileNotFoundError(inference)

    run_name = slug(args.run_name or default_run_name)
    gaussian_model = resolve_gaussian_model(root, args.scene, args.gaussian_model)
    gaussian_source = resolve_gaussian_source(
        root,
        args.scene,
        args.gaussian_source,
    )
    gt_root = resolve_gt_root(root, args.scene, args.gt_root)
    human_root = resolve_human_root(root, args.scene, args.human_root)

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else root
        / "results"
        / "render_refinement"
        / args.scene
        / run_name
    )
    variants_root = output_root / "variants"
    renders_root = output_root / "renders"
    pose_cache = output_root / "dynamic_pose_cache.pt"
    output_root.mkdir(parents=True, exist_ok=True)

    env = build_subprocess_env(root, REPO)
    python_bin = sys.executable

    print("=" * 100)
    print("GENERIC RENDER REFINEMENT PIPELINE")
    print("scene           :", args.scene)
    print("run name        :", run_name)
    print("inference       :", inference)
    print("gaussian model  :", gaussian_model)
    print("gaussian source :", gaussian_source)
    print("gt root         :", gt_root)
    print("human root      :", human_root)
    print("variants        :", variants)
    print("output          :", output_root)
    print("=" * 100)

    variant_models: dict[str, Path] = {}

    for variant in variants:
        variant_dir = variants_root / variant
        summary = variant_dir / "summary.json"
        output_model = variant_dir / "gaussian_model"

        if not (
            args.resume
            and summary.is_file()
            and output_model.is_dir()
        ):
            cmd: list[object] = [
                python_bin,
                SCRIPTS / "finetune_gaussian_render.py",
                "--source_path",
                gaussian_source,
                "--model_path",
                gaussian_model,
                "--inference",
                inference,
                "--gt-root",
                gt_root,
                "--out-dir",
                variants_root,
                "--pose-cache",
                pose_cache,
                "--variant",
                variant,
                "--steps",
                args.steps,
                "--num-train-frames",
                args.num_train_frames,
                "--camera-indices",
                *args.camera_indices,
                "--gt-view-ids",
                *args.gt_view_ids,
                "--lr-rgb",
                args.lr_rgb,
                "--lr-opacity",
                args.lr_opacity,
                "--lr-scale",
                args.lr_scale,
                "--lambda-ssim",
                args.lambda_ssim,
                "--lambda-alpha",
                args.lambda_alpha,
                "--lambda-reg",
                args.lambda_reg,
                "--backend",
                args.backend,
                "--seed",
                args.seed,
            ]
            if human_root is not None:
                cmd += ["--human-root", human_root]
            run(cmd, env)

        if not output_model.is_dir():
            raise FileNotFoundError(output_model)
        variant_models[variant] = output_model

    render_rows: list[dict[str, object]] = []
    render_mappings: list[tuple[str, Path]] = []

    if args.render or args.evaluate:
        default_inference = (
            root / "experiments" / args.scene / "inference.pkl"
        )
        if not default_inference.is_file():
            raise FileNotFoundError(default_inference)

        render_models: list[tuple[str, Path]] = [
            ("baseline", gaussian_model),
            *[(variant, variant_models[variant]) for variant in variants],
        ]

        renders_root.mkdir(parents=True, exist_ok=True)

        # All render variants use exactly the same fixed physical trajectory.
        with temporarily_replace_file(default_inference, inference):
            for label, model_path in render_models:
                output_dir = renders_root / label
                expected_scene_dir = output_dir / args.scene

                if (
                    args.resume
                    and expected_scene_dir.is_dir()
                    and any(expected_scene_dir.rglob("*.png"))
                ):
                    print("[RESUME] render:", label)
                    render_mappings.append((label, output_dir))
                    continue

                if output_dir.exists():
                    shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                command = [
                    python_bin,
                    root / "gs_render_dynamics.py",
                    "--source_path",
                    gaussian_source,
                    "--model_path",
                    model_path,
                    "--name",
                    args.scene,
                    "--output_dir",
                    output_dir,
                ]

                print("\n" + "#" * 100)
                print("RENDER:", label)
                print("#" * 100)
                start = time.perf_counter()
                result = subprocess.run(
                    [str(x) for x in command],
                    cwd=str(root),
                    env=env,
                )
                elapsed = time.perf_counter() - start
                png_count = len(list(output_dir.rglob("*.png")))

                render_rows.append(
                    {
                        "Method": label,
                        "Gaussian Model": str(model_path),
                        "Render Root": str(output_dir),
                        "Rendered PNGs": png_count,
                        "Seconds": elapsed,
                        "Return Code": result.returncode,
                    }
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Rendering failed: {label}")
                if png_count == 0:
                    raise RuntimeError(f"Rendering produced no PNGs: {label}")
                render_mappings.append((label, output_dir))

        if render_rows:
            render_status = output_root / "render_status.csv"
            with render_status.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(render_rows[0].keys()),
                )
                writer.writeheader()
                writer.writerows(render_rows)

    metrics_csv: Path | None = None
    if args.evaluate:
        if not render_mappings:
            render_mappings = [
                ("baseline", renders_root / "baseline"),
                *[(variant, renders_root / variant) for variant in variants],
            ]

        metrics_csv = output_root / "render_metrics_multiview.csv"
        cmd = [
            python_bin,
            SCRIPTS / "evaluate_rendering_multiview.py",
            "--phystwin-root",
            root,
            "--scene",
            args.scene,
            "--gt-root",
            gt_root,
            "--view-ids",
            *args.gt_view_ids,
            "--output-csv",
            metrics_csv,
        ]
        if human_root is not None:
            cmd += ["--human-root", human_root]
        for label, render_dir in render_mappings:
            cmd += ["--render", f"{label}={render_dir}"]
        run(cmd, env)

    summary = {
        "protocol": "generic_fixed_physics_render_refinement",
        "scene": args.scene,
        "run_name": run_name,
        "inference": str(inference),
        "physics_trajectory_frozen": True,
        "physics_topology_frozen": True,
        "source_gaussian_model": str(gaussian_model),
        "gaussian_source": str(gaussian_source),
        "gt_root": str(gt_root),
        "human_root": str(human_root) if human_root else None,
        "variants": {
            variant: str(variant_models[variant])
            for variant in variants
        },
        "pose_cache": str(pose_cache),
        "render_root": str(renders_root) if (args.render or args.evaluate) else None,
        "metrics_csv": str(metrics_csv) if metrics_csv else None,
        "test_frames_used_for_optimization": False,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("[DONE] Generic render refinement")
    print("summary:", summary_path)
    if metrics_csv:
        print("metrics:", metrics_csv)
    print("=" * 100)


if __name__ == "__main__":
    main()
