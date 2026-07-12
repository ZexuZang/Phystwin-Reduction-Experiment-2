#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/inference.pkl")
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Strategy-2 inference trajectories with PhysTwin Gaussian renderer."
    )
    parser.add_argument("--phystwin-root", required=True, type=Path)
    parser.add_argument("--scene", default="double_stretch_sloth")
    parser.add_argument("--node-keep", default="80")
    parser.add_argument(
        "--run",
        action="append",
        type=parse_mapping,
        help="Custom trajectory mapping: NAME=/path/to/inference.pkl",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gaussian-model", type=Path)
    return parser.parse_args()


@contextmanager
def temporarily_replace_inference(default_path: Path, source_path: Path):
    backup = default_path.with_name(
        default_path.stem + "_backup_before_strategy2_render" + default_path.suffix
    )
    had_original = default_path.exists()
    if had_original and not backup.exists():
        shutil.copy2(default_path, backup)
    try:
        if source_path.resolve() != default_path.resolve():
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            default_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, default_path)
        yield
    finally:
        if backup.exists():
            shutil.copy2(backup, default_path)
        elif not had_original and default_path.exists():
            default_path.unlink()


def add_torch_library_path(env: dict[str, str]) -> None:
    try:
        import torch
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    except Exception:
        pass


def slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("%", "")
    )


def main() -> None:
    args = parse_args()
    root = args.phystwin_root.expanduser().resolve()
    online_root = root / "results" / "online_adaptation" / args.scene
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else online_root / "renders"
    )
    source_path = root / "data" / "gaussian_data" / args.scene

    if args.gaussian_model:
        model_path = args.gaussian_model.expanduser().resolve()
    else:
        candidates = sorted((root / "gaussian_output" / args.scene).glob("init=*"))
        if not candidates:
            raise FileNotFoundError(
                f"No Gaussian model found under {root / 'gaussian_output' / args.scene}"
            )
        model_path = candidates[0]

    default_inference = root / "experiments" / args.scene / "inference.pkl"
    if args.run:
        runs = dict(args.run)
    else:
        runs = {
            "Original Full Train": default_inference,
            "Stage1-50 Only": online_root / "runs" / "stage1_full_rollout" / "inference.pkl",
            f"NodeCluster keep {args.node_keep}%": (
                online_root
                / "runs"
                / f"node_cluster_keep_{args.node_keep}_reconstructed"
                / "inference.pkl"
            ),
            "Online Stiffness": online_root / "runs" / "online_stiffness" / "inference.pkl",
            "Online BT-guided": online_root / "runs" / "online_bt_guided" / "inference.pkl",
        }

    env = os.environ.copy()
    env.update(
        {
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "WANDB_SILENT": "true",
        }
    )
    add_torch_library_path(env)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for method, inference_path in runs.items():
        inference_path = Path(inference_path)
        if not inference_path.is_file():
            print("[SKIP] Missing:", inference_path)
            continue
        output_dir = output_root / slug(method)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "gs_render_dynamics.py",
            "--source_path",
            str(source_path),
            "--model_path",
            str(model_path),
            "--name",
            args.scene,
            "--output_dir",
            str(output_dir),
        ]
        print("=" * 100)
        print("Rendering:", method)
        start = time.perf_counter()
        with temporarily_replace_inference(default_inference, inference_path):
            result = subprocess.run(command, cwd=root, env=env)
        elapsed = time.perf_counter() - start
        png_count = len(list(output_dir.rglob("*.png")))
        fps = png_count / elapsed if elapsed > 0 else float("nan")
        rows.append(
            {
                "Method": method,
                "Inference": str(inference_path),
                "Render Output": str(output_dir),
                "Rendered PNGs": png_count,
                "Rendering Seconds": elapsed,
                "Rendering FPS": fps,
                "Return Code": result.returncode,
            }
        )
        if result.returncode != 0:
            print(f"[WARNING] Rendering failed for {method}")

    csv_path = output_root / "rendering_fps.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("[DONE]", csv_path)


if __name__ == "__main__":
    main()
