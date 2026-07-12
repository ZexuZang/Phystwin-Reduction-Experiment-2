#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SCRIPT = REPO_ROOT / "scripts" / "run_external_topology_inference.py"


def parse_job(value: str) -> tuple[str, Path, Path]:
    if "=" not in value or "|" not in value:
        raise argparse.ArgumentTypeError(
            "Use NAME=/path/to/model.pth|/path/to/topology.npz"
        )
    name, payload = value.split("=", 1)
    model, topology = payload.split("|", 1)
    return (
        name.strip(),
        Path(model).expanduser().resolve(),
        Path(topology).expanduser().resolve(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Strategy-2 simulation FPS by rerunning inference."
    )
    parser.add_argument("--phystwin-root", required=True, type=Path)
    parser.add_argument("--scene", default="double_stretch_sloth")
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--train-frame", type=int, required=True)
    parser.add_argument(
        "--job",
        action="append",
        type=parse_job,
        required=True,
        help="NAME=/path/to/model.pth|/path/to/topology.npz",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def main() -> None:
    args = parse_args()
    root = args.phystwin_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else root / "results" / "online_adaptation" / args.scene / "fps_measure"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, model, topology in args.job:
        run_dir = output_root / slug(name)
        command = [
            sys.executable,
            str(INFERENCE_SCRIPT),
            "--phystwin-root",
            str(root),
            "--scene",
            args.scene,
            "--train-frame",
            str(args.train_frame),
            "--model-path",
            str(model),
            "--topology-path",
            str(topology),
            "--output-dir",
            str(run_dir),
            "--seed",
            str(args.seed),
        ]
        if args.base_path:
            command += ["--base-path", str(args.base_path.expanduser().resolve())]

        print("=" * 100)
        print("$", " ".join(command))
        start = time.perf_counter()
        result = subprocess.run(command, cwd=REPO_ROOT)
        elapsed = time.perf_counter() - start
        inference_path = run_dir / "inference.pkl"
        frames = 0
        if inference_path.is_file():
            with inference_path.open("rb") as handle:
                frames = int(np.asarray(pickle.load(handle)).shape[0])
        topology_data = np.load(topology, allow_pickle=True)
        rows.append(
            {
                "Method": name,
                "Model": str(model),
                "Topology": str(topology),
                "Object Springs": int(topology_data["num_object_springs"]),
                "Total Springs": int(topology_data["springs"].shape[0]),
                "Frames": frames,
                "Simulation Seconds": elapsed,
                "Simulation FPS": frames / elapsed if elapsed > 0 else float("nan"),
                "Return Code": result.returncode,
            }
        )

    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else output_root / "simulation_fps.csv"
    )
    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("[DONE]", output_csv)


if __name__ == "__main__":
    main()
