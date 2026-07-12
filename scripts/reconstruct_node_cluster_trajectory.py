#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phystwin_reduction.online_adaptation import reconstruct_full_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct original-node trajectory from clustered-node inference."
    )
    parser.add_argument("--reduced-inference-path", required=True, type=Path)
    parser.add_argument("--topology-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path, metadata = reconstruct_full_trajectory(
        args.reduced_inference_path,
        args.topology_path,
        args.output_path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print("[DONE]", path)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
