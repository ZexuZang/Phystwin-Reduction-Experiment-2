#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Use NAME=/path/to/inference.pkl"
        )
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one-directional L1 CD and 3D tracking error."
    )
    parser.add_argument("--phystwin-root", required=True, type=Path)
    parser.add_argument("--scene", default="double_stretch_sloth")
    parser.add_argument("--stage1-ratio", type=float, default=0.5)
    parser.add_argument("--node-keep", default="80")
    parser.add_argument(
        "--run",
        action="append",
        type=parse_mapping,
        help="Custom mapping: NAME=/path/to/inference.pkl",
    )
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def chamfer_single_direction_l1(
    predicted_points: np.ndarray,
    visible_gt_points: np.ndarray,
) -> float:
    tree = cKDTree(predicted_points)
    distances, _ = tree.query(visible_gt_points, k=1, p=1)
    return float(np.mean(distances))


def main() -> None:
    args = parse_args()
    root = args.phystwin_root.expanduser().resolve()
    scene_root = root / "data" / "different_types" / args.scene

    if args.run:
        runs = dict(args.run)
    else:
        online_root = root / "results" / "online_adaptation" / args.scene
        runs = {
            "Original Full Train": (
                root / "experiments" / args.scene / "inference.pkl"
            ),
            "Stage1-50 Only": (
                online_root / "runs" / "stage1_full_rollout" / "inference.pkl"
            ),
            f"NodeCluster keep {args.node_keep}%": (
                online_root
                / "runs"
                / f"node_cluster_keep_{args.node_keep}_reconstructed"
                / "inference.pkl"
            ),
            "Online Stiffness": (
                online_root / "runs" / "online_stiffness" / "inference.pkl"
            ),
            "Online BT-guided": (
                online_root / "runs" / "online_bt_guided" / "inference.pkl"
            ),
        }

    with (scene_root / "final_data.pkl").open("rb") as handle:
        data = pickle.load(handle)
    with (scene_root / "gt_track_3d.pkl").open("rb") as handle:
        gt_track_3d = np.asarray(pickle.load(handle))
    with (scene_root / "split.json").open(
        "r", encoding="utf-8"
    ) as handle:
        split = json.load(handle)

    object_points = np.asarray(data["object_points"])
    object_visibilities = np.asarray(data["object_visibilities"])
    num_surface_points = (
        object_points.shape[1]
        + np.asarray(data["surface_points"]).shape[0]
    )
    train_end = int(split["train"][1])
    test_start = int(split["test"][0])
    test_end = int(split["test"][1])

    def evaluate_cd(
        predicted: np.ndarray, start: int, end: int
    ) -> dict[str, float | int]:
        values: list[float] = []
        max_frame = min(end, len(predicted), len(object_points))
        for frame_idx in range(start, max_frame):
            gt_visible = object_points[frame_idx][
                object_visibilities[frame_idx]
            ]
            pred_surface = predicted[frame_idx][:num_surface_points]
            if len(gt_visible) and len(pred_surface):
                values.append(
                    chamfer_single_direction_l1(
                        pred_surface, gt_visible
                    )
                )
        return {
            "frame_num": len(values),
            "cd": (
                float(np.mean(values))
                if values
                else float("nan")
            ),
        }

    def evaluate_track(
        predicted: np.ndarray, start: int, end: int
    ) -> float:
        initial_mask = ~np.isnan(gt_track_3d[0]).any(axis=1)
        tree = cKDTree(predicted[0])
        _, predicted_indices = tree.query(
            gt_track_3d[0][initial_mask], k=1
        )
        errors: list[float] = []
        max_frame = min(end, len(predicted), len(gt_track_3d))
        for frame_idx in range(start, max_frame):
            visible = ~np.isnan(
                gt_track_3d[frame_idx][initial_mask]
            ).any(axis=1)
            gt_points = gt_track_3d[frame_idx][initial_mask][
                visible
            ]
            pred_points = predicted[frame_idx][
                predicted_indices
            ][visible]
            errors.append(
                float(
                    np.mean(
                        np.linalg.norm(
                            pred_points - gt_points, axis=1
                        )
                    )
                )
                if len(pred_points)
                else 0.0
            )
        return (
            float(np.mean(errors))
            if errors
            else float("nan")
        )

    rows: list[dict[str, float | int | str]] = []
    for method, inference_path in runs.items():
        inference_path = Path(inference_path)
        if not inference_path.is_file():
            print("[SKIP] Missing:", inference_path)
            continue
        with inference_path.open("rb") as handle:
            predicted = np.asarray(pickle.load(handle))
        cd_train = evaluate_cd(predicted, 1, train_end)
        cd_test = evaluate_cd(
            predicted, test_start, test_end
        )
        rows.append(
            {
                "Method": method,
                "Train Frame Num": cd_train["frame_num"],
                "CD Train": cd_train["cd"],
                "Test Frame Num": cd_test["frame_num"],
                "CD Test": cd_test["cd"],
                "Track Error Train": evaluate_track(
                    predicted, 1, train_end
                ),
                "Track Error Test": evaluate_track(
                    predicted, test_start, test_end
                ),
            }
        )

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    output = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else root / "results" / "online_adaptation" / args.scene / "cd_track_metrics.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print("Saved:", output)


if __name__ == "__main__":
    main()
