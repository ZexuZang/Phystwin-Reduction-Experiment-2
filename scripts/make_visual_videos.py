#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image


def parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/render_directory")
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create GT / prediction / absolute-error videos."
    )
    parser.add_argument("--phystwin-root", required=True, type=Path)
    parser.add_argument("--scene", default="double_stretch_sloth")
    parser.add_argument("--stage1-ratio", type=float, default=0.5)
    parser.add_argument("--node-keep", default="80")
    parser.add_argument(
        "--render",
        action="append",
        type=parse_mapping,
        help="Custom render mapping: NAME=/path/to/render_directory",
    )
    parser.add_argument(
        "--frame-mode",
        choices=["stage1", "online", "test", "all"],
        default="test",
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def numeric_stem(path: Path) -> int | None:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else None


def collect_images(root: Path) -> list[Path]:
    return sorted([*root.rglob("*.png"), *root.rglob("*.jpg"), *root.rglob("*.jpeg")])


def build_map(files: list[Path]) -> dict[int, Path]:
    result = {}
    for path in files:
        idx = numeric_stem(path)
        if idx is not None and idx not in result:
            result[idx] = path
    return result


def load_rgb(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image[:, :, :3].astype(np.uint8)


def resize_like(image: np.ndarray, target: np.ndarray) -> np.ndarray:
    if image.shape[:2] == target.shape[:2]:
        return image
    return np.asarray(
        Image.fromarray(image).resize(
            (target.shape[1], target.shape[0]), Image.Resampling.BILINEAR
        )
    )


def title(image: np.ndarray, text: str, height: int = 42) -> np.ndarray:
    canvas = np.full((image.shape[0] + height, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[height:] = image
    cv2.putText(
        canvas,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


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
    scene_root = root / "data" / "different_types" / args.scene
    online_root = root / "results" / "online_adaptation" / args.scene
    with (scene_root / "split.json").open("r", encoding="utf-8") as handle:
        split = json.load(handle)

    train_start, train_end = map(int, split["train"])
    test_start, test_end = map(int, split["test"])
    stage1_end = train_start + int((train_end - train_start) * args.stage1_ratio)
    groups = {
        "stage1": list(range(train_start + 1, stage1_end)),
        "online": list(range(stage1_end, train_end)),
        "test": list(range(test_start, test_end)),
        "all": list(range(train_start + 1, test_end)),
    }
    frame_indices = groups[args.frame_mode]

    if args.render:
        render_roots = dict(args.render)
    else:
        render_root = online_root / "renders"
        render_roots = {
            "Original Full Train": render_root / "original_full_train",
            "Stage1-50 Only": render_root / "stage1_50_only",
            f"NodeCluster keep {args.node_keep}%": render_root / f"nodecluster_keep_{args.node_keep}",
            "Online Stiffness": render_root / "online_stiffness",
            "Online BT-guided": render_root / "online_bt_guided",
        }

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else online_root / "visual_videos"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_files = collect_images(scene_root / "color")
    gt_map = build_map(gt_files)

    for method, pred_root in render_roots.items():
        pred_files = collect_images(Path(pred_root))
        pred_map = build_map(pred_files)
        usable = [
            idx for idx in frame_indices if idx in gt_map and idx in pred_map
        ]
        if not usable:
            print("[SKIP] No matched frames:", method)
            continue

        first_gt = load_rgb(gt_map[usable[0]])
        first_pred = resize_like(load_rgb(pred_map[usable[0]]), first_gt)
        first_err = cv2.applyColorMap(
            np.mean(
                np.abs(first_gt.astype(np.float32) - first_pred.astype(np.float32)),
                axis=2,
            ).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        first_err = cv2.cvtColor(first_err, cv2.COLOR_BGR2RGB)
        sample = np.concatenate(
            [title(first_gt, "GT"), title(first_pred, "Prediction"), title(first_err, "Abs Error")],
            axis=1,
        )

        output_path = output_dir / f"{slug(method)}_{args.frame_mode}.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (sample.shape[1], sample.shape[0]),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {output_path}")

        for frame_idx in usable:
            gt = load_rgb(gt_map[frame_idx])
            pred = resize_like(load_rgb(pred_map[frame_idx]), gt)
            error = np.mean(
                np.abs(gt.astype(np.float32) - pred.astype(np.float32)), axis=2
            )
            error = cv2.applyColorMap(
                np.clip(error, 0, 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            error = cv2.cvtColor(error, cv2.COLOR_BGR2RGB)
            panel = np.concatenate(
                [
                    title(gt, f"GT | frame {frame_idx}"),
                    title(pred, f"{method} | Prediction"),
                    title(error, "Abs Error"),
                ],
                axis=1,
            )
            writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        writer.release()
        print("[DONE]", output_path)


if __name__ == "__main__":
    main()
