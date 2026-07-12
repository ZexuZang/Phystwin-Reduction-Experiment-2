#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch


def parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/render_directory")
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Strategy-2 renders with PSNR, SSIM, LPIPS, and IoU."
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
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def numeric_stem(path: Path) -> int | None:
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1]) if numbers else None


def collect_images(root: Path) -> list[Path]:
    return sorted([*root.rglob("*.png"), *root.rglob("*.jpg"), *root.rglob("*.jpeg")])


def build_frame_map(files: list[Path]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in files:
        idx = numeric_stem(path)
        if idx is not None and idx not in result:
            result[idx] = path
    return result


def frame_path(
    frame_idx: int,
    files: list[Path],
    mapping: dict[int, Path],
    indices: list[int],
) -> Path | None:
    if frame_idx in mapping:
        return mapping[frame_idx]
    try:
        position = indices.index(frame_idx)
    except ValueError:
        return None
    return files[position] if position < len(files) else None


def load_rgb(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image[:, :, :3].astype(np.uint8)


def load_gt_mask(path: Path | None, rgb: np.ndarray) -> np.ndarray:
    if path is None:
        return (rgb.astype(np.float32).sum(axis=-1) > 5).astype(np.float32)
    mask = np.asarray(Image.open(path)).astype(np.float32)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.size and mask.max() > 1:
        mask /= 255.0
    return mask


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(path))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    rgb = image[:, :, :3].astype(np.uint8)
    if image.shape[-1] == 4:
        mask = image[:, :, 3].astype(np.float32)
        if mask.size and mask.max() > 1:
            mask /= 255.0
    else:
        mask = (rgb.astype(np.float32).sum(axis=-1) > 5).astype(np.float32)
    return rgb, mask


def resize_prediction(
    rgb: np.ndarray, mask: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if rgb.shape[:2] == target.shape[:2]:
        return rgb, mask
    size = (target.shape[1], target.shape[0])
    rgb_resized = np.asarray(
        Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)
    )
    mask_resized = (
        np.asarray(
            Image.fromarray((mask * 255).astype(np.uint8)).resize(
                size, Image.Resampling.NEAREST
            )
        ).astype(np.float32)
        / 255.0
    )
    return rgb_resized, mask_resized


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool, b_bool = a > 0.5, b > 0.5
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a_bool, b_bool).sum() / union)


def main() -> None:
    args = parse_args()
    root = args.phystwin_root.expanduser().resolve()
    online_root = root / "results" / "online_adaptation" / args.scene
    scene_root = root / "data" / "different_types" / args.scene
    with (scene_root / "split.json").open("r", encoding="utf-8") as handle:
        split = json.load(handle)

    train_start, train_end = map(int, split["train"])
    test_start, test_end = map(int, split["test"])
    stage1_end = train_start + int((train_end - train_start) * args.stage1_ratio)
    frame_groups = {
        "Stage1": list(range(train_start + 1, stage1_end)),
        "Online": list(range(stage1_end, train_end)),
        "Test": list(range(test_start, test_end)),
    }

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
        # Accept the slugs produced by render_methods.py.
        aliases = {
            "Original Full Train": render_root / "original_full_train",
            "Stage1-50 Only": render_root / "stage1_50_only",
            f"NodeCluster keep {args.node_keep}%": render_root / f"nodecluster_keep_{args.node_keep}",
            "Online Stiffness": render_root / "online_stiffness",
            "Online BT-guided": render_root / "online_bt_guided",
        }
        render_roots.update(aliases)

    gt_color_files = collect_images(scene_root / "color")
    gt_mask_files = collect_images(scene_root / "mask")
    gt_color_map = build_frame_map(gt_color_files)
    gt_mask_map = build_frame_map(gt_mask_files)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:
        print("[WARNING] LPIPS unavailable:", exc)
        lpips_model = None

    def lpips_value(gt: np.ndarray, pred: np.ndarray) -> float:
        if lpips_model is None:
            return float("nan")
        gt_tensor = (
            torch.from_numpy(gt.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
            * 2
            - 1
        )
        pred_tensor = (
            torch.from_numpy(pred.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
            * 2
            - 1
        )
        with torch.no_grad():
            return float(lpips_model(gt_tensor, pred_tensor).item())

    rows: list[dict[str, object]] = []
    for method, pred_root in render_roots.items():
        pred_root = Path(pred_root)
        pred_files = collect_images(pred_root)
        if not pred_files:
            print("[SKIP] No images:", pred_root)
            continue
        pred_map = build_frame_map(pred_files)
        row: dict[str, object] = {"Method": method, "Render Root": str(pred_root)}

        for split_name, indices in frame_groups.items():
            psnr_values: list[float] = []
            ssim_values: list[float] = []
            lpips_values: list[float] = []
            iou_values: list[float] = []
            for frame_idx in indices:
                gt_path = frame_path(frame_idx, gt_color_files, gt_color_map, indices)
                pred_path = frame_path(frame_idx, pred_files, pred_map, indices)
                mask_path = frame_path(frame_idx, gt_mask_files, gt_mask_map, indices)
                if gt_path is None or pred_path is None:
                    continue
                gt = load_rgb(gt_path)
                pred, pred_mask = load_prediction(pred_path)
                pred, pred_mask = resize_prediction(pred, pred_mask, gt)
                gt_mask = load_gt_mask(mask_path, gt)
                psnr_values.append(
                    float(peak_signal_noise_ratio(gt, pred, data_range=255))
                )
                ssim_values.append(
                    float(structural_similarity(gt, pred, channel_axis=2, data_range=255))
                )
                lpips_values.append(lpips_value(gt, pred))
                iou_values.append(compute_iou(gt_mask, pred_mask))

            row[f"{split_name} Frames"] = len(psnr_values)
            row[f"{split_name} PSNR"] = (
                float(np.mean(psnr_values)) if psnr_values else float("nan")
            )
            row[f"{split_name} SSIM"] = (
                float(np.mean(ssim_values)) if ssim_values else float("nan")
            )
            row[f"{split_name} LPIPS"] = (
                float(np.nanmean(lpips_values)) if lpips_values else float("nan")
            )
            row[f"{split_name} IoU"] = (
                float(np.mean(iou_values)) if iou_values else float("nan")
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    output = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else online_root / "render_metrics.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print("[DONE]", output)


if __name__ == "__main__":
    main()
