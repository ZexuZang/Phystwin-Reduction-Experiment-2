#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate dynamic Gaussian renders over all configured views.

Unlike the legacy evaluator, this script keeps view identity instead of merging
same-numbered frame names from different view folders.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from phystwin_reduction.render_refinement import (
    detect_render_view_root,
    find_human_mask,
    infer_object_mask_id,
    load_split,
)


def parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/render_root")
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser().resolve()


def load_rgb(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image[..., :3].astype(np.uint8)


def load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(np.float32)
    if mask.size and mask.max() > 1.0:
        mask /= 255.0
    return mask


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(path))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    rgb = image[..., :3].astype(np.uint8)
    if image.ndim == 3 and image.shape[-1] >= 4:
        alpha = image[..., 3].astype(np.float32)
        if alpha.size and alpha.max() > 1.0:
            alpha /= 255.0
    else:
        alpha = (rgb.astype(np.float32).sum(-1) > 5.0).astype(np.float32)
    return rgb, alpha


def resize_prediction(
    rgb: np.ndarray,
    alpha: np.ndarray,
    target_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = target_hw
    if rgb.shape[:2] == (h, w):
        return rgb, alpha
    rgb = np.asarray(
        Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR)
    )
    alpha = (
        np.asarray(
            Image.fromarray((alpha * 255.0).astype(np.uint8)).resize(
                (w, h),
                Image.Resampling.NEAREST,
            )
        ).astype(np.float32)
        / 255.0
    )
    return rgb, alpha


def lpips_value(model, device, gt: np.ndarray, pred: np.ndarray) -> float:
    if model is None:
        return float("nan")
    gt_t = (
        torch.from_numpy(gt.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
        * 2.0
        - 1.0
    )
    pred_t = (
        torch.from_numpy(pred.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
        * 2.0
        - 1.0
    )
    with torch.no_grad():
        return float(model(gt_t, pred_t).item())


def iou_value(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0.5
    b = b > 0.5
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def find_pred_path(view_root: Path, frame: int) -> Path | None:
    candidates = [
        view_root / f"{frame:05d}.png",
        view_root / f"{frame}.png",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phystwin-root", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--gt-root", type=Path)
    parser.add_argument("--human-root", type=Path)
    parser.add_argument(
        "--render",
        action="append",
        type=parse_mapping,
        required=True,
        help="NAME=/path/to/output_dir passed to gs_render_dynamics.py",
    )
    parser.add_argument(
        "--view-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Rendered view folder ids and GT view ids.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    root = args.phystwin_root.expanduser().resolve()
    gt_root = (
        args.gt_root.expanduser().resolve()
        if args.gt_root is not None
        else root / "data" / "different_types" / args.scene
    )
    human_root = (
        args.human_root.expanduser().resolve()
        if args.human_root is not None
        else root / "data" / "different_types_human_mask" / args.scene
    )
    if not human_root.is_dir():
        human_root = None
    if not gt_root.is_dir():
        raise FileNotFoundError(gt_root)

    split = load_split(gt_root)
    train_start, train_end = map(int, split["train"])
    test_start, test_end = map(int, split["test"])
    groups = {
        "Train": list(range(train_start + 1, train_end)),
        "Test": list(range(test_start, test_end)),
    }

    object_ids = {
        int(view): infer_object_mask_id(gt_root, int(view))
        for view in args.view_ids
    }

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

    rows: list[dict] = []
    for method, render_root in args.render:
        render_root = detect_render_view_root(render_root, args.scene)
        row: dict[str, object] = {
            "Method": method,
            "Render Root": str(render_root),
        }

        for split_name, frames in groups.items():
            psnr_values: list[float] = []
            ssim_values: list[float] = []
            lpips_values: list[float] = []
            iou_values: list[float] = []

            for view in args.view_ids:
                pred_view_root = render_root / str(view)
                object_id = object_ids[int(view)]

                for frame in frames:
                    gt_rgb_path = (
                        gt_root / "color" / str(view) / f"{frame}.png"
                    )
                    gt_mask_path = (
                        gt_root
                        / "mask"
                        / str(view)
                        / str(object_id)
                        / f"{frame}.png"
                    )
                    pred_path = find_pred_path(pred_view_root, frame)
                    if (
                        pred_path is None
                        or not gt_rgb_path.is_file()
                        or not gt_mask_path.is_file()
                    ):
                        continue

                    gt_rgb = load_rgb(gt_rgb_path)
                    gt_mask = load_mask(gt_mask_path)
                    pred_rgb, pred_alpha = load_prediction(pred_path)
                    pred_rgb, pred_alpha = resize_prediction(
                        pred_rgb,
                        pred_alpha,
                        gt_rgb.shape[:2],
                    )

                    human_path = find_human_mask(human_root, int(view), frame)
                    if human_path is not None:
                        human = load_mask(human_path)
                        if human.shape != gt_mask.shape:
                            human = (
                                np.asarray(
                                    Image.fromarray(
                                        (human * 255.0).astype(np.uint8)
                                    ).resize(
                                        (gt_mask.shape[1], gt_mask.shape[0]),
                                        Image.Resampling.NEAREST,
                                    )
                                ).astype(np.float32)
                                / 255.0
                            )
                        keep = 1.0 - np.clip(human, 0.0, 1.0)
                    else:
                        keep = np.ones_like(gt_mask)

                    target_mask = np.clip(gt_mask, 0.0, 1.0) * keep
                    target_rgb = (
                        gt_rgb.astype(np.float32)
                        * target_mask[..., None]
                    ).round().clip(0, 255).astype(np.uint8)
                    pred_eval = (
                        pred_rgb.astype(np.float32)
                        * keep[..., None]
                    ).round().clip(0, 255).astype(np.uint8)

                    psnr_values.append(
                        float(
                            peak_signal_noise_ratio(
                                target_rgb,
                                pred_eval,
                                data_range=255,
                            )
                        )
                    )
                    ssim_values.append(
                        float(
                            structural_similarity(
                                target_rgb,
                                pred_eval,
                                channel_axis=2,
                                data_range=255,
                            )
                        )
                    )
                    lpips_values.append(
                        lpips_value(
                            lpips_model,
                            device,
                            target_rgb,
                            pred_eval,
                        )
                    )
                    iou_values.append(
                        iou_value(
                            target_mask,
                            pred_alpha * keep,
                        )
                    )

            row[f"{split_name} Images"] = len(psnr_values)
            row[f"PSNR {split_name}"] = (
                float(np.mean(psnr_values)) if psnr_values else float("nan")
            )
            row[f"SSIM {split_name}"] = (
                float(np.mean(ssim_values)) if ssim_values else float("nan")
            )
            row[f"LPIPS {split_name}"] = (
                float(np.nanmean(lpips_values))
                if lpips_values
                else float("nan")
            )
            row[f"IoU {split_name}"] = (
                float(np.mean(iou_values)) if iou_values else float("nan")
            )

        rows.append(row)

    output = args.output_csv.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("=" * 120)
    for row in rows:
        print(json.dumps(row, indent=2))
    print("[DONE]", output)


if __name__ == "__main__":
    main()
