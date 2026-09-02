#!/usr/bin/env python3
from __future__ import annotations

"""Fine-tune Gaussian RGB / opacity / scale for a fixed PhysTwin trajectory.

The physical trajectory and spring topology are never changed by this script.
Only Gaussian rendering parameters are optimized.  Train/test boundaries are
read from split.json and test frames are never used for optimization.
"""

import argparse
import copy
import csv
import json
import math
import pickle
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from phystwin_reduction.render_refinement import (
    RENDER_VARIANTS,
    find_human_mask,
    infer_object_mask_id,
    load_split,
    uniform_train_frames,
)


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = np.asarray(Image.open(path))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    image = image[..., :3].astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).to(device)


def load_mask(path: Path, device: torch.device) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(np.float32)
    if mask.size and mask.max() > 1.0:
        mask /= 255.0
    return torch.from_numpy(mask).to(device)


def resize_chw(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(
        x[None],
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0]


def resize_hw(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(
        x[None, None],
        size=size,
        mode="nearest",
    )[0, 0]


@torch.no_grad()
def build_pose_cache(
    *,
    gaussians,
    ctrl_pts_np: np.ndarray,
    selected_frames: list[int],
    cache_path: Path,
    device: torch.device,
) -> dict:
    """Cache only dynamic Gaussian xyz/quaternion for requested train frames.

    The propagation follows PhysTwin's gs_render_dynamics.py:
    fixed initial KNN relations, sequential control-point motions, and smooth
    interpolation between motion change-points.  Appearance parameters are not
    cached because they stay learnable during fine-tuning.
    """

    from gaussian_splatting.dynamic_utils import (
        get_topk_indices,
        interpolate_motions,
        knn_weights,
    )

    selected_frames = sorted(set(map(int, selected_frames)))
    ctrl_pts_np = np.asarray(ctrl_pts_np, dtype=np.float32)
    if ctrl_pts_np.ndim != 3 or ctrl_pts_np.shape[-1] != 3:
        raise ValueError(
            f"Expected inference trajectory [T,N,3], got {ctrl_pts_np.shape}"
        )
    n_frames = int(ctrl_pts_np.shape[0])
    invalid = [f for f in selected_frames if not 0 <= f < n_frames]
    if invalid:
        raise ValueError(f"Train frames outside trajectory: {invalid[:10]}")

    ctrl_pts = torch.as_tensor(ctrl_pts_np, dtype=torch.float32, device=device)
    motion = np.linalg.norm(ctrl_pts_np[1:] - ctrl_pts_np[:-1], axis=-1).sum(-1)

    change_points = [0]
    change_points.extend(
        i for i, value in enumerate(motion, start=1) if value > 1e-10
    )
    change_points = sorted(set(change_points))

    needed_cp = {0}
    for frame in selected_frames:
        prev_cp = max(cp for cp in change_points if cp <= frame)
        needed_cp.add(prev_cp)
        next_cp = [cp for cp in change_points if cp > frame]
        if next_cp:
            needed_cp.add(min(next_cp))

    print("=" * 80)
    print("DYNAMIC GAUSSIAN POSE CACHE")
    print("trajectory frames :", n_frames)
    print("selected frames   :", selected_frames)
    print("change points     :", len(change_points))
    print("cached states     :", len(needed_cp))
    print("=" * 80)

    all_pos = gaussians.get_xyz.detach().clone()
    all_rot = gaussians.get_rotation.detach().clone()
    relations = get_topk_indices(ctrl_pts[0], K=16)

    cp_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    if 0 in needed_cp:
        cp_states[0] = (
            all_pos.detach().cpu().half(),
            all_rot.detach().cpu().half(),
        )

    chunk_size = 20_000
    for frame_idx in range(1, n_frames):
        if motion[frame_idx - 1] <= 1e-10:
            continue

        prev_particle_pos = ctrl_pts[frame_idx - 1]
        cur_particle_pos = ctrl_pts[frame_idx]

        for start in range(0, len(all_pos), chunk_size):
            end = min(start + chunk_size, len(all_pos))
            pos = all_pos[start:end]
            quat = all_rot[start:end]
            weights = knn_weights(prev_particle_pos, pos, K=16)
            pos, quat, _ = interpolate_motions(
                bones=prev_particle_pos,
                motions=cur_particle_pos - prev_particle_pos,
                relations=relations,
                weights=weights,
                xyz=pos,
                quat=quat,
            )
            all_pos[start:end] = pos
            all_rot[start:end] = quat

        if frame_idx in needed_cp:
            cp_states[frame_idx] = (
                all_pos.detach().cpu().half(),
                all_rot.detach().cpu().half(),
            )
            print(f"cached motion state {frame_idx}/{n_frames - 1}")

    poses: dict[int, dict[str, torch.Tensor]] = {}
    for frame in selected_frames:
        prev_cp = max(cp for cp in change_points if cp <= frame)
        next_candidates = [cp for cp in change_points if cp > frame]
        p0, q0 = cp_states[prev_cp]

        if frame == prev_cp or not next_candidates:
            xyz = p0
            quat = q0
        else:
            next_cp = min(next_candidates)
            p1, q1 = cp_states[next_cp]
            alpha = float(frame - prev_cp) / float(next_cp - prev_cp)
            xyz = (
                p0.float() * (1.0 - alpha)
                + p1.float() * alpha
            ).half()
            quat = F.normalize(
                q0.float() * (1.0 - alpha)
                + q1.float() * alpha,
                dim=-1,
            ).half()

        poses[int(frame)] = {
            "xyz": xyz,
            "quat": quat,
        }

    payload = {
        "frames": selected_frames,
        "gaussian_count": int(gaussians.get_xyz.shape[0]),
        "trajectory_frames": n_frames,
        "poses": poses,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print("[DONE] pose cache:", cache_path)
    return payload


def load_or_build_pose_cache(
    *,
    gaussians,
    inference_path: Path,
    selected_frames: list[int],
    cache_path: Path,
    device: torch.device,
) -> dict:
    if cache_path.is_file():
        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )
        cached = set(map(int, payload.get("frames", [])))
        if (
            set(selected_frames).issubset(cached)
            and int(payload.get("gaussian_count", -1))
            == int(gaussians.get_xyz.shape[0])
        ):
            print("Using pose cache:", cache_path)
            return payload
        print("Pose cache is incompatible; rebuilding:", cache_path)

    with inference_path.open("rb") as handle:
        trajectory = np.asarray(pickle.load(handle))
    return build_pose_cache(
        gaussians=gaussians,
        ctrl_pts_np=trajectory,
        selected_frames=selected_frames,
        cache_path=cache_path,
        device=device,
    )


def save_complete_gaussian_model(
    *,
    source_model: Path,
    output_model: Path,
    gaussians,
    loaded_iteration: int,
) -> Path:
    if output_model.exists():
        shutil.rmtree(output_model)
    shutil.copytree(source_model, output_model)

    ply_out = (
        output_model
        / "point_cloud"
        / f"iteration_{loaded_iteration}"
        / "point_cloud.ply"
    )
    ply_out.parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(ply_out))
    return ply_out


def main() -> None:
    # Import PhysTwin only after the repository root has been added to PYTHONPATH
    # by run_render_refinement_pipeline.py or by the caller.
    from argparse import ArgumentParser
    from gaussian_splatting.arguments import (
        ModelParams,
        PipelineParams,
        get_combined_args,
    )
    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.scene import Scene
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.utils.loss_utils import ssim

    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--inference", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--human-root", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--pose-cache", required=True, type=Path)
    parser.add_argument(
        "--variant",
        required=True,
        choices=sorted(RENDER_VARIANTS),
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-train-frames", type=int, default=24)
    parser.add_argument(
        "--camera-indices",
        type=int,
        nargs="+",
        default=[0, 50, 100],
        help="Indices into Scene.getTestCameras(), matching gs_render_dynamics.py.",
    )
    parser.add_argument(
        "--gt-view-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="GT view folder ids aligned with --camera-indices.",
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

    args = get_combined_args(parser)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.num_train_frames <= 0:
        raise ValueError("--num-train-frames must be positive")
    if len(args.camera_indices) != len(args.gt_view_ids):
        raise ValueError("--camera-indices and --gt-view-ids must have equal length")
    if not args.camera_indices:
        raise ValueError("At least one camera is required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda")
    inference_path = args.inference.expanduser().resolve()
    gt_root = args.gt_root.expanduser().resolve()
    human_root = (
        args.human_root.expanduser().resolve()
        if args.human_root is not None
        else None
    )
    out_dir = args.out_dir.expanduser().resolve() / args.variant
    pose_cache_path = args.pose_cache.expanduser().resolve()
    source_model_dir = Path(args.model_path).expanduser().resolve()

    for path in (inference_path,):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not gt_root.is_dir():
        raise FileNotFoundError(gt_root)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
    )
    loaded_iteration = int(
        getattr(scene, "loaded_iter", args.iteration if args.iteration >= 0 else 10000)
    )

    split = load_split(gt_root)
    train_frames = uniform_train_frames(
        split,
        args.num_train_frames,
        skip_rest_frame=True,
    )

    all_cameras = scene.getTestCameras()
    if max(args.camera_indices) >= len(all_cameras):
        raise IndexError(
            f"Requested camera index {max(args.camera_indices)} but only "
            f"{len(all_cameras)} test cameras are available"
        )
    cameras = [all_cameras[i] for i in args.camera_indices]
    object_ids = {
        int(view): infer_object_mask_id(gt_root, int(view))
        for view in args.gt_view_ids
    }

    pose_cache = load_or_build_pose_cache(
        gaussians=gaussians,
        inference_path=inference_path,
        selected_frames=train_frames,
        cache_path=pose_cache_path,
        device=device,
    )

    # Freeze everything first.  The dynamic xyz and rotation are supplied from
    # the fixed physical trajectory and are never optimized.
    parameter_tensors = {
        "xyz": gaussians._xyz,
        "features_dc": gaussians._features_dc,
        "features_rest": gaussians._features_rest,
        "opacity": gaussians._opacity,
        "scaling": gaussians._scaling,
        "rotation": gaussians._rotation,
    }
    for tensor in parameter_tensors.values():
        tensor.requires_grad_(False)

    learnable = RENDER_VARIANTS[args.variant]
    optimizer_groups = []

    gaussians._features_dc.requires_grad_("features_dc" in learnable)
    if gaussians._features_dc.requires_grad:
        optimizer_groups.append(
            {"params": [gaussians._features_dc], "lr": args.lr_rgb}
        )

    gaussians._opacity.requires_grad_("opacity" in learnable)
    if gaussians._opacity.requires_grad:
        optimizer_groups.append(
            {"params": [gaussians._opacity], "lr": args.lr_opacity}
        )

    gaussians._scaling.requires_grad_("scaling" in learnable)
    if gaussians._scaling.requires_grad:
        optimizer_groups.append(
            {"params": [gaussians._scaling], "lr": args.lr_scale}
        )

    if not optimizer_groups:
        raise RuntimeError(f"No learnable parameters for {args.variant}")

    optimizer = torch.optim.Adam(optimizer_groups, eps=1e-8)

    dc0 = gaussians._features_dc.detach().clone()
    opa0 = gaussians._opacity.detach().clone()
    scale0 = gaussians._scaling.detach().clone()

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device=device,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    use_gsplat = args.backend == "gsplat"

    print("=" * 80)
    print("GAUSSIAN RENDER REFINEMENT")
    print("variant        :", args.variant)
    print("learnable      :", learnable)
    print("gaussians      :", int(gaussians.get_xyz.shape[0]))
    print("train frames   :", train_frames)
    print("camera indices :", args.camera_indices)
    print("gt view ids    :", args.gt_view_ids)
    print("inference      :", inference_path)
    print("backend        :", args.backend)
    print("=" * 80)

    # Seeded random sampling is retained because it was the protocol used in the
    # successful exploratory experiment.  The seed makes the sequence reusable.
    rng = random.Random(args.seed)

    for step in range(args.steps):
        frame = int(rng.choice(train_frames))
        pair_idx = rng.randrange(len(cameras))
        camera = cameras[pair_idx]
        gt_view = int(args.gt_view_ids[pair_idx])

        pose = pose_cache["poses"][frame]
        xyz = pose["xyz"].float().to(device)
        quat = F.normalize(pose["quat"].float().to(device), dim=-1)

        dynamic_gaussians = copy.copy(gaussians)
        dynamic_gaussians._xyz = xyz
        dynamic_gaussians._rotation = quat
        dynamic_gaussians._features_dc = gaussians._features_dc
        dynamic_gaussians._features_rest = gaussians._features_rest
        dynamic_gaussians._opacity = gaussians._opacity
        dynamic_gaussians._scaling = gaussians._scaling

        result = render(
            camera,
            dynamic_gaussians,
            pipe,
            background,
            use_gsplat=use_gsplat,
        )
        rendered = result["render"]
        pred_rgb = rendered[:3]
        pred_alpha = (
            rendered[3]
            if rendered.shape[0] >= 4
            else torch.ones_like(pred_rgb[0])
        )

        gt_rgb = load_rgb(
            gt_root / "color" / str(gt_view) / f"{frame}.png",
            device,
        )
        gt_mask = load_mask(
            gt_root
            / "mask"
            / str(gt_view)
            / str(object_ids[gt_view])
            / f"{frame}.png",
            device,
        )

        target_size = tuple(pred_rgb.shape[-2:])
        gt_rgb = resize_chw(gt_rgb, target_size)
        gt_mask = resize_hw(gt_mask, target_size)

        human_path = find_human_mask(human_root, gt_view, frame)
        if human_path is not None:
            human = resize_hw(load_mask(human_path, device), target_size)
            keep = 1.0 - human.clamp(0.0, 1.0)
        else:
            keep = torch.ones_like(gt_mask)

        target_mask = gt_mask.clamp(0.0, 1.0) * keep
        target_rgb = gt_rgb * target_mask[None]
        pred_eval = pred_rgb * keep[None]

        loss_rgb = torch.abs(pred_eval - target_rgb).mean()
        ssim_value = ssim(pred_eval[None], target_rgb[None])
        loss_ssim = 1.0 - ssim_value
        loss_alpha = torch.abs(pred_alpha * keep - target_mask).mean()

        reg = (gaussians._features_dc - dc0).pow(2).mean()
        if "opacity" in learnable:
            reg = reg + (gaussians._opacity - opa0).pow(2).mean()
        if "scaling" in learnable:
            reg = reg + (gaussians._scaling - scale0).pow(2).mean()

        loss = (
            loss_rgb
            + args.lambda_ssim * loss_ssim
            + args.lambda_alpha * loss_alpha
            + args.lambda_reg * reg
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        learnable_tensors = [
            tensor
            for group in optimizer_groups
            for tensor in group["params"]
        ]
        torch.nn.utils.clip_grad_norm_(learnable_tensors, max_norm=1.0)
        optimizer.step()

        row = {
            "step": step,
            "frame": frame,
            "gt_view": gt_view,
            "loss": float(loss.detach()),
            "rgb_loss": float(loss_rgb.detach()),
            "ssim": float(ssim_value.detach()),
            "alpha_loss": float(loss_alpha.detach()),
            "regularization": float(reg.detach()),
        }
        history.append(row)

        current = row["loss"]
        if current < best_loss:
            best_loss = current
            best_state = {
                "features_dc": gaussians._features_dc.detach().cpu().clone(),
                "opacity": gaussians._opacity.detach().cpu().clone(),
                "scaling": gaussians._scaling.detach().cpu().clone(),
            }

        if step % 10 == 0 or step == args.steps - 1:
            print(json.dumps(row, indent=2))

    if best_state is not None:
        gaussians._features_dc.data.copy_(best_state["features_dc"].to(device))
        gaussians._opacity.data.copy_(best_state["opacity"].to(device))
        gaussians._scaling.data.copy_(best_state["scaling"].to(device))

    history_csv = out_dir / "train_history.csv"
    with history_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    output_model = out_dir / "gaussian_model"
    ply_out = save_complete_gaussian_model(
        source_model=source_model_dir,
        output_model=output_model,
        gaussians=gaussians,
        loaded_iteration=loaded_iteration,
    )

    summary = {
        "protocol": "fixed_physics_gaussian_render_refinement",
        "variant": args.variant,
        "learnable_parameters": list(learnable),
        "physics_trajectory_frozen": True,
        "physics_topology_frozen": True,
        "test_frames_used_for_optimization": False,
        "inference": str(inference_path),
        "source_gaussian_model": str(source_model_dir),
        "output_gaussian_model": str(output_model),
        "point_cloud": str(ply_out),
        "pose_cache": str(pose_cache_path),
        "train_frames": train_frames,
        "camera_indices": list(map(int, args.camera_indices)),
        "gt_view_ids": list(map(int, args.gt_view_ids)),
        "steps": int(args.steps),
        "best_sample_loss": float(best_loss),
        "backend": args.backend,
        "history": str(history_csv),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("RENDER REFINEMENT DONE")
    print("variant :", args.variant)
    print("model   :", output_model)
    print("summary :", out_dir / "summary.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
