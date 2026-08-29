#!/usr/bin/env python3
from __future__ import annotations

"""Formal Stage-1 retraining for a PhysTwin node-coarsened graph.

Unlike the original Node-Coarsening helper, this version accepts --train_frame
so the formal online protocol can train only on Stage-1 frames (for example
0..66 when --train_frame 67), leaving the remaining train split exclusively
for online update/error estimation.
"""

import argparse
import csv
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_stats(x: torch.Tensor) -> tuple[float, float, float]:
    x = x.detach().float().cpu()
    return float(x.min()), float(x.mean()), float(x.max())


def current_model_and_stats(trainer, wp) -> tuple[dict, dict]:
    spring_y = torch.exp(
        wp.to_torch(trainer.simulator.wp_spring_Y, requires_grad=False)
    ).detach().clone()
    collide_elas = wp.to_torch(
        trainer.simulator.wp_collide_elas, requires_grad=False
    ).detach().clone()
    collide_fric = wp.to_torch(
        trainer.simulator.wp_collide_fric, requires_grad=False
    ).detach().clone()
    collide_object_elas = wp.to_torch(
        trainer.simulator.wp_collide_object_elas, requires_grad=False
    ).detach().clone()
    collide_object_fric = wp.to_torch(
        trainer.simulator.wp_collide_object_fric, requires_grad=False
    ).detach().clone()

    y_min, y_mean, y_max = tensor_stats(spring_y)
    stats = {
        "spring_Y_min": y_min,
        "spring_Y_mean": y_mean,
        "spring_Y_max": y_max,
        "collide_elas": float(collide_elas.cpu().flatten()[0]),
        "collide_fric": float(collide_fric.cpu().flatten()[0]),
        "collide_object_elas": float(collide_object_elas.cpu().flatten()[0]),
        "collide_object_fric": float(collide_object_fric.cpu().flatten()[0]),
    }
    model = {
        "epoch": None,
        "num_object_springs": trainer.num_object_springs,
        "num_physical_points": trainer.num_physical_points,
        "num_dense_points": trainer.num_all_points,
        "coarsening_mode": trainer.coarsening_mode,
        "coarsened_data": trainer.coarsened_data_path,
        "spring_Y": spring_y,
        "collide_elas": collide_elas,
        "collide_fric": collide_fric,
        "collide_object_elas": collide_object_elas,
        "collide_object_fric": collide_object_fric,
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "protocol": "node_coarsening_stage1_retrain",
    }
    return model, stats


def train_best(
    trainer,
    cfg,
    wp,
    logger,
    tqdm,
    out_dir: Path,
    checkpoint_interval: int,
) -> Path:
    train_dir = out_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    history_path = train_dir / "coarsening_train_history.csv"
    fields = [
        "epoch",
        "loss",
        "chamfer_loss",
        "track_loss",
        "spring_Y_min",
        "spring_Y_mean",
        "spring_Y_max",
        "collide_elas",
        "collide_fric",
        "collide_object_elas",
        "collide_object_fric",
        "is_new_best",
        "is_final_best",
    ]

    best_loss: float | None = None
    best_epoch: int | None = None
    best_path: Path | None = None
    rows: list[dict] = []

    for epoch in range(cfg.iterations):
        total_loss = total_chamfer = total_track = 0.0
        trainer.simulator.set_init_state(
            trainer.simulator.wp_init_vertices,
            trainer.simulator.wp_init_velocities,
        )

        for frame_idx in tqdm(range(1, cfg.train_frame)):
            trainer.simulator.set_controller_target(frame_idx)
            if trainer.simulator.object_collision_flag:
                trainer.simulator.update_collision_graph()

            if cfg.use_graph:
                wp.capture_launch(trainer.simulator.graph)
            else:
                with trainer.simulator.tape:
                    trainer.simulator.step()
                    trainer.simulator.calculate_loss()
                trainer.simulator.tape.backward(trainer.simulator.loss)

            wp.synchronize_device()
            trainer.optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            chamfer = wp.to_torch(trainer.simulator.chamfer_loss, requires_grad=False)
            track = wp.to_torch(trainer.simulator.track_loss, requires_grad=False)
            loss = wp.to_torch(trainer.simulator.loss, requires_grad=False)
            total_chamfer += float(chamfer.item())
            total_track += float(track.item())
            total_loss += float(loss.item())

            if cfg.use_graph:
                trainer.simulator.tape.zero()
            else:
                trainer.simulator.tape.reset()
            trainer.simulator.clear_loss()
            trainer.simulator.set_init_state(
                trainer.simulator.wp_states[-1].wp_x,
                trainer.simulator.wp_states[-1].wp_v,
            )

        divisor = max(1, cfg.train_frame - 1)
        total_loss /= divisor
        total_chamfer /= divisor
        total_track /= divisor
        if not np.isfinite([total_loss, total_chamfer, total_track]).all():
            raise FloatingPointError(
                f"Non-finite metric at epoch={epoch}: loss={total_loss}, "
                f"chamfer={total_chamfer}, track={total_track}"
            )

        model, stats = current_model_and_stats(trainer, wp)
        model["epoch"] = epoch
        is_best = best_loss is None or total_loss < best_loss
        if is_best:
            if best_path is not None and best_path.exists():
                best_path.unlink(missing_ok=True)
            best_loss = total_loss
            best_epoch = epoch
            best_path = train_dir / f"best_{epoch}.pth"
            torch.save(model, best_path)

        iteration = epoch + 1
        if checkpoint_interval > 0 and (
            iteration % checkpoint_interval == 0 or iteration == cfg.iterations
        ):
            torch.save(model, train_dir / f"iter_{iteration}.pth")

        rows.append(
            {
                "epoch": epoch,
                "loss": total_loss,
                "chamfer_loss": total_chamfer,
                "track_loss": total_track,
                **stats,
                "is_new_best": int(is_best),
                "is_final_best": 0,
            }
        )
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(
            f"[COARSENING TRAIN] epoch={epoch}, loss={total_loss:.8g}, "
            f"chamfer={total_chamfer:.8g}, track={total_track:.8g}"
        )

    if best_path is None or best_epoch is None:
        raise RuntimeError("Training produced no valid checkpoint")
    for row in rows:
        row["is_final_best"] = int(row["epoch"] == best_epoch)
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"[COARSENING DONE] best_epoch={best_epoch}, best_loss={best_loss}")
    return best_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phystwin_root", required=True, type=Path)
    p.add_argument("--base_path", type=Path)
    p.add_argument("--case_name", required=True)
    p.add_argument("--coarsened_data", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train_frame", type=int, required=True,
                   help="Exclusive Stage-1 training end; update/test frames are not optimized here.")
    p.add_argument("--dt_scale", type=float, default=1.0)
    p.add_argument("--retrain_epochs", type=int, default=200)
    p.add_argument("--checkpoint_interval", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--spring_Y_min_override", type=float)
    p.add_argument("--spring_Y_max_override", type=float)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.dt_scale) or args.dt_scale <= 0.0:
        raise ValueError("--dt_scale must be finite and positive")

    root = args.phystwin_root.expanduser().resolve()
    base_path = (
        args.base_path.expanduser().resolve()
        if args.base_path is not None
        else root / "data" / "different_types"
    )
    scene_root = base_path / args.case_name
    coarsened_data = args.coarsened_data.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not coarsened_data.is_file():
        raise FileNotFoundError(coarsened_data)

    split = json.loads((scene_root / "split.json").read_text(encoding="utf-8"))
    split_train_end = int(split["train"][1])
    train_frame = int(args.train_frame)
    if train_frame < 2 or train_frame > split_train_end:
        raise ValueError(
            f"train_frame must be in [2, {split_train_end}], got {train_frame}"
        )

    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("SKIP_OPEN3D_VIDEO", "1")
    os.chdir(root)
    sys.path.insert(0, str(root))
    set_all_seeds(args.seed)

    from qqtt.engine.trainer_warp_coarsening import InvPhyTrainerWarpCoarsening
    from qqtt.utils import cfg, logger
    import warp as wp
    from tqdm import tqdm

    if "cloth" in args.case_name or "package" in args.case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    optimal_path = root / "experiments_optimization" / args.case_name / "optimal_params.pkl"
    if not optimal_path.is_file():
        raise FileNotFoundError(optimal_path)
    with optimal_path.open("rb") as f:
        cfg.set_optimal_params(pickle.load(f))

    with (scene_root / "calibrate.pkl").open("rb") as f:
        c2ws = pickle.load(f)
    cfg.c2ws = np.asarray(c2ws)
    cfg.w2cs = np.asarray([np.linalg.inv(c2w) for c2w in c2ws])
    with (scene_root / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    cfg.intrinsics = np.asarray(metadata["intrinsics"])
    cfg.WH = metadata["WH"]
    cfg.overlay_path = str(scene_root / "color")
    cfg.iterations = int(args.retrain_epochs)
    cfg.train_frame = train_frame

    original_dt = float(cfg.dt)
    original_substeps = int(cfg.num_substeps)
    case_scale = 0.5 if args.case_name == "double_stretch_zebra" else 1.0
    effective_scale = case_scale * float(args.dt_scale)
    cfg.dt = original_dt * effective_scale
    cfg.num_substeps = int(round(original_substeps / effective_scale))

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.set_log_file(path=str(out_dir), name="coarsening_retrain_log")

    coarse = np.load(coarsened_data, allow_pickle=True)
    print("=== Stage-1 node-coarsened retraining ===")
    print("scene           =", args.case_name)
    print("mode            =", str(np.asarray(coarse["mode"]).item()))
    print("physical nodes  =", int(coarse["n_object_original"]), "->", int(coarse["n_object_reduced"]))
    print("springs         =", len(coarse["reduced_edges"]))
    print("train frames    = [0,", train_frame, ")")
    print("epochs          =", cfg.iterations)

    trainer = InvPhyTrainerWarpCoarsening(
        data_path=str(scene_root / "final_data.pkl"),
        base_dir=str(out_dir),
        train_frame=train_frame,
        pure_inference_mode=False,
        device=args.device,
        coarsened_data=str(coarsened_data),
        spring_Y_min_override=args.spring_Y_min_override,
        spring_Y_max_override=args.spring_Y_max_override,
    )
    best = train_best(
        trainer, cfg, wp, logger, tqdm, out_dir, args.checkpoint_interval
    )

    print("best checkpoint =", best)
    print("Running fresh dense inference from the best Stage-1 reduced checkpoint ...")
    infer_trainer = InvPhyTrainerWarpCoarsening(
        data_path=str(scene_root / "final_data.pkl"),
        base_dir=str(out_dir),
        train_frame=None,
        pure_inference_mode=True,
        device=args.device,
        coarsened_data=str(coarsened_data),
        spring_Y_min_override=args.spring_Y_min_override,
        spring_Y_max_override=args.spring_Y_max_override,
    )
    infer_trainer.test(model_path=str(best))
    print("[DONE] dense inference   :", out_dir / "inference.pkl")
    print("[DONE] physical inference:", out_dir / "inference_physical.pkl")
    print("[DONE] history           :", out_dir / "train" / "coarsening_train_history.csv")


if __name__ == "__main__":
    main()
