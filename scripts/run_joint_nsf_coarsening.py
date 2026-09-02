#!/usr/bin/env python3
from __future__ import annotations

"""Joint residual Neural Spring Field training for any coarsened topology.

The script has no Colab path, scene-specific budget, or external NeuSpring repo
dependency.  It optimizes a zero-initialized residual field through a manual
Warp -> PyTorch gradient bridge, then bakes ordinary PhysTwin checkpoints.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phystwin_reduction.neuspring import (
    JointResidualSpringField,
    build_spring_features,
    load_topology_npz,
)
from phystwin_reduction.phystwin_runtime import (
    prepare_phystwin,
    resolve_scene_root,
    set_all_seeds,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Joint residual NSF optimization through PhysTwin simulation loss."
    )
    p.add_argument("--phystwin-root", required=True, type=Path)
    p.add_argument("--scene", required=True)
    p.add_argument("--base-path", type=Path)
    p.add_argument("--coarsened-data", required=True, type=Path)
    p.add_argument("--base-checkpoint", required=True, type=Path)
    p.add_argument("--nsf-topology", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--train-frame",
        required=True,
        type=int,
        help="Exclusive optimization end. Keep test frames outside this interval.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--region-embed-dim", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-regions", type=int, default=8)
    p.add_argument(
        "--max-delta-log-y",
        type=float,
        default=0.10,
        help="Bound on residual log stiffness; 0.10 is roughly x0.90..x1.11.",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def state_dict_cpu(model: torch.nn.Module) -> dict:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def parameter_grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().item())
    return total ** 0.5


def main() -> None:
    a = parse_args()
    if a.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if a.train_frame < 2:
        raise ValueError("--train-frame must be >= 2")
    if a.lr <= 0 or a.adam_eps <= 0:
        raise ValueError("--lr and --adam-eps must be positive")
    if a.max_delta_log_y <= 0:
        raise ValueError("--max-delta-log-y must be positive")

    root = a.phystwin_root.expanduser().resolve()
    scene_root = resolve_scene_root(root, a.scene, a.base_path)
    coarsened_data = a.coarsened_data.expanduser().resolve()
    base_checkpoint = a.base_checkpoint.expanduser().resolve()
    nsf_topology = a.nsf_topology.expanduser().resolve()
    out = a.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    for path in (coarsened_data, base_checkpoint, nsf_topology):
        if not path.is_file():
            raise FileNotFoundError(path)

    cfg = prepare_phystwin(root, a.scene, scene_root, seed=a.seed)
    cfg.use_graph = False
    cfg.train_frame = int(a.train_frame)
    set_all_seeds(a.seed)

    from qqtt.engine.trainer_warp_coarsening import InvPhyTrainerWarpCoarsening

    trainer = InvPhyTrainerWarpCoarsening(
        data_path=str(scene_root / "final_data.pkl"),
        base_dir=str(out),
        coarsened_data=str(coarsened_data),
        train_frame=int(a.train_frame),
        pure_inference_mode=False,
        device=a.device,
    )
    simulator = trainer.simulator

    base = torch.load(
        base_checkpoint,
        map_location=a.device,
        weights_only=False,
    )
    required_checkpoint_keys = (
        "spring_Y",
        "collide_elas",
        "collide_fric",
        "collide_object_elas",
        "collide_object_fric",
    )
    missing = [key for key in required_checkpoint_keys if key not in base]
    if missing:
        raise KeyError(f"Base checkpoint is missing keys: {missing}")

    base_y = base["spring_Y"].detach().float().reshape(-1).to(a.device)
    base_shape = base["spring_Y"].shape
    n_total = len(base_y)
    n_object = int(trainer.num_object_springs)

    if n_total != len(trainer.init_springs):
        raise ValueError(
            f"checkpoint has {n_total} springs but trainer has {len(trainer.init_springs)}"
        )

    trainer.simulator.set_collide(
        base["collide_elas"].detach().clone(),
        base["collide_fric"].detach().clone(),
    )
    trainer.simulator.set_collide_object(
        base["collide_object_elas"].detach().clone(),
        base["collide_object_fric"].detach().clone(),
    )

    topology = load_topology_npz(nsf_topology)
    features = build_spring_features(
        topology["points"],
        topology["edges"],
        region_ids=topology.get("region_ids"),
        num_regions=a.num_regions,
        seed=a.seed,
    )
    numeric = torch.from_numpy(features["numeric"]).float().to(a.device)
    region_ids = torch.from_numpy(features["region_ids"]).long().to(a.device)
    num_regions = int(features["num_regions"])

    if len(numeric) != n_object:
        raise ValueError(
            f"NSF topology has {len(numeric)} object springs; trainer has {n_object}"
        )

    field = JointResidualSpringField(
        numeric_dim=int(numeric.shape[-1]),
        num_regions=num_regions,
        region_embed_dim=a.region_embed_dim,
        hidden_dim=a.hidden_dim,
        num_layers=a.num_layers,
    ).to(a.device)

    optimizer = torch.optim.AdamW(
        field.parameters(),
        lr=a.lr,
        betas=(0.9, 0.99),
        eps=a.adam_eps,
        weight_decay=1e-6,
    )
    base_log_y = torch.log(base_y.clamp_min(1e-8))

    def current_parameters() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = field(numeric, region_ids).reshape(-1)
        delta_log = float(a.max_delta_log_y) * torch.tanh(raw)
        object_log_y = base_log_y[:n_object] + delta_log
        full_log_y = torch.cat(
            [object_log_y, base_log_y[n_object:]],
            dim=0,
        )
        return object_log_y, full_log_y, delta_log

    with torch.no_grad():
        _, initial_log_y, _ = current_parameters()
        initial_y = torch.exp(initial_log_y)
    initial_error = float((initial_y - base_y).abs().max().item())
    if initial_error > 1e-5:
        raise RuntimeError(
            f"Zero-initialized Joint NSF does not reproduce base checkpoint: {initial_error}"
        )

    print("=" * 80)
    print("JOINT NEURAL SPRING FIELD")
    print("scene          :", a.scene)
    print("physical nodes :", trainer.num_physical_points)
    print("total springs  :", n_total)
    print("object springs :", n_object)
    print("train frames   : [0,", a.train_frame, ")")
    print("epochs         :", a.epochs)
    print("base checkpoint:", base_checkpoint)
    print("NSF topology   :", nsf_topology)
    print("NSF features   :", tuple(numeric.shape))
    print("NSF regions    :", num_regions)
    print("initial error  :", initial_error)
    print("=" * 80)

    history: list[dict] = []
    best_loss: float | None = None
    best_checkpoint: Path | None = None

    for epoch in range(int(a.epochs)):
        simulator.set_init_state(
            simulator.wp_init_vertices,
            simulator.wp_init_velocities,
        )
        totals = {
            "loss": 0.0,
            "chamfer_loss": 0.0,
            "track_loss": 0.0,
            "warp_grad_abs_mean": 0.0,
            "nsf_grad_norm": 0.0,
        }
        frames = 0

        for frame_idx in range(1, int(a.train_frame)):
            optimizer.zero_grad(set_to_none=True)
            object_log_y, full_log_y, _ = current_parameters()

            # PhysTwin's Warp spring parameter is log(Y).  The detach is
            # intentional; the Warp gradient is bridged back manually below.
            simulator.set_spring_Y(full_log_y.detach())
            simulator.set_controller_target(frame_idx)
            if simulator.object_collision_flag:
                simulator.update_collision_graph()

            with simulator.tape:
                simulator.step()
                simulator.calculate_loss()
            simulator.tape.backward(simulator.loss)
            wp.synchronize_device()

            wp_grad_array = simulator.wp_spring_Y.grad
            if wp_grad_array is None:
                raise RuntimeError("wp_spring_Y.grad is None")
            warp_grad = (
                wp.to_torch(wp_grad_array, requires_grad=False)
                .detach()
                .clone()
            )
            if not torch.isfinite(warp_grad).all():
                raise FloatingPointError("Non-finite Warp spring gradient")

            object_log_y.backward(gradient=warp_grad[:n_object])
            nsf_grad = parameter_grad_norm(field)
            torch.nn.utils.clip_grad_norm_(
                field.parameters(),
                float(a.grad_clip),
            )
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            loss = float(
                wp.to_torch(simulator.loss, requires_grad=False).item()
            )
            chamfer = float(
                wp.to_torch(simulator.chamfer_loss, requires_grad=False).item()
            )
            track = float(
                wp.to_torch(simulator.track_loss, requires_grad=False).item()
            )

            totals["loss"] += loss
            totals["chamfer_loss"] += chamfer
            totals["track_loss"] += track
            totals["warp_grad_abs_mean"] += float(warp_grad.abs().mean().item())
            totals["nsf_grad_norm"] += float(nsf_grad)
            frames += 1

            simulator.tape.reset()
            simulator.clear_loss()
            simulator.set_init_state(
                simulator.wp_states[-1].wp_x,
                simulator.wp_states[-1].wp_v,
            )

        divisor = max(1, frames)
        for key in totals:
            totals[key] /= divisor

        with torch.no_grad():
            _, final_log_y, delta_log = current_parameters()
            final_y = torch.exp(final_log_y)

        row = {
            "epoch": epoch,
            **totals,
            "delta_logY_abs_mean": float(delta_log.abs().mean().item()),
            "delta_logY_abs_max": float(delta_log.abs().max().item()),
            "spring_Y_min": float(final_y.min().item()),
            "spring_Y_mean": float(final_y.mean().item()),
            "spring_Y_max": float(final_y.max().item()),
        }
        if not np.isfinite(list(row.values())).all():
            raise FloatingPointError(f"Non-finite Joint NSF metrics: {row}")
        history.append(row)
        print("[JOINT NSF]", json.dumps(row, indent=2))

        baked = dict(base)
        baked["spring_Y"] = final_y.detach().cpu().reshape(base_shape)
        baked["epoch"] = epoch
        baked["protocol"] = "joint_residual_neural_spring_field"
        baked["joint_nsf_base_checkpoint"] = str(base_checkpoint)
        baked["joint_nsf_topology"] = str(nsf_topology)
        baked["joint_nsf_max_delta_log_y"] = float(a.max_delta_log_y)
        baked.pop("optimizer_state_dict", None)

        torch.save(baked, out / f"joint_epoch_{epoch}.pth")
        field_state = {
            "model_state_dict": state_dict_cpu(field),
            "epoch": epoch,
            "numeric_dim": int(numeric.shape[-1]),
            "num_regions": num_regions,
            "hidden_dim": int(a.hidden_dim),
            "region_embed_dim": int(a.region_embed_dim),
            "num_layers": int(a.num_layers),
            "max_delta_log_y": float(a.max_delta_log_y),
            "base_checkpoint": str(base_checkpoint),
            "nsf_topology": str(nsf_topology),
            "feature_description": (
                "midpoint_xyz_norm + length_norm + direction_xyz + region_embedding"
            ),
        }
        torch.save(field_state, out / f"joint_field_epoch_{epoch}.pt")

        if best_loss is None or totals["loss"] < best_loss:
            best_loss = totals["loss"]
            best_checkpoint = out / "best_joint_nsf.pth"
            torch.save(baked, best_checkpoint)
            torch.save(field_state, out / "best_joint_field.pt")

    history_csv = out / "joint_nsf_history.csv"
    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "protocol": "joint_residual_neural_spring_field",
        "scene": a.scene,
        "train_frame": int(a.train_frame),
        "epochs": int(a.epochs),
        "base_checkpoint": str(base_checkpoint),
        "coarsened_data": str(coarsened_data),
        "nsf_topology": str(nsf_topology),
        "best_loss": best_loss,
        "best_checkpoint": str(best_checkpoint),
        "history": str(history_csv),
        "test_frames_used_for_optimization": False,
    }
    (out / "joint_nsf_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("JOINT NSF TRAINING DONE")
    print("best loss      :", best_loss)
    print("best checkpoint:", best_checkpoint)
    print("history        :", history_csv)
    print("=" * 80)


if __name__ == "__main__":
    main()
