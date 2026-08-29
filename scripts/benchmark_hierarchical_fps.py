#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phystwin_reduction.pytorch3d_stub import install_pytorch3d_stub
install_pytorch3d_stub()


def load_cfg(root: Path, scene: str):
    sys.path.insert(0, str(root))
    os.chdir(root)
    from qqtt.utils import cfg

    cfg.load_from_yaml(str(root / ("configs/cloth.yaml" if "cloth" in scene or "package" in scene else "configs/real.yaml")))
    with (root / "experiments_optimization" / scene / "optimal_params.pkl").open("rb") as f:
        cfg.set_optimal_params(pickle.load(f))
    scene_root = root / "data" / "different_types" / scene
    with (scene_root / "calibrate.pkl").open("rb") as f:
        c2ws = pickle.load(f)
    cfg.c2ws = np.asarray(c2ws)
    cfg.w2cs = np.asarray([np.linalg.inv(x) for x in c2ws])
    with (scene_root / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    cfg.intrinsics = np.asarray(metadata["intrinsics"])
    cfg.WH = metadata["WH"]
    cfg.overlay_path = str(scene_root / "color")
    cfg.device = "cuda:0"
    return cfg, scene_root


def load_checkpoint(sim, path: Path):
    ckpt = torch.load(path, map_location="cuda:0")
    spring_y = ckpt["spring_Y"]
    if len(spring_y) != sim.n_springs:
        raise RuntimeError(f"checkpoint springs={len(spring_y)} simulator springs={sim.n_springs}")
    sim.set_spring_Y(torch.log(spring_y).detach().clone())
    sim.set_collide(ckpt["collide_elas"].detach().clone(), ckpt["collide_fric"].detach().clone())
    sim.set_collide_object(
        ckpt["collide_object_elas"].detach().clone(),
        ckpt["collide_object_fric"].detach().clone(),
    )


def make_trainer(root: Path, scene: str, mode: str, coarse_path: Path | None, checkpoint: Path):
    cfg, scene_root = load_cfg(root, scene)
    if mode == "full":
        from qqtt import InvPhyTrainerWarp
        trainer = InvPhyTrainerWarp(
            data_path=str(scene_root / "final_data.pkl"),
            base_dir=str(root / "results" / "fps_tmp_full"),
            pure_inference_mode=True,
            device="cuda:0",
        )
    else:
        if coarse_path is None:
            raise ValueError("--coarse-path is required for --mode reduced")
        from qqtt.engine.trainer_warp_coarsening import InvPhyTrainerWarpCoarsening
        trainer = InvPhyTrainerWarpCoarsening(
            data_path=str(scene_root / "final_data.pkl"),
            base_dir=str(root / "results" / "fps_tmp_reduced"),
            coarsened_data=str(coarse_path),
            pure_inference_mode=True,
            device="cuda:0",
        )
    load_checkpoint(trainer.simulator, checkpoint)
    return trainer, cfg, scene_root


def advance(sim, cfg, frame_idx: int, reconstruct: bool):
    sim.set_controller_target(frame_idx, pure_inference=True)
    if sim.object_collision_flag:
        sim.update_collision_graph()
    if cfg.use_graph:
        wp.capture_launch(sim.forward_graph)
    else:
        sim.step()
    if reconstruct:
        sim.reconstruct_dense(sim.wp_states[-1].wp_x)
    sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)


def benchmark(trainer, cfg, start: int, end: int, repeats: int, reconstruct: bool):
    sim = trainer.simulator
    values = []
    for repeat in range(repeats + 1):
        sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
        for frame in range(1, start):
            advance(sim, cfg, frame, reconstruct)
        wp.synchronize_device()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for frame in range(start, end):
            advance(sim, cfg, frame, reconstruct)
        wp.synchronize_device()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        fps = (end - start) / elapsed
        print(f"repeat={repeat:02d} elapsed={elapsed:.6f}s fps={fps:.3f}")
        if repeat > 0:
            values.append(fps)
    return {
        "fps_mean": float(np.mean(values)),
        "fps_std": float(np.std(values)),
        "fps_min": float(np.min(values)),
        "fps_max": float(np.max(values)),
        "repeats": int(repeats),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phystwin-root", required=True, type=Path)
    p.add_argument("--scene", required=True)
    p.add_argument("--mode", choices=["full", "reduced"], required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--coarse-path", type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--measure-start", type=int)
    p.add_argument("--measure-end", type=int)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    root = args.phystwin_root.expanduser().resolve()
    trainer, cfg, scene_root = make_trainer(
        root,
        args.scene,
        args.mode,
        args.coarse_path.expanduser().resolve() if args.coarse_path else None,
        args.checkpoint.expanduser().resolve(),
    )
    split = json.loads((scene_root / "split.json").read_text(encoding="utf-8"))
    start = int(args.measure_start) if args.measure_start is not None else int(split["test"][0])
    end = int(args.measure_end) if args.measure_end is not None else int(split["test"][1])

    print("===== FPS BENCHMARK =====")
    print("label   :", args.label)
    print("scene   :", args.scene)
    print("GPU     :", torch.cuda.get_device_name(0))
    print("frames  :", [start, end])
    print("springs :", trainer.simulator.n_springs)

    physics = benchmark(trainer, cfg, start, end, args.repeats, reconstruct=False)
    result = {
        "label": args.label,
        "mode": args.mode,
        "scene": args.scene,
        "gpu": torch.cuda.get_device_name(0),
        "dt": float(cfg.dt),
        "substeps": int(cfg.num_substeps),
        "springs": int(trainer.simulator.n_springs),
        "measure_start": start,
        "measure_end": end,
        "physics": physics,
    }
    if args.mode == "reduced":
        result["physics_plus_dense"] = benchmark(
            trainer, cfg, start, end, args.repeats, reconstruct=True
        )
        result["physical_nodes"] = int(trainer.num_physical_points)
        result["dense_nodes"] = int(trainer.num_all_points)
    else:
        result["physics_plus_dense"] = physics
        result["physical_nodes"] = int(trainer.num_all_points)
        result["dense_nodes"] = int(trainer.num_all_points)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("[SAVED]", args.output)


if __name__ == "__main__":
    main()
