#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phystwin_reduction.stage2_post_retrain import generate_final_budget_topology


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build the stable Stage-2 Online-BT graph at a FINAL spring budget. "
            "BT/stiffness are computed from the post-node-retraining topology."
        )
    )
    p.add_argument("--full-topology", required=True, type=Path)
    p.add_argument("--retrained-topology", required=True, type=Path)
    p.add_argument("--coarse-node-error", required=True, type=Path)
    p.add_argument("--base-trainer", required=True, type=Path)
    p.add_argument("--node-checkpoint", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--target-final-spring-ratio", required=True, type=float)
    p.add_argument("--bt-weight", type=float, default=0.7)
    p.add_argument("--online-error-weight", type=float, default=0.3)
    p.add_argument("--min-degree", type=int, default=1)
    p.add_argument("--local-budget", type=int, default=300)
    p.add_argument("--reduced-order", type=int, default=20)
    p.add_argument("--label", default="final_budget")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    retrained_topology_path = args.retrained_topology.expanduser().resolve()
    node_ckpt_path = args.node_checkpoint.expanduser().resolve()
    retrained_raw = np.load(retrained_topology_path, allow_pickle=True)
    if "retrained_checkpoint" not in retrained_raw.files:
        raise RuntimeError(
            f"{retrained_topology_path} is not marked as post-node-retraining. "
            "Run apply_checkpoint_to_topology.py first."
        )
    marker = Path(str(np.asarray(retrained_raw["retrained_checkpoint"]).item())).expanduser().resolve()
    if marker != node_ckpt_path:
        raise RuntimeError(
            "The retrained topology and node checkpoint do not match:\n"
            f"  topology marker: {marker}\n"
            f"  checkpoint     : {node_ckpt_path}"
        )

    error_file = np.load(args.coarse_node_error.expanduser().resolve(), allow_pickle=True)
    error_key = "node_error_normalized" if "node_error_normalized" in error_file.files else "node_error"
    coarse_error = np.asarray(error_file[error_key], dtype=np.float64)

    topology_path, meta = generate_final_budget_topology(
        args.full_topology,
        retrained_topology_path,
        coarse_error,
        out / "topology.npz",
        target_final_ratio=args.target_final_spring_ratio,
        bt_weight=args.bt_weight,
        online_error_weight=args.online_error_weight,
        min_degree=args.min_degree,
        local_budget=args.local_budget,
        reduced_order=args.reduced_order,
    )

    # Build trainer NPZ by keeping the Stage-1 node mapping and replacing only edges.
    base = np.load(args.base_trainer.expanduser().resolve(), allow_pickle=True)
    pruned = np.load(topology_path, allow_pickle=True)
    arrays = {k: base[k] for k in base.files}
    e_obj = int(np.asarray(pruned["num_object_springs"]).item())
    arrays["reduced_edges"] = np.asarray(pruned["springs"])
    arrays["reduced_edges_oo"] = np.asarray(pruned["springs"])[:e_obj]
    arrays["reduced_rest_lengths"] = np.asarray(pruned["rest_lengths"])
    arrays["reduced_spring_Y_init"] = np.asarray(pruned["spring_Y"])
    arrays["mode"] = np.asarray(args.label)
    trainer_path = out / "trainer.npz"
    np.savez_compressed(trainer_path, **arrays)

    # Build a matching pure-inference checkpoint. The pruned spring_Y values are
    # exactly the retained post-node-retraining parameters; there is no Stage-2 retrain.
    ckpt = torch.load(node_ckpt_path, map_location="cpu")
    final_ckpt = dict(ckpt)
    final_ckpt["spring_Y"] = torch.as_tensor(
        np.asarray(pruned["spring_Y"]), dtype=torch.float32
    )
    final_ckpt["num_object_springs"] = e_obj
    final_ckpt.pop("optimizer_state_dict", None)
    final_ckpt["stage2_protocol"] = "post_node_retrain_final_budget_online_bt"
    final_ckpt["stage2_target_final_ratio"] = float(args.target_final_spring_ratio)
    final_ckpt["stage2_actual_final_ratio"] = float(meta["actual_final_ratio"])
    final_ckpt["source_node_checkpoint"] = str(node_ckpt_path)
    checkpoint_path = out / "initial.pth"
    torch.save(final_ckpt, checkpoint_path)

    if len(arrays["reduced_edges"]) != len(final_ckpt["spring_Y"]):
        raise RuntimeError("Final trainer/checkpoint spring counts do not match")
    summary = {
        **meta,
        "topology": str(topology_path),
        "trainer": str(trainer_path),
        "checkpoint": str(checkpoint_path),
        "node_checkpoint": str(node_ckpt_path),
        "coarse_node_error": str(args.coarse_node_error.expanduser().resolve()),
    }
    (out / "stage2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("===== STAGE-2 FINAL BUDGET =====")
    print("target final ratio :", args.target_final_spring_ratio)
    print("actual final ratio :", meta["actual_final_ratio"])
    print("total springs      :", meta["actual_total_springs"])
    print("object keep ratio  :", meta["stage2_object_keep_ratio"])
    print("graph              :", meta["graph_stats"])
    print("[DONE] topology    :", topology_path)
    print("[DONE] trainer     :", trainer_path)
    print("[DONE] checkpoint  :", checkpoint_path)


if __name__ == "__main__":
    main()
