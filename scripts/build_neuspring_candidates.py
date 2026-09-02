#!/usr/bin/env python3
from __future__ import annotations

"""Generate generic NeuSpring recovery candidates for any Stage-2 budget.

This replaces notebook code that was tied to one scene and one target such as
"Final54".  The target is now an argument, the train/update interval is an
argument, and every candidate is recorded in a manifest.
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phystwin_reduction.neuspring import (
    RecoveryInputs,
    prepare_recovery_context,
    select_recovery_candidate,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build prior-control and NeuSpring topology-recovery candidates."
    )
    p.add_argument("--stage1-trainer", required=True, type=Path)
    p.add_argument("--node-checkpoint", required=True, type=Path)
    p.add_argument("--source-stage2-topology", required=True, type=Path)
    p.add_argument("--physical-inference", required=True, type=Path)
    p.add_argument("--full-topology", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)

    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-final-spring-ratio", type=float)
    target.add_argument("--target-total-springs", type=int)

    p.add_argument(
        "--selection-frame-start",
        type=int,
        default=0,
        help="Inclusive trajectory frame used for topology construction.",
    )
    p.add_argument(
        "--selection-frame-end",
        type=int,
        required=True,
        help="Exclusive end. Test frames must not be included.",
    )
    p.add_argument("--num-candidates", type=int, default=6)
    p.add_argument("--num-regions", type=int, default=6)
    p.add_argument("--dynamics-weight", type=float, default=0.5)

    p.add_argument("--radius-scale-min", type=float, default=0.75)
    p.add_argument("--radius-scale-max", type=float, default=1.35)
    p.add_argument("--knn-scale-min", type=float, default=0.75)
    p.add_argument("--knn-scale-max", type=float, default=1.30)

    p.add_argument("--prior-weight", type=float, default=0.55)
    p.add_argument("--eligibility-weight", type=float, default=0.25)
    p.add_argument("--edge-dynamics-weight", type=float, default=0.15)
    p.add_argument("--length-weight", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def full_spring_count(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        for key in ("springs", "edges", "reduced_edges"):
            if key in data.files:
                return int(len(data[key]))
    raise KeyError(
        f"Cannot determine full spring count from {path}; "
        "expected one of springs/edges/reduced_edges"
    )


def read_source_indices_and_prior(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        if "kept_stage1_object_indices" not in data.files:
            raise KeyError(
                f"{path} does not contain kept_stage1_object_indices. "
                "Use a Stage-2 topology produced by build_final_budget_stage2.py."
            )
        kept = np.asarray(data["kept_stage1_object_indices"], dtype=np.int64)

        if "final_edge_score" in data.files:
            prior = np.asarray(data["final_edge_score"], dtype=np.float64)
        elif "object_edge_scores" in data.files:
            prior = np.asarray(data["object_edge_scores"], dtype=np.float64)
        else:
            raise KeyError(
                f"{path} does not contain final_edge_score or object_edge_scores"
            )
    return kept, prior


def main() -> None:
    a = parse_args()

    stage1_trainer_path = require_file(a.stage1_trainer)
    node_checkpoint_path = require_file(a.node_checkpoint)
    source_topology_path = require_file(a.source_stage2_topology)
    physical_inference_path = require_file(a.physical_inference)
    full_topology_path = require_file(a.full_topology)
    out = a.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if a.num_candidates < 2:
        raise ValueError("--num-candidates must be >= 2 (prior control + NeuSpring)")
    if a.num_regions < 1:
        raise ValueError("--num-regions must be positive")
    if not (a.selection_frame_start < a.selection_frame_end):
        raise ValueError("selection frame range is empty")

    stage1 = np.load(stage1_trainer_path, allow_pickle=True)
    required = (
        "reduced_points",
        "reduced_edges",
        "reduced_edges_oo",
        "reduced_rest_lengths",
    )
    missing = [key for key in required if key not in stage1.files]
    if missing:
        raise KeyError(f"Stage-1 trainer is missing keys: {missing}")

    points = np.asarray(stage1["reduced_points"], dtype=np.float64)
    all_edges = np.asarray(stage1["reduced_edges"], dtype=np.int64)
    object_edges = np.asarray(stage1["reduced_edges_oo"], dtype=np.int64)
    all_rest = np.asarray(stage1["reduced_rest_lengths"], dtype=np.float64)

    checkpoint = torch.load(
        node_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "spring_Y" not in checkpoint:
        raise KeyError(f"{node_checkpoint_path} has no spring_Y")
    spring_y = (
        checkpoint["spring_Y"]
        .detach()
        .cpu()
        .float()
        .reshape(-1)
        .numpy()
    )
    if len(spring_y) != len(all_edges):
        raise ValueError(
            f"node checkpoint has {len(spring_y)} springs but trainer has {len(all_edges)}"
        )

    source_indices, prior_score = read_source_indices_and_prior(source_topology_path)
    if len(prior_score) != len(object_edges):
        raise ValueError(
            f"Stage-2 prior has {len(prior_score)} object scores; "
            f"Stage-1 trainer has {len(object_edges)} object springs"
        )

    with physical_inference_path.open("rb") as f:
        trajectory = np.asarray(pickle.load(f))
    if trajectory.ndim != 3:
        raise ValueError(
            f"Expected physical inference [T,N,3], got {trajectory.shape}"
        )
    if a.selection_frame_end > len(trajectory):
        raise ValueError(
            f"selection-frame-end={a.selection_frame_end} exceeds trajectory length "
            f"{len(trajectory)}"
        )
    trajectory = trajectory[
        int(a.selection_frame_start) : int(a.selection_frame_end),
        : len(points),
    ]

    full_total = full_spring_count(full_topology_path)
    if a.target_total_springs is not None:
        target_total = int(a.target_total_springs)
        target_ratio = target_total / full_total
    else:
        target_ratio = float(a.target_final_spring_ratio)
        if not 0.0 < target_ratio <= 1.0:
            raise ValueError("--target-final-spring-ratio must be in (0, 1]")
        target_total = int(round(full_total * target_ratio))

    context = prepare_recovery_context(
        RecoveryInputs(
            points=points,
            all_edges=all_edges,
            object_edges=object_edges,
            all_rest_lengths=all_rest,
            spring_y=spring_y,
            source_object_indices=source_indices,
            prior_score=prior_score,
            trajectory=trajectory,
            target_total_springs=target_total,
            num_regions=a.num_regions,
            seed=a.seed,
            dynamics_weight=a.dynamics_weight,
        )
    )

    rng = np.random.default_rng(a.seed)
    rows: list[dict] = []

    def write_candidate(
        candidate_id: str,
        *,
        kind: str,
        radius_scale: np.ndarray,
        knn_scale: np.ndarray,
        prior_only: bool,
    ) -> None:
        selected = select_recovery_candidate(
            context,
            radius_scale=radius_scale,
            knn_scale=knn_scale,
            prior_only=prior_only,
            prior_weight=a.prior_weight,
            eligibility_weight=a.eligibility_weight,
            dynamics_weight=a.edge_dynamics_weight,
            length_weight=a.length_weight,
        )

        idx = selected["selected_indices"]
        restore = selected["restored_indices"]

        new_object_edges = context.object_edges[idx]
        new_object_rest = context.object_rest_lengths[idx]
        new_object_y = context.object_spring_y[idx]

        new_edges = np.concatenate(
            [new_object_edges, context.controller_edges],
            axis=0,
        )
        new_rest = np.concatenate(
            [new_object_rest, context.controller_rest_lengths],
            axis=0,
        )
        new_y = np.concatenate(
            [new_object_y, context.controller_spring_y],
            axis=0,
        )
        if len(new_edges) != target_total or len(new_y) != target_total:
            raise RuntimeError("Candidate did not hit exact total spring budget")

        candidate_dir = out / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)

        arrays = {key: stage1[key] for key in stage1.files}
        arrays["reduced_edges"] = new_edges
        arrays["reduced_edges_oo"] = new_object_edges
        arrays["reduced_rest_lengths"] = new_rest
        arrays["reduced_spring_Y_init"] = new_y
        arrays["mode"] = np.asarray("neuspring_recovery")
        arrays["neuspring_candidate_kind"] = np.asarray(kind)
        arrays["target_final_spring_ratio"] = np.asarray(target_ratio)
        arrays["target_total_springs"] = np.asarray(target_total)
        np.savez_compressed(candidate_dir / "trainer.npz", **arrays)

        new_checkpoint = dict(checkpoint)
        new_checkpoint["spring_Y"] = torch.as_tensor(
            new_y,
            dtype=checkpoint["spring_Y"].dtype,
        ).reshape(-1)
        new_checkpoint["num_object_springs"] = int(len(new_object_edges))
        new_checkpoint.pop("optimizer_state_dict", None)
        new_checkpoint["protocol"] = "generic_neuspring_topology_recovery"
        new_checkpoint["neuspring_candidate_id"] = candidate_id
        new_checkpoint["target_final_spring_ratio"] = float(target_ratio)
        torch.save(new_checkpoint, candidate_dir / "initial.pth")

        u = new_object_edges[:, 0]
        v = new_object_edges[:, 1]
        edge_region = np.where(
            context.node_region[u] == context.node_region[v],
            context.node_region[u],
            context.num_regions,
        )
        np.savez_compressed(
            candidate_dir / "nsf_topology.npz",
            points=context.points.astype(np.float32),
            springs=new_object_edges.astype(np.int64),
            region_ids=edge_region.astype(np.int64),
        )

        metadata = {
            "candidate_id": candidate_id,
            "kind": kind,
            "prior_only": bool(prior_only),
            "target_final_spring_ratio": float(target_ratio),
            "target_total_springs": int(target_total),
            "object_springs": int(len(new_object_edges)),
            "controller_springs": int(len(context.controller_edges)),
            "restored_springs": int(len(restore)),
            "eligible_removed": int(selected["eligible"][context.removed_object_indices].sum()),
            "radius_scale": list(map(float, radius_scale)),
            "knn_scale": list(map(float, knn_scale)),
            "selected_indices": list(map(int, idx)),
            "restored_indices": list(map(int, restore)),
            "trainer": str(candidate_dir / "trainer.npz"),
            "initial_checkpoint": str(candidate_dir / "initial.pth"),
            "nsf_topology": str(candidate_dir / "nsf_topology.npz"),
        }
        (candidate_dir / "candidate.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        rows.append(metadata)

    ones = np.ones(context.num_regions, dtype=np.float64)
    write_candidate(
        "cand_000_prior",
        kind="prior",
        radius_scale=ones,
        knn_scale=ones,
        prior_only=True,
    )

    for idx in range(1, int(a.num_candidates)):
        write_candidate(
            f"cand_{idx:03d}_neuspring",
            kind="neuspring",
            radius_scale=rng.uniform(
                a.radius_scale_min,
                a.radius_scale_max,
                size=context.num_regions,
            ),
            knn_scale=rng.uniform(
                a.knn_scale_min,
                a.knn_scale_max,
                size=context.num_regions,
            ),
            prior_only=False,
        )

    manifest = {
        "protocol": "generic_neuspring_topology_recovery_candidates",
        "source_stage2_topology": str(source_topology_path),
        "stage1_trainer": str(stage1_trainer_path),
        "node_checkpoint": str(node_checkpoint_path),
        "physical_inference": str(physical_inference_path),
        "full_topology": str(full_topology_path),
        "full_total_springs": int(full_total),
        "source_object_springs": int(len(source_indices)),
        "target_final_spring_ratio": float(target_ratio),
        "target_total_springs": int(target_total),
        "selection_frames": [
            int(a.selection_frame_start),
            int(a.selection_frame_end),
        ],
        "test_frames_used_for_topology_selection": False,
        "num_regions": int(context.num_regions),
        "num_candidates": int(len(rows)),
        "candidates": [
            {
                key: value
                for key, value in row.items()
                if key not in ("selected_indices", "restored_indices")
            }
            for row in rows
        ],
    }
    (out / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    csv_fields = [
        "candidate_id",
        "kind",
        "prior_only",
        "target_final_spring_ratio",
        "target_total_springs",
        "object_springs",
        "controller_springs",
        "restored_springs",
        "eligible_removed",
        "trainer",
        "initial_checkpoint",
        "nsf_topology",
    ]
    with (out / "candidate_generation.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in csv_fields})

    print("=" * 80)
    print("NeuSpring candidates generated")
    print("source object springs :", len(source_indices))
    print("target total springs  :", target_total)
    print("target final ratio    :", target_ratio)
    print("candidates            :", len(rows))
    print("manifest              :", out / "candidate_manifest.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
