from __future__ import annotations

"""Stage-2 spring reduction on a *retrained node-coarsened* PhysTwin graph.

This module intentionally does not infer the coarse object-node count from a
controller-edge column.  Hierarchical coarsening may store controller springs
as either [object, controller] or [controller, object].  Instead, the object
count is read from the coarsening arrays and controller seeds are extracted
orientation-independently.

The intended protocol is:

    full Stage-1 -> node coarsening -> physics retraining -> best checkpoint
    -> write retrained spring_Y back to the coarse topology
    -> online residual on update frames -> project residual to coarse nodes
    -> BT + stiffness + online-error scoring on the retrained coarse dynamics
    -> connectivity-constrained pruning to a target FINAL spring budget.

Test frames must not be used for topology selection.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .bt_guided import (
    _reduce_with_prbt_or_bt,
    bfs_local_nodes,
    build_local_lti_from_topology,
    state_scores_to_node_scores,
)
from .topology import (
    TopologyData,
    add_min_degree_edges,
    graph_stats,
    load_topology,
    maximum_spanning_forest_indices,
    normalize_score,
)


@dataclass(frozen=True)
class FinalBudget:
    full_total_springs: int
    stage1_total_springs: int
    stage1_object_springs: int
    stage1_controller_springs: int
    target_final_ratio: float
    target_total_springs: int
    target_object_springs: int
    stage2_object_keep_ratio: float


def infer_coarse_object_points(topology_path: str | Path, data: TopologyData | None = None) -> int:
    """Infer object-node count without assuming controller-edge orientation."""
    path = Path(topology_path).expanduser().resolve()
    z = np.load(path, allow_pickle=True)

    if "reduced_object_points" in z.files:
        return int(len(z["reduced_object_points"]))
    if "n_object_reduced" in z.files:
        return int(np.asarray(z["n_object_reduced"]).item())
    if "original_num_object_points" in z.files and "reduction_type" not in z.files:
        return int(np.asarray(z["original_num_object_points"]).item())

    topo = data if data is not None else load_topology(path)
    if len(topo.object_springs):
        return int(np.max(topo.object_springs)) + 1
    return int(len(topo.points_full))


def require_retrained_topology(topology_path: str | Path) -> None:
    """Reject the old incorrect Stage-2 protocol that scores pre-retrain topology."""
    path = Path(topology_path).expanduser().resolve()
    z = np.load(path, allow_pickle=True)
    if "retrained_checkpoint" not in z.files:
        raise RuntimeError(
            "Stage-2 requires a post-node-retraining topology. "
            f"Missing 'retrained_checkpoint' marker in {path}. "
            "Run scripts/apply_checkpoint_to_topology.py first."
        )


def controller_object_seeds(data: TopologyData, num_object_points: int) -> np.ndarray:
    """Return object-side endpoints of controller springs, independent of edge order."""
    seeds: list[int] = []
    for a_raw, b_raw in np.asarray(data.controller_springs, dtype=np.int64):
        a, b = int(a_raw), int(b_raw)
        a_obj = 0 <= a < num_object_points
        b_obj = 0 <= b < num_object_points
        if a_obj and not b_obj:
            seeds.append(a)
        elif b_obj and not a_obj:
            seeds.append(b)
    return np.unique(np.asarray(seeds, dtype=np.int64)) if seeds else np.empty(0, dtype=np.int64)


def compute_bt_node_scores_post_retrain(
    topology_path: str | Path,
    *,
    local_budget: int = 300,
    max_input_nodes: int = 2,
    reduced_order: int = 20,
    damping: float = 1e-1,
    anchor: float = 1e-3,
    feedthrough: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute BT importance on the retrained node-coarsened dynamics."""
    require_retrained_topology(topology_path)
    data = load_topology(topology_path)
    n_obj = infer_coarse_object_points(topology_path, data)

    if len(data.object_springs) and int(np.max(data.object_springs)) >= n_obj:
        raise RuntimeError(
            f"Object-edge index exceeds coarse object-node count: max="
            f"{int(np.max(data.object_springs))}, n_obj={n_obj}"
        )

    seed_nodes = controller_object_seeds(data, n_obj)
    if not len(seed_nodes):
        degree = np.zeros(n_obj, dtype=np.int64)
        for i, j in data.object_springs:
            degree[int(i)] += 1
            degree[int(j)] += 1
        seed_nodes = np.argsort(-degree)[:max_input_nodes]

    local_nodes = bfs_local_nodes(
        n_obj,
        data.object_springs,
        seed_nodes,
        max_nodes=local_budget,
    )
    input_nodes = [int(x) for x in seed_nodes[:max_input_nodes]]

    A, B, C, D, local_map = build_local_lti_from_topology(
        points_full=data.points_full[:n_obj],
        masses=data.masses[:n_obj],
        object_springs=data.object_springs,
        object_rest_lengths=data.object_rest_lengths,
        object_spring_Y=data.object_spring_Y,
        local_nodes=local_nodes,
        input_nodes=input_nodes,
        damping=damping,
        anchor=anchor,
        feedthrough=feedthrough,
    )
    method, state_scores = _reduce_with_prbt_or_bt(A, B, C, D, reduced_order)
    local_scores = normalize_score(
        state_scores_to_node_scores(state_scores, len(local_nodes))
    )
    global_scores = np.zeros(n_obj, dtype=np.float64)
    for global_node, local_idx in local_map.items():
        global_scores[int(global_node)] = local_scores[int(local_idx)]

    return global_scores, {
        "guidance_method": method,
        "num_object_points": n_obj,
        "local_budget": int(local_budget),
        "max_input_nodes": int(max_input_nodes),
        "reduced_order": int(reduced_order),
        "damping": float(damping),
        "anchor": float(anchor),
        "feedthrough": float(feedthrough),
        "local_node_count": int(len(local_nodes)),
        "input_nodes": input_nodes,
        "seed_nodes": seed_nodes,
        "local_nodes": local_nodes,
    }


def build_online_bt_scores_post_retrain(
    topology_path: str | Path,
    coarse_node_error: np.ndarray,
    *,
    bt_weight: float = 0.7,
    online_error_weight: float = 0.3,
    local_budget: int = 300,
    reduced_order: int = 20,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fuse BT, retrained stiffness and online residual on the coarse graph."""
    if not 0.0 <= bt_weight <= 1.0:
        raise ValueError("bt_weight must be in [0, 1]")
    if not 0.0 <= online_error_weight <= 1.0:
        raise ValueError("online_error_weight must be in [0, 1]")

    require_retrained_topology(topology_path)
    data = load_topology(topology_path)
    n_obj = infer_coarse_object_points(topology_path, data)
    err = np.asarray(coarse_node_error, dtype=np.float64).reshape(-1)
    if len(err) != n_obj:
        raise ValueError(f"coarse_node_error length={len(err)} but n_obj={n_obj}")

    i = data.object_springs[:, 0].astype(np.int64)
    j = data.object_springs[:, 1].astype(np.int64)

    stiffness = normalize_score(
        data.object_spring_Y / np.maximum(data.object_rest_lengths, 1e-8)
    )
    err_n = normalize_score(err)
    edge_err = normalize_score(0.5 * (err_n[i] + err_n[j]))

    bt_node, bt_info = compute_bt_node_scores_post_retrain(
        topology_path,
        local_budget=local_budget,
        reduced_order=reduced_order,
    )
    bt_edge = normalize_score(0.5 * (bt_node[i] + bt_node[j]))

    prior = normalize_score(bt_weight * bt_edge + (1.0 - bt_weight) * stiffness)
    final = normalize_score(
        (1.0 - online_error_weight) * prior + online_error_weight * edge_err
    )
    return final, {
        "stiffness_edge_score": stiffness,
        "bt_node_score": bt_node,
        "bt_edge_score": bt_edge,
        "online_edge_error": edge_err,
        "prior_edge_score": prior,
        "final_edge_score": final,
        "bt_info": bt_info,
    }


def compute_final_budget(
    full_topology_path: str | Path,
    retrained_coarse_topology_path: str | Path,
    target_final_ratio: float,
) -> FinalBudget:
    if not 0.0 < target_final_ratio <= 1.0:
        raise ValueError("target_final_ratio must be in (0, 1]")
    require_retrained_topology(retrained_coarse_topology_path)

    full = load_topology(full_topology_path)
    coarse = load_topology(retrained_coarse_topology_path)
    full_total = int(len(full.springs))
    stage1_total = int(len(coarse.springs))
    stage1_obj = int(len(coarse.object_springs))
    stage1_ctl = int(len(coarse.controller_springs))

    target_total = int(round(full_total * float(target_final_ratio)))
    if target_total > stage1_total:
        raise ValueError(
            f"Target final springs {target_total} exceeds Stage-1 reduced springs "
            f"{stage1_total}; Stage-2 only prunes and cannot add springs."
        )
    target_obj = target_total - stage1_ctl
    if target_obj < 1:
        raise ValueError(
            f"Target final budget leaves {target_obj} object springs after preserving "
            f"{stage1_ctl} controller springs."
        )
    keep = target_obj / max(stage1_obj, 1)
    if keep > 1.0 + 1e-12:
        raise ValueError(f"Computed Stage-2 keep ratio > 1: {keep}")

    return FinalBudget(
        full_total_springs=full_total,
        stage1_total_springs=stage1_total,
        stage1_object_springs=stage1_obj,
        stage1_controller_springs=stage1_ctl,
        target_final_ratio=float(target_final_ratio),
        target_total_springs=target_total,
        target_object_springs=target_obj,
        stage2_object_keep_ratio=float(min(1.0, keep)),
    )


def _prune_indices(
    n_obj: int,
    object_springs: np.ndarray,
    scores: np.ndarray,
    target_object_springs: int,
    min_degree: int,
) -> np.ndarray:
    target = max(int(target_object_springs), max(0, n_obj - 1))
    mandatory = maximum_spanning_forest_indices(n_obj, object_springs, scores)
    keep = add_min_degree_edges(
        n_obj, object_springs, scores, mandatory, int(min_degree)
    )
    keep_set = set(int(x) for x in np.asarray(keep).tolist())
    remaining = np.asarray(
        [idx for idx in range(len(object_springs)) if idx not in keep_set],
        dtype=np.int64,
    )
    if len(remaining):
        remaining = remaining[np.argsort(scores[remaining])[::-1]]
    for idx in remaining[: max(0, target - len(keep_set))]:
        keep_set.add(int(idx))
    return np.asarray(sorted(keep_set), dtype=np.int64)


def generate_final_budget_topology(
    full_topology_path: str | Path,
    retrained_coarse_topology_path: str | Path,
    coarse_node_error: np.ndarray,
    output_path: str | Path,
    *,
    target_final_ratio: float,
    bt_weight: float = 0.7,
    online_error_weight: float = 0.3,
    min_degree: int = 1,
    local_budget: int = 300,
    reduced_order: int = 20,
) -> tuple[Path, dict[str, Any]]:
    """Generate Stage-2 topology at a FINAL budget relative to the full graph."""
    full_topology_path = Path(full_topology_path).expanduser().resolve()
    coarse_path = Path(retrained_coarse_topology_path).expanduser().resolve()
    out = Path(output_path).expanduser().resolve()

    budget = compute_final_budget(full_topology_path, coarse_path, target_final_ratio)
    data = load_topology(coarse_path)
    n_obj = infer_coarse_object_points(coarse_path, data)

    score, detail = build_online_bt_scores_post_retrain(
        coarse_path,
        coarse_node_error,
        bt_weight=bt_weight,
        online_error_weight=online_error_weight,
        local_budget=local_budget,
        reduced_order=reduced_order,
    )
    keep_indices = _prune_indices(
        n_obj,
        data.object_springs,
        score,
        budget.target_object_springs,
        min_degree,
    )

    object_springs = data.object_springs[keep_indices]
    object_rest = data.object_rest_lengths[keep_indices]
    object_y = data.object_spring_Y[keep_indices]
    stats = graph_stats(n_obj, object_springs)

    if stats["components"] != 1:
        raise RuntimeError(f"Pruned object graph disconnected: {stats}")
    if stats["isolated"] != 0:
        raise RuntimeError(f"Pruned object graph has isolated nodes: {stats}")
    if stats["min_degree"] < int(min_degree):
        raise RuntimeError(f"Pruned object graph violates min_degree={min_degree}: {stats}")

    all_springs = np.concatenate([object_springs, data.controller_springs], axis=0)
    all_rest = np.concatenate([object_rest, data.controller_rest_lengths], axis=0)
    all_y = np.concatenate([object_y, data.controller_spring_Y], axis=0)

    raw = np.load(coarse_path, allow_pickle=True)
    arrays: dict[str, Any] = {k: raw[k] for k in raw.files}
    arrays.update(
        {
            "points_full": data.points_full,
            "springs": all_springs,
            "rest_lengths": all_rest,
            "masses": data.masses,
            "spring_Y": all_y,
            "num_object_springs": np.asarray(len(object_springs), dtype=np.int64),
            "kept_stage1_object_indices": keep_indices,
            "object_edge_scores": score,
            "stiffness_edge_score": detail["stiffness_edge_score"],
            "bt_node_score": detail["bt_node_score"],
            "bt_edge_score": detail["bt_edge_score"],
            "online_edge_error": detail["online_edge_error"],
            "prior_edge_score": detail["prior_edge_score"],
            "final_edge_score": detail["final_edge_score"],
            "stage2_target_final_ratio": np.asarray(target_final_ratio),
            "stage2_actual_final_ratio": np.asarray(len(all_springs) / budget.full_total_springs),
            "stage2_object_keep_ratio": np.asarray(len(object_springs) / budget.stage1_object_springs),
            "stage2_bt_weight": np.asarray(bt_weight),
            "stage2_online_error_weight": np.asarray(online_error_weight),
            "stage2_min_degree": np.asarray(min_degree),
            "stage2_protocol": np.asarray("post_node_retrain_final_budget_online_bt"),
        }
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)

    scalar_bt = {
        k: v
        for k, v in detail["bt_info"].items()
        if np.isscalar(v) or isinstance(v, (str, bool))
    }
    meta = {
        "protocol": "post_node_retrain_final_budget_online_bt",
        "full_topology": str(full_topology_path),
        "source_retrained_coarse_topology": str(coarse_path),
        "retrained_checkpoint": str(np.asarray(raw["retrained_checkpoint"]).item()),
        "target_final_ratio": float(target_final_ratio),
        "target_total_springs": budget.target_total_springs,
        "actual_total_springs": int(len(all_springs)),
        "actual_final_ratio": float(len(all_springs) / budget.full_total_springs),
        "stage1_total_springs": budget.stage1_total_springs,
        "stage1_object_springs": budget.stage1_object_springs,
        "controller_springs_preserved": budget.stage1_controller_springs,
        "pruned_object_springs": int(len(object_springs)),
        "stage2_object_keep_ratio": float(len(object_springs) / budget.stage1_object_springs),
        "bt_weight": float(bt_weight),
        "online_error_weight": float(online_error_weight),
        "min_degree": int(min_degree),
        "graph_stats": stats,
        **scalar_bt,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, meta
