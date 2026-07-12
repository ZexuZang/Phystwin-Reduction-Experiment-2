from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .topology import TopologyData, infer_num_object_points, load_topology, normalize_score


def compute_online_node_error(
    inference_path: str | Path,
    final_data_path: str | Path,
    topology_path: str | Path,
    *,
    online_start: int,
    online_end: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Assign visible GT point errors to nearest predicted object nodes."""
    inference_path = Path(inference_path).expanduser().resolve()
    final_data_path = Path(final_data_path).expanduser().resolve()
    topology_path = Path(topology_path).expanduser().resolve()

    with inference_path.open("rb") as handle:
        predicted = np.asarray(pickle.load(handle), dtype=np.float64)
    with final_data_path.open("rb") as handle:
        data = pickle.load(handle)

    object_points = np.asarray(data["object_points"], dtype=np.float64)
    object_visibilities = np.asarray(data["object_visibilities"])
    topology = load_topology(topology_path)
    num_object_points = infer_num_object_points(topology)

    node_error = np.zeros(num_object_points, dtype=np.float64)
    node_count = np.zeros(num_object_points, dtype=np.float64)

    start = max(0, int(online_start))
    end = min(int(online_end), len(predicted), len(object_points))
    if end <= start:
        raise ValueError(
            f"Online frame range is empty after clipping: start={start}, end={end}"
        )

    valid_frames = 0
    for frame_idx in range(start, end):
        pred_object = predicted[frame_idx, :num_object_points]
        gt_visible = object_points[frame_idx][object_visibilities[frame_idx]]
        if len(pred_object) == 0 or len(gt_visible) == 0:
            continue
        distances, indices = cKDTree(pred_object).query(gt_visible, k=1)
        np.add.at(node_error, indices.astype(np.int64), distances.astype(np.float64))
        np.add.at(node_count, indices.astype(np.int64), 1.0)
        valid_frames += 1

    averaged = node_error / np.maximum(node_count, 1.0)
    normalized = normalize_score(averaged)
    metadata = {
        "online_start": start,
        "online_end": end,
        "valid_frames": valid_frames,
        "num_object_points": num_object_points,
        "nonzero_nodes": int(np.count_nonzero(node_count)),
    }
    return averaged, normalized, metadata


def _empty_edges(dtype: np.dtype = np.int64) -> np.ndarray:
    return np.empty((0, 2), dtype=dtype)


def generate_node_cluster_topology(
    topology_path: str | Path,
    inference_path: str | Path,
    output_path: str | Path,
    *,
    online_start: int,
    online_end: int,
    node_keep_ratio: float = 0.8,
    max_cluster_size: int = 8,
    node_error: np.ndarray | None = None,
    protect_controller_attached: bool = True,
    protect_high_error: bool = True,
    high_error_percentile: float = 90.0,
) -> tuple[Path, dict[str, Any]]:
    """Merge neighboring object nodes with similar online displacement trajectories."""
    if not 0 < node_keep_ratio <= 1:
        raise ValueError("node_keep_ratio must be in (0, 1]")
    if max_cluster_size < 1:
        raise ValueError("max_cluster_size must be >= 1")

    output_path = Path(output_path).expanduser().resolve()
    topology = load_topology(topology_path)
    with Path(inference_path).expanduser().resolve().open("rb") as handle:
        vertices = np.asarray(pickle.load(handle), dtype=np.float64)

    num_object_points = infer_num_object_points(topology)
    num_total_points = topology.points_full.shape[0]
    object_points = topology.points_full[:num_object_points].astype(np.float64)
    controller_points = topology.points_full[num_object_points:].astype(np.float64)
    object_masses = topology.masses[:num_object_points].astype(np.float64)
    controller_masses = topology.masses[num_object_points:].astype(np.float64)

    object_springs = topology.object_springs.astype(np.int64)
    controller_springs = topology.controller_springs.astype(np.int64)
    object_rest = topology.object_rest_lengths.astype(np.float64)
    controller_rest = topology.controller_rest_lengths.astype(np.float64)
    object_y = topology.object_spring_Y.astype(np.float64)
    controller_y = topology.controller_spring_Y.astype(np.float64)

    if vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError(f"Expected inference shape [T, N, 3], got {vertices.shape}")
    if vertices.shape[1] < num_object_points:
        raise ValueError(
            f"Inference has {vertices.shape[1]} nodes but topology needs "
            f"{num_object_points} object nodes"
        )

    start = max(0, int(online_start))
    end = min(int(online_end), len(vertices))
    if end <= start:
        raise ValueError(
            f"Online frame range is empty after clipping: start={start}, end={end}"
        )
    online_vertices = vertices[start:end, :num_object_points]
    displacement = online_vertices - object_points[None]

    protected = np.zeros(num_object_points, dtype=bool)
    attached_nodes: list[int] = []
    for a, b in controller_springs:
        if int(a) < num_object_points:
            attached_nodes.append(int(a))
        if int(b) < num_object_points:
            attached_nodes.append(int(b))
    attached = np.unique(np.asarray(attached_nodes, dtype=np.int64))
    if protect_controller_attached and len(attached):
        protected[attached] = True

    if node_error is None:
        error_for_protection = np.linalg.norm(displacement, axis=2).mean(axis=0)
    else:
        error_for_protection = np.asarray(node_error, dtype=np.float64)[:num_object_points]
        if len(error_for_protection) != num_object_points:
            raise ValueError("node_error length does not match object node count")

    if protect_high_error:
        threshold = float(np.percentile(error_for_protection, high_error_percentile))
        high_error_nodes = np.where(error_for_protection >= threshold)[0]
        protected[high_error_nodes] = True
    else:
        threshold = float("nan")
        high_error_nodes = np.empty(0, dtype=np.int64)

    edge_dissimilarity = np.zeros(len(object_springs), dtype=np.float64)
    merge_allowed = np.ones(len(object_springs), dtype=bool)
    for edge_idx, (i, j) in enumerate(object_springs):
        i, j = int(i), int(j)
        trajectory_delta = displacement[:, i] - displacement[:, j]
        edge_dissimilarity[edge_idx] = np.linalg.norm(
            trajectory_delta, axis=1
        ).mean()
        if protected[i] or protected[j]:
            merge_allowed[edge_idx] = False

    parent = np.arange(num_object_points, dtype=np.int64)
    cluster_size = np.ones(num_object_points, dtype=np.int64)
    cluster_protected = protected.copy()

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(a: int, b: int) -> bool:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return False
        if cluster_protected[root_a] or cluster_protected[root_b]:
            return False
        if cluster_size[root_a] + cluster_size[root_b] > max_cluster_size:
            return False
        if cluster_size[root_a] < cluster_size[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        cluster_size[root_a] += cluster_size[root_b]
        cluster_protected[root_a] |= cluster_protected[root_b]
        return True

    target_clusters = max(
        int(np.ceil(num_object_points * node_keep_ratio)),
        int(protected.sum()),
    )
    current_clusters = num_object_points
    candidates = np.where(merge_allowed)[0]
    candidates = candidates[np.argsort(edge_dissimilarity[candidates])]

    merges = 0
    for edge_idx in candidates:
        if current_clusters <= target_clusters:
            break
        i, j = object_springs[edge_idx]
        if union(int(i), int(j)):
            current_clusters -= 1
            merges += 1

    roots = np.asarray([find(i) for i in range(num_object_points)], dtype=np.int64)
    unique_roots = sorted(np.unique(roots).tolist())
    root_to_cluster = {root: idx for idx, root in enumerate(unique_roots)}
    object_cluster = np.asarray(
        [root_to_cluster[find(i)] for i in range(num_object_points)],
        dtype=np.int64,
    )
    num_reduced_object = len(unique_roots)

    cluster_members: list[list[int]] = [[] for _ in range(num_reduced_object)]
    for old_idx, cluster_idx in enumerate(object_cluster):
        cluster_members[int(cluster_idx)].append(int(old_idx))
    cluster_sizes = np.asarray([len(x) for x in cluster_members], dtype=np.int64)

    reduced_object_points = np.zeros((num_reduced_object, 3), dtype=np.float64)
    reduced_object_masses = np.zeros(num_reduced_object, dtype=np.float64)
    for cluster_idx, members_list in enumerate(cluster_members):
        members = np.asarray(members_list, dtype=np.int64)
        masses = object_masses[members]
        points = object_points[members]
        total_mass = max(float(masses.sum()), 1e-12)
        reduced_object_masses[cluster_idx] = total_mass
        reduced_object_points[cluster_idx] = (
            points * masses[:, None]
        ).sum(axis=0) / total_mass

    reduced_points = np.concatenate([reduced_object_points, controller_points], axis=0)
    reduced_masses = np.concatenate([reduced_object_masses, controller_masses], axis=0)

    def add_accumulator(
        accumulator: defaultdict[tuple[int, int], float],
        a: int,
        b: int,
        stiffness: float,
    ) -> None:
        if a == b:
            return
        accumulator[tuple(sorted((int(a), int(b))))] += float(stiffness)

    object_edge_stiffness: defaultdict[tuple[int, int], float] = defaultdict(float)
    for (i, j), rest, spring_y in zip(object_springs, object_rest, object_y):
        cluster_i = int(object_cluster[int(i)])
        cluster_j = int(object_cluster[int(j)])
        add_accumulator(
            object_edge_stiffness,
            cluster_i,
            cluster_j,
            float(spring_y) / max(float(rest), 1e-8),
        )

    reduced_object_edges: list[list[int]] = []
    reduced_object_rest: list[float] = []
    reduced_object_y: list[float] = []
    for (a, b), stiffness_sum in object_edge_stiffness.items():
        rest = max(float(np.linalg.norm(reduced_points[a] - reduced_points[b])), 1e-8)
        reduced_object_edges.append([a, b])
        reduced_object_rest.append(rest)
        reduced_object_y.append(stiffness_sum * rest)

    controller_edge_stiffness: defaultdict[tuple[int, int], float] = defaultdict(float)
    for (a, b), rest, spring_y in zip(
        controller_springs, controller_rest, controller_y
    ):
        a, b = int(a), int(b)
        new_a = (
            int(object_cluster[a])
            if a < num_object_points
            else num_reduced_object + (a - num_object_points)
        )
        new_b = (
            int(object_cluster[b])
            if b < num_object_points
            else num_reduced_object + (b - num_object_points)
        )
        add_accumulator(
            controller_edge_stiffness,
            new_a,
            new_b,
            float(spring_y) / max(float(rest), 1e-8),
        )

    reduced_controller_edges: list[list[int]] = []
    reduced_controller_rest: list[float] = []
    reduced_controller_y: list[float] = []
    for (a, b), stiffness_sum in controller_edge_stiffness.items():
        rest = max(float(np.linalg.norm(reduced_points[a] - reduced_points[b])), 1e-8)
        reduced_controller_edges.append([a, b])
        reduced_controller_rest.append(rest)
        reduced_controller_y.append(stiffness_sum * rest)

    object_edges_array = (
        np.asarray(reduced_object_edges, dtype=np.int64).reshape(-1, 2)
        if reduced_object_edges
        else _empty_edges()
    )
    controller_edges_array = (
        np.asarray(reduced_controller_edges, dtype=np.int64).reshape(-1, 2)
        if reduced_controller_edges
        else _empty_edges()
    )
    reduced_springs = np.concatenate([object_edges_array, controller_edges_array], axis=0)
    reduced_rest = np.concatenate(
        [
            np.asarray(reduced_object_rest, dtype=np.float64),
            np.asarray(reduced_controller_rest, dtype=np.float64),
        ]
    )
    reduced_y = np.concatenate(
        [
            np.asarray(reduced_object_y, dtype=np.float64),
            np.asarray(reduced_controller_y, dtype=np.float64),
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points_full=reduced_points,
        springs=reduced_springs,
        rest_lengths=reduced_rest,
        masses=reduced_masses,
        spring_Y=reduced_y,
        num_object_springs=len(object_edges_array),
        reduction_type=np.asarray("trajectory_neighbor_clustering"),
        node_keep_ratio=node_keep_ratio,
        max_cluster_size=max_cluster_size,
        original_num_object_points=num_object_points,
        original_num_total_points=num_total_points,
        object_cluster=object_cluster,
        cluster_sizes=cluster_sizes,
        original_object_points=object_points,
        reduced_object_points=reduced_object_points,
        protected_nodes=protected,
        attached_object_nodes=attached,
        high_error_nodes=high_error_nodes,
        edge_dissimilarity=edge_dissimilarity,
        online_start=start,
        online_end=end,
    )
    metadata = {
        "output_path": str(output_path),
        "original_object_nodes": num_object_points,
        "reduced_object_nodes": num_reduced_object,
        "node_keep_ratio_requested": node_keep_ratio,
        "node_keep_ratio_actual": num_reduced_object / max(num_object_points, 1),
        "original_total_springs": len(topology.springs),
        "reduced_total_springs": len(reduced_springs),
        "original_object_springs": len(object_springs),
        "reduced_object_springs": len(object_edges_array),
        "protected_nodes": int(protected.sum()),
        "merges": merges,
        "online_start": start,
        "online_end": end,
        "high_error_threshold": threshold,
    }
    return output_path, metadata


def reconstruct_full_trajectory(
    reduced_inference_path: str | Path,
    node_cluster_topology_path: str | Path,
    output_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Lift reduced cluster displacements back to the original object nodes."""
    reduced_inference_path = Path(reduced_inference_path).expanduser().resolve()
    node_cluster_topology_path = (
        Path(node_cluster_topology_path).expanduser().resolve()
    )
    output_path = Path(output_path).expanduser().resolve()

    with reduced_inference_path.open("rb") as handle:
        reduced_vertices = np.asarray(pickle.load(handle), dtype=np.float64)
    topology = np.load(node_cluster_topology_path, allow_pickle=True)

    required = {
        "object_cluster",
        "original_object_points",
        "reduced_object_points",
        "original_num_object_points",
        "original_num_total_points",
    }
    missing = required.difference(topology.files)
    if missing:
        raise KeyError(
            f"Node-cluster topology is missing reconstruction arrays: {sorted(missing)}"
        )

    object_cluster = topology["object_cluster"].astype(np.int64)
    original_object_points = topology["original_object_points"].astype(np.float64)
    reduced_object_points = topology["reduced_object_points"].astype(np.float64)
    original_num_object = int(topology["original_num_object_points"])
    original_num_total = int(topology["original_num_total_points"])
    num_reduced_object = len(reduced_object_points)

    if reduced_vertices.ndim != 3 or reduced_vertices.shape[2] != 3:
        raise ValueError(
            f"Expected reduced inference shape [T, N, 3], got {reduced_vertices.shape}"
        )
    if reduced_vertices.shape[1] < num_reduced_object:
        raise ValueError("Reduced inference has fewer nodes than reduced topology")
    if len(object_cluster) != original_num_object:
        raise ValueError("object_cluster does not match original object node count")

    reduced_object_trajectory = reduced_vertices[:, :num_reduced_object]
    reduced_controller_trajectory = reduced_vertices[:, num_reduced_object:]
    cluster_displacement = (
        reduced_object_trajectory - reduced_object_points[None]
    )
    full_object_trajectory = (
        original_object_points[None]
        + cluster_displacement[:, object_cluster]
    )
    full_reconstructed = np.concatenate(
        [full_object_trajectory, reduced_controller_trajectory], axis=1
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(full_reconstructed.astype(np.float32), handle)

    metadata = {
        "output_path": str(output_path),
        "frames": int(full_reconstructed.shape[0]),
        "reconstructed_nodes": int(full_reconstructed.shape[1]),
        "expected_original_nodes": original_num_total,
        "node_count_matches": bool(full_reconstructed.shape[1] == original_num_total),
    }
    return output_path, metadata
