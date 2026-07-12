from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .topology import TopologyData, infer_num_object_points, normalize_score


def build_object_adjacency(
    num_object_points: int, object_springs: np.ndarray
) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(num_object_points)]
    for i, j in object_springs:
        a, b = int(i), int(j)
        adjacency[a].append(b)
        adjacency[b].append(a)
    return adjacency


def bfs_local_nodes(
    num_object_points: int,
    object_springs: np.ndarray,
    seed_nodes: np.ndarray,
    max_nodes: int = 200,
) -> np.ndarray:
    adjacency = build_object_adjacency(num_object_points, object_springs)
    seeds = [int(x) for x in seed_nodes if 0 <= int(x) < num_object_points]
    if not seeds:
        raise ValueError("No valid seed nodes were found")
    visited = set(seeds)
    queue: deque[int] = deque(seeds)
    order = list(seeds)

    while queue and len(order) < max_nodes:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
            order.append(neighbor)
            if len(order) >= max_nodes:
                break
    return np.asarray(order, dtype=np.int64)


def build_local_lti_from_topology(
    *,
    points_full: np.ndarray,
    masses: np.ndarray,
    object_springs: np.ndarray,
    object_rest_lengths: np.ndarray,
    object_spring_Y: np.ndarray,
    local_nodes: np.ndarray,
    input_nodes: list[int],
    damping: float = 1e-1,
    anchor: float = 1e-3,
    feedthrough: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    local_nodes = np.asarray(local_nodes, dtype=np.int64)
    n_local = len(local_nodes)
    if n_local == 0:
        raise ValueError("Local BT subgraph is empty")

    dof = 3 * n_local
    local_map = {
        int(global_idx): local_idx
        for local_idx, global_idx in enumerate(local_nodes)
    }
    mask = np.asarray(
        [
            int(i) in local_map and int(j) in local_map
            for i, j in object_springs
        ],
        dtype=bool,
    )
    springs_local = object_springs[mask]
    rest_local = object_rest_lengths[mask]
    y_local = object_spring_Y[mask]

    mass_local = np.asarray(masses)[local_nodes]
    mass_matrix = np.zeros((dof, dof), dtype=np.float64)
    for local_idx, mass in enumerate(mass_local):
        block = slice(3 * local_idx, 3 * local_idx + 3)
        mass_matrix[block, block] = np.eye(3) * max(float(mass), 1e-8)

    stiffness = np.zeros((dof, dof), dtype=np.float64)
    for (global_i, global_j), rest_length, spring_y in zip(
        springs_local, rest_local, y_local
    ):
        global_i, global_j = int(global_i), int(global_j)
        local_i = local_map[global_i]
        local_j = local_map[global_j]
        direction = points_full[global_j] - points_full[global_i]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        unit = direction / norm
        edge_stiffness = (
            float(spring_y) / max(float(rest_length), 1e-8)
        ) * np.outer(unit, unit)

        block_i = slice(3 * local_i, 3 * local_i + 3)
        block_j = slice(3 * local_j, 3 * local_j + 3)
        stiffness[block_i, block_i] += edge_stiffness
        stiffness[block_j, block_j] += edge_stiffness
        stiffness[block_i, block_j] -= edge_stiffness
        stiffness[block_j, block_i] -= edge_stiffness

    stiffness += anchor * np.eye(dof)
    damping_matrix = damping * np.eye(dof)
    inverse_mass = np.linalg.inv(mass_matrix + 1e-10 * np.eye(dof))

    system_a = np.block(
        [
            [np.zeros((dof, dof)), np.eye(dof)],
            [-inverse_mass @ stiffness, -inverse_mass @ damping_matrix],
        ]
    )

    valid_inputs = [int(node) for node in input_nodes if int(node) in local_map]
    input_dim = 3 * len(valid_inputs)
    if input_dim == 0:
        raise ValueError("No valid input nodes remain in the local BT subgraph")

    input_position = np.zeros((dof, input_dim), dtype=np.float64)
    input_velocity = np.zeros((dof, input_dim), dtype=np.float64)
    for input_idx, global_node in enumerate(valid_inputs):
        local_idx = local_map[global_node]
        rows = slice(3 * local_idx, 3 * local_idx + 3)
        cols = slice(3 * input_idx, 3 * input_idx + 3)
        input_velocity[rows, cols] = inverse_mass[rows, rows]
    system_b = np.vstack([input_position, input_velocity])

    system_c = np.zeros((input_dim, 2 * dof), dtype=np.float64)
    for output_idx, global_node in enumerate(valid_inputs):
        local_idx = local_map[global_node]
        rows = slice(3 * output_idx, 3 * output_idx + 3)
        cols = slice(3 * local_idx, 3 * local_idx + 3)
        system_c[rows, cols] = np.eye(3)

    system_d = feedthrough * np.eye(input_dim)
    return system_a, system_b, system_c, system_d, local_map


def _projection_state_scores(reductor: Any, state_dim: int) -> np.ndarray:
    if not hasattr(reductor, "V") or reductor.V is None:
        raise RuntimeError("The pyMOR reductor did not expose projection basis V")
    basis = reductor.V.to_numpy()
    if basis.shape[0] != state_dim and basis.shape[1] == state_dim:
        basis = basis.T
    if basis.shape[0] != state_dim:
        raise RuntimeError(
            f"Unexpected projection basis shape {basis.shape}; state_dim={state_dim}"
        )
    return np.sum(basis**2, axis=1)


def _reduce_with_prbt_or_bt(
    system_a: np.ndarray,
    system_b: np.ndarray,
    system_c: np.ndarray,
    system_d: np.ndarray,
    reduced_order: int,
) -> tuple[str, np.ndarray]:
    try:
        from pymor.models.iosys import LTIModel
        from pymor.reductors.bt import BTReductor, PRBTReductor
    except ImportError as exc:
        raise RuntimeError(
            "pyMOR is required for BT-guided pruning. Install it with "
            "`pip install pymor`."
        ) from exc

    full_model = LTIModel.from_matrices(
        system_a, system_b, system_c, system_d
    )
    state_dim = system_a.shape[0]
    safe_order = max(1, min(int(reduced_order), state_dim - 1))

    try:
        reductor = PRBTReductor(full_model)
        reductor.reduce(safe_order)
        return "PRBT", _projection_state_scores(reductor, state_dim)
    except Exception as prbt_error:
        print(f"[BT] PRBT failed; falling back to BT: {prbt_error!r}")

    reductor = BTReductor(full_model)
    reductor.reduce(safe_order)
    return "BT", _projection_state_scores(reductor, state_dim)


def state_scores_to_node_scores(
    state_scores: np.ndarray, n_local_nodes: int
) -> np.ndarray:
    node_scores = np.zeros(n_local_nodes, dtype=np.float64)
    for node_idx in range(n_local_nodes):
        q_score = state_scores[3 * node_idx : 3 * node_idx + 3].sum()
        v_start = 3 * n_local_nodes + 3 * node_idx
        v_score = state_scores[v_start : v_start + 3].sum()
        node_scores[node_idx] = q_score + v_score
    return node_scores


def compute_bt_node_scores(
    data: TopologyData,
    *,
    local_budget: int = 200,
    max_input_nodes: int = 2,
    reduced_order: int = 20,
    damping: float = 1e-1,
    anchor: float = 1e-3,
    feedthrough: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    num_object_points = infer_num_object_points(data)
    if len(data.controller_springs):
        seed_nodes = np.unique(
            data.controller_springs[:, 1].astype(np.int64)
        )
    else:
        degree = np.zeros(num_object_points, dtype=np.int64)
        for i, j in data.object_springs:
            degree[int(i)] += 1
            degree[int(j)] += 1
        seed_nodes = np.argsort(-degree)[:max_input_nodes]

    local_nodes = bfs_local_nodes(
        num_object_points,
        data.object_springs,
        seed_nodes,
        max_nodes=local_budget,
    )
    input_nodes = [int(x) for x in seed_nodes[:max_input_nodes]]

    system_a, system_b, system_c, system_d, local_map = (
        build_local_lti_from_topology(
            points_full=data.points_full[:num_object_points],
            masses=data.masses[:num_object_points],
            object_springs=data.object_springs,
            object_rest_lengths=data.object_rest_lengths,
            object_spring_Y=data.object_spring_Y,
            local_nodes=local_nodes,
            input_nodes=input_nodes,
            damping=damping,
            anchor=anchor,
            feedthrough=feedthrough,
        )
    )
    method, state_scores = _reduce_with_prbt_or_bt(
        system_a,
        system_b,
        system_c,
        system_d,
        reduced_order,
    )
    local_node_scores = normalize_score(
        state_scores_to_node_scores(state_scores, len(local_nodes))
    )
    global_scores = np.zeros(num_object_points, dtype=np.float64)
    for global_node, local_idx in local_map.items():
        global_scores[global_node] = local_node_scores[local_idx]

    return global_scores, {
        "guidance_method": method,
        "local_budget": local_budget,
        "max_input_nodes": max_input_nodes,
        "reduced_order": reduced_order,
        "damping": damping,
        "anchor": anchor,
        "feedthrough": feedthrough,
        "local_node_count": len(local_nodes),
        "input_nodes": input_nodes,
        "local_nodes": local_nodes,
        "seed_nodes": seed_nodes,
    }
