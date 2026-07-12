from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TopologyData:
    points_full: np.ndarray
    springs: np.ndarray
    rest_lengths: np.ndarray
    masses: np.ndarray
    spring_Y: np.ndarray
    num_object_springs: int

    @property
    def object_springs(self) -> np.ndarray:
        return self.springs[: self.num_object_springs]

    @property
    def controller_springs(self) -> np.ndarray:
        return self.springs[self.num_object_springs :]

    @property
    def object_rest_lengths(self) -> np.ndarray:
        return self.rest_lengths[: self.num_object_springs]

    @property
    def controller_rest_lengths(self) -> np.ndarray:
        return self.rest_lengths[self.num_object_springs :]

    @property
    def object_spring_Y(self) -> np.ndarray:
        return self.spring_Y[: self.num_object_springs]

    @property
    def controller_spring_Y(self) -> np.ndarray:
        return self.spring_Y[self.num_object_springs :]


def load_topology(path: str | Path) -> TopologyData:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Topology file not found: {path}")
    topo = np.load(path, allow_pickle=True)
    required = {
        "points_full", "springs", "rest_lengths",
        "masses", "spring_Y", "num_object_springs",
    }
    missing = sorted(required.difference(topo.files))
    if missing:
        raise KeyError(f"{path} is missing arrays: {missing}")
    data = TopologyData(
        points_full=np.asarray(topo["points_full"]),
        springs=np.asarray(topo["springs"]),
        rest_lengths=np.asarray(topo["rest_lengths"]),
        masses=np.asarray(topo["masses"]),
        spring_Y=np.asarray(topo["spring_Y"]),
        num_object_springs=int(topo["num_object_springs"]),
    )
    if data.springs.ndim != 2 or data.springs.shape[1] != 2:
        raise ValueError(f"springs must have shape [E, 2], got {data.springs.shape}")
    if len(data.rest_lengths) != len(data.springs):
        raise ValueError("rest_lengths length does not match springs")
    if len(data.spring_Y) != len(data.springs):
        raise ValueError("spring_Y length does not match springs")
    return data


def normalize_score(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    if not np.all(np.isfinite(x)):
        raise ValueError("Score contains NaN or infinity")
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn + eps)


def infer_num_object_points(data: TopologyData) -> int:
    if len(data.controller_springs):
        return int(np.min(data.controller_springs[:, 0]))
    if len(data.object_springs):
        return int(np.max(data.object_springs)) + 1
    return len(data.points_full)


def _edge_tuple(edge: np.ndarray | tuple[int, int]) -> tuple[int, int]:
    i, j = int(edge[0]), int(edge[1])
    return (i, j) if i <= j else (j, i)


def _component_labels(num_nodes: int, springs: np.ndarray) -> tuple[int, np.ndarray]:
    if len(springs) == 0:
        return num_nodes, np.arange(num_nodes, dtype=np.int64)
    i = springs[:, 0].astype(np.int64)
    j = springs[:, 1].astype(np.int64)
    graph = coo_matrix(
        (
            np.ones(len(i) * 2, dtype=np.float64),
            (np.concatenate([i, j]), np.concatenate([j, i])),
        ),
        shape=(num_nodes, num_nodes),
    )
    return connected_components(graph, directed=False)


def graph_stats(num_nodes: int, springs: np.ndarray) -> dict[str, float | int]:
    degree = np.zeros(num_nodes, dtype=np.int64)
    for i, j in np.asarray(springs):
        degree[int(i)] += 1
        degree[int(j)] += 1
    components, _ = _component_labels(num_nodes, np.asarray(springs))
    return {
        "components": int(components),
        "isolated": int(np.sum(degree == 0)),
        "min_degree": int(degree.min()) if len(degree) else 0,
        "degree_le_1": int(np.sum(degree <= 1)),
        "degree_le_2": int(np.sum(degree <= 2)),
        "mean_degree": float(degree.mean()) if len(degree) else 0.0,
        "max_degree": int(degree.max()) if len(degree) else 0,
    }


def maximum_spanning_forest_indices(
    num_nodes: int,
    object_springs: np.ndarray,
    edge_scores: np.ndarray,
) -> np.ndarray:
    scores = normalize_score(edge_scores)
    i = object_springs[:, 0].astype(np.int64)
    j = object_springs[:, 1].astype(np.int64)
    weights = (scores.max() - scores) + 1e-6
    graph = coo_matrix((weights, (i, j)), shape=(num_nodes, num_nodes))
    forest = minimum_spanning_tree(graph + graph.T).tocoo()
    edge_to_idx = {_edge_tuple(edge): idx for idx, edge in enumerate(object_springs)}
    keep = []
    for a, b in zip(forest.row, forest.col):
        idx = edge_to_idx.get(_edge_tuple((int(a), int(b))))
        if idx is not None:
            keep.append(idx)
    return np.asarray(sorted(set(keep)), dtype=np.int64)


def add_min_degree_edges(
    num_nodes: int,
    object_springs: np.ndarray,
    edge_scores: np.ndarray,
    keep_indices: np.ndarray,
    min_degree: int,
) -> np.ndarray:
    keep = set(int(x) for x in np.asarray(keep_indices).tolist())
    incident: list[list[int]] = [[] for _ in range(num_nodes)]
    for idx, (i, j) in enumerate(object_springs):
        incident[int(i)].append(idx)
        incident[int(j)].append(idx)

    while True:
        degree = np.zeros(num_nodes, dtype=np.int64)
        for idx in keep:
            i, j = object_springs[idx]
            degree[int(i)] += 1
            degree[int(j)] += 1
        changed = False
        for node in np.where(degree < min_degree)[0]:
            candidates = [idx for idx in incident[int(node)] if idx not in keep]
            if candidates:
                keep.add(max(candidates, key=lambda idx: float(edge_scores[idx])))
                changed = True
        if not changed:
            break
    return np.asarray(sorted(keep), dtype=np.int64)


def add_bridge_springs(
    points: np.ndarray,
    springs: np.ndarray,
    rest_lengths: np.ndarray,
    spring_y: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n_components, labels = _component_labels(num_nodes, springs)
    if n_components <= 1:
        return springs, rest_lengths, spring_y, 0

    default_y = float(np.median(spring_y)) if len(spring_y) else 1.0
    new_springs = springs.astype(np.int64).tolist()
    new_rest = rest_lengths.astype(np.float64).tolist()
    new_y = spring_y.astype(np.float64).tolist()
    component_ids, counts = np.unique(labels, return_counts=True)
    main_component = int(component_ids[np.argmax(counts)])
    main_nodes = np.where(labels == main_component)[0]
    tree = cKDTree(points[main_nodes])
    added = 0

    for component in component_ids:
        if int(component) == main_component:
            continue
        nodes = np.where(labels == component)[0]
        distances, nearest = tree.query(points[nodes], k=1)
        local_idx = int(np.argmin(distances))
        a = int(nodes[local_idx])
        b = int(main_nodes[int(nearest[local_idx])])
        new_springs.append([a, b])
        new_rest.append(float(np.linalg.norm(points[a] - points[b])))
        new_y.append(default_y)
        added += 1
        main_nodes = np.concatenate([main_nodes, nodes])
        tree = cKDTree(points[main_nodes])

    return (
        np.asarray(new_springs, dtype=np.int64),
        np.asarray(new_rest, dtype=np.float64),
        np.asarray(new_y, dtype=np.float64),
        added,
    )


def generate_connected_pruned_topology(
    topology_path: str | Path,
    out_path: str | Path,
    object_edge_scores: np.ndarray,
    *,
    keep_ratio: float = 0.5,
    min_degree: int = 1,
    allow_bridge: bool = False,
    method: str = "custom",
    extra_metadata: Mapping[str, Any] | None = None,
    extra_arrays: Mapping[str, np.ndarray] | None = None,
) -> tuple[Path, dict[str, Any]]:
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")

    data = load_topology(topology_path)
    num_object_points = infer_num_object_points(data)
    scores = np.asarray(object_edge_scores, dtype=np.float64)
    if len(scores) != len(data.object_springs):
        raise ValueError("object_edge_scores length does not match object springs")

    target_keep = max(
        int(len(data.object_springs) * keep_ratio),
        max(0, num_object_points - 1),
    )
    mandatory = maximum_spanning_forest_indices(
        num_object_points, data.object_springs, scores
    )
    keep_indices = add_min_degree_edges(
        num_object_points,
        data.object_springs,
        scores,
        mandatory,
        min_degree,
    )
    keep_set = set(int(x) for x in keep_indices.tolist())
    remaining = np.asarray(
        [idx for idx in range(len(data.object_springs)) if idx not in keep_set],
        dtype=np.int64,
    )
    if len(remaining):
        remaining = remaining[np.argsort(scores[remaining])[::-1]]
    for idx in remaining[: max(0, target_keep - len(keep_set))]:
        keep_set.add(int(idx))
    keep_indices = np.asarray(sorted(keep_set), dtype=np.int64)

    object_springs = data.object_springs[keep_indices]
    object_rest = data.object_rest_lengths[keep_indices]
    object_y = data.object_spring_Y[keep_indices]
    before_stats = graph_stats(num_object_points, object_springs)

    bridge_added = 0
    if before_stats["components"] > 1 and allow_bridge:
        object_springs, object_rest, object_y, bridge_added = add_bridge_springs(
            data.points_full[:num_object_points],
            object_springs,
            object_rest,
            object_y,
            num_object_points,
        )
    after_stats = graph_stats(num_object_points, object_springs)

    all_springs = np.concatenate([object_springs, data.controller_springs], axis=0)
    all_rest = np.concatenate([object_rest, data.controller_rest_lengths], axis=0)
    all_y = np.concatenate([object_y, data.controller_spring_Y], axis=0)

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "points_full": data.points_full,
        "springs": all_springs,
        "rest_lengths": all_rest,
        "masses": data.masses,
        "spring_Y": all_y,
        "num_object_springs": np.asarray(len(object_springs), dtype=np.int64),
        "keep_ratio": np.asarray(keep_ratio),
        "min_degree": np.asarray(min_degree),
        "allow_bridge": np.asarray(allow_bridge),
        "bridge_added": np.asarray(bridge_added),
        "method": np.asarray(method),
        "kept_original_object_indices": keep_indices,
        "object_edge_scores": scores,
    }
    if extra_arrays:
        arrays.update({key: np.asarray(value) for key, value in extra_arrays.items()})
    np.savez_compressed(out_path, **arrays)

    metadata: dict[str, Any] = {
        "source_topology": str(Path(topology_path).expanduser().resolve()),
        "output_topology": str(out_path),
        "method": method,
        "keep_ratio": keep_ratio,
        "min_degree": min_degree,
        "allow_bridge": allow_bridge,
        "bridge_added": bridge_added,
        "num_object_points": num_object_points,
        "original_object_springs": len(data.object_springs),
        "pruned_object_springs": len(object_springs),
        "controller_springs_kept": len(data.controller_springs),
        "total_springs": len(all_springs),
        "before_stats": before_stats,
        "after_stats": after_stats,
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    out_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return out_path, metadata
