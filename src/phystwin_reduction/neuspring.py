#!/usr/bin/env python3
from __future__ import annotations

"""Reusable NeuSpring utilities for PhysTwin reduction experiments.

The functions in this module are intentionally independent from Colab paths,
a specific scene name, and a specific spring budget.  They are used by the
generic NeuSpring candidate generator and Joint NSF trainer.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree


POINT_KEYS = (
    "points",
    "points_full",
    "object_points",
    "vertices",
    "mass_points",
    "point_xyz",
    "xyz",
    "x",
)
EDGE_KEYS = (
    "springs",
    "edges",
    "object_springs",
    "spring_indices",
    "spring_ij",
    "edge_index",
)
REGION_KEYS = (
    "region_ids",
    "region_id",
    "spring_region_ids",
    "spring_region_id",
    "object_spring_region_ids",
    "object_spring_region_id",
)


def _first_key(data: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in data:
            return key
    return None


def load_topology_npz(path: str | Path) -> Dict[str, np.ndarray]:
    """Load a topology NPZ using common PhysTwin/NeuSpring key aliases."""

    path = Path(path)
    with np.load(path, allow_pickle=True) as raw:
        data = {k: raw[k] for k in raw.files}

    point_key = _first_key(data, POINT_KEYS)
    edge_key = _first_key(data, EDGE_KEYS)
    region_key = _first_key(data, REGION_KEYS)

    if point_key is None:
        raise KeyError(
            f"Cannot find points in {path}; available keys: {sorted(data)}"
        )
    if edge_key is None:
        raise KeyError(
            f"Cannot find springs/edges in {path}; available keys: {sorted(data)}"
        )

    points = np.asarray(data[point_key], dtype=np.float32)
    edges = np.asarray(data[edge_key])
    if edges.ndim == 2 and edges.shape[0] == 2 and edges.shape[1] != 2:
        edges = edges.T
    if edges.ndim != 2 or edges.shape[1] < 2:
        raise ValueError(f"Invalid edge array shape in {path}: {edges.shape}")
    edges = edges[:, :2].astype(np.int64)

    result: Dict[str, np.ndarray] = {
        "points": points,
        "edges": edges,
    }
    if region_key is not None:
        result["region_ids"] = np.asarray(data[region_key], dtype=np.int64).reshape(-1)
    return result


def simple_kmeans_numpy(
    x: np.ndarray,
    k: int,
    *,
    iters: int = 40,
    seed: int = 0,
) -> np.ndarray:
    """Small deterministic-enough k-means without a scikit-learn dependency."""

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"k-means expects a 2-D array, got {x.shape}")
    n = len(x)
    if n == 0:
        raise ValueError("Cannot cluster an empty array")
    k = int(max(1, min(k, n)))

    rng = np.random.default_rng(seed)
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)

    for _ in range(int(iters)):
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            labels = new_labels
            break
        labels = new_labels
        for idx in range(k):
            mask = labels == idx
            if mask.any():
                centers[idx] = x[mask].mean(axis=0)
            else:
                centers[idx] = x[rng.integers(0, n)]
    return labels


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(axis=0, keepdims=True)) / (
        x.std(axis=0, keepdims=True) + 1e-8
    )


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def build_spring_features(
    points: np.ndarray,
    edges: np.ndarray,
    *,
    region_ids: Optional[np.ndarray] = None,
    num_regions: int = 8,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Build the canonical NeuSpring edge features used by the NSF."""

    points = np.asarray(points, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int64)
    p0 = points[edges[:, 0]]
    p1 = points[edges[:, 1]]
    midpoint = 0.5 * (p0 + p1)
    vec = p1 - p0
    length = np.linalg.norm(vec, axis=-1, keepdims=True).astype(np.float32)
    direction = vec / np.maximum(length, 1e-8)

    center = midpoint.mean(axis=0, keepdims=True)
    scale = float(max(np.std(midpoint, axis=0).mean(), 1e-6))
    midpoint_norm = (midpoint - center) / scale
    length_norm = length / max(float(length.mean()), 1e-8)

    if region_ids is None or len(region_ids) != len(edges):
        region_ids = simple_kmeans_numpy(
            midpoint,
            k=num_regions,
            iters=40,
            seed=seed,
        )
    region_ids = np.asarray(region_ids, dtype=np.int64).reshape(-1)
    n_regions = max(int(num_regions), int(region_ids.max()) + 1)

    numeric = np.concatenate(
        [midpoint_norm, length_norm, direction],
        axis=-1,
    ).astype(np.float32)

    return {
        "numeric": numeric,
        "midpoint": midpoint.astype(np.float32),
        "length": length.astype(np.float32),
        "direction": direction.astype(np.float32),
        "region_ids": region_ids,
        "num_regions": np.asarray(n_regions, dtype=np.int64),
        "norm_center": center.astype(np.float32),
        "norm_scale": np.asarray(scale, dtype=np.float32),
    }


class EnhancedNeuralSpringField(nn.Module):
    """Absolute NSF: edge features -> positive spring stiffness."""

    def __init__(
        self,
        numeric_dim: int = 7,
        num_regions: int = 8,
        region_embed_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 4,
        out_dim: int = 1,
        min_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.region_embedding = nn.Embedding(num_regions, region_embed_dim)
        layers: list[nn.Module] = []
        dim = numeric_dim + region_embed_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(dim, hidden_dim), nn.SiLU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.min_value = float(min_value)

    def forward(self, numeric: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        emb = self.region_embedding(region_ids.long())
        return torch.nn.functional.softplus(
            self.net(torch.cat([numeric, emb], dim=-1))
        ) + self.min_value


class JointResidualSpringField(nn.Module):
    """Residual NSF initialized to exactly reproduce a supplied base checkpoint."""

    def __init__(
        self,
        numeric_dim: int,
        num_regions: int,
        region_embed_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.region_embedding = nn.Embedding(num_regions, region_embed_dim)
        layers: list[nn.Module] = []
        dim = numeric_dim + region_embed_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(dim, hidden_dim), nn.SiLU()])
            dim = hidden_dim

        final = nn.Linear(dim, 1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, numeric: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        emb = self.region_embedding(region_ids.long())
        return self.net(torch.cat([numeric, emb], dim=-1))


def best_checkpoint(train_output: str | Path) -> Path:
    """Resolve the actual best checkpoint instead of assuming the last epoch."""

    train_output = Path(train_output)
    history = train_output / "train" / "coarsening_train_history.csv"

    if history.is_file():
        import csv

        with history.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            final_best = [
                r for r in rows if str(r.get("is_final_best", "0")).strip() == "1"
            ]
            if final_best:
                epoch = int(float(final_best[-1]["epoch"]))
            else:
                epoch = int(float(min(rows, key=lambda r: float(r["loss"]))["epoch"]))
            candidate = train_output / "train" / f"best_{epoch}.pth"
            if candidate.is_file():
                return candidate

    files = list((train_output / "train").glob("best_*.pth"))
    if not files:
        files = list(train_output.rglob("best_*.pth"))
    if not files:
        raise FileNotFoundError(f"No best_*.pth found under {train_output}")

    def epoch_of(path: Path) -> int:
        match = re.search(r"best_(\d+)", path.stem)
        return int(match.group(1)) if match else -1

    return max(files, key=lambda p: (epoch_of(p), p.stat().st_mtime))


@dataclass
class RecoveryInputs:
    points: np.ndarray
    all_edges: np.ndarray
    object_edges: np.ndarray
    all_rest_lengths: np.ndarray
    spring_y: np.ndarray
    source_object_indices: np.ndarray
    prior_score: np.ndarray
    trajectory: np.ndarray
    target_total_springs: int
    num_regions: int = 6
    seed: int = 42
    dynamics_weight: float = 0.5


@dataclass
class RecoveryContext:
    points: np.ndarray
    object_edges: np.ndarray
    controller_edges: np.ndarray
    object_rest_lengths: np.ndarray
    controller_rest_lengths: np.ndarray
    object_spring_y: np.ndarray
    controller_spring_y: np.ndarray
    source_object_indices: np.ndarray
    removed_object_indices: np.ndarray
    restore_count: int
    target_object_springs: int
    target_total_springs: int
    node_region: np.ndarray
    num_regions: int
    prior_norm: np.ndarray
    edge_complexity: np.ndarray
    length_prior: np.ndarray
    length: np.ndarray
    base_radius: np.ndarray
    base_knn: np.ndarray


def prepare_recovery_context(inputs: RecoveryInputs) -> RecoveryContext:
    """Prepare all scene-agnostic quantities needed for topology candidates."""

    points = np.asarray(inputs.points, dtype=np.float64)
    all_edges = np.asarray(inputs.all_edges, dtype=np.int64)
    object_edges = np.asarray(inputs.object_edges, dtype=np.int64)
    all_rest = np.asarray(inputs.all_rest_lengths, dtype=np.float64)
    spring_y = np.asarray(inputs.spring_y, dtype=np.float64).reshape(-1)
    source = np.asarray(inputs.source_object_indices, dtype=np.int64).reshape(-1)
    prior_score = np.asarray(inputs.prior_score, dtype=np.float64).reshape(-1)
    trajectory = np.asarray(inputs.trajectory, dtype=np.float64)

    e_obj = len(object_edges)
    if len(all_edges) != len(all_rest) or len(all_edges) != len(spring_y):
        raise ValueError("Edge/rest-length/spring-Y counts do not match")
    if len(prior_score) != e_obj:
        raise ValueError(
            f"prior score has {len(prior_score)} values, expected {e_obj}"
        )
    if trajectory.ndim != 3 or trajectory.shape[1] < len(points):
        raise ValueError(
            "trajectory must have shape [frames, >=physical_nodes, xyz]"
        )

    controller_edges = all_edges[e_obj:]
    object_rest = all_rest[:e_obj]
    controller_rest = all_rest[e_obj:]
    y_obj = spring_y[:e_obj]
    y_ctrl = spring_y[e_obj:]
    n_ctrl = len(controller_edges)

    target_total = int(inputs.target_total_springs)
    target_object = target_total - n_ctrl
    restore_count = target_object - len(source)
    if restore_count <= 0:
        raise ValueError(
            "NeuSpring recovery adds object springs, so target budget must be "
            "larger than the source Stage-2 budget"
        )
    if target_object > e_obj:
        raise ValueError(
            f"Target object springs {target_object} exceed Stage-1 pool {e_obj}"
        )

    source_set = set(map(int, source))
    removed = np.asarray(
        [idx for idx in range(e_obj) if idx not in source_set],
        dtype=np.int64,
    )
    if restore_count > len(removed):
        raise ValueError(
            f"Need to restore {restore_count} springs but only {len(removed)} are available"
        )

    motion = np.diff(trajectory[:, : len(points)], axis=0)
    if len(motion) == 0:
        raise ValueError("At least two trajectory frames are required")
    motion_mag = np.linalg.norm(motion, axis=-1)
    motion_mean = motion_mag.mean(axis=0)
    motion_std = motion_mag.std(axis=0)

    node_features = np.concatenate(
        [
            standardize(points),
            float(inputs.dynamics_weight)
            * standardize(np.stack([motion_mean, motion_std], axis=1)),
        ],
        axis=1,
    )
    node_region = simple_kmeans_numpy(
        node_features,
        k=int(inputs.num_regions),
        seed=int(inputs.seed),
    )
    num_regions = int(node_region.max()) + 1

    p0 = points[object_edges[:, 0]]
    p1 = points[object_edges[:, 1]]
    vec = p1 - p0
    length = np.linalg.norm(vec, axis=1)

    node_complexity = minmax(motion_mean)
    edge_complexity = 0.5 * (
        node_complexity[object_edges[:, 0]]
        + node_complexity[object_edges[:, 1]]
    )
    prior_norm = minmax(prior_score)
    length_prior = 1.0 - minmax(length)

    tree = cKDTree(points)
    nn_dist, _ = tree.query(points, k=min(2, len(points)))
    if len(points) == 1:
        spacing = np.ones(1, dtype=np.float64)
    else:
        spacing = np.asarray(nn_dist)[:, 1]

    base_radius = np.zeros(num_regions, dtype=np.float64)
    base_knn = np.zeros(num_regions, dtype=np.int64)
    for region in range(num_regions):
        idx = np.where(node_region == region)[0]
        if len(idx) == 0:
            continue
        spacing_r = float(np.median(spacing[idx]))
        motion_r = float(np.mean(node_complexity[idx]))
        base_radius[region] = np.clip(
            spacing_r * (2.6 + 1.2 * motion_r),
            0.006,
            0.08,
        )
        base_knn[region] = int(
            np.clip(
                12 + 16 * motion_r + np.sqrt(len(idx)) * 0.25,
                8,
                48,
            )
        )

    return RecoveryContext(
        points=points,
        object_edges=object_edges,
        controller_edges=controller_edges,
        object_rest_lengths=object_rest,
        controller_rest_lengths=controller_rest,
        object_spring_y=y_obj,
        controller_spring_y=y_ctrl,
        source_object_indices=source,
        removed_object_indices=removed,
        restore_count=restore_count,
        target_object_springs=target_object,
        target_total_springs=target_total,
        node_region=node_region,
        num_regions=num_regions,
        prior_norm=prior_norm,
        edge_complexity=edge_complexity,
        length_prior=length_prior,
        length=length,
        base_radius=base_radius,
        base_knn=base_knn,
    )


def select_recovery_candidate(
    context: RecoveryContext,
    *,
    radius_scale: np.ndarray,
    knn_scale: np.ndarray,
    prior_only: bool,
    prior_weight: float = 0.55,
    eligibility_weight: float = 0.25,
    dynamics_weight: float = 0.15,
    length_weight: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Select exactly the requested number of restored object springs."""

    c = context
    radius_scale = np.asarray(radius_scale, dtype=np.float64)
    knn_scale = np.asarray(knn_scale, dtype=np.float64)
    if radius_scale.shape != (c.num_regions,) or knn_scale.shape != (c.num_regions,):
        raise ValueError("Region scale arrays must have one value per region")

    radius = c.base_radius * radius_scale
    knn = np.clip(np.round(c.base_knn * knn_scale), 4, 64).astype(np.int64)

    if prior_only:
        eligible = np.ones(len(c.object_edges), dtype=np.float64)
        recovery_score = c.prior_norm.copy()
    else:
        tree = cKDTree(c.points)
        max_k = min(int(knn.max()) + 1, len(c.points))
        _, neighbors = tree.query(c.points, k=max_k)
        if max_k == 1:
            neighbors = np.asarray(neighbors)[:, None]

        eligible = np.zeros(len(c.object_edges), dtype=np.float64)
        for edge_idx in c.removed_object_indices:
            u, v = c.object_edges[edge_idx]
            ru = c.node_region[u]
            rv = c.node_region[v]
            if c.length[edge_idx] > max(radius[ru], radius[rv]):
                continue

            ku = min(int(knn[ru]), max_k - 1)
            kv = min(int(knn[rv]), max_k - 1)
            u_neighbors = neighbors[u, 1 : ku + 1]
            v_neighbors = neighbors[v, 1 : kv + 1]
            if np.any(u_neighbors == v) or np.any(v_neighbors == u):
                eligible[edge_idx] = 1.0

        total_weight = (
            prior_weight
            + eligibility_weight
            + dynamics_weight
            + length_weight
        )
        if total_weight <= 0:
            raise ValueError("Candidate fusion weights must sum to a positive value")
        recovery_score = (
            prior_weight * c.prior_norm
            + eligibility_weight * eligible
            + dynamics_weight * c.edge_complexity
            + length_weight * c.length_prior
        ) / total_weight

    removed_sorted = c.removed_object_indices[
        np.argsort(recovery_score[c.removed_object_indices])[::-1]
    ]
    restore = removed_sorted[: c.restore_count]
    selected = np.asarray(
        sorted(
            list(map(int, c.source_object_indices))
            + list(map(int, restore))
        ),
        dtype=np.int64,
    )
    if len(selected) != c.target_object_springs:
        raise RuntimeError("Candidate selection did not hit the requested budget")

    return {
        "selected_indices": selected,
        "restored_indices": restore,
        "eligible": eligible,
        "recovery_score": recovery_score,
        "radius": radius,
        "knn": knn,
    }
