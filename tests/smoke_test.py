#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pickle
import shutil
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phystwin_reduction.online_adaptation import (
    compute_online_node_error,
    generate_node_cluster_topology,
    reconstruct_full_trajectory,
)
from phystwin_reduction.topology import (
    generate_connected_pruned_topology,
    load_topology,
    normalize_score,
)


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="phystwin_strategy2_smoke_"))
    try:
        points = np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [2, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        object_springs = np.asarray(
            [
                [0, 1],
                [1, 2],
                [3, 4],
                [4, 5],
                [0, 3],
                [1, 4],
                [2, 5],
                [0, 4],
                [1, 5],
            ],
            dtype=np.int64,
        )
        controller_springs = np.asarray([[6, 0], [6, 3]], dtype=np.int64)
        springs = np.concatenate([object_springs, controller_springs])
        rest = np.linalg.norm(points[springs[:, 0]] - points[springs[:, 1]], axis=1)
        spring_y = np.linspace(1.0, 2.0, len(springs))
        topology_path = root / "topology.npz"
        np.savez_compressed(
            topology_path,
            points_full=points,
            springs=springs,
            rest_lengths=rest,
            masses=np.ones(len(points)),
            spring_Y=spring_y,
            num_object_springs=len(object_springs),
        )

        frames = 8
        trajectory = np.repeat(points[None], frames, axis=0)
        for frame_idx in range(frames):
            trajectory[frame_idx, :6, 0] += 0.05 * frame_idx
        inference_path = root / "inference.pkl"
        with inference_path.open("wb") as handle:
            pickle.dump(trajectory.astype(np.float32), handle)

        gt_object = trajectory[:, :6].copy()
        final_data_path = root / "final_data.pkl"
        with final_data_path.open("wb") as handle:
            pickle.dump(
                {
                    "object_points": gt_object,
                    "object_visibilities": np.ones((frames, 6), dtype=bool),
                    "surface_points": np.empty((0, 3)),
                },
                handle,
            )

        _, normalized_error, _ = compute_online_node_error(
            inference_path,
            final_data_path,
            topology_path,
            online_start=3,
            online_end=7,
        )

        cluster_path = root / "cluster.npz"
        generate_node_cluster_topology(
            topology_path,
            inference_path,
            cluster_path,
            online_start=3,
            online_end=7,
            node_keep_ratio=0.67,
            max_cluster_size=3,
            node_error=normalized_error,
            protect_high_error=False,
        )
        clustered = np.load(cluster_path)
        reduced_points = clustered["points_full"]
        reduced_trajectory = np.repeat(reduced_points[None], frames, axis=0)
        reduced_path = root / "reduced.pkl"
        with reduced_path.open("wb") as handle:
            pickle.dump(reduced_trajectory.astype(np.float32), handle)

        reconstructed_path = root / "reconstructed.pkl"
        _, reconstruction_meta = reconstruct_full_trajectory(
            reduced_path, cluster_path, reconstructed_path
        )
        assert reconstruction_meta["node_count_matches"]

        topology = load_topology(topology_path)
        stiffness = normalize_score(
            topology.object_spring_Y
            / np.maximum(topology.object_rest_lengths, 1e-8)
        )
        pruned_path = root / "pruned.npz"
        _, pruning_meta = generate_connected_pruned_topology(
            topology_path,
            pruned_path,
            stiffness,
            keep_ratio=0.6,
            min_degree=1,
            method="smoke_test",
        )
        assert pruning_meta["after_stats"]["components"] == 1
        print("Strategy-2 smoke test passed.")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
