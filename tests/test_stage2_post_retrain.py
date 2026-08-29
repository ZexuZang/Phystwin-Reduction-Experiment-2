from pathlib import Path

import numpy as np
import pytest

from phystwin_reduction.stage2_post_retrain import (
    compute_final_budget,
    controller_object_seeds,
    infer_coarse_object_points,
    require_retrained_topology,
)
from phystwin_reduction.topology import load_topology


def _write_topology(path: Path, retrained: bool, reduced: bool = True):
    # 3 object nodes + 2 controllers. Controller edges deliberately use the
    # [object, controller] order that broke the old BT implementation.
    points = np.zeros((5, 3), dtype=np.float64)
    springs = np.asarray([[0, 1], [1, 2], [0, 2], [0, 3], [2, 4]], dtype=np.int64)
    rest = np.ones(5, dtype=np.float64)
    masses = np.ones(5, dtype=np.float64)
    y = np.arange(1, 6, dtype=np.float64)
    arrays = {
        "points_full": points,
        "springs": springs,
        "rest_lengths": rest,
        "masses": masses,
        "spring_Y": y,
        "num_object_springs": np.asarray(3),
    }
    if reduced:
        arrays["reduced_object_points"] = points[:3]
        arrays["mapping_indices"] = np.asarray([[0], [1], [2]])
        arrays["mapping_weights"] = np.ones((3, 1))
    if retrained:
        arrays["retrained_checkpoint"] = np.asarray("best_10.pth")
    np.savez_compressed(path, **arrays)


def test_infer_object_count_is_orientation_independent(tmp_path):
    p = tmp_path / "coarse.npz"
    _write_topology(p, retrained=True)
    data = load_topology(p)
    assert infer_coarse_object_points(p, data) == 3
    assert controller_object_seeds(data, 3).tolist() == [0, 2]


def test_stage2_rejects_pre_retrain_topology(tmp_path):
    p = tmp_path / "coarse.npz"
    _write_topology(p, retrained=False)
    with pytest.raises(RuntimeError, match="post-node-retraining"):
        require_retrained_topology(p)


def test_final_budget_is_relative_to_full_graph(tmp_path):
    full = tmp_path / "full.npz"
    coarse = tmp_path / "coarse.npz"

    # Full: 8 total springs; coarse: 5 total = 3 object + 2 controller.
    points = np.zeros((5, 3))
    np.savez_compressed(
        full,
        points_full=points,
        springs=np.asarray([[0,1],[1,2],[0,2],[0,1],[1,2],[0,2],[0,3],[2,4]]),
        rest_lengths=np.ones(8),
        masses=np.ones(5),
        spring_Y=np.ones(8),
        num_object_springs=np.asarray(6),
    )
    _write_topology(coarse, retrained=True)

    budget = compute_final_budget(full, coarse, 0.50)
    assert budget.target_total_springs == 4
    assert budget.stage1_controller_springs == 2
    assert budget.target_object_springs == 2
    assert budget.stage2_object_keep_ratio == pytest.approx(2 / 3)
