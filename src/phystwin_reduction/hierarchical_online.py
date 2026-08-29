from __future__ import annotations

"""Compatibility layer for the old hierarchical-online API.

The former Stage-2 implementation could score a pre-retraining coarse graph.
That protocol is intentionally disabled.  The formal method lives in
``stage2_post_retrain.py`` and requires ``topology_retrained.npz``.

Only dense->coarse error projection is kept here for backward compatibility.
"""

from pathlib import Path

import numpy as np


def project_dense_error_to_coarse(
    dense_error: np.ndarray,
    coarsened_topology_path: str | Path,
) -> np.ndarray:
    z = np.load(Path(coarsened_topology_path).expanduser().resolve(), allow_pickle=True)
    required = {"mapping_indices", "mapping_weights", "reduced_object_points"}
    missing = sorted(required.difference(z.files))
    if missing:
        raise KeyError(f"Coarsened topology missing arrays: {missing}")

    weights = np.asarray(z["mapping_weights"], dtype=np.float64)
    indices = np.asarray(z["mapping_indices"], dtype=np.int64)
    n_red = int(len(z["reduced_object_points"]))
    error = np.asarray(dense_error, dtype=np.float64).reshape(-1)
    if len(error) < len(indices):
        raise ValueError(
            f"Dense error has {len(error)} entries but mapping needs {len(indices)}"
        )
    error = error[: len(indices)]

    accum = np.zeros(n_red, dtype=np.float64)
    denom = np.zeros(n_red, dtype=np.float64)
    for k in range(indices.shape[1]):
        np.add.at(accum, indices[:, k], weights[:, k] * error)
        np.add.at(denom, indices[:, k], weights[:, k])
    return accum / np.maximum(denom, 1e-12)


def _legacy_disabled() -> None:
    raise RuntimeError(
        "Legacy pre-retrain Stage-2 Online-BT is disabled. "
        "Use phystwin_reduction.stage2_post_retrain and "
        "scripts/build_final_budget_stage2.py."
    )


def build_online_bt_scores(*args, **kwargs):
    _legacy_disabled()


def generate_coarse_online_spring_topology(*args, **kwargs):
    _legacy_disabled()
