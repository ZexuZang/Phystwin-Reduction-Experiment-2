#!/usr/bin/env python3
from __future__ import annotations

"""Project dense update error to the Stage-1 node-coarsened graph.

This script deliberately reads the node mapping directly from the node-coarsened
NPZ.  It does not call the legacy Stage-2 implementation.
"""

import argparse
from pathlib import Path

import numpy as np


def project_dense_error(dense_error: np.ndarray, topology_path: Path) -> np.ndarray:
    z = np.load(topology_path, allow_pickle=True)
    required = {"mapping_indices", "mapping_weights", "reduced_object_points"}
    missing = sorted(required.difference(z.files))
    if missing:
        raise KeyError(f"{topology_path} missing coarsening arrays: {missing}")

    indices = np.asarray(z["mapping_indices"], dtype=np.int64)
    weights = np.asarray(z["mapping_weights"], dtype=np.float64)
    n_reduced = int(len(z["reduced_object_points"]))
    error = np.asarray(dense_error, dtype=np.float64).reshape(-1)

    if len(error) < len(indices):
        raise ValueError(
            f"Dense error has {len(error)} nodes but mapping needs {len(indices)}"
        )
    error = error[: len(indices)]
    if indices.shape != weights.shape:
        raise ValueError(f"mapping shape mismatch: {indices.shape} vs {weights.shape}")
    if np.any(indices < 0) or np.any(indices >= n_reduced):
        raise ValueError("mapping_indices contains out-of-range coarse-node indices")

    accum = np.zeros(n_reduced, dtype=np.float64)
    denom = np.zeros(n_reduced, dtype=np.float64)
    for k in range(indices.shape[1]):
        np.add.at(accum, indices[:, k], weights[:, k] * error)
        np.add.at(denom, indices[:, k], weights[:, k])
    return accum / np.maximum(denom, 1e-12)


def main() -> None:
    p = argparse.ArgumentParser(description="Project dense update error to node-coarsened graph.")
    p.add_argument("--dense-node-error", required=True, type=Path)
    p.add_argument("--coarsened-topology", required=True, type=Path)
    p.add_argument("--output-path", required=True, type=Path)
    a = p.parse_args()

    dense_path = a.dense_node_error.expanduser().resolve()
    topology_path = a.coarsened_topology.expanduser().resolve()
    z = np.load(dense_path, allow_pickle=True)
    key = "node_error_normalized" if "node_error_normalized" in z.files else "node_error"
    coarse = project_dense_error(np.asarray(z[key]), topology_path)
    normalized = (coarse - coarse.min()) / (np.ptp(coarse) + 1e-12)

    out = a.output_path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        node_error=coarse,
        node_error_normalized=normalized,
        source_dense_error=np.asarray(str(dense_path)),
        source_coarsened_topology=np.asarray(str(topology_path)),
    )
    print("[DONE]", out)
    print("coarse nodes  :", len(coarse))
    print("min/mean/max  :", float(coarse.min()), float(coarse.mean()), float(coarse.max()))


if __name__ == "__main__":
    main()
