#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert hierarchical node topology NPZ to PhysTwin coarsening-trainer NPZ."
    )
    p.add_argument("--phystwin-root", required=True, type=Path)
    p.add_argument("--scene", required=True)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--mode", required=True, choices=["geometry", "trajectory", "krylov", "soar"])
    p.add_argument("--base-path", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.phystwin_root.expanduser().resolve()
    scene_root = (
        args.base_path.expanduser().resolve() / args.scene
        if args.base_path is not None
        else root / "data" / "different_types" / args.scene
    )
    src = args.input.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    data_path = scene_root / "final_data.pkl"
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    z = np.load(src, allow_pickle=True)
    required = {
        "original_num_object_points",
        "reduced_object_points",
        "num_object_springs",
        "points_full",
        "masses",
        "springs",
        "rest_lengths",
        "spring_Y",
        "original_object_points",
        "mapping_indices",
        "mapping_weights",
    }
    missing = sorted(required.difference(z.files))
    if missing:
        raise KeyError(f"{src} missing arrays: {missing}")

    with data_path.open("rb") as f:
        data = pickle.load(f)

    n_dense = int(np.asarray(z["original_num_object_points"]).item())
    n_red = int(len(z["reduced_object_points"]))
    e_obj = int(np.asarray(z["num_object_springs"]).item())
    n_controller = int(data["controller_points"].shape[1])
    n_original_tracked = int(data["object_points"].shape[1])
    n_surface = int(data["surface_points"].shape[0])
    n_interior = int(data["interior_points"].shape[0])
    n_surface_total = n_original_tracked + n_surface

    if n_original_tracked + n_surface + n_interior != n_dense:
        raise RuntimeError(
            "Dense physical-node accounting mismatch: "
            f"{n_original_tracked}+{n_surface}+{n_interior} != {n_dense}"
        )

    mapping_weights = np.asarray(z["mapping_weights"], dtype=np.float64)
    if not np.allclose(mapping_weights.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("mapping_weights rows do not sum to one")

    metadata = {
        "source_topology": str(src),
        "scene": args.scene,
        "method": args.mode,
        "schema": "node_coarsening_trainer_compatible",
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        reduced_points=np.asarray(z["reduced_object_points"]),
        reduced_points_all=np.asarray(z["points_full"]),
        reduced_masses=np.asarray(z["masses"]),
        reduced_edges=np.asarray(z["springs"]),
        reduced_edges_oo=np.asarray(z["springs"])[:e_obj],
        reduced_rest_lengths=np.asarray(z["rest_lengths"]),
        reduced_spring_Y_init=np.asarray(z["spring_Y"]),
        dense_rest_points=np.asarray(z["original_object_points"]),
        mapping_indices=np.asarray(z["mapping_indices"]),
        mapping_weights=mapping_weights,
        n_object_original=np.asarray(n_dense),
        n_object_reduced=np.asarray(n_red),
        n_controller=np.asarray(n_controller),
        n_original_tracked=np.asarray(n_original_tracked),
        n_surface_total=np.asarray(n_surface_total),
        n_interior=np.asarray(n_interior),
        controller_points=np.asarray(data["controller_points"][0]),
        mode=np.asarray(args.mode),
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    print("[DONE]", out)
    print("scene          :", args.scene)
    print("mode           :", args.mode)
    print("dense nodes    :", n_dense)
    print("reduced nodes  :", n_red)
    print("object springs :", e_obj)
    print("total springs  :", len(z["springs"]))
    print("mapping        :", np.asarray(z["mapping_indices"]).shape)


if __name__ == "__main__":
    main()
