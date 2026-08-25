from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from .bt_guided import compute_bt_node_scores
from .topology import generate_connected_pruned_topology, load_topology, normalize_score


def project_dense_error_to_coarse(
    dense_error: np.ndarray,
    coarsened_topology_path: str | Path,
) -> np.ndarray:
    z=np.load(coarsened_topology_path,allow_pickle=True)
    W=np.asarray(z["mapping_weights"],dtype=np.float64)
    I=np.asarray(z["mapping_indices"],dtype=np.int64)
    n_red=int(np.asarray(z["num_object_springs"]).item())  # placeholder overwritten below
    n_red=len(np.asarray(z["reduced_object_points"]))
    e=np.asarray(dense_error,dtype=np.float64)[:len(I)]
    accum=np.zeros(n_red,dtype=np.float64)
    denom=np.zeros(n_red,dtype=np.float64)
    for k in range(I.shape[1]):
        np.add.at(accum,I[:,k],W[:,k]*e)
        np.add.at(denom,I[:,k],W[:,k])
    return accum/np.maximum(denom,1e-12)


def build_online_bt_scores(
    topology_path: str | Path,
    coarse_node_error: np.ndarray,
    *,
    bt_weight: float=0.7,
    online_error_weight: float=0.3,
    local_budget: int=300,
    reduced_order: int=20,
):
    data=load_topology(topology_path)
    i=data.object_springs[:,0].astype(np.int64)
    j=data.object_springs[:,1].astype(np.int64)

    stiffness=normalize_score(
        data.object_spring_Y/np.maximum(data.object_rest_lengths,1e-8)
    )
    err=normalize_score(np.asarray(coarse_node_error,dtype=np.float64))
    edge_err=normalize_score(0.5*(err[i]+err[j]))

    bt_node,bt_info=compute_bt_node_scores(
        data,local_budget=local_budget,reduced_order=reduced_order
    )
    bt_edge=normalize_score(0.5*(bt_node[i]+bt_node[j]))
    prior=normalize_score(bt_weight*bt_edge+(1.0-bt_weight)*stiffness)
    final=normalize_score(
        (1.0-online_error_weight)*prior+online_error_weight*edge_err
    )
    return final,{
        "stiffness_edge_score":stiffness,
        "bt_node_score":bt_node,
        "bt_edge_score":bt_edge,
        "online_edge_error":edge_err,
        "bt_info":bt_info,
    }


def generate_coarse_online_spring_topology(
    topology_path,
    coarse_node_error,
    output_path,
    *,
    keep_ratio=0.5,
    bt_weight=0.7,
    online_error_weight=0.3,
    min_degree=1,
    local_budget=300,
    reduced_order=20,
):
    score,detail=build_online_bt_scores(
        topology_path,coarse_node_error,
        bt_weight=bt_weight,online_error_weight=online_error_weight,
        local_budget=local_budget,reduced_order=reduced_order,
    )
    scalar_bt={
        k:v for k,v in detail["bt_info"].items()
        if np.isscalar(v) or isinstance(v,(str,bool))
    }
    path,meta=generate_connected_pruned_topology(
        topology_path,output_path,score,
        keep_ratio=keep_ratio,min_degree=min_degree,allow_bridge=False,
        method="hierarchical_online_bt",
        extra_metadata={
            "bt_weight":bt_weight,
            "online_error_weight":online_error_weight,
            **scalar_bt,
        },
        extra_arrays={
            "stiffness_edge_score":detail["stiffness_edge_score"],
            "bt_node_score":detail["bt_node_score"],
            "bt_edge_score":detail["bt_edge_score"],
            "online_edge_error":detail["online_edge_error"],
        },
    )
    return path,meta
