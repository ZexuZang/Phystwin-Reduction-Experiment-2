#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from phystwin_reduction.hierarchical_online import generate_coarse_online_spring_topology
p=argparse.ArgumentParser()
p.add_argument("--topology-path",required=True,type=Path)
p.add_argument("--coarse-node-error",required=True,type=Path)
p.add_argument("--output-path",required=True,type=Path)
p.add_argument("--keep-ratio",type=float,default=0.5)
p.add_argument("--bt-weight",type=float,default=0.7)
p.add_argument("--online-error-weight",type=float,default=0.3)
p.add_argument("--min-degree",type=int,default=1)
p.add_argument("--local-budget",type=int,default=300)
p.add_argument("--reduced-order",type=int,default=20)
a=p.parse_args()
z=np.load(a.coarse_node_error)
key="node_error_normalized" if "node_error_normalized" in z.files else "node_error"
path,meta=generate_coarse_online_spring_topology(
    a.topology_path,np.asarray(z[key]),a.output_path,
    keep_ratio=a.keep_ratio,bt_weight=a.bt_weight,
    online_error_weight=a.online_error_weight,min_degree=a.min_degree,
    local_budget=a.local_budget,reduced_order=a.reduced_order,
)
print("[DONE]",path)
print(meta)
