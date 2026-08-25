#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from phystwin_reduction.hierarchical_coarsening import generate_hierarchical_node_topology

p=argparse.ArgumentParser()
p.add_argument("--topology-path",required=True,type=Path)
p.add_argument("--inference-path",required=True,type=Path)
p.add_argument("--output-path",required=True,type=Path)
p.add_argument("--method",choices=["trajectory","geometry","krylov","soar"],default="soar")
p.add_argument("--frame-start",type=int,default=0)
p.add_argument("--frame-end",type=int,required=True)
p.add_argument("--keep-ratio",type=float,default=0.5)
p.add_argument("--rank",type=int)
p.add_argument("--alpha-dyn",type=float,default=1.0)
p.add_argument("--beta-geo",type=float,default=1.0)
p.add_argument("--protect-top-pct",type=float,default=10.0)
p.add_argument("--max-cluster-size",type=int,default=16)
p.add_argument("--mapping-k",type=int,default=4)
p.add_argument("--dashpot",type=float,default=100.0)
p.add_argument("--drag",type=float,default=3.0)
a=p.parse_args()
r=generate_hierarchical_node_topology(
    a.topology_path,a.inference_path,a.output_path,
    method=a.method,frame_start=a.frame_start,frame_end=a.frame_end,
    keep_ratio=a.keep_ratio,signature_rank=a.rank,
    alpha_dyn=a.alpha_dyn,beta_geo=a.beta_geo,
    protect_top_pct=a.protect_top_pct,max_cluster_size=a.max_cluster_size,
    mapping_k=a.mapping_k,dashpot=a.dashpot,drag=a.drag,
)
print("[DONE]",r.topology_path)
print(r.metadata)
