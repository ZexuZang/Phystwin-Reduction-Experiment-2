#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from phystwin_reduction.hierarchical_coarsening import reconstruct_dense_trajectory
p=argparse.ArgumentParser()
p.add_argument("--reduced-inference-path",required=True,type=Path)
p.add_argument("--topology-path",required=True,type=Path)
p.add_argument("--output-path",required=True,type=Path)
a=p.parse_args()
print("[DONE]",reconstruct_dense_trajectory(a.reduced_inference_path,a.topology_path,a.output_path))
