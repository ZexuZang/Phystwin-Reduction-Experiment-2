#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from phystwin_reduction.hierarchical_online import project_dense_error_to_coarse
p=argparse.ArgumentParser()
p.add_argument("--dense-node-error",required=True,type=Path)
p.add_argument("--coarsened-topology",required=True,type=Path)
p.add_argument("--output-path",required=True,type=Path)
a=p.parse_args()
z=np.load(a.dense_node_error,allow_pickle=True)
key="node_error_normalized" if "node_error_normalized" in z.files else "node_error"
coarse=project_dense_error_to_coarse(np.asarray(z[key]),a.coarsened_topology)
a.output_path.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.output_path,node_error=coarse,node_error_normalized=(coarse-coarse.min())/(np.ptp(coarse)+1e-12))
print("[DONE]",a.output_path)
