#!/usr/bin/env python3
from pathlib import Path
import pickle, tempfile, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

# Lightweight local stubs for topology dependency if running only the overlay.
try:
    from phystwin_reduction.hierarchical_coarsening import build_graph_mapping
except Exception as e:
    print("IMPORT FAILED:",repr(e))
    print("This overlay must be copied into Experiment-2 so topology.py is available.")
    raise

def main():
    original=np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]],dtype=float)
    reduced=np.array([[0.5,0,0],[2.5,0,0]],dtype=float)
    cluster=np.array([0,0,1,1])
    edges=np.array([[0,1]])
    idx,w=build_graph_mapping(original,reduced,cluster,edges,k=2)
    assert idx.shape==(4,2)
    assert w.shape==(4,2)
    assert np.allclose(w.sum(axis=1),1.0)
    print("[OK] graph mapping")
    print("[OK] hierarchical overlay smoke test")

if __name__=="__main__":
    main()
