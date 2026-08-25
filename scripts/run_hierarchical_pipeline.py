#!/usr/bin/env python3
"""
Sequential paper pipeline.

Requires the existing Experiment-2 scripts in the same repository:
  train_stage1.py
  export_stage1_topology.py
  run_external_topology_inference.py
  compute_online_node_error.py

The new hierarchy is:
Stage1 -> node coarsening -> coarse rollout -> dense reconstruction
-> online dense residual -> project to coarse nodes -> online BT spring pruning
-> final reduced rollout -> dense reconstruction.
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parent

def run(cmd):
    print("\n"+"="*100)
    print("$"," ".join(map(str,cmd)))
    subprocess.run([str(x) for x in cmd],check=True,cwd=REPO)

def latest_ckpt(d):
    c=list((d/"train").glob("best_*.pth"))+list((d/"train").glob("iter_*.pth"))
    if not c: raise FileNotFoundError(d/"train")
    def key(p):
        m=re.findall(r"\d+",p.stem)
        return int(m[-1]) if m else -1
    return max(c,key=key)

p=argparse.ArgumentParser()
p.add_argument("--phystwin-root",required=True,type=Path)
p.add_argument("--scene",required=True)
p.add_argument("--stage1-ratio",type=float,default=0.5)
p.add_argument("--update-ratio",type=float,default=0.5,
               help="Fraction of remaining training interval used for online update.")
p.add_argument("--node-method",choices=["trajectory","geometry","krylov","soar"],default="soar")
p.add_argument("--node-keep-ratio",type=float,default=0.5)
p.add_argument("--spring-keep-ratio",type=float,default=0.5)
p.add_argument("--bt-weight",type=float,default=0.7)
p.add_argument("--online-error-weight",type=float,default=0.3)
p.add_argument("--node-rank",type=int)
p.add_argument("--alpha-dyn",type=float,default=1.0)
p.add_argument("--beta-geo",type=float,default=1.0)
p.add_argument("--protect-top-pct",type=float,default=10.0)
p.add_argument("--max-cluster-size",type=int,default=16)
p.add_argument("--mapping-k",type=int,default=4)
p.add_argument("--local-budget",type=int,default=300)
p.add_argument("--reduced-order",type=int,default=20)
p.add_argument("--stage1-model-path",type=Path)
p.add_argument("--skip-stage1-training",action="store_true")
a=p.parse_args()

root=a.phystwin_root.expanduser().resolve()
scene_root=root/"data/different_types"/a.scene
split=json.loads((scene_root/"split.json").read_text())
# Supports either explicit lists or common [start,end] ranges.
train=split.get("train",split.get("train_frames"))
test=split.get("test",split.get("test_frames"))
if train is None: raise KeyError("split.json has no train/train_frames")
if isinstance(train,list) and len(train)==2 and all(isinstance(x,(int,float)) for x in train):
    train_start,train_end=int(train[0]),int(train[1])
else:
    train_idx=[int(x) for x in train]
    train_start,train_end=min(train_idx),max(train_idx)+1
if test is not None:
    if isinstance(test,list) and len(test)==2 and all(isinstance(x,(int,float)) for x in test):
        test_start,test_end=int(test[0]),int(test[1])
    else:
        ti=[int(x) for x in test]; test_start,test_end=min(ti),max(ti)+1
else:
    test_start=test_end=train_end

span=max(1,train_end-train_start)
stage1_end=train_start+int(round(span*a.stage1_ratio))
update_end=stage1_end+int(round((train_end-stage1_end)*a.update_ratio))
stage1_end=max(train_start+2,min(stage1_end,train_end))
update_end=max(stage1_end+1,min(update_end,train_end))

out=root/"results/hierarchical_reduction"/a.scene
stage1_dir=out/"stage1"
node_dir=out/"node"; online_dir=out/"online"; spring_dir=out/"spring"; final_dir=out/"final"
for d in [stage1_dir,node_dir,online_dir,spring_dir,final_dir]: d.mkdir(parents=True,exist_ok=True)

model=a.stage1_model_path.expanduser().resolve() if a.stage1_model_path else None
if not a.skip_stage1_training and model is None:
    run([sys.executable,HERE/"train_stage1.py","--phystwin-root",root,"--scene",a.scene,
         "--stage1-ratio",a.stage1_ratio,"--output-dir",stage1_dir])
if model is None: model=latest_ckpt(stage1_dir)

full_topo=out/"full_topology.npz"
run([sys.executable,HERE/"export_stage1_topology.py","--phystwin-root",root,"--scene",a.scene,
     "--stage1-ratio",a.stage1_ratio,"--model-path",model,"--output-path",full_topo])

train_roll=out/"train_rollout"
run([sys.executable,HERE/"run_external_topology_inference.py","--phystwin-root",root,"--scene",a.scene,
     "--train-frame",train_end,"--model-path",model,"--topology-path",full_topo,
     "--output-dir",train_roll])
full_inf=train_roll/"inference.pkl"

node_topo=node_dir/f"coarse_{a.node_method}_keep_{int(round(a.node_keep_ratio*100))}.npz"
run([sys.executable,HERE/"generate_hierarchical_node_topology.py",
     "--topology-path",full_topo,"--inference-path",full_inf,"--output-path",node_topo,
     "--method",a.node_method,"--frame-start",train_start,"--frame-end",stage1_end,
     "--keep-ratio",a.node_keep_ratio,"--alpha-dyn",a.alpha_dyn,"--beta-geo",a.beta_geo,
     "--protect-top-pct",a.protect_top_pct,"--max-cluster-size",a.max_cluster_size,
     "--mapping-k",a.mapping_k] + (["--rank",a.node_rank] if a.node_rank else []))

coarse_roll=node_dir/"coarse_inference"
run([sys.executable,HERE/"run_external_topology_inference.py","--phystwin-root",root,"--scene",a.scene,
     "--train-frame",train_end,"--model-path",model,"--topology-path",node_topo,
     "--output-dir",coarse_roll])

dense_update=node_dir/"dense_train_reconstruction"/"inference.pkl"
run([sys.executable,HERE/"reconstruct_hierarchical_trajectory.py",
     "--reduced-inference-path",coarse_roll/"inference.pkl","--topology-path",node_topo,
     "--output-path",dense_update])

dense_err=online_dir/"dense_node_error.npz"
run([sys.executable,HERE/"compute_online_node_error.py","--phystwin-root",root,"--scene",a.scene,
     "--inference-path",dense_update,"--topology-path",full_topo,
     "--output-path",dense_err,
     "--online-start",stage1_end,"--online-end",update_end])

coarse_err=online_dir/"coarse_node_error.npz"
run([sys.executable,HERE/"project_online_error_to_coarse.py",
     "--dense-node-error",dense_err,"--coarsened-topology",node_topo,"--output-path",coarse_err])

spring_topo=spring_dir/f"online_bt_keep_{int(round(a.spring_keep_ratio*100))}.npz"
run([sys.executable,HERE/"generate_coarse_online_spring_topology.py",
     "--topology-path",node_topo,"--coarse-node-error",coarse_err,"--output-path",spring_topo,
     "--keep-ratio",a.spring_keep_ratio,"--bt-weight",a.bt_weight,
     "--online-error-weight",a.online_error_weight,
     "--local-budget",a.local_budget,"--reduced-order",a.reduced_order])

final_red=final_dir/"reduced_inference"
run([sys.executable,HERE/"run_external_topology_inference.py","--phystwin-root",root,"--scene",a.scene,
     "--train-frame",train_end,"--model-path",model,"--topology-path",spring_topo,
     "--output-dir",final_red])

final_dense=final_dir/"dense_reconstruction"/"inference.pkl"
run([sys.executable,HERE/"reconstruct_hierarchical_trajectory.py",
     "--reduced-inference-path",final_red/"inference.pkl","--topology-path",node_topo,
     "--output-path",final_dense])

summary={
 "scene":a.scene,
 "train":[train_start,train_end],
 "node_signature_frames":[train_start,stage1_end],
 "online_update_frames":[stage1_end,update_end],
 "test":[test_start,test_end],
 "stage1_model":str(model),
 "full_topology":str(full_topo),
 "node_topology":str(node_topo),
 "coarse_error":str(coarse_err),
 "spring_topology":str(spring_topo),
 "final_dense_inference":str(final_dense),
}
(out/"hierarchical_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print("\n[DONE]",out/"hierarchical_summary.json")
