# Merge Guide: from two repos to one paper repository

## 0. What changes conceptually

Old Experiment-2:

```text
Stage1
├─ trajectory node clustering
├─ online stiffness spring pruning
└─ online BT spring pruning
```

New paper pipeline:

```text
Stage1
  ↓
SOAR/Krylov node coarsening
  ↓
coarse rollout
  ↓
dense reconstruction
  ↓
UPDATE-only residual
  ↓
dense → coarse error projection
  ↓
BT + stiffness + online error
  ↓
spring pruning on the coarse graph
  ↓
final rollout
  ↓
dense reconstruction
  ↓
TEST-only evaluation
```

## 1. Keep these files from Experiment-2

Do not delete them:

- `src/phystwin_reduction/topology.py`
- `src/phystwin_reduction/bt_guided.py`
- `src/phystwin_reduction/online_adaptation.py`
- `src/phystwin_reduction/phystwin_runtime.py`
- `scripts/train_stage1.py`
- `scripts/export_stage1_topology.py`
- `scripts/run_external_topology_inference.py`
- `scripts/compute_online_node_error.py`
- `scripts/evaluate_geometry.py`
- `scripts/render_methods.py`
- `scripts/evaluate_rendering.py`
- `scripts/benchmark_simulation.py`

The old `generate_node_cluster_topology.py` becomes a **trajectory-clustering baseline**.

The old `generate_online_spring_topologies.py` becomes a **spring-only baseline**.

## 2. Add the new files from this integration bundle

Copy:

```bash
cp -a Phystwin-Hierarchical-Reduction/src/. Phystwin-Reduction-Experiment-2/src/
cp -a Phystwin-Hierarchical-Reduction/scripts/. Phystwin-Reduction-Experiment-2/scripts/
cp -a Phystwin-Hierarchical-Reduction/configs/. Phystwin-Reduction-Experiment-2/configs/
cp -a Phystwin-Hierarchical-Reduction/docs/. Phystwin-Reduction-Experiment-2/docs/
cp -a Phystwin-Hierarchical-Reduction/tests/. Phystwin-Reduction-Experiment-2/tests/
```

## 3. First test without GPU

```bash
cd Phystwin-Reduction-Experiment-2
python -m py_compile \
  src/phystwin_reduction/hierarchical_coarsening.py \
  src/phystwin_reduction/hierarchical_online.py \
  scripts/generate_hierarchical_node_topology.py \
  scripts/reconstruct_hierarchical_trajectory.py \
  scripts/project_online_error_to_coarse.py \
  scripts/generate_coarse_online_spring_topology.py \
  scripts/run_hierarchical_pipeline.py

python tests/smoke_test_hierarchical.py
```

## 4. First real-data test: do not run everything

Use one scene only.

### 4.1 Generate/export your Stage-1 topology exactly as before

First make sure these exist:

```text
PhysTwin/data/different_types/<scene>/final_data.pkl
PhysTwin/data/different_types/<scene>/split.json
PhysTwin/experiments_optimization/<scene>/optimal_params.pkl
your Stage-1 checkpoint
```

### 4.2 Run only node coarsening

```bash
python scripts/generate_hierarchical_node_topology.py \
  --topology-path <stage1_topology.npz> \
  --inference-path <stage1_train_rollout/inference.pkl> \
  --output-path /tmp/coarse_soar.npz \
  --method soar \
  --frame-start 0 \
  --frame-end <END_OF_STAGE1_TRAIN_ONLY> \
  --keep-ratio 0.75 \
  --rank 5 \
  --alpha-dyn 1 \
  --beta-geo 1 \
  --protect-top-pct 10 \
  --max-cluster-size 16 \
  --mapping-k 4
```

Inspect:

```python
import numpy as np
z=np.load("/tmp/coarse_soar.npz",allow_pickle=True)
print(z["points_full"].shape)
print(z["springs"].shape)
print(z["mapping_indices"].shape)
print(z["mapping_weights"].sum(axis=1).min(), z["mapping_weights"].sum(axis=1).max())
```

Expected:
- node count goes down;
- no NaN;
- every mapping row sums to ~1.

### 4.3 Run reduced topology inference

Use your existing:

```bash
python scripts/run_external_topology_inference.py ...
```

If this fails because checkpoint spring parameters have the old edge count,
do **not** hack around the shape mismatch. This is the point where the
coarsened trainer/retraining overlay from the Node-Coarsening repo must be
ported into your PhysTwin checkout.

The final paper pipeline should retrain the coarsened topology instead of
blindly reusing the full graph's per-edge spring checkpoint.

## 5. Important implementation decision: retraining

For the paper, the safe implementation is:

```text
full Stage-1 checkpoint
        ↓
build coarse topology
        ↓
initialize coarse Y by axial aggregation
        ↓
short/full coarse retraining
        ↓
coarse rollout
```

This follows the Node-Coarsening repository and avoids a hidden shape mismatch.

Port these PhysTwin overlay files if your external-topology runner cannot
train a new topology:

```text
Node-Coarsening/phystwin_changes/overlay/qqtt/engine/trainer_warp_coarsening.py
Node-Coarsening/phystwin_changes/overlay/qqtt/model/diff_simulator/spring_mass_warp_coarsening.py
Node-Coarsening/node_coarsening/run_coarsening_retrain.py
```

The current integration bundle intentionally does not overwrite PhysTwin
automatically because your local PhysTwin checkout/environment must be
validated first.

## 6. Online update

Once the coarse rollout works:

1. reconstruct coarse trajectory to dense points;
2. compute error **only on update frames**;
3. project dense error to coarse nodes;
4. compute BT on the coarse graph;
5. fuse BT + stiffness + online residual;
6. prune coarse springs;
7. retrain/fine-tune final topology;
8. evaluate only on test frames.

## 7. Git workflow

```bash
cd Phystwin-Reduction-Experiment-2
git checkout -b hierarchical-reduction
git add src scripts configs docs tests README_HIERARCHICAL.md
git commit -m "Add hierarchical node-and-spring reduction pipeline"
git push -u origin hierarchical-reduction
```

After one scene passes end-to-end:

```bash
git tag -a v0.1-hierarchical -m "First hierarchical pipeline"
git push origin v0.1-hierarchical
```

## 8. Do not merge to main yet

Keep `main` as your old reproducible Experiment-2 baseline until:
- one scene runs end-to-end;
- test leakage is ruled out;
- coarse retraining works;
- geometry metrics work;
- rendering works;
- FPS benchmark works.
