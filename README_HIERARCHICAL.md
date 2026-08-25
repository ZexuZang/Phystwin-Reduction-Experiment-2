# PhysTwin Hierarchical Reduction

Unified implementation plan for a paper-oriented PhysTwin reduction pipeline:

1. **Dynamics-aware physical-node coarsening** (Trajectory / Krylov / SOAR).
2. **Reduced-to-dense graph reconstruction** with fixed K-neighbor weights.
3. **Online residual estimation** on update frames only.
4. **Coarse-node residual projection**.
5. **BT/stiffness/online-error guided spring pruning** on the coarsened graph.
6. **Connectivity-preserving final topology**.
7. **Final reduced inference and dense reconstruction** for PhysTwin geometry/rendering evaluation.

The intended paper method is:

> **SOAR node coarsening + online BT-guided spring pruning**

Trajectory, geometry, random, Krylov, stiffness-only and BT-only variants are retained as baselines/ablations.

## Important split rule

Never use test frames to build topology.

```text
TRAIN                  UPDATE                 TEST
|----------------------|----------------------|---------------------|
baseline fitting        online residual        evaluation only
Krylov/SOAR signatures  spring adaptation      no topology selection
node coarsening
```

## How this repository is meant to be used

This bundle is an **integration overlay** for
`ZexuZang/Phystwin-Reduction-Experiment-2`.

Start from your existing Experiment-2 checkout and copy this bundle over it.
It does not delete the old scripts; the old trajectory clustering and old
parallel node/spring branches remain useful as baselines.

Recommended local layout:

```text
<ROOT>/
├── PhysTwin/
├── Phystwin-Reduction-Experiment-2/
├── Node-Coarsening/                     # optional reference checkout
└── Phystwin-Hierarchical-Reduction/     # this integration bundle
```

Apply:

```bash
cd <ROOT>/Phystwin-Reduction-Experiment-2
git checkout -b hierarchical-reduction

cp -a ../Phystwin-Hierarchical-Reduction/src/. src/
cp -a ../Phystwin-Hierarchical-Reduction/scripts/. scripts/
cp -a ../Phystwin-Hierarchical-Reduction/configs/. configs/
cp -a ../Phystwin-Hierarchical-Reduction/docs/. docs/
cp -a ../Phystwin-Hierarchical-Reduction/tests/. tests/
cp ../Phystwin-Hierarchical-Reduction/README_HIERARCHICAL.md .
```

Then run the smoke test:

```bash
python tests/smoke_test_hierarchical.py
```

## New files

```text
src/phystwin_reduction/
├── hierarchical_coarsening.py
└── hierarchical_online.py

scripts/
├── generate_hierarchical_node_topology.py
├── reconstruct_hierarchical_trajectory.py
├── project_online_error_to_coarse.py
├── generate_coarse_online_spring_topology.py
└── run_hierarchical_pipeline.py

configs/
└── hierarchical_reduction.yaml

docs/
├── MERGE_GUIDE.md
└── PAPER_EXPERIMENT_PLAN.md
```

## Main command

After the integration files are copied into your Experiment-2 repository:

```bash
python scripts/run_hierarchical_pipeline.py \
  --phystwin-root ~/PhysTwin \
  --scene double_stretch_sloth \
  --node-method soar \
  --node-keep-ratio 0.50 \
  --spring-keep-ratio 0.50 \
  --stage1-ratio 0.50 \
  --update-ratio 0.50 \
  --bt-weight 0.70 \
  --online-error-weight 0.30
```

This orchestrator deliberately calls the existing Experiment-2 scripts for
Stage-1 training/export/inference and uses the new integrated modules only for
the hierarchical parts.

## Output layout

```text
<PhysTwin>/results/hierarchical_reduction/<scene>/
├── stage1/
├── full_topology.npz
├── train_rollout/
├── node/
│   ├── coarse_<method>_keep_XX.npz
│   ├── coarse_inference/
│   └── dense_train_reconstruction/
├── online/
│   ├── dense_node_error.npz
│   └── coarse_node_error.npz
├── spring/
│   └── online_bt_keep_XX.npz
├── final/
│   ├── reduced_inference/
│   └── dense_reconstruction/inference.pkl
└── hierarchical_summary.json
```

## Paper-facing method

### Level I: physical state reduction

For the full graph \(G=(V,E)\), obtain a node signature \(z_i\) from a
trajectory, first-order Krylov basis, or SOAR-inspired second-order basis.
Only adjacent clusters may merge.

For dynamic methods the pair cost is

\[
c_{ij} =
\alpha\, d_{\mathrm{dyn}}(z_i,z_j)
+\beta\,d_{\mathrm{geo}}(p_i,p_j).
\]

Mass is conserved and parallel contracted springs are aggregated by axial
stiffness:

\[
k_{IJ}=\sum_{e\in E_{IJ}} Y_e/L_e,\qquad
Y_{IJ}=k_{IJ}L_{IJ}.
\]

### Dense reconstruction

\[
P(t)=P(0)+W(Q(t)-Q(0)).
\]

The default map uses four graph-local reduced nodes per original physical
point.

### Level II: interaction reduction

After online/update residuals are observed, dense node error is projected to
the coarse graph and fused with stiffness and BT importance:

\[
s_e^{prior}
=
\lambda_{BT}s_e^{BT}
+(1-\lambda_{BT})s_e^{stiff},
\]

\[
s_e
=
(1-\lambda_{on})s_e^{prior}
+\lambda_{on}s_e^{err}.
\]

A maximum spanning forest and minimum-degree repair preserve graph structure.

## Status

The integration code is designed to be syntax-testable without PhysTwin.
Full end-to-end verification still requires your actual PhysTwin checkout,
data, checkpoints, CUDA/Warp environment, and Gaussian renderer.
