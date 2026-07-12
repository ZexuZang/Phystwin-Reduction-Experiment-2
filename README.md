# PhysTwin Reduction Experiment 2

Server-ready Python implementation of the notebook
`pruning_strategy2_spring_and_node.ipynb`.

This experiment combines two reduction directions:

1. **Node reduction** through trajectory-aware neighboring-node clustering.
2. **Spring reduction** through online-error-aware stiffness and BT/PRBT scores.

The Colab notebook has been replaced with independent `.py` scripts. There are
no `%cd`, `!pip`, Google Drive mounts, or `/content/PhysTwin` paths in the
experiment code.

## Workflow

```text
Train first 50% of training frames
        ↓
Export Stage-1 topology
        ↓
Roll out Stage-1 model over the sequence
        ↓
Compute online per-node error
        ├── Neighbor-only node clustering
        │       ↓
        │   Reduced-node inference
        │       ↓
        │   Reconstruct original-node trajectory
        │
        ├── Online stiffness spring pruning
        │
        └── Online BT/PRBT-guided spring pruning
                ↓
        Final inference, rendering and evaluation
```

## Project structure

```text
Phystwin-Reduction-Experiment-2/
├── configs/default_experiment.yaml
├── scripts/
│   ├── check_environment.py
│   ├── patch_phystwin.py
│   ├── train_stage1.py
│   ├── export_stage1_topology.py
│   ├── run_external_topology_inference.py
│   ├── compute_online_node_error.py
│   ├── generate_node_cluster_topology.py
│   ├── reconstruct_node_cluster_trajectory.py
│   ├── generate_online_spring_topologies.py
│   ├── run_strategy2_pipeline.py
│   ├── evaluate_geometry.py
│   ├── render_methods.py
│   ├── evaluate_rendering.py
│   ├── benchmark_simulation.py
│   └── make_visual_videos.py
├── src/phystwin_reduction/
├── requirements.txt
└── environment.yml
```

## 1. Clone on the server

```bash
cd ~

git clone https://github.com/Jianghanxiao/PhysTwin.git
git clone https://github.com/ZexuZang/Phystwin-Reduction-Experiment-2.git

cd ~/PhysTwin
git checkout gradio_playground
git submodule update --init --recursive
```

The repository URL for Experiment 2 is an example until you create that GitHub
repository.

## 2. Create the environment

Use a PyTorch version matching the server CUDA driver. Do not blindly copy the
Colab Python 3.12 installation cell.

```bash
conda create -n phystwin-strategy2 python=3.10 -y
conda activate phystwin-strategy2

# Example for CUDA 12.1. Change this when the server uses another CUDA version.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

cd ~/Phystwin-Reduction-Experiment-2
pip install -r requirements.txt
python scripts/check_environment.py
```

Install PhysTwin Gaussian custom operators:

```bash
cd ~/PhysTwin/gaussian_splatting/submodules/diff-gaussian-rasterization
pip install --no-build-isolation .

cd ~/PhysTwin/gaussian_splatting/submodules/simple-knn
pip install --no-build-isolation .
```

## 3. Expected PhysTwin data

```text
~/PhysTwin/
├── data/different_types/double_stretch_sloth/
│   ├── split.json
│   ├── final_data.pkl
│   ├── calibrate.pkl
│   ├── metadata.json
│   ├── color/
│   └── mask/
├── data/gaussian_data/double_stretch_sloth/
├── experiments_optimization/double_stretch_sloth/optimal_params.pkl
├── gaussian_output/double_stretch_sloth/
└── results/
```

## 4. Patch PhysTwin once

External topology inference requires `trainer_warp.py` to read the environment
variable `EXTERNAL_TOPOLOGY_NPZ`.

```bash
cd ~/Phystwin-Reduction-Experiment-2

python scripts/patch_phystwin.py \
  --phystwin-root ~/PhysTwin
```

When optional PyTorch3D imports prevent the code from starting:

```bash
python scripts/patch_phystwin.py \
  --phystwin-root ~/PhysTwin \
  --patch-pytorch3d
```

The patch creates backups and can be run repeatedly.

## 5. Run the complete pipeline

```bash
cd ~/Phystwin-Reduction-Experiment-2
conda activate phystwin-strategy2

python scripts/run_strategy2_pipeline.py \
  --phystwin-root ~/PhysTwin \
  --scene double_stretch_sloth \
  --stage1-ratio 0.5 \
  --node-keep-ratio 0.8 \
  --max-cluster-size 8 \
  --spring-keep-ratio 0.3 \
  --online-error-weight 0.3 \
  --bt-weight 0.7
```

The default values reproduce the main settings found in the notebook.

For a long server run:

```bash
tmux new -s phystwin-strategy2

conda activate phystwin-strategy2
cd ~/Phystwin-Reduction-Experiment-2

python scripts/run_strategy2_pipeline.py \
  --phystwin-root ~/PhysTwin \
  --scene double_stretch_sloth
```

Detach with `Ctrl+B`, then `D`. Reconnect with:

```bash
tmux attach -t phystwin-strategy2
```

## 6. Reuse an existing Stage-1 checkpoint

```bash
python scripts/run_strategy2_pipeline.py \
  --phystwin-root ~/PhysTwin \
  --scene double_stretch_sloth \
  --skip-stage1-training \
  --stage1-model-path ~/PhysTwin/results/online_adaptation/double_stretch_sloth/stage1_train_50/train/iter_80.pth
```

## 7. Run stages separately

### Stage-1 training

```bash
python scripts/train_stage1.py \
  --phystwin-root ~/PhysTwin \
  --scene double_stretch_sloth \
  --stage1-ratio 0.5
```

### Export Stage-1 topology

```bash
python scripts/export_stage1_topology.py \
  --phystwin-root ~/PhysTwin \
  --scene double_stretch_sloth \
  --model-path /absolute/path/to/stage1_checkpoint.pth
```

### Stage-1 rollout

```bash
python scripts/run_external_topology_inference.py \
  --phystwin-root ~/PhysTwin \
  --scene double_stretch_sloth \
  --train-frame 67 \
  --model-path /absolute/path/to/stage1_checkpoint.pth \
  --topology-path ~/PhysTwin/results/online_adaptation/double_stretch_sloth/topologies/stage1_official_topology.npz \
  --output-dir ~/PhysTwin/results/online_adaptation/double_stretch_sloth/runs/stage1_full_rollout
```

Do not copy `67` blindly. The complete pipeline reads the correct split and
calculates this value automatically.

### Compute online node error

```bash
python scripts/compute_online_node_error.py \
  --phystwin-root ~/PhysTwin \
  --scene double_stretch_sloth \
  --inference-path ~/PhysTwin/results/online_adaptation/double_stretch_sloth/runs/stage1_full_rollout/inference.pkl \
  --topology-path ~/PhysTwin/results/online_adaptation/double_stretch_sloth/topologies/stage1_official_topology.npz
```

### Generate node-cluster topology

```bash
python scripts/generate_node_cluster_topology.py \
  --phystwin-root ~/PhysTwin \
  --scene double_stretch_sloth \
  --topology-path ~/PhysTwin/results/online_adaptation/double_stretch_sloth/topologies/stage1_official_topology.npz \
  --inference-path ~/PhysTwin/results/online_adaptation/double_stretch_sloth/runs/stage1_full_rollout/inference.pkl \
  --node-error-path ~/PhysTwin/results/online_adaptation/double_stretch_sloth/node_error_online.npz \
  --node-keep-ratio 0.8 \
  --max-cluster-size 8
```

### Reconstruct the original-node trajectory

```bash
python scripts/reconstruct_node_cluster_trajectory.py \
  --reduced-inference-path /path/to/node_cluster/inference.pkl \
  --topology-path /path/to/online_cluster_node_keep_80.npz \
  --output-path /path/to/reconstructed/inference.pkl
```

### Generate online spring topologies

```bash
python scripts/generate_online_spring_topologies.py \
  --topology-path /path/to/stage1_official_topology.npz \
  --node-error-path /path/to/node_error_online.npz \
  --output-dir /path/to/topologies \
  --keep-ratio 0.3 \
  --online-error-weight 0.3 \
  --bt-weight 0.7
```

## 8. Evaluate geometry and tracking

```bash
python scripts/evaluate_geometry.py \
  --phystwin-root ~/PhysTwin \
  --scene double_stretch_sloth
```

You may register custom inference files:

```bash
python scripts/evaluate_geometry.py \
  --phystwin-root ~/PhysTwin \
  --scene double_stretch_sloth \
  --run "My Method=/absolute/path/to/inference.pkl"
```

## 9. Render and evaluate images

```bash
python scripts/render_methods.py \
  --phystwin-root ~/PhysTwin \
  --scene double_stretch_sloth

python scripts/evaluate_rendering.py \
  --phystwin-root ~/PhysTwin \
  --scene double_stretch_sloth
```

Create videos:

```bash
python scripts/make_visual_videos.py \
  --phystwin-root ~/PhysTwin \
  --scene double_stretch_sloth \
  --frame-mode test
```


## Standalone smoke test

This test does not require PhysTwin or a GPU. It validates node error,
node clustering, trajectory reconstruction, and connected spring pruning on a
small synthetic topology.

```bash
python tests/smoke_test.py
```

## Important conversion notes

- The Notebook cell order is no longer used as hidden state.
- All experiment paths are derived from `--phystwin-root`.
- Stage boundaries are read from the scene's `split.json`.
- Node clustering preserves controller-attached and high-error nodes by default.
- Controller nodes are retained during node reduction.
- Reduced node trajectories are reconstructed to the original node count before
  Gaussian rendering.
- Spring pruning keeps a maximum spanning forest and enforces minimum degree.
- PRBT is attempted first; BT is used when PRBT fails.
- The generated project passed Python syntax checks and standalone topology
  tests. It has not been executed against your actual offline server, PhysTwin
  checkout, dataset, CUDA driver, or checkpoints.
