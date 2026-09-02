# Gaussian Render Refinement

This stage reproduces the RGB / Opacity / Scale experiments without any
`Final54`, `cand_001`, `double_stretch_sloth`, or Colab-specific path in the
algorithm.

## Purpose

The stage is applied **after** a physical reduced model has already produced an
`inference.pkl` trajectory.

The physical system stays frozen:

- node topology: unchanged;
- spring topology: unchanged;
- spring stiffness: unchanged;
- physical trajectory: unchanged;
- physics FPS: unchanged by this optimization stage.

Only Gaussian rendering parameters are updated.

## Variants

| Variant | Trainable Gaussian parameters |
|---|---|
| `rgb_only` | `_features_dc` |
| `rgb_opacity` | `_features_dc`, `_opacity` |
| `rgb_opa_scale` | `_features_dc`, `_opacity`, `_scaling` |

`_xyz`, `_rotation`, and `_features_rest` remain frozen.  Dynamic `_xyz` and
`_rotation` are reconstructed from the supplied PhysTwin trajectory.

## Files

```text
src/phystwin_reduction/render_refinement.py
scripts/finetune_gaussian_render.py
scripts/evaluate_rendering_multiview.py
scripts/run_render_refinement_pipeline.py
tests/smoke_test_render_refinement.py
```

## Direct use with any inference trajectory

```bash
python scripts/run_render_refinement_pipeline.py \
  --phystwin-root ~/PhysTwin \
  --scene <scene> \
  --inference /path/to/inference.pkl \
  --run-name recovered_model \
  --variants rgb_only rgb_opacity rgb_opa_scale \
  --steps 200 \
  --num-train-frames 24 \
  --render \
  --evaluate
```

The default output is:

```text
<PhysTwin>/results/render_refinement/<scene>/<run-name>/
├── dynamic_pose_cache.pt
├── variants/
│   ├── rgb_only/
│   │   ├── gaussian_model/
│   │   ├── train_history.csv
│   │   └── summary.json
│   ├── rgb_opacity/
│   └── rgb_opa_scale/
├── renders/
│   ├── baseline/
│   ├── rgb_only/
│   ├── rgb_opacity/
│   └── rgb_opa_scale/
├── render_status.csv
├── render_metrics_multiview.csv
└── summary.json
```

## Use directly after NeuSpring

`run_neuspring_pipeline.py` writes one `target_*/summary.json`.  Render
refinement can resolve a physical trajectory from that summary:

```bash
python scripts/run_render_refinement_pipeline.py \
  --phystwin-root ~/PhysTwin \
  --scene <scene> \
  --neuspring-summary /path/to/target_x/summary.json \
  --summary-model prior-adapt \
  --run-name recovered_prior \
  --render \
  --evaluate
```

Available `--summary-model` values:

- `prior-adapt`
- `prior-joint`
- `winner-adapt`
- `winner-joint`

This is intentionally not tied to which candidate happened to win in one
experiment.  `winner-*` is resolved from `selection.json`/`summary.json`.

## Train/test protocol

Training frames are read from:

```text
data/different_types/<scene>/split.json
```

Only the train interval is used for Gaussian optimization.  Test frames are
used only by `evaluate_rendering_multiview.py`.

The training script writes:

```json
{
  "physics_trajectory_frozen": true,
  "physics_topology_frozen": true,
  "test_frames_used_for_optimization": false
}
```

into each variant summary.

## Loss

The default objective is:

```text
L =
    L1_RGB
  + lambda_ssim * (1 - SSIM)
  + lambda_alpha * L1_alpha
  + lambda_reg * parameter_deviation
```

Defaults:

```text
lr_rgb       = 1e-3
lr_opacity   = 5e-4
lr_scale     = 2e-4
lambda_ssim  = 0.20
lambda_alpha = 0.10
lambda_reg   = 1e-4
```

These are command-line parameters, not hard-coded experimental cases.

## Dynamic Gaussian pose

The Gaussian center and quaternion at the selected train frames are propagated
from the supplied PhysTwin `inference.pkl`, using the same KNN motion
interpolation logic as `gs_render_dynamics.py`.

A pose cache is created once and reused by all three variants.

## Rendering evaluation

The new evaluator preserves the view dimension.  It evaluates all configured
view folders instead of merging files with identical frame numbers across
different cameras.

Metrics:

- PSNR
- SSIM
- LPIPS
- foreground IoU

## Smoke test

The CPU-only integration smoke test does not require PhysTwin:

```bash
python tests/smoke_test_render_refinement.py
```

The actual fine-tuning requires the PhysTwin Gaussian renderer and CUDA
environment.
