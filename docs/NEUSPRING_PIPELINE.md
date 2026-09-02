# Generic NeuSpring Recovery Pipeline

This document describes the reproducible NeuSpring path integrated into
`Phystwin-Reduction-Experiment-2`.

## What was removed from the notebook workflow

The notebook experiments contained several one-off assumptions:

- fixed scene paths such as `/content/PhysTwin/.../double_stretch_sloth`;
- fixed frame numbers such as `134` and `192`;
- fixed budget names such as `Final54`;
- fixed winner directories such as `cand_001_neuspring`;
- selecting `max(best_*.pth)` instead of reading the actual training history;
- importing NeuSpring feature code from another checkout under
  `/content/phystwin-neuspring-exp`;
- repeated Colab-only `subprocess` cells.

The Python implementation removes these assumptions.

## Files

- `src/phystwin_reduction/neuspring.py`
  - NeuSpring edge features;
  - lightweight region clustering;
  - recovery candidate scoring;
  - exact-budget spring restoration;
  - Joint residual NSF model;
  - best-checkpoint resolver.

- `scripts/build_neuspring_candidates.py`
  - generates one prior-only control and N NeuSpring candidates at exactly the
    same target spring budget;
  - uses only the requested train/update trajectory interval;
  - saves `candidate_manifest.json`.

- `scripts/run_joint_nsf_coarsening.py`
  - optimizes a zero-initialized residual NSF through the PhysTwin simulation
    loss;
  - manually bridges Warp gradients back into PyTorch;
  - bakes ordinary PhysTwin checkpoints.

- `scripts/run_neuspring_pipeline.py`
  - end-to-end orchestration;
  - reads `split.json` automatically;
  - finds the actual best node checkpoint;
  - adapts every candidate;
  - selects the NeuSpring winner using **CD Train + Track Error Train only**;
  - never uses test metrics for topology selection;
  - runs Joint NSF on the automatically selected winner;
  - optionally benchmarks FPS.

- `tests/smoke_test_neuspring.py`
  - CPU-only test for exact budget, feature construction, and zero-init Joint NSF.

## Prerequisite

First run the hierarchical pipeline so that this structure exists:

```text
results/hierarchical_v2/<scene>/<method>_nodeXX/
├── full_stage1_topology.npz
├── node/
│   ├── trainer.npz
│   └── retrain/
│       ├── inference.pkl
│       ├── inference_physical.pkl
│       └── train/best_*.pth
└── stage2/
    └── finalXX/
        ├── topology.npz
        ├── trainer.npz
        ├── initial.pth
        └── inference/inference.pkl
```

## Reproduce a single target budget

For example, to reproduce the experiment that happens to use a 0.54 final
spring ratio:

```bash
python scripts/run_neuspring_pipeline.py \
  --phystwin-root ~/PhysTwin \
  --scene double_stretch_sloth \
  --source-final-ratio 0.50 \
  --target-final-ratios 0.54 \
  --run-prior-joint \
  --benchmark-fps
```

`0.54` is only a command-line experiment parameter.  There is no `Final54`
branch in the code and no `cand_001` assumption.

## Sweep multiple budgets

```bash
python scripts/run_neuspring_pipeline.py \
  --phystwin-root ~/PhysTwin \
  --scene double_stretch_sloth \
  --source-final-ratio 0.50 \
  --target-final-ratios 0.52 0.54 0.56 0.58 \
  --run-prior-joint \
  --benchmark-fps \
  --resume
```

Every target gets an independent directory, candidate manifest, selection
record, final geometry table, and optional FPS results.

## Important protocol rule

Topology construction and candidate adaptation use the scene's training interval
from `split.json`.  The test interval is used only after the winner has already
been selected.

The pipeline writes this explicitly into `selection.json`:

```json
{
  "selection_metric": "CD Train + Track Error Train",
  "test_used_for_selection": false
}
```

## Output structure

```text
results/neuspring/<scene>/<hierarchical-run>/from_finalXX/
└── target_0p54/
    ├── candidates/
    │   ├── candidate_manifest.json
    │   ├── cand_000_prior/
    │   ├── cand_001_neuspring/
    │   └── ...
    ├── candidate_geometry.csv
    ├── selection.json
    ├── final_geometry.csv
    ├── fps/
    └── summary.json
```

The candidate number is not treated as the winner.  The winning directory is
read from the train-only evaluation and recorded in `selection.json`.

## Smoke test

```bash
python tests/smoke_test_neuspring.py
```

This test does not require PhysTwin or a GPU.
