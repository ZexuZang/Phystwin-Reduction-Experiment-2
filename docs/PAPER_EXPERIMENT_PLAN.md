# Paper Experiment Plan

## Main method

**Ours = SOAR node coarsening + online BT-guided spring pruning**

## Node ablation

Hold spring pruning disabled after graph contraction.

| Node method | Node keep | CD | Track | Sim FPS |
|---|---:|---:|---:|---:|
| Full | 1.00 | | | |
| Random | 0.50 | | | |
| Geometry | 0.50 | | | |
| Trajectory | 0.50 | | | |
| Krylov | 0.50 | | | |
| SOAR | 0.50 | | | |

## Spring ablation

Fix the same SOAR-coarsened graph.

| Spring method | Spring keep | CD | Track | Sim FPS |
|---|---:|---:|---:|---:|
| no extra pruning | 1.00 | | | |
| random | 0.50 | | | |
| stiffness | 0.50 | | | |
| BT | 0.50 | | | |
| online stiffness | 0.50 | | | |
| online BT | 0.50 | | | |

## Hierarchical ablation

| Method | Node keep | Spring keep | CD | Track | PSNR | FPS |
|---|---:|---:|---:|---:|---:|---:|
| Full | 1.0 | 1.0 | | | | |
| Spring only | 1.0 | 0.5 | | | | |
| Node only | 0.5 | 1.0 | | | | |
| Node + stiffness | 0.5 | 0.5 | | | | |
| Node + BT | 0.5 | 0.5 | | | | |
| Node + online BT (Ours) | 0.5 | 0.5 | | | | |

## Ratio sweep

Node keep:
- 1.00
- 0.75
- 0.50

Additional spring keep after coarsening:
- 1.00
- 0.70
- 0.50
- 0.30

Select the paper operating point by Pareto trade-off, not by assuming
50% nodes + 30% springs is best.

## Split discipline

For every scene record exact absolute frame ranges:

```text
TRAIN/signature: [...]
UPDATE/error:     [...]
TEST/evaluation:  [...]
```

No test frame may enter:
- final displacement seed;
- trajectory signature;
- node protection;
- BT inputs chosen from observations;
- online error;
- topology hyperparameter selection.
