#!/usr/bin/env python3
"""Legacy Stage-2 entry point intentionally disabled.

The old workflow accepted an *additional* spring keep ratio and could be called
on the pre-retraining node topology.  That is not the formal method anymore.

Use ``build_final_budget_stage2.py`` instead.  It requires
``topology_retrained.npz``, recomputes BT/stiffness on the retrained coarse
dynamics, fuses projected update error, preserves connectivity, and targets a
FINAL spring ratio relative to the full graph.
"""

raise SystemExit(
    "Legacy Stage-2 script disabled. Use scripts/build_final_budget_stage2.py "
    "with --retrained-topology and --target-final-spring-ratio."
)
