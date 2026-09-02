#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from phystwin_reduction.render_refinement import (
    RENDER_VARIANTS,
    resolve_inference_from_neuspring_summary,
    slug,
    uniform_train_frames,
    validate_variants,
)


def main() -> None:
    assert validate_variants(RENDER_VARIANTS) == list(RENDER_VARIANTS)
    assert slug("Recovered 54 + RGB") == "Recovered_54_RGB"

    split = {"train": [0, 20], "test": [20, 30]}
    sampled = uniform_train_frames(split, 5)
    assert sampled[0] == 1
    assert sampled[-1] == 19
    assert len(sampled) == 5

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        winner = root / "cand_003_neuspring"
        prior = root / "cand_000_prior"

        for path in [
            winner / "adapt" / "inference.pkl",
            winner / "joint_nsf" / "inference" / "inference.pkl",
            prior / "adapt" / "inference.pkl",
            prior / "joint_nsf" / "inference" / "inference.pkl",
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test")

        summary = {
            "selection": {
                "winner_directory": str(winner),
                "prior_control_directory": str(prior),
            },
            "winner_joint_inference": str(
                winner / "joint_nsf" / "inference" / "inference.pkl"
            ),
        }
        summary_path = root / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        assert resolve_inference_from_neuspring_summary(
            summary_path, "winner-adapt"
        ) == (winner / "adapt" / "inference.pkl").resolve()
        assert resolve_inference_from_neuspring_summary(
            summary_path, "winner-joint"
        ) == (
            winner / "joint_nsf" / "inference" / "inference.pkl"
        ).resolve()
        assert resolve_inference_from_neuspring_summary(
            summary_path, "prior-adapt"
        ) == (prior / "adapt" / "inference.pkl").resolve()
        assert resolve_inference_from_neuspring_summary(
            summary_path, "prior-joint"
        ) == (
            prior / "joint_nsf" / "inference" / "inference.pkl"
        ).resolve()

    print("[OK] render variants")
    print("[OK] generic train-frame sampling")
    print("[OK] NeuSpring summary trajectory resolution")
    print("[OK] no fixed scene/budget/candidate required")


if __name__ == "__main__":
    main()
