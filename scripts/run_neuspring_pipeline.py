#!/usr/bin/env python3
from __future__ import annotations

"""End-to-end NeuSpring recovery pipeline for PhysTwin hierarchical reduction.

Pipeline
--------
existing hierarchical run
    -> choose a source Stage-2 budget
    -> generate equal-budget prior + NeuSpring recovery candidates
    -> adapt every candidate on train/update frames only
    -> select the NeuSpring topology using train metrics only
    -> Joint NSF optimization on the automatically selected winner
    -> inference/evaluation, and optional FPS benchmark

Nothing in this file is tied to "Final54", cand_001, double_stretch_sloth, or a
specific best epoch.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(REPO / "src"))

from phystwin_reduction.neuspring import best_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generic NeuSpring recovery + adaptation + Joint NSF pipeline."
    )
    p.add_argument("--phystwin-root", required=True, type=Path)
    p.add_argument("--scene", required=True)
    p.add_argument("--base-path", type=Path)
    p.add_argument(
        "--hierarchical-run",
        type=Path,
        help=(
            "Existing run produced by run_hierarchical_pipeline.py. "
            "When omitted it is derived from node method and keep ratio."
        ),
    )
    p.add_argument("--node-method", default="soar")
    p.add_argument("--node-keep-ratio", type=float, default=0.75)
    p.add_argument(
        "--source-final-ratio",
        type=float,
        required=True,
        help="Existing Stage-2 budget to recover from, e.g. 0.50.",
    )
    target_group = p.add_mutually_exclusive_group(
        required=True
    )

    target_group.add_argument(
        "--target-final-ratios",
        type=float,
        nargs="+",
        help=(
            "Explicit recovery budgets, e.g. "
            "0.52 0.54 0.56 0.58."
        ),
    )

    target_group.add_argument(
        "--auto-budget-max-ratio",
        type=float,
        help=(
            "Automatically sweep recovery budgets from "
            "source_final_ratio + auto_budget_step up to "
            "this ratio."
        ),
    )

    p.add_argument(
        "--auto-budget-step",
        type=float,
        default=0.02,
        help="Budget sweep step. Default: 0.02.",
    )

    p.add_argument(
        "--auto-select-budget",
        action="store_true",
        help=(
            "Automatically select the final recovery budget "
            "using train-only geometry under an FPS constraint."
        ),
    )

    p.add_argument(
        "--min-fps-retention",
        type=float,
        default=0.95,
        help=(
            "Minimum Full-State FPS relative to source budget. "
            "Default 0.95 means at most 5%% FPS loss."
        ),
    )
    p.add_argument("--output-root", type=Path)

    p.add_argument("--num-candidates", type=int, default=6)
    p.add_argument("--num-regions", type=int, default=6)
    p.add_argument("--candidate-adapt-epochs", type=int, default=10)
    p.add_argument("--candidate-checkpoint-interval", type=int, default=2)

    p.add_argument("--joint-epochs", type=int, default=10)
    p.add_argument("--joint-lr", type=float, default=5e-5)
    p.add_argument("--joint-adam-eps", type=float, default=1e-8)
    p.add_argument("--joint-max-delta-log-y", type=float, default=0.10)
    p.add_argument("--joint-hidden-dim", type=int, default=128)
    p.add_argument("--joint-region-embed-dim", type=int, default=8)
    p.add_argument("--joint-grad-clip", type=float, default=1.0)
    p.add_argument(
        "--run-prior-joint",
        action="store_true",
        help="Also run Joint NSF on the prior-only control candidate.",
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--python", dest="python_bin", default=sys.executable)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--benchmark-fps", action="store_true")
    p.add_argument("--fps-repeats", type=int, default=5)
    return p.parse_args()


def run(cmd: list[object], *, env: dict[str, str]) -> None:
    printable = " ".join(str(x) for x in cmd)
    print("\n" + "=" * 100)
    print("$", printable)
    print("=" * 100)
    subprocess.run(
        [str(x) for x in cmd],
        check=True,
        cwd=str(REPO),
        env=env,
    )


def ratio_tag(ratio: float, prefix: str = "final") -> str:
    pct = ratio * 100.0
    if abs(pct - round(pct)) < 1e-9:
        suffix = str(int(round(pct)))
    else:
        suffix = f"{pct:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{prefix}{suffix}"


def target_dir_name(ratio: float) -> str:
    return f"target_{ratio:.6f}".rstrip("0").rstrip(".").replace(".", "p")

def build_target_ratios(
    source_ratio: float,
    explicit_ratios: list[float] | None,
    auto_max_ratio: float | None,
    step: float,
) -> list[float]:

    if explicit_ratios is not None:
        return sorted(
            {
                float(x)
                for x in explicit_ratios
            }
        )

    if auto_max_ratio is None:
        raise ValueError(
            "No target budget specification."
        )

    if step <= 0:
        raise ValueError(
            "--auto-budget-step must be positive."
        )

    if auto_max_ratio <= source_ratio:
        raise ValueError(
            "--auto-budget-max-ratio must be "
            "larger than source ratio."
        )

    ratios = []

    current = (
        source_ratio
        + step
    )

    while (
        current
        <= auto_max_ratio + 1e-9
    ):

        ratios.append(
            round(
                float(current),
                8,
            )
        )

        current += step

    if not ratios:
        raise RuntimeError(
            "Automatic budget sweep is empty."
        )

    return ratios


def read_fullstate_fps(
    path: Path,
) -> float:

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if "physics_plus_dense" not in data:
        raise KeyError(
            f"{path} has no physics_plus_dense result"
        )

    return float(
        data[
            "physics_plus_dense"
        ][
            "fps_mean"
        ]
    )


def select_recovery_budget(
    rows: list[dict],
    *,
    min_fps_retention: float,
) -> dict:

    feasible = [
        row
        for row in rows
        if (
            float(
                row["fps_retention"]
            )
            >= min_fps_retention
        )
    ]

    if not feasible:

        table = "\n".join(
            (
                f"ratio={row['target_final_ratio']:.4f}, "
                f"score={row['train_score']:.6f}, "
                f"fps={row['full_state_fps']:.3f}, "
                f"retention={row['fps_retention']:.4f}"
            )
            for row in rows
        )

        raise RuntimeError(
            "No recovery budget satisfies the FPS "
            f"retention constraint "
            f"{min_fps_retention:.3f}.\n"
            + table
        )

    # Main criterion:
    # best TRAIN geometry under FPS constraint.
    #
    # Tie-break:
    # smaller target ratio = fewer recovered springs.
    winner = min(
        feasible,
        key=lambda row: (
            float(
                row["train_score"]
            ),
            float(
                row["target_final_ratio"]
            ),
        ),
    )

    return winner

def require(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description}: {path}")
    return path


def load_candidates(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"No candidates in {manifest_path}")
    return candidates


def evaluate_runs(
    *,
    python_bin: str,
    root: Path,
    scene: str,
    runs: list[tuple[str, Path]],
    output_csv: Path,
    env: dict[str, str],
) -> None:
    cmd: list[object] = [
        python_bin,
        SCRIPTS / "evaluate_geometry.py",
        "--phystwin-root",
        root,
        "--scene",
        scene,
    ]
    for label, inference in runs:
        require(inference, f"inference for {label}")
        cmd += ["--run", f"{label}={inference}"]
    cmd += ["--output-csv", output_csv]
    run(cmd, env=env)


def read_geometry_scores(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    result: dict[str, dict] = {}
    for row in rows:
        method = str(row["Method"])
        cd_train = float(row["CD Train"])
        track_train = float(row["Track Error Train"])
        result[method] = {
            **row,
            "train_score": cd_train + track_train,
        }
    return result


def benchmark_one(
    *,
    python_bin: str,
    root: Path,
    scene: str,
    label: str,
    trainer: Path,
    checkpoint: Path,
    test_start: int,
    test_end: int,
    repeats: int,
    output: Path,
    env: dict[str, str],
) -> None:
    run(
        [
            python_bin,
            SCRIPTS / "benchmark_hierarchical_fps.py",
            "--phystwin-root",
            root,
            "--scene",
            scene,
            "--mode",
            "reduced",
            "--label",
            label,
            "--coarse-path",
            trainer,
            "--checkpoint",
            checkpoint,
            "--measure-start",
            test_start,
            "--measure-end",
            test_end,
            "--repeats",
            repeats,
            "--output",
            output,
        ],
        env=env,
    )


def run_joint_and_inference(
    *,
    a: argparse.Namespace,
    python_bin: str,
    root: Path,
    base_path: Path,
    train_end: int,
    candidate_dir: Path,
    base_checkpoint: Path,
    joint_dir: Path,
    env: dict[str, str],
) -> Path:
    joint_checkpoint = joint_dir / "best_joint_nsf.pth"
    joint_inference = joint_dir / "inference" / "inference.pkl"

    if not (a.resume and joint_checkpoint.is_file()):
        run(
            [
                python_bin,
                SCRIPTS / "run_joint_nsf_coarsening.py",
                "--phystwin-root",
                root,
                "--scene",
                a.scene,
                "--base-path",
                base_path,
                "--coarsened-data",
                candidate_dir / "trainer.npz",
                "--base-checkpoint",
                base_checkpoint,
                "--nsf-topology",
                candidate_dir / "nsf_topology.npz",
                "--output-dir",
                joint_dir,
                "--train-frame",
                train_end,
                "--epochs",
                a.joint_epochs,
                "--lr",
                a.joint_lr,
                "--adam-eps",
                a.joint_adam_eps,
                "--max-delta-log-y",
                a.joint_max_delta_log_y,
                "--hidden-dim",
                a.joint_hidden_dim,
                "--region-embed-dim",
                a.joint_region_embed_dim,
                "--num-regions",
                a.num_regions,
                "--grad-clip",
                a.joint_grad_clip,
                "--seed",
                a.seed,
                "--device",
                a.device,
            ],
            env=env,
        )

    if not (a.resume and joint_inference.is_file()):
        run(
            [
                python_bin,
                SCRIPTS / "run_coarsened_inference.py",
                "--phystwin-root",
                root,
                "--scene",
                a.scene,
                "--base-path",
                base_path,
                "--coarsened-data",
                candidate_dir / "trainer.npz",
                "--checkpoint",
                joint_checkpoint,
                "--output-dir",
                joint_dir / "inference",
                "--train-frame",
                train_end,
                "--device",
                a.device,
                "--seed",
                a.seed,
            ],
            env=env,
        )
    return joint_inference


def main() -> None:
    a = parse_args()
    target_ratios = build_target_ratios(
        source_ratio=float(
            a.source_final_ratio
        ),
        explicit_ratios=(
            a.target_final_ratios
            if a.target_final_ratios
            else None
        ),
        auto_max_ratio=(
            a.auto_budget_max_ratio
        ),
        step=float(
            a.auto_budget_step
        ),
    )

    auto_select_budget = (
        a.auto_select_budget
        or
        a.auto_budget_max_ratio is not None
    )

    print(
        "Recovery target budgets:",
        target_ratios,
    )

    print(
        "Automatic budget selection:",
        auto_select_budget,
    )

    if auto_select_budget:

        print(
            "Minimum FPS retention:",
            a.min_fps_retention,
        )
    if not 0.0 < a.source_final_ratio <= 1.0:
        raise ValueError("--source-final-ratio must be in (0,1]")
    if any(
        not 0.0 < ratio <= 1.0
        for ratio in target_ratios
    ):
        raise ValueError(
            "Every target final ratio must be in (0,1]"
        )
    if any(
        ratio <= a.source_final_ratio
        for ratio in target_ratios
    ):
        raise ValueError(
            "NeuSpring recovery restores springs, "
            "so every target ratio must be larger "
            "than --source-final-ratio"
        )


    root = require(a.phystwin_root, "PhysTwin root")
    base_path = (
        a.base_path.expanduser().resolve()
        if a.base_path is not None
        else root / "data" / "different_types"
    )
    scene_root = require(base_path / a.scene, "scene directory")

    split = json.loads((scene_root / "split.json").read_text(encoding="utf-8"))
    train_start, train_end = map(int, split["train"])
    test_start, test_end = map(int, split["test"])

    if a.hierarchical_run is not None:
        hierarchy = require(a.hierarchical_run, "hierarchical run")
    else:
        hierarchy = require(
            root
            / "results"
            / "hierarchical_v2"
            / a.scene
            / f"{a.node_method}_node{int(round(a.node_keep_ratio * 100))}",
            "derived hierarchical run",
        )

    full_topology = require(
        hierarchy / "full_stage1_topology.npz",
        "full Stage-1 topology",
    )
    node_trainer = require(hierarchy / "node" / "trainer.npz", "node trainer")
    node_retrain = require(hierarchy / "node" / "retrain", "node retrain directory")
    node_best = best_checkpoint(node_retrain)
    physical_inference = require(
        node_retrain / "inference_physical.pkl",
        "node physical inference",
    )

    source_tag = ratio_tag(a.source_final_ratio)
    source_stage2 = require(
        hierarchy / "stage2" / source_tag,
        "source Stage-2 run",
    )
    source_topology = require(source_stage2 / "topology.npz", "source topology")
    source_inference = require(
        source_stage2 / "inference" / "inference.pkl",
        "source inference",
    )

    output_root = (
        a.output_root.expanduser().resolve()
        if a.output_root is not None
        else root / "results" / "neuspring"
    )
    experiment_root = (
        output_root
        / a.scene
        / hierarchy.name
        / f"from_{source_tag}"
    )
    experiment_root.mkdir(parents=True, exist_ok=True)

    python_bin = str(Path(a.python_bin).expanduser())
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(REPO / "src"), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("SKIP_OPEN3D_VIDEO", "1")

    all_target_summaries: list[dict] = []
    budget_selection_rows: list[dict] = []

    source_selection_fps = None

    if auto_select_budget:

        source_trainer = require(
            source_stage2 / "trainer.npz",
            "source Stage-2 trainer",
        )

        source_checkpoint = require(
            source_stage2 / "initial.pth",
            "source Stage-2 checkpoint",
        )

        selection_fps_dir = (
            experiment_root
            / "budget_selection_fps"
        )

        selection_fps_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        source_fps_json = (
            selection_fps_dir
            / f"Source-{source_tag}.json"
        )

        # IMPORTANT:
        # selection FPS is measured on TRAIN interval,
        # not the test interval.
        fps_start = max(
            1,
            train_start,
        )

        if not (
            a.resume
            and source_fps_json.is_file()
        ):

            benchmark_one(
                python_bin=python_bin,
                root=root,
                scene=a.scene,
                label=f"Source-{source_tag}",
                trainer=source_trainer,
                checkpoint=source_checkpoint,
                test_start=fps_start,
                test_end=train_end,
                repeats=a.fps_repeats,
                output=source_fps_json,
                env=env,
            )

        source_selection_fps = (
            read_fullstate_fps(
                source_fps_json
            )
        )

        print(
            "\nSource Full-State FPS:",
            source_selection_fps,
        )

    for target_ratio in target_ratios:
        target_root = experiment_root / target_dir_name(target_ratio)
        candidates_root = target_root / "candidates"
        target_root.mkdir(parents=True, exist_ok=True)

        manifest = candidates_root / "candidate_manifest.json"
        if not (a.resume and manifest.is_file()):
            run(
                [
                    python_bin,
                    SCRIPTS / "build_neuspring_candidates.py",
                    "--stage1-trainer",
                    node_trainer,
                    "--node-checkpoint",
                    node_best,
                    "--source-stage2-topology",
                    source_topology,
                    "--physical-inference",
                    physical_inference,
                    "--full-topology",
                    full_topology,
                    "--output-dir",
                    candidates_root,
                    "--target-final-spring-ratio",
                    target_ratio,
                    "--selection-frame-start",
                    train_start,
                    "--selection-frame-end",
                    train_end,
                    "--num-candidates",
                    a.num_candidates,
                    "--num-regions",
                    a.num_regions,
                    "--seed",
                    a.seed,
                ],
                env=env,
            )

        candidates = load_candidates(manifest)
        label_to_candidate: dict[str, dict] = {}
        candidate_runs: list[tuple[str, Path]] = [
            (f"Source-{source_tag}", source_inference)
        ]

        for candidate in candidates:
            candidate_dir = Path(candidate["trainer"]).parent
            adapt_dir = candidate_dir / "adapt"
            adapt_inference = adapt_dir / "inference.pkl"
            if not (a.resume and adapt_inference.is_file()):
                run(
                    [
                        python_bin,
                        SCRIPTS / "run_coarsening_retrain.py",
                        "--phystwin_root",
                        root,
                        "--base_path",
                        base_path,
                        "--case_name",
                        a.scene,
                        "--coarsened_data",
                        candidate_dir / "trainer.npz",
                        "--parent_checkpoint",
                        candidate_dir / "initial.pth",
                        "--out_dir",
                        adapt_dir,
                        "--device",
                        a.device,
                        "--train_frame",
                        train_end,
                        "--retrain_epochs",
                        a.candidate_adapt_epochs,
                        "--checkpoint_interval",
                        a.candidate_checkpoint_interval,
                        "--seed",
                        a.seed,
                    ],
                    env=env,
                )

            if candidate["kind"] == "prior":
                label = "Prior-Adapt"
            else:
                label = f"NeuSpring-{candidate['candidate_id']}-Adapt"
            label_to_candidate[label] = candidate
            candidate_runs.append((label, adapt_inference))

        candidate_geometry = target_root / "candidate_geometry.csv"
        if not (a.resume and candidate_geometry.is_file()):
            evaluate_runs(
                python_bin=python_bin,
                root=root,
                scene=a.scene,
                runs=candidate_runs,
                output_csv=candidate_geometry,
                env=env,
            )

        geometry_scores = read_geometry_scores(candidate_geometry)
        neuspring_labels = [
            label
            for label, candidate in label_to_candidate.items()
            if candidate["kind"] == "neuspring"
        ]
        winner_label = min(
            neuspring_labels,
            key=lambda label: geometry_scores[label]["train_score"],
        )
        winner_candidate = label_to_candidate[winner_label]
        winner_dir = Path(winner_candidate["trainer"]).parent
        winner_adapt_dir = winner_dir / "adapt"
        winner_best = best_checkpoint(winner_adapt_dir)

        prior_candidate = next(
            candidate for candidate in candidates if candidate["kind"] == "prior"
        )
        prior_dir = Path(prior_candidate["trainer"]).parent
        prior_adapt_dir = prior_dir / "adapt"
        prior_best = best_checkpoint(prior_adapt_dir)

        selection = {
            "protocol": "train_only_neuspring_topology_selection",
            "target_final_spring_ratio": float(target_ratio),
            "selection_metric": "CD Train + Track Error Train",
            "test_used_for_selection": False,
            "winner_label": winner_label,
            "winner_candidate_id": winner_candidate["candidate_id"],
            "winner_directory": str(winner_dir),
            "winner_train_score": float(
                geometry_scores[winner_label]["train_score"]
            ),
            "winner_adapt_checkpoint": str(winner_best),
            "prior_control_directory": str(prior_dir),
            "prior_adapt_checkpoint": str(prior_best),
        }
        (target_root / "selection.json").write_text(
            json.dumps(selection, indent=2),
            encoding="utf-8",
        )

        winner_joint_dir = winner_dir / "joint_nsf"
        winner_joint_inference = run_joint_and_inference(
            a=a,
            python_bin=python_bin,
            root=root,
            base_path=base_path,
            train_end=train_end,
            candidate_dir=winner_dir,
            base_checkpoint=winner_best,
            joint_dir=winner_joint_dir,
            env=env,
        )

        prior_joint_inference: Path | None = None
        if a.run_prior_joint:
            prior_joint_inference = run_joint_and_inference(
                a=a,
                python_bin=python_bin,
                root=root,
                base_path=base_path,
                train_end=train_end,
                candidate_dir=prior_dir,
                base_checkpoint=prior_best,
                joint_dir=prior_dir / "joint_nsf",
                env=env,
            )

        final_runs: list[tuple[str, Path]] = [
            (f"Source-{source_tag}", source_inference),
            ("Prior-Adapt", prior_adapt_dir / "inference.pkl"),
            ("NeuSpring-Winner-Adapt", winner_adapt_dir / "inference.pkl"),
            ("NeuSpring-Winner-JointNSF", winner_joint_inference),
        ]
        if prior_joint_inference is not None:
            final_runs.insert(3, ("Prior-JointNSF", prior_joint_inference))

        final_geometry = target_root / "final_geometry.csv"
        evaluate_runs(
            python_bin=python_bin,
            root=root,
            scene=a.scene,
            runs=final_runs,
            output_csv=final_geometry,
            env=env,
        )
        if auto_select_budget:

            final_scores = read_geometry_scores(
                final_geometry
            )

            final_label = (
                "NeuSpring-Winner-JointNSF"
            )

            if final_label not in final_scores:

                raise KeyError(
                    f"{final_label} missing from "
                    f"{final_geometry}"
                )

            train_score = float(
                final_scores[
                    final_label
                ][
                    "train_score"
                ]
            )


            # ========================================================
            # TRAIN-INTERVAL Full-State FPS
            # ========================================================

            selection_fps_dir = (
                experiment_root
                / "budget_selection_fps"
            )

            target_fps_json = (
                selection_fps_dir
                / (
                    f"target_"
                    f"{target_ratio:.4f}"
                    f"_joint.json"
                )
            )

            if not (
                a.resume
                and target_fps_json.is_file()
            ):

                benchmark_one(
                    python_bin=python_bin,
                    root=root,
                    scene=a.scene,
                    label=(
                        f"Recovery-{target_ratio:.4f}"
                    ),
                    trainer=(
                        winner_dir
                        / "trainer.npz"
                    ),
                    checkpoint=(
                        winner_joint_dir
                        / "best_joint_nsf.pth"
                    ),
                    test_start=max(
                        1,
                        train_start,
                    ),
                    test_end=train_end,
                    repeats=a.fps_repeats,
                    output=target_fps_json,
                    env=env,
                )

            target_fps = read_fullstate_fps(
                target_fps_json
            )

            fps_retention = (
                target_fps
                / source_selection_fps
            )

            budget_selection_rows.append(
                {
                    "target_final_ratio":
                        float(
                            target_ratio
                        ),

                    "winner_candidate_id":
                        winner_candidate[
                            "candidate_id"
                        ],

                    "train_score":
                        train_score,

                    "full_state_fps":
                        float(
                            target_fps
                        ),

                    "source_full_state_fps":
                        float(
                            source_selection_fps
                        ),

                    "fps_retention":
                        float(
                            fps_retention
                        ),

                    "fps_feasible":
                        bool(
                            fps_retention
                            >=
                            a.min_fps_retention
                        ),

                    "winner_directory":
                        str(
                            winner_dir
                        ),

                    "joint_checkpoint":
                        str(
                            winner_joint_dir
                            / "best_joint_nsf.pth"
                        ),

                    "joint_inference":
                        str(
                            winner_joint_inference
                        ),

                    "final_geometry":
                        str(
                            final_geometry
                        ),
                }
            )

        fps_outputs: dict[str, str] = {}
        if a.benchmark_fps:
            fps_dir = target_root / "fps"
            fps_dir.mkdir(parents=True, exist_ok=True)
            benchmark_models: list[tuple[str, Path, Path]] = [
                (
                    "Prior-Adapt",
                    prior_dir / "trainer.npz",
                    prior_best,
                ),
                (
                    "NeuSpring-Winner-Adapt",
                    winner_dir / "trainer.npz",
                    winner_best,
                ),
                (
                    "NeuSpring-Winner-JointNSF",
                    winner_dir / "trainer.npz",
                    winner_joint_dir / "best_joint_nsf.pth",
                ),
            ]
            if a.run_prior_joint:
                benchmark_models.append(
                    (
                        "Prior-JointNSF",
                        prior_dir / "trainer.npz",
                        prior_dir / "joint_nsf" / "best_joint_nsf.pth",
                    )
                )
            for label, trainer, checkpoint in benchmark_models:
                output = fps_dir / f"{label}.json"
                if not (a.resume and output.is_file()):
                    benchmark_one(
                        python_bin=python_bin,
                        root=root,
                        scene=a.scene,
                        label=label,
                        trainer=trainer,
                        checkpoint=checkpoint,
                        test_start=test_start,
                        test_end=test_end,
                        repeats=a.fps_repeats,
                        output=output,
                        env=env,
                    )
                fps_outputs[label] = str(output)

        target_summary = {
            "protocol": "generic_neuspring_recovery_joint_nsf",
            "scene": a.scene,
            "hierarchical_run": str(hierarchy),
            "source_final_ratio": float(a.source_final_ratio),
            "target_final_ratio": float(target_ratio),
            "train_frames": [train_start, train_end],
            "test_frames": [test_start, test_end],
            "candidate_manifest": str(manifest),
            "candidate_geometry": str(candidate_geometry),
            "selection": selection,
            "winner_joint_checkpoint": str(
                winner_joint_dir / "best_joint_nsf.pth"
            ),
            "winner_joint_inference": str(winner_joint_inference),
            "final_geometry": str(final_geometry),
            "fps": fps_outputs,
        }
        (target_root / "summary.json").write_text(
            json.dumps(target_summary, indent=2),
            encoding="utf-8",
        )
        all_target_summaries.append(target_summary)

        print("\n" + "#" * 100)
        print("TARGET FINISHED:", target_ratio)
        print("winner:", winner_candidate["candidate_id"])
        print("selection metric:", selection["selection_metric"])
        print("final geometry:", final_geometry)
        print("#" * 100)
    selected_budget = None

    if auto_select_budget:

        selected_budget = (
            select_recovery_budget(
                budget_selection_rows,
                min_fps_retention=float(
                    a.min_fps_retention
                ),
            )
        )

        # ========================================================
        # Save CSV
        # ========================================================

        selection_csv = (
            experiment_root
            / "budget_selection.csv"
        )

        fields = [
            "target_final_ratio",
            "winner_candidate_id",
            "train_score",
            "full_state_fps",
            "source_full_state_fps",
            "fps_retention",
            "fps_feasible",
            "winner_directory",
            "joint_checkpoint",
            "joint_inference",
            "final_geometry",
        ]

        with selection_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()

            for row in budget_selection_rows:
                writer.writerow(
                    {
                        key:
                            row[key]
                        for key in fields
                    }
                )


        # ========================================================
        # Save winner JSON
        # ========================================================

        selection_json = {
            "protocol":
                "automatic_budget_aware_neuspring_recovery",

            "selection_data":
                "train_only",

            "test_metrics_used_for_selection":
                False,

            "selection_metric":
                (
                    "minimize "
                    "CD Train + Track Error Train "
                    "subject to Full-State FPS retention"
                ),

            "source_final_ratio":
                float(
                    a.source_final_ratio
                ),

            "candidate_target_ratios":
                list(
                    map(
                        float,
                        target_ratios,
                    )
                ),

            "minimum_fps_retention":
                float(
                    a.min_fps_retention
                ),

            "source_full_state_fps":
                float(
                    source_selection_fps
                ),

            "selected":
                selected_budget,

            "all_candidates":
                budget_selection_rows,
        }

        selected_json_path = (
            experiment_root
            / "selected_budget.json"
        )

        selected_json_path.write_text(
            json.dumps(
                selection_json,
                indent=2,
            ),
            encoding="utf-8",
        )


        print(
            "\n"
            + "=" * 100
        )

        print(
            "AUTOMATIC RECOVERY BUDGET SELECTED"
        )

        print(
            "=" * 100
        )

        print(
            "ratio        :",
            selected_budget[
                "target_final_ratio"
            ],
        )

        print(
            "candidate    :",
            selected_budget[
                "winner_candidate_id"
            ],
        )

        print(
            "train score  :",
            selected_budget[
                "train_score"
            ],
        )

        print(
            "FullState FPS:",
            selected_budget[
                "full_state_fps"
            ],
        )

        print(
            "FPS retention:",
            selected_budget[
                "fps_retention"
            ],
        )

        print(
            "checkpoint   :",
            selected_budget[
                "joint_checkpoint"
            ],
        )

        print(
            "=" * 100
        )
    overall = {

        "protocol":
            "generic_neuspring_multi_budget_sweep",

        "scene":
            a.scene,

        "source_final_ratio":
            float(
                a.source_final_ratio
            ),

        "target_final_ratios":
            list(
                map(
                    float,
                    target_ratios,
                )
            ),

        "automatic_budget_selection":
            bool(
                auto_select_budget
            ),

        "minimum_fps_retention":
            (
                float(
                    a.min_fps_retention
                )
                if auto_select_budget
                else None
            ),

        "selected_budget":
            selected_budget,

        "targets":
            all_target_summaries,
    }
    (experiment_root / "summary.json").write_text(
        json.dumps(overall, indent=2),
        encoding="utf-8",
    )
    print("\n[DONE]", experiment_root / "summary.json")


if __name__ == "__main__":
    main()
