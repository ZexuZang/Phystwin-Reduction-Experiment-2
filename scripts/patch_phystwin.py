#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


INIT_SENTINEL = "[EXTERNAL_TOPOLOGY_NPZ:init]"
TEST_SENTINEL = "[EXTERNAL_TOPOLOGY_NPZ:test]"


INIT_PATCH = r'''
        # ===== External pruned topology override before simulator construction =====
        import os as _os
        external_topology_npz = _os.environ.get("EXTERNAL_TOPOLOGY_NPZ", None)
        self.external_init_spring_Y = None

        if external_topology_npz is not None and _os.path.exists(external_topology_npz):
            print(f"[EXTERNAL_TOPOLOGY_NPZ:init] Loading topology from: {external_topology_npz}")
            import numpy as _np
            topo = _np.load(external_topology_npz, allow_pickle=True)

            def _read(primary, fallback):
                if primary in topo.files:
                    return topo[primary]
                if fallback in topo.files:
                    return topo[fallback]
                raise KeyError(
                    f"Topology file must contain {primary!r} or {fallback!r}."
                )

            self.init_vertices = torch.tensor(
                _read("points_full", "init_vertices"),
                dtype=torch.float32,
                device=cfg.device,
            )
            self.init_springs = torch.tensor(
                _read("springs", "init_springs"),
                dtype=torch.int32,
                device=cfg.device,
            )
            self.init_rest_lengths = torch.tensor(
                _read("rest_lengths", "init_rest_lengths"),
                dtype=torch.float32,
                device=cfg.device,
            )
            self.init_masses = torch.tensor(
                _read("masses", "init_masses"),
                dtype=torch.float32,
                device=cfg.device,
            )
            self.num_object_springs = int(topo["num_object_springs"])
            if "spring_Y" in topo.files:
                self.external_init_spring_Y = torch.tensor(
                    topo["spring_Y"],
                    dtype=torch.float32,
                    device=cfg.device,
                )

            print("[EXTERNAL_TOPOLOGY_NPZ:init] init_vertices:", self.init_vertices.shape)
            print("[EXTERNAL_TOPOLOGY_NPZ:init] init_springs:", self.init_springs.shape)
            print("[EXTERNAL_TOPOLOGY_NPZ:init] num_object_springs:", self.num_object_springs)
        # ========================================================================
'''


TEST_PATCH = r'''            num_object_springs = checkpoint["num_object_springs"]

            # ===== External pruned topology override for test-time spring parameters =====
            import os as _os
            external_topology_npz = _os.environ.get("EXTERNAL_TOPOLOGY_NPZ", None)
            if external_topology_npz is not None and _os.path.exists(external_topology_npz):
                print(f"[EXTERNAL_TOPOLOGY_NPZ:test] Loading spring_Y from: {external_topology_npz}")
                import numpy as _np
                topo = _np.load(external_topology_npz, allow_pickle=True)
                if "spring_Y" in topo.files:
                    spring_Y = torch.tensor(
                        topo["spring_Y"],
                        dtype=torch.float32,
                        device=cfg.device,
                    )
                    num_object_springs = int(topo["num_object_springs"])
                    print("[EXTERNAL_TOPOLOGY_NPZ:test] spring_Y:", spring_Y.shape)
                    print("[EXTERNAL_TOPOLOGY_NPZ:test] num_object_springs:", num_object_springs)
            # =============================================================================
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch trainer_warp.py to accept an external pruned topology."
    )
    parser.add_argument("--phystwin-root", required=True, type=Path)
    parser.add_argument("--patch-pytorch3d", action="store_true")
    return parser.parse_args()


def patch_trainer(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    backup = path.with_suffix(
        path.suffix + ".backup_before_external_topology_patch"
    )
    if not backup.exists():
        shutil.copy2(path, backup)
        print("Backup:", backup)

    changed = False
    init_marker = "        self.simulator = SpringMassSystemWarp("
    if INIT_SENTINEL not in text:
        if init_marker not in text:
            raise RuntimeError(
                f"Could not find simulator marker in {path}. "
                "The upstream PhysTwin file may have changed."
            )
        text = text.replace(
            init_marker, INIT_PATCH + "\n" + init_marker, 1
        )
        changed = True
        print("Inserted initialization patch.")
    else:
        print("Initialization patch already exists.")

    test_marker = (
        '            num_object_springs = checkpoint["num_object_springs"]\n'
    )
    if TEST_SENTINEL not in text:
        if test_marker not in text:
            raise RuntimeError(
                f"Could not find test marker in {path}. "
                "The upstream PhysTwin file may have changed."
            )
        text = text.replace(test_marker, TEST_PATCH, 1)
        changed = True
        print("Inserted test-time spring patch.")
    else:
        print("Test patch already exists.")

    if changed:
        path.write_text(text, encoding="utf-8")
        print("Patched:", path)


def make_pytorch3d_optional(root: Path) -> None:
    replacements = {
        "from pytorch3d.loss import chamfer_distance": (
            "try:\n"
            "    from pytorch3d.loss import chamfer_distance\n"
            "except Exception:\n"
            "    chamfer_distance = None"
        ),
        "import pytorch3d.ops as ops": (
            "try:\n"
            "    import pytorch3d.ops as ops\n"
            "except Exception:\n"
            "    ops = None"
        ),
        "from pytorch3d.structures import Meshes": (
            "try:\n"
            "    from pytorch3d.structures import Meshes\n"
            "except Exception:\n"
            "    Meshes = None"
        ),
        "from pytorch3d.io import load_objs_as_meshes": (
            "try:\n"
            "    from pytorch3d.io import load_objs_as_meshes\n"
            "except Exception:\n"
            "    load_objs_as_meshes = None"
        ),
    }

    candidates = list((root / "qqtt").rglob("*.py"))
    if (root / "gs_render.py").is_file():
        candidates.append(root / "gs_render.py")

    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="ignore")
        original = text
        for old, new in replacements.items():
            if old in text and new not in text:
                text = text.replace(old, new)
        if text != original:
            backup = path.with_suffix(
                path.suffix + ".backup_before_optional_pytorch3d"
            )
            if not backup.exists():
                shutil.copy2(path, backup)
            path.write_text(text, encoding="utf-8")
            print("Made PyTorch3D import optional:", path)


def main() -> None:
    args = parse_args()
    root = args.phystwin_root.expanduser().resolve()
    trainer = root / "qqtt" / "engine" / "trainer_warp.py"
    if not trainer.is_file():
        raise FileNotFoundError(f"trainer_warp.py not found: {trainer}")

    patch_trainer(trainer)
    if args.patch_pytorch3d:
        make_pytorch3d_optional(root)


if __name__ == "__main__":
    main()
