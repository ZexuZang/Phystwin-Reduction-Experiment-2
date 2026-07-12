#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform
import sys


PACKAGES = ["numpy", "scipy", "pandas", "torch", "open3d", "cv2", "PIL", "skimage", "lpips", "pymor"]


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", platform.python_version())
    failed = False
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            print(f"{package}: {getattr(module, '__version__', 'unknown')}")
        except Exception as exc:
            failed = True
            print(f"{package}: IMPORT FAILED: {exc!r}")

    try:
        import torch

        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA device:", torch.cuda.get_device_name(0))
    except Exception:
        pass

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
