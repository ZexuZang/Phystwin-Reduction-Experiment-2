from __future__ import annotations

import sys
import types

import torch


def install_pytorch3d_stub() -> bool:
    """Install a minimal in-process PyTorch3D stub when PyTorch3D is unavailable."""
    try:
        import pytorch3d  # noqa: F401
        return False
    except Exception:
        pass

    p3d = types.ModuleType("pytorch3d")
    ops = types.ModuleType("pytorch3d.ops")
    loss = types.ModuleType("pytorch3d.loss")
    structures = types.ModuleType("pytorch3d.structures")
    io = types.ModuleType("pytorch3d.io")

    def chamfer_distance(
        x,
        y,
        *,
        batch_reduction="mean",
        point_reduction="mean",
        norm=2,
        single_directional=False,
        **_,
    ):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if y.dim() == 2:
            y = y.unsqueeze(0)
        distances = torch.cdist(x, y, p=1 if norm == 1 else 2)
        x_to_y = distances.min(dim=2).values
        if single_directional:
            value = x_to_y
        else:
            value = x_to_y + distances.min(dim=1).values
        if point_reduction == "mean":
            value = value.mean(dim=1)
        elif point_reduction == "sum":
            value = value.sum(dim=1)
        if batch_reduction == "mean":
            value = value.mean()
        elif batch_reduction == "sum":
            value = value.sum()
        return value, None

    class _KNNResult:
        pass

    def knn_points(p1, p2, K=1, return_nn=False, **_):
        squared = torch.cdist(p1, p2) ** 2
        dists, indices = torch.topk(squared, k=K, dim=-1, largest=False)
        result = _KNNResult()
        result.dists = dists
        result.idx = indices
        if return_nn:
            batch = torch.arange(p2.shape[0], device=p2.device)[:, None, None]
            result.knn = p2[batch, indices]
        return result

    class Meshes:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def load_objs_as_meshes(*_, **__):
        raise ImportError("PyTorch3D is unavailable in this environment.")

    loss.chamfer_distance = chamfer_distance
    ops.knn_points = knn_points
    structures.Meshes = Meshes
    io.load_objs_as_meshes = load_objs_as_meshes

    p3d.ops = ops
    p3d.loss = loss
    p3d.structures = structures
    p3d.io = io

    sys.modules["pytorch3d"] = p3d
    sys.modules["pytorch3d.ops"] = ops
    sys.modules["pytorch3d.loss"] = loss
    sys.modules["pytorch3d.structures"] = structures
    sys.modules["pytorch3d.io"] = io
    return True
