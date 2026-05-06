"""Compatibility stub for comfy_aimdo.torch."""

from __future__ import annotations

import torch


def aimdo_to_tensor(obj, device=None):
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device) if device is not None else obj
    target = device if device is not None else "cpu"
    return torch.empty((0,), dtype=torch.uint8, device=target)


def hostbuf_to_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    return torch.empty((0,), dtype=torch.uint8, device="cpu")
