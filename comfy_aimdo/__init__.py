"""Compatibility stub for comfy_aimdo on unsupported GPUs.

This package satisfies upstream ComfyUI imports without enabling DynamicVRAM.
"""

from . import control

__all__ = ["control"]
