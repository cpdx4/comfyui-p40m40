"""Compatibility stub for comfy_aimdo on unsupported GPUs.

This package satisfies upstream ComfyUI imports without enabling DynamicVRAM.
"""

from . import control
from . import host_buffer
from . import model_mmap
from . import model_vbar
from . import torch
from . import vram_buffer

__all__ = [
	"control",
	"host_buffer",
	"model_mmap",
	"model_vbar",
	"torch",
	"vram_buffer",
]
