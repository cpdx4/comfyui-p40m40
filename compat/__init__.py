"""
compat/__init__.py
==================
Compatibility layer for ComfyUI on Pascal (sm_61) and Maxwell (sm_52) GPUs.
Call install() before importing any ComfyUI module.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("comfyui_compat")
_installed = False


def install(verbose: bool = True) -> None:
    """
    Install all compatibility shims.  Must be called before any ComfyUI import.
    Calling more than once is safe (idempotent).
    """
    global _installed
    if _installed:
        return
    _installed = True

    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="[compat] %(levelname)s %(message)s",
        )

    logger.info("=== ComfyUI P40/M40 Compatibility Layer ===")

    # 1. Detect GPUs first — other modules read FEATURE_FLAGS
    from compat.gpu_compat import detect_gpus, FEATURE_FLAGS
    detect_gpus()
    logger.info("Feature flags: %s", FEATURE_FLAGS)

    # 2. Patch PyTorch built-ins (fp8 stubs, compile noop, sdpa)
    from compat import torch_compat
    torch_compat.apply_all()

    # 3. Pre-register import hooks so ComfyUI attention module is patched
    #    the moment it is first imported, not after.
    from compat import attention_compat
    attention_compat.register_import_hook()

    # 4. Patch aiohttp so Host: hostname:port headers don't 500 in yarl
    from compat import aiohttp_compat
    aiohttp_compat.patch_aiohttp_host()

    logger.info("Compatibility layer installed successfully.")


__all__ = ["install"]
