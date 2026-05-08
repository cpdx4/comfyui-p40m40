"""
patches/patch_model_patcher.py
==============================
Patches ComfyUI/comfy/model_patcher.py for PyTorch 2.0 compatibility.

PyTorch 2.0 dtypes do not expose `dtype.itemsize` in this environment.
Use a tensor-based fallback to compute byte size instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("patches")

PATCH_ID = "model_patcher_itemsize"
TARGET_FILE = "comfy/model_patcher.py"
_SENTINEL = "# [P40-COMPAT] model_patcher itemsize patched"


def _target() -> Path:
    from patches.apply_all import COMFYUI_ROOT
    return COMFYUI_ROOT / TARGET_FILE


def check() -> bool:
    t = _target()
    if not t.exists():
        return False
    return _SENTINEL in t.read_text(encoding="utf-8")


def revert() -> None:
    t = _target()
    orig = t.parent / (t.name + ".orig")
    if orig.exists():
        t.write_bytes(orig.read_bytes())
        logger.info("Reverted %s", TARGET_FILE)


def apply() -> None:
    t = _target()
    if not t.exists():
        raise FileNotFoundError(f"Target not found: {t}")

    src = t.read_text(encoding="utf-8")
    if _SENTINEL in src:
        logger.info("Already patched: %s", TARGET_FILE)
        return

    helper = (
        "\n\n# [P40-COMPAT] injected by patches/patch_model_patcher.py\n"
        "def _p40_dtype_itemsize(dtype):\n"
        "    try:\n"
        "        return dtype.itemsize\n"
        "    except Exception:\n"
        "        return torch.tensor([], dtype=dtype).element_size()\n"
    )

    anchor = "class LowVramPatch:"
    if anchor in src and "def _p40_dtype_itemsize(dtype):" not in src:
        src = src.replace(anchor, helper + "\n\n" + anchor, 1)

    src = src.replace(
        "return weight.numel() * model_dtype.itemsize * LOWVRAM_PATCH_ESTIMATE_MATH_FACTOR",
        "return weight.numel() * _p40_dtype_itemsize(model_dtype) * LOWVRAM_PATCH_ESTIMATE_MATH_FACTOR",
    )
    src = src.replace(
        "return weight.numel() * model_dtype.itemsize",
        "return weight.numel() * _p40_dtype_itemsize(model_dtype)",
    )

    src += f"\n{_SENTINEL}\n"
    t.write_text(src, encoding="utf-8")
    logger.info("Patched: %s", TARGET_FILE)
