"""
patches/patch_attention.py
===========================
Patches ComfyUI/comfy/ldm/modules/attention.py to:
  1. Remove / guard Triton imports (crash on sm_52/sm_61)
  2. Force xformers to disabled
  3. Replace SDPA flash/mem-efficient selection with math-only fallback
  4. Patch F.scaled_dot_product_attention `scale` kwarg (2.1 API on 2.0)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger("patches")

PATCH_ID = "attention_triton_flash"
TARGET_FILE = "comfy/ldm/modules/attention.py"
_SENTINEL = "# [P40-COMPAT] attention patched"


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


def apply() -> None:
    t = _target()
    if not t.exists():
        raise FileNotFoundError(f"Target not found: {t}")

    src = t.read_text(encoding="utf-8")
    if _SENTINEL in src:
        return

    src = _patch_triton_imports(src)
    src = _patch_xformers_flag(src)
    src = _patch_flash_attention_import(src)
    src = _patch_sdpa_backend_selection(src)
    src = _patch_scaled_dot_product_scale_kwarg(src)
    src += f"\n{_SENTINEL}\n"

    t.write_text(src, encoding="utf-8")
    logger.info("Patched: %s", TARGET_FILE)


# ---------------------------------------------------------------------------

def _patch_triton_imports(src: str) -> str:
    """
    Guard `import triton` / `from triton import ...` so they don't crash.
    The compat layer's _TritonBlocker handles runtime; this patch adds a
    source-level try/except so ComfyUI itself doesn't fail on startup.
    """
    # Replace bare triton imports with guarded versions
    src = re.sub(
        r'^(import triton\b[^\n]*)',
        r'try:\n    \1\nexcept (ImportError, RuntimeError):  # [P40-COMPAT]\n    triton = None',
        src,
        flags=re.MULTILINE,
    )
    src = re.sub(
        r'^(from triton[^\n]+)',
        r'try:\n    \1\nexcept (ImportError, RuntimeError):  # [P40-COMPAT]\n    pass',
        src,
        flags=re.MULTILINE,
    )
    return src


def _patch_xformers_flag(src: str) -> str:
    """
    xformers availability checks — force to False.
    ComfyUI patterns:
        XFORMERS_IS_AVAILABLE = True
        if XFORMERS_IS_AVAILABLE:
    """
    # Disable the flag assignment
    src = re.sub(
        r'^(XFORMERS_IS_AVAILABLE\s*=\s*)True\b',
        r'\1False  # [P40-COMPAT] xformers disabled (requires sm_70+)',
        src,
        flags=re.MULTILINE,
    )
    src = re.sub(
        r'^(XFORMERS_ENABLED\s*=\s*)True\b',
        r'\1False  # [P40-COMPAT]',
        src,
        flags=re.MULTILINE,
    )
    # Guard the xformers import
    src = re.sub(
        r'^(import xformers\b[^\n]*)',
        r'try:\n    \1\nexcept ImportError:  # [P40-COMPAT]\n    pass',
        src,
        flags=re.MULTILINE,
    )
    src = re.sub(
        r'^(from xformers[^\n]+)',
        r'try:\n    \1\nexcept ImportError:  # [P40-COMPAT]\n    pass',
        src,
        flags=re.MULTILINE,
    )
    return src


def _patch_flash_attention_import(src: str) -> str:
    """Guard flash_attn imports."""
    src = re.sub(
        r'^(import flash_attn\b[^\n]*)',
        r'try:\n    \1\nexcept (ImportError, RuntimeError):  # [P40-COMPAT]\n    flash_attn = None',
        src,
        flags=re.MULTILINE,
    )
    src = re.sub(
        r'^(from flash_attn[^\n]+)',
        r'try:\n    \1\nexcept (ImportError, RuntimeError):  # [P40-COMPAT]\n    pass',
        src,
        flags=re.MULTILINE,
    )
    return src


def _patch_sdpa_backend_selection(src: str) -> str:
    """
    ComfyUI selects attention backends in a priority order:
      flash → mem_efficient → math
    Force selection to always fall through to math.
    """
    # Disable flash attention selection
    src = re.sub(
        r'enable_flash\s*=\s*True',
        'enable_flash=False  # [P40-COMPAT]',
        src,
    )
    src = re.sub(
        r'enable_mem_efficient\s*=\s*True',
        'enable_mem_efficient=False  # [P40-COMPAT]',
        src,
    )

    # Patterns like: if FLASH_ATTN_AVAILABLE: → if False:  # [P40-COMPAT]
    for pattern in (
        "FLASH_ATTN_AVAILABLE",
        "FLASH_ATTENTION_AVAILABLE",
        "flash_attn_available",
        "HAS_FLASH_ATTN",
    ):
        src = re.sub(
            rf'\b{pattern}\b',
            'False  # [P40-COMPAT]',
            src,
        )

    return src


def _patch_scaled_dot_product_scale_kwarg(src: str) -> str:
    """
    PyTorch 2.1 added `scale` kwarg to F.scaled_dot_product_attention.
    PyTorch 2.0 raises TypeError if passed.
    Remove `scale=...` from call sites.
    """
    src = re.sub(
        r'(F\.scaled_dot_product_attention\([^)]+?),\s*scale\s*=[^,)]+',
        r'\1  # [P40-COMPAT scale kwarg removed]',
        src,
    )
    return src
