"""
patches/patch_model_management.py
==================================
Patches ComfyUI/comfy/model_management.py to:
  1. Remove / guard FP8 dtype references (torch.float8_* not in PyTorch 2.0)
  2. Disable BF16 as a default dtype on Pascal/Maxwell GPUs
  3. Wrap torch.compile calls (belt-and-suspenders, compat layer also does this)
  4. Gate SDPA mem-efficient / flash paths

All edits are string-replacement based and idempotent.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger("patches")

PATCH_ID = "model_management_fp8_bf16"
TARGET_FILE = "comfy/model_management.py"

# Sentinel comment injected to detect if patch is already applied
_SENTINEL = "# [P40-COMPAT] patched"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

    src = _patch_fp8_dtype_references(src)
    src = _patch_bf16_unet_dtype(src)
    src = _patch_torch_compile(src)
    src = _patch_sdpa_selection(src)
    src = _add_sentinel(src)

    t.write_text(src, encoding="utf-8")
    logger.info("Patched: %s", TARGET_FILE)


# ---------------------------------------------------------------------------
# Individual transformations
# ---------------------------------------------------------------------------

def _patch_fp8_dtype_references(src: str) -> str:
    """
    Replace direct torch.float8_* attribute accesses with a safe helper call.
    ComfyUI uses these in unet_dtype() and a few other places.

    Pattern: `torch.float8_e4m3fn` → `_compat_get_fp8_or_fp16()`
    """
    # Inject a helper function near the top of the file
    helper = '''
# [P40-COMPAT] FP8 dtype helper — injected by patches/patch_model_management.py
def _compat_get_fp8_or_fp16(preferred_fp8_attr: str = "float8_e4m3fn"):
    """Return FP8 dtype if available, else FP16."""
    import torch
    dtype = getattr(torch, preferred_fp8_attr, None)
    if dtype is None or not _compat_is_real_fp8(dtype):
        return torch.float16
    return dtype

def _compat_is_real_fp8(dtype) -> bool:
    """True only if dtype is a genuine PyTorch FP8 type (not a stub)."""
    import torch
    for name in ("float8_e4m3fn", "float8_e5m2"):
        real = getattr(torch, name, None)
        if real is not None and real is dtype:
            try:
                torch.zeros(1, dtype=dtype)  # will fail on CUDA 11.8 / sm_61
                return True
            except Exception:
                return False
    return False
'''

    # Insert helper after the first import block (after the last top-level import)
    # Find position of first non-import, non-comment, non-blank line
    import_end = 0
    for match in re.finditer(r'^(?:import |from )', src, re.MULTILINE):
        import_end = match.end()

    # Move to end of that import line
    newline_pos = src.find('\n', import_end)
    if newline_pos == -1:
        newline_pos = len(src)

    src = src[:newline_pos + 1] + helper + src[newline_pos + 1:]

    # Replace bare torch.float8_e4m3fn with the helper
    src = re.sub(
        r'\btorch\.float8_e4m3fn\b',
        '_compat_get_fp8_or_fp16("float8_e4m3fn")',
        src,
    )
    src = re.sub(
        r'\btorch\.float8_e5m2\b',
        '_compat_get_fp8_or_fp16("float8_e5m2")',
        src,
    )
    src = re.sub(
        r'\btorch\.float8_e4m3fnuz\b',
        '_compat_get_fp8_or_fp16("float8_e4m3fnuz")',
        src,
    )

    return src


def _patch_bf16_unet_dtype(src: str) -> str:
    """
    In unet_dtype() / get_computation_dtype(), if BF16 is selected but
    the GPU doesn't support it, downgrade to FP16.

    We inject a post-processing call at the return site.
    """
    helper = '''
# [P40-COMPAT] BF16 downgrade helper
def _compat_downgrade_bf16(dtype):
    """Downgrade bfloat16 to float16 on GPUs without BF16 tensor cores."""
    import torch
    if dtype != torch.bfloat16:
        return dtype
    try:
        from compat.gpu_compat import FEATURE_FLAGS
        if not FEATURE_FLAGS.get("has_bf16_compute", False):
            return torch.float16
    except ImportError:
        # compat layer not installed — check CUDA capability directly
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            sm = props.major * 10 + props.minor
            if sm < 80:
                return torch.float16
    return dtype
'''

    # Inject helper (append to existing compat helpers)
    src = src.replace(
        "# [P40-COMPAT] BF16 downgrade helper",
        "",
        # idempotent — remove if already present before re-adding
    )
    # Insert after first def in file
    first_def = src.find('\ndef ')
    if first_def != -1:
        src = src[:first_def] + '\n' + helper + src[first_def:]

    # Wrap returns from unet_dtype-like functions
    # Pattern: return torch.bfloat16  →  return _compat_downgrade_bf16(torch.bfloat16)
    src = re.sub(
        r'\breturn (torch\.bfloat16)\b',
        r'return _compat_downgrade_bf16(\1)',
        src,
    )

    return src


def _patch_torch_compile(src: str) -> str:
    """
    Replace torch.compile(model, ...) with the model unchanged.
    Belt-and-suspenders — compat/torch_compat.py patches at runtime too.
    """
    # @torch.compile decorator → no-op
    src = re.sub(
        r'@torch\.compile\b[^\n]*\n',
        '# [P40-COMPAT] @torch.compile removed\n',
        src,
    )
    # torch.compile(expr) as a call
    src = re.sub(
        r'\btorch\.compile\(([^)]+)\)',
        r'\1  # [P40-COMPAT] torch.compile removed',
        src,
    )
    return src


def _patch_sdpa_selection(src: str) -> str:
    """
    Where model_management explicitly enables flash or mem-efficient SDPA,
    clamp those flags to False for Pascal/Maxwell.
    """
    # enable_flash=True → enable_flash=False  # P40-COMPAT
    src = re.sub(
        r'\benable_flash\s*=\s*True\b',
        'enable_flash=False  # [P40-COMPAT]',
        src,
    )
    src = re.sub(
        r'\benable_mem_efficient\s*=\s*True\b',
        'enable_mem_efficient=False  # [P40-COMPAT]',
        src,
    )
    return src


def _add_sentinel(src: str) -> str:
    return src + f"\n{_SENTINEL}\n"
