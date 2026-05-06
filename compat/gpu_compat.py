"""
compat/gpu_compat.py
====================
GPU detection and feature-flag registry.

FEATURE_FLAGS is populated once by detect_gpus() and read by all other
compat modules to decide which patches to apply.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("comfyui_compat")

# ---------------------------------------------------------------------------
# Feature flag registry
# ---------------------------------------------------------------------------
FEATURE_FLAGS: Dict[str, bool] = {
    "has_fp8": False,          # torch.float8_* dtypes (PyTorch 2.1+, sm_89+)
    "has_bf16_compute": False,  # BF16 tensor cores (Ampere sm_80+)
    "has_fp16_tensor_cores": False,  # FP16 tensor cores (Volta sm_70+)
    "has_flash_attention": False,  # Flash Attention 2 (Ampere sm_80+)
    "has_mem_efficient_attn": False,  # xformers mem-efficient (Volta sm_70+)
    "has_triton": False,       # Triton 2.x JIT (Volta sm_70+)
    "has_torch_compile": False,  # torch.compile Triton backend (sm_70+)
    "safe_dtype": "float16",   # Best safe dtype for this GPU
    "min_sm": 0,               # Lowest SM version across all detected GPUs
    "max_sm": 0,               # Highest SM version
}

# GPU info list populated by detect_gpus()
GPU_INFO: List[Dict] = []

# Minimum SM versions for features
_SM_REQUIREMENTS = {
    "has_fp16_tensor_cores": 70,   # Volta
    "has_bf16_compute": 80,        # Ampere
    "has_fp8": 89,                 # Ada Lovelace
    "has_flash_attention": 80,     # Ampere
    "has_mem_efficient_attn": 70,  # Volta (via xformers)
    "has_triton": 70,              # Volta
    "has_torch_compile": 70,       # Volta (for Triton backend)
}


def _sm_from_capability(major: int, minor: int) -> int:
    """Convert (major, minor) to a two-digit SM number."""
    return major * 10 + minor


def detect_gpus() -> None:
    """Detect all CUDA GPUs and populate FEATURE_FLAGS and GPU_INFO."""
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not found — GPU detection skipped.")
        return

    if not torch.cuda.is_available():
        logger.warning("CUDA not available — running on CPU.")
        FEATURE_FLAGS["safe_dtype"] = "float32"
        return

    device_count = torch.cuda.device_count()
    sm_values: List[int] = []

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        sm = _sm_from_capability(props.major, props.minor)
        sm_values.append(sm)

        info = {
            "index": i,
            "name": props.name,
            "sm": sm,
            "vram_gb": round(props.total_memory / 1024**3, 1),
            "major": props.major,
            "minor": props.minor,
        }
        GPU_INFO.append(info)
        logger.info(
            "GPU %d: %s  sm_%d  %.1f GB VRAM",
            i,
            props.name,
            sm,
            info["vram_gb"],
        )

    if not sm_values:
        return

    min_sm = min(sm_values)
    max_sm = max(sm_values)
    FEATURE_FLAGS["min_sm"] = min_sm
    FEATURE_FLAGS["max_sm"] = max_sm

    # Enable features only when ALL GPUs meet the minimum requirement
    # (weakest GPU determines the capability floor for the whole session)
    for flag, required_sm in _SM_REQUIREMENTS.items():
        FEATURE_FLAGS[flag] = min_sm >= required_sm

    # Check PyTorch version for fp8 dtype availability
    _check_torch_fp8()

    # Determine safe dtype
    if FEATURE_FLAGS["has_bf16_compute"]:
        FEATURE_FLAGS["safe_dtype"] = "bfloat16"
    else:
        FEATURE_FLAGS["safe_dtype"] = "float16"

    logger.info(
        "SM range: sm_%d – sm_%d | safe_dtype: %s | fp8: %s | bf16: %s | triton: %s",
        min_sm,
        max_sm,
        FEATURE_FLAGS["safe_dtype"],
        FEATURE_FLAGS["has_fp8"],
        FEATURE_FLAGS["has_bf16_compute"],
        FEATURE_FLAGS["has_triton"],
    )


def _check_torch_fp8() -> None:
    """
    PyTorch 2.1+ added torch.float8_e4m3fn / torch.float8_e5m2.
    Disable fp8 flag if these dtypes are absent.
    """
    try:
        import torch
        _ = torch.float8_e4m3fn  # noqa: F841  — will raise AttributeError on 2.0
        _ = torch.float8_e5m2    # noqa: F841
        # Only mark True if GPU also supports it
        FEATURE_FLAGS["has_fp8"] = FEATURE_FLAGS["has_fp8"] and True
    except AttributeError:
        FEATURE_FLAGS["has_fp8"] = False


def require_sm(minimum: int, feature_name: str = "this feature") -> None:
    """
    Raise RuntimeError with a helpful message if the GPU doesn't meet `minimum` SM.
    Call this from nodes or model loaders that truly cannot work below a threshold.
    """
    min_sm = FEATURE_FLAGS.get("min_sm", 0)
    if min_sm < minimum:
        raise RuntimeError(
            f"{feature_name} requires sm_{minimum}+ but your GPU has sm_{min_sm}. "
            f"This is expected on P40/M40 hardware."
        )


def get_safe_torch_dtype() -> "torch.dtype":
    """Return the best compute dtype for this GPU without importing at module level."""
    import torch
    safe = FEATURE_FLAGS.get("safe_dtype", "float16")
    return getattr(torch, safe)


def is_compatible_dtype(dtype: "torch.dtype") -> bool:
    """
    Return True if the given dtype can be used for compute on this GPU.
    Helps callers decide whether to downgrade.
    """
    import torch
    incompatible = []
    if not FEATURE_FLAGS["has_bf16_compute"]:
        incompatible.append(torch.bfloat16)
    if not FEATURE_FLAGS["has_fp8"]:
        # FP8 dtypes may not even exist in torch 2.0, guard with hasattr
        for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
            t = getattr(torch, name, None)
            if t is not None:
                incompatible.append(t)
    return dtype not in incompatible


def downgrade_dtype(dtype: "torch.dtype") -> "torch.dtype":
    """
    If dtype is incompatible with this GPU, return the best compatible fallback.
    FP8  → FP16
    BF16 → FP16  (on non-Ampere GPUs)
    """
    if is_compatible_dtype(dtype):
        return dtype
    import torch
    return torch.float16
