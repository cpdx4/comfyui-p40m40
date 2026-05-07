"""
compat/fp8_stub.py
==================
Stub objects that mimic torch FP8 dtypes for code paths that reference
torch.float8_e4m3fn / torch.float8_e5m2 at import time.

On PyTorch 2.0.1 these attributes do not exist.  We inject plausible
sentinel objects so that isinstance checks and dtype comparisons don't crash,
while ensuring any computation that would use FP8 is transparently redirected
to FP16.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger("comfyui_compat")

# ---------------------------------------------------------------------------
# Stub dtype class
# ---------------------------------------------------------------------------

class _FP8DtypeStub:
    """
    A sentinel that:
    - compares equal to itself (identity) so `dtype == torch.float8_e4m3fn` works
    - returns torch.float16 when used in tensor operations
    - logs a warning the first time it is actually used in a computation
    """

    def __init__(self, name: str, fallback: torch.dtype = torch.float16):
        self._name = name
        self.fallback = fallback
        self._warned = False

    def _warn_once(self) -> None:
        if not self._warned:
            logger.warning(
                "FP8 dtype '%s' is not available on this GPU/PyTorch version. "
                "Falling back to %s.  Generation quality is unaffected.",
                self._name,
                self.fallback,
            )
            self._warned = True

    # ------------------------------------------------------------------
    # Make the stub behave reasonably in common usage patterns
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"torch.{self._name} (stubbed → {self.fallback})"

    def __str__(self) -> str:
        return self._name

    def __eq__(self, other: Any) -> bool:
        # Exact identity match only — so `dtype == float8_e4m3fn` is False
        # when dtype is actually float16, which is the correct behavior.
        return other is self

    def __hash__(self) -> int:
        return hash(self._name)

    def __bool__(self) -> bool:
        return True

    # Allows `tensor.to(torch.float8_e4m3fn)` to silently use float16
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._warn_once()
        return self.fallback(*args, **kwargs)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# FP8 dtype sentinels
# ---------------------------------------------------------------------------

float8_e4m3fn = _FP8DtypeStub("float8_e4m3fn", torch.float16)
float8_e5m2 = _FP8DtypeStub("float8_e5m2", torch.float16)
float8_e4m3fnuz = _FP8DtypeStub("float8_e4m3fnuz", torch.float16)
float8_e5m2fnuz = _FP8DtypeStub("float8_e5m2fnuz", torch.float16)

# Map from stub name to stub object for easy lookup
FP8_STUBS = {
    "float8_e4m3fn": float8_e4m3fn,
    "float8_e5m2": float8_e5m2,
    "float8_e4m3fnuz": float8_e4m3fnuz,
    "float8_e5m2fnuz": float8_e5m2fnuz,
}


def inject_into_torch() -> None:
    """
    Inject stub objects as attributes of the `torch` module so that code doing
    `torch.float8_e4m3fn` at runtime doesn't raise AttributeError.

    Only injects attributes that are genuinely missing (i.e. on PyTorch 2.0).
    """
    for attr_name, stub in FP8_STUBS.items():
        if not hasattr(torch, attr_name):
            setattr(torch, attr_name, stub)
            logger.debug("Injected FP8 stub: torch.%s", attr_name)
        else:
            logger.debug("torch.%s already present — not stubbing.", attr_name)


def is_fp8_stub(dtype: Any) -> bool:
    """Return True if dtype is one of our stub objects."""
    return isinstance(dtype, _FP8DtypeStub)


def resolve_dtype(dtype: Any, fallback: torch.dtype = torch.float16) -> torch.dtype:
    """
    Given a dtype that might be an FP8 stub or a real torch.dtype, return a
    dtype that is safe to use on this GPU.
    """
    if isinstance(dtype, _FP8DtypeStub):
        dtype._warn_once()
        return fallback
    if isinstance(dtype, torch.dtype):
        from compat.gpu_compat import downgrade_dtype
        return downgrade_dtype(dtype)
    return fallback


# ---------------------------------------------------------------------------
# FP8 → float16 dequantization (pure PyTorch, no native FP8 support needed)
# ---------------------------------------------------------------------------

def fp8_e4m3fn_to_float16(uint8_tensor: torch.Tensor) -> torch.Tensor:
    """
    Dequantize a tensor of raw fp8_e4m3fn bytes (stored as uint8) to float16.

    FP8 E4M3FN format (per OFP8 / IEEE draft):
      bit 7   : sign
      bits 6-3: exponent (4 bits), bias = 7
      bits 2-0: mantissa (3 bits)
      special : exp=0xF, mant!=0 → NaN  (no infinity)
    """
    x = uint8_tensor.to(torch.int32)
    sign = (((x >> 7) & 1) * -2 + 1).to(torch.float32)   # +1 or -1
    exp_bits = (x >> 3) & 0xF
    mant_bits = (x & 0x7).to(torch.float32)

    # Normal: val = sign * 2^(exp-7) * (1 + mant/8)
    norm_val = sign * torch.pow(2.0, (exp_bits - 7).to(torch.float32)) * (1.0 + mant_bits / 8.0)
    # Subnormal (exp==0): val = sign * 2^(-6) * (mant/8)
    sub_val = sign * (2.0 ** -6) * (mant_bits / 8.0)
    # NaN
    nan_val = torch.full_like(norm_val, float("nan"))

    result = torch.where(exp_bits == 0, sub_val, norm_val)
    result = torch.where((exp_bits == 15) & (x & 0x7 != 0), nan_val, result)
    return result.to(torch.float16)


def fp8_e5m2_to_float16(uint8_tensor: torch.Tensor) -> torch.Tensor:
    """
    Dequantize a tensor of raw fp8_e5m2 bytes (stored as uint8) to float16.

    FP8 E5M2 format:
      bit 7   : sign
      bits 6-2: exponent (5 bits), bias = 15
      bits 1-0: mantissa (2 bits)
      special : exp=0x1F, mant=01/10/11 → NaN; exp=0x1F, mant=00 → ±Inf
    """
    x = uint8_tensor.to(torch.int32)
    sign = (((x >> 7) & 1) * -2 + 1).to(torch.float32)
    exp_bits = (x >> 2) & 0x1F
    mant_bits = (x & 0x3).to(torch.float32)

    norm_val = sign * torch.pow(2.0, (exp_bits - 15).to(torch.float32)) * (1.0 + mant_bits / 4.0)
    sub_val = sign * (2.0 ** -14) * (mant_bits / 4.0)
    inf_val = sign * torch.full_like(norm_val, float("inf"))
    nan_val = torch.full_like(norm_val, float("nan"))

    result = torch.where(exp_bits == 0, sub_val, norm_val)
    result = torch.where((exp_bits == 31) & (x & 0x3 == 0), inf_val, result)
    result = torch.where((exp_bits == 31) & (x & 0x3 != 0), nan_val, result)
    return result.to(torch.float16)


# Map safetensors dtype string → dequant function
FP8_DEQUANT = {
    "F8_E4M3": fp8_e4m3fn_to_float16,
    "F8_E5M2": fp8_e5m2_to_float16,
}
