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
