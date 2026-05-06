"""
compat/torch_compat.py
======================
PyTorch version shims for running ComfyUI on PyTorch 2.0.1 + CUDA 11.8.

Patches applied:
  1. FP8 dtype stubs  (torch.float8_* missing in 2.0)
  2. torch.compile → identity no-op
  3. SDPA forced to math-only backend (flash/mem-efficient unavailable on Pascal/Maxwell)
  4. torch.autocast BF16 guard (BF16 compute not available on Pascal/Maxwell)
  5. torch.nn.attention.sdpa_kernel back-compat shim (2.1 API → 2.0 API)
"""

from __future__ import annotations

import contextlib
import logging
import sys
from typing import Any, Callable, Generator, Optional, TypeVar

import torch

logger = logging.getLogger("comfyui_compat")
F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# 1. FP8 dtype stubs
# ---------------------------------------------------------------------------

def patch_fp8_dtypes() -> None:
    """Inject FP8 stub attributes into the torch namespace."""
    from compat.fp8_stub import inject_into_torch
    inject_into_torch()
    logger.info("FP8 dtype stubs installed.")


# ---------------------------------------------------------------------------
# 2. torch.compile → identity no-op
# ---------------------------------------------------------------------------

_original_compile: Optional[Callable] = None
_compile_patched = False


def _compile_noop(model: Any = None, *args: Any, **kwargs: Any) -> Any:
    """
    Drop-in replacement for torch.compile that returns the model unchanged.
    Logs once so the user knows compilation was skipped.
    """
    if not hasattr(_compile_noop, "_logged"):
        logger.info(
            "torch.compile() is disabled (CUDA 11.8 / sm_61 incompatibility). "
            "Models run in eager mode."
        )
        _compile_noop._logged = True  # type: ignore[attr-defined]
    if model is None:
        # Used as @torch.compile decorator with no arguments
        return lambda fn: fn
    if callable(model):
        return model
    return model


def patch_torch_compile() -> None:
    global _original_compile, _compile_patched
    if _compile_patched:
        return
    if hasattr(torch, "compile"):
        _original_compile = torch.compile
        torch.compile = _compile_noop  # type: ignore[attr-defined]
        logger.info("torch.compile replaced with identity no-op.")
    else:
        logger.debug("torch.compile not present in this torch version — no patch needed.")
    _compile_patched = True


def restore_torch_compile() -> None:
    global _original_compile, _compile_patched
    if _original_compile is not None:
        torch.compile = _original_compile  # type: ignore[attr-defined]
    _compile_patched = False


# ---------------------------------------------------------------------------
# 3. SDPA — force math-only backend
# ---------------------------------------------------------------------------
#
# PyTorch 2.0 context manager: torch.backends.cuda.sdp_kernel(...)
# PyTorch 2.1 added:           torch.nn.attention.sdpa_kernel(...)
#
# On Pascal/Maxwell:
#   enable_flash=False     (requires Ampere)
#   enable_mem_efficient=False  (requires Volta via xformers)
#   enable_math=True       (pure CUDA, works everywhere)

_sdpa_patched = False


def patch_sdpa() -> None:
    """
    Force SDPA to always use the math (software) backend.
    Monkey-patches both the 2.0 and 2.1 context manager APIs.
    """
    global _sdpa_patched
    if _sdpa_patched:
        return

    # --- Patch 2.0 context manager ---
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        _patch_sdp_kernel_20()

    # --- Install 2.1 API shim (in case upstream uses it) ---
    _install_sdpa_kernel_21_shim()

    _sdpa_patched = True
    logger.info("SDPA forced to math-only backend (flash and mem-efficient disabled).")


def _patch_sdp_kernel_20() -> None:
    """
    Replace torch.backends.cuda.sdp_kernel so it always uses math backend
    regardless of what arguments the caller passes.
    """
    original = torch.backends.cuda.sdp_kernel

    @contextlib.contextmanager
    def _math_only_sdp_kernel(
        enable_flash: bool = False,
        enable_math: bool = True,
        enable_mem_efficient: bool = False,
        **kwargs: Any,
    ) -> Generator[None, None, None]:
        with original(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        ):
            yield

    torch.backends.cuda.sdp_kernel = _math_only_sdp_kernel  # type: ignore[attr-defined]


def _install_sdpa_kernel_21_shim() -> None:
    """
    Install torch.nn.attention.sdpa_kernel shim for code written against
    the PyTorch 2.1 API that may be present in newer ComfyUI.
    """
    # Ensure torch.nn.attention exists as a module-like object
    if not hasattr(torch.nn, "attention"):
        import types
        torch.nn.attention = types.ModuleType("torch.nn.attention")  # type: ignore[attr-defined]

    if not hasattr(torch.nn.attention, "sdpa_kernel"):
        @contextlib.contextmanager
        def _sdpa_kernel_21_shim(*backends: Any, **kwargs: Any) -> Generator[None, None, None]:
            """
            2.1-style sdpa_kernel shim → delegates to 2.0 math-only context manager.
            Accepts SDPBackend enum values and ignores them (we always use math).
            """
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
                with torch.backends.cuda.sdp_kernel(  # type: ignore[attr-defined]
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=False,
                ):
                    yield
            else:
                yield

        torch.nn.attention.sdpa_kernel = _sdpa_kernel_21_shim  # type: ignore[attr-defined]
        logger.debug("Installed torch.nn.attention.sdpa_kernel shim (2.1 API on 2.0).")

    # Also shim SDPBackend enum if absent
    if not hasattr(torch.nn.attention, "SDPBackend"):
        from enum import Enum
        class SDPBackend(Enum):  # type: ignore[no-redef]
            MATH = 0
            FLASH_ATTENTION = 1
            EFFICIENT_ATTENTION = 2
            CUDNN_ATTENTION = 3
        torch.nn.attention.SDPBackend = SDPBackend  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 4. autocast BF16 guard
# ---------------------------------------------------------------------------
#
# Some ComfyUI code does:  with torch.autocast("cuda", dtype=torch.bfloat16):
# On Pascal/Maxwell, BF16 autocast silently computes in FP32 (very slow) or
# produces incorrect results.  We intercept autocast and substitute FP16.

_autocast_patched = False
_original_autocast: Optional[type] = None


def patch_autocast() -> None:
    """Substitute BF16 autocast with FP16 autocast on non-BF16 GPUs."""
    global _autocast_patched, _original_autocast
    if _autocast_patched:
        return

    from compat.gpu_compat import FEATURE_FLAGS
    if FEATURE_FLAGS.get("has_bf16_compute", False):
        # GPU supports BF16 — no patch needed
        return

    _original_autocast = torch.autocast

    class _FP16AutocastGuard(torch.autocast):  # type: ignore[misc]
        """
        Wraps torch.autocast to convert BF16 → FP16 when BF16 is unavailable.
        """
        def __init__(
            self,
            device_type: str,
            dtype: Optional[torch.dtype] = None,
            enabled: bool = True,
            cache_enabled: Optional[bool] = None,
        ) -> None:
            if dtype == torch.bfloat16 and device_type == "cuda":
                if not hasattr(_FP16AutocastGuard, "_logged"):
                    logger.info(
                        "autocast BF16 → FP16 (BF16 compute not available on sm_%d)",
                        FEATURE_FLAGS.get("min_sm", 0),
                    )
                    _FP16AutocastGuard._logged = True
                dtype = torch.float16
            super().__init__(device_type, dtype=dtype, enabled=enabled)

    torch.autocast = _FP16AutocastGuard  # type: ignore[attr-defined,misc]
    # Also patch the older torch.cuda.amp.autocast if present
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        torch.cuda.amp.autocast = _FP16AutocastGuard  # type: ignore[attr-defined]

    _autocast_patched = True
    logger.info("autocast BF16 guard installed (BF16 → FP16 on Pascal/Maxwell).")


# ---------------------------------------------------------------------------
# 5. Misc missing attrs shims (PyTorch 2.0 vs 2.1 gaps)
# ---------------------------------------------------------------------------

def patch_misc() -> None:
    """Patch miscellaneous missing attrs added in PyTorch 2.1."""
    _patch_scaled_dot_product_attention_scale()
    _patch_missing_unsigned_dtypes()


def _patch_scaled_dot_product_attention_scale() -> None:
    """
    F.scaled_dot_product_attention in PyTorch 2.1 added an explicit `scale`
    keyword argument.  PyTorch 2.0 ignores unknown kwargs via **kwargs in some
    builds but raises TypeError in others.  Wrap to be safe.
    """
    import torch.nn.functional as F

    if not hasattr(F, "scaled_dot_product_attention"):
        return  # very old torch, nothing to do

    _orig_sdpa = F.scaled_dot_product_attention

    def _safe_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,  # 2.1+ argument — silently ignored on 2.0
        **kwargs: Any,
    ) -> torch.Tensor:
        # PyTorch 2.0 signature does not accept `scale`; drop it.
        try:
            return _orig_sdpa(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        except TypeError:
            # Fallback: compute manually
            import math
            head_dim = query.size(-1)
            s = scale if scale is not None else (1.0 / math.sqrt(head_dim))
            scores = torch.matmul(query, key.transpose(-2, -1)) * s
            if attn_mask is not None:
                scores = scores + attn_mask
            if is_causal:
                seq_len = query.size(-2)
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=query.device), diagonal=1
                )
                scores = scores + causal_mask
            weights = torch.softmax(scores, dim=-1)
            if dropout_p > 0.0:
                weights = torch.nn.functional.dropout(weights, p=dropout_p)
            return torch.matmul(weights, value)

    F.scaled_dot_product_attention = _safe_sdpa  # type: ignore[attr-defined]
    logger.debug("F.scaled_dot_product_attention wrapped for 2.0/2.1 compat.")


def _patch_missing_unsigned_dtypes() -> None:
    """Shim torch unsigned dtypes that are not exposed in torch 2.0.x."""
    aliases = {
        "uint16": torch.int16,
        "uint32": torch.int32,
        "uint64": torch.int64,
    }
    installed = []
    for name, target in aliases.items():
        if not hasattr(torch, name):
            setattr(torch, name, target)
            installed.append(name)

    if installed:
        logger.warning(
            "Installed torch dtype aliases for missing unsigned types on torch 2.0: %s",
            ", ".join(installed),
        )


# ---------------------------------------------------------------------------
# 6. torch.serialization shims (PyTorch 2.4 API on 2.0)
# ---------------------------------------------------------------------------
#
# PyTorch 2.4 added add_safe_globals / get_safe_globals / clear_safe_globals
# to torch.serialization for use with weights_only=True loading.
# ComfyUI's comfy/utils.py calls add_safe_globals at import time; without a
# shim this raises AttributeError and crashes the whole process.

def patch_serialization() -> None:
    """Install torch.serialization shims for APIs added in PyTorch 2.4."""
    import torch.serialization as _ser

    if not hasattr(_ser, "add_safe_globals"):
        _safe_globals_registry: list = []

        def add_safe_globals(cls_list: list) -> None:
            """No-op shim: torch 2.0 doesn't use the safe-globals mechanism."""
            _safe_globals_registry.extend(cls_list)

        def get_safe_globals() -> list:
            return list(_safe_globals_registry)

        def clear_safe_globals() -> None:
            _safe_globals_registry.clear()

        _ser.add_safe_globals = add_safe_globals  # type: ignore[attr-defined]
        _ser.get_safe_globals = get_safe_globals  # type: ignore[attr-defined]
        _ser.clear_safe_globals = clear_safe_globals  # type: ignore[attr-defined]
        logger.debug("Installed torch.serialization.add_safe_globals shim (2.4 API on 2.0).")


# ---------------------------------------------------------------------------
# 7. torch.nn.RMSNorm backfill (added in PyTorch 2.1, missing in 2.0)
# ---------------------------------------------------------------------------

def patch_rmsnorm() -> None:
    """Inject a pure-Python RMSNorm into torch.nn if missing (PyTorch < 2.1)."""
    if hasattr(torch.nn, "RMSNorm"):
        return

    class RMSNorm(torch.nn.Module):
        """Minimal RMSNorm compatible with the ComfyUI ops.py usage."""

        def __init__(self, normalized_shape, eps: float = 1e-5,
                     elementwise_affine: bool = True, device=None, dtype=None) -> None:
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if elementwise_affine:
                self.weight = torch.nn.Parameter(
                    torch.ones(self.normalized_shape, device=device, dtype=dtype)
                )
            else:
                self.register_parameter("weight", None)
            self.bias: Optional[torch.nn.Parameter] = None  # ComfyUI expects this attr

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # rms_norm not available in PyTorch < 2.1, implement manually
            orig_dtype = x.dtype
            x = x.float()
            norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            norm = norm.to(orig_dtype)
            if self.weight is not None:
                norm = norm * self.weight
            return norm

        def extra_repr(self) -> str:
            return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"

    torch.nn.RMSNorm = RMSNorm  # type: ignore[attr-defined]
    logger.info("Installed torch.nn.RMSNorm backfill (PyTorch 2.0 compat).")

    # Also backfill torch.nn.functional.rms_norm if missing
    if not hasattr(torch.nn.functional, "rms_norm"):
        def _rms_norm(input: torch.Tensor, normalized_shape, weight=None,
                      eps: float = 1e-5) -> torch.Tensor:
            orig_dtype = input.dtype
            x = input.float()
            norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            norm = norm.to(orig_dtype)
            if weight is not None:
                norm = norm * weight
            return norm
        torch.nn.functional.rms_norm = _rms_norm  # type: ignore[attr-defined]
        logger.info("Installed torch.nn.functional.rms_norm backfill (PyTorch 2.0 compat).")


# ---------------------------------------------------------------------------
# 8. Module.load_state_dict `assign` kwarg (added in PyTorch 2.1)
# ---------------------------------------------------------------------------
#
# ComfyUI calls:
#   model.load_state_dict(sd, strict=False, assign=self.patcher.is_dynamic())
#
# PyTorch 2.0 does not accept `assign`; this raises TypeError at runtime
# whenever a VAE, checkpoint, or CLIP model is loaded.
# Fix: wrap load_state_dict to silently drop `assign` on torch < 2.1.

def patch_load_state_dict() -> None:
    """Wrap Module.load_state_dict to ignore `assign` kwarg on PyTorch < 2.1."""
    import torch.nn as nn

    # torch 2.1 accepts assign natively — nothing to do
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2] if x.isdigit())
    if torch_version >= (2, 1):
        return

    _orig_lsd = nn.Module.load_state_dict

    def _load_state_dict_compat(self, state_dict, strict=True, assign=False, **kwargs):  # type: ignore[override]
        # `assign` was added in 2.1 — drop it silently on 2.0
        return _orig_lsd(self, state_dict, strict=strict, **kwargs)

    nn.Module.load_state_dict = _load_state_dict_compat  # type: ignore[method-assign]
    logger.info("Wrapped Module.load_state_dict to drop `assign` kwarg (PyTorch 2.0 compat).")


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def apply_all() -> None:
    """Apply all torch compatibility patches."""
    patch_fp8_dtypes()
    patch_torch_compile()
    patch_sdpa()
    patch_autocast()
    patch_misc()
    patch_serialization()
    patch_rmsnorm()
    patch_load_state_dict()
