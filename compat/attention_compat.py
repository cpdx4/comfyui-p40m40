"""
compat/attention_compat.py
==========================
Patches the ComfyUI attention mechanism to work on Pascal (sm_61) and
Maxwell (sm_52) GPUs where Flash Attention, xformers, and Triton are
unavailable.

Strategy:
  - Register a sys.meta_path import hook so the patch fires the instant
    `comfy.ldm.modules.attention` is first imported.
  - Monkey-patch `optimized_attention` and disable xformers/flash guards.
  - Disable Triton import at the sys.modules level so ComfyUI never tries
    to JIT-compile a Triton kernel.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import logging
import sys
import types
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger("comfyui_compat")

# ---------------------------------------------------------------------------
# Triton blocker — prevents import triton from succeeding even if triton
# is installed, because Triton 2.x silently corrupts on sm_52/sm_61.
# ---------------------------------------------------------------------------

class _TritonBlocker(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """
    Intercepts `import triton` and returns a harmless dummy module.
    This prevents ComfyUI from crashing when it probes for Triton availability.
    """

    _BLOCKED = frozenset({"triton", "triton.language", "triton.ops", "triton.runtime"})

    def find_module(
        self, fullname: str, path: Optional[Any] = None
    ) -> Optional["_TritonBlocker"]:
        if fullname in self._BLOCKED or fullname.startswith("triton."):
            return self
        return None

    def find_spec(
        self,
        fullname: str,
        path: Any,
        target: Optional[types.ModuleType] = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        if fullname in self._BLOCKED or fullname.startswith("triton."):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(
        self, spec: importlib.machinery.ModuleSpec
    ) -> Optional[types.ModuleType]:
        mod = types.ModuleType(spec.name)
        mod.__package__ = spec.name.split(".")[0]
        mod.__spec__ = spec
        return mod

    def exec_module(self, module: types.ModuleType) -> None:
        # Provide stub `jit` decorator so @triton.jit decorated functions
        # compile to a no-op lambda rather than crashing.
        if module.__name__ == "triton":
            module.jit = lambda fn=None, **kw: (fn if fn is not None else lambda f: f)  # type: ignore[attr-defined]
            module.cdiv = lambda a, b: -(-a // b)  # type: ignore[attr-defined]
            logger.debug("triton import blocked and stubbed (sm_%s incompatible).", 
                         sys.modules.get("__main__", object).__class__.__name__)


def _install_triton_blocker() -> None:
    # Only block if real triton is absent or incompatible
    if "triton" in sys.modules:
        # Already imported — check if it's ours
        existing = sys.modules["triton"]
        if hasattr(existing, "__file__") and existing.__file__:
            # Real triton is loaded — check GPU capability
            from compat.gpu_compat import FEATURE_FLAGS
            if not FEATURE_FLAGS.get("has_triton", False):
                logger.warning(
                    "Real triton is installed but GPU has sm_%d < sm_70. "
                    "Replacing triton module with stub to prevent crashes.",
                    FEATURE_FLAGS.get("min_sm", 0),
                )
                stub = types.ModuleType("triton")
                stub.jit = lambda fn=None, **kw: (fn if fn is not None else lambda f: f)  # type: ignore[attr-defined]
                stub.cdiv = lambda a, b: -(-a // b)  # type: ignore[attr-defined]
                sys.modules["triton"] = stub
    else:
        blocker = _TritonBlocker()
        sys.meta_path.insert(0, blocker)
        logger.info("Triton import blocker installed (sm_70+ required).")


# ---------------------------------------------------------------------------
# Safe math-only attention implementation
# ---------------------------------------------------------------------------

def _math_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: Optional[torch.Tensor] = None,
    attn_precision: Optional[Any] = None,
    skip_reshape: bool = False,
) -> torch.Tensor:
    """
    Pure-PyTorch scaled dot-product attention.
    Works on any GPU; no Flash Attention, no xformers, no Triton.
    Uses FP32 accumulation for numerics, then casts back to input dtype.
    """
    if skip_reshape:
        b, n, _ = q.shape
        dim_head = q.shape[-1] // heads
    else:
        b, n, _ = q.shape
        dim_head = q.shape[-1] // heads
        q = q.view(b, n, heads, dim_head).transpose(1, 2)
        k = k.view(b, k.shape[1], heads, dim_head).transpose(1, 2)
        v = v.view(b, v.shape[1], heads, dim_head).transpose(1, 2)

    scale = dim_head ** -0.5
    original_dtype = q.dtype

    # Upcast to FP32 for attention weight computation to avoid FP16 overflow
    q_fp32 = q.float()
    k_fp32 = k.float()

    scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) * scale

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores + mask.float()

    weights = torch.softmax(scores, dim=-1).to(original_dtype)
    out = torch.matmul(weights, v)

    if not skip_reshape:
        out = out.transpose(1, 2).reshape(b, n, heads * dim_head)

    return out


# ---------------------------------------------------------------------------
# ComfyUI optimized_attention replacement
# ---------------------------------------------------------------------------

def _make_patched_optimized_attention() -> Any:
    """
    Returns a replacement for comfy.ldm.modules.attention.optimized_attention
    that always uses the safe math backend.
    """

    def optimized_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: Optional[torch.Tensor] = None,
        attn_precision: Optional[Any] = None,
        skip_reshape: bool = False,
    ) -> torch.Tensor:
        return _math_attention(q, k, v, heads, mask, attn_precision, skip_reshape)

    def optimized_attention_masked_fill(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: Optional[torch.Tensor] = None,
        attn_precision: Optional[Any] = None,
    ) -> torch.Tensor:
        return _math_attention(q, k, v, heads, mask, attn_precision)

    return optimized_attention, optimized_attention_masked_fill


# ---------------------------------------------------------------------------
# Import hook — fires when comfy.ldm.modules.attention is first imported
# ---------------------------------------------------------------------------

class _AttentionPatchHook(importlib.abc.MetaPathFinder):
    """
    After the real module loads, replace attention functions with safe variants.
    """

    _TARGETS = {
        "comfy.ldm.modules.attention",
        "comfy.ldm.modules.diffusionmodules.model",
    }

    def find_spec(
        self,
        fullname: str,
        path: Any,
        target: Optional[types.ModuleType] = None,
    ) -> None:
        return None  # Never intercept loading — only patch after load

    def find_module(self, fullname: str, path: Optional[Any] = None) -> None:
        return None


def _patch_attention_module(module: types.ModuleType) -> None:
    """Apply patches to a loaded attention module."""
    opt_attn, opt_attn_mf = _make_patched_optimized_attention()

    patched = []

    for attr in (
        "optimized_attention",
        "attention_basic",
        "attention_split",
        "attention_sub_quad",
    ):
        if hasattr(module, attr):
            setattr(module, attr, opt_attn)
            patched.append(attr)

    for attr in ("optimized_attention_masked_fill",):
        if hasattr(module, attr):
            setattr(module, attr, opt_attn_mf)
            patched.append(attr)

    # Disable xformers flag
    for flag in ("XFORMERS_IS_AVAILABLE", "XFORMERS_ENABLED", "_use_xformers"):
        if hasattr(module, flag):
            setattr(module, flag, False)
            patched.append(f"{flag}=False")

    if patched:
        logger.info("Patched attention module '%s': %s", module.__name__, patched)


class _PostImportPatcher:
    """
    sys.meta_path finder that calls a callback after a target module loads.
    Uses the loader protocol to intercept exec_module.
    """

    def __init__(self, targets: dict[str, Any]) -> None:
        self._targets = targets  # module_name → callback(module)
        self._done: set[str] = set()

    def find_spec(
        self,
        fullname: str,
        path: Any,
        target: Optional[types.ModuleType] = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        if fullname not in self._targets or fullname in self._done:
            return None
        # Find the real spec via remaining finders
        spec = self._find_real_spec(fullname, path)
        if spec is None:
            return None
        # Wrap the loader
        spec.loader = _WrappedLoader(spec.loader, fullname, self._targets[fullname], self._done)
        return spec

    def _find_real_spec(
        self, fullname: str, path: Any
    ) -> Optional[importlib.machinery.ModuleSpec]:
        for finder in sys.meta_path:
            if finder is self:
                continue
            if hasattr(finder, "find_spec"):
                spec = finder.find_spec(fullname, path, None)
                if spec is not None:
                    return spec
        return None


class _WrappedLoader(importlib.abc.Loader):
    def __init__(
        self,
        real_loader: Any,
        fullname: str,
        callback: Any,
        done_set: set[str],
    ) -> None:
        self._real = real_loader
        self._fullname = fullname
        self._callback = callback
        self._done = done_set

    def create_module(
        self, spec: importlib.machinery.ModuleSpec
    ) -> Optional[types.ModuleType]:
        if hasattr(self._real, "create_module"):
            return self._real.create_module(spec)
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        if hasattr(self._real, "exec_module"):
            self._real.exec_module(module)
        self._done.add(self._fullname)
        try:
            self._callback(module)
        except Exception as exc:
            logger.error(
                "Attention patch callback failed for '%s': %s", self._fullname, exc
            )


def register_import_hook() -> None:
    """
    Install the import hook so attention modules are patched on first import.
    Also immediately patch any modules already loaded.
    """
    _install_triton_blocker()

    targets = {
        "comfy.ldm.modules.attention": _patch_attention_module,
    }

    patcher = _PostImportPatcher(targets)
    sys.meta_path.insert(0, patcher)
    logger.info("Attention import hook registered.")

    # Patch any modules already loaded (e.g. if attention was imported before install())
    for mod_name, callback in targets.items():
        if mod_name in sys.modules:
            callback(sys.modules[mod_name])
            logger.info("Patched already-loaded module: %s", mod_name)
