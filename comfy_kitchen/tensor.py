"""
comfy_kitchen.tensor stub — P40/M40 compatibility shim.

The real comfy_kitchen.tensor provides FP8/FP4 quantised tensor ops that
require sm_89+ (Ada Lovelace).  On Pascal (sm_61) and Maxwell (sm_52)
these are unavailable; this stub lets ComfyUI import the module cleanly
and raises NotImplementedError if any op is actually called at runtime.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("comfyui_compat.comfy_kitchen")
logger.debug("comfy_kitchen.tensor stub loaded — FP8/FP4 ops are unavailable on sm<89.")


def _unavailable(name: str):
    def _raise(*args: Any, **kwargs: Any):
        raise NotImplementedError(
            f"comfy_kitchen.tensor.{name} is not available on Pascal/Maxwell (sm<89). "
            "FP8/FP4 quantisation requires Ada Lovelace or newer."
        )
    _raise.__name__ = name
    return _raise


# ---------------------------------------------------------------------------
# Stub the public API surface that ComfyUI / custom nodes reference.
# Add more stubs here if new names are needed.
# ---------------------------------------------------------------------------

cast_to_fp8      = _unavailable("cast_to_fp8")
cast_to_fp4      = _unavailable("cast_to_fp4")
dequantize_fp8   = _unavailable("dequantize_fp8")
dequantize_fp4   = _unavailable("dequantize_fp4")
fp8_gemm         = _unavailable("fp8_gemm")
fp4_gemm         = _unavailable("fp4_gemm")
quantize_weight  = _unavailable("quantize_weight")
