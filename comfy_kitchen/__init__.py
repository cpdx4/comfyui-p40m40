"""
comfy_kitchen stub — P40/M40 compatibility shim.

comfy_kitchen provides FP8/FP4 tensor primitives that require sm_89+
(Ada Lovelace / H100).  On Pascal (sm_61) and Maxwell (sm_52) these
operations are not available.  This stub satisfies the unconditional
import in ComfyUI's nodes.py without crashing at startup.
"""
from . import tensor  # noqa: F401

__all__ = ["tensor"]
