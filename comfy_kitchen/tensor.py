"""
comfy_kitchen.tensor compatibility sentinel.

On P40/M40-class systems this backend is intentionally unavailable.
Importing this module raises ImportError so ComfyUI can gracefully
fall back to non-comfy_kitchen quantization paths.
"""

raise ImportError("comfy_kitchen.tensor backend unavailable on this platform")
