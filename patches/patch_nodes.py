"""
patches/patch_nodes.py
======================
Patches ComfyUI/nodes.py and comfy_extras/nodes_flux.py to:
  1. Remove / stub FP8-specific node classes (FP8CheckpointLoader, etc.)
  2. Remove comfy_kitchen imports
  3. Ensure Qwen-VL / Qwen-Image-Edit nodes load using patched diffusers

These patches preserve node registry integrity so the frontend doesn't show
red "missing node" errors — the nodes still appear but operate in FP16 mode.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

logger = logging.getLogger("patches")

PATCH_ID = "nodes_fp8_comfy_kitchen"
TARGET_FILE = "nodes.py"  # checked first; flux nodes handled separately
_SENTINEL = "# [P40-COMPAT] nodes patched"


def _target() -> Path:
    from patches.apply_all import COMFYUI_ROOT
    return COMFYUI_ROOT / TARGET_FILE


def check() -> bool:
    t = _target()
    if not t.exists():
        return False
    return _SENTINEL in t.read_text(encoding="utf-8")


def revert() -> None:
    for rel_path in [TARGET_FILE, "comfy_extras/nodes_flux.py"]:
        from patches.apply_all import COMFYUI_ROOT
        t = COMFYUI_ROOT / rel_path
        orig = t.parent / (t.name + ".orig")
        if orig.exists():
            t.write_bytes(orig.read_bytes())


def apply() -> None:
    _patch_file(TARGET_FILE)
    _patch_flux_nodes()
    _patch_comfy_kitchen_references()


def _patch_file(rel_path: str) -> None:
    from patches.apply_all import COMFYUI_ROOT
    t = COMFYUI_ROOT / rel_path
    if not t.exists():
        logger.debug("Optional target not found, skipping: %s", rel_path)
        return

    src = t.read_text(encoding="utf-8")
    if _SENTINEL in src:
        return

    src = _patch_comfy_kitchen_imports(src)
    src = _patch_fp8_node_classes(src)
    src = _patch_diffusers_bf16_default(src)
    src += f"\n{_SENTINEL}\n"

    t.write_text(src, encoding="utf-8")
    logger.info("Patched: %s", rel_path)


def _patch_flux_nodes() -> None:
    from patches.apply_all import COMFYUI_ROOT
    flux_path = COMFYUI_ROOT / "comfy_extras" / "nodes_flux.py"
    if not flux_path.exists():
        return

    src = flux_path.read_text(encoding="utf-8")
    if _SENTINEL in src:
        return

    src = _patch_fp8_node_classes(src)
    src = _patch_comfy_kitchen_imports(src)

    # Flux-specific: UNETLoader with FP8 → fall back to FP16
    src = re.sub(
        r'(model_options\["dtype"\]\s*=\s*)torch\.float8_e4m3fn',
        r'\1torch.float16  # [P40-COMPAT] FP8→FP16',
        src,
    )
    src = re.sub(
        r'"FP8 e4m3fn"',
        '"FP8 e4m3fn (→FP16 on this GPU)"',
        src,
    )

    # Instead of crashing when FP8 is selected, silently downgrade
    _fp8_fallback = '''
# [P40-COMPAT] FP8 dtype resolver for Flux nodes
def _flux_resolve_dtype(dtype_str):
    import torch
    _MAP = {
        "default": None,
        "fp8_e4m3fn": torch.float16,      # FP8→FP16
        "fp8_e5m2": torch.float16,         # FP8→FP16
        "fp16": torch.float16,
        "bf16": torch.float16,             # BF16→FP16 on Pascal
        "fp32": torch.float32,
    }
    return _MAP.get(dtype_str.lower(), None)
'''
    if "_flux_resolve_dtype" not in src:
        src = _fp8_fallback + src

    src += f"\n{_SENTINEL}\n"
    flux_path.write_text(src, encoding="utf-8")
    logger.info("Patched: comfy_extras/nodes_flux.py")


def _patch_comfy_kitchen_references() -> None:
    """
    comfy_kitchen may be referenced in extra_nodes or custom_nodes directories.
    We create a stub module that satisfies imports without crashing.
    """
    from patches.apply_all import COMFYUI_ROOT
    stub_path = COMFYUI_ROOT / "comfy_kitchen" / "__init__.py"
    if stub_path.exists():
        return  # already present (either real or our stub)

    stub_path.parent.mkdir(parents=True, exist_ok=True)
    stub_path.write_text(
        '"""comfy_kitchen stub — inserted by patches/patch_nodes.py [P40-COMPAT]"""\n'
        '# This stub satisfies imports from code that references comfy_kitchen.\n'
        '# All functions are no-ops. On hardware that supports comfy_kitchen\n'
        '# (PyTorch 2.1+, sm_89+), remove this stub and install the real package.\n\n'
        'def __getattr__(name):\n'
        '    import warnings\n'
        '    warnings.warn(f"comfy_kitchen.{name} is not available (P40/M40 stub)", stacklevel=2)\n'
        '    return None\n',
        encoding="utf-8",
    )
    logger.info("Created comfy_kitchen stub at %s", stub_path)


# ---------------------------------------------------------------------------

def _patch_comfy_kitchen_imports(src: str) -> str:
    """Replace comfy_kitchen imports with no-ops."""
    src = re.sub(
        r'^(import comfy_kitchen\b[^\n]*)',
        r'try:\n    \1\nexcept ImportError:  # [P40-COMPAT]\n    pass',
        src,
        flags=re.MULTILINE,
    )
    src = re.sub(
        r'^(from comfy_kitchen[^\n]+)',
        r'try:\n    \1\nexcept ImportError:  # [P40-COMPAT]\n    pass',
        src,
        flags=re.MULTILINE,
    )
    return src


def _patch_fp8_node_classes(src: str) -> str:
    """
    FP8 checkpoint loader nodes: add a warning and dtype downgrade so the
    node still loads (avoiding red nodes) but uses FP16.
    """
    # Wrap FP8 dtype usage in node INPUT_TYPES and forward methods
    src = re.sub(
        r'\btorch\.float8_e4m3fn\b',
        'torch.float16  # [P40-COMPAT FP8→FP16]',
        src,
    )
    src = re.sub(
        r'\btorch\.float8_e5m2\b',
        'torch.float16  # [P40-COMPAT FP8→FP16]',
        src,
    )
    return src


def _patch_diffusers_bf16_default(src: str) -> str:
    """
    diffusers / transformers loaders default to BF16 on CUDA.
    Override to FP16 for Pascal/Maxwell.
    """
    # Pattern: torch_dtype=torch.bfloat16
    src = re.sub(
        r'torch_dtype\s*=\s*torch\.bfloat16',
        'torch_dtype=torch.float16  # [P40-COMPAT BF16→FP16]',
        src,
    )
    return src
