"""
patches/patch_ops.py
====================
Patches ComfyUI/comfy/ops.py to remove FP8 weight casting paths and
replace them with FP16 fallbacks.

ComfyUI ops.py defines:
  - disable_weight_init.Linear   — linear layer that optionally casts to FP8
  - manual_cast_weight()         — casts weights before a forward pass
  - CastWeightsTo context manager

All FP8 paths are gated to FP16 on hardware that doesn't support FP8.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger("patches")

PATCH_ID = "ops_fp8_removal"
TARGET_FILE = "comfy/ops.py"
_SENTINEL = "# [P40-COMPAT] ops patched v2"


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


def apply() -> None:
    t = _target()
    if not t.exists():
        raise FileNotFoundError(f"Target not found: {t}")

    src = t.read_text(encoding="utf-8")
    # Strip old v1 sentinel so we can re-apply with new v2 patches
    src = src.replace("\n# [P40-COMPAT] ops patched\n", "\n")
    if _SENTINEL in src:
        return

    src = _inject_compat_helper(src)
    src = _patch_fp8_cast_in_linear(src)
    src = _patch_fp8_manual_cast(src)
    src = _patch_fp8_isinstance_checks(src)
    src = _patch_torch_compile(src)
    src = _patch_layout_cls_none_fallback(src)
    src += f"\n{_SENTINEL}\n"

    t.write_text(src, encoding="utf-8")
    logger.info("Patched: %s", TARGET_FILE)


# ---------------------------------------------------------------------------

_HELPER = '''
# [P40-COMPAT] injected by patches/patch_ops.py
def _compat_resolve_dtype(dtype):
    """
    If dtype is FP8 or BF16 and the GPU doesn't support it, return FP16.
    Falls back gracefully if compat layer is not installed.
    """
    import torch
    try:
        from compat.fp8_stub import is_fp8_stub, resolve_dtype
        return resolve_dtype(dtype)
    except ImportError:
        pass
    # Fallback: check torch dtype name
    dtype_str = str(dtype)
    if "float8" in dtype_str:
        return torch.float16
    if dtype == torch.bfloat16:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            if props.major * 10 + props.minor < 80:
                return torch.float16
    return dtype
'''


def _inject_compat_helper(src: str) -> str:
    # Insert after the import block
    pos = src.rfind('\nimport ')
    if pos == -1:
        pos = src.rfind('\nfrom ')
    end_of_line = src.find('\n', pos + 1) if pos != -1 else 0
    return src[:end_of_line + 1] + _HELPER + src[end_of_line + 1:]


def _patch_fp8_cast_in_linear(src: str) -> str:
    """
    Pattern in ops.py:
        weight = weight.to(torch.float8_e4m3fn)
    Replace with:
        weight = weight.to(_compat_resolve_dtype(torch.float8_e4m3fn))
    """
    src = re.sub(
        r'(weight\s*=\s*weight\.to\()torch\.float8_e4m3fn(\))',
        r'\1_compat_resolve_dtype(torch.float8_e4m3fn)\2',
        src,
    )
    src = re.sub(
        r'(weight\s*=\s*weight\.to\()torch\.float8_e5m2(\))',
        r'\1_compat_resolve_dtype(torch.float8_e5m2)\2',
        src,
    )
    # Generic: any .to(torch.float8_*) call
    src = re.sub(
        r'\.to\(torch\.(float8_e4m3fn|float8_e5m2|float8_e4m3fnuz|float8_e5m2fnuz)\)',
        lambda m: f'.to(_compat_resolve_dtype(torch.{m.group(1)}))',
        src,
    )
    return src


def _patch_fp8_manual_cast(src: str) -> str:
    """
    manual_cast_weight often contains:
        if weight.dtype == torch.float8_e4m3fn:
            ...
    Wrap the dtype comparison so it evaluates False on PyTorch 2.0.
    """
    src = re.sub(
        r'(if\s+\w+\.dtype\s*==\s*)torch\.(float8_e4m3fn|float8_e5m2)\b',
        r'\1_compat_resolve_dtype(torch.\2) and False  # [P40-COMPAT]',
        src,
    )
    return src


def _patch_fp8_isinstance_checks(src: str) -> str:
    """
    Some versions do: if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
    Replace with a safe test.
    """
    src = re.sub(
        r'dtype\s+in\s+\(torch\.float8_e4m3fn,\s*torch\.float8_e5m2\)',
        'False  # [P40-COMPAT] FP8 not available',
        src,
    )
    src = re.sub(
        r'dtype\s+in\s+\[torch\.float8_e4m3fn,\s*torch\.float8_e5m2\]',
        'False  # [P40-COMPAT] FP8 not available',
        src,
    )
    return src


def _patch_torch_compile(src: str) -> str:
    src = re.sub(
        r'@torch\.compile\b[^\n]*\n',
        '# [P40-COMPAT] @torch.compile removed\n',
        src,
    )
    return src


def _patch_layout_cls_none_fallback(src: str) -> str:
    """
    When comfy_kitchen is unavailable, get_layout_class() returns None and
    calling None.Params(...) raises AttributeError.  Insert an early-return
    fallback right after the layout_cls assignment: when layout_cls is None
    we pop the scale params and store the (already-dequantized) weight as a
    plain float16 nn.Parameter, then return from _load_from_state_dict.
    """
    old = (
        "                    layout_cls = get_layout_class(self.layout_type)\n"
        "\n"
        "                    # Load format-specific parameters\n"
    )
    new = (
        "                    layout_cls = get_layout_class(self.layout_type)\n"
        "\n"
        "                    if layout_cls is None:  # [P40-COMPAT] comfy_kitchen unavailable\n"
        "                        logging.warning(\n"
        "                            '[P40-COMPAT] No layout class for %s at %s, loading as float16',\n"
        "                            self.quant_format, layer_name)\n"
        "                        for _p40_pn in qconfig['parameters']:\n"
        "                            _p40_pk = f'{prefix}{_p40_pn}'\n"
        "                            if _p40_pk in state_dict:\n"
        "                                manually_loaded_keys.append(_p40_pk)\n"
        "                                state_dict.pop(_p40_pk)\n"
        "                        self.weight = torch.nn.Parameter(\n"
        "                            weight.to(device=device, dtype=MixedPrecisionOps._compute_dtype),\n"
        "                            requires_grad=False,\n"
        "                        )\n"
        "                        super()._load_from_state_dict(\n"
        "                            state_dict, prefix, local_metadata, strict,\n"
        "                            missing_keys, unexpected_keys, error_msgs)\n"
        "                        for _p40_key in manually_loaded_keys:\n"
        "                            if _p40_key in missing_keys:\n"
        "                                missing_keys.remove(_p40_key)\n"
        "                        return\n"
        "\n"
        "                    # Load format-specific parameters\n"
    )
    if old not in src:
        logging.warning("_patch_layout_cls_none_fallback: target string not found, skipping")
        return src
    return src.replace(old, new, 1)
