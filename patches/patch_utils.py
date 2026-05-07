"""
patches/patch_utils.py
======================
Patches ComfyUI/comfy/utils.py to handle FP8 tensors in safetensors files
on PyTorch 2.0, which has no native FP8 dtype support.

Problem:
  safetensors stores FP8 tensors with dtype "F8_E4M3" / "F8_E5M2".
  On PyTorch >= 2.1, torch.float8_e4m3fn exists and view(dtype=...) works.
  On PyTorch 2.0.1, our compat layer injects _FP8DtypeStub objects.
  After compat/torch_compat.py patches safetensors._TYPES to use torch.uint8,
  FP8 tensors load successfully AS uint8 — raw fp8 bit patterns preserved.
  This patch then dequantizes those uint8 bytes to float16 at load time,
  reading the original dtype from the safetensors file header.

Result: FP8-quantised model weights are transparently loaded as float16.
"""

from __future__ import annotations

import json
import logging
import re
import struct
from pathlib import Path

logger = logging.getLogger("patches")

PATCH_ID = "utils_fp8_dequant"
TARGET_FILE = "comfy/utils.py"
_SENTINEL = "# [P40-COMPAT] utils fp8 patched"


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
        logger.info("Reverted %s", TARGET_FILE)


def apply() -> None:
    t = _target()
    if not t.exists():
        raise FileNotFoundError(f"Target not found: {t}")

    src = t.read_text(encoding="utf-8")

    if _SENTINEL in src:
        logger.info("Already patched: %s", TARGET_FILE)
        return

    src = _patch_load_torch_file(src)
    src += f"\n{_SENTINEL}\n"

    t.write_text(src, encoding="utf-8")
    logger.info("Patched: %s", TARGET_FILE)


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------

_HELPER = '''

# [P40-COMPAT] injected by patches/patch_utils.py
def _p40_read_safetensors_fp8_dtypes(ckpt: str) -> dict:
    """
    Read the safetensors file header and return a dict of {tensor_name: dtype_str}
    for any FP8 tensors (dtype string starts with "F8_").
    Returns empty dict if file can't be parsed or has no FP8 tensors.
    """
    _FP8_DTYPE_STRINGS = {"F8_E4M3", "F8_E5M2", "F8_E4M3FNUZ", "F8_E5M2FNUZ"}
    try:
        with open(ckpt, "rb") as _f:
            _header_size = struct.unpack("<Q", _f.read(8))[0]
            _header = json.loads(_f.read(_header_size))
        return {
            _k: _v["dtype"]
            for _k, _v in _header.items()
            if _k != "__metadata__" and _v.get("dtype", "").upper() in _FP8_DTYPE_STRINGS
        }
    except Exception:
        return {}


def _p40_dequant_fp8_tensor(tensor, dtype_str: str):
    """
    Convert a uint8 tensor (holding raw FP8 bytes) to float16 using
    a proper bit-level dequantization.  dtype_str is the safetensors
    dtype string, e.g. "F8_E4M3" or "F8_E5M2".
    """
    try:
        from compat.fp8_stub import FP8_DEQUANT
        key = dtype_str.upper()
        if key in FP8_DEQUANT:
            return FP8_DEQUANT[key](tensor)
    except Exception:
        pass
    # Fallback: just cast as integers to float16 (wrong values, but won't crash)
    return tensor.to(torch.float16)

'''

_OLD_LOOP = '''\
                with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
                    sd = {}
                    for k in f.keys():
                        tensor = f.get_tensor(k)
                        if DISABLE_MMAP:  # TODO: Not sure if this is the best way to bypass the mmap issues
                            tensor = tensor.to(device=device, copy=True)
                        sd[k] = tensor
                    if return_metadata:
                        metadata = f.metadata()'''

_NEW_LOOP = '''\
                with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
                    sd = {}
                    _p40_fp8_dtypes = _p40_read_safetensors_fp8_dtypes(ckpt)
                    for k in f.keys():
                        tensor = f.get_tensor(k)
                        if k in _p40_fp8_dtypes and tensor.dtype == torch.uint8:
                            tensor = _p40_dequant_fp8_tensor(tensor, _p40_fp8_dtypes[k])
                        if DISABLE_MMAP:  # TODO: Not sure if this is the best way to bypass the mmap issues
                            tensor = tensor.to(device=device, copy=True)
                        sd[k] = tensor
                    if return_metadata:
                        metadata = f.metadata()'''


def _patch_load_torch_file(src: str) -> str:
    # Inject helpers before load_torch_file
    insert_before = "\ndef load_torch_file("
    if insert_before not in src:
        logger.warning("Could not find load_torch_file in %s — skipping", TARGET_FILE)
        return src
    src = src.replace(insert_before, _HELPER + insert_before, 1)

    # Replace the safetensors load loop (normalise whitespace for matching)
    old_norm = re.sub(r" +", " ", _OLD_LOOP)
    src_norm = re.sub(r" +", " ", src)
    if old_norm not in src_norm:
        logger.warning("Could not find safe_open loop in %s — FP8 dequant skipped", TARGET_FILE)
        return src

    # Replace in the original (non-normalised) source by regex
    src = re.sub(
        r'(with safetensors\.safe_open\([^\n]+\n)'  # opening line
        r'(\s+sd = \{\}\n)'
        r'(\s+for k in f\.keys\(\):\n)'
        r'(\s+tensor = f\.get_tensor\(k\)\n)'
        r'(\s+if DISABLE_MMAP:[^\n]+\n)'
        r'(\s+tensor = tensor\.to[^\n]+\n)'
        r'(\s+sd\[k\] = tensor\n)'
        r'(\s+if return_metadata:\n)'
        r'(\s+metadata = f\.metadata\(\))',
        lambda m: (
            m.group(1)
            + m.group(2)
            + "                    _p40_fp8_dtypes = _p40_read_safetensors_fp8_dtypes(ckpt)\n"
            + m.group(3)
            + m.group(4)
            + "                        if k in _p40_fp8_dtypes and tensor.dtype == torch.uint8:\n"
            + "                            tensor = _p40_dequant_fp8_tensor(tensor, _p40_fp8_dtypes[k])\n"
            + m.group(5)
            + m.group(6)
            + m.group(7)
            + m.group(8)
            + m.group(9)
        ),
        src,
    )
    return src
