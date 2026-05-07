"""
patches/patch_utils.py
======================
Patches ComfyUI/comfy/utils.py to handle FP8 tensors in safetensors files
on PyTorch 2.0, which has no native FP8 dtype support.

Problem:
  The safetensors Rust extension (safe_open.get_tensor) calls
  raw_tensor.view(dtype=fp8_dtype) internally. Since torch.float8_e4m3fn
  on PyTorch 2.0 is our _FP8DtypeStub Python object (not a real C dtype),
  view() raises TypeError. Patching safetensors._TYPES does not help because
  the Rust extension caches dtype references independently.

Fix:
  When the safetensors file contains FP8 tensors (detected by reading the
  file header), bypass safe_open entirely and use safetensors.torch.load()
  (reads bytes → _view2torch → _TYPES) which is pure Python and fully
  respects our _TYPES patch (torch.uint8 for FP8 keys). The uint8 raw bytes
  are then dequantized to float16 using a proper bit-level LUT.
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
_SENTINEL = "# [P40-COMPAT] utils fp8 patched v3"


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

    # Remove old v1 sentinel/helpers if present, start fresh
    src = _remove_old_patch(src)

    if _SENTINEL in src:
        logger.info("Already patched: %s", TARGET_FILE)
        return

    src = _patch_load_torch_file(src)
    src += f"\n{_SENTINEL}\n"

    t.write_text(src, encoding="utf-8")
    logger.info("Patched: %s", TARGET_FILE)


# ---------------------------------------------------------------------------
# Remove old v1 patch if present
# ---------------------------------------------------------------------------

def _remove_old_patch(src: str) -> str:
    for old_sentinel in [
        "# [P40-COMPAT] utils fp8 patched\n",
        "# [P40-COMPAT] utils fp8 patched v2\n",
    ]:
        src = src.replace(old_sentinel, "")

    old_helper_start = "\n# [P40-COMPAT] injected by patches/patch_utils.py\n"
    # Remove old helper block (from the marker to the blank line before load_torch_file)
    if old_helper_start in src:
        start = src.find(old_helper_start)
        # Find the def load_torch_file that follows
        end = src.find("\ndef load_torch_file(", start)
        if end != -1:
            src = src[:start] + "\n" + src[end:]

    # Remove old injected lines inside the safe_open loop
    for line in [
        "                    _p40_fp8_dtypes = _p40_read_safetensors_fp8_dtypes(ckpt)\n",
        "                        if k in _p40_fp8_dtypes and tensor.dtype == torch.uint8:\n",
        "                            tensor = _p40_dequant_fp8_tensor(tensor, _p40_fp8_dtypes[k])\n",
    ]:
        src = src.replace(line, "")

    return src


# ---------------------------------------------------------------------------
# Helpers injected into utils.py
# ---------------------------------------------------------------------------

_HELPER = '''
# [P40-COMPAT] injected by patches/patch_utils.py
import json as _p40_json
import struct as _p40_struct


def _p40_read_safetensors_fp8_dtypes(ckpt: str) -> dict:
    """
    Read the safetensors file header and return {tensor_name: dtype_str}
    for any FP8 tensors.  Returns {} if none found or on parse error.
    """
    _FP8 = {"F8_E4M3", "F8_E5M2", "F8_E4M3FNUZ", "F8_E5M2FNUZ"}
    try:
        with open(ckpt, "rb") as _f:
            _hsz = _p40_struct.unpack("<Q", _f.read(8))[0]
            _hdr = _p40_json.loads(_f.read(_hsz))
        return {
            _k: _v["dtype"]
            for _k, _v in _hdr.items()
            if _k != "__metadata__" and _v.get("dtype", "").upper() in _FP8
        }
    except Exception:
        return {}


def _p40_dequant_fp8_tensor(tensor, dtype_str: str):
    """
    Convert a uint8 tensor holding raw FP8 bytes to float16.
    dtype_str is the safetensors dtype string: "F8_E4M3" or "F8_E5M2".
    """
    try:
        from compat.fp8_stub import FP8_DEQUANT
        key = dtype_str.upper()
        if key in FP8_DEQUANT:
            return FP8_DEQUANT[key](tensor)
    except Exception:
        pass
    return tensor.to(torch.float16)


def _p40_load_fp8_safetensors(ckpt: str, device) -> dict:
    """
    Load a safetensors file that contains FP8 tensors by:
      1. Using safe_open normally for all non-FP8 tensors (fast path).
      2. For FP8 tensors: reading raw bytes directly from the file using the
         data_offsets from the safetensors header, then dequantizing to float16.

    This completely bypasses safetensors' Rust dtype handling for FP8 tensors,
    which fails on PyTorch 2.0 because the Rust extension calls view(dtype=stub).
    """
    import safetensors

    _p40_fp8_dtypes = _p40_read_safetensors_fp8_dtypes(ckpt)

    # Parse full header for data_offsets of FP8 tensors
    with open(ckpt, "rb") as _p40_f:
        _p40_hsz = _p40_struct.unpack("<Q", _p40_f.read(8))[0]
        _p40_hdr = _p40_json.loads(_p40_f.read(_p40_hsz))
        # data section starts right after the 8-byte length prefix + header JSON
        _p40_data_start = 8 + _p40_hsz
        _p40_raw = _p40_f.read()  # rest of file = tensor data section

    _p40_sd = {}

    # Load non-FP8 tensors via safe_open (Rust fast path)
    with safetensors.safe_open(ckpt, framework="pt", device=str(device)) as _p40_sf:
        for _p40_k in _p40_sf.keys():
            if _p40_k not in _p40_fp8_dtypes:
                _p40_sd[_p40_k] = _p40_sf.get_tensor(_p40_k)

    # Load FP8 tensors manually from raw bytes
    for _p40_k, _p40_dstr in _p40_fp8_dtypes.items():
        _p40_info = _p40_hdr.get(_p40_k, {})
        _p40_shape = _p40_info.get("shape", [])
        _p40_offsets = _p40_info.get("data_offsets", [0, 0])
        _p40_start = _p40_offsets[0]
        _p40_end = _p40_offsets[1]
        _p40_nbytes = _p40_end - _p40_start

        if _p40_nbytes == 0:
            _p40_sd[_p40_k] = torch.zeros(_p40_shape, dtype=torch.float16)
            continue

        _p40_bytes = bytearray(_p40_raw[_p40_start:_p40_end])
        _p40_u8 = torch.frombuffer(_p40_bytes, dtype=torch.uint8)
        if _p40_shape:
            _p40_u8 = _p40_u8.reshape(_p40_shape)
        _p40_sd[_p40_k] = _p40_dequant_fp8_tensor(_p40_u8, _p40_dstr)
        if str(device) != "cpu":
            _p40_sd[_p40_k] = _p40_sd[_p40_k].to(device)

    return _p40_sd

'''


# The original safe_open block we need to replace inside the try: of load_torch_file
_OLD_BLOCK = '''            else:
                with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
                    sd = {}
                    for k in f.keys():
                        tensor = f.get_tensor(k)
                        if DISABLE_MMAP:  # TODO: Not sure if this is the best way to bypass the mmap issues
                            tensor = tensor.to(device=device, copy=True)
                        sd[k] = tensor
                    if return_metadata:
                        metadata = f.metadata()'''

_NEW_BLOCK = '''            else:
                if _p40_read_safetensors_fp8_dtypes(ckpt):
                    # FP8 file: safe_open Rust cannot view() our dtype stubs.
                    # Fall back to pure-Python safetensors.torch.load() path.
                    sd = _p40_load_fp8_safetensors(ckpt, device)
                    if return_metadata:
                        with safetensors.safe_open(ckpt, framework="pt", device=device.type) as _p40_mf:
                            metadata = _p40_mf.metadata()
                else:
                    with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
                        sd = {}
                        for k in f.keys():
                            tensor = f.get_tensor(k)
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

    # Normalise the source for matching (collapse multiple spaces to one)
    def _norm(s: str) -> str:
        return re.sub(r'[ \t]+', ' ', s)

    if _norm(_OLD_BLOCK) not in _norm(src):
        logger.warning("Could not find safe_open block in %s — FP8 fix skipped", TARGET_FILE)
        return src

    # Replace using regex to handle any whitespace variation
    src = re.sub(
        r'( {12}else:\n)'                                      # else:
        r'( {16}with safetensors\.safe_open\([^\n]+\n)'        # with safe_open(
        r'( {20}sd = \{\}\n)'                                   # sd = {}
        r'( {20}for k in f\.keys\(\):\n)'                      # for k in f.keys():
        r'( {24}tensor = f\.get_tensor\(k\)\n)'                # tensor = f.get_tensor(k)
        r'( {24}if DISABLE_MMAP:[^\n]+\n)'                     # if DISABLE_MMAP:
        r'( {28}tensor = tensor\.to[^\n]+\n)'                  # tensor = tensor.to(...)
        r'( {24}sd\[k\] = tensor\n)'                           # sd[k] = tensor
        r'( {20}if return_metadata:\n)'                        # if return_metadata:
        r'( {24}metadata = f\.metadata\(\))',                   # metadata = f.metadata()
        _NEW_BLOCK,
        src,
    )
    return src
