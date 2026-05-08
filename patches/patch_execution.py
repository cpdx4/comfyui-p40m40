"""
patches/patch_execution.py
==========================
Patches ComfyUI/comfy/execution.py to pre-validate oversized UNET files
before node execution starts.

Why:
- On P40/M40 systems with limited host RAM, very large safetensors UNET files
  fail with mmap ENOMEM.
- Without this precheck, ComfyUI may spend minutes loading CLIP/other nodes
  before failing late at UNET load time.

Behavior:
- If any UNETLoader node references a diffusion model larger than
  P40_MAX_SAFETENSORS_GB (default 32), validation fails immediately with a
  clear error message.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("patches")

PATCH_ID = "execution_unet_size_precheck"
TARGET_FILE = "execution.py"
_SENTINEL = "# [P40-COMPAT] execution unet size precheck patched"


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

    helper = '''

def _p40_validate_unet_size_limits(prompt):
    import os
    try:
        import folder_paths
    except Exception:
        return None

    try:
        _max_gb = float(os.environ.get("P40_MAX_SAFETENSORS_GB", "32"))
    except Exception:
        _max_gb = 32.0

    # max <= 0 disables this precheck.
    if _max_gb <= 0:
        return None

    for _node_id, _node in prompt.items():
        if _node.get("class_type") != "UNETLoader":
            continue
        _inputs = _node.get("inputs", {})
        _unet_name = _inputs.get("unet_name")
        if not isinstance(_unet_name, str) or len(_unet_name) == 0:
            continue
        try:
            _path = folder_paths.get_full_path_or_raise("diffusion_models", _unet_name)
            _size_gb = os.path.getsize(_path) / (1024.0 ** 3)
        except Exception:
            continue

        if _size_gb > _max_gb:
            return {
                "node_id": _node_id,
                "class_type": "UNETLoader",
                "unet_name": _unet_name,
                "size_gb": _size_gb,
                "max_gb": _max_gb,
                "path": _path,
            }
    return None
'''

    insert_after = "def full_type_name(klass):\n"
    if insert_after not in src:
        logger.warning("Could not find full_type_name in %s — precheck skipped", TARGET_FILE)
    else:
        # Insert helper right after full_type_name function definition block.
        anchor = "    return module + '.' + klass.__qualname__\n"
        if anchor in src and "def _p40_validate_unet_size_limits(prompt):" not in src:
            src = src.replace(anchor, anchor + helper, 1)

    old = "    if len(outputs) == 0:\n"
    new = '''    _p40_unet_too_large = _p40_validate_unet_size_limits(prompt)
    if _p40_unet_too_large is not None:
        _nid = _p40_unet_too_large["node_id"]
        _cls = _p40_unet_too_large["class_type"]
        _uname = _p40_unet_too_large["unet_name"]
        _size = _p40_unet_too_large["size_gb"]
        _max = _p40_unet_too_large["max_gb"]
        _path = _p40_unet_too_large["path"]
        _details = (
            f"UNET '{_uname}' is {_size:.2f} GiB (limit {_max:.2f} GiB via P40_MAX_SAFETENSORS_GB)\\n"
            f"Path: {_path}"
        )
        error = {
            "type": "prompt_unet_too_large",
            "message": "Prompt rejected: UNET file is too large for this host policy",
            "details": _details,
            "extra_info": {
                "node_id": _nid,
                "class_type": _cls,
                "unet_name": _uname,
                "size_gb": _size,
                "max_gb": _max,
                "path": _path,
            }
        }
        node_errors = {
            _nid: {
                "errors": [error],
                "dependent_outputs": list(outputs),
                "class_type": _cls,
            }
        }
        return (False, error, [], node_errors)

    if len(outputs) == 0:
'''

    if old not in src:
        logger.warning("Could not find validate_prompt outputs check in %s — precheck hook skipped", TARGET_FILE)
    else:
        src = src.replace(old, new, 1)

    src += f"\n{_SENTINEL}\n"
    t.write_text(src, encoding="utf-8")
    logger.info("Patched: %s", TARGET_FILE)
