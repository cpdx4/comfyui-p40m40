"""Compatibility stub for comfy_aimdo.

Upstream ComfyUI imports comfy_aimdo unconditionally in newer builds to support
DynamicVRAM. That package is not usable on the P40/M40 compatibility target, so
this stub exposes the small API surface ComfyUI expects and always disables the
feature cleanly.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("comfyui_compat")
_initialized = False


def init() -> bool:
    global _initialized
    _initialized = True
    logger.info("comfy_aimdo stub active: DynamicVRAM disabled for compatibility")
    return False


def init_device(device_index: int | None) -> bool:
    logger.info(
        "comfy_aimdo stub ignoring init_device(%s): DynamicVRAM unsupported on this fork",
        device_index,
    )
    return False


def set_log_debug() -> None:
    logger.setLevel(logging.DEBUG)


def set_log_critical() -> None:
    logger.setLevel(logging.CRITICAL)


def set_log_error() -> None:
    logger.setLevel(logging.ERROR)


def set_log_warning() -> None:
    logger.setLevel(logging.WARNING)


def set_log_info() -> None:
    logger.setLevel(logging.INFO)
