"""Compatibility stub for comfy_aimdo.model_vbar."""

from __future__ import annotations


class ModelVBAR:
    def __init__(self, *args, **kwargs) -> None:
        pass


def vbar_fault(*args, **kwargs):
    return None


def vbar_signature_compare(*args, **kwargs) -> bool:
    return False


def vbar_unpin(*args, **kwargs) -> None:
    return None


def vbars_analyze() -> int:
    return 0


def vbars_reset_watermark_limits() -> None:
    return None
