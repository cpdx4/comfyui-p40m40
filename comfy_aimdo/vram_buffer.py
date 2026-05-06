"""Compatibility stub for comfy_aimdo.vram_buffer."""

from __future__ import annotations


class VRAMBuffer:
    def __init__(self, size: int, device_index: int | None = None) -> None:
        self._size = int(size)
        self._device_index = device_index

    def get(self, size: int, offset: int = 0):
        # Return None-like payload; conversion shim handles fallback.
        return None

    def size(self) -> int:
        return self._size
