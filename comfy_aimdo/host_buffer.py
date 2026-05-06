"""Compatibility stub for comfy_aimdo.host_buffer."""

from __future__ import annotations


class HostBuffer:
    def __init__(self, size: int) -> None:
        self._size = int(size)

    def get(self, size: int, offset: int = 0):
        return None

    def size(self) -> int:
        return self._size
