"""
comfy_kitchen stub — P40/M40 compatibility shim.

comfy_kitchen provides FP8/FP4 tensor primitives that require sm_89+
(Ada Lovelace / H100). On Pascal (sm_61) and Maxwell (sm_52) these
operations are not available. This stub provides the minimal API shape
newer ComfyUI code expects and safely disables all accelerated backends.
"""

from __future__ import annotations

from . import tensor  # noqa: F401


class _Registry:
	def __init__(self) -> None:
		self.disabled = set()

	def disable(self, backend: str) -> None:
		self.disabled.add(backend)


registry = _Registry()


def list_backends() -> dict:
	# Present but disabled so callers can proceed without crashing.
	return {
		"cuda": "disabled",
		"triton": "disabled",
	}


__all__ = ["tensor", "registry", "list_backends"]
