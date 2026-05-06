"""
compat/aiohttp_compat.py
========================
Patches aiohttp to handle `Host: hostname:port` headers correctly.

In aiohttp 3.9.x, `BaseRequest.url` is built as:
    URL.build(scheme=self.scheme, host=self.host)
where `self.host` returns the raw Host header including port (e.g.
"localhost:8188"). yarl 1.9+ rejects a colon in the `host` argument of
`URL.build()` because that parameter must be a bare hostname.

Fix: replace the `URL` name in `aiohttp.web_request`'s module globals
with a subclass whose `build()` classmethod splits `host:port` before
delegating, so yarl receives only the bare hostname. This matches the
behaviour fixed in aiohttp 3.10+.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("comfyui_compat")

_patched = False


def patch_aiohttp_host() -> None:
    """Inject a permissive URL subclass into aiohttp.web_request."""
    global _patched
    if _patched:
        return

    try:
        import aiohttp.web_request as _wr
        from yarl import URL as _OrigURL

        # yarl.URL is a final C extension — cannot be subclassed.
        # Instead, inject a wrapper object whose .build() method strips the
        # port from `host` before delegating to the real URL.build().
        class _URLBuildWrapper:
            """Thin proxy for yarl.URL that fixes up host:port in build()."""

            def __getattr__(self, name: str):
                return getattr(_OrigURL, name)

            def __instancecheck__(self, instance) -> bool:  # type: ignore[override]
                return isinstance(instance, _OrigURL)

            def __call__(self, *args, **kwargs):
                return _OrigURL(*args, **kwargs)

            @staticmethod
            def build(*, host: str = "", port: int | None = None, **kw):
                # aiohttp 3.9 passes "hostname:port" as host; split it out.
                if host and not host.startswith("[") and ":" in host:
                    bare, _, port_str = host.rpartition(":")
                    if port_str.isdigit():
                        host = bare
                        if port is None:
                            port = int(port_str)
                return _OrigURL.build(host=host, port=port, **kw)

        _wr.URL = _URLBuildWrapper()  # type: ignore[attr-defined]
        _patched = True
        logger.info(
            "Patched aiohttp.web_request.URL to accept host:port (aiohttp 3.9 / yarl 1.9+ compat)."
        )
    except Exception as exc:
        logger.warning("Could not patch aiohttp URL: %s", exc)
