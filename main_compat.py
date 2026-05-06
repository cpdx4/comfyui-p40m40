"""
main_compat.py
==============
Drop-in replacement for ComfyUI/main.py that installs the P40/M40
compatibility layer before importing any ComfyUI module.

Usage (exactly like the original main.py):
  python main_compat.py [--listen 0.0.0.0] [--port 8188] [...]

Extra flags:
  --base-directory PATH    Path to the ComfyUI submodule (default: ./ComfyUI)
  --skip-patch-check       Skip verifying source patches are applied
    --skip-preflight         Skip startup preflight checks
  --compat-verbose         Enable verbose compat layer logging
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve paths before any other import
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
COMFYUI_DEFAULT = THIS_DIR / "ComfyUI"
UPSTREAM_COMFYUI_URL = "https://github.com/comfyanonymous/ComfyUI.git"


def _parse_compat_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse our extra args, pass the rest through to ComfyUI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--base-directory",
        default=str(COMFYUI_DEFAULT),
        help="Path to the ComfyUI directory (default: ./ComfyUI)",
    )
    parser.add_argument(
        "--skip-patch-check",
        action="store_true",
        help="Skip checking that source patches are applied",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip startup preflight checks",
    )
    parser.add_argument(
        "--compat-verbose",
        action="store_true",
        help="Enable verbose compat layer output",
    )
    return parser.parse_known_args()


def _verify_patches(comfyui_dir: Path) -> None:
    """Warn if source patches haven't been applied."""
    try:
        sys.path.insert(0, str(THIS_DIR))
        from patches.apply_all import COMFYUI_ROOT, PATCH_MODULES
        import importlib

        missing = []
        for mod_name in PATCH_MODULES:
            try:
                mod = importlib.import_module(mod_name)
                if not mod.check():
                    missing.append(getattr(mod, "PATCH_ID", mod_name))
            except Exception:
                pass

        if missing:
            print(
                f"\n[compat] WARNING: The following source patches are not applied:\n"
                f"  {missing}\n"
                f"  Run: python patches/apply_all.py\n"
                f"  Proceeding with runtime-only patches (may miss some fixes).\n",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"[compat] Could not verify patches: {exc}", file=sys.stderr)


def _ensure_comfyui_source(comfyui_dir: Path) -> Path:
    """
    Ensure a valid ComfyUI checkout exists and contains main.py.
    If the directory exists but is empty (common with a bind mount), clone upstream.
    """
    main_py = comfyui_dir / "main.py"
    if comfyui_dir.exists() and main_py.exists():
        return comfyui_dir

    if not comfyui_dir.exists():
        comfyui_dir.mkdir(parents=True, exist_ok=True)

    # If the directory is empty or missing main.py, bootstrap it.
    is_empty = not any(comfyui_dir.iterdir())
    if is_empty or not main_py.exists():
        print(
            f"[compat] ComfyUI source missing at {comfyui_dir}; bootstrapping from upstream...",
            file=sys.stderr,
        )
        try:
            if any(comfyui_dir.iterdir()):
                # Non-empty but invalid checkout; keep files and only fail with guidance.
                raise RuntimeError(
                    f"Directory {comfyui_dir} exists but does not contain main.py"
                )
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    UPSTREAM_COMFYUI_URL,
                    str(comfyui_dir),
                ]
            )
        except Exception as exc:
            print(
                "ERROR: Failed to bootstrap ComfyUI source.\n"
                f"Target: {comfyui_dir}\n"
                f"Reason: {exc}\n"
                "Fix options:\n"
                "  1) Initialize the submodule on host: git submodule update --init --recursive\n"
                "  2) Remove empty bind-mount folder and restart container\n"
                "  3) Run with --base-directory pointing to a valid ComfyUI checkout",
                file=sys.stderr,
            )
            sys.exit(1)

    if not main_py.exists():
        print(
            f"ERROR: ComfyUI checkout at {comfyui_dir} is invalid (missing main.py)",
            file=sys.stderr,
        )
        sys.exit(1)

    return comfyui_dir


def _apply_patches_if_needed() -> None:
    """
    Apply source-level patches automatically when they are missing.
    This prevents runtime-only mode in fresh Docker bootstraps.
    """
    try:
        from patches import apply_all
        if not apply_all.run_patches("check"):
            print("[compat] Applying missing source patches...", file=sys.stderr)
            ok = apply_all.run_patches("apply")
            if not ok:
                print(
                    "ERROR: Failed to apply one or more source patches. "
                    "Run: python patches/apply_all.py and inspect output.",
                    file=sys.stderr,
                )
                sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Could not run patch manager: {exc}", file=sys.stderr)
        sys.exit(1)


def _run_startup_preflight(comfyui_dir: Path) -> None:
    """Run lightweight boot diagnostics for GPU visibility, source health, and patch status."""
    try:
        from scripts.startup_preflight import run_preflight

        run_preflight(comfyui_dir)
    except Exception as exc:
        print(f"[compat] WARNING: preflight failed to run: {exc}", file=sys.stderr)


def main() -> None:
    compat_args, comfyui_args = _parse_compat_args()
    comfyui_dir = _ensure_comfyui_source(Path(compat_args.base_directory).resolve())

    # ---------------------------------------------------------------------------
    # 1. Add compat/ and ComfyUI/ to sys.path BEFORE any ComfyUI import
    # ---------------------------------------------------------------------------
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    if str(comfyui_dir) not in sys.path:
        sys.path.insert(1, str(comfyui_dir))

    if not compat_args.skip_preflight:
        _run_startup_preflight(comfyui_dir)

    # ---------------------------------------------------------------------------
    # 2. Install the compatibility layer
    #    This must happen before `import comfy.*` or `import nodes`
    # ---------------------------------------------------------------------------
    from compat import install
    install(verbose=compat_args.compat_verbose)

    # Ensure source patches are present before ComfyUI imports.
    _apply_patches_if_needed()

    # ---------------------------------------------------------------------------
    # 3. Optionally verify source patches
    # ---------------------------------------------------------------------------
    if not compat_args.skip_patch_check:
        _verify_patches(comfyui_dir)

    # ---------------------------------------------------------------------------
    # 4. Set environment variables that ComfyUI reads at import time
    # ---------------------------------------------------------------------------
    # Ensure ComfyUI's folder resolution finds the right paths
    os.environ.setdefault("COMFYUI_PATH", str(comfyui_dir))

    # ---------------------------------------------------------------------------
    # 5. Reconstruct sys.argv for ComfyUI's own arg parser
    #    ComfyUI's main.py reads sys.argv, so we pass our cleaned args through.
    # ---------------------------------------------------------------------------
    sys.argv = [str(comfyui_dir / "main.py")] + comfyui_args

    # ---------------------------------------------------------------------------
    # 6. Change working directory to ComfyUI so its relative path lookups work
    # ---------------------------------------------------------------------------
    os.chdir(comfyui_dir)

    # ---------------------------------------------------------------------------
    # 7. Run ComfyUI's main module
    # ---------------------------------------------------------------------------
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "comfyui_main", comfyui_dir / "main.py"
    )
    if spec is None or spec.loader is None:
        print(
            f"ERROR: Could not load ComfyUI main.py from {comfyui_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    mod = importlib.util.module_from_spec(spec)
    sys.modules["__comfyui_main__"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]


if __name__ == "__main__":
    main()
