"""
patches/apply_all.py
====================
Master patch runner.  Applies (or verifies) all source-level patches to the
ComfyUI submodule.

Usage:
  python patches/apply_all.py            # apply all patches
  python patches/apply_all.py --check    # verify state, exit 1 if stale
  python patches/apply_all.py --revert   # restore .orig backups
  python patches/apply_all.py --status   # show which patches are applied

Each individual patch module must export:
  PATCH_ID       : str  — unique identifier
  TARGET_FILE    : str  — path relative to ComfyUI/ submodule root
  check()        : bool — True if patch is already applied
  apply()        : None — apply the patch (idempotent)
  revert()       : None — restore .orig backup
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("patches")
logging.basicConfig(level=logging.INFO, format="[patch] %(levelname)s %(message)s")

# Root of this repository
REPO_ROOT = Path(__file__).resolve().parent.parent
COMFYUI_ROOT = REPO_ROOT / "ComfyUI"
STATE_FILE = Path(__file__).resolve().parent / ".patch_state.json"

# Ordered list of patch modules (applied in this order)
PATCH_MODULES = [
    "patches.patch_model_management",
    "patches.patch_model_patcher",
    "patches.patch_ops",
    "patches.patch_attention",
    "patches.patch_nodes",
    "patches.patch_utils",
]


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_patches(mode: str) -> bool:
    """
    mode: "apply" | "check" | "revert" | "status"
    Returns True if all patches are in the expected state.
    """
    if not COMFYUI_ROOT.exists():
        logger.error(
            "ComfyUI submodule not found at %s. "
            "Run: git submodule update --init --recursive",
            COMFYUI_ROOT,
        )
        return False

    state = _load_state()
    all_ok = True

    for mod_name in PATCH_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as exc:
            logger.error("Could not import patch module %s: %s", mod_name, exc)
            all_ok = False
            continue

        patch_id: str = getattr(mod, "PATCH_ID", mod_name)
        target_rel: str = getattr(mod, "TARGET_FILE", "")
        target_path = COMFYUI_ROOT / target_rel if target_rel else None

        if mode == "status":
            applied = mod.check()
            print(f"  {'✓' if applied else '✗'} {patch_id}")
            continue

        if mode == "check":
            applied = mod.check()
            if not applied:
                logger.warning("Patch NOT applied: %s", patch_id)
                all_ok = False
            else:
                logger.info("OK: %s", patch_id)
            continue

        if mode == "revert":
            if target_path and (target_path.parent / (target_path.name + ".orig")).exists():
                mod.revert()
                logger.info("Reverted: %s", patch_id)
            else:
                logger.debug("No .orig backup for %s — skipping revert.", patch_id)
            continue

        # mode == "apply"
        if mod.check():
            logger.info("Already applied: %s", patch_id)
        else:
            logger.info("Applying: %s ...", patch_id)
            try:
                if target_path and target_path.exists():
                    # Backup before first modification
                    orig = target_path.parent / (target_path.name + ".orig")
                    if not orig.exists():
                        orig.write_bytes(target_path.read_bytes())
                        state[patch_id] = {
                            "orig_sha256": _file_sha256(orig),
                            "target": str(target_path.relative_to(REPO_ROOT)),
                        }
                        _save_state(state)
                mod.apply()
                logger.info("Applied: %s", patch_id)
            except Exception as exc:
                logger.error("FAILED to apply %s: %s", patch_id, exc)
                all_ok = False

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="ComfyUI P40/M40 patch manager")
    parser.add_argument(
        "mode",
        nargs="?",
        default="apply",
        choices=["apply", "check", "revert", "status"],
    )
    args = parser.parse_args()

    # Add repo root to sys.path so patch modules can be imported
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    ok = run_patches(args.mode)

    if args.mode == "check" and not ok:
        logger.error("Some patches are not applied.  Run: python patches/apply_all.py")
        sys.exit(1)
    elif args.mode == "apply" and not ok:
        logger.error("Some patches failed.  Check output above.")
        sys.exit(1)

    if args.mode == "apply":
        logger.info("All patches applied successfully.")


if __name__ == "__main__":
    main()
