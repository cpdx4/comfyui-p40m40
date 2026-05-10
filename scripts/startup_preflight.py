"""
Startup preflight checks for the ComfyUI P40/M40 fork.

This script is intentionally lightweight and safe to run on every boot.
It reports environment readiness before ComfyUI is imported.
"""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def _print(status: str, message: str) -> None:
    print(f"[preflight] {status} {message}")


def _check_python() -> Tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    return ok, f"Python {v.major}.{v.minor}.{v.micro}"


def _check_torch() -> Tuple[bool, str, bool]:
    try:
        import torch
    except Exception as exc:
        return False, f"PyTorch import failed: {exc}", False

    cuda_ok = bool(torch.cuda.is_available())
    details = []
    try:
        details.append(f"device_count={torch.cuda.device_count()}")
    except Exception as exc:
        details.append(f"device_count_error={exc}")

    if not cuda_ok:
        try:
            torch.zeros(1, device="cuda")
        except Exception as exc:
            details.append(f"cuda_init_error={exc}")

    msg = (
        f"PyTorch {torch.__version__} | CUDA build {torch.version.cuda} | "
        f"cuda_available={cuda_ok}"
    )
    if details:
        msg = f"{msg} | {' | '.join(details)}"
    return True, msg, cuda_ok


def _check_nvidia_smi() -> Tuple[bool, str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False, "nvidia-smi not found in container PATH"

    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=8,
        ).strip()
        if not out:
            return False, "nvidia-smi returned no GPU rows"
        first = out.splitlines()[0]
        count = len(out.splitlines())
        return True, f"nvidia-smi sees {count} GPU(s), first={first}"
    except Exception as exc:
        return False, f"nvidia-smi failed: {exc}"


def _check_comfyui_source(comfyui_dir: Path) -> Tuple[bool, str]:
    main_py = comfyui_dir / "main.py"
    if not comfyui_dir.exists():
        return False, f"ComfyUI directory missing: {comfyui_dir}"
    if not main_py.exists():
        return False, f"Missing ComfyUI entrypoint: {main_py}"
    return True, f"ComfyUI source OK at {comfyui_dir}"


def _ensure_dependency_compat() -> Tuple[bool, str]:
    """
    Keep critical deps in ranges compatible with torch 2.0.1 + transformers 4.37.
    Some custom node installers can upgrade these and break startup.
    """
    actions: List[str] = []

    def parse_major(ver: str) -> int:
        try:
            return int(ver.split(".", 1)[0])
        except Exception:
            return -1

    def get_ver(pkg: str) -> str | None:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None

    def pip_install(spec: str) -> None:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", spec],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    hub_ver = get_ver("huggingface-hub")
    if hub_ver is not None and parse_major(hub_ver) >= 1:
        pip_install("huggingface-hub>=0.19.3,<1.0")
        new_ver = get_ver("huggingface-hub")
        actions.append(f"huggingface-hub {hub_ver} -> {new_ver}")

    np_ver = get_ver("numpy")
    if np_ver is not None and parse_major(np_ver) >= 2:
        pip_install("numpy<2")
        new_ver = get_ver("numpy")
        actions.append(f"numpy {np_ver} -> {new_ver}")

    if actions:
        return True, "Adjusted incompatible deps: " + "; ".join(actions)
    return True, "Dependency compatibility OK"


def _check_patch_state() -> Tuple[bool, str]:
    try:
        from patches import apply_all
    except Exception as exc:
        return False, f"Patch manager import failed: {exc}"

    missing: List[str] = []
    try:
        import importlib

        for mod_name in apply_all.PATCH_MODULES:
            mod = importlib.import_module(mod_name)
            if not mod.check():
                missing.append(getattr(mod, "PATCH_ID", mod_name))
    except Exception as exc:
        return False, f"Patch state read failed: {exc}"

    if missing:
        return False, f"Missing source patches: {missing}"
    return True, "All source patches applied"


def run_preflight(comfyui_dir: Path) -> bool:
    """
    Runs checks and prints a concise report.
    Returns True when all critical checks pass.
    """
    _print("INFO", "Starting startup preflight checks")

    ok_all = True

    ok, msg = _check_python()
    _print("OK" if ok else "FAIL", msg)
    ok_all = ok_all and ok

    torch_ok, torch_msg, cuda_ok = _check_torch()
    _print("OK" if torch_ok else "FAIL", torch_msg)
    ok_all = ok_all and torch_ok

    smi_ok, smi_msg = _check_nvidia_smi()
    _print("OK" if smi_ok else "WARN", smi_msg)

    source_ok, source_msg = _check_comfyui_source(comfyui_dir)
    _print("OK" if source_ok else "FAIL", source_msg)
    ok_all = ok_all and source_ok

    patch_ok, patch_msg = _check_patch_state()
    _print("OK" if patch_ok else "WARN", patch_msg)

    dep_ok, dep_msg = _ensure_dependency_compat()
    _print("OK" if dep_ok else "WARN", dep_msg)

    if torch_ok and not cuda_ok:
        _print(
            "WARN",
            "CUDA not available to torch. Check Docker GPU runtime: use 'gpus: all' and verify host NVIDIA Container Toolkit.",
        )

    _print("INFO", "Preflight completed")
    return ok_all


if __name__ == "__main__":
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("./ComfyUI").resolve()
    success = run_preflight(target)
    raise SystemExit(0 if success else 1)
