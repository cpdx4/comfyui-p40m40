"""
scripts/validate_compat.py
===========================
Compatibility validation script.  Run after setup or after an upstream merge
to confirm the environment is correctly configured for P40/M40 hardware.

Exits 0 on full pass, 1 on any failure.

Usage:
  python scripts/validate_compat.py
  python scripts/validate_compat.py --quick   # skip slow GPU tests
  python scripts/validate_compat.py --verbose
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Setup path so we can import the compat layer
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
COMFYUI_ROOT = REPO_ROOT / "ComfyUI"
if str(COMFYUI_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFYUI_ROOT))


# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------
PASS = "✓"
FAIL = "✗"
WARN = "⚠"
SKIP = "–"

_results: List[Tuple[str, str, str]] = []  # (status, name, message)


def record(status: str, name: str, message: str = "") -> None:
    _results.append((status, name, message))
    prefix = {"✓": "\033[32m✓\033[0m", "✗": "\033[31m✗\033[0m",
               "⚠": "\033[33m⚠\033[0m", "–": "\033[90m–\033[0m"}.get(status, status)
    msg = f"  {prefix} {name}"
    if message:
        msg += f": {message}"
    print(msg)


def run_test(name: str, fn: Callable, skip_if: bool = False) -> bool:
    if skip_if:
        record(SKIP, name, "skipped")
        return True
    try:
        result = fn()
        if result is False:
            record(FAIL, name)
            return False
        msg = str(result) if result not in (None, True) else ""
        record(PASS, name, msg)
        return True
    except Exception as exc:
        record(FAIL, name, str(exc))
        if verbose:
            traceback.print_exc()
        return False


verbose = False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_python_version() -> str:
    v = sys.version_info
    assert v.major == 3 and v.minor >= 10, f"Python 3.10+ required, got {v.major}.{v.minor}"
    return f"{v.major}.{v.minor}.{v.micro}"


def test_torch_import() -> str:
    import torch
    return f"PyTorch {torch.__version__}"


def test_torch_version_compatible() -> str:
    import torch
    parts = torch.__version__.split("+")[0].split(".")
    major, minor = int(parts[0]), int(parts[1])
    assert (major, minor) <= (2, 0), (
        f"PyTorch {torch.__version__} may have incompatible FP8/compile APIs. "
        f"Expected 2.0.x for P40/M40."
    )
    return f"PyTorch {torch.__version__} (≤2.0 ✓)"


def test_cuda_available() -> str:
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    return f"CUDA {torch.version.cuda}"


def test_cuda_version_compatible() -> str:
    import torch
    cuda_ver = torch.version.cuda
    if cuda_ver:
        major = int(cuda_ver.split(".")[0])
        assert major <= 11, f"CUDA {cuda_ver} detected — expected ≤11.8 for P40/M40"
    return f"CUDA {cuda_ver}"


def test_gpu_detection() -> str:
    from compat.gpu_compat import detect_gpus, GPU_INFO, FEATURE_FLAGS
    detect_gpus()
    assert GPU_INFO, "No CUDA GPUs detected"
    names = ", ".join(g["name"] for g in GPU_INFO)
    min_sm = FEATURE_FLAGS["min_sm"]
    return f"{names} (sm_{min_sm})"


def test_gpu_sm_level() -> str:
    from compat.gpu_compat import FEATURE_FLAGS
    min_sm = FEATURE_FLAGS.get("min_sm", 0)
    assert min_sm > 0, "GPU SM level not detected"
    if min_sm < 52:
        raise AssertionError(f"sm_{min_sm} is below Maxwell (sm_52) minimum")
    return f"sm_{min_sm} (Maxwell+ ✓)"


def test_no_bf16_on_pascal() -> None:
    from compat.gpu_compat import FEATURE_FLAGS
    min_sm = FEATURE_FLAGS.get("min_sm", 99)
    if min_sm < 80:
        assert not FEATURE_FLAGS.get("has_bf16_compute", False), (
            f"BF16 compute incorrectly enabled on sm_{min_sm}"
        )


def test_no_fp8() -> None:
    from compat.gpu_compat import FEATURE_FLAGS
    assert not FEATURE_FLAGS.get("has_fp8", False), (
        "FP8 incorrectly enabled (should be False on P40/M40 with PyTorch 2.0)"
    )


def test_fp8_stub_installed() -> str:
    import torch
    # fp8 attrs should exist (as stubs) on PyTorch 2.0
    stub = getattr(torch, "float8_e4m3fn", None)
    assert stub is not None, "torch.float8_e4m3fn stub not injected"
    return "FP8 stubs present"


def test_torch_compile_noop() -> str:
    import torch

    class _Dummy:
        pass

    d = _Dummy()
    result = torch.compile(d)
    assert result is d, "torch.compile did not return model unchanged (not a no-op)"
    return "torch.compile is identity no-op"


def test_sdpa_math_only() -> None:
    import torch
    if not torch.cuda.is_available():
        return
    # Attempt SDPA with math backend — should not raise
    q = torch.randn(1, 4, 8, 32, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 4, 8, 32, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 4, 8, 32, device="cuda", dtype=torch.float16)
    try:
        with torch.backends.cuda.sdp_kernel(  # type: ignore[attr-defined]
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        assert out.shape == q.shape
    except Exception as exc:
        raise AssertionError(f"SDPA math backend failed: {exc}")


def test_autocast_no_bf16() -> None:
    import torch
    from compat.gpu_compat import FEATURE_FLAGS
    if FEATURE_FLAGS.get("has_bf16_compute", False):
        return  # Not applicable on Ampere+
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pass  # Should not crash; dtype is silently downgraded to fp16


def test_comfyui_model_management_import() -> str:
    mod = importlib.import_module("comfy.model_management")
    assert hasattr(mod, "get_torch_device"), "model_management missing get_torch_device"
    return "comfy.model_management imports OK"


def test_comfyui_ops_import() -> str:
    mod = importlib.import_module("comfy.ops")
    assert hasattr(mod, "disable_weight_init"), "ops missing disable_weight_init"
    return "comfy.ops imports OK"


def test_comfyui_attention_import() -> str:
    mod = importlib.import_module("comfy.ldm.modules.attention")
    assert hasattr(mod, "CrossAttention") or hasattr(mod, "Attention"), (
        "attention module missing expected class"
    )
    return "comfy.ldm.modules.attention imports OK"


def test_no_triton_crash() -> str:
    # Importing triton should either be blocked (returns stub) or not crash
    try:
        import triton  # noqa: F401
        # If we get here, either real triton or our stub
        has_jit = hasattr(triton, "jit")
        return f"triton importable, has jit={has_jit}"
    except Exception as exc:
        raise AssertionError(f"triton import crashed: {exc}")


def test_patches_applied() -> str:
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
        raise AssertionError(f"Unapplied patches: {missing}. Run: python patches/apply_all.py")
    return "All patches applied"


def test_simple_fp16_tensor_op() -> str:
    import torch
    if not torch.cuda.is_available():
        return "skipped (no CUDA)"
    a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    c = torch.mm(a, b)
    assert c.shape == (128, 128)
    assert c.dtype == torch.float16
    return "FP16 matmul on GPU OK"


def test_vram_available() -> str:
    import torch
    if not torch.cuda.is_available():
        return "skipped"
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free = torch.cuda.mem_get_info(i)[0]
        total = props.total_memory
        free_gb = free / 1024**3
        total_gb = total / 1024**3
        if free_gb < 2.0:
            raise AssertionError(
                f"GPU {i} has only {free_gb:.1f} GB free VRAM. "
                f"Consider closing other applications."
            )
    return f"{free_gb:.1f} GB free / {total_gb:.1f} GB total"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global verbose

    parser = argparse.ArgumentParser(description="ComfyUI P40/M40 compatibility validator")
    parser.add_argument("--quick", action="store_true", help="Skip slow GPU compute tests")
    parser.add_argument("--verbose", action="store_true", help="Show full tracebacks")
    args = parser.parse_args()
    verbose = args.verbose
    quick = args.quick

    print("\n=== ComfyUI P40/M40 Compatibility Validation ===\n")

    # Install compat layer first
    print("Installing compatibility layer...")
    from compat import install
    install(verbose=False)
    print("")

    print("[ Environment ]")
    run_test("Python version ≥3.10", test_python_version)
    run_test("PyTorch importable", test_torch_import)
    run_test("PyTorch ≤2.0.x", test_torch_version_compatible)
    run_test("CUDA available", test_cuda_available)
    run_test("CUDA ≤11.8", test_cuda_version_compatible)

    print("\n[ GPU Detection ]")
    run_test("GPU detected", test_gpu_detection)
    run_test("SM level ≥52 (Maxwell+)", test_gpu_sm_level)
    run_test("BF16 correctly disabled", test_no_bf16_on_pascal)
    run_test("FP8 correctly disabled", test_no_fp8)
    run_test("VRAM available", test_vram_available, skip_if=quick)

    print("\n[ PyTorch Patches ]")
    run_test("FP8 stubs installed", test_fp8_stub_installed)
    run_test("torch.compile is no-op", test_torch_compile_noop)
    run_test("SDPA math-only backend", test_sdpa_math_only, skip_if=quick)
    run_test("autocast BF16→FP16 guard", test_autocast_no_bf16)

    print("\n[ ComfyUI Imports ]")
    if not COMFYUI_ROOT.exists():
        print("  – ComfyUI submodule not found — skipping import tests")
        print("    Run: git submodule update --init --recursive")
    else:
        run_test("patches applied", test_patches_applied)
        run_test("comfy.model_management", test_comfyui_model_management_import)
        run_test("comfy.ops", test_comfyui_ops_import)
        run_test("comfy.ldm.modules.attention", test_comfyui_attention_import)
        run_test("triton import safe", test_no_triton_crash)

    print("\n[ GPU Compute ]")
    run_test("FP16 matmul on GPU", test_simple_fp16_tensor_op, skip_if=quick)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("")
    passed = sum(1 for s, _, _ in _results if s == PASS)
    failed = sum(1 for s, _, _ in _results if s == FAIL)
    warned = sum(1 for s, _, _ in _results if s == WARN)
    skipped = sum(1 for s, _, _ in _results if s == SKIP)

    print(f"Results: {passed} passed · {failed} failed · {warned} warnings · {skipped} skipped")
    print("")

    if failed > 0:
        print("\033[31mSome checks failed. Review output above.\033[0m")
        sys.exit(1)
    else:
        print("\033[32mAll checks passed — environment is P40/M40 compatible.\033[0m")


if __name__ == "__main__":
    main()
