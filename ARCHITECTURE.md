# Fork Architecture & Patch Plan

## 1. Design Principles

### Upstream Tracking Strategy
The fork uses ComfyUI as a **git submodule** pinned to `origin/master`.  
All compatibility work lives **outside** the submodule so upstream diffs stay clean.

Three mechanisms are used in layers:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3 – main_compat.py                               │
│  Entry point. Installs compat/ shims before any import  │
├─────────────────────────────────────────────────────────┤
│  Layer 2 – compat/ (runtime monkey-patches)             │
│  Patches torch.* and ComfyUI internals in memory.       │
│  Zero source modifications, survives upstream updates.  │
├─────────────────────────────────────────────────────────┤
│  Layer 1 – patches/ (source-level idempotent patches)   │
│  Applied once after git submodule update.               │
│  Covers cases where monkey-patching is insufficient.    │
└─────────────────────────────────────────────────────────┘
```

The goal is to keep Layer 1 (source patches) as small as possible so that
`git submodule update` + `python patches/apply_all.py` re-establishes a
working state after any upstream bump.

---

## 2. GPU Capability Matrix

| Feature | sm_52 (M40) | sm_61 (P40) | sm_70 (V100) | sm_80 (A100) |
|---------|-------------|-------------|--------------|--------------|
| FP32 | ✅ | ✅ | ✅ | ✅ |
| FP16 (storage) | ✅ | ✅ | ✅ | ✅ |
| FP16 (tensor cores) | ❌ | ❌ | ✅ | ✅ |
| BF16 | ❌ | ❌ | ❌ | ✅ |
| FP8 | ❌ | ❌ | ❌ | ❌* |
| INT8 (bitsandbytes) | ⚠️ partial | ⚠️ partial | ✅ | ✅ |
| SDPA (math) | ✅ | ✅ | ✅ | ✅ |
| SDPA (mem-efficient) | ❌ | ❌ | ✅ | ✅ |
| Flash Attention 2 | ❌ | ❌ | ❌ | ✅ |
| Triton | ❌ | ❌ | ✅ | ✅ |
| torch.compile | ❌† | ❌† | ✅ | ✅ |
| xformers | ❌ | ❌ | ✅ | ✅ |

*FP8 requires sm_89 (Ada Lovelace) or sm_90 (Hopper) †Compile works for eager, but Triton backend needs sm_70+; disabled to avoid silent perf regression

---

## 3. ComfyUI Files That Require Patching

### 3.1 `comfy/model_management.py`

**Issues:**
1. **FP8 dtype references** — `torch.float8_e4m3fn`, `torch.float8_e5m2` are PyTorch 2.1+ only.  
   Present in `unet_dtype()`, `get_computation_dtype()`, and `ModelPatcher`.
2. **BF16 on Pascal/Maxwell** — `torch.bfloat16` as storage works but compute ops silently fall
   back to FP32, causing correctness and OOM issues. Must be disabled for sm_52/sm_61.
3. **`torch.compile` wrapping** — Models are compiled at load time in some paths. On CUDA 11.8
   with sm_61 this triggers a Triton attempt that crashes.
4. **SDPA context manager API** — PyTorch 2.0 uses `torch.backends.cuda.sdp_kernel(...)` as a
   context manager; PyTorch 2.1 deprecated it for `torch.nn.attention.sdpa_kernel(...)`. The
   upstream code may mix these.
5. **`is_device_mps()`** — safe, no patch needed.
6. **`get_autocast_device()`** — safe for CUDA path.

**Patch strategy:** Runtime monkey-patch via `compat/torch_compat.py` + `compat/gpu_compat.py`.
Source patch only for FP8 dtype guard (cannot be done safely at runtime).

### 3.2 `comfy/ops.py`

**Issues:**
1. **`CastWeightsTo` with FP8** — `disable_weight_init.Linear` casts weights to
   `torch.float8_e4m3fn` when the model dtype is FP8. Crashes on 2.0.1.
2. **FP8 `manual_cast` path** — `manual_cast_weight` and related functions reference FP8 dtypes.
3. **`torch.compile` decorator** — Some op functions are decorated.

**Patch strategy:** Source patch — replace FP8 branches with FP16 fallback.

### 3.3 `comfy/ldm/modules/attention.py`

**Issues:**
1. **Triton import** — Top-level `import triton` or `from triton import ...` will crash because
   Triton 2.x requires sm_70+. Even if triton is not installed, ComfyUI may do a capability check
   that accesses unavailable CUDA functions.
2. **`optimized_attention`** — Selects between flash, mem-efficient, math backends. On P40/M40
   only `enable_math=True` is valid. The selection logic must be overridden.
3. **`xformers` import guard** — xformers for CUDA 11.8 doesn't support Pascal/Maxwell. Must be
   unconditionally disabled.
4. **`F.scaled_dot_product_attention` with `scale` kwarg** — Added in PyTorch 2.1; not present
   in 2.0. Code must use the 2.0 signature.

**Patch strategy:** Runtime monkey-patch + source patch for Triton import guard.

### 3.4 `comfy/latent_formats.py`

**Issues:**
1. May define FP8 as a latent storage format for certain models (Flux NF4/FP8 variants).

**Patch strategy:** Source patch — gate FP8 format checks behind `HAS_FP8` flag.

### 3.5 `comfy/supported_models.py` / `comfy/supported_models_base.py`

**Issues:**
1. FP8 model configs — e.g., `Flux.unet_config` may set `dtype=torch.float8_e4m3fn`.
2. Models that hard-require BF16.

**Patch strategy:** Source patch — add dtype downgrade logic in `ModelPatcher.get_model_object`.

### 3.6 `comfy/k_diffusion/sampling.py`

**Issues:**
1. Some samplers use `torch.compile` for inner loops.
2. Custom CUDA extensions for certain samplers.

**Patch strategy:** Runtime monkey-patch `torch.compile` to identity.

### 3.7 `nodes.py` (root-level)

**Issues:**
1. `FP8Linear` / `CheckpointLoaderFP8` nodes reference FP8 dtypes.
2. Node registry may import from `comfy_kitchen`.

**Patch strategy:** Source patch — replace FP8 node class bodies with deprecation notices that
fall back gracefully.

### 3.8 `comfy/diffusers_convert.py` / `comfy/model_detection.py`

**Issues:**
1. Newer diffusers model loading uses `torch.bfloat16` as default. Must override to FP16 on
   Pascal/Maxwell.
2. Some model detection heuristics reference FP8 weight shapes.

**Patch strategy:** Runtime monkey-patch `diffusers` default dtype setting.

### 3.9 `comfy_extras/nodes_flux.py` (if present)

**Issues:**
1. Flux models with FP8 quantization paths.
2. May import from `comfy_kitchen`.

**Patch strategy:** Source patch — strip FP8 quantization, keep FP16 path.

---

## 4. Compatibility Layer Architecture

```
compat/
├── __init__.py               ← install() entry point called from main_compat.py
├── gpu_compat.py             ← GPU detection; exports FEATURE_FLAGS dict
├── torch_compat.py           ← Stubs for fp8 dtypes, compile noop, sdpa patches,
│                                 RMSNorm backfill (torch.nn.RMSNorm +
│                                 torch.nn.functional.rms_norm; added in PyTorch 2.1,
│                                 missing in 2.0.1)
├── attention_compat.py       ← Replaces optimized_attention with safe fallback
├── aiohttp_compat.py         ← Patches aiohttp.web_request.URL.build() to accept
│                                 "host:port" Host headers (aiohttp 3.9 / yarl 1.9+ bug)
└── fp8_stub.py               ← Dummy dtype objects that behave like torch dtypes
```

### Stub packages

```
comfy_aimdo/            ← Replaces the comfy-aimdo pip package (requires PyTorch 2.8+).
├── __init__.py         ← Prints warning; disables DynamicVRAM; legacy ModelPatcher used.
├── control.py          ← init_device, get_total_vram_usage, analyze, set_log_* stubs.
├── vram_buffer.py      ← VRAMBuffer(size, device_index) no-op.
├── model_vbar.py       ← ModelVBAR + vbar_fault/signature_compare/unpin/analyze stubs.
├── torch.py            ← aimdo_to_tensor / hostbuf_to_tensor no-ops.
├── host_buffer.py      ← HostBuffer no-op.
└── model_mmap.py       ← ModelMMap.get() returns 0.

comfy_kitchen/          ← Replaces the comfy-kitchen pip package.
├── __init__.py         ← registry stub with disable()/list_backends().
└── tensor.py           ← Raises ImportError → quant_ops.py activates its no-kitchen path.
```

### Execution Order in `main_compat.py`

```python
# 1. Detect GPUs, build feature flags
from compat import install
install()   # Must happen before any ComfyUI import

# 2. Apply source patches if missing (idempotent)
from patches import apply_all
apply_all.run_patches("apply")

# 3. Run ComfyUI as __main__ so its startup block executes
import runpy
runpy.run_path(comfyui_dir / "main.py", run_name="__main__")
```

> **Note:** `runpy.run_path(..., run_name="__main__")` is required because
> ComfyUI's server startup lives inside `if __name__ == "__main__":`.  Using
> `importlib.util.exec_module` silently skips that block.

### `install()` call graph

```
install()
  ├── gpu_compat.detect_gpus()
  ├── torch_compat.patch_fp8_dtypes()      # inject dummy fp8 dtype attrs
  ├── torch_compat.patch_torch_compile()   # replace with identity
  ├── torch_compat.patch_sdpa()            # force math-only backend
  ├── torch_compat.patch_autocast()        # force fp16 instead of bf16
  ├── torch_compat.patch_misc()            # uint16/32/64 dtype aliases
  ├── torch_compat.patch_serialization()   # add_safe_globals shim (2.4 API → 2.0)
  ├── torch_compat.patch_rmsnorm()         # backfill torch.nn.RMSNorm (missing in 2.0)
  ├── attention_compat.register_import_hook()  # patch attention on first import
  └── aiohttp_compat.patch_aiohttp_host()  # fix host:port in URL.build()
```

---

## 5. Patch Script Design

Each `patches/patch_*.py` script:
1. Is **idempotent** — detects if the patch is already applied and skips.
2. Writes a backup `*.orig` before modifying.
3. Validates by importing the patched module and running a smoke test.
4. Records a SHA256 of the original file in `patches/.patch_state.json`.

```bash
python patches/apply_all.py          # apply all patches
python patches/apply_all.py --check  # verify state without modifying
python patches/apply_all.py --revert # restore .orig backups
```

---

## 6. Upstream Merge Safety

### Commit Triage Categories

| Category | Action | Detection Pattern |
|----------|--------|-------------------|
| UI / frontend | Auto-accept | `web/`, `*.js`, `*.css`, `*.ts` |
| New nodes (non-FP8) | Auto-accept | `nodes_*.py` without FP8 keywords |
| FP8 / FP4 additions | Skip commit | `float8`, `float4`, `fp8`, `fp4` |
| Triton kernels | Skip commit | `import triton`, `@triton.jit` |
| torch.compile additions | Patch | `torch.compile(` |
| Flash attention additions | Patch | `flash_attn`, `enable_flash=True` |
| CUDA 12.x kernels | Skip commit | `sm_90`, `sm_89`, `cuda_12` |
| comfy_kitchen | Skip commit | `comfy_kitchen` |
| BF16-only model configs | Patch | `bfloat16` as sole dtype option |
| SDPA API 2.1 | Patch | `sdpa_kernel(` |

### Merge Script Flow

```
upstream_merge.sh
  1. git fetch upstream
  2. For each new commit on upstream/master:
     a. Run commit_triage.py → category
     b. If SKIP: record in skip_log.txt, continue
     c. If AUTO_ACCEPT: git cherry-pick
     d. If PATCH: cherry-pick + run relevant patch_*.py
     e. If CONFLICT: pause, open diff for human review
  3. Run patches/apply_all.py --check
  4. Run scripts/validate_compat.py
  5. Report summary
```

---

## 7. Long-Term Maintenance Checklist

### Monthly
- [ ] Run `./scripts/upstream_merge.sh --dry-run` to see pending commits
- [ ] Review `skip_log.txt` for any skipped commits that might now be patchable
- [ ] Run `python scripts/validate_compat.py` to verify all checks still pass

### On upstream minor release (e.g. ComfyUI 0.4.x → 0.5.x)
- [ ] Review `CHANGELOG.md` in upstream for new GPU-specific features
- [ ] Run `patches/apply_all.py --check` to see which patches still apply cleanly
- [ ] Update `patches/.patch_state.json` with new file SHAs
- [ ] Update `requirements-compat.txt` if new optional deps were added upstream

### On PyTorch upgrade (when 2.0.1 → 2.1+ becomes viable for P40)
- [ ] Check CUDA 11.8 → CUDA 12.x driver support on your system
- [ ] Rerun `scripts/validate_compat.py` with new torch version
- [ ] Gradually remove compat shims that are no longer needed
- [ ] The FP8 stubs can be retired once `torch.float8_e4m3fn` is native in your torch

---

## 8. Testing Strategy

```
scripts/validate_compat.py runs:
  ✓ GPU detection and sm_ version check
  ✓ FP8 dtype stub smoke test
  ✓ torch.compile noop verification
  ✓ SDPA math-only mode verification
  ✓ BF16 compute guard check
  ✓ ComfyUI model_management import
  ✓ ComfyUI ops import
  ✓ ComfyUI attention import (no triton crash)
  ✓ Simple SD1.5 generation (if model present)
  ✓ VRAM usage within expected bounds
```

---

## 9. FP8 Fallback Behavior

When a model requests FP8 (e.g., Flux FP8 checkpoint):

```
Requested dtype:  torch.float8_e4m3fn
                        │
              compat/fp8_stub.py detects
                        │
              Downgrades to torch.float16
                        │
              Logs: "FP8 not available on sm_61, using FP16"
                        │
              Model loads in FP16 (24 GB VRAM on P40 is sufficient)
```

VRAM budget for Flux.1 FP16 on P40 (24 GB):
- UNet weights: ~16 GB
- VAE: ~1 GB  
- CLIP/T5: ~2 GB
- Activations: ~3 GB
- **Total: ~22 GB — fits on P40**
