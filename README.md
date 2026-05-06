# ComfyUI — P40/M40 Compatibility Fork

A maintainable fork of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that tracks upstream's latest UI, workflow engine, and node definitions while removing or patching every code path that requires PyTorch 2.1+, CUDA 12.x, FP8/FP4, Triton 2.x, Flash Attention, or `comfy_kitchen`.

## Target Hardware

| GPU | Architecture | Compute Capability | VRAM |
|-----|-------------|-------------------|------|
| NVIDIA Tesla P40 | Pascal | sm_61 | 24 GB |
| NVIDIA Tesla M40 | Maxwell | sm_52 | 24 GB |

**Software stack:** CUDA 11.8 · cuDNN 8.7 · PyTorch 2.0.1 · Python 3.10

## What This Fork Does

### Keeps (from upstream)
- All modern ComfyUI frontend (React/LiteGraph UI, workflow metadata v2+)
- All new nodes that do not require FP8 dtypes
- All new sampler and scheduler definitions
- All new model loader architecture (diffusers-style, safetensors)
- Workflow API v2 and prompt queue improvements

### Removes / Patches
| Feature | Why Removed | Replacement |
|---------|-------------|-------------|
| `torch.float8_e4m3fn` / `torch.float8_e5m2` | PyTorch 2.1+ only | Stubbed → falls back to FP16 |
| FP8 linear / matmul paths | Requires sm_89 (Ada Lovelace) | FP16 path used instead |
| `torch.compile()` | Unstable on CUDA 11.8, needs sm_70+ for Triton backend | Disabled (identity pass-through) |
| Triton 2.x attention kernels | Requires sm_70+ | Software SDPA fallback |
| Flash Attention 2 | Requires sm_80+ (Ampere) | Math SDPA fallback |
| SDPA fused kernels | Flash/mem-efficient require Ampere | `enable_math=True` only |
| BF16 compute | No hardware BF16 on Pascal/Maxwell | FP16 used instead |
| `comfy_kitchen` | Third-party dependency, not open | Stubbed with noop |
| CUDA 12.x-only kernels | Driver incompatibility | Removed |
| xformers | Builds require sm_70+ | Replaced with manual attention |

## Quick Start (Docker — recommended)

```bash
git clone https://github.com/YOUR_ORG/comfyui-p40m40.git
cd comfyui-p40m40

# First time: build image
docker compose build

# Run with GPU(s)
docker compose up

# Open browser
open http://localhost:8188
```

## Quick Start (bare-metal)

```bash
git clone --recurse-submodules https://github.com/YOUR_ORG/comfyui-p40m40.git
cd comfyui-p40m40

# Create venv
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install compatible dependencies
pip install -r requirements-compat.txt

# Apply source patches to ComfyUI submodule
python patches/apply_all.py

# Validate environment
python scripts/validate_compat.py

# Launch
python main_compat.py --listen 0.0.0.0 --port 8188
```

## Repository Layout

```
comfyui-p40m40/
├── ComfyUI/                  ← upstream as git submodule (tracked, unmodified)
├── compat/                   ← compatibility layer (injected at runtime)
│   ├── __init__.py
│   ├── gpu_compat.py         ← GPU detection & feature-flag registry
│   ├── torch_compat.py       ← PyTorch version shims (fp8 stubs, compile noop)
│   ├── attention_compat.py   ← attention mechanism patches
│   └── fp8_stub.py           ← FP8 dtype + quantization stubs
├── patches/                  ← idempotent Python patch scripts
│   ├── apply_all.py          ← master patch runner
│   ├── patch_model_management.py
│   ├── patch_attention.py
│   ├── patch_ops.py
│   └── patch_nodes.py
├── scripts/
│   ├── upstream_merge.sh     ← safe upstream merge with conflict guard
│   ├── setup_fork.sh         ← first-time setup
│   └── validate_compat.py    ← environment + GPU sanity checks
├── .github/workflows/
│   └── upstream_check.yml    ← weekly PR from upstream main
├── main_compat.py            ← drop-in replacement for ComfyUI/main.py
├── Dockerfile
├── docker-compose.yml
└── requirements-compat.txt
```

## Merging Upstream Changes

```bash
# Fetch and merge upstream, skip FP8/Triton/compile commits automatically
./scripts/upstream_merge.sh

# Or review first
./scripts/upstream_merge.sh --dry-run
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full patch plan and long-term maintenance guide.

## Supported Models

| Model | Works | Notes |
|-------|-------|-------|
| SD 1.5 | ✅ | Full support |
| SD 2.x | ✅ | Full support |
| SDXL | ✅ | Runs in FP16 on 24 GB VRAM |
| Flux.1 | ✅ | FP16 only (no FP8 quant) |
| Qwen-VL / Qwen-Image-Edit | ✅ | Patched diffusers loader |
| ControlNet | ✅ | Full support |
| IP-Adapter | ✅ | Full support |
| AnimateDiff | ✅ | Full support |
| LCM / Lightning | ✅ | Full support |

## Known Limitations

- No FP8 quantization (requires Ada Lovelace / PyTorch 2.1)
- No BF16 compute (P40/M40 lack native BF16 tensor cores)
- No Triton-accelerated ops (requires sm_70+)
- No Flash Attention (requires Ampere sm_80+)
- Generation speed ~15–25% slower than RTX 3090 at equivalent model size due to no flash attn

## License

This fork inherits ComfyUI's GPL-3.0 license. Compatibility layer code (the `compat/` directory) is also GPL-3.0.
