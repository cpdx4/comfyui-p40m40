# syntax=docker/dockerfile:1
# ComfyUI P40/M40 Compatibility Fork
# Base: PyTorch 2.0.1 + CUDA 11.8 + cuDNN 8.7
#
# Build:  docker compose build
# Run:    docker compose up
# Shell:  docker compose exec comfyui bash

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    git-lfs \
    curl \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# PyTorch 2.0.1 + CUDA 11.8
# This is the newest PyTorch that supports Pascal (sm_61) and Maxwell (sm_52)
# under CUDA 11.8 with full FP16 compute support.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,id=pytorch-wheelhouse,target=/opt/wheelhouse,sharing=locked \
        --mount=type=cache,id=pip-cache,target=/root/.cache/pip,sharing=locked \
        bash -lc 'set -euo pipefail; \
            TORCH_WHL="/opt/wheelhouse/torch-2.0.1+cu118-cp310-cp310-linux_x86_64.whl"; \
            TV_WHL="/opt/wheelhouse/torchvision-0.15.2+cu118-cp310-cp310-linux_x86_64.whl"; \
            TA_WHL="/opt/wheelhouse/torchaudio-2.0.2+cu118-cp310-cp310-linux_x86_64.whl"; \
            if [ ! -f "$TORCH_WHL" ] || [ ! -f "$TV_WHL" ] || [ ! -f "$TA_WHL" ]; then \
                echo "[build] PyTorch wheelhouse cache miss: downloading wheels once"; \
                pip download --no-deps \
                    --dest /opt/wheelhouse \
                    --index-url https://download.pytorch.org/whl/cu118 \
                    torch==2.0.1+cu118 \
                    torchvision==0.15.2+cu118 \
                    torchaudio==2.0.2+cu118; \
            else \
                echo "[build] PyTorch wheelhouse cache hit: installing local wheels"; \
            fi; \
            pip install --no-index --no-deps --find-links=/opt/wheelhouse \
                torch==2.0.1+cu118 \
                torchvision==0.15.2+cu118 \
                torchaudio==2.0.2+cu118'

# ---------------------------------------------------------------------------
# Core ComfyUI dependencies (pinned for reproducibility)
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    accelerate==0.21.0 \
    aiohttp==3.9.3 \
    alembic==1.14.0 \
    av==14.2.0 \
    blake3==1.0.4 \
    comfyui-embedded-docs==0.4.4 \
    comfyui-frontend-package==1.43.17 \
    comfyui-workflow-templates==0.9.69 \
    einops==0.7.0 \
    filelock==3.16.1 \
    huggingface-hub==0.20.3 \
    kornia==0.7.0 \
    numpy==1.26.4 \
    Pillow==10.2.0 \
    pydantic==2.10.6 \
    pydantic-settings==2.7.1 \
    psutil==5.9.8 \
    PyYAML==6.0.1 \
    requests==2.32.3 \
    safetensors==0.4.2 \
    scipy==1.12.0 \
    sentencepiece==0.2.0 \
    simpleeval==1.0.3 \
    SQLAlchemy==2.0.36 \
    spandrel==0.3.4 \
    tokenizers==0.15.1 \
    torchsde==0.2.6 \
    tqdm==4.66.1 \
    transformers==4.37.2 \
    yarl==1.18.3

# ---------------------------------------------------------------------------
# diffusers — pinned to last version that works with PyTorch 2.0.1
# (newer diffusers assume BF16 + torch.compile which breaks on P40/M40)
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    diffusers==0.25.1 \
    invisible-watermark==0.2.0 \
    open-clip-torch==2.24.0

# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    opencv-python-headless==4.9.0.80 \
    imageio==2.33.1 \
    imageio-ffmpeg==0.4.9

# ---------------------------------------------------------------------------
# OpenGL for GLSL shader nodes (nodes_glsl.py requires glfw + PyOpenGL)
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    glfw==2.7.0 \
    PyOpenGL==3.1.7 \
    PyOpenGL-accelerate==3.1.7 \
    || echo "PyOpenGL install failed — GLSL nodes unavailable"

# ---------------------------------------------------------------------------
# Optional: bitsandbytes for INT8 quantization (partial sm_61 support)
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    bitsandbytes==0.41.3 \
    || echo "bitsandbytes install failed — INT8 quantization unavailable"

# ---------------------------------------------------------------------------
# NO xformers: xformers requires sm_70+ (Volta).
# NO triton: triton requires sm_70+ for Triton 2.x JIT.
# NO flash-attn: flash-attn requires sm_80+ (Ampere).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ComfyUI-Manager — node installation UI (bundled by default)
# Installed as a pip package; enabled with --enable-manager flag at runtime.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U --pre comfyui-manager

# ---------------------------------------------------------------------------
# Force numpy<2 — open-clip-torch / other packages may upgrade it past 1.x.
# torch 2.0.1 was compiled against NumPy 1.x and will warn/crash with 2.x.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy==1.26.4"

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
WORKDIR /app

# Copy compatibility layer and patches first (rarely changes)
COPY compat/ /app/compat/
COPY patches/ /app/patches/
COPY comfy_aimdo/ /app/comfy_aimdo/
COPY comfy_kitchen/ /app/comfy_kitchen/
COPY scripts/ /app/scripts/

# Copy the entry point
COPY main_compat.py /app/main_compat.py
COPY requirements-compat.txt /app/requirements-compat.txt

# Mount point for ComfyUI submodule / models / outputs
# In production use docker compose volumes instead of COPY
ARG COMFYUI_DIR=/app/ComfyUI
RUN mkdir -p ${COMFYUI_DIR} /app/models /app/output /app/input

# ---------------------------------------------------------------------------
# GPU environment variables
# ---------------------------------------------------------------------------
ENV NVIDIA_VISIBLE_DEVICES=all
# Disable CUDA graph capture (unstable on Pascal under CUDA 11.8)
ENV PYTORCH_NO_CUDA_MEMORY_CACHING=0
# Force FP16 for memory efficiency
ENV COMFYUI_FORCE_FP16=1
# Prevent torch from probing unavailable CUDA capabilities
ENV TORCH_CUDA_ARCH_LIST="6.1;6.0;5.2"
# Disable Triton autotune cache (Triton is blocked anyway)
ENV TRITON_CACHE_DIR=/tmp/triton_cache

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf -H "Host: comfyui" http://127.0.0.1:8188/system_stats || exit 1

EXPOSE 8188

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
CMD ["python", "/app/main_compat.py", \
     "--listen", "0.0.0.0", \
     "--port", "8188", \
     "--base-directory", "/app/ComfyUI", \
     "--output-directory", "/app/output", \
     "--input-directory", "/app/input", \
     "--enable-manager"]
