#!/usr/bin/env bash
# scripts/setup_fork.sh
# =====================
# First-time setup for the ComfyUI P40/M40 fork.
# Run once after cloning the repository.
#
# Usage: ./scripts/setup_fork.sh [--venv] [--docker]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

USE_VENV=false
USE_DOCKER=false

for arg in "$@"; do
    case "$arg" in
        --venv)   USE_VENV=true ;;
        --docker) USE_DOCKER=true ;;
    esac
done

echo "=== ComfyUI P40/M40 Fork Setup ==="
echo "Repo root: $REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Initialize submodule
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Initializing ComfyUI submodule..."
cd "$REPO_ROOT"
git submodule update --init --recursive --depth 1
echo "  Submodule ready at: $REPO_ROOT/ComfyUI"

# ---------------------------------------------------------------------------
# 2. Set up upstream remote in the submodule
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Setting up upstream remote in ComfyUI submodule..."
COMFYUI_DIR="$REPO_ROOT/ComfyUI"
if ! git -C "$COMFYUI_DIR" remote | grep -q "^upstream$"; then
    git -C "$COMFYUI_DIR" remote add upstream \
        "https://github.com/comfyanonymous/ComfyUI.git"
    echo "  Added remote 'upstream'"
else
    echo "  Remote 'upstream' already exists"
fi

# ---------------------------------------------------------------------------
# 3. Python / venv setup
# ---------------------------------------------------------------------------
if $USE_DOCKER; then
    echo ""
    echo "[3/6] Docker mode — skipping Python venv setup."
    echo "  Run: docker compose build && docker compose up"
else
    echo ""
    echo "[3/6] Setting up Python virtual environment..."
    python3.10 -m venv "$REPO_ROOT/.venv" || python3 -m venv "$REPO_ROOT/.venv"
    source "$REPO_ROOT/.venv/bin/activate"

    echo "  Installing PyTorch 2.0.1 + CUDA 11.8..."
    pip install --quiet --upgrade pip
    pip install --quiet \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118 \
        torchaudio==2.0.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

    echo "  Installing remaining dependencies..."
    pip install --quiet -r "$REPO_ROOT/requirements-compat.txt"
    echo "  Python environment ready."
fi

# ---------------------------------------------------------------------------
# 4. Apply source patches to ComfyUI
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Applying compatibility patches to ComfyUI source..."
cd "$REPO_ROOT"
if ! $USE_DOCKER; then
    source "$REPO_ROOT/.venv/bin/activate" 2>/dev/null || true
fi
python patches/apply_all.py
echo "  Patches applied."

# ---------------------------------------------------------------------------
# 5. Create model directories
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Creating model directories..."
mkdir -p "$REPO_ROOT/models/checkpoints"
mkdir -p "$REPO_ROOT/models/vae"
mkdir -p "$REPO_ROOT/models/loras"
mkdir -p "$REPO_ROOT/models/controlnet"
mkdir -p "$REPO_ROOT/models/clip"
mkdir -p "$REPO_ROOT/models/unet"
mkdir -p "$REPO_ROOT/output"
mkdir -p "$REPO_ROOT/input"
echo "  Directories created."

# ---------------------------------------------------------------------------
# 6. Validate environment
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Running compatibility validation..."
python scripts/validate_compat.py
echo ""
echo "=== Setup complete! ==="
echo ""
if $USE_DOCKER; then
    echo "Start ComfyUI:"
    echo "  docker compose up"
else
    echo "Start ComfyUI:"
    echo "  source .venv/bin/activate"
    echo "  python main_compat.py --listen 0.0.0.0 --port 8188"
fi
