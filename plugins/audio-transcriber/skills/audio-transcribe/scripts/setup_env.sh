#!/usr/bin/env bash
# Set up Python venv with FunASR and dependencies.
# Detects GPU/CUDA automatically; falls back to CPU-only PyTorch.
#
# Requires Python 3.12 (needed by the --lang mimo preset even if INSTALL_MIMO
# is not set, to keep dependency resolution consistent across presets).
#
# Usage:
#   bash setup_env.sh            # auto-detect CUDA
#   bash setup_env.sh cpu        # force CPU-only
#   bash setup_env.sh cu121      # force CUDA 12.1
#
#   INSTALL_MIMO=1 bash setup_env.sh
#       After the base install, also run setup_mimo.sh to clone the MiMo repo,
#       install flash-attn, and download MiMo weights. Opt-in because the
#       download is ~20 GB and flash-attn compile takes 10–30 min.

set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
FORCE_VARIANT="${1:-auto}"
AUTO_YES="${AUTO_YES:-}"
INSTALL_MIMO="${INSTALL_MIMO:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Audio Transcriber Environment Setup ==="
echo ""
echo "This script will:"
echo "  - Install ffmpeg (system package) if not present"
echo "  - Require Python 3.12; rebuild $VENV_DIR if it was made with another version"
echo "  - Install PyTorch, FunASR, modelscope, boto3 into the venv"
echo "  - Patch FunASR's clustering for long-audio performance"
if [ -n "$INSTALL_MIMO" ]; then
    echo "  - INSTALL_MIMO=1 set: also clone MiMo repo + flash-attn + download weights"
fi
echo ""

if [ -z "$AUTO_YES" ]; then
    if [ ! -t 0 ]; then
        echo "Error: Running non-interactively without AUTO_YES=1."
        echo "  Set AUTO_YES=1 to skip the confirmation prompt."
        exit 1
    fi
    read -rp "Proceed? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy] ]]; then
        echo "Aborted."
        exit 2
    fi
fi

# Ensure ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "Installing ffmpeg..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
    elif command -v brew &>/dev/null; then
        brew install ffmpeg
    else
        echo "Error: ffmpeg not found. Install it manually."
        exit 1
    fi
fi

# Require Python 3.12
if ! command -v python3.12 &>/dev/null; then
    echo "ERROR: python3.12 is required but not found."
    echo ""
    echo "Install instructions:"
    echo "  Ubuntu 24.04:  sudo apt install python3.12 python3.12-venv"
    echo "  Ubuntu 22.04:  add deadsnakes PPA, then:"
    echo "                 sudo apt install python3.12 python3.12-venv"
    echo "                 https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa"
    echo "  macOS:         brew install python@3.12"
    echo "  Other:         https://www.python.org/downloads/"
    exit 1
fi
PY=python3.12

# Rebuild venv if it exists but isn't 3.12
if [ -d "$VENV_DIR" ]; then
    EXISTING_PY="$VENV_DIR/bin/python3"
    if [ -x "$EXISTING_PY" ]; then
        EXISTING_VER=$("$EXISTING_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")
        if [ "$EXISTING_VER" != "3.12" ]; then
            echo "Existing venv is Python $EXISTING_VER, rebuilding for 3.12..."
            rm -rf "$VENV_DIR"
        fi
    else
        echo "Existing venv looks broken (no python3), rebuilding..."
        rm -rf "$VENV_DIR"
    fi
fi

# Create venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv: $VENV_DIR (Python 3.12)"
    "$PY" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Detect CUDA
if [ "$FORCE_VARIANT" = "auto" ]; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
        MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        if [ "$MAJOR" -ge 12 ] 2>/dev/null; then
            FORCE_VARIANT="cu121"
        elif [ "$MAJOR" -ge 11 ] 2>/dev/null; then
            FORCE_VARIANT="cu118"
        else
            FORCE_VARIANT="cpu"
        fi
        echo "Detected CUDA $CUDA_VER -> $FORCE_VARIANT"
    else
        FORCE_VARIANT="cpu"
        echo "No NVIDIA GPU detected -> CPU mode"
    fi
fi

# Install PyTorch
echo "Installing PyTorch ($FORCE_VARIANT)..."
if [ "$FORCE_VARIANT" = "cpu" ]; then
    pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    pip install -q torch torchaudio --index-url "https://download.pytorch.org/whl/$FORCE_VARIANT"
fi

# Install FunASR + deps (scikit-learn is new: MiMo path uses KMeans)
echo "Installing FunASR and dependencies..."
pip install -q -U funasr modelscope boto3 scikit-learn soundfile

# Patch clustering for long audio
if [ -f "$SCRIPT_DIR/patch_clustering.py" ]; then
    echo "Applying clustering optimization patch..."
    if ! python3 "$SCRIPT_DIR/patch_clustering.py" --yes; then
        echo "WARNING: Clustering patch failed. Long recordings (>1h) may be very slow."
        echo "  You can retry manually: python3 $SCRIPT_DIR/patch_clustering.py --yes"
    fi
else
    echo "WARNING: patch_clustering.py not found at $SCRIPT_DIR"
    echo "  Long-audio clustering optimization will not be applied."
fi

# Optional: install MiMo
if [ -n "$INSTALL_MIMO" ]; then
    echo ""
    echo "=== INSTALL_MIMO=1 detected — invoking setup_mimo.sh ==="
    bash "$SCRIPT_DIR/setup_mimo.sh"
fi

echo ""
echo "=== Setup complete ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
import funasr
print(f'FunASR {funasr.__version__}')
"
echo ""
echo "Activate with: source $VENV_DIR/bin/activate"
echo ""
echo "Optional LLM provider SDKs (for Phase 3 cleanup):"
echo "  pip install anthropic   # for --model claude-*"
echo "  pip install openai      # for --model gpt-* / deepseek-* / vLLM / Ollama"
echo "  (boto3 is already installed for AWS Bedrock, the default provider)"
