#!/usr/bin/env bash
# Install MiMo-V2.5-ASR: clone the reference repo, install flash-attn, and
# download HF weights. Called automatically by setup_env.sh when
# INSTALL_MIMO=1, or can be run standalone.
#
# Environment:
#   VENV_DIR             — path to the active venv (default: .venv)
#   MIMO_WEIGHTS_PATH    — cache dir for MiMo weights (default: $HF_HOME
#                          or ~/.cache/huggingface)

set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
MIMO_WEIGHTS_PATH="${MIMO_WEIGHTS_PATH:-${HF_HOME:-$HOME/.cache/huggingface}}"
MIMO_REPO_URL="https://github.com/XiaomiMiMo/MiMo-V2.5-ASR.git"
MIMO_REPO_DIR="$VENV_DIR/mimo"

echo "=== MiMo-V2.5-ASR Install ==="
echo "  venv:     $VENV_DIR"
echo "  weights:  $MIMO_WEIGHTS_PATH"
echo ""

# Require active venv
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR. Run setup_env.sh first."
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 1. Clone MiMo repo (idempotent)
if [ ! -d "$MIMO_REPO_DIR/.git" ]; then
    echo "[1/4] Cloning $MIMO_REPO_URL into $MIMO_REPO_DIR..."
    git clone --depth 1 "$MIMO_REPO_URL" "$MIMO_REPO_DIR"
else
    echo "[1/4] MiMo repo already present at $MIMO_REPO_DIR — skipping clone."
fi

# 2. Install MiMo's Python dependencies
if [ -f "$MIMO_REPO_DIR/requirements.txt" ]; then
    echo "[2/4] Installing MiMo requirements..."
    pip install -q -r "$MIMO_REPO_DIR/requirements.txt"
else
    echo "  WARNING: $MIMO_REPO_DIR/requirements.txt missing — skipping."
fi

# 3. Install flash-attn (requires nvcc; compile can take 10–30 min)
if python3 -c "import flash_attn" 2>/dev/null; then
    echo "[3/4] flash-attn already installed — skipping."
else
    if ! command -v nvcc &>/dev/null; then
        echo "ERROR: nvcc not found."
        echo "  flash-attn requires the CUDA toolkit (not just the driver)."
        echo "  Install: https://developer.nvidia.com/cuda-toolkit"
        exit 1
    fi
    echo "[3/4] Installing flash-attn==2.7.4.post1 (this can take 10–30 min)..."
    pip install flash-attn==2.7.4.post1 --no-build-isolation
fi

# 4. Download weights (idempotent — huggingface-cli skips cached files)
echo "[4/4] Downloading MiMo weights to $MIMO_WEIGHTS_PATH..."
mkdir -p "$MIMO_WEIGHTS_PATH"
HF_HOME="$MIMO_WEIGHTS_PATH" \
    python3 -m huggingface_hub.commands.huggingface_cli download \
        XiaomiMiMo/MiMo-V2.5-ASR
HF_HOME="$MIMO_WEIGHTS_PATH" \
    python3 -m huggingface_hub.commands.huggingface_cli download \
        XiaomiMiMo/MiMo-Audio-Tokenizer

echo ""
echo "=== MiMo install complete ==="
echo "  Use with: python3 transcribe_funasr.py <audio> --lang mimo \\"
echo "               --mimo-weights-path $MIMO_WEIGHTS_PATH"
