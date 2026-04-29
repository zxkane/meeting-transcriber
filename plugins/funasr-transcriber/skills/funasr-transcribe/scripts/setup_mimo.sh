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

# 2. Install MiMo's Python dependencies. Upstream requirements.txt is
# incomplete — the runtime code imports einops (via internal audio modules)
# and addict (via the 3D-Speaker gender classifier path) without declaring
# them. Install both alongside the declared deps so first-run inference
# doesn't fail on ModuleNotFoundError.
if [ -f "$MIMO_REPO_DIR/requirements.txt" ]; then
    echo "[2/4] Installing MiMo requirements..."
    pip install -q -r "$MIMO_REPO_DIR/requirements.txt"
else
    echo "  WARNING: $MIMO_REPO_DIR/requirements.txt missing — skipping."
fi
echo "  Installing additional runtime deps (upstream missed these): einops, addict"
pip install -q einops addict

# 3. Install flash-attn. Prefer the pre-built wheel matching the installed
# torch + python + cxx11abi. This works on CUDA-driver-only hosts (most AWS
# GPU instances) and avoids a 10–30 min nvcc compile. Fall back to source
# build only when no matching wheel is published.
if python3 -c "import flash_attn" 2>/dev/null; then
    echo "[3/4] flash-attn already installed — skipping."
else
    FA_VER="2.7.4.post1"
    echo "[3/4] Detecting pre-built flash-attn wheel for installed torch..."
    read -r TORCH_MINOR ABI <<EOF_DETECT
$(python3 -c "import torch; v=torch.__version__.split('+')[0].rsplit('.',1)[0]; print(v, 'TRUE' if torch.compiled_with_cxx11_abi() else 'FALSE')")
EOF_DETECT
    PY_MINOR=$(python3 -c "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')")
    WHEEL_NAME="flash_attn-${FA_VER}+cu12torch${TORCH_MINOR}cxx11abi${ABI}-${PY_MINOR}-${PY_MINOR}-linux_x86_64.whl"
    WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/${WHEEL_NAME}"
    echo "  torch=${TORCH_MINOR} abi=${ABI} py=${PY_MINOR}"
    echo "  wheel: ${WHEEL_NAME}"

    if pip install --no-deps "${WHEEL_URL}" 2>&1 | tail -5; then
        echo "  Installed from pre-built wheel."
    else
        echo "  Pre-built wheel install failed — falling back to source build."
        if ! command -v nvcc &>/dev/null; then
            echo "ERROR: no matching pre-built wheel and nvcc not found."
            echo "  Either (a) ensure your torch/python combo matches a wheel at"
            echo "  https://github.com/Dao-AILab/flash-attention/releases/tag/v${FA_VER}"
            echo "  or (b) install the CUDA toolkit:"
            echo "  https://developer.nvidia.com/cuda-toolkit"
            exit 1
        fi
        echo "  Building flash-attn==${FA_VER} from source (10–30 min)..."
        pip install "flash-attn==${FA_VER}" --no-build-isolation
    fi
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
