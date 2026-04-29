#!/usr/bin/env python3
"""MiMo-V2.5-ASR local inference integration for funasr-transcribe.

Runs XiaomiMiMo/MiMo-V2.5-ASR on a local CUDA GPU, reusing the existing
pipeline's FSMN VAD segmentation and CAM++ speaker clustering so the output
format matches --lang zh exactly.

Public entry points:
  - require_cuda_and_vram(min_gb): pre-flight GPU capacity check
  - require_mimo_installed(weights_path, repo_path): pre-flight install check
  - transcribe_with_mimo(audio_path, num_speakers, ...): full Phase 1 path
  - save_partial / load_partial: resume state management
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence


def require_cuda_and_vram(min_gb: int = 20) -> None:
    """Pre-flight: require a CUDA device with at least min_gb VRAM.

    Raises:
        RuntimeError: if CUDA is unavailable or VRAM is below min_gb.
    """
    import torch  # imported lazily so test can patch sys.modules
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--lang mimo requires a CUDA GPU. CUDA is not available "
            "on this machine. Use --lang zh for CPU."
        )
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    if total_gb < min_gb:
        raise RuntimeError(
            f"--lang mimo requires ≥{min_gb} GB VRAM. "
            f"Detected: {props.name} ({total_gb:.1f} GB). "
            f"MiMo-V2.5-ASR is 8B params in fp16 + tokenizer + KV cache. "
            f"Use --lang zh for low-VRAM GPUs."
        )


def require_mimo_installed(weights_path: str, repo_path: str) -> None:
    """Pre-flight: require the MiMo GitHub repo cloned and HF weights downloaded.

    Raises:
        RuntimeError: with a user-facing message pointing to the install command.
    """
    repo = Path(repo_path)
    if not (repo.is_dir() and (repo / "src").is_dir()):
        raise RuntimeError(
            f"--lang mimo requires MiMo to be installed, but the MiMo repo "
            f"was not found at {repo_path}. "
            f"Run: INSTALL_MIMO=1 bash $SCRIPTS/setup_env.sh"
        )

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    for repo_id in ("XiaomiMiMo/MiMo-V2.5-ASR",
                    "XiaomiMiMo/MiMo-Audio-Tokenizer"):
        try:
            snapshot_download(repo_id, cache_dir=weights_path,
                              local_files_only=True)
        except LocalEntryNotFoundError as e:
            raise RuntimeError(
                f"MiMo weights not found at {weights_path} "
                f"(missing: {repo_id}). "
                f"Run: INSTALL_MIMO=1 MIMO_WEIGHTS_PATH={weights_path} "
                f"bash $SCRIPTS/setup_env.sh"
            ) from e


def transcribe_with_mimo(audio_path: str,
                         num_speakers: Optional[int] = None,
                         audio_tag: str = "<chinese>",
                         batch: int = 1,
                         weights_path: Optional[str] = None,
                         resume: bool = False,
                         device: str = "cuda:0",
                         spk_model_id: str = "iic/speech_campplus_sv_zh-cn_16k-common",
                         vad_model_id: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                         repo_path: Optional[str] = None,
                         backoffs: Sequence[float] = (0.5, 2.0, 5.0)) -> list:
    """Phase 1 MiMo path: VAD -> MiMo ASR -> CAM++ speaker labels. Not implemented yet."""
    raise NotImplementedError


def save_partial(partial_path: Path, audio_hash: str, audio_tag: str,
                 weights_path: str, vad_segments: list, completed: list,
                 failed_at: dict) -> None:
    """Persist in-progress MiMo inference state. Not implemented yet."""
    raise NotImplementedError


def load_partial(partial_path: Path, audio_hash: str, audio_tag: str) -> dict:
    """Load resume state; raise on hash/tag mismatch. Not implemented yet."""
    raise NotImplementedError
