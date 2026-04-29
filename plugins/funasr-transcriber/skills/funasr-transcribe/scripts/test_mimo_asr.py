#!/usr/bin/env python3
"""Tests for mimo_asr module. All CUDA / MiMo / HF calls are mocked.

Safe for CI: does not require a GPU, MiMo weights, or network access.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import mimo_asr


def test_module_imports():
    """mimo_asr module exists and exposes expected public names."""
    assert hasattr(mimo_asr, "require_cuda_and_vram")
    assert hasattr(mimo_asr, "require_mimo_installed")
    assert hasattr(mimo_asr, "transcribe_with_mimo")
    assert hasattr(mimo_asr, "save_partial")
    assert hasattr(mimo_asr, "load_partial")


class TestRequireCudaAndVram:
    def test_no_cuda_raises_clear_error(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with pytest.raises(RuntimeError, match="CUDA"):
                mimo_asr.require_cuda_and_vram(min_gb=20)

    def test_insufficient_vram_raises_clear_error(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.name = "NVIDIA GeForce RTX 4080"
        props.total_memory = 16 * 1024**3  # 16 GB
        fake_torch.cuda.get_device_properties.return_value = props
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with pytest.raises(RuntimeError, match=r"≥20|at least 20|20 GB"):
                mimo_asr.require_cuda_and_vram(min_gb=20)

    def test_sufficient_vram_passes(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.name = "NVIDIA A100-SXM4-40GB"
        props.total_memory = 40 * 1024**3  # 40 GB
        fake_torch.cuda.get_device_properties.return_value = props
        with patch.dict(sys.modules, {"torch": fake_torch}):
            mimo_asr.require_cuda_and_vram(min_gb=20)  # must not raise


class TestRequireMimoInstalled:
    def test_missing_repo_raises(self, tmp_path):
        weights = tmp_path / "hf"
        weights.mkdir()
        repo = tmp_path / "mimo"  # does NOT exist
        with pytest.raises(RuntimeError, match=r"INSTALL_MIMO|setup_env\.sh"):
            mimo_asr.require_mimo_installed(str(weights), str(repo))

    def test_missing_weights_raises(self, tmp_path):
        weights = tmp_path / "hf"
        weights.mkdir()
        repo = tmp_path / "mimo"
        repo.mkdir()
        (repo / "src").mkdir()  # looks like a clone

        class LocalEntryNotFoundError(Exception):
            pass

        fake_hf = MagicMock()
        fake_hf.snapshot_download.side_effect = LocalEntryNotFoundError("not cached")
        fake_errs = MagicMock()
        fake_errs.LocalEntryNotFoundError = LocalEntryNotFoundError
        with patch.dict(sys.modules, {"huggingface_hub": fake_hf,
                                      "huggingface_hub.errors": fake_errs}):
            with pytest.raises(RuntimeError, match=r"MiMo weights not found"):
                mimo_asr.require_mimo_installed(str(weights), str(repo))

    def test_everything_present_passes(self, tmp_path):
        weights = tmp_path / "hf"
        weights.mkdir()
        repo = tmp_path / "mimo"
        repo.mkdir()
        (repo / "src").mkdir()

        fake_hf = MagicMock()
        fake_hf.snapshot_download.return_value = str(tmp_path / "snap")
        with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            mimo_asr.require_mimo_installed(str(weights), str(repo))  # no raise
        # Called once per repo id
        assert fake_hf.snapshot_download.call_count == 2
