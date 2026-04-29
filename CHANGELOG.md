# Changelog

## 1.7.0 (2026-04-29)

### Added
- **`--lang mimo`:** local inference with Xiaomi's MiMo-V2.5-ASR (8B,
  GPU-only), reusing FSMN VAD + CAM++ diarization so output format matches
  `--lang zh`.
- **`scripts/setup_mimo.sh`:** opt-in installer (`INSTALL_MIMO=1 bash
  setup_env.sh`) that clones the MiMo repo, installs `flash-attn`, and
  downloads ~20 GB of weights to `$MIMO_WEIGHTS_PATH` (defaults to
  `$HF_HOME` → `~/.cache/huggingface`). Prefers pre-built `flash-attn`
  wheels matching the installed torch + Python + ABI so the install
  completes on CUDA-driver-only hosts (most AWS GPU instances) without
  requiring the CUDA toolkit (`nvcc`). Falls back to a source build only
  when no matching wheel is published.
- **`--mimo-audio-tag`, `--mimo-batch`, `--mimo-weights-path`,
  `--resume-mimo`:** CLI flags for the new preset. `--resume-mimo` picks up
  from a mid-run failure using `*_mimo_partial.json` with audio-hash
  verification.
- New tests in `scripts/test_mimo_asr.py` (mocked; GPU-free, CI-safe).

### Changed — BREAKING ENVIRONMENT CHANGE
- **`setup_env.sh` now requires Python 3.12.** Existing `.venv/`
  directories created with earlier Python versions are detected and
  rebuilt on the next run of `setup_env.sh`. Expect a 2–3 GB re-download
  of FunASR dependencies.
- New dependencies: `scikit-learn` (for KMeans clustering in the MiMo
  path), `soundfile` (already transitively present, now explicit).

### Notes
- `--lang mimo` hard-fails if CUDA is unavailable or VRAM < 20 GB. There
  is no CPU fallback for this preset.
- `--hotwords` and `--batch-size` are silently ignored with
  `--lang mimo`; use `--mimo-batch` for per-call concurrency.
