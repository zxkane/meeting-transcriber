# Changelog

## 1.7.1 (2026-05-01)

### Rebranding — plugin and skill renamed to reflect multi-engine support

Since 1.7.0 added MiMo-V2.5-ASR alongside FunASR, the plugin is no longer
FunASR-only. This release renames repo, plugin, skill, and script paths
accordingly. **This affects users who invoke the CLI by path or import
the script as a Python module; does not affect users on ClawHub or
skills.sh.**

Renames:
- GitHub repo: `zxkane/audio-transcriber-funasr` → `zxkane/audio-transcriber`
  (old URLs continue to work via GitHub's automatic redirect)
- Plugin dir: `plugins/funasr-transcriber/` → `plugins/audio-transcriber/`
- Skill dir: `skills/funasr-transcribe/` → `skills/audio-transcribe/`
- Skill name (SKILL.md `name:`): `funasr-transcribe` → `audio-transcribe`
- Main CLI script: `scripts/transcribe_funasr.py` → `scripts/transcribe.py`
- `.claude/skills/funasr-transcribe` symlink → `.claude/skills/audio-transcribe`

Not renamed (intentional, backward compat):
- **ClawHub package slug `zxkane-audio-transcriber-funasr` is preserved** —
  existing installs continue to work, no migration needed for
  `/plugin install audio-transcriber@zxkane-audio-transcriber-funasr`.
- Internal function `transcribe_with_funasr()` keeps its name (it is
  specifically the FunASR-backed path, so the name is accurate).

### Migration for direct-invocation users

If you were invoking the script by path, update your command:

```diff
- python3 plugins/funasr-transcriber/skills/funasr-transcribe/scripts/transcribe_funasr.py
+ python3 plugins/audio-transcriber/skills/audio-transcribe/scripts/transcribe.py
```

If you were importing as a Python module (rare, mostly test-only):

```diff
- import transcribe_funasr
+ import transcribe
```

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
  when no matching wheel is published. **The MiMo GitHub repo is pinned
  to a known-good commit** (validated against our `MimoAudio.__init__`
  kwarg contract and runtime deps); override with
  `MIMO_PINNED_COMMIT=<sha>` to trial a newer upstream.
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
