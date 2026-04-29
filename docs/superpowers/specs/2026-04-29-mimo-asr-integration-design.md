# MiMo-V2.5-ASR Integration — Design

**Date:** 2026-04-29
**Status:** Draft — awaiting user review

## Problem

The `funasr-transcribe` skill currently supports five ASR presets
(`zh`, `zh-basic`, `en`, `auto`, `whisper`), all built on FunASR's `AutoModel`
interface. Users have asked for support for Xiaomi's
[`XiaomiMiMo/MiMo-V2.5-ASR`](https://huggingface.co/spaces/XiaomiMiMo/MiMo-V2.5-ASR) —
an 8B LLM-based ASR model claimed to match or beat Paraformer / SenseVoice /
Whisper on Chinese, English, Chinese-English code-switching, dialects (Wu,
Cantonese, Hokkien, Sichuanese), lyrics, and proper nouns.

This design adds a `--lang mimo` preset that runs the model **locally on the
user's own GPU**, integrated with the existing pipeline's speaker diarization
and LLM cleanup stages.

## Goals

- Add `--lang mimo` as a new preset alongside existing presets.
- Run MiMo inference entirely locally (no HuggingFace Space dependency at
  inference time).
- Reuse the existing FSMN VAD and CAM++ speaker clustering so output format
  matches `--lang zh` exactly (per-sentence timestamps + speaker labels).
- Support 4+ hour audio files with clean failure handling and resume.
- Fail fast with clear errors when GPU / VRAM / weights prerequisites are
  missing, rather than silently degrading.

## Non-Goals

- Calling the MiMo HuggingFace Space (rejected: cannot handle 4h+ files due to
  HTTP timeouts, shared queue of 4, and IP rate limits).
- Running MiMo on CPU (rejected: 8B params, unworkable without a GPU).
- MiMo with VRAM < 20 GB (rejected: will OOM; cleaner to pre-check and abort).
- Custom hotword support for MiMo (MiMo does not expose a hotword API).
- Streaming / real-time transcription.

## Research Findings

Summary of MiMo-V2.5-ASR characteristics driving the design (full research
in conversation):

| Property | Value | Design implication |
|---|---|---|
| Runtime | HuggingFace Transformers, not FunASR `AutoModel` | Needs a new inference path, cannot be a one-line `MODEL_PRESETS` entry |
| Python | **3.12 required** | Must upgrade `setup_env.sh` (breaking change) |
| CUDA | ≥12.0 | Document; predicate check at runtime |
| `flash-attn` | 2.7.4.post1 (requires `nvcc`, compile takes 10–30 min) | Isolate in optional `setup_mimo.sh` |
| Weights | `XiaomiMiMo/MiMo-V2.5-ASR` + `XiaomiMiMo/MiMo-Audio-Tokenizer` (~20 GB) | Opt-in download via `INSTALL_MIMO=1` |
| Code | `src.mimo_audio.mimo_audio.MimoAudio` from MiMo GitHub repo (not a pip package) | `git clone` into `venv/mimo/`, add to `sys.path` |
| Output | Single text string per `asr_sft()` call | No timestamps, no speaker labels — must layer VAD + CAM++ on top |
| VRAM | ~20 GB (8B fp16 + tokenizer + KV cache) | Pre-check required; 16 GB cards will OOM |
| License | MIT, weights openly redistributable | No legal blocker to local use |

## Architecture

### Pipeline integration

The skill's Phase 1 is split into two paths that **share VAD and diarization**
and diverge only in ASR text generation:

```
audio
  │
  ├─ existing presets (zh / en / ...): FunASR AutoModel produces
  │       {sentence_info: [{text, speaker, start_ms, end_ms}, ...]}
  │
  └─ mimo preset:
         (1) FSMN VAD → [(start_ms, end_ms), ...]         ← reuse existing model
         (2) extract each VAD segment to a temp WAV file
         (3) MimoAudio.asr_sft(wav, audio_tag) → text     ← MiMo only does this
         (4) CAM++ embeddings on each segment → cluster → speaker label
         (5) assemble {text, speaker, start_ms, end_ms}   ← format matches (zh)
```

Phase 2 (`merge_consecutive`, speaker map, gender, reference correction) and
Phase 3 (LLM cleanup) are **unchanged** — step (5) emits the same structure
existing presets produce.

### File layout

- **`scripts/mimo_asr.py`** — new module. Encapsulates:
  - `require_cuda_and_vram(min_gb=20)` — pre-flight GPU check
  - `require_mimo_installed(weights_path, repo_path)` — pre-flight weights/repo check
  - `run_fsmn_vad(audio_path, ...)` — wrapper around FunASR VAD model returning
    `[(start_ms, end_ms), ...]`
  - `extract_segment(audio_path, s_ms, e_ms, tmpdir)` — ffmpeg cut to WAV
  - `transcribe_with_mimo(...)` — the full Phase 1 MiMo path (returns
    `sentence_info`-shaped list)
  - `assign_speakers_via_cam(...)` — CAM++ embedding + clustering adapter
  - Partial-save / resume helpers (`save_partial`, `load_partial`, audio hash)

- **`scripts/transcribe_funasr.py`** — existing main pipeline. Modified:
  - Add `"mimo"` to `MODEL_PRESETS` (metadata only — no ASR model id, no VAD id
    beyond what we reference directly).
  - Add `SUPPORTED_LANGS` entry.
  - In `transcribe_with_funasr(...)`, branch early:
    `if lang == "mimo": return mimo_asr.transcribe_with_mimo(...)`.
  - Add CLI flags: `--mimo-audio-tag`, `--mimo-batch`, `--mimo-weights-path`,
    `--resume-mimo`.
  - Warnings on incompatible flags (`--hotwords`, `--batch-size` with `--lang mimo`).

- **`scripts/setup_mimo.sh`** — new script. Clones MiMo repo, installs
  requirements + flash-attn, downloads weights. Idempotent; safe to re-run.

- **`scripts/setup_env.sh`** — modified:
  - Detect `python3.12`; fail clearly on missing with platform-specific guidance.
  - Rebuild `venv/` if existing Python version ≠ 3.12.
  - Install all existing FunASR dependencies under Python 3.12.
  - If `INSTALL_MIMO=1`, call `setup_mimo.sh` at the end.

- **`scripts/test_mimo_asr.py`** — new unit tests (mocked, no GPU, no weights).

### Config surface

Environment variables:

| Variable | Consumed by | Purpose |
|---|---|---|
| `INSTALL_MIMO` | `setup_env.sh` | If set, also run `setup_mimo.sh` |
| `MIMO_WEIGHTS_PATH` | `setup_mimo.sh`, runtime (via `HF_HOME`) | Override default `~/.cache/huggingface` |
| `HF_HOME` | huggingface_hub, transformers | Standard HF cache env var; respected as fallback |

CLI flags (new):

| Flag | Default | Scope | Notes |
|---|---|---|---|
| `--mimo-audio-tag {<chinese>\|<english>\|<auto>}` | `<chinese>` | mimo only | MiMo's language hint passed to `asr_sft()` |
| `--mimo-batch N` | `1` | mimo only | Concurrent VAD segments per inference call. Keep at 1 unless on H100/80GB |
| `--mimo-weights-path DIR` | `$HF_HOME` → `~/.cache/huggingface` | mimo only | Cache directory for MiMo weights. Sets `HF_HOME` internally |
| `--resume-mimo` | off | mimo only | Resume Phase 1 MiMo inference from `*_mimo_partial.json` |

### Environment setup

**`setup_env.sh` Python 3.12 migration (breaking change for existing users):**

1. Detect `python3.12` availability. On missing:
   - Ubuntu 24.04: suggest `apt install python3.12 python3.12-venv` (bundled).
   - Ubuntu 22.04: suggest `deadsnakes` PPA or manual install (do NOT add PPA
     automatically — requires `sudo` and breaks CI/containers).
   - macOS: suggest `brew install python@3.12`.
   - Other: error with links to https://www.python.org/downloads/.
2. If `venv/` exists and its Python ≠ 3.12: print `"venv is Python 3.X,
   rebuilding for 3.12"` and delete + recreate.
3. Install all current dependencies under Python 3.12.
4. If `INSTALL_MIMO=1`:
   ```bash
   bash "$(dirname "$0")/setup_mimo.sh"
   ```

**`setup_mimo.sh` (new):**

```bash
#!/usr/bin/env bash
set -euo pipefail

MIMO_WEIGHTS_PATH="${MIMO_WEIGHTS_PATH:-${HF_HOME:-$HOME/.cache/huggingface}}"
VENV_ROOT="${VENV_ROOT:-./venv}"

# 1. Clone MiMo repo (provides src.mimo_audio — not a pip package)
if [[ ! -d "$VENV_ROOT/mimo" ]]; then
    git clone --depth 1 \
        https://github.com/XiaomiMiMo/MiMo-V2.5-ASR.git \
        "$VENV_ROOT/mimo"
fi

# 2. Install MiMo's Python dependencies
pip install -r "$VENV_ROOT/mimo/requirements.txt"

# 3. flash-attn (requires nvcc; can take 10–30 min to compile)
# Verify nvcc is present before attempting
if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found. flash-attn requires CUDA toolkit."
    echo "Install: https://developer.nvidia.com/cuda-toolkit"
    exit 1
fi
pip install flash-attn==2.7.4.post1 --no-build-isolation

# 4. Download weights to MIMO_WEIGHTS_PATH
HF_HOME="$MIMO_WEIGHTS_PATH" \
    huggingface-cli download XiaomiMiMo/MiMo-V2.5-ASR
HF_HOME="$MIMO_WEIGHTS_PATH" \
    huggingface-cli download XiaomiMiMo/MiMo-Audio-Tokenizer
```

Typical usage:

```bash
# Machine with persistent model volume
INSTALL_MIMO=1 MIMO_WEIGHTS_PATH=/mnt/models/hf bash setup_env.sh

# Runtime — same path
python3 $SCRIPTS/transcribe_funasr.py podcast.m4a \
    --lang mimo --mimo-weights-path /mnt/models/hf
```

Runtime path resolution inside `mimo_asr.py`:
`--mimo-weights-path` (if set) > `$HF_HOME` (if set) > `~/.cache/huggingface`.
The resolved value is written to `os.environ["HF_HOME"]` before importing
`MimoAudio`, so `huggingface_hub` / `transformers` pick it up automatically.

### Pre-flight checks

Run in this order before any expensive work:

1. **`require_cuda_and_vram(min_gb=20)`**
   - `torch.cuda.is_available()` → else exit with "MiMo requires a CUDA GPU.
     Use --lang zh for CPU."
   - `torch.cuda.get_device_properties(0).total_memory >= 20 * 1024**3` →
     else exit with "MiMo requires ≥20 GB VRAM. Detected: {name} ({gb:.1f} GB)."
2. **`require_mimo_installed(weights_path, repo_path)`**
   - `repo_path` (e.g. `venv/mimo/`) exists → else exit with
     "Run: INSTALL_MIMO=1 bash $SCRIPTS/setup_env.sh"
   - Both weights resolve via `huggingface_hub.snapshot_download(repo_id,
     cache_dir=weights_path, local_files_only=True)` without raising. If the
     call raises `LocalEntryNotFoundError`, exit with: "MiMo weights not found
     at {weights_path}. Run: INSTALL_MIMO=1 MIMO_WEIGHTS_PATH={weights_path}
     bash $SCRIPTS/setup_env.sh"
   - Checked repos: `XiaomiMiMo/MiMo-V2.5-ASR`, `XiaomiMiMo/MiMo-Audio-Tokenizer`.

### Inference loop

```python
def transcribe_with_mimo(audio_path, num_speakers, audio_tag="<chinese>",
                         batch=1, weights_path=None, resume=False):
    # 1. Pre-checks (fail fast)
    require_cuda_and_vram(min_gb=20)
    require_mimo_installed(weights_path, repo_path="venv/mimo")

    # 2. Resume path OR fresh VAD
    partial_path = audio_path.with_suffix(".mimo_partial.json")
    if resume:
        state = load_partial(partial_path, audio_path, audio_tag)
        vad_segments = state["vad_segments"]
        completed = {c["idx"]: c for c in state["completed"]}
        start_idx = state["failed_at"]["idx"]
    else:
        vad_segments = run_fsmn_vad(audio_path, max_single_segment_time=60000)
        completed = {}
        start_idx = 0

    # 3. Load MiMo.
    #    We resolve snapshot directories via huggingface_hub.snapshot_download
    #    (with local_files_only=True, since pre-check confirmed download).
    #    This returns the canonical path under hub/.../snapshots/<hash>/
    #    without hardcoding snapshot hashes.
    #    venv_root is resolved from sys.prefix at runtime (the active venv dir).
    venv_root = Path(sys.prefix)
    sys.path.insert(0, str(venv_root / "mimo"))
    from huggingface_hub import snapshot_download
    from src.mimo_audio.mimo_audio import MimoAudio
    model_dir = snapshot_download("XiaomiMiMo/MiMo-V2.5-ASR",
                                  cache_dir=weights_path, local_files_only=True)
    tokenizer_dir = snapshot_download("XiaomiMiMo/MiMo-Audio-Tokenizer",
                                      cache_dir=weights_path, local_files_only=True)
    mimo = MimoAudio(model_path=model_dir, tokenizer_path=tokenizer_dir)

    # 4. Per-segment loop with retry
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(start_idx, len(vad_segments)):
            if i in completed:
                continue
            s_ms, e_ms = vad_segments[i]
            chunk_wav = extract_segment(audio_path, s_ms, e_ms, tmpdir)
            text = infer_with_retry(mimo, chunk_wav, audio_tag,
                                    max_retries=3, backoffs=[0.5, 2.0, 5.0])
            if text is None:
                # Final failure — save partial, raise
                save_partial(partial_path, audio_path, audio_tag, weights_path,
                             vad_segments, list(completed.values()),
                             failed_at={"idx": i, "start_ms": s_ms,
                                        "error": "... after 3 retries"})
                raise RuntimeError(f"MiMo failed at segment {i}/{len(vad_segments)}")
            completed[i] = {"idx": i, "text": text,
                            "start_ms": s_ms, "end_ms": e_ms}
            print(f"  [{i+1:3d}/{len(vad_segments)}] {format_time(s_ms)} "
                  f"({e_ms-s_ms}ms) → {text[:40]}...")

    # 5. Free MiMo VRAM before loading CAM++
    del mimo
    torch.cuda.empty_cache()

    # 6. CAM++ speaker clustering on the same VAD segments
    sentences = [completed[i] for i in range(len(vad_segments))]
    sentences = assign_speakers_via_cam(sentences, audio_path, num_speakers)

    # 7. Success → clean up partial
    if partial_path.exists():
        partial_path.unlink()

    # 8. Report
    wall = time.time() - t0
    duration = (vad_segments[-1][1] - vad_segments[0][0]) / 1000
    print(f"MiMo inference: {wall:.1f}s (RTF {wall/duration:.3f})")
    return sentences
```

### Failure handling

| Failure | Action |
|---|---|
| Pre-check (CUDA / VRAM / weights / repo) | Exit immediately, non-zero code, clear message |
| Single segment inference (OOM, CUDA error, corrupt audio) | Retry up to 3× with backoffs [0.5s, 2s, 5s], calling `gc.collect() + torch.cuda.empty_cache()` between attempts |
| 3 retries all fail | Save partial to `{stem}_mimo_partial.json`, raise, exit non-zero, instruct user to `--resume-mimo` |
| CAM++ clustering failure (Phase 2 of MiMo path) | Same as today's presets — surfaces as regular exception |
| Phase 2/3 failure after Phase 1 completed | `*_raw_transcript.json` already written; `--skip-transcribe` resumes as today |

### Partial state format

`{stem}_mimo_partial.json`:

```json
{
  "audio_hash": "sha256:...",
  "audio_tag": "<chinese>",
  "mimo_weights_path": "/mnt/models/hf",
  "vad_segments": [[3000, 4200], [5100, 8500], ...],
  "completed": [
    {"idx": 0, "text": "...", "start_ms": 3000, "end_ms": 4200},
    ...
  ],
  "failed_at": {
    "idx": 97,
    "start_ms": 5025000,
    "error": "CUDA OOM after 3 retries"
  }
}
```

**`audio_hash`** is SHA256 of the preprocessed FLAC (not the original file —
avoids false mismatches when the user renames an m4a or switches
`--audio-format`). Cost: ~0.5s for a 4h / 200MB FLAC, negligible.

### Resume logic

`--resume-mimo`:

1. Load `{stem}_mimo_partial.json`.
2. Verify `audio_hash` matches current preprocessed FLAC. Mismatch → error:
   "audio file changed since partial was saved. Delete *_mimo_partial.json to
   restart."
3. Verify `audio_tag` matches current CLI value. Mismatch → error.
4. Skip VAD; use cached `vad_segments`.
5. Continue inference from `failed_at.idx`.
6. On full success, proceed through CAM++ and write `*_raw_transcript.json`,
   then **delete** `*_mimo_partial.json`.

Interaction with existing `--skip-transcribe`: independent. If
`*_raw_transcript.json` exists, `--skip-transcribe` takes precedence (Phase 1
entirely skipped). If only `*_mimo_partial.json` exists, use `--resume-mimo`.

### Progress output

```
Phase 1.5/3: VAD segmentation (FSMN)
  Detected 237 segments across 14834.2s
Phase 2/3: MiMo ASR (local, GPU)
  Loading MiMo-V2.5-ASR from /mnt/models/hf...  12.4s
  [  1/237] 00:00:03 (1200ms) → 欢迎收听...
  [  2/237] 00:00:05 (3400ms) → 我们今天请到的嘉宾是...
  ...
  [237/237] 04:07:12 (2100ms) → 谢谢大家收听
  MiMo inference: 1847.3s (RTF 0.124)
Phase 2.5/3: CAM++ speaker clustering
```

RTF (real-time factor) is explicitly reported to enable direct speed
comparison against `--lang zh` (Paraformer GPU RTF typically 0.02–0.05).

## Testing

### Unit tests (`scripts/test_mimo_asr.py`)

No GPU, no weights, all mocked. CI-safe.

| Test | Asserts |
|---|---|
| `test_require_cuda_and_vram_no_cuda` | Clear error mentioning "CUDA" when `is_available()` mocked False |
| `test_require_cuda_and_vram_insufficient` | Clear error mentioning "≥20 GB" for 16 GB mock |
| `test_require_cuda_and_vram_ok` | Returns None for 24 GB mock |
| `test_require_mimo_installed_missing_weights` | Error mentions `INSTALL_MIMO=1` |
| `test_require_mimo_installed_missing_repo` | Error mentions missing repo clone |
| `test_retry_on_transient_failure` | 2 failures + 1 success → returns correct text, 3 calls |
| `test_fail_after_max_retries` | 3 failures → raises; partial.json written with completed segments |
| `test_partial_json_roundtrip` | Write → read returns equivalent dict |
| `test_resume_audio_hash_mismatch` | Raises with clear error |
| `test_resume_audio_tag_mismatch` | Raises with clear error |
| `test_resume_skips_completed_segments` | 96/237 in partial → only 141 MimoAudio calls |
| `test_cli_flag_precedence` | `--mimo-weights-path` > `$HF_HOME` > default |
| `test_warn_on_hotwords_with_mimo` | stderr contains warning, no raise |

### Not tested automatically

- End-to-end GPU inference — requires 20 GB GPU + 20 GB weights; developer-local only.
- Accuracy benchmark — manual: user runs xiaoyuzhou podcast through `--lang mimo`
  and `--lang zh`, diffs output.
- `flash-attn` compile — too heavy for CI; verified manually via `setup_mimo.sh`.

### Existing test suite

`test_speaker_verification.py` is unchanged — MiMo preset's Phase 1 output
format is identical to `zh`, so downstream speaker verification is shared.

Run:
```
cd plugins/funasr-transcriber/skills/funasr-transcribe/scripts
python3 -m pytest test_speaker_verification.py test_mimo_asr.py -v
```

### Manual acceptance checklist

1. On a GPU machine:
   `INSTALL_MIMO=1 MIMO_WEIGHTS_PATH=/mnt/models/hf bash setup_env.sh` completes.
2. `--lang mimo` + short (< 5 min) test audio produces a transcript with
   speaker labels and timestamps in the same format as `--lang zh`.
3. xiaoyuzhou 4h+ podcast:
   - Record RTF (for speed comparison vs `--lang zh`).
   - Diff output against `--lang zh` on the same audio; judge accuracy.
   - Note any failed segments in the progress output.
4. Ctrl-C mid-run then `--resume-mimo` → completes successfully.
5. On a CPU-only or <20 GB GPU machine: `--lang mimo` exits immediately with
   a clear error (does not attempt to run).

## Migration Impact

- **Breaking change:** all users re-running `setup_env.sh` will have their
  `venv/` rebuilt under Python 3.12. Expect a 2–3 GB re-download of FunASR
  dependencies. CHANGELOG entry needs prominent warning.
- **Backward compatibility:** all existing CLI flags and presets continue to
  work identically. Default `--lang zh` path is untouched; only the Python
  version under it changes.
- **Version:** bump to 1.7.0 (minor — additive feature with a breaking
  environment change).

## Documentation Updates

- `SKILL.md`:
  - Add `mimo` row to "Supported Languages" table with footnote on GPU-only requirement.
  - Add `--mimo-*` flags to "Key Flags" table.
  - Add `setup_mimo.sh` / `INSTALL_MIMO=1` note under "Environment Setup".
  - Add a brief "When to use `--lang mimo`" blurb: dialects, code-switching,
    lyrics, or evaluating alternatives to Paraformer.
- `references/pipeline-details.md`: document the VAD + MiMo + CAM++ Phase 1
  architecture and RTF expectations.
- `README.md`: add MiMo to the feature list, link to GitHub/HF.
- `CHANGELOG.md`: 1.7.0 entry with Python 3.12 upgrade warning.

## Open Questions

None — all decisions confirmed during brainstorming:

- Local weights (not HF Space) — confirmed
- Python 3.12 main venv upgrade (not isolated venv) — confirmed
- `INSTALL_MIMO=1` env var trigger — confirmed
- Reuse FSMN VAD + CAM++ for diarization — confirmed
- Hard-fail on CUDA missing or VRAM < 20 GB — confirmed
- Retry 3× per segment then hard-fail with resume — confirmed
- `--mimo-weights-path` flag for container / shared-volume scenarios — confirmed
