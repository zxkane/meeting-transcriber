---
name: funasr-transcribe
version: 1.2.0
description: >
  This skill should be used when the user asks to "transcribe a meeting",
  "transcribe audio", "transcribe a meeting recording",
  "convert audio to text", "generate meeting minutes from audio",
  "do speech-to-text", "transcribe with speaker diarization",
  "identify speakers in audio", "transcribe Chinese audio",
  "transcribe English audio", "transcribe Japanese audio",
  "multi-speaker transcription", "transcribe a podcast",
  "transcribe podcast episode", "transcribe an interview",
  or mentions FunASR, Paraformer, SenseVoice, Whisper, meeting
  transcription, podcast transcription, or speaker diarization.
  Supports multi-speaker meeting and podcast transcription in Chinese,
  English, Japanese, Korean, Cantonese, and 99 languages (via Whisper)
  with automatic speaker diarization and hotword biasing.
  Works on both GPU and CPU.
---

# FunASR Meeting Transcription

Transcribe multi-speaker meetings, podcasts, and interviews into
structured Markdown with automatic speaker diarization, hotword biasing,
and optional LLM cleanup, using the open-source FunASR pipeline.

**Optimized for long-form audio**: handles arbitrarily long recordings
(4+ hours tested), separates speakers via CAM++ diarization,
merges consecutive utterances, and maps speaker IDs to real names.
Works for both large meetings (10+ speakers) and podcasts (2–3 speakers).

## Supported Languages

| `--lang` | Model | Languages | Hotword |
|----------|-------|-----------|---------|
| `zh` (default) | SeACo-Paraformer | Chinese (CER 1.95%) | Yes |
| `zh-basic` | Paraformer-large | Chinese | No |
| `en` | Paraformer-en | English | No |
| `auto` | SenseVoiceSmall | Auto-detect: zh/en/ja/ko/yue | No |
| `whisper` | Whisper-large-v3-turbo | 99 languages | No |

All presets include **speaker diarization** (CAM++) and **VAD** (FSMN).

## Workflow

Before starting transcription, **always ask the user** the following:

1. **Audio file** — path to the recording (required)
2. **Type** — meeting, podcast, or interview? (affects defaults)
3. **Language** — what language is the recording in? (default: Chinese)
4. **Number of speakers** — how many participants? (improves diarization)
5. **Speaker names** — for podcasts: host + guest names; for meetings: attendee list
6. **Supporting files** — ask:
   > "Do you have any of the following to improve transcription accuracy?"
   > - **Attendee / guest list** — used for hotwords and speaker mapping
   > - **Meeting agenda or episode topic** — used for hotwords (terms, names)
   > - **Reference documents** (prior notes, show notes, etc.) — used to identify speakers via keyword matching after transcription
   >
   > These are optional but significantly improve speaker identification
   > and domain-specific term recognition.

If the user provides supporting materials:
- Extract participant names and key terms → create `hotwords.txt`
- Extract per-person context → create `speaker-context.json`
- Use both with `--hotwords` and `--speaker-context` flags

If no supporting files are available, proceed with `--num-speakers` only.

## Quick Start

### 1. Environment Setup

```bash
bash ${CLAUDE_PLUGIN_ROOT}/skills/funasr-transcribe/scripts/setup_env.sh
# Or force CPU:  bash setup_env.sh cpu
```

**Critical for long meetings**: The setup script patches FunASR's spectral
clustering for O(N^2*k) performance. Without this, recordings over ~1 hour
hang for hours during speaker clustering.

### 2. Audio Preprocessing

Convert to 16kHz mono. Two options depending on priority:

```bash
# Smallest file (lossy, -3% sentences vs lossless — fine for most meetings)
ffmpeg -i recording.m4a -ar 16000 -ac 1 -c:a libopus -b:a 32k meeting.opus

# Lossless (larger, best quality)
ffmpeg -i recording.m4a -ar 16000 -ac 1 -sample_fmt s16 meeting.flac
```

| Format | 4h14m meeting | Quality | ASR impact |
|--------|--------------|---------|-----------|
| **Opus 32kbps** | **55MB** | Lossy | **-3% sentences, keywords intact** |
| FLAC 16-bit | 219MB | Lossless | Baseline |
| WAV | 465MB | Lossless | Same as FLAC |
| Original M4A | 173MB | Source | Also works directly |

FunASR natively reads FLAC, WAV, Opus, M4A, and MP3.
**Do NOT split long recordings** — splitting breaks speaker ID consistency.

### 3. Prepare Supporting Files (Recommended)

Prepare these from the meeting invite / attendee list:

- **`hotwords.txt`** — One term per line: participant names, project names,
  domain jargon (Chinese terms work best; English terms may regress)
- **`speaker-context.json`** — Per-person keywords for LLM speaker identification

See `references/pipeline-details.md` for format details and effectiveness data.

### 4. Run Transcription

Copy the script to the working directory (output files written to CWD):

```bash
cp ${CLAUDE_PLUGIN_ROOT}/skills/funasr-transcribe/scripts/transcribe_funasr.py .
```

**Prerequisites for Phase 3 (LLM cleanup):** AWS credentials with Bedrock
`InvokeModel` permission. Skip with `--skip-llm` if unavailable.

```bash
# Chinese meeting with hotwords (recommended)
python3 transcribe_funasr.py meeting.wav --lang zh --num-speakers 9 \
    --hotwords hotwords.txt

# English meeting with speaker names
python3 transcribe_funasr.py meeting.wav --lang en \
    --speakers "Alice,Bob,Carol,Dave"

# Auto-detect language (zh/en/ja/ko/yue)
python3 transcribe_funasr.py meeting.wav --lang auto --num-speakers 6

# Whisper for any language
python3 transcribe_funasr.py meeting.wav --lang whisper --num-speakers 4

# Full pipeline with all supporting files
python3 transcribe_funasr.py meeting.wav --lang zh --num-speakers 9 \
    --hotwords hotwords.txt \
    --speakers "Alice,Bob,Carol" \
    --speaker-context speaker-context.json

# Raw transcription only (no LLM)
python3 transcribe_funasr.py meeting.wav --skip-llm

# Resume interrupted LLM cleanup
python3 transcribe_funasr.py meeting.wav --skip-transcribe
```

### 5. Speaker Identification (Post-Processing)

FunASR's CAM++ may merge acoustically similar speakers. To improve:

1. **`--num-speakers N`** — Hint expected count
2. **`--hotwords`** — Include participant names (Chinese names work best)
3. **Keyword matching** — Search `*_raw_transcript.json` for unique phrases
4. **`--speaker-context`** — Provide per-person keywords for LLM splitting

## Key Flags

| Flag | Purpose |
|------|---------|
| `--lang` | `zh` (default), `zh-basic`, `en`, `auto`, `whisper` |
| `--hotwords` | Hotword file or string — biases ASR toward terms (zh only) |
| `--num-speakers N` | Expected speaker count (improves diarization) |
| `--speakers "A,B,C"` | Assign real names by first-appearance order |
| `--speaker-context F` | JSON with per-speaker keywords for LLM |
| `--device cpu` | Force CPU mode |
| `--batch-size N` | Adjust for memory (60 for CPU, 100 if GPU OOM) |
| `--skip-transcribe` | Resume from saved `*_raw_transcript.json` |
| `--skip-llm` | Skip LLM cleanup |
| `--bedrock-model ID` | Override LLM model (default: `us.anthropic.claude-sonnet-4-6`) |
| `--clean-cache` | Delete LLM chunk cache after completion |

## Outputs

- `<stem>-transcript.md` — Final Markdown with speaker labels
- `<stem>_raw_transcript.json` — Raw Phase 1 output (for resume/analysis)

## Additional Resources

- **`references/pipeline-details.md`** — Architecture, model specs, benchmarks,
  hotword effectiveness data, clustering patch, diarization limitations,
  supporting file preparation guide
- **`scripts/transcribe_funasr.py`** — Main transcription pipeline
- **`scripts/setup_env.sh`** — Environment setup (venv + deps + patch)
- **`scripts/patch_clustering.py`** — Sparse eigsh patch for long meetings

## CPU-only / Low-Memory Machines

Long recordings (2+ hours) on resource-constrained machines may hit two silent
failure modes: **exec timeouts** (agent kills the process) and **OOM kills**
(kernel kills the process when RAM is exhausted).

See `references/pipeline-details.md § Running on CPU-only / Low-Memory Machines`
for detailed workarounds:
- Detach from agent timeouts with `systemd-run` or `nohup`
- Prevent OOM via swap and/or `--lang zh-basic` (lighter model)

## Podcast / Interview Transcription

The pipeline also handles podcasts and interviews. Key differences from meetings:
- Fewer speakers (2–3), usually known upfront — always provide `--num-speakers`
  and `--speakers`
- Use `--lang auto` for bilingual shows, `--lang whisper` for other languages
- Provide `--speaker-context` describing host/guest roles for better LLM cleanup

See `references/pipeline-details.md § Podcast Transcription` for recommended
settings and examples.
