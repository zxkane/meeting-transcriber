# Audio Transcriber

[![View on ClawHub](https://img.shields.io/badge/ClawHub-audio--transcriber--funasr-6E56CF?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTUtMTAtNXptMCAxMUwyIDhsLjA1IDkuOTVMMTIgMjNsOS45NS01LjA1TDIyIDhsLTEwIDV6Ii8+PC9zdmc+)](https://clawhub.ai/zxkane/zxkane-audio-transcriber-funasr)

Claude Code plugin for multi-speaker meeting and podcast transcription with automatic speaker diarization and LLM cleanup. Supports **two ASR engine families**:

- **[FunASR](https://github.com/modelscope/FunASR)** — Paraformer / SenseVoice / Whisper, fast and cheap, 99 languages, CPU or GPU
- **[MiMo-V2.5-ASR](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR)** — 8B parameters, local GPU only, stronger on proper nouns, code-switching, and rare terms (opt-in since 1.7.0)

Both engines share the same VAD (FSMN) + speaker-clustering (CAM++) pipeline, so speaker diarization behavior is consistent across engines.

## Features

- **Meetings & podcasts** — Handles large meetings (10+ speakers) and podcasts/interviews (2–3 speakers) with CAM++ speaker diarization, `--num-speakers` hint, real name mapping, and speaker context for LLM identification
- **Hotword biasing** — SeACo-Paraformer accepts participant names and domain terms to improve recognition accuracy (+50% on tested Chinese terms, FunASR path only)
- **Multi-language** — Chinese (SeACo-Paraformer, CER 1.95%), English (Paraformer-en), auto-detect (SenseVoiceSmall: zh/en/ja/ko/yue), 99 languages (Whisper-large-v3-turbo), or Chinese + dialects + code-switching (MiMo-V2.5-ASR)
- **Long recordings** — Handles 6+ hour recordings without splitting (includes spectral clustering performance patch)
- **LLM cleanup** — Bedrock Claude / Anthropic / OpenAI-compatible — removes fillers, fixes ASR errors, polishes grammar
- **GPU & CPU** — FunASR auto-detects CUDA and falls back to CPU ([low-memory guidance](plugins/audio-transcriber/skills/audio-transcribe/references/pipeline-details.md#running-on-cpu-only--low-memory-machines)); MiMo requires ≥20 GB VRAM
- **Resume support** — Checkpoint at every phase for interrupted runs (FunASR `--skip-transcribe`; MiMo `--resume-mimo` with audio-hash verification)

## Installation

### As Agent Skill (via [skills.sh](https://skills.sh))

```bash
npx skills add zxkane/audio-transcriber
```

### As Claude Code Plugin

Add as a marketplace, then install:

```bash
# In Claude Code
/plugin marketplace add zxkane/audio-transcriber
/plugin install audio-transcriber@zxkane-audio-transcriber-funasr
```

> Note: the ClawHub package slug is still `zxkane-audio-transcriber-funasr` for
> backward compatibility with users who already have 1.6.x or earlier installed.
> The repository was renamed from `audio-transcriber-funasr` in 1.7.1 as the
> plugin now supports both FunASR and MiMo. Old GitHub URLs redirect.

### Manual Usage

```bash
# 1. Set up environment (auto-detects GPU/CPU; requires Python 3.12 since 1.7.0)
bash plugins/audio-transcriber/skills/audio-transcribe/scripts/setup_env.sh
source .venv/bin/activate

# 2. (Optional) install MiMo-V2.5-ASR locally — ~34 GB download, ≥20 GB VRAM
INSTALL_MIMO=1 MIMO_WEIGHTS_PATH=/path/to/hf-cache \
  bash plugins/audio-transcriber/skills/audio-transcribe/scripts/setup_env.sh

# 3. Convert audio to 16kHz mono FLAC (lossless, ~50% smaller than WAV)
ffmpeg -i recording.m4a -ar 16000 -ac 1 -sample_fmt s16 meeting.flac

# 4. Chinese meeting, 9 speakers (FunASR SeACo-Paraformer)
python3 plugins/audio-transcriber/skills/audio-transcribe/scripts/transcribe.py \
  meeting.flac --lang zh --num-speakers 9

# 5. English meeting, with real names
python3 plugins/audio-transcriber/skills/audio-transcribe/scripts/transcribe.py \
  meeting.flac --lang en --speakers "Alice,Bob,Carol,Dave"

# 6. Chinese podcast with proper-noun-heavy content (MiMo, local GPU)
python3 plugins/audio-transcriber/skills/audio-transcribe/scripts/transcribe.py \
  episode.flac --lang mimo --num-speakers 2 --speakers "Host,Guest" \
  --mimo-weights-path /path/to/hf-cache

# 7. Auto-detect language (zh/en/ja/ko/yue)
python3 plugins/audio-transcriber/skills/audio-transcribe/scripts/transcribe.py \
  meeting.flac --lang auto --num-speakers 6
```

## Plugin Structure

```
.
├── .claude-plugin/
│   └── marketplace.json          # skills.sh marketplace registration
├── .claude/
│   └── skills/
│       └── audio-transcribe -> ../../plugins/audio-transcriber/skills/audio-transcribe
├── plugins/
│   └── audio-transcriber/
│       └── skills/
│           └── audio-transcribe/
│               ├── SKILL.md              # Skill entry point
│               ├── references/
│               │   └── pipeline-details.md
│               └── scripts/
│                   ├── transcribe.py             # Main pipeline
│                   ├── mimo_asr.py               # MiMo-V2.5-ASR integration
│                   ├── patch_clustering.py       # Long-audio perf fix
│                   ├── setup_env.sh              # One-click env setup
│                   └── setup_mimo.sh             # Opt-in MiMo installer
├── CLAUDE.md
├── README.md
└── .gitignore
```

## Pipeline

```
Audio (.m4a/.mp3) ─► ffmpeg ─► 16kHz WAV
                                  │
Phase 1: ASR                     │
  ├─ FSMN-VAD (voice detection)  │
  ├─ ASR engine:                 │
  │    FunASR (Paraformer/       ├─► raw_transcript.json
  │    SenseVoice/Whisper)       │
  │    OR MiMo-V2.5-ASR (local)  │
  ├─ Hotword biasing (SeACo-zh)  │
  ├─ Punctuation restoration     │
  └─ CAM++ (speaker clustering)  │
                                  │
Phase 2: Post-process            │
  ├─ Merge consecutive utterances├─► merged segments
  └─ Map speaker IDs to names    │
                                  │
Phase 3: LLM cleanup (optional)  │
  └─ Bedrock / Anthropic /       └─► transcript.md
     OpenAI-compatible
```

## Performance

Benchmarked on a 4h14m, 9-speaker Chinese meeting recording (FunASR):

| Phase | GPU (L40S) | CPU |
|-------|-----------|-----|
| Model load | 14s | ~30s |
| Transcription | 169s | ~30-60 min |
| Clustering (patched) | ~10s | ~2-5 min |
| LLM cleanup (17 chunks) | ~35 min | ~35 min |

**Without the clustering patch**, speaker clustering on long audio takes 10+ hours.
The patch replaces O(N^3) `scipy.linalg.eigh` with O(N^2·k) `scipy.sparse.linalg.eigsh`.

### FunASR vs MiMo at a glance (6h45m Chinese podcast, same GPU L40S)

| Engine | Wall | RTF | Cost / episode (Spot L4 est.) | Accuracy on proper nouns |
|---|---|---|---|---|
| FunASR (Paraformer) | ~6 min | 0.014 | ~$0.05 | Good |
| MiMo-V2.5-ASR | ~49 min | 0.12 | ~$0.94 | **Visibly better** |

See `docs/superpowers/reports/2026-04-30-mimo-vs-funasr-perf-cost.md` for the full comparison.

## License

MIT
