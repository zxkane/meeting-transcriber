# FunASR Meeting Transcriber

Claude Code plugin for multi-speaker meeting transcription with automatic speaker diarization and LLM cleanup, powered by [FunASR](https://github.com/modelscope/FunASR).

## Features

- **Multi-speaker meetings** — CAM++ speaker diarization separates who said what, with `--num-speakers` hint, real name mapping, and speaker context for LLM identification
- **Hotword biasing** — SeACo-Paraformer accepts participant names and domain terms to improve recognition accuracy (+50% on tested Chinese terms)
- **Multi-language** — Chinese (SeACo-Paraformer, CER 1.95%), English (Paraformer-en), auto-detect (SenseVoiceSmall: zh/en/ja/ko/yue), or 99 languages (Whisper-large-v3-turbo)
- **Long recordings** — Handles 4+ hour meetings without splitting (includes spectral clustering performance patch)
- **LLM cleanup** — Bedrock Claude removes fillers, fixes ASR errors, polishes grammar
- **GPU & CPU** — Auto-detects CUDA; fully functional on CPU (slower)
- **Resume support** — Checkpoint at every phase for interrupted runs

## Installation

### As Agent Skill (via [skills.sh](https://skills.sh))

```bash
npx skills add zxkane/meeting-transcriber
```

### As Claude Code Plugin

Add as a marketplace, then install:

```bash
# In Claude Code
/plugin marketplace add zxkane/meeting-transcriber
/plugin install funasr-transcriber@zxkane-meeting-transcriber
```

### Manual Usage

```bash
# 1. Set up environment (auto-detects GPU/CPU)
bash plugins/funasr-transcriber/skills/funasr-transcribe/scripts/setup_env.sh
source .venv/bin/activate

# 2. Convert audio to 16kHz mono FLAC (lossless, ~50% smaller than WAV)
ffmpeg -i recording.m4a -ar 16000 -ac 1 -sample_fmt s16 meeting.flac

# 3. Chinese meeting, 9 speakers
python3 plugins/funasr-transcriber/skills/funasr-transcribe/scripts/transcribe_funasr.py \
  meeting.flac --lang zh --num-speakers 9

# 4. English meeting, with real names
python3 plugins/funasr-transcriber/skills/funasr-transcribe/scripts/transcribe_funasr.py \
  meeting.flac --lang en --speakers "Alice,Bob,Carol,Dave"

# 5. Auto-detect language (zh/en/ja/ko/yue)
python3 plugins/funasr-transcriber/skills/funasr-transcribe/scripts/transcribe_funasr.py \
  meeting.flac --lang auto --num-speakers 6
```

## Plugin Structure

```
.
├── .claude-plugin/
│   └── marketplace.json          # skills.sh marketplace registration
├── .claude/
│   └── skills/
│       └── funasr-transcribe -> ../../plugins/funasr-transcriber/skills/funasr-transcribe
├── plugins/
│   └── funasr-transcriber/
│       └── skills/
│           └── funasr-transcribe/
│               ├── SKILL.md              # Skill entry point
│               ├── references/
│               │   └── pipeline-details.md
│               └── scripts/
│                   ├── transcribe_funasr.py    # Main pipeline
│                   ├── patch_clustering.py     # Long-audio perf fix
│                   └── setup_env.sh            # One-click env setup
├── CLAUDE.md
├── README.md
└── .gitignore
```

## Pipeline

```
Audio (.m4a/.mp3) ─► ffmpeg ─► 16kHz WAV
                                  │
Phase 1: FunASR ASR              │
  ├─ FSMN-VAD (voice detection)  │
  ├─ ASR model (lang-dependent)  ├─► raw_transcript.json
  ├─ Hotword biasing (zh only)   │
  ├─ Punctuation restoration     │
  └─ CAM++ (speaker clustering)  │
                                  │
Phase 2: Post-process            │
  ├─ Merge consecutive utterances├─► merged segments
  └─ Map speaker IDs to names    │
                                  │
Phase 3: LLM cleanup (optional)  │
  └─ Bedrock Claude              └─► transcript.md
```

## Performance

Benchmarked on a 4h14m, 9-speaker Chinese meeting recording:

| Phase | GPU (L40S) | CPU |
|-------|-----------|-----|
| Model load | 14s | ~30s |
| Transcription | 169s | ~30-60 min |
| Clustering (patched) | ~10s | ~2-5 min |
| LLM cleanup (17 chunks) | ~35 min | ~35 min |

**Without the clustering patch**, speaker clustering on long audio takes 10+ hours.
The patch replaces O(N^3) `scipy.linalg.eigh` with O(N^2·k) `scipy.sparse.linalg.eigsh`.

## License

MIT
