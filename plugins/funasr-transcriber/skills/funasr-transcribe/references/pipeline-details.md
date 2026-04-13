# FunASR Meeting Transcription Pipeline — Technical Details

## Architecture

```
Audio File (.m4a/.mp3/.wav)
  │
  ├─ [ffmpeg] ──► 16kHz mono WAV
  │
  ├─ [Phase 1: FunASR] ──► raw_transcript.json
  │   ├─ FSMN-VAD: segment speech vs silence
  │   ├─ ASR model (language-dependent, see below)
  │   ├─ (Optional) Hotword biasing (SeACo-Paraformer only)
  │   ├─ Punctuation restoration (model-dependent)
  │   └─ CAM++: speaker embeddings → spectral clustering
  │
  ├─ [Phase 2: Post-process]
  │   ├─ Merge consecutive same-speaker utterances (<2s gap)
  │   └─ Map speaker IDs to names (if provided)
  │
  └─ [Phase 3: LLM cleanup] ──► transcript.md
      ├─ Remove fillers (um, uh, 嗯, 啊, etc.)
      ├─ Fix ASR errors (homophones, context-based)
      ├─ Polish grammar while preserving meaning
      └─ (Optional) Identify merged speakers via context
```

## Language Presets & Models

### `--lang zh` (Chinese, default) — SeACo-Paraformer with hotword support

| Component | Model ID | Params | Role |
|-----------|----------|--------|------|
| ASR | `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | 220M | Chinese ASR (CER 1.95%), hotword-customizable |
| VAD | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | 0.4M | Voice activity detection |
| Punctuation | `iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` | 290M | Punctuation restoration |
| Speaker | `iic/speech_campplus_sv_zh-cn_16k-common` | 7.2M | Speaker diarization |

SeACo-Paraformer accepts a `--hotwords` parameter (space-separated string or .txt file)
to bias recognition toward specific terms. See [Hotword Biasing](#hotword-biasing) below.

### `--lang zh-basic` (Chinese, no hotword)

Same as `zh` but uses the base Paraformer-large without hotword support.
Use when hotword biasing is unnecessary or causing issues with English terms.

| Component | Model ID | Params | Role |
|-----------|----------|--------|------|
| ASR | `iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | 220M | Chinese ASR (CER 1.95%) |

### `--lang en` (English)

| Component | Model ID | Params | Role |
|-----------|----------|--------|------|
| ASR | `iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020` | 220M | English ASR |

### `--lang auto` (Auto-detect: zh/en/ja/ko/yue)

| Component | Model ID | Params | Role |
|-----------|----------|--------|------|
| ASR | `iic/SenseVoiceSmall` | 234M | Multi-language ASR with auto language detection |

SenseVoiceSmall includes built-in punctuation and supports emotion detection.

### `--lang whisper` (Multilingual, 99 languages)

| Component | Model ID | Params | Role |
|-----------|----------|--------|------|
| ASR | `iic/Whisper-large-v3-turbo` | 809M | OpenAI Whisper via FunASR, broadest language coverage |

All presets share the same VAD (`fsmn-vad`) and speaker diarization (`cam++`) models.
Models are auto-downloaded from ModelScope on first run.

## Hotword Biasing

SeACo-Paraformer (`--lang zh`) supports hotword customization to improve recognition
of specific terms — particularly useful for participant names, project names, and
domain-specific jargon in meetings.

### How to provide hotwords

```bash
# Space-separated string
python3 transcribe_funasr.py meeting.wav --lang zh --hotwords "张三 李四 ClawCon Rebase"

# Text file (one word per line)
python3 transcribe_funasr.py meeting.wav --lang zh --hotwords hotwords.txt
```

### What to include in hotwords

For meeting transcription, a good hotwords file includes:
- **Participant names** (full names in the meeting's language)
- **Project / product names** (internal codenames, product brands)
- **Domain-specific Chinese terms** (technical jargon, acronyms in Chinese)
- **Organization names** (company, team, department names)

### Effectiveness — empirical results

Tested on a 4h14m, 9-speaker Chinese meeting with 27 hotwords:

| Term | Without hotwords | With hotwords | Change |
|------|-----------------|---------------|--------|
| 龙虾 (lobechat) | 28 | 42 | **+50%** |
| 高琦 (person name) | 0 | 7 | **0 → 7** |
| 搬瓦工 (BandwagonHost) | 0 | 1 | **0 → 1** |
| 谢锐 (person name) | 0 | 1 | improved |
| 鲲鹏 (org name) | 6 | 7 | slight improvement |
| Rebase (English) | 5 | 0 | **regression** |
| Tailwind (English) | 3 | 1 | **regression** |

**Key findings:**
1. **Chinese terms benefit significantly** — names, brands, and Chinese jargon
   see clear improvement (龙虾 +50%, 高琦 from zero)
2. **English terms may regress** — SeACo's hotword biasing operates on Chinese
   token vocabulary; English loanwords can be disrupted
3. **Person names have limited uplift** — meeting participants rarely say full
   names aloud; hotwords help only when names do appear in speech

**Recommendation:** Include Chinese terms and names in hotwords. For English
technical terms, rely on Phase 3 LLM cleanup rather than hotword biasing.
If English term accuracy is critical and hotword biasing causes regressions,
use `--lang zh-basic` instead.

## Performance Benchmarks

Tested on a 4h14m, 9-speaker Chinese meeting recording (GPU: L40S 46GB):

| Metric | Paraformer-large | SeACo + Hotword | Notes |
|--------|-----------------|-----------------|-------|
| Model load | 14s | 14s (422s first run, downloading 944MB model) | Cached after first run |
| Transcription | 169s | 168s | Virtually identical |
| Raw sentences | 6672 | 6725 | Comparable |
| Merged segments | 1695 | 1724 | Comparable |
| Speakers detected | 7 (of 9) | 7 (of 9) | Same diarization result |

| Metric | GPU (L40S 46GB) | CPU (estimated) |
|--------|-----------------|-----------------|
| Model load | 14s | ~30s |
| Transcription | 169s | ~30-60 min |
| Speaker clustering | ~10s (patched) | ~2-5 min (patched) |
| LLM cleanup (17 chunks) | ~35 min | ~35 min (network-bound) |
| Total | ~38 min | ~70-100 min |

**Without the clustering patch**, the original `scipy.linalg.eigh()` on the full Laplacian
matrix was O(N^3) and took **10+ hours** on this recording. The patch reduces it to O(N^2*k)
via `scipy.sparse.linalg.eigsh()`.

## Clustering Patch (Critical for Long Meetings)

FunASR's `SpectralCluster.get_spec_embs()` uses `scipy.linalg.eigh(L)` which computes
ALL eigenvalues of the NxN Laplacian. For a 4-hour recording, N can be 6000+, making
this O(N^3) operation take hours.

The patch (`scripts/patch_clustering.py`) replaces this with:
- `scipy.sparse.linalg.eigsh(L_sparse, k=num_speakers, which='SM')` — only computes
  the k smallest eigenvalues needed, reducing complexity to O(N^2 * k)
- Vectorized `p_pruning()` — replaces Python loop with numpy broadcasting

**Always run the patch before processing meetings longer than ~1 hour.**

## Speaker Diarization Limitations

FunASR's CAM++ speaker diarization may merge acoustically similar speakers into one ID.
In the tested 9-person meeting, only 7 unique IDs were detected (two pairs merged).

Workarounds:
1. **Provide `--num-speakers N`** to hint expected count (uses `preset_spk_num`)
2. **Post-hoc keyword matching**: use reference documents (meeting agendas, attendee notes)
   to identify which speaker ID maps to which person
3. **LLM-assisted splitting**: provide `--speaker-context` with per-person keywords;
   the LLM can then split merged speakers when context is clear (~73% success rate)

## Supporting Files for Better Results

Prepare these files before transcription for best results:

| File | Used in | Purpose |
|------|---------|---------|
| `hotwords.txt` | Phase 1 (`--hotwords`) | Bias ASR toward names and terms |
| `speaker-context.json` | Phase 3 (`--speaker-context`) | Help LLM identify and split speakers |
| Meeting agenda | Manual reference | Identify meeting phases for post-analysis |
| Attendee list | Build hotwords + speaker names | Map speaker IDs to real names |

### Example: preparing supporting files from a meeting invite

```bash
# 1. Create hotwords.txt from attendee list and agenda
cat > hotwords.txt << 'EOF'
Alice
Bob
Carol
ProjectAlpha
Sprint Review
Q2 OKR
EOF

# 2. Create speaker-context.json from attendee roles
cat > speaker-context.json << 'EOF'
{
  "Alice": "Engineering manager, discusses sprint velocity and tech debt",
  "Bob": "Product manager, presents roadmap and customer feedback",
  "Carol": "Designer, shows mockups, mentions Figma and user testing"
}
EOF

# 3. Run with both
python3 transcribe_funasr.py meeting.wav \
  --lang zh --num-speakers 3 \
  --speakers "Alice,Bob,Carol" \
  --hotwords hotwords.txt \
  --speaker-context speaker-context.json
```

## Audio Preprocessing

FunASR works best with 16kHz mono audio. **FLAC is recommended** over WAV — lossless
quality at ~50% the file size, and FunASR reads it natively via soundfile.

```bash
# Recommended: FLAC (lossless, compact)
ffmpeg -i recording.m4a -ar 16000 -ac 1 -sample_fmt s16 meeting.flac

# Alternative: WAV (lossless, larger)
ffmpeg -i recording.m4a -ar 16000 -ac 1 meeting.wav
```

**Important:** Use `-sample_fmt s16` when converting to FLAC — without it, ffmpeg
may output 24-bit samples (s32/24bit) which doubles the file size with no ASR benefit.

### Format comparison (4h14m meeting)

| Format | Size | Quality | FunASR support |
|--------|------|---------|---------------|
| **FLAC (16kHz mono s16)** | **219MB** | Lossless | Native (soundfile) |
| WAV (16kHz mono) | 465MB | Lossless | Native (soundfile) |
| Opus (32kbps) | 54MB | Lossy | Native (soundfile) |
| M4A/AAC (original 48kHz) | 173MB | Source | Via librosa |
| M4A/AAC (16kHz 32kbps) | 60MB | Lossy | Via librosa |

FunASR accepts all common audio formats. FLAC offers the best trade-off: lossless
quality, reasonable size, and native reader support without librosa fallback.

For long recordings, do NOT split the audio — FunASR handles arbitrarily long files
and splitting breaks speaker consistency across segments.

## Resume / Checkpoint Support

The pipeline supports resuming interrupted runs:
- **Phase 1 output**: `<stem>_raw_transcript.json` — use `--skip-transcribe` to skip ASR
- **Phase 3 cache**: `<stem>_llm_cache/chunk_NNN.txt` — already-cleaned chunks are reused
  (kept by default; add `--clean-cache` to delete after completion)

## Speaker Context JSON Format

The `--speaker-context` file helps the LLM identify speakers and fix ASR errors:

```json
{
  "Alice": "Discussed Q1 revenue targets, mentioned Chicago office relocation",
  "Bob": "Presented the new CI/CD pipeline, uses Terraform and ArgoCD",
  "Carol": "HR updates, mentioned hiring freeze and new PTO policy"
}
```

The context is injected into the LLM system prompt for each cleanup chunk.
