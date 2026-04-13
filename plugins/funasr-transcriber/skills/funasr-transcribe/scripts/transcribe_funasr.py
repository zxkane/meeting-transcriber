#!/usr/bin/env python3
"""FunASR: Multi-language meeting transcription with speaker diarization + LLM cleanup.

Three-phase pipeline optimized for multi-speaker meetings:
  Phase 1: FunASR ASR + speaker diarization (+ optional hotword biasing)
  Phase 2: Post-processing (merge consecutive utterances + speaker mapping)
  Phase 3: LLM cleanup via Bedrock Claude (optional)

Language presets:
  zh        — SeACo-Paraformer (best Chinese, CER 1.95%, hotword support)
  zh-basic  — Paraformer-large (Chinese, no hotword, lighter)
  en        — Paraformer-en (English)
  auto      — SenseVoiceSmall (auto-detect: zh/en/ja/ko/yue)
  whisper   — Whisper-large-v3-turbo (99 languages)

Supports GPU (recommended) and CPU (slower but functional).
First run auto-downloads models from ModelScope.

Usage:
  # Chinese meeting with hotwords (names, terms)
  python3 transcribe_funasr.py meeting.wav --lang zh --num-speakers 9 \\
      --hotwords "张三 李四 ClawCon Rebase"

  # Hotwords from file (one per line)
  python3 transcribe_funasr.py meeting.wav --lang zh --hotwords hotwords.txt

  # English meeting
  python3 transcribe_funasr.py meeting.wav --lang en --num-speakers 4

  # Auto-detect language
  python3 transcribe_funasr.py meeting.wav --lang auto --num-speakers 6

  # Whisper for any language
  python3 transcribe_funasr.py meeting.wav --lang whisper --num-speakers 4

  # With real speaker names
  python3 transcribe_funasr.py meeting.wav --speakers "Alice,Bob,Carol"

  # CPU mode
  python3 transcribe_funasr.py meeting.wav --lang zh --device cpu

  # Raw transcription only (no LLM)
  python3 transcribe_funasr.py meeting.wav --skip-llm

  # Resume interrupted LLM cleanup
  python3 transcribe_funasr.py meeting.wav --skip-transcribe

  # Speaker context JSON to help LLM identify speakers
  python3 transcribe_funasr.py meeting.wav --speaker-context context.json
"""

import json
import sys
import time
import argparse
from pathlib import Path


# ──────────────────────────────────────────────
# Language-specific model presets
# ──────────────────────────────────────────────

MODEL_PRESETS = {
    "zh": {
        "label": "Chinese (SeACo-Paraformer, hotword-enabled)",
        # SeACo-Paraformer: hotword-customizable Paraformer variant
        # Paper: "SeACo-Paraformer: A Non-Autoregressive ASR System with
        # Flexible and Effective Hotword Customization Ability"
        "asr": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "spk": "iic/speech_campplus_sv_zh-cn_16k-common",
        "hotword_support": True,
    },
    "zh-basic": {
        "label": "Chinese (Paraformer-large, no hotword)",
        "asr": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "spk": "iic/speech_campplus_sv_zh-cn_16k-common",
        "hotword_support": False,
    },
    "en": {
        "label": "English (Paraformer-en)",
        "asr": "iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "spk": "iic/speech_campplus_sv_zh-cn_16k-common",
        "hotword_support": False,
    },
    "auto": {
        "label": "Auto-detect (SenseVoiceSmall: zh/en/ja/ko/yue)",
        "asr": "iic/SenseVoiceSmall",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": None,  # SenseVoiceSmall includes punctuation
        "spk": "iic/speech_campplus_sv_zh-cn_16k-common",
        "hotword_support": False,
    },
    "whisper": {
        "label": "Multilingual (Whisper-large-v3-turbo, 99 languages)",
        "asr": "iic/Whisper-large-v3-turbo",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": None,  # Whisper includes punctuation
        "spk": "iic/speech_campplus_sv_zh-cn_16k-common",
        "hotword_support": False,
    },
}

SUPPORTED_LANGS = list(MODEL_PRESETS.keys())


# ──────────────────────────────────────────────
# Phase 1: FunASR transcription
# ──────────────────────────────────────────────

def transcribe_with_funasr(audio_path: str, lang: str = "zh",
                           num_speakers: int = None,
                           device: str = "cuda:0",
                           batch_size_s: int = 300,
                           hotwords: str = None) -> list:
    """Run FunASR for ASR and speaker diarization.

    Loads language-specific models and runs the full pipeline:
    VAD -> ASR -> punctuation -> speaker embedding -> clustering.

    For 'zh' mode with hotwords, uses SeACo-Paraformer which biases
    recognition toward provided hotwords (names, terms, etc.).
    """
    from funasr import AutoModel

    preset = MODEL_PRESETS[lang]
    print(f"[Phase 1] Language: {preset['label']}")
    print(f"  Loading models (device={device})...")
    t0 = time.time()

    model_kwargs = {
        "model": preset["asr"],
        "vad_model": preset["vad"],
        "vad_kwargs": {"max_single_segment_time": 60000},
        "spk_model": preset["spk"],
        "device": device,
        "disable_update": True,
    }
    if preset.get("punc"):
        model_kwargs["punc_model"] = preset["punc"]

    # SeACo-Paraformer: pass hotwords at model init
    if hotwords and preset.get("hotword_support"):
        model_kwargs["hotword"] = hotwords
        print(f"  Hotwords: {hotwords[:100]}{'...' if len(hotwords) > 100 else ''}")

    model = AutoModel(**model_kwargs)
    print(f"  Models loaded: {time.time() - t0:.1f}s")

    generate_kwargs = {"input": audio_path, "batch_size_s": batch_size_s}
    if num_speakers:
        generate_kwargs["preset_spk_num"] = num_speakers

    print(f"  Transcribing: {audio_path} (speakers={num_speakers}, batch={batch_size_s}s)")
    t1 = time.time()
    res = model.generate(**generate_kwargs)
    elapsed = time.time() - t1
    print(f"  Transcription done: {elapsed:.1f}s")

    # Parse results — handle both composite and SenseVoice/Whisper output formats
    transcript = []
    for result in res:
        if "sentence_info" in result:
            for sent in result["sentence_info"]:
                transcript.append({
                    "speaker": int(sent.get("spk", 0)),
                    "start_ms": sent["start"],
                    "end_ms": sent["end"],
                    "text": sent.get("text", sent.get("sentence", "")),
                })
        elif "text" in result and "timestamp" not in result:
            # Fallback: plain text without sentence-level info
            transcript.append({
                "speaker": 0,
                "start_ms": 0,
                "end_ms": 0,
                "text": result["text"],
            })

    speakers = sorted(set(s["speaker"] for s in transcript))
    print(f"  Sentences: {len(transcript)}, speakers detected: {len(speakers)}")
    for spk in speakers:
        count = sum(1 for s in transcript if s["speaker"] == spk)
        print(f"    spk {spk}: {count}")

    return transcript


# ──────────────────────────────────────────────
# Phase 2: Post-processing
# ──────────────────────────────────────────────

def merge_consecutive(transcript: list, gap_ms: int = 2000) -> list:
    """Merge consecutive sentences from the same speaker within gap_ms."""
    if not transcript:
        return []
    merged = []
    cur = dict(transcript[0])
    for sent in transcript[1:]:
        if sent["speaker"] == cur["speaker"] and (sent["start_ms"] - cur["end_ms"]) < gap_ms:
            cur["end_ms"] = sent["end_ms"]
            cur["text"] += sent["text"]
        else:
            merged.append(cur)
            cur = dict(sent)
    merged.append(cur)
    return merged


def build_speaker_map(transcript: list, speakers: list = None) -> dict:
    """Build speaker ID -> display name mapping.

    If speaker names are provided, assign by first-appearance order.
    Otherwise use generic labels.
    """
    seen_ids = []
    for s in transcript:
        if s["speaker"] not in seen_ids:
            seen_ids.append(s["speaker"])

    if speakers:
        mapping = {}
        for i, spk_id in enumerate(seen_ids):
            mapping[spk_id] = speakers[i] if i < len(speakers) else f"Speaker {spk_id + 1}"
        return mapping

    return {spk_id: f"Speaker {spk_id + 1}" for spk_id in seen_ids}


# ──────────────────────────────────────────────
# Phase 3: LLM cleanup (Bedrock Claude)
# ──────────────────────────────────────────────

def format_time_ms(ms: int) -> str:
    s = ms / 1000
    return f"{int(s // 3600):02d}:{int((s % 3600) // 60):02d}:{int(s % 60):02d}"


def format_chunk(chunk: list, speaker_map: dict) -> str:
    lines = []
    for item in chunk:
        name = speaker_map.get(item["speaker"], f"Unknown{item['speaker']}")
        lines.append(f"[{format_time_ms(item['start_ms'])}] {name}: {item['text']}")
    return "\n".join(lines)


def chunk_by_duration(items: list, duration_ms: int = 900000) -> list:
    """Split items into time-based chunks (default 15 min)."""
    if not items:
        return []
    chunks, cur, start = [], [], items[0]["start_ms"]
    for item in items:
        if item["start_ms"] - start >= duration_ms and cur:
            chunks.append(cur)
            cur, start = [], item["start_ms"]
        cur.append(item)
    if cur:
        chunks.append(cur)
    return chunks


DEFAULT_SYSTEM_PROMPT = """You are a professional meeting transcript editor.

Rules:
1. Remove filler words (um, uh, like, you know, 嗯, 啊, 那个, 就是说, 对对对, etc.)
2. Fix grammar errors; make sentences more readable and polished
3. Merge stuttered/repeated expressions into fluent sentences
4. Fix obvious ASR errors based on context
5. Preserve original meaning — do not add content not in the original
6. Keep speaker labels and timestamps unchanged
7. Preserve technical terms and proper nouns
8. Output cleaned text only, format: [timestamp] Name: content"""


def cleanup_with_bedrock(chunk_text: str, chunk_idx: int, total: int,
                         system_prompt: str, model_id: str, region: str):
    """Call Bedrock Claude to clean one chunk."""
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    client = boto3.client(
        "bedrock-runtime", region_name=region,
        config=Config(read_timeout=300, connect_timeout=10, retries={"max_attempts": 3}),
    )
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8192,
        "system": system_prompt,
        "messages": [{"role": "user",
                       "content": f"Clean the following meeting transcript segment "
                                  f"({chunk_idx+1}/{total}):\n\n{chunk_text}"}],
    })
    for attempt in range(3):
        try:
            resp = client.invoke_model(modelId=model_id, contentType="application/json",
                                       accept="application/json", body=body)
            return json.loads(resp["body"].read())["content"][0]["text"]
        except ClientError as e:
            if "ThrottlingException" in str(e) and attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def run_llm_cleanup(merged: list, speaker_map: dict, model_id: str, region: str,
                    speaker_context: dict = None, cache_dir: Path = None) -> list:
    """Chunk merged transcript and clean each via LLM. Supports resume via cache_dir."""
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if speaker_context:
        extra = "\n\nMeeting participant context (use to fix ASR errors and identify speakers):\n"
        for name, info in speaker_context.items():
            extra += f"- {name}: {info}\n"
        system_prompt += extra

    chunks = chunk_by_duration(merged)
    cleaned = []
    if cache_dir:
        cache_dir.mkdir(exist_ok=True)

    print(f"  LLM cleanup: {len(chunks)} chunks, model: {model_id}")
    for i, chunk in enumerate(chunks):
        cache_file = cache_dir / f"chunk_{i:03d}.txt" if cache_dir else None

        if cache_file and cache_file.exists():
            cleaned.append(cache_file.read_text(encoding="utf-8"))
            print(f"  chunk {i+1}/{len(chunks)} (cached)")
            continue

        chunk_text = format_chunk(chunk, speaker_map)
        try:
            result = cleanup_with_bedrock(chunk_text, i, len(chunks),
                                          system_prompt, model_id, region)
            cleaned.append(result)
            if cache_file:
                cache_file.write_text(result, encoding="utf-8")
        except Exception as e:
            print(f"  chunk {i+1} cleanup failed: {e}, using raw text")
            cleaned.append(chunk_text)

        print(f"  chunk {i+1}/{len(chunks)} done")
        time.sleep(1)

    return cleaned


# ──────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────

def assemble_markdown(cleaned_parts: list, metadata: dict) -> str:
    speakers_list = "\n".join(f"- {name}" for name in metadata.get("speakers", []))
    duration_s = metadata.get("duration_ms", 0) / 1000
    h, m = int(duration_s // 3600), int((duration_s % 3600) // 60)

    md = f"""# Meeting Transcript

## Info

| Field | Value |
|-------|-------|
| **File** | {metadata.get('filename', '')} |
| **Duration** | {h}h{m}m |
| **Speakers** | {metadata.get('num_speakers', '?')} |
| **Language** | {metadata.get('language', 'N/A')} |
| **ASR Engine** | {metadata.get('asr_engine', 'FunASR')} |

## Speaker List

{speakers_list}

---

## Transcript

"""
    md += "\n\n".join(cleaned_parts)
    return md


# ──────────────────────────────────────────────
# Hotword helpers
# ──────────────────────────────────────────────

def resolve_hotwords(hotwords_arg: str) -> str:
    """Resolve --hotwords argument.

    If it's a .txt file path, return the path (FunASR reads it directly).
    Otherwise treat as space-separated hotword string.
    """
    if hotwords_arg and hotwords_arg.endswith(".txt") and Path(hotwords_arg).exists():
        count = sum(1 for line in open(hotwords_arg) if line.strip())
        print(f"  Hotwords file: {hotwords_arg} ({count} words)")
        return hotwords_arg
    return hotwords_arg


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="FunASR multi-speaker meeting transcription with speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Supported languages: {', '.join(SUPPORTED_LANGS)}")
    p.add_argument("audio_file", help="Audio file (WAV recommended; M4A/MP3 also supported)")
    p.add_argument("--lang", default="zh", choices=SUPPORTED_LANGS,
                   help="Language preset (default: zh). "
                        "zh=SeACo-Paraformer(hotword), zh-basic=Paraformer-large, "
                        "en=Paraformer-en, auto=SenseVoiceSmall, whisper=Whisper-v3-turbo")
    p.add_argument("--num-speakers", type=int, default=None,
                   help="Number of speakers in the meeting (improves diarization accuracy)")
    p.add_argument("--speakers", type=str, default=None,
                   help="Comma-separated speaker names, e.g. 'Alice,Bob,Carol'")
    p.add_argument("--hotwords", type=str, default=None,
                   help="Hotwords to bias ASR (space-separated string or .txt file). "
                        "Use for participant names, technical terms, project names. "
                        "Only effective with --lang zh (SeACo-Paraformer)")
    p.add_argument("--device", default=None,
                   help="Device: cuda:0 / cpu (auto-detected by default)")
    p.add_argument("--batch-size", type=int, default=300,
                   help="Batch size in seconds. Use 60 for CPU, 100 if GPU OOM (default: 300)")
    p.add_argument("--output", default=None,
                   help="Output file (default: <stem>-transcript.md)")
    p.add_argument("--bedrock-model", default="us.anthropic.claude-sonnet-4-6",
                   help="Bedrock model ID for LLM cleanup")
    p.add_argument("--bedrock-region", default="us-west-2", help="Bedrock region")
    p.add_argument("--speaker-context", type=str, default=None,
                   help="JSON file with per-speaker context to help LLM identify speakers")
    p.add_argument("--skip-transcribe", action="store_true",
                   help="Skip ASR, load from *_raw_transcript.json")
    p.add_argument("--skip-llm", action="store_true", help="Skip LLM cleanup")
    p.add_argument("--clean-cache", action="store_true",
                   help="Delete LLM chunk cache after completion")
    args = p.parse_args()

    audio_path = Path(args.audio_file)
    raw_json = Path(f"{audio_path.stem}_raw_transcript.json")
    output_path = Path(args.output) if args.output else Path(f"{audio_path.stem}-transcript.md")

    # Auto-detect device
    if args.device is None:
        try:
            import torch
            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
        print(f"Device: {args.device}")
        if args.device == "cpu" and args.batch_size > 60:
            print(f"  CPU mode: batch_size adjusted to 60 (was {args.batch_size})")
            args.batch_size = 60

    # Parse speaker names
    speaker_names = [s.strip() for s in args.speakers.split(",")] if args.speakers else None
    num_speakers = args.num_speakers or (len(speaker_names) if speaker_names else None)

    # Resolve hotwords
    hotwords = resolve_hotwords(args.hotwords) if args.hotwords else None
    preset = MODEL_PRESETS[args.lang]
    if hotwords and not preset.get("hotword_support"):
        print(f"  Warning: --hotwords ignored for --lang {args.lang} "
              f"(only supported with --lang zh / SeACo-Paraformer)")
        hotwords = None

    # ── Phase 1: Transcribe ──
    if args.skip_transcribe:
        if not raw_json.exists():
            print(f"Error: {raw_json} not found. Run full transcription first.")
            sys.exit(1)
        with open(raw_json, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        print(f"Loaded {len(transcript)} sentences from {raw_json}")
    else:
        if not audio_path.exists():
            print(f"Error: {audio_path} not found")
            sys.exit(1)
        transcript = transcribe_with_funasr(str(audio_path), args.lang, num_speakers,
                                            args.device, args.batch_size, hotwords)
        with open(raw_json, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        print(f"Raw transcript saved: {raw_json}")

    if not transcript:
        print("Error: empty transcript")
        sys.exit(1)

    # ── Phase 2: Post-process ──
    merged = merge_consecutive(transcript)
    speaker_map = build_speaker_map(transcript, speaker_names)
    print(f"[Phase 2] Merged: {len(transcript)} sentences -> {len(merged)} segments")

    # ── Phase 3: LLM cleanup ──
    speaker_context = None
    if args.speaker_context:
        with open(args.speaker_context, "r", encoding="utf-8") as f:
            speaker_context = json.load(f)

    if args.skip_llm:
        chunks = chunk_by_duration(merged)
        cleaned_parts = [format_chunk(chunk, speaker_map) for chunk in chunks]
    else:
        cache_dir = Path(f"{audio_path.stem}_llm_cache")
        print("[Phase 3] LLM cleanup...")
        cleaned_parts = run_llm_cleanup(merged, speaker_map, args.bedrock_model,
                                        args.bedrock_region, speaker_context, cache_dir)
        if args.clean_cache and cache_dir.exists():
            for f in cache_dir.glob("chunk_*.txt"):
                f.unlink()
            cache_dir.rmdir()
            print("  LLM cache cleaned")

    # ── Output ──
    duration_ms = transcript[-1]["end_ms"] - transcript[0]["start_ms"]
    actual_speakers = sorted(set(s["speaker"] for s in transcript))
    md = assemble_markdown(cleaned_parts, {
        "filename": audio_path.name,
        "duration_ms": duration_ms,
        "num_speakers": len(actual_speakers),
        "language": preset["label"],
        "asr_engine": f"FunASR ({preset['asr'].split('/')[-1]})",
        "speakers": [speaker_map.get(s, f"Speaker {s+1}") for s in actual_speakers],
    })
    output_path.write_text(md, encoding="utf-8")
    print(f"\nDone: {output_path} ({len(merged)} segments, "
          f"{len(actual_speakers)} speakers, {format_time_ms(duration_ms)})")


if __name__ == "__main__":
    main()
