#!/usr/bin/env python3
"""Tests for speaker verification pipeline.

Covers: llm_utils, verify_speakers, and speaker-related functions
in transcribe. All LLM calls are mocked.
"""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from llm_utils import detect_llm_provider, is_retryable, call_llm
import speaker_gender as sg
import transcribe as tf
import verify_speakers as vs


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

def make_segment(speaker, start_ms, end_ms, text):
    return {"speaker": speaker, "start_ms": start_ms, "end_ms": end_ms, "text": text}


@pytest.fixture
def two_speaker_transcript():
    return [
        make_segment(0, 0, 5000, "大家好，欢迎来到我们的节目"),
        make_segment(0, 5000, 15000, "今天我们请到了一位特别的嘉宾"),
        make_segment(1, 15000, 30000, "谢谢主持人，很高兴来到这里"),
        make_segment(0, 30000, 45000, "能不能先给大家介绍一下你自己"),
        make_segment(1, 45000, 90000, "好的，我是做人工智能研究的"),
        make_segment(0, 90000, 120000, "那我们今天就来聊聊AI的发展"),
    ]


@pytest.fixture
def four_speaker_transcript():
    return [
        make_segment(0, 0, 10000, "Let's start the meeting"),
        make_segment(1, 10000, 25000, "I'll present the engineering update"),
        make_segment(2, 25000, 40000, "Design review is ready for Q3"),
        make_segment(3, 40000, 55000, "Budget looks good this quarter"),
        make_segment(0, 55000, 70000, "Great, any questions?"),
        make_segment(1, 70000, 90000, "We need more time for the migration"),
    ]


@pytest.fixture
def speaker_context_podcast():
    return {
        "关羽": "Host, asks questions, introduces topics",
        "张飞": "Guest, AI researcher, shares expertise",
    }


@pytest.fixture
def speaker_context_meeting():
    return {
        "Alice": "Team lead, runs the meeting",
        "Bob": "Engineer, gives technical updates",
        "Carol": "Designer, presents design work",
        "Dave": "Finance, reports on budget",
    }


# ──────────────────────────────────────────────
# llm_utils: detect_llm_provider
# ──────────────────────────────────────────────

class TestDetectLLMProvider:
    def test_bedrock_arn(self):
        assert detect_llm_provider("arn:aws:bedrock:us-west-2:123:inference-profile/abc") == "bedrock"

    def test_bedrock_cross_region(self):
        assert detect_llm_provider("us.anthropic.claude-sonnet-4-6") == "bedrock"

    def test_anthropic_bare_claude(self):
        assert detect_llm_provider("claude-sonnet-4-6") == "anthropic"

    def test_anthropic_claude_opus(self):
        assert detect_llm_provider("claude-opus-4-7") == "anthropic"

    def test_openai_gpt(self):
        assert detect_llm_provider("gpt-4o") == "openai"

    def test_openai_deepseek(self):
        assert detect_llm_provider("deepseek-chat") == "openai"

    def test_openai_default(self):
        assert detect_llm_provider("some-custom-model") == "openai"


# ──────────────────────────────────────────────
# llm_utils: is_retryable
# ──────────────────────────────────────────────

class TestIsRetryable:
    def test_rate_limit(self):
        assert is_retryable(Exception("rate_limit_exceeded"))

    def test_throttle(self):
        assert is_retryable(Exception("throttling exception"))

    def test_429(self):
        assert is_retryable(Exception("HTTP 429 Too Many Requests"))

    def test_529(self):
        assert is_retryable(Exception("529 overloaded"))

    def test_not_retryable(self):
        assert not is_retryable(Exception("invalid api key"))

    def test_not_retryable_generic(self):
        assert not is_retryable(Exception("something went wrong"))

    def test_read_timeout_retryable(self):
        assert is_retryable(Exception("Read timeout on endpoint URL: https://bedrock..."))

    def test_connect_timeout_retryable(self):
        assert is_retryable(Exception("Connect timeout on endpoint URL"))

    def test_timed_out_retryable(self):
        assert is_retryable(Exception("Request timed out after 300s"))


# ──────────────────────────────────────────────
# llm_utils: call_llm
# ──────────────────────────────────────────────

class TestCallLLM:
    @patch("llm_utils._call_anthropic", return_value="test response")
    def test_anthropic_routing(self, mock_call):
        result = call_llm("sys", "user", "claude-sonnet-4-6")
        assert result == "test response"
        mock_call.assert_called_once_with("sys", "user", "claude-sonnet-4-6")

    @patch("llm_utils._call_bedrock", return_value="bedrock response")
    def test_bedrock_routing(self, mock_call):
        result = call_llm("sys", "user", "us.anthropic.claude-sonnet-4-6", region="us-east-1")
        assert result == "bedrock response"
        mock_call.assert_called_once_with("sys", "user", "us.anthropic.claude-sonnet-4-6", "us-east-1")

    @patch("llm_utils._call_openai", return_value="openai response")
    def test_openai_routing(self, mock_call):
        result = call_llm("sys", "user", "gpt-4o")
        assert result == "openai response"
        mock_call.assert_called_once_with("sys", "user", "gpt-4o")

    @patch("llm_utils._call_anthropic")
    def test_retry_on_rate_limit(self, mock_call):
        mock_call.side_effect = [Exception("rate_limit_exceeded"), "ok"]
        result = call_llm("sys", "user", "claude-sonnet-4-6", max_retries=2)
        assert result == "ok"
        assert mock_call.call_count == 2

    @patch("llm_utils._call_anthropic")
    def test_no_retry_on_auth_error(self, mock_call):
        mock_call.side_effect = Exception("invalid api key")
        with pytest.raises(Exception, match="invalid api key"):
            call_llm("sys", "user", "claude-sonnet-4-6", max_retries=3)
        assert mock_call.call_count == 1

    @patch("llm_utils._call_bedrock", return_value="response")
    def test_default_region(self, mock_call):
        call_llm("sys", "user", "us.anthropic.claude-sonnet-4-6")
        mock_call.assert_called_once_with("sys", "user", "us.anthropic.claude-sonnet-4-6", "us-west-2")


# ──────────────────────────────────────────────
# verify_speakers: helpers
# ──────────────────────────────────────────────

class TestFormatTimeMs:
    def test_zero(self):
        assert vs.format_time_ms(0) == "00:00:00"

    def test_seconds(self):
        assert vs.format_time_ms(5000) == "00:00:05"

    def test_minutes(self):
        assert vs.format_time_ms(90000) == "00:01:30"

    def test_hours(self):
        assert vs.format_time_ms(3661000) == "01:01:01"


class TestBuildSpeakerMap:
    def test_two_speakers(self, two_speaker_transcript):
        m = vs.build_speaker_map(two_speaker_transcript, ["Host", "Guest"])
        assert m == {0: "Host", 1: "Guest"}

    def test_more_names_than_speakers(self, two_speaker_transcript):
        m = vs.build_speaker_map(two_speaker_transcript, ["A", "B", "C"])
        assert m == {0: "A", 1: "B"}

    def test_fewer_names_than_speakers(self, two_speaker_transcript):
        m = vs.build_speaker_map(two_speaker_transcript, ["OnlyOne"])
        assert m[0] == "OnlyOne"
        assert m[1] == "Speaker 2"

    def test_preserves_appearance_order(self):
        transcript = [
            make_segment(2, 0, 1000, "first"),
            make_segment(0, 1000, 2000, "second"),
        ]
        m = vs.build_speaker_map(transcript, ["Alpha", "Beta"])
        assert m == {2: "Alpha", 0: "Beta"}


class TestExtractEarlyTranscript:
    def test_respects_cutoff(self, two_speaker_transcript):
        speaker_map = {0: "Host", 1: "Guest"}
        result = vs.extract_early_transcript(two_speaker_transcript, 1, speaker_map)
        lines = result.strip().split("\n")
        for line in lines:
            assert "Host:" in line or "Guest:" in line

    def test_empty_transcript(self):
        assert vs.extract_early_transcript([], 5, {}) == ""

    def test_includes_timestamp(self, two_speaker_transcript):
        speaker_map = {0: "H", 1: "G"}
        result = vs.extract_early_transcript(two_speaker_transcript, 5, speaker_map)
        assert "[00:00:00]" in result


class TestComputeSpeakerStats:
    def test_counts_segments_and_chars(self, two_speaker_transcript):
        speaker_map = {0: "Host", 1: "Guest"}
        stats = vs.compute_speaker_stats(two_speaker_transcript, speaker_map)
        assert stats["Host"]["segments"] == 4
        assert stats["Guest"]["segments"] == 2
        assert stats["Host"]["chars"] > 0
        assert stats["Guest"]["chars"] > 0

    def test_with_minutes_cutoff(self, two_speaker_transcript):
        speaker_map = {0: "Host", 1: "Guest"}
        stats_full = vs.compute_speaker_stats(two_speaker_transcript, speaker_map)
        stats_limited = vs.compute_speaker_stats(two_speaker_transcript, speaker_map, minutes=1)
        total_full = sum(s["segments"] for s in stats_full.values())
        total_limited = sum(s["segments"] for s in stats_limited.values())
        assert total_limited <= total_full


# ──────────────────────────────────────────────
# verify_speakers: apply_swap
# ──────────────────────────────────────────────

class TestApplySwap:
    def test_swaps_two_speakers(self):
        transcript = [
            make_segment(0, 0, 1000, "hello"),
            make_segment(1, 1000, 2000, "world"),
            make_segment(0, 2000, 3000, "foo"),
        ]
        speaker_map = {0: "Alice", 1: "Bob"}
        vs.apply_swap(transcript, speaker_map, 0, 1)
        assert transcript[0]["speaker"] == 1
        assert transcript[1]["speaker"] == 0
        assert transcript[2]["speaker"] == 1
        assert speaker_map[0] == "Bob"
        assert speaker_map[1] == "Alice"

    def test_swap_is_reversible(self):
        transcript = [make_segment(0, 0, 1000, "a"), make_segment(1, 1000, 2000, "b")]
        speaker_map = {0: "X", 1: "Y"}
        vs.apply_swap(transcript, speaker_map, 0, 1)
        vs.apply_swap(transcript, speaker_map, 0, 1)
        assert transcript[0]["speaker"] == 0
        assert transcript[1]["speaker"] == 1
        assert speaker_map == {0: "X", 1: "Y"}


# ──────────────────────────────────────────────
# verify_speakers: apply_meeting_mapping
# ──────────────────────────────────────────────

class TestApplyMeetingMapping:
    def test_no_changes(self):
        transcript = [make_segment(0, 0, 1000, "hello")]
        speaker_map = {0: "Alice", 1: "Bob"}
        mapping = {"Alice": "Alice", "Bob": "Bob"}
        vs.apply_meeting_mapping(transcript, speaker_map, mapping)
        assert transcript[0]["speaker"] == 0

    def test_simple_swap(self):
        transcript = [
            make_segment(0, 0, 1000, "a"),
            make_segment(1, 1000, 2000, "b"),
        ]
        speaker_map = {0: "Alice", 1: "Bob"}
        mapping = {"Alice": "Bob", "Bob": "Alice"}
        vs.apply_meeting_mapping(transcript, speaker_map, mapping)
        assert transcript[0]["speaker"] == 1
        assert transcript[1]["speaker"] == 0

    def test_partial_mapping_ignored(self):
        transcript = [
            make_segment(0, 0, 1000, "a"),
            make_segment(1, 1000, 2000, "b"),
        ]
        speaker_map = {0: "Alice", 1: "Bob"}
        mapping = {"Alice": "Alice"}
        vs.apply_meeting_mapping(transcript, speaker_map, mapping)
        assert transcript[0]["speaker"] == 0
        assert transcript[1]["speaker"] == 1

    def test_three_way_rotation(self):
        transcript = [
            make_segment(0, 0, 1000, "a"),
            make_segment(1, 1000, 2000, "b"),
            make_segment(2, 2000, 3000, "c"),
        ]
        speaker_map = {0: "A", 1: "B", 2: "C"}
        mapping = {"A": "B", "B": "A", "C": "C"}
        vs.apply_meeting_mapping(transcript, speaker_map, mapping)
        assert transcript[0]["speaker"] == 1
        assert transcript[1]["speaker"] == 0
        assert transcript[2]["speaker"] == 2


# ──────────────────────────────────────────────
# verify_speakers: verify_podcast (mocked LLM)
# ──────────────────────────────────────────────

class TestVerifyPodcast:
    @patch("verify_speakers.call_llm")
    def test_correct_verdict(self, mock_llm):
        mock_llm.return_value = "VERDICT: CORRECT\nCONFIDENCE: HIGH\nEVIDENCE: labels match roles"
        speaker_map = {0: "Host", 1: "Guest"}
        ctx = {"Host": "asks questions", "Guest": "answers"}
        result = vs.verify_podcast("transcript text", speaker_map, ctx, "model", "us-west-2")
        assert result["verdict"] == "CORRECT"
        assert result["confidence"] == "HIGH"

    @patch("verify_speakers.call_llm")
    def test_swap_verdict(self, mock_llm):
        mock_llm.return_value = "VERDICT: SWAP\nCONFIDENCE: HIGH\nEVIDENCE: host is answering"
        speaker_map = {0: "Host", 1: "Guest"}
        ctx = {"Host": "asks questions", "Guest": "answers"}
        result = vs.verify_podcast("transcript text", speaker_map, ctx, "model", "us-west-2")
        assert result["verdict"] == "SWAP"

    @patch("verify_speakers.call_llm")
    def test_malformed_response(self, mock_llm):
        mock_llm.return_value = "I cannot determine the speaker roles from this text."
        speaker_map = {0: "Host", 1: "Guest"}
        ctx = {"Host": "asks questions", "Guest": "answers"}
        result = vs.verify_podcast("transcript text", speaker_map, ctx, "model", "us-west-2")
        assert result["verdict"] == "UNKNOWN"
        assert result["confidence"] == "LOW"


# ──────────────────────────────────────────────
# verify_speakers: verify_meeting (mocked LLM)
# ──────────────────────────────────────────────

class TestVerifyMeeting:
    @patch("verify_speakers.call_llm")
    def test_correct_mapping(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "correct": True,
            "confidence": "HIGH",
            "mapping": {"Alice": "Alice", "Bob": "Bob"},
            "evidence": {"Alice": "leads meeting", "Bob": "gives updates"},
        })
        speaker_map = {0: "Alice", 1: "Bob"}
        ctx = {"Alice": "team lead", "Bob": "engineer"}
        result = vs.verify_meeting("text", speaker_map, ctx, "model", "us-west-2")
        assert result["correct"] is True
        assert result["mapping"]["Alice"] == "Alice"

    @patch("verify_speakers.call_llm")
    def test_swapped_mapping(self, mock_llm):
        mock_llm.return_value = '```json\n{"correct": false, "confidence": "HIGH", "mapping": {"Alice": "Bob", "Bob": "Alice"}, "evidence": {}}\n```'
        speaker_map = {0: "Alice", 1: "Bob"}
        ctx = {"Alice": "team lead", "Bob": "engineer"}
        result = vs.verify_meeting("text", speaker_map, ctx, "model", "us-west-2")
        assert result["correct"] is False
        assert result["mapping"]["Alice"] == "Bob"

    @patch("verify_speakers.call_llm")
    def test_no_json_in_response(self, mock_llm):
        mock_llm.return_value = "I'm not sure about the speaker assignments."
        speaker_map = {0: "Alice", 1: "Bob"}
        ctx = {"Alice": "leader", "Bob": "engineer"}
        result = vs.verify_meeting("text", speaker_map, ctx, "model", "us-west-2")
        assert result["correct"] is None
        assert "raw" in result

    @patch("verify_speakers.call_llm")
    def test_invalid_json_in_response(self, mock_llm):
        mock_llm.return_value = '{"mapping": invalid json here}'
        speaker_map = {0: "Alice", 1: "Bob"}
        ctx = {"Alice": "leader", "Bob": "engineer"}
        result = vs.verify_meeting("text", speaker_map, ctx, "model", "us-west-2")
        assert result["correct"] is None


# ──────────────────────────────────────────────
# transcribe: speaker verification functions
# ──────────────────────────────────────────────


class TestTranscribeMergeConsecutive:
    def test_merges_same_speaker(self):
        transcript = [
            make_segment(0, 0, 1000, "hello "),
            make_segment(0, 1500, 2500, "world"),
        ]
        merged = tf.merge_consecutive(transcript)
        assert len(merged) == 1
        assert merged[0]["text"] == "hello world"

    def test_no_merge_different_speakers(self):
        transcript = [
            make_segment(0, 0, 1000, "hello"),
            make_segment(1, 1500, 2500, "world"),
        ]
        merged = tf.merge_consecutive(transcript)
        assert len(merged) == 2

    def test_no_merge_large_gap(self):
        transcript = [
            make_segment(0, 0, 1000, "hello"),
            make_segment(0, 5000, 6000, "world"),
        ]
        merged = tf.merge_consecutive(transcript)
        assert len(merged) == 2

    def test_empty_transcript(self):
        assert tf.merge_consecutive([]) == []

    def test_solo_podcast_keeps_timestamps(self):
        # Long single-speaker run (solo podcast) with small gaps should NOT
        # collapse into a single segment — we cap merged duration so
        # periodic timestamps remain in the output.
        transcript = [
            make_segment(0, i * 10000, i * 10000 + 8000, f"sentence {i} ")
            for i in range(60)  # 10 minutes of tightly-packed speech
        ]
        merged = tf.merge_consecutive(transcript, max_merge_ms=120000)
        assert len(merged) >= 5, (
            f"Expected periodic timestamps for solo podcast, got {len(merged)} segment(s)"
        )
        for seg in merged:
            assert seg["end_ms"] - seg["start_ms"] <= 120000 + 8000


class TestTranscribeBuildSpeakerMap:
    def test_with_names(self):
        transcript = [make_segment(0, 0, 1000, "a"), make_segment(1, 1000, 2000, "b")]
        m = tf.build_speaker_map(transcript, ["Alice", "Bob"])
        assert m == {0: "Alice", 1: "Bob"}

    def test_without_names(self):
        transcript = [make_segment(0, 0, 1000, "a"), make_segment(1, 1000, 2000, "b")]
        m = tf.build_speaker_map(transcript)
        assert m == {0: "Speaker 1", 1: "Speaker 2"}

    def test_solo_podcast_with_host_name(self):
        transcript = [make_segment(0, 0, 1000, "a"), make_segment(0, 2000, 3000, "b")]
        m = tf.build_speaker_map(transcript, ["李雷"])
        assert m == {0: "李雷"}


class TestExtractSpeakerNamesFromReference:
    def test_chinese_host_label(self):
        text = "节目信息\n主播：李雷\n更多内容..."
        assert tf.extract_speaker_names_from_reference(text) == ["李雷"]

    def test_host_and_guest(self):
        text = "主播：关羽\n嘉宾：张飞"
        assert tf.extract_speaker_names_from_reference(text) == ["关羽", "张飞"]

    def test_english_labels(self):
        text = "Host: Alice\nGuest - Bob"
        assert tf.extract_speaker_names_from_reference(text) == ["Alice", "Bob"]

    def test_no_labels_returns_empty(self):
        assert tf.extract_speaker_names_from_reference("random show notes text") == []

    def test_empty_input(self):
        assert tf.extract_speaker_names_from_reference("") == []
        assert tf.extract_speaker_names_from_reference(None) == []

    def test_stops_at_punctuation(self):
        text = "主播：李雷，资深宏观研究员"
        assert tf.extract_speaker_names_from_reference(text) == ["李雷"]

    def test_stops_at_chinese_period(self):
        text = "主播：李雷。本期节目我们将..."
        assert tf.extract_speaker_names_from_reference(text) == ["李雷"]

    def test_stops_at_ascii_period(self):
        text = "Host: Alice.senior analyst"
        assert tf.extract_speaker_names_from_reference(text) == ["Alice"]

    def test_stops_at_parenthesis(self):
        text = "Host: Alice (senior analyst)\nGuest: Bob"
        assert tf.extract_speaker_names_from_reference(text) == ["Alice", "Bob"]

    def test_dedup_across_host_and_guest(self):
        text = "主播：Alice\n嘉宾：Alice"
        assert tf.extract_speaker_names_from_reference(text) == ["Alice"]


class TestDetectAliasInSpeakers:
    def test_alias_detected_chinese(self):
        ref = "主播：张三（张三的播客）"
        result = tf.detect_alias_in_speakers(["张三的播客"], ref)
        assert result == [("张三的播客", "张三")]

    def test_real_name_accepted(self):
        ref = "主播：张三（张三的播客）"
        assert tf.detect_alias_in_speakers(["张三"], ref) == []

    def test_english_alias_detected(self):
        ref = "Host: Alice (AlicePodcast)"
        result = tf.detect_alias_in_speakers(["AlicePodcast"], ref)
        assert result == [("AlicePodcast", "Alice")]

    def test_ascii_parens_detected(self):
        ref = "Host: 张三(张三的播客)\n嘉宾: 李四"
        result = tf.detect_alias_in_speakers(["张三的播客", "李四"], ref)
        assert result == [("张三的播客", "张三")]

    def test_no_parens_in_reference(self):
        ref = "主播：张三\n嘉宾：李四"
        assert tf.detect_alias_in_speakers(["张三"], ref) == []

    def test_multiple_mismatches(self):
        ref = "主播：张三（张三的播客）\n嘉宾：李四（四哥）"
        result = tf.detect_alias_in_speakers(["张三的播客", "四哥"], ref)
        assert ("张三的播客", "张三") in result
        assert ("四哥", "李四") in result

    def test_partial_mismatch(self):
        ref = "主播：张三（张三的播客）\n嘉宾：李四"
        result = tf.detect_alias_in_speakers(["张三的播客", "李四"], ref)
        assert result == [("张三的播客", "张三")]

    def test_unknown_name_not_flagged(self):
        ref = "主播：张三（张三的播客）"
        assert tf.detect_alias_in_speakers(["王五"], ref) == []

    def test_empty_inputs(self):
        assert tf.detect_alias_in_speakers([], "主播：张三（张三的播客）") == []
        assert tf.detect_alias_in_speakers(["张三"], None) == []
        assert tf.detect_alias_in_speakers(["张三"], "") == []

    def test_reference_without_role_label(self):
        ref = "张三（张三的播客）是一位资深投资人"
        assert tf.detect_alias_in_speakers(["张三的播客"], ref) == []


class TestDetectMontageEnd:
    def test_no_montage_few_segments(self):
        transcript = [make_segment(0, 0, 30000, "long intro")]
        assert tf.detect_montage_end(transcript) == 0

    def test_no_montage_all_long(self):
        transcript = [
            make_segment(0, 0, 20000, "first long"),
            make_segment(1, 20000, 45000, "second long"),
        ]
        assert tf.detect_montage_end(transcript) == 0

    def test_detects_cold_open(self):
        """Short clips followed by a long segment = montage."""
        transcript = [
            make_segment(0, 0, 3000, "clip one"),
            make_segment(1, 3000, 6000, "clip two"),
            make_segment(0, 6000, 9000, "clip three"),
            make_segment(1, 9000, 12000, "clip four"),
            make_segment(0, 12000, 30000, "welcome to the show, real intro starts"),
        ]
        assert tf.detect_montage_end(transcript) == 4

    def test_no_montage_mixed_lengths(self):
        """One long segment among early ones breaks the montage pattern."""
        transcript = [
            make_segment(0, 0, 3000, "short"),
            make_segment(1, 3000, 20000, "long early on"),
            make_segment(0, 20000, 23000, "short again"),
        ]
        assert tf.detect_montage_end(transcript) == 0

    def test_many_short_clips_then_long_intro(self):
        """Typical podcast cold open: 8 highlight clips then a long real intro."""
        transcript = [
            make_segment(0, 7000, 12000, "highlight clip one"),
            make_segment(1, 17000, 21000, "highlight clip two"),
            make_segment(0, 21000, 27000, "highlight clip three"),
            make_segment(0, 28000, 38000, "highlight clip four"),
            make_segment(0, 39000, 47000, "highlight clip five"),
            make_segment(1, 48000, 50000, "highlight clip six"),
            make_segment(0, 50000, 53000, "highlight clip seven"),
            make_segment(0, 55000, 60000, "highlight clip eight"),
            make_segment(1, 62000, 103000, "Hi everyone welcome to the show, this is the real intro that goes on for a while"),
        ]
        assert tf.detect_montage_end(transcript) == 8


class TestRescoreMontageSpakers:
    def test_noop_when_montage_end_zero(self):
        transcript = [make_segment(0, 0, 5000, "hello")]
        result = tf.rescore_montage_speakers(transcript, 0, "/fake.wav", "model")
        assert result is transcript

    def test_noop_when_montage_end_exceeds_length(self):
        transcript = [make_segment(0, 0, 5000, "hello")]
        result = tf.rescore_montage_speakers(transcript, 5, "/fake.wav", "model")
        assert result is transcript

    def test_noop_when_montage_end_negative(self):
        transcript = [make_segment(0, 0, 5000, "hello")]
        result = tf.rescore_montage_speakers(transcript, -1, "/fake.wav", "model")
        assert result is transcript


class TestDetectMontageEndBoundary:
    def test_exactly_four_segments_montage(self):
        """Minimum montage: 3 short clips + 1 long at index 3."""
        transcript = [
            make_segment(0, 0, 3000, "clip one"),
            make_segment(1, 3000, 6000, "clip two"),
            make_segment(0, 6000, 9000, "clip three"),
            make_segment(1, 9000, 30000, "real intro starts here"),
        ]
        assert tf.detect_montage_end(transcript) == 3

    def test_exactly_four_segments_no_montage_long_not_enough(self):
        """4 segments but the 4th is only 14s — below 15s threshold."""
        transcript = [
            make_segment(0, 0, 3000, "clip one"),
            make_segment(1, 3000, 6000, "clip two"),
            make_segment(0, 6000, 9000, "clip three"),
            make_segment(1, 9000, 23000, "almost long enough"),
        ]
        assert tf.detect_montage_end(transcript) == 0


class TestVerifySpeakerAssignment:
    def test_detects_chinese_self_intro_mismatch(self):
        transcript = [
            make_segment(0, 0, 5000, "大家好我是张飞"),
            make_segment(1, 5000, 10000, "你好"),
        ]
        speaker_map = {0: "关羽", 1: "张飞"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["关羽", "张飞"])
        assert result[0] == "张飞"
        assert result[1] == "关羽"

    def test_confirms_correct_assignment(self):
        transcript = [
            make_segment(0, 0, 5000, "大家好我是关羽"),
            make_segment(1, 5000, 10000, "你好"),
        ]
        speaker_map = {0: "关羽", 1: "张飞"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["关羽", "张飞"])
        assert result[0] == "关羽"

    def test_english_self_intro(self):
        transcript = [
            make_segment(0, 0, 5000, "Hi everyone I'm Bob"),
            make_segment(1, 5000, 10000, "Welcome"),
        ]
        speaker_map = {0: "Alice", 1: "Bob"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["Alice", "Bob"])
        assert result[0] == "Bob"
        assert result[1] == "Alice"

    def test_no_intro_returns_unchanged(self):
        transcript = [
            make_segment(0, 0, 5000, "今天天气不错"),
            make_segment(1, 5000, 10000, "是的"),
        ]
        speaker_map = {0: "A", 1: "B"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["A", "B"])
        assert result == {0: "A", 1: "B"}

    def test_given_name_match_chinese(self):
        """'我是丽华' should match full name '王丽华'."""
        transcript = [
            make_segment(0, 0, 5000, "嗨，朋友们好，欢迎收听我们的节目，我是丽华"),
            make_segment(1, 5000, 10000, "好，我是某某频道的主播赵大明"),
        ]
        speaker_map = {0: "赵大明", 1: "王丽华"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["赵大明", "王丽华"])
        assert result[0] == "王丽华"
        assert result[1] == "赵大明"

    def test_filler_between_intro_and_name(self):
        """'我是某某频道的主播赵大明' should match despite filler words."""
        transcript = [
            make_segment(0, 0, 5000, "好，感谢节目的邀请，我是某某频道的主播赵大明"),
            make_segment(1, 5000, 10000, "欢迎"),
        ]
        speaker_map = {0: "赵大明", 1: "王丽华"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["赵大明", "王丽华"])
        assert result[0] == "赵大明"

    def test_two_char_name_given_name_match(self):
        """Two-char Chinese name: '我是磊' should match '陈磊'."""
        transcript = [
            make_segment(0, 0, 5000, "大家好我是磊"),
            make_segment(1, 5000, 10000, "你好"),
        ]
        speaker_map = {0: "林峰", 1: "陈磊"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["林峰", "陈磊"])
        assert result[0] == "陈磊"
        assert result[1] == "林峰"

    def test_name_variants_helper(self):
        """_name_variants produces correct variants."""
        assert tf._name_variants("王丽华") == [("王丽华", "王丽华"), ("丽华", "王丽华")]
        assert tf._name_variants("赵大明") == [("赵大明", "赵大明"), ("大明", "赵大明")]
        assert tf._name_variants("Alice") == [("Alice", "Alice")]
        assert tf._name_variants("陈磊") == [("陈磊", "陈磊"), ("磊", "陈磊")]

    def test_name_variants_four_char_name(self):
        """4-char Chinese name (compound surname)."""
        assert tf._name_variants("欧阳明月") == [("欧阳明月", "欧阳明月"), ("阳明月", "欧阳明月")]

    def test_filler_exceeding_limit_no_match(self):
        """16 chars of filler between '我是' and name should NOT match."""
        transcript = [
            make_segment(0, 0, 5000, "我是这个节目的特别邀请来的嘉宾主持人赵大明"),
            make_segment(1, 5000, 10000, "你好"),
        ]
        speaker_map = {0: "王丽华", 1: "赵大明"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["王丽华", "赵大明"])
        assert result == {0: "王丽华", 1: "赵大明"}

    def test_punctuation_cutoff_prevents_cross_sentence_match(self):
        """Punctuation between '我是' and name should block matching."""
        transcript = [
            make_segment(0, 0, 5000, "我是主持人。赵大明你好"),
            make_segment(1, 5000, 10000, "你好"),
        ]
        speaker_map = {0: "王丽华", 1: "赵大明"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["王丽华", "赵大明"])
        assert result == {0: "王丽华", 1: "赵大明"}

    def test_no_speaker_names_returns_unchanged(self):
        transcript = [make_segment(0, 0, 5000, "hello")]
        speaker_map = {0: "Speaker 1"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, None)
        assert result == {0: "Speaker 1"}


class TestChunkByDuration:
    def test_single_chunk(self):
        items = [make_segment(0, 0, 1000, "a"), make_segment(0, 1000, 2000, "b")]
        chunks = tf.chunk_by_duration(items, duration_ms=900000)
        assert len(chunks) == 1

    def test_multiple_chunks(self):
        items = [
            make_segment(0, 0, 1000, "first"),
            make_segment(0, 1000000, 1001000, "second"),
        ]
        chunks = tf.chunk_by_duration(items, duration_ms=900000)
        assert len(chunks) == 2

    def test_empty(self):
        assert tf.chunk_by_duration([]) == []


class TestFormatChunk:
    def test_formats_with_names(self):
        chunk = [make_segment(0, 0, 1000, "hello"), make_segment(1, 1000, 2000, "world")]
        speaker_map = {0: "Alice", 1: "Bob"}
        result = tf.format_chunk(chunk, speaker_map)
        assert "[00:00:00] Alice: hello" in result
        assert "[00:00:01] Bob: world" in result


class TestBuildSystemPrompt:
    def test_base_prompt(self):
        prompt = tf.build_system_prompt()
        assert "transcript editor" in prompt
        assert "Preserve timestamps" in prompt

    def test_with_speaker_names(self):
        prompt = tf.build_system_prompt(speaker_names=["Alice", "Bob"])
        assert "Alice" in prompt
        assert "Bob" in prompt

    def test_with_speaker_context(self):
        ctx = {"Alice": "team lead", "Bob": "engineer"}
        prompt = tf.build_system_prompt(speaker_context=ctx)
        assert "team lead" in prompt
        assert "engineer" in prompt

    def test_with_reference_text(self):
        prompt = tf.build_system_prompt(reference_text="Meeting about Q3 roadmap")
        assert "Q3 roadmap" in prompt

    def test_reference_truncation(self):
        long_ref = "x" * 5000
        prompt = tf.build_system_prompt(reference_text=long_ref)
        assert "[...truncated]" in prompt

    def test_prompt_instructs_to_preserve_periodic_timestamps(self):
        # Solo-speaker bug: when a 15-minute chunk has only one speaker the LLM
        # used to collapse the entire chunk into a single timestamped block,
        # losing every intermediate anchor. The prompt must require the model
        # to keep a periodic timestamp (every ~2 minutes) even when the speaker
        # does not change.
        prompt = tf.build_system_prompt()
        low = prompt.lower()
        assert "every" in low and "minute" in low, (
            "prompt must require LLM to emit a timestamp every ~2 minutes"
        )
        # Must explicitly cover the single-speaker case
        assert "speaker" in low and ("same" in low or "single" in low or "solo" in low), (
            "prompt must mention that the rule applies even when speaker is unchanged"
        )


# ──────────────────────────────────────────────
# transcribe: validate_lang_diarization
# ──────────────────────────────────────────────

class TestValidateLangDiarization:
    def test_auto_with_speakers_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            tf.validate_lang_diarization("auto", 2)
        assert exc_info.value.code == 1

    def test_whisper_with_speakers_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            tf.validate_lang_diarization("whisper", 3)
        assert exc_info.value.code == 1

    def test_zh_with_speakers_ok(self):
        tf.validate_lang_diarization("zh", 2)

    def test_auto_without_speakers_ok(self):
        tf.validate_lang_diarization("auto", None)

    def test_zh_basic_with_speakers_ok(self):
        tf.validate_lang_diarization("zh-basic", 5)

    def test_en_with_speakers_ok(self):
        tf.validate_lang_diarization("en", 4)


# ──────────────────────────────────────────────
# transcribe: --model-cache-dir
# ──────────────────────────────────────────────

class TestModelCacheDir:
    def test_argparse_accepts_model_cache_dir(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("audio_file")
        parser.add_argument("--model-cache-dir", type=str, default=None)
        args = parser.parse_args(["test.wav", "--model-cache-dir", "/tmp/models"])
        assert args.model_cache_dir == "/tmp/models"

    def test_argparse_default_is_none(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("audio_file")
        parser.add_argument("--model-cache-dir", type=str, default=None)
        args = parser.parse_args(["test.wav"])
        assert args.model_cache_dir is None


# ──────────────────────────────────────────────
# transcribe: parse_funasr_results
# ──────────────────────────────────────────────

class TestParseFunasrResults:
    def test_sentence_info_shape(self):
        """Standard Paraformer output with sentence_info (speaker diarization)."""
        res = [{"sentence_info": [
            {"spk": 0, "start": 0, "end": 5000, "text": "hello"},
            {"spk": 1, "start": 5000, "end": 10000, "text": "world"},
        ]}]
        transcript = tf.parse_funasr_results(res)
        assert len(transcript) == 2
        assert transcript[0] == {"speaker": 0, "start_ms": 0, "end_ms": 5000, "text": "hello"}
        assert transcript[1] == {"speaker": 1, "start_ms": 5000, "end_ms": 10000, "text": "world"}

    def test_sentence_info_uses_sentence_key(self):
        """Some models use 'sentence' instead of 'text' in sentence_info."""
        res = [{"sentence_info": [
            {"spk": 0, "start": 0, "end": 1000, "sentence": "fallback text"},
        ]}]
        transcript = tf.parse_funasr_results(res)
        assert transcript[0]["text"] == "fallback text"

    def test_text_only_no_timestamp(self):
        """SenseVoice/Whisper output: text only, no timestamps."""
        res = [{"text": "auto detected speech"}]
        transcript = tf.parse_funasr_results(res)
        assert len(transcript) == 1
        assert transcript[0]["text"] == "auto detected speech"
        assert transcript[0]["speaker"] == 0
        assert transcript[0]["start_ms"] == 0
        assert transcript[0]["end_ms"] == 0

    def test_text_with_timestamp_no_sentence_info(self):
        """Result with text + timestamp but no sentence_info — previously dropped."""
        res = [{"text": "detected speech", "timestamp": [[0, 500], [500, 1200]]}]
        transcript = tf.parse_funasr_results(res)
        assert len(transcript) == 1
        assert transcript[0]["text"] == "detected speech"
        assert transcript[0]["speaker"] == 0
        assert transcript[0]["start_ms"] == 0
        assert transcript[0]["end_ms"] == 1200

    def test_text_with_empty_timestamp(self):
        """Result with text + empty timestamp list."""
        res = [{"text": "speech with empty ts", "timestamp": []}]
        transcript = tf.parse_funasr_results(res)
        assert len(transcript) == 1
        assert transcript[0]["text"] == "speech with empty ts"
        assert transcript[0]["start_ms"] == 0
        assert transcript[0]["end_ms"] == 0

    def test_unknown_shape_warns(self, capsys):
        """Result with no recognized keys should warn, not silently drop."""
        res = [{"unknown_key": "value"}]
        transcript = tf.parse_funasr_results(res)
        assert len(transcript) == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "unrecognized" in captured.out.lower() or "unknown" in captured.out.lower()

    def test_multiple_results_mixed(self):
        """Multiple results with different shapes in one response."""
        res = [
            {"sentence_info": [{"spk": 0, "start": 0, "end": 5000, "text": "first"}]},
            {"text": "second", "timestamp": [[5000, 6000], [6000, 8000]]},
            {"text": "third"},
        ]
        transcript = tf.parse_funasr_results(res)
        assert len(transcript) == 3
        assert transcript[0]["text"] == "first"
        assert transcript[1]["text"] == "second"
        assert transcript[1]["end_ms"] == 8000
        assert transcript[2]["text"] == "third"
        assert transcript[2]["start_ms"] == 0

    def test_empty_results(self):
        """Empty results list."""
        assert tf.parse_funasr_results([]) == []

    def test_sentence_info_missing_spk_defaults_to_zero(self):
        """sentence_info entries without 'spk' key default to speaker 0."""
        res = [{"sentence_info": [{"start": 0, "end": 1000, "text": "no spk"}]}]
        transcript = tf.parse_funasr_results(res)
        assert transcript[0]["speaker"] == 0


# ──────────────────────────────────────────────
# transcribe: _verify_speaker_roles_via_llm
# ──────────────────────────────────────────────

class TestVerifySpeakerRolesViaLLM:
    @patch("transcribe.call_llm", return_value="CORRECT")
    def test_correct_keeps_map(self, mock_llm):
        speaker_map = {0: "Host", 1: "Guest"}
        ctx = {"Host": "asks questions", "Guest": "answers"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result[0] == "Host"
        assert result[1] == "Guest"

    @patch("transcribe.call_llm", return_value="SWAP")
    def test_swap_two_speakers(self, mock_llm):
        speaker_map = {0: "Host", 1: "Guest"}
        ctx = {"Host": "asks questions", "Guest": "answers"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result[0] == "Guest"
        assert result[1] == "Host"

    @patch("transcribe.call_llm", return_value="I'm not sure about this")
    def test_ambiguous_keeps_map(self, mock_llm):
        speaker_map = {0: "Host", 1: "Guest"}
        ctx = {"Host": "asks questions", "Guest": "answers"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result[0] == "Host"
        assert result[1] == "Guest"

    @patch("transcribe.call_llm")
    def test_llm_failure_keeps_map(self, mock_llm):
        mock_llm.side_effect = RuntimeError("API error")
        speaker_map = {0: "Host", 1: "Guest"}
        ctx = {"Host": "asks questions", "Guest": "answers"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result[0] == "Host"
        assert result[1] == "Guest"

    @patch("transcribe.call_llm")
    def test_import_error_propagates(self, mock_llm):
        mock_llm.side_effect = ImportError("No module named 'boto3'")
        speaker_map = {0: "Host", 1: "Guest"}
        ctx = {"Host": "asks questions", "Guest": "answers"}
        with pytest.raises(ImportError):
            tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")

    @patch("transcribe.call_llm")
    def test_multi_speaker_json_swap(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "correct": False,
            "mapping": {"Alice": "Bob", "Bob": "Alice", "Carol": "Carol"},
        })
        speaker_map = {0: "Alice", 1: "Bob", 2: "Carol"}
        ctx = {"Alice": "leader", "Bob": "eng", "Carol": "design"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result[0] == "Bob"
        assert result[1] == "Alice"
        assert result[2] == "Carol"

    @patch("transcribe.call_llm")
    def test_multi_speaker_correct(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "correct": True,
            "mapping": {"Alice": "Alice", "Bob": "Bob", "Carol": "Carol"},
        })
        speaker_map = {0: "Alice", 1: "Bob", 2: "Carol"}
        ctx = {"Alice": "leader", "Bob": "eng", "Carol": "design"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result[0] == "Alice"

    @patch("transcribe.call_llm")
    def test_multi_speaker_invalid_json_keeps_map(self, mock_llm):
        mock_llm.return_value = "not valid json at all"
        speaker_map = {0: "Alice", 1: "Bob", 2: "Carol"}
        ctx = {"Alice": "leader", "Bob": "eng", "Carol": "design"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result == {0: "Alice", 1: "Bob", 2: "Carol"}

    @patch("transcribe.call_llm")
    def test_multi_speaker_three_way_cycle(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "correct": False,
            "mapping": {"Alice": "Bob", "Bob": "Carol", "Carol": "Alice"},
        })
        speaker_map = {0: "Alice", 1: "Bob", 2: "Carol"}
        ctx = {"Alice": "leader", "Bob": "eng", "Carol": "design"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result[0] == "Bob"
        assert result[1] == "Carol"
        assert result[2] == "Alice"

    @patch("transcribe.call_llm")
    def test_multi_speaker_duplicate_targets_rejected(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "correct": False,
            "mapping": {"Alice": "Carol", "Bob": "Carol", "Carol": "Alice"},
        })
        speaker_map = {0: "Alice", 1: "Bob", 2: "Carol"}
        ctx = {"Alice": "leader", "Bob": "eng", "Carol": "design"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result == {0: "Alice", 1: "Bob", 2: "Carol"}


# ──────────────────────────────────────────────
# Integration: main() with mock LLM
# ──────────────────────────────────────────────

class TestVerifySpeakersMain:
    def _write_fixtures(self, tmpdir, transcript, context):
        json_path = tmpdir / "transcript.json"
        ctx_path = tmpdir / "context.json"
        json_path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
        ctx_path.write_text(json.dumps(context, ensure_ascii=False), encoding="utf-8")
        return json_path, ctx_path

    @patch("verify_speakers.call_llm")
    def test_podcast_swap_dryrun(self, mock_llm, two_speaker_transcript,
                                  speaker_context_podcast, tmp_path):
        mock_llm.return_value = "VERDICT: SWAP\nCONFIDENCE: HIGH\nEVIDENCE: labels are swapped"
        json_path, ctx_path = self._write_fixtures(
            tmp_path, two_speaker_transcript, speaker_context_podcast)
        sys.argv = ["verify_speakers.py", str(json_path),
                     "--speakers", "关羽,张飞",
                     "--speaker-context", str(ctx_path)]
        vs.main()
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["speaker"] == 0  # unchanged — dry-run doesn't modify

    @patch("verify_speakers.call_llm")
    def test_podcast_swap_fix(self, mock_llm, two_speaker_transcript,
                               speaker_context_podcast, tmp_path):
        mock_llm.return_value = "VERDICT: SWAP\nCONFIDENCE: HIGH\nEVIDENCE: labels are swapped"
        json_path, ctx_path = self._write_fixtures(
            tmp_path, two_speaker_transcript, speaker_context_podcast)
        sys.argv = ["verify_speakers.py", str(json_path),
                     "--speakers", "关羽,张飞",
                     "--speaker-context", str(ctx_path), "--fix"]
        vs.main()
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["speaker"] == 1  # swapped

    @patch("verify_speakers.call_llm")
    def test_podcast_correct_no_changes(self, mock_llm, two_speaker_transcript,
                                         speaker_context_podcast, tmp_path):
        mock_llm.return_value = "VERDICT: CORRECT\nCONFIDENCE: HIGH\nEVIDENCE: looks good"
        json_path, ctx_path = self._write_fixtures(
            tmp_path, two_speaker_transcript, speaker_context_podcast)
        sys.argv = ["verify_speakers.py", str(json_path),
                     "--speakers", "关羽,张飞",
                     "--speaker-context", str(ctx_path), "--fix"]
        vs.main()
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["speaker"] == 0  # unchanged

    @patch("verify_speakers.call_llm")
    def test_unknown_verdict_exits_2(self, mock_llm, two_speaker_transcript,
                                      speaker_context_podcast, tmp_path):
        mock_llm.return_value = "VERDICT: MAYBE\nCONFIDENCE: LOW\nEVIDENCE: unclear"
        json_path, ctx_path = self._write_fixtures(
            tmp_path, two_speaker_transcript, speaker_context_podcast)
        sys.argv = ["verify_speakers.py", str(json_path),
                     "--speakers", "关羽,张飞",
                     "--speaker-context", str(ctx_path)]
        with pytest.raises(SystemExit) as exc_info:
            vs.main()
        assert exc_info.value.code == 2

    @patch("verify_speakers.call_llm")
    def test_meeting_inconclusive_exits_2(self, mock_llm, four_speaker_transcript,
                                           speaker_context_meeting, tmp_path):
        mock_llm.return_value = "I don't know"
        json_path, ctx_path = self._write_fixtures(
            tmp_path, four_speaker_transcript, speaker_context_meeting)
        sys.argv = ["verify_speakers.py", str(json_path),
                     "--speakers", "Alice,Bob,Carol,Dave",
                     "--speaker-context", str(ctx_path)]
        with pytest.raises(SystemExit) as exc_info:
            vs.main()
        assert exc_info.value.code == 2

    def test_missing_context_file_exits_1(self, two_speaker_transcript, tmp_path):
        json_path = tmp_path / "transcript.json"
        json_path.write_text(json.dumps(two_speaker_transcript), encoding="utf-8")
        sys.argv = ["verify_speakers.py", str(json_path),
                     "--speakers", "A,B",
                     "--speaker-context", str(tmp_path / "nonexistent.json")]
        with pytest.raises(SystemExit) as exc_info:
            vs.main()
        assert exc_info.value.code == 1

    def test_invalid_json_context_exits_1(self, two_speaker_transcript, tmp_path):
        json_path = tmp_path / "transcript.json"
        json_path.write_text(json.dumps(two_speaker_transcript), encoding="utf-8")
        ctx_path = tmp_path / "bad.json"
        ctx_path.write_text("not valid json {{{", encoding="utf-8")
        sys.argv = ["verify_speakers.py", str(json_path),
                     "--speakers", "A,B",
                     "--speaker-context", str(ctx_path)]
        with pytest.raises(SystemExit) as exc_info:
            vs.main()
        assert exc_info.value.code == 1


# ──────────────────────────────────────────────
# Phase1-only and json-out flag tests
# ──────────────────────────────────────────────

class TestPhase1Flags:
    """Tests for --phase1-only and --json-out CLI flags."""

    def _parse(self, extra_args):
        """Parse CLI args with defaults suitable for testing."""
        base = ["test.wav"]
        with patch("sys.argv", ["transcribe.py"] + base + extra_args):
            p = argparse.ArgumentParser()
            p.add_argument("audio_file")
            p.add_argument("--phase1-only", action="store_true")
            p.add_argument("--json-out", type=str, default=None)
            p.add_argument("--skip-transcribe", action="store_true")
            p.add_argument("--skip-llm", action="store_true")
            p.add_argument("--model", default=None)
            p.add_argument("--output", default=None)
            return p.parse_args(base + extra_args)

    def test_phase1_only_flag_parsed(self):
        args = self._parse(["--phase1-only"])
        assert args.phase1_only is True

    def test_phase1_only_default_false(self):
        args = self._parse([])
        assert args.phase1_only is False

    def test_json_out_flag_parsed(self):
        args = self._parse(["--json-out", "/tmp/out.json"])
        assert args.json_out == "/tmp/out.json"

    def test_json_out_default_none(self):
        args = self._parse([])
        assert args.json_out is None

    def test_json_out_overrides_default_path(self):
        from pathlib import Path
        args = self._parse(["--json-out", "/tmp/custom.json"])
        raw_json = Path(args.json_out) if args.json_out else Path(f"{Path(args.audio_file).stem}_raw_transcript.json")
        assert raw_json == Path("/tmp/custom.json")

    def test_default_raw_json_path(self):
        from pathlib import Path
        args = self._parse([])
        raw_json = Path(args.json_out) if args.json_out else Path(f"{Path(args.audio_file).stem}_raw_transcript.json")
        assert raw_json == Path("test_raw_transcript.json")

    def test_phase1_only_with_json_out(self):
        args = self._parse(["--phase1-only", "--json-out", "/tmp/out.json"])
        assert args.phase1_only is True
        assert args.json_out == "/tmp/out.json"

    def test_phase1_only_early_exit(self, tmp_path):
        """--phase1-only exits after Phase 1 without producing .md output."""
        transcript = [
            make_segment(0, 0, 5000, "Hello world"),
            make_segment(1, 5000, 10000, "Hi there"),
        ]
        raw_json = tmp_path / "test_raw_transcript.json"
        with open(raw_json, "w") as f:
            json.dump(transcript, f)

        md_path = tmp_path / "test-transcript.md"
        test_args = [
            "transcribe.py",
            str(tmp_path / "test.wav"),
            "--phase1-only",
            "--skip-transcribe",
            "--json-out", str(raw_json),
            "--device", "cpu",
        ]
        with patch("sys.argv", test_args), \
             patch.object(sys, "exit") as mock_exit:
            mock_exit.side_effect = SystemExit(0)
            with pytest.raises(SystemExit) as exc_info:
                tf.main()
            assert exc_info.value.code == 0
        assert not md_path.exists()

    def test_json_out_writes_to_custom_path(self, tmp_path):
        """--json-out writes transcript JSON to the specified path."""
        custom_json = tmp_path / "custom_output.json"
        transcript = [
            make_segment(0, 0, 5000, "Hello"),
            make_segment(1, 5000, 10000, "World"),
        ]
        with patch("sys.argv", [
                "transcribe.py", str(tmp_path / "test.wav"),
                "--json-out", str(custom_json),
                "--skip-llm", "--device", "cpu",
            ]), \
             patch.object(tf, "transcribe_with_funasr", return_value=transcript), \
             patch.object(tf, "preprocess_audio", return_value=str(tmp_path / "test.wav")), \
             patch.object(tf, "detect_montage_end", return_value=0), \
             patch("pathlib.Path.exists", return_value=True):
            tf.main()
        assert custom_json.exists()
        saved = json.loads(custom_json.read_text())
        assert len(saved) == 2
        assert saved[0]["text"] == "Hello"

    def test_without_new_flags_runs_all_phases(self, tmp_path):
        """Without new flags, all phases run (backward compatibility)."""
        transcript = [
            make_segment(0, 0, 5000, "Hello"),
            make_segment(1, 5000, 10000, "World"),
        ]
        output_md = tmp_path / "test-transcript.md"
        raw_json = tmp_path / "test_raw_transcript.json"

        def fake_exists(self_path=None):
            return True

        with patch("sys.argv", [
                "transcribe.py", str(tmp_path / "test.wav"),
                "--skip-llm", "--device", "cpu",
                "--output", str(output_md),
                "--json-out", str(raw_json),
            ]), \
             patch.object(tf, "transcribe_with_funasr", return_value=transcript), \
             patch.object(tf, "preprocess_audio", return_value=str(tmp_path / "test.wav")), \
             patch.object(tf, "detect_montage_end", return_value=0), \
             patch("pathlib.Path.exists", return_value=True):
            tf.main()
        assert output_md.exists()
        assert raw_json.exists()
        content = output_md.read_text()
        assert "Transcript" in content

    def test_json_out_nonexistent_parent_exits(self, tmp_path):
        """--json-out to a nonexistent directory exits with code 1."""
        bad_path = tmp_path / "nonexistent" / "dir" / "out.json"
        test_args = [
            "transcribe.py",
            str(tmp_path / "test.wav"),
            "--json-out", str(bad_path),
            "--device", "cpu",
        ]
        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                tf.main()
            assert exc_info.value.code == 1

    def test_phase1_only_empty_transcript_exits_error(self, tmp_path):
        """--phase1-only with empty transcript exits with error, not success."""
        raw_json = tmp_path / "empty_raw_transcript.json"
        with open(raw_json, "w") as f:
            json.dump([], f)
        test_args = [
            "transcribe.py",
            str(tmp_path / "test.wav"),
            "--phase1-only",
            "--skip-transcribe",
            "--json-out", str(raw_json),
            "--device", "cpu",
        ]
        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                tf.main()
            assert exc_info.value.code == 1

    def test_phase1_only_writes_json_then_exits(self, tmp_path):
        """--phase1-only writes raw JSON and exits 0 without --skip-transcribe."""
        transcript = [
            make_segment(0, 0, 5000, "Hello"),
            make_segment(1, 5000, 10000, "World"),
        ]
        custom_json = tmp_path / "phase1_out.json"
        with patch("sys.argv", [
                "transcribe.py", str(tmp_path / "test.wav"),
                "--phase1-only",
                "--json-out", str(custom_json),
                "--device", "cpu",
            ]), \
             patch.object(tf, "transcribe_with_funasr", return_value=transcript), \
             patch.object(tf, "preprocess_audio", return_value=str(tmp_path / "test.wav")), \
             patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                tf.main()
            assert exc_info.value.code == 0
        assert custom_json.exists()
        saved = json.loads(custom_json.read_text())
        assert len(saved) == 2
        assert saved[0]["text"] == "Hello"
        md_path = tmp_path / "test-transcript.md"
        assert not md_path.exists()


# ──────────────────────────────────────────────
# speaker_gender: classifier helpers
# ──────────────────────────────────────────────

class TestNormalizeGenderLabel:
    def test_english_male_variants(self):
        assert sg._normalize_gender_label("male") == "male"
        assert sg._normalize_gender_label("M") == "male"
        assert sg._normalize_gender_label("  Man ") == "male"

    def test_english_female_variants(self):
        assert sg._normalize_gender_label("Female") == "female"
        assert sg._normalize_gender_label("f") == "female"
        assert sg._normalize_gender_label("woman") == "female"

    def test_chinese_variants(self):
        assert sg._normalize_gender_label("男") == "male"
        assert sg._normalize_gender_label("女") == "female"
        assert sg._normalize_gender_label("男性") == "male"
        assert sg._normalize_gender_label("女性") == "female"

    def test_unknown_returns_none(self):
        assert sg._normalize_gender_label("other") is None
        assert sg._normalize_gender_label("") is None
        assert sg._normalize_gender_label(None) is None


class TestMajorityVote:
    def test_clear_majority(self):
        assert sg._majority_vote(["male", "male", "female"]) == "male"

    def test_tie_returns_none(self):
        assert sg._majority_vote(["male", "female"]) is None

    def test_empty_returns_none(self):
        assert sg._majority_vote([]) is None

    def test_filters_invalid_labels(self):
        assert sg._majority_vote(["male", "unknown", "male"]) == "male"


class TestSelectSampleSegments:
    def test_picks_longest_segments(self):
        transcript = [
            make_segment(0, 0, 2000, "short"),
            make_segment(0, 2000, 10000, "long"),
            make_segment(1, 10000, 12000, "other"),
            make_segment(0, 12000, 18000, "medium"),
        ]
        picks = sg._select_sample_segments(transcript, speaker_id=0, max_samples=2)
        durations = [p["end_ms"] - p["start_ms"] for p in picks]
        assert durations == sorted(durations, reverse=True)
        assert len(picks) == 2

    def test_filters_short_segments(self):
        transcript = [
            make_segment(0, 0, 500, "too short"),
            make_segment(0, 500, 1000, "also short"),
        ]
        picks = sg._select_sample_segments(transcript, speaker_id=0, min_duration_ms=1500)
        assert picks == []

    def test_returns_empty_when_speaker_absent(self):
        transcript = [make_segment(1, 0, 5000, "hi")]
        assert sg._select_sample_segments(transcript, speaker_id=0) == []


class TestClassifySpeakerGender:
    def test_uses_model_loader_hook(self):
        transcript = [
            make_segment(0, 0, 5000, "host talks"),
            make_segment(0, 5000, 12000, "host talks more"),
            make_segment(1, 12000, 18000, "guest talks"),
        ]

        class FakeModel:
            def infer(self, start_ms, end_ms):
                return "male" if start_ms < 12000 else "female"

        result = sg.classify_speaker_gender(
            audio_path="/dev/null",
            transcript=transcript,
            _model_loader=lambda: FakeModel(),
        )
        assert result == {0: "male", 1: "female"}

    def test_majority_vote_per_speaker(self):
        transcript = [
            make_segment(0, 0, 5000, "a"),
            make_segment(0, 5000, 10000, "b"),
            make_segment(0, 10000, 15000, "c"),
        ]
        calls = {"n": 0}

        class FakeModel:
            def infer(self, start_ms, end_ms):
                calls["n"] += 1
                # 2x male, 1x female — majority male
                return ["female", "male", "male"][calls["n"] - 1]

        result = sg.classify_speaker_gender(
            audio_path="/dev/null",
            transcript=transcript,
            _model_loader=lambda: FakeModel(),
            max_samples=3,
        )
        assert result == {0: "male"}

    def test_tie_produces_no_entry(self):
        transcript = [
            make_segment(0, 0, 5000, "a"),
            make_segment(0, 5000, 10000, "b"),
        ]

        class FakeModel:
            def infer(self, start_ms, end_ms):
                return "male" if start_ms == 0 else "female"

        result = sg.classify_speaker_gender(
            audio_path="/dev/null",
            transcript=transcript,
            _model_loader=lambda: FakeModel(),
            max_samples=2,
        )
        assert result == {}

    def test_skips_speakers_without_long_segments(self):
        transcript = [
            make_segment(0, 0, 500, "too short"),
            make_segment(1, 500, 10000, "long"),
        ]

        class FakeModel:
            def infer(self, start_ms, end_ms):
                return "female"

        result = sg.classify_speaker_gender(
            audio_path="/dev/null",
            transcript=transcript,
            _model_loader=lambda: FakeModel(),
        )
        assert result == {1: "female"}

    def test_inference_exception_does_not_break_other_speakers(self, capsys):
        transcript = [
            make_segment(0, 0, 5000, "a"),
            make_segment(1, 5000, 10000, "b"),
        ]

        class FakeModel:
            def infer(self, start_ms, end_ms):
                if start_ms == 0:
                    raise RuntimeError("model blew up")
                return "female"

        result = sg.classify_speaker_gender(
            audio_path="/dev/null",
            transcript=transcript,
            _model_loader=lambda: FakeModel(),
        )
        assert result == {1: "female"}

    def test_loader_failure_returns_empty(self, capsys):
        transcript = [make_segment(0, 0, 5000, "a")]

        def broken_loader():
            raise ImportError("modelscope not installed")

        result = sg.classify_speaker_gender(
            audio_path="/dev/null",
            transcript=transcript,
            _model_loader=broken_loader,
        )
        assert result == {}

    def test_empty_transcript(self):
        assert sg.classify_speaker_gender("/dev/null", []) == {}

    def test_respects_speaker_ids_filter(self):
        transcript = [
            make_segment(0, 0, 5000, "a"),
            make_segment(1, 5000, 10000, "b"),
        ]

        class FakeModel:
            def infer(self, start_ms, end_ms):
                return "male"

        result = sg.classify_speaker_gender(
            audio_path="/dev/null",
            transcript=transcript,
            speaker_ids=[1],
            _model_loader=lambda: FakeModel(),
        )
        assert result == {1: "male"}


# ──────────────────────────────────────────────
# speaker_gender: reference-text extraction
# ──────────────────────────────────────────────

class TestExtractGenderFromReference:
    def test_role_with_parenthetical_gender_chinese(self):
        text = "主播（女）：韩梅梅\n嘉宾（男）：李雷"
        assert sg.extract_gender_from_reference(text) == {
            "韩梅梅": "female", "李雷": "male"}

    def test_role_with_parenthetical_gender_english(self):
        text = "Host (female): Alice\nGuest (male): Bob"
        assert sg.extract_gender_from_reference(text) == {
            "Alice": "female", "Bob": "male"}

    def test_gender_prefixed_role(self):
        text = "男主播 李雷\n女嘉宾 韩梅梅"
        result = sg.extract_gender_from_reference(text)
        assert result.get("李雷") == "male"
        assert result.get("韩梅梅") == "female"

    def test_name_followed_by_gender(self):
        text = "Alice (female) is a researcher. 韩梅梅（女）"
        result = sg.extract_gender_from_reference(text)
        assert result.get("Alice") == "female"
        assert result.get("韩梅梅") == "female"

    def test_no_matches(self):
        assert sg.extract_gender_from_reference("just some plain text") == {}

    def test_empty_input(self):
        assert sg.extract_gender_from_reference("") == {}
        assert sg.extract_gender_from_reference(None) == {}

    def test_first_match_wins(self):
        text = "Host (female): Alice\nAlice (male) appears again"
        assert sg.extract_gender_from_reference(text)["Alice"] == "female"


# ──────────────────────────────────────────────
# speaker_gender: merge + CLI parsing
# ──────────────────────────────────────────────

class TestMergeGenderSources:
    def test_reference_overrides_auto(self):
        auto = {0: "male", 1: "female"}
        reference = {"Alice": "female"}
        speaker_map = {0: "Alice", 1: "Bob"}
        merged = sg.merge_gender_sources(auto, reference, speaker_map)
        assert merged == {0: "female", 1: "female"}

    def test_reference_fills_missing_auto(self):
        auto = {}
        reference = {"Alice": "female", "Bob": "male"}
        speaker_map = {0: "Alice", 1: "Bob"}
        assert sg.merge_gender_sources(auto, reference, speaker_map) == {
            0: "female", 1: "male"}

    def test_reference_without_matching_name_ignored(self):
        auto = {0: "male"}
        reference = {"Carol": "female"}
        speaker_map = {0: "Alice"}
        assert sg.merge_gender_sources(auto, reference, speaker_map) == {0: "male"}

    def test_empty_inputs(self):
        assert sg.merge_gender_sources(None, None, None) == {}
        assert sg.merge_gender_sources({}, {}, {}) == {}


class TestParseGenderCliArg:
    def test_name_gender_pairs(self):
        speaker_map = {0: "Alice", 1: "Bob"}
        assert sg.parse_gender_cli_arg("Alice:female,Bob:male", speaker_map) == {
            0: "female", 1: "male"}

    def test_equals_separator(self):
        speaker_map = {0: "Alice", 1: "Bob"}
        assert sg.parse_gender_cli_arg("Alice=F,Bob=M", speaker_map) == {
            0: "female", 1: "male"}

    def test_bare_gender_list_positional(self):
        speaker_map = {0: "Alice", 1: "Bob"}
        assert sg.parse_gender_cli_arg("female,male", speaker_map) == {
            0: "female", 1: "male"}

    def test_ignores_unknown_names(self):
        speaker_map = {0: "Alice"}
        assert sg.parse_gender_cli_arg("Nobody:male", speaker_map) == {}

    def test_empty_input(self):
        assert sg.parse_gender_cli_arg("", {0: "Alice"}) == {}
        assert sg.parse_gender_cli_arg(None, {0: "Alice"}) == {}

    def test_chinese_name_mapping(self):
        speaker_map = {0: "韩梅梅", 1: "李雷"}
        assert sg.parse_gender_cli_arg("韩梅梅:女,李雷:男", speaker_map) == {
            0: "female", 1: "male"}


class TestFormatGenderLabel:
    def test_known_labels(self):
        assert sg.format_gender_label("male") == "(male)"
        assert sg.format_gender_label("female") == "(female)"

    def test_unknown_is_empty(self):
        assert sg.format_gender_label(None) == ""
        assert sg.format_gender_label("other") == ""


# ──────────────────────────────────────────────
# transcribe: gender-aware speaker list rendering
# ──────────────────────────────────────────────

class TestAssembleMarkdownWithGender:
    def test_renders_gender_in_speaker_list(self):
        md = tf.assemble_markdown(
            ["[00:00:00] Alice: hi"],
            {
                "title": "Test",
                "filename": "x.wav",
                "duration_ms": 1000,
                "num_speakers": 2,
                "language": "zh",
                "asr_engine": "FunASR",
                "speakers": ["Alice", "Bob"],
                "speaker_genders": {"Alice": "female", "Bob": "male"},
            },
        )
        assert "Alice (female)" in md
        assert "Bob (male)" in md

    def test_omits_gender_when_unknown(self):
        md = tf.assemble_markdown(
            ["[00:00:00] Alice: hi"],
            {
                "title": "Test",
                "filename": "x.wav",
                "duration_ms": 1000,
                "num_speakers": 1,
                "language": "zh",
                "asr_engine": "FunASR",
                "speakers": ["Alice"],
                "speaker_genders": {},
            },
        )
        assert "Alice" in md
        assert "(male)" not in md
        assert "(female)" not in md

    def test_works_without_speaker_genders_key(self):
        md = tf.assemble_markdown(
            ["line"],
            {
                "title": "T", "filename": "f", "duration_ms": 1,
                "num_speakers": 1, "language": "zh", "asr_engine": "E",
                "speakers": ["Alice"],
            },
        )
        assert "Alice" in md


class TestBuildSystemPromptWithGender:
    def test_injects_gender_hints(self):
        prompt = tf.build_system_prompt(
            speaker_context=None,
            reference_text=None,
            speaker_names=["Alice", "Bob"],
            speaker_genders={"Alice": "female", "Bob": "male"},
        )
        assert "Alice" in prompt and "female" in prompt
        assert "Bob" in prompt and "male" in prompt

    def test_no_gender_hints_when_empty(self):
        prompt = tf.build_system_prompt(
            speaker_context=None,
            reference_text=None,
            speaker_names=["Alice"],
            speaker_genders=None,
        )
        assert "female" not in prompt.lower()


# ──────────────────────────────────────────────
# Issue #27 — shownotes parser for title-line format
# ──────────────────────────────────────────────

class TestExtractSpeakerNamesTitleLineFormat:
    """Shownotes with role as a standalone heading, names below as
    'Name: description' lines (小宇宙 / Apple Podcasts common export)."""

    def test_chinese_guest_title_block(self):
        text = (
            "💬 本期嘉宾\n"
            "李雷：资深宏观研究员\n"
            "韩梅梅：独立投资人\n"
        )
        assert tf.extract_speaker_names_from_reference(text) == ["李雷", "韩梅梅"]

    def test_chinese_host_title_block(self):
        text = (
            "主播\n"
            "张三：节目主理人\n"
        )
        assert tf.extract_speaker_names_from_reference(text) == ["张三"]

    def test_english_guests_title_block(self):
        text = (
            "Guests\n"
            "Alice: senior analyst at ACME\n"
            "Bob: independent researcher\n"
        )
        assert tf.extract_speaker_names_from_reference(text) == ["Alice", "Bob"]

    def test_mixed_inline_host_and_title_guests(self):
        text = (
            "Host: 张三\n"
            "\n"
            "💬 本期嘉宾\n"
            "李雷：资深宏观研究员\n"
            "韩梅梅：独立投资人\n"
        )
        result = tf.extract_speaker_names_from_reference(text)
        assert result[0] == "张三"
        assert "李雷" in result
        assert "韩梅梅" in result

    def test_title_block_stops_at_blank_line(self):
        text = (
            "嘉宾\n"
            "李雷：资深宏观研究员\n"
            "\n"
            "相关链接：\n"
            "王五：unrelated article title writer\n"
        )
        assert tf.extract_speaker_names_from_reference(text) == ["李雷"]

    def test_title_block_stops_at_next_heading(self):
        text = (
            "嘉宾\n"
            "李雷：资深宏观研究员\n"
            "时间戳\n"
            "王五：unrelated timestamp label\n"
        )
        assert tf.extract_speaker_names_from_reference(text) == ["李雷"]

    def test_prose_not_mistaken_for_title_block(self):
        """Regression guard: generic prose with 'Name: ...' lines must
        not be captured as guests."""
        text = (
            "In today's episode we talk about topics.\n"
            "Alice said: something interesting.\n"
            "Bob replied: agreed.\n"
        )
        assert tf.extract_speaker_names_from_reference(text) == []

    def test_long_heading_not_treated_as_title(self):
        """A long line with 'Guest' somewhere is not a standalone heading."""
        text = (
            "In this Guest series we explore many perspectives on topics.\n"
            "Alice: senior analyst\n"
        )
        assert tf.extract_speaker_names_from_reference(text) == []

    def test_heading_with_decorative_chars_chinese(self):
        """Emoji/whitespace decoration around heading should still match."""
        text = (
            "  💬  本期嘉宾  \n"
            "李雷：投资人\n"
        )
        assert tf.extract_speaker_names_from_reference(text) == ["李雷"]

    def test_real_world_xiaoyuzhou_style(self):
        """Realistic shownotes excerpt (paraphrased with placeholder names
        per CLAUDE.md convention — never use real hosts/guests in tests)."""
        text = (
            "🔗 官网地址example.com\n"
            "💬 本期嘉宾\n"
            "李雷：资深宏观研究员，业余哲学爱好者\n"
            "韩梅梅：播客《某某节目》主理人\n"
            "🪐 时间戳\n"
            "01:57 开场聊天\n"
        )
        result = tf.extract_speaker_names_from_reference(text)
        assert "李雷" in result
        assert "韩梅梅" in result
        assert len(result) == 2


# ──────────────────────────────────────────────
# Issue #28 — iterative swap for N-way speaker rotations
# ──────────────────────────────────────────────

class TestVerifySpeakerAssignmentNWay:
    """Pairwise swap cannot fix rotations of 3+ speakers. The function
    must iterate until all mismatches are resolved (or cap at N-1 swaps)."""

    def test_three_speaker_full_rotation(self):
        """True mapping is a 3-cycle: id 0→C, 1→A, 2→B.
        Initial labels (wrong): {0:A, 1:B, 2:C}.
        After iteration: {0:C, 1:A, 2:B}."""
        transcript = [
            make_segment(0, 0, 5000, "大家好我是王五"),       # id 0 is actually 王五
            make_segment(1, 5000, 10000, "大家好我是张三"),   # id 1 is actually 张三
            make_segment(2, 10000, 15000, "大家好我是李四"),  # id 2 is actually 李四
        ]
        speaker_map = {0: "张三", 1: "李四", 2: "王五"}
        result = tf.verify_speaker_assignment(
            transcript, speaker_map, ["张三", "李四", "王五"])
        assert result[0] == "王五"
        assert result[1] == "张三"
        assert result[2] == "李四"

    def test_three_speaker_partial_mismatch(self):
        """Only two speakers swapped; third is already correct.
        One pairwise swap suffices."""
        transcript = [
            make_segment(0, 0, 5000, "大家好我是张三"),
            make_segment(1, 5000, 10000, "大家好我是王五"),
            make_segment(2, 10000, 15000, "你好，我是李四"),
        ]
        # id 0 says 张三, currently labeled 张三 → correct.
        # id 1 says 王五, currently labeled 李四 → mismatch.
        # id 2 says 李四, currently labeled 王五 → mismatch.
        speaker_map = {0: "张三", 1: "李四", 2: "王五"}
        result = tf.verify_speaker_assignment(
            transcript, speaker_map, ["张三", "李四", "王五"])
        assert result[0] == "张三"
        assert result[1] == "王五"
        assert result[2] == "李四"

    def test_all_correct_no_swap(self):
        """Regression: all self-intros match labels → no swaps."""
        transcript = [
            make_segment(0, 0, 5000, "大家好我是张三"),
            make_segment(1, 5000, 10000, "大家好我是李四"),
            make_segment(2, 10000, 15000, "大家好我是王五"),
        ]
        speaker_map = {0: "张三", 1: "李四", 2: "王五"}
        result = tf.verify_speaker_assignment(
            transcript, speaker_map, ["张三", "李四", "王五"])
        assert result == {0: "张三", 1: "李四", 2: "王五"}

    def test_iteration_cap_prevents_infinite_loop(self):
        """Pathological input: repeated irreconcilable mismatches should
        terminate within N-1 iterations, not loop forever."""
        # Contradictory self-intros — same speaker id claims two names.
        transcript = [
            make_segment(0, 0, 5000, "大家好我是张三"),
            make_segment(0, 5000, 10000, "其实我是李四"),
            make_segment(1, 10000, 15000, "大家好我是王五"),
            make_segment(2, 15000, 20000, "我是李四"),
        ]
        speaker_map = {0: "张三", 1: "李四", 2: "王五"}
        # Must return, must not hang. No assertion on final labels — the
        # test's goal is termination.
        result = tf.verify_speaker_assignment(
            transcript, speaker_map, ["张三", "李四", "王五"])
        assert isinstance(result, dict)
        assert set(result.values()) == {"张三", "李四", "王五"}

    def test_four_speaker_rotation(self):
        """4-speaker rotation: id 0→D, 1→A, 2→B, 3→C. Needs 3 iterations."""
        transcript = [
            make_segment(0, 0, 5000, "大家好我是赵六"),
            make_segment(1, 5000, 10000, "大家好我是张三"),
            make_segment(2, 10000, 15000, "大家好我是李四"),
            make_segment(3, 15000, 20000, "大家好我是王五"),
        ]
        speaker_map = {0: "张三", 1: "李四", 2: "王五", 3: "赵六"}
        result = tf.verify_speaker_assignment(
            transcript, speaker_map,
            ["张三", "李四", "王五", "赵六"])
        assert result[0] == "赵六"
        assert result[1] == "张三"
        assert result[2] == "李四"
        assert result[3] == "王五"


# ──────────────────────────────────────────────
# Issue #29 — LLM provider detection for Bedrock prefixes
# ──────────────────────────────────────────────

class TestDetectLLMProviderBedrockPrefixes:
    """Expanded Bedrock detection: global., apac., eu., amazon-bedrock/
    wrapper, bedrock/ wrapper."""

    def test_bedrock_global_prefix(self):
        assert detect_llm_provider("global.anthropic.claude-sonnet-4-6") == "bedrock"

    def test_bedrock_apac_prefix(self):
        assert detect_llm_provider("apac.anthropic.claude-sonnet-4-6") == "bedrock"

    def test_bedrock_eu_prefix(self):
        assert detect_llm_provider("eu.anthropic.claude-sonnet-4-6") == "bedrock"

    def test_bedrock_amazon_bedrock_wrapper(self):
        assert detect_llm_provider(
            "amazon-bedrock/global.anthropic.claude-sonnet-4-6") == "bedrock"

    def test_bedrock_amazon_bedrock_wrapper_us_prefix(self):
        assert detect_llm_provider(
            "amazon-bedrock/us.anthropic.claude-sonnet-4-6") == "bedrock"

    def test_bedrock_plain_bedrock_wrapper(self):
        assert detect_llm_provider("bedrock/us.anthropic.claude-sonnet-4-6") == "bedrock"

    def test_bare_claude_still_anthropic(self):
        """Regression: no Bedrock marker → bare claude ID stays Anthropic."""
        assert detect_llm_provider("claude-sonnet-4-6") == "anthropic"

    def test_bare_claude_with_version_still_anthropic(self):
        assert detect_llm_provider("claude-sonnet-4-6-20250929") == "anthropic"


class TestCallLLMExplicitProvider:
    """Explicit provider arg must override auto-detection."""

    def test_explicit_bedrock_over_anthropic_id(self):
        """Force Bedrock even when model ID is a bare 'claude-*' form."""
        with patch("llm_utils._call_bedrock", return_value="ok") as mock_bed, \
             patch("llm_utils._call_anthropic", return_value="wrong") as mock_ant:
            result = call_llm(
                "sys", "msg", "claude-sonnet-4-6",
                region="us-west-2", provider="bedrock")
            assert result == "ok"
            mock_bed.assert_called_once()
            mock_ant.assert_not_called()

    def test_explicit_anthropic_over_bedrock_id(self):
        with patch("llm_utils._call_anthropic", return_value="ok") as mock_ant, \
             patch("llm_utils._call_bedrock", return_value="wrong"):
            result = call_llm(
                "sys", "msg", "us.anthropic.claude-sonnet-4-6",
                provider="anthropic")
            assert result == "ok"
            mock_ant.assert_called_once()

    def test_no_provider_falls_back_to_detect(self):
        with patch("llm_utils._call_bedrock", return_value="ok") as mock_bed:
            result = call_llm(
                "sys", "msg", "global.anthropic.claude-sonnet-4-6",
                region="us-west-2")
            assert result == "ok"
            mock_bed.assert_called_once()


class TestCallLLMBedrockWrapperStripped:
    """`amazon-bedrock/` and `bedrock/` wrapper prefixes must be stripped
    before being passed to the boto3 client — AWS doesn't accept them."""

    def test_amazon_bedrock_prefix_stripped(self):
        captured = {}

        def fake_bedrock(system_prompt, user_message, model_id, region,
                        max_tokens=8192):
            captured["model_id"] = model_id
            return "ok"

        with patch("llm_utils._call_bedrock", side_effect=fake_bedrock):
            call_llm("sys", "msg",
                     "amazon-bedrock/global.anthropic.claude-sonnet-4-6",
                     region="us-west-2")
        assert captured["model_id"] == "global.anthropic.claude-sonnet-4-6"

    def test_plain_bedrock_prefix_stripped(self):
        captured = {}

        def fake_bedrock(system_prompt, user_message, model_id, region,
                        max_tokens=8192):
            captured["model_id"] = model_id
            return "ok"

        with patch("llm_utils._call_bedrock", side_effect=fake_bedrock):
            call_llm("sys", "msg",
                     "bedrock/us.anthropic.claude-sonnet-4-6",
                     region="us-west-2")
        assert captured["model_id"] == "us.anthropic.claude-sonnet-4-6"

    def test_no_prefix_unchanged(self):
        captured = {}

        def fake_bedrock(system_prompt, user_message, model_id, region,
                        max_tokens=8192):
            captured["model_id"] = model_id
            return "ok"

        with patch("llm_utils._call_bedrock", side_effect=fake_bedrock):
            call_llm("sys", "msg",
                     "us.anthropic.claude-sonnet-4-6",
                     region="us-west-2")
        assert captured["model_id"] == "us.anthropic.claude-sonnet-4-6"


# ──────────────────────────────────────────────
# PR-review feedback: additional guards
# ──────────────────────────────────────────────

class TestStripBedrockWrapperMalformed:
    """Typo-class inputs (`bedrock//foo`, lone `amazon-bedrock/`) must
    raise loudly instead of forwarding a bad ID to boto3."""

    def test_double_slash_typo_raises(self):
        from llm_utils import strip_bedrock_wrapper
        with pytest.raises(ValueError, match="Malformed Bedrock model ID"):
            strip_bedrock_wrapper("bedrock//foo")

    def test_amazon_bedrock_double_slash_raises(self):
        from llm_utils import strip_bedrock_wrapper
        with pytest.raises(ValueError, match="Malformed Bedrock model ID"):
            strip_bedrock_wrapper("amazon-bedrock//global.anthropic.claude-sonnet-4-6")

    def test_bare_wrapper_raises(self):
        from llm_utils import strip_bedrock_wrapper
        with pytest.raises(ValueError, match="Malformed Bedrock model ID"):
            strip_bedrock_wrapper("bedrock/")

    def test_valid_input_unchanged(self):
        from llm_utils import strip_bedrock_wrapper
        assert strip_bedrock_wrapper("us.anthropic.claude-sonnet-4-6") == \
               "us.anthropic.claude-sonnet-4-6"


class TestCallLLMProviderMismatch:
    """When --provider disagrees with auto-detected provider, a
    'Note:' line must be printed so users can connect SDK errors to
    the override choice."""

    def test_mismatch_note_printed(self, capsys):
        with patch("llm_utils._call_anthropic", return_value="ok"):
            call_llm("sys", "msg",
                     "us.anthropic.claude-sonnet-4-6",
                     provider="anthropic")
        captured = capsys.readouterr().out
        assert "explicit --provider=anthropic overrides detected provider=bedrock" in captured

    def test_matching_provider_no_note(self, capsys):
        with patch("llm_utils._call_bedrock", return_value="ok"):
            call_llm("sys", "msg",
                     "us.anthropic.claude-sonnet-4-6",
                     region="us-west-2", provider="bedrock")
        captured = capsys.readouterr().out
        assert "overrides detected provider" not in captured

    def test_implicit_no_note(self, capsys):
        with patch("llm_utils._call_bedrock", return_value="ok"):
            call_llm("sys", "msg",
                     "us.anthropic.claude-sonnet-4-6",
                     region="us-west-2")
        captured = capsys.readouterr().out
        assert "overrides detected provider" not in captured


class TestVerifySpeakerAssignmentCapWarning:
    """When pairwise-swap iteration exhausts the cap with unresolved
    mismatches (contradictory self-intros), the code must surface a
    WARNING rather than silently returning a partially-corrected map."""

    def test_cap_exhaustion_warning(self, capsys):
        # Contradictory: id 0 claims two different names.
        transcript = [
            make_segment(0, 0, 5000, "大家好我是张三"),
            make_segment(0, 5000, 10000, "其实我是李四"),
            make_segment(1, 10000, 15000, "大家好我是王五"),
            make_segment(2, 15000, 20000, "我是李四"),
        ]
        speaker_map = {0: "张三", 1: "李四", 2: "王五"}
        tf.verify_speaker_assignment(
            transcript, speaker_map, ["张三", "李四", "王五"])
        captured = capsys.readouterr().out
        assert "hit iteration cap" in captured

    def test_clean_convergence_no_cap_warning(self, capsys):
        """4-speaker rotation resolves in exactly N-1=3 swaps → no warning."""
        transcript = [
            make_segment(0, 0, 5000, "大家好我是赵六"),
            make_segment(1, 5000, 10000, "大家好我是张三"),
            make_segment(2, 10000, 15000, "大家好我是李四"),
            make_segment(3, 15000, 20000, "大家好我是王五"),
        ]
        speaker_map = {0: "张三", 1: "李四", 2: "王五", 3: "赵六"}
        tf.verify_speaker_assignment(
            transcript, speaker_map,
            ["张三", "李四", "王五", "赵六"])
        captured = capsys.readouterr().out
        assert "hit iteration cap" not in captured


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
