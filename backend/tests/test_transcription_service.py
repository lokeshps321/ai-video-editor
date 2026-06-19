import base64
import math
import os
from pathlib import Path

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")
os.environ["TRANSCRIBE_BACKEND"] = "local"
os.environ["TRANSCRIBE_CLOUD_ONLY"] = "false"
os.environ["TRANSCRIBE_GROQ_LANGUAGE_FALLBACKS"] = ""
os.environ["TRANSCRIBE_GROQ_LANGUAGE_FALLBACK_MAX_ATTEMPTS"] = "0"

from app import transcription_service as ts
from app.transcription_service import TranscriptPayload, TranscriptWordPayload


def _payload(word_count: int) -> TranscriptPayload:
    words: list[TranscriptWordPayload] = []
    text_parts: list[str] = []
    for idx in range(word_count):
        token = f"w{idx}"
        text_parts.append(token)
        words.append(
            TranscriptWordPayload(
                id=str(idx),
                text=token,
                start_sec=idx * 0.1,
                end_sec=(idx * 0.1) + 0.09,
            )
        )
    return TranscriptPayload(
        source="faster_whisper",
        language="en",
        text=" ".join(text_parts),
        words=words,
        is_mock=False,
    )


def _payload_with_times(
    times: list[tuple[float, float]], *, source: str = "groq"
) -> TranscriptPayload:
    words: list[TranscriptWordPayload] = []
    for idx, (start_sec, end_sec) in enumerate(times):
        words.append(
            TranscriptWordPayload(
                id=f"t{idx}",
                text=f"w{idx}",
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )
    return TranscriptPayload(
        source=source,
        language="en",
        text=" ".join(word.text for word in words),
        words=words,
        is_mock=False,
    )


def _payload_with_entries(
    entries: list[tuple[float, float, str]], *, source: str = "groq"
) -> TranscriptPayload:
    words: list[TranscriptWordPayload] = []
    for idx, (start_sec, end_sec, text) in enumerate(entries):
        words.append(
            TranscriptWordPayload(
                id=f"e{idx}",
                text=text,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )
    return TranscriptPayload(
        source=source,
        language="en",
        text=" ".join(word.text for word in words),
        words=words,
        is_mock=False,
    )


def test_detect_hallucinations_collapses_duplicate_bursts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_DUP_GAP_SEC", "0.35")
    words = [
        TranscriptWordPayload(id="1", text="I'm", start_sec=1.00, end_sec=1.10),
        TranscriptWordPayload(id="2", text="I'm", start_sec=1.12, end_sec=1.20),
        TranscriptWordPayload(id="3", text="worried", start_sec=1.21, end_sec=1.34),
        TranscriptWordPayload(id="4", text="worried", start_sec=1.35, end_sec=1.45),
        TranscriptWordPayload(id="5", text="now", start_sec=1.46, end_sec=1.60),
        TranscriptWordPayload(id="6", text="finish", start_sec=1.70, end_sec=1.90),
    ]

    result = ts._detect_hallucinations(words)
    assert [item.text for item in result] == ["I'm", "worried", "now", "finish"]


def test_detect_hallucinations_removes_tight_repeat_phrase_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_STRICT", "true")
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_REPEAT_MIN_OCCURRENCES", "3")
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_REPEAT_WINDOW_SEC", "8")
    words = [
        TranscriptWordPayload(id="w1", text="I", start_sec=10.00, end_sec=10.10),
        TranscriptWordPayload(id="w2", text="go", start_sec=10.11, end_sec=10.20),
        TranscriptWordPayload(id="w3", text="to", start_sec=10.21, end_sec=10.30),
        TranscriptWordPayload(id="w4", text="win", start_sec=10.31, end_sec=10.40),
        TranscriptWordPayload(id="w5", text="I", start_sec=11.00, end_sec=11.10),
        TranscriptWordPayload(id="w6", text="go", start_sec=11.11, end_sec=11.20),
        TranscriptWordPayload(id="w7", text="to", start_sec=11.21, end_sec=11.30),
        TranscriptWordPayload(id="w8", text="win", start_sec=11.31, end_sec=11.40),
        TranscriptWordPayload(id="w9", text="I", start_sec=12.00, end_sec=12.10),
        TranscriptWordPayload(id="w10", text="go", start_sec=12.11, end_sec=12.20),
        TranscriptWordPayload(id="w11", text="to", start_sec=12.21, end_sec=12.30),
        TranscriptWordPayload(id="w12", text="win", start_sec=12.31, end_sec=12.40),
        TranscriptWordPayload(id="w13", text="finish", start_sec=13.00, end_sec=13.20),
        TranscriptWordPayload(id="w14", text="now", start_sec=13.21, end_sec=13.35),
    ]

    result = ts._detect_hallucinations(words)
    assert [item.text for item in result] == ["I", "go", "to", "win", "finish", "now"]


def test_trim_prompt_leakage_phrases_strips_music_instruction_fragments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_PROMPT_LEAKAGE_FILTER", "true")
    ts._TRANSCRIPTION_RUNTIME.profile = "music"
    words = [
        TranscriptWordPayload(id="1", text="We've", start_sec=1.00, end_sec=1.10),
        TranscriptWordPayload(id="2", text="been", start_sec=1.11, end_sec=1.20),
        TranscriptWordPayload(id="3", text="spending", start_sec=1.21, end_sec=1.35),
        TranscriptWordPayload(id="4", text="most", start_sec=1.36, end_sec=1.48),
        TranscriptWordPayload(id="5", text="our", start_sec=1.49, end_sec=1.58),
        TranscriptWordPayload(id="6", text="lives", start_sec=1.59, end_sec=1.70),
        TranscriptWordPayload(id="7", text="living", start_sec=1.71, end_sec=1.84),
        TranscriptWordPayload(id="8", text="in", start_sec=1.85, end_sec=1.92),
        TranscriptWordPayload(id="9", text="a", start_sec=1.93, end_sec=1.98),
        TranscriptWordPayload(id="10", text="gangsta's", start_sec=1.99, end_sec=2.12),
        TranscriptWordPayload(id="11", text="paradise", start_sec=2.13, end_sec=2.30),
        TranscriptWordPayload(id="12", text="Transcribe", start_sec=2.31, end_sec=2.40),
        TranscriptWordPayload(id="13", text="speech", start_sec=2.41, end_sec=2.50),
        TranscriptWordPayload(id="14", text="and", start_sec=2.51, end_sec=2.58),
        TranscriptWordPayload(id="15", text="sung", start_sec=2.59, end_sec=2.70),
        TranscriptWordPayload(id="16", text="lyrics", start_sec=2.71, end_sec=2.85),
        TranscriptWordPayload(id="17", text="Do", start_sec=2.86, end_sec=2.91),
        TranscriptWordPayload(id="18", text="not", start_sec=2.92, end_sec=2.98),
        TranscriptWordPayload(id="19", text="translate", start_sec=2.99, end_sec=3.10),
        TranscriptWordPayload(id="20", text="again", start_sec=3.11, end_sec=3.24),
    ]

    try:
        result = ts._trim_prompt_leakage_phrases(words)
    finally:
        ts._TRANSCRIPTION_RUNTIME.profile = None

    assert [item.text for item in result] == [
        "We've",
        "been",
        "spending",
        "most",
        "our",
        "lives",
        "living",
        "in",
        "a",
        "gangsta's",
        "paradise",
        "again",
    ]


def test_detect_hallucinations_preserves_repeat_lines_in_music_profile() -> None:
    previous = getattr(ts._TRANSCRIPTION_RUNTIME, "profile", None)
    ts._TRANSCRIPTION_RUNTIME.profile = "music"
    try:
        words = [
            TranscriptWordPayload(id="w1", text="lose", start_sec=1.00, end_sec=1.10),
            TranscriptWordPayload(id="w2", text="my", start_sec=1.10, end_sec=1.20),
            TranscriptWordPayload(id="w3", text="mind", start_sec=1.20, end_sec=1.35),
            TranscriptWordPayload(id="w4", text="lose", start_sec=2.10, end_sec=2.20),
            TranscriptWordPayload(id="w5", text="my", start_sec=2.20, end_sec=2.30),
            TranscriptWordPayload(id="w6", text="mind", start_sec=2.30, end_sec=2.45),
            TranscriptWordPayload(id="w7", text="lose", start_sec=3.10, end_sec=3.20),
            TranscriptWordPayload(id="w8", text="my", start_sec=3.20, end_sec=3.30),
            TranscriptWordPayload(id="w9", text="mind", start_sec=3.30, end_sec=3.45),
        ]

        result = ts._detect_hallucinations(words)
        assert [item.text for item in result] == [item.text for item in words]
    finally:
        ts._TRANSCRIPTION_RUNTIME.profile = previous


def test_detect_hallucinations_preserves_non_latin_sequences() -> None:
    words = [
        TranscriptWordPayload(id="1", text="ಓ", start_sec=0.0, end_sec=0.2),
        TranscriptWordPayload(id="2", text="ಬಿಸಿಲು", start_sec=0.2, end_sec=0.4),
        TranscriptWordPayload(id="3", text="ಕುದುರೆಯೊಂದು", start_sec=0.4, end_sec=0.6),
        TranscriptWordPayload(id="4", text="ಎದೆಯಿಂದ", start_sec=0.6, end_sec=0.8),
        TranscriptWordPayload(id="5", text="ಓಡಿದಂತೆ", start_sec=0.8, end_sec=1.0),
        TranscriptWordPayload(id="6", text="ನೆನಪಿನಿಂದ", start_sec=1.0, end_sec=1.2),
        TranscriptWordPayload(id="7", text="ನದಿಯೊಂದು", start_sec=1.2, end_sec=1.4),
        TranscriptWordPayload(id="8", text="ಮೂಡಿದಂತೆ", start_sec=1.4, end_sec=1.6),
    ]

    result = ts._detect_hallucinations(words)
    assert [item.text for item in result] == [item.text for item in words]


def test_trim_known_tail_hallucination_drops_phrase_after_long_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_TAIL_PHRASE_FILTER", "true")
    monkeypatch.setenv("TRANSCRIBE_TAIL_PHRASE_MIN_GAP_SEC", "0.5")
    words = [
        TranscriptWordPayload(id="w1", text="we", start_sec=0.10, end_sec=0.24),
        TranscriptWordPayload(id="w2", text="go", start_sec=0.24, end_sec=0.38),
        TranscriptWordPayload(id="w3", text="on", start_sec=0.38, end_sec=0.52),
        TranscriptWordPayload(id="w4", text="and", start_sec=0.52, end_sec=0.66),
        TranscriptWordPayload(id="w5", text="on", start_sec=0.66, end_sec=0.82),
        TranscriptWordPayload(id="w6", text="the", start_sec=5.80, end_sec=5.95),
        TranscriptWordPayload(id="w7", text="end", start_sec=5.95, end_sec=6.10),
        TranscriptWordPayload(id="w8", text="thank", start_sec=6.10, end_sec=6.24),
        TranscriptWordPayload(id="w9", text="you", start_sec=6.24, end_sec=6.40),
    ]

    trimmed = ts._trim_known_tail_hallucination(words, 6.6)
    assert [item.text for item in trimmed] == ["we", "go", "on", "and", "on"]


def test_trim_known_tail_hallucination_keeps_phrase_without_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_TAIL_PHRASE_FILTER", "true")
    monkeypatch.setenv("TRANSCRIBE_TAIL_PHRASE_MIN_GAP_SEC", "0.5")
    words = [
        TranscriptWordPayload(id="w1", text="this", start_sec=0.10, end_sec=0.22),
        TranscriptWordPayload(id="w2", text="was", start_sec=0.22, end_sec=0.34),
        TranscriptWordPayload(id="w3", text="great", start_sec=0.34, end_sec=0.50),
        TranscriptWordPayload(id="w4", text="thank", start_sec=0.50, end_sec=0.62),
        TranscriptWordPayload(id="w5", text="you", start_sec=0.62, end_sec=0.80),
    ]

    trimmed = ts._trim_known_tail_hallucination(words, 1.0)
    assert [item.text for item in trimmed] == [item.text for item in words]


def test_apply_word_filters_drops_words_inside_nonvocal_music_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_FILTER", "true")
    monkeypatch.setenv("TRANSCRIBE_NONVOCAL_REGION_FILTER", "true")
    monkeypatch.setattr(
        ts,
        "detect_silence_ranges",
        lambda *_args, **_kwargs: [(8.0, 15.0)],
    )
    words = [
        TranscriptWordPayload(id="w1", text="real", start_sec=1.0, end_sec=1.3),
        TranscriptWordPayload(id="w2", text="fake", start_sec=10.0, end_sec=10.3),
        TranscriptWordPayload(id="w3", text="lyrics", start_sec=18.0, end_sec=18.4),
    ]
    previous_profile = getattr(ts._TRANSCRIPTION_RUNTIME, "profile", None)
    ts._TRANSCRIPTION_RUNTIME.profile = "music"
    try:
        filtered = ts._apply_word_filters(words, 24.0, audio_path="prepared.mp3")
    finally:
        ts._TRANSCRIPTION_RUNTIME.profile = previous_profile

    assert [word.text for word in filtered] == ["real", "lyrics"]


def test_apply_word_filters_keeps_words_in_speech_profile_even_if_silence_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_FILTER", "true")
    monkeypatch.setenv("TRANSCRIBE_NONVOCAL_REGION_FILTER", "true")
    monkeypatch.setattr(
        ts,
        "detect_silence_ranges",
        lambda *_args, **_kwargs: [(8.0, 15.0)],
    )
    words = [
        TranscriptWordPayload(id="w1", text="hello", start_sec=1.0, end_sec=1.3),
        TranscriptWordPayload(id="w2", text="world", start_sec=10.0, end_sec=10.3),
    ]
    previous_profile = getattr(ts._TRANSCRIPTION_RUNTIME, "profile", None)
    ts._TRANSCRIPTION_RUNTIME.profile = "speech"
    try:
        filtered = ts._apply_word_filters(words, 24.0, audio_path="prepared.mp3")
    finally:
        ts._TRANSCRIPTION_RUNTIME.profile = previous_profile

    assert [word.text for word in filtered] == ["hello", "world"]


def test_apply_word_filters_trims_sparse_music_tail_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_FILTER", "true")
    monkeypatch.setenv("TRANSCRIBE_SPARSE_MUSIC_TAIL_FILTER", "true")
    words = [
        TranscriptWordPayload(id="w1", text="you", start_sec=92.0, end_sec=92.2),
        TranscriptWordPayload(id="w2", text="and", start_sec=92.21, end_sec=92.34),
        TranscriptWordPayload(id="w3", text="me", start_sec=92.35, end_sec=92.5),
        TranscriptWordPayload(id="w4", text="casting", start_sec=95.4, end_sec=95.58),
        TranscriptWordPayload(id="w5", text="calling", start_sec=95.62, end_sec=95.84),
    ]
    previous_profile = getattr(ts._TRANSCRIPTION_RUNTIME, "profile", None)
    ts._TRANSCRIPTION_RUNTIME.profile = "music"
    try:
        filtered = ts._apply_word_filters(words, 106.0, audio_path="prepared.mp3")
    finally:
        ts._TRANSCRIPTION_RUNTIME.profile = previous_profile

    assert [word.text for word in filtered] == ["you", "and", "me"]


def test_apply_word_filters_keeps_tail_when_song_ends_without_music_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_FILTER", "true")
    monkeypatch.setenv("TRANSCRIBE_SPARSE_MUSIC_TAIL_FILTER", "true")
    words = [
        TranscriptWordPayload(id="w1", text="you", start_sec=92.0, end_sec=92.2),
        TranscriptWordPayload(id="w2", text="and", start_sec=92.21, end_sec=92.34),
        TranscriptWordPayload(id="w3", text="me", start_sec=92.35, end_sec=92.5),
        TranscriptWordPayload(id="w4", text="forever", start_sec=92.54, end_sec=92.9),
    ]
    previous_profile = getattr(ts._TRANSCRIPTION_RUNTIME, "profile", None)
    ts._TRANSCRIPTION_RUNTIME.profile = "music"
    try:
        filtered = ts._apply_word_filters(words, 93.4, audio_path="prepared.mp3")
    finally:
        ts._TRANSCRIPTION_RUNTIME.profile = previous_profile

    assert [word.text for word in filtered] == ["you", "and", "me", "forever"]


def test_trim_songlike_tail_hallucination_drops_sparse_outro_cluster() -> None:
    payload = TranscriptPayload(
        source="groq",
        language="en",
        text="you and me casting calling",
        words=[
            TranscriptWordPayload(id="w1", text="you", start_sec=92.0, end_sec=92.2),
            TranscriptWordPayload(id="w2", text="and", start_sec=92.21, end_sec=92.34),
            TranscriptWordPayload(id="w3", text="me", start_sec=92.35, end_sec=92.5),
            TranscriptWordPayload(
                id="w4", text="casting", start_sec=95.4, end_sec=95.58
            ),
            TranscriptWordPayload(
                id="w5", text="calling", start_sec=95.62, end_sec=95.84
            ),
        ],
        is_mock=False,
    )

    trimmed = ts.trim_songlike_tail_hallucination(payload, duration_sec=106.0)

    assert [word.text for word in trimmed.words] == ["you", "and", "me"]
    assert trimmed.text == "you and me"


def test_trim_songlike_tail_hallucination_keeps_normal_ending() -> None:
    payload = TranscriptPayload(
        source="groq",
        language="en",
        text="you and me forever",
        words=[
            TranscriptWordPayload(id="w1", text="you", start_sec=92.0, end_sec=92.2),
            TranscriptWordPayload(id="w2", text="and", start_sec=92.21, end_sec=92.34),
            TranscriptWordPayload(id="w3", text="me", start_sec=92.35, end_sec=92.5),
            TranscriptWordPayload(
                id="w4", text="forever", start_sec=92.54, end_sec=92.9
            ),
        ],
        is_mock=False,
    )

    trimmed = ts.trim_songlike_tail_hallucination(payload, duration_sec=93.4)

    assert [word.text for word in trimmed.words] == ["you", "and", "me", "forever"]


def test_trim_song_mode_to_manual_lyrics_span_drops_nonmanual_edges() -> None:
    payload = TranscriptPayload(
        source="groq_gapfill_lyrics_ref",
        language="en",
        text="intro As I walk outro",
        words=[
            TranscriptWordPayload(
                id="w1", text="intro", start_sec=7.0, end_sec=7.4, source_pass="primary"
            ),
            TranscriptWordPayload(
                id="w2", text="As", start_sec=25.5, end_sec=25.7, source_pass="manual"
            ),
            TranscriptWordPayload(
                id="w3", text="I", start_sec=25.8, end_sec=26.0, source_pass="manual"
            ),
            TranscriptWordPayload(
                id="w4", text="walk", start_sec=26.1, end_sec=26.4, source_pass="manual"
            ),
            TranscriptWordPayload(
                id="w5",
                text="outro",
                start_sec=240.0,
                end_sec=240.4,
                source_pass="primary",
            ),
        ],
        is_mock=False,
    )

    trimmed = ts.trim_song_mode_to_manual_lyrics_span(payload)

    assert [word.text for word in trimmed.words] == ["As", "I", "walk"]
    assert trimmed.text == "As I walk"


def test_stabilize_song_words_with_validator_drops_short_unsupported_clusters() -> None:
    primary_words = [
        TranscriptWordPayload(id="1", text="we", start_sec=0.10, end_sec=0.20),
        TranscriptWordPayload(id="2", text="ride", start_sec=0.21, end_sec=0.33),
        TranscriptWordPayload(id="3", text="fast", start_sec=0.34, end_sec=0.48),
        TranscriptWordPayload(id="4", text="check", start_sec=9.60, end_sec=9.76),
        TranscriptWordPayload(id="5", text="latest", start_sec=9.77, end_sec=9.98),
        TranscriptWordPayload(id="6", text="thank", start_sec=10.10, end_sec=10.24),
        TranscriptWordPayload(id="7", text="you", start_sec=10.25, end_sec=10.36),
        TranscriptWordPayload(id="8", text="for", start_sec=10.37, end_sec=10.46),
        TranscriptWordPayload(id="9", text="watching", start_sec=10.47, end_sec=10.70),
    ]
    validator_words = [
        TranscriptWordPayload(id="v1", text="we", start_sec=0.10, end_sec=0.20),
        TranscriptWordPayload(id="v2", text="ride", start_sec=0.21, end_sec=0.33),
        TranscriptWordPayload(id="v3", text="fast", start_sec=0.34, end_sec=0.48),
    ]

    result = ts._stabilize_song_words_with_validator(
        primary_words, validator_words, 11.0
    )
    assert [word.text for word in result] == ["we", "ride", "fast"]


def test_stabilize_song_transcript_uses_local_validator_in_song_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = TranscriptPayload(
        source="groq",
        language="en",
        text="we ride fast check latest thank you for watching",
        words=[
            TranscriptWordPayload(id="1", text="we", start_sec=0.10, end_sec=0.20),
            TranscriptWordPayload(id="2", text="ride", start_sec=0.21, end_sec=0.33),
            TranscriptWordPayload(id="3", text="fast", start_sec=0.34, end_sec=0.48),
            TranscriptWordPayload(id="4", text="check", start_sec=9.60, end_sec=9.76),
            TranscriptWordPayload(id="5", text="latest", start_sec=9.77, end_sec=9.98),
            TranscriptWordPayload(id="6", text="thank", start_sec=10.10, end_sec=10.24),
            TranscriptWordPayload(id="7", text="you", start_sec=10.25, end_sec=10.36),
            TranscriptWordPayload(id="8", text="for", start_sec=10.37, end_sec=10.46),
            TranscriptWordPayload(
                id="9", text="watching", start_sec=10.47, end_sec=10.70
            ),
        ],
        is_mock=False,
    )
    validator = TranscriptPayload(
        source="faster_whisper",
        language="en",
        text="we ride fast",
        words=[
            TranscriptWordPayload(
                id="v1", text="we", start_sec=0.10, end_sec=0.20, confidence=0.94
            ),
            TranscriptWordPayload(
                id="v2", text="ride", start_sec=0.21, end_sec=0.33, confidence=0.94
            ),
            TranscriptWordPayload(
                id="v3", text="fast", start_sec=0.34, end_sec=0.48, confidence=0.94
            ),
        ],
        is_mock=False,
    )

    monkeypatch.setattr(
        ts, "_build_from_faster_whisper", lambda *args, **kwargs: validator
    )
    previous_mode = getattr(ts._TRANSCRIPTION_RUNTIME, "mode", None)
    ts._TRANSCRIPTION_RUNTIME.mode = "song"
    try:
        result = ts.stabilize_song_transcript(
            payload,
            path="/tmp/fake.wav",
            duration_sec=11.0,
        )
    finally:
        ts._TRANSCRIPTION_RUNTIME.mode = previous_mode

    assert result.source == "groq_validated"
    assert [word.text for word in result.words] == ["we", "ride", "fast"]


def test_generate_transcript_retries_low_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_MODEL", "base.en")
    monkeypatch.setenv("TRANSCRIBE_RETRY_MODEL", "medium")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_QUALITY_RETRY", "true")
    monkeypatch.setenv("TRANSCRIBE_MIN_WORDS_PER_SEC", "0.45")
    monkeypatch.setenv("TRANSCRIBE_RETRY_MIN_DURATION_SEC", "0")

    calls: list[tuple[str | None, int | None, bool | None]] = []

    def fake_build(
        _path: str,
        _duration: float,
        *,
        model_name: str | None = None,
        beam_size: int | None = None,
        force_vad_filter: bool | None = None,
    ) -> TranscriptPayload | None:
        calls.append((model_name, beam_size, force_vad_filter))
        if len(calls) == 1:
            return _payload(10)
        return _payload(180)

    monkeypatch.setattr(ts, "_build_from_faster_whisper", fake_build)

    result = ts.generate_transcript("sample.mp4", 240.0)
    assert len(result.words) == 180
    assert len(calls) == 2
    assert calls[0][0] == "base.en"
    assert calls[1][0] == "medium"
    assert calls[1][2] is False


def test_generate_transcript_retry_when_primary_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_MODEL", "base.en")
    monkeypatch.setenv("TRANSCRIBE_RETRY_MODEL", "medium")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_QUALITY_RETRY", "true")
    monkeypatch.setenv("TRANSCRIBE_RETRY_MIN_DURATION_SEC", "0")

    calls: list[str | None] = []

    def fake_build(
        _path: str,
        _duration: float,
        *,
        model_name: str | None = None,
        beam_size: int | None = None,
        force_vad_filter: bool | None = None,
    ) -> TranscriptPayload | None:
        calls.append(model_name)
        if len(calls) == 1:
            return None
        return _payload(90)

    monkeypatch.setattr(ts, "_build_from_faster_whisper", fake_build)

    result = ts.generate_transcript("sample.mp4", 120.0)
    assert result.is_mock is False
    assert len(result.words) == 90
    assert calls == ["base.en", "medium"]


def test_generate_transcript_retries_groq_on_long_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "groq")
    monkeypatch.setenv("TRANSCRIBE_ALLOW_MOCK_FALLBACK", "false")
    monkeypatch.setenv("TRANSCRIBE_GROQ_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MODEL", "whisper-large-v3")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MIN_DURATION_SEC", "0")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT", "false")
    monkeypatch.setenv("TRANSCRIBE_MAX_WORD_GAP_SEC", "12")
    monkeypatch.setenv("TRANSCRIBE_GAP_CHECK_MIN_WORDS", "2")

    calls: list[str] = []

    def fake_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str = "whisper-large-v3-turbo",
        prompt: str | None = None,
    ) -> TranscriptPayload | None:
        calls.append(model_name)
        if model_name == "whisper-large-v3-turbo":
            # Big gap between 20s and 45s should trigger retry.
            return _payload_with_times(
                [(1.0, 2.0), (10.0, 11.0), (20.0, 21.0), (45.0, 46.0)], source="groq"
            )
        return _payload_with_times(
            [
                (1.0, 2.0),
                (10.0, 11.0),
                (20.0, 21.0),
                (30.0, 31.0),
                (40.0, 41.0),
                (45.0, 46.0),
            ],
            source="groq",
        )

    monkeypatch.setattr(ts, "_build_from_groq", fake_groq)
    monkeypatch.setattr(
        ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3)
    )

    result = ts.generate_transcript("sample.mp4", 60.0)
    assert len(result.words) == 6
    assert calls == ["whisper-large-v3-turbo", "whisper-large-v3"]


def test_generate_transcript_groq_gap_fill_preserves_primary_dialogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "groq")
    monkeypatch.setenv("TRANSCRIBE_GROQ_MODEL", "whisper-large-v3")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MIN_DURATION_SEC", "0")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT", "false")
    monkeypatch.setenv("TRANSCRIBE_MAX_WORD_GAP_SEC", "4")
    monkeypatch.setenv("TRANSCRIBE_GAP_CHECK_MIN_WORDS", "2")
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_FILTER", "false")

    calls: list[str] = []
    primary_entries = [
        (0.90, 1.10, "Plan"),
        (1.10, 1.24, "C"),
        (1.24, 1.45, "for"),
        (1.45, 1.72, "combat"),
        (2.00, 2.18, "find"),
        (2.18, 2.28, "a"),
        (2.28, 2.56, "solution"),
        (2.56, 2.88, "quickly"),
        (16.00, 16.20, "radio"),
        (16.20, 16.40, "check"),
        (17.00, 17.20, "finish"),
        (18.00, 18.20, "now"),
    ]
    retry_entries = [
        (0.90, 1.10, "Plan"),
        (1.10, 1.24, "A"),
        (1.24, 1.45, "siege"),
        (1.45, 1.72, "plan"),
        (2.00, 2.18, "find"),
        (2.18, 2.28, "a"),
        (2.28, 2.56, "solution"),
        (2.56, 2.88, "quickly"),
        (8.00, 8.20, "chorus"),
        (8.20, 8.40, "line"),
        (9.00, 9.20, "again"),
        (16.00, 16.20, "radio"),
        (16.20, 16.40, "check"),
        (17.00, 17.20, "finish"),
        (18.00, 18.20, "now"),
    ]

    def fake_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str = "whisper-large-v3",
        prompt: str | None = None,
    ) -> TranscriptPayload | None:
        calls.append(model_name)
        if model_name == "whisper-large-v3":
            return _payload_with_entries(primary_entries, source="groq")
        return _payload_with_entries(retry_entries, source="groq")

    monkeypatch.setattr(ts, "_build_from_groq", fake_groq)
    monkeypatch.setattr(
        ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3)
    )

    result = ts.generate_transcript("sample.mp4", 20.0)
    words_lower = [word.text.lower() for word in result.words]
    assert calls == ["whisper-large-v3", "whisper-large-v3-turbo"]
    assert "plan" in words_lower
    assert "c" in words_lower
    assert "siege" not in words_lower
    assert "chorus" in words_lower
    assert len(result.words) > len(primary_entries)


def test_generate_transcript_uses_retry_prompt_for_groq_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "groq")
    monkeypatch.setenv("TRANSCRIBE_GROQ_MODEL", "whisper-large-v3")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MIN_DURATION_SEC", "0")
    monkeypatch.setenv("TRANSCRIBE_GROQ_PROMPT", "")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_PROMPT", "lyrics retry prompt")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT", "false")
    monkeypatch.setenv("TRANSCRIBE_MAX_WORD_GAP_SEC", "4")
    monkeypatch.setenv("TRANSCRIBE_GAP_CHECK_MIN_WORDS", "2")

    seen_calls: list[tuple[str, str | None]] = []

    def fake_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str = "whisper-large-v3",
        prompt: str | None = None,
    ) -> TranscriptPayload | None:
        seen_calls.append((model_name, prompt))
        if model_name == "whisper-large-v3":
            return _payload_with_entries(
                [
                    (0.9, 1.1, "Plan"),
                    (1.1, 1.3, "C"),
                    (1.3, 1.5, "for"),
                    (1.5, 1.7, "combat"),
                    (2.0, 2.2, "find"),
                    (2.2, 2.4, "a"),
                    (2.4, 2.6, "solution"),
                    (2.6, 2.8, "quickly"),
                    (16.0, 16.2, "radio"),
                    (16.2, 16.4, "check"),
                ],
                source="groq",
            )
        return _payload_with_entries(
            [
                (0.9, 1.1, "Plan"),
                (1.1, 1.3, "A"),
                (1.3, 1.5, "siege"),
                (8.0, 8.2, "chorus"),
                (8.2, 8.4, "line"),
                (16.0, 16.2, "radio"),
                (16.2, 16.4, "check"),
            ],
            source="groq",
        )

    monkeypatch.setattr(ts, "_build_from_groq", fake_groq)
    monkeypatch.setattr(
        ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3)
    )

    ts.generate_transcript("sample.mp4", 20.0)
    assert seen_calls[0] == ("whisper-large-v3", None)
    assert seen_calls[1] == ("whisper-large-v3-turbo", "lyrics retry prompt")


def test_generate_transcript_groq_retry_falls_back_to_no_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "groq")
    monkeypatch.setenv("TRANSCRIBE_GROQ_MODEL", "whisper-large-v3")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MIN_DURATION_SEC", "0")
    monkeypatch.setenv("TRANSCRIBE_GROQ_PROMPT", "")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_PROMPT", "lyrics retry prompt")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT", "true")
    monkeypatch.setenv("TRANSCRIBE_MAX_WORD_GAP_SEC", "4")
    monkeypatch.setenv("TRANSCRIBE_GAP_CHECK_MIN_WORDS", "2")
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_FILTER", "false")

    seen_calls: list[tuple[str, str | None]] = []

    def fake_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str = "whisper-large-v3",
        prompt: str | None = None,
    ) -> TranscriptPayload | None:
        seen_calls.append((model_name, prompt))
        if model_name == "whisper-large-v3":
            return _payload_with_entries(
                [
                    (0.9, 1.1, "Plan"),
                    (1.1, 1.3, "C"),
                    (1.3, 1.5, "for"),
                    (1.5, 1.7, "combat"),
                    (2.0, 2.2, "find"),
                    (2.2, 2.4, "a"),
                    (2.4, 2.6, "solution"),
                    (2.6, 2.8, "quickly"),
                    (16.0, 16.2, "radio"),
                    (16.2, 16.4, "check"),
                ],
                source="groq",
            )
        if prompt == "lyrics retry prompt":
            return None
        return _payload_with_entries(
            [
                (0.9, 1.1, "Plan"),
                (1.1, 1.3, "A"),
                (1.3, 1.5, "siege"),
                (8.0, 8.2, "chorus"),
                (8.2, 8.4, "line"),
                (16.0, 16.2, "radio"),
                (16.2, 16.4, "check"),
            ],
            source="groq",
        )

    monkeypatch.setattr(ts, "_build_from_groq", fake_groq)
    monkeypatch.setattr(
        ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3)
    )

    result = ts.generate_transcript("sample.mp4", 20.0)
    words_lower = [word.text.lower() for word in result.words]
    assert seen_calls == [
        ("whisper-large-v3", None),
        ("whisper-large-v3-turbo", "lyrics retry prompt"),
        ("whisper-large-v3-turbo", None),
    ]
    assert "chorus" in words_lower


def test_generate_transcript_retries_groq_on_sparse_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "groq")
    monkeypatch.setenv("TRANSCRIBE_GROQ_MODEL", "whisper-large-v3")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MIN_DURATION_SEC", "0")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT", "false")
    monkeypatch.setenv("TRANSCRIBE_MAX_WORD_GAP_SEC", "40")
    monkeypatch.setenv("TRANSCRIBE_GAP_CHECK_MIN_WORDS", "2")
    monkeypatch.setenv("TRANSCRIBE_SPARSE_WINDOW_SEC", "20")
    monkeypatch.setenv("TRANSCRIBE_SPARSE_WINDOW_MIN_WORDS", "4")
    monkeypatch.setenv("TRANSCRIBE_SPARSE_WINDOW_STEP_SEC", "10")
    monkeypatch.setenv("TRANSCRIBE_SPARSE_WINDOW_START_SEC", "20")
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_FILTER", "false")

    calls: list[str] = []
    primary_entries: list[tuple[float, float, str]] = []
    for idx in range(10):
        start = 0.8 + (idx * 1.8)
        primary_entries.append((start, start + 0.14, f"p{idx}"))
    primary_entries.extend(
        [
            (22.0, 22.2, "s1"),
            (34.0, 34.2, "s2"),
        ]
    )
    for idx in range(34):
        start = 40.0 + (idx * 1.1)
        primary_entries.append((start, start + 0.14, f"q{idx}"))

    retry_entries = list(primary_entries) + [
        (24.0, 24.2, "chorus"),
        (26.0, 26.2, "line"),
        (28.0, 28.2, "again"),
        (30.0, 30.2, "chorus"),
        (32.0, 32.2, "line"),
        (36.0, 36.2, "again"),
    ]

    def fake_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str = "whisper-large-v3",
        prompt: str | None = None,
    ) -> TranscriptPayload | None:
        calls.append(model_name)
        if model_name == "whisper-large-v3":
            return _payload_with_entries(primary_entries, source="groq")
        return _payload_with_entries(retry_entries, source="groq")

    monkeypatch.setattr(ts, "_build_from_groq", fake_groq)
    monkeypatch.setattr(
        ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3)
    )

    result = ts.generate_transcript("sample.mp4", 80.0)
    words_lower = [word.text.lower() for word in result.words]
    assert calls == ["whisper-large-v3", "whisper-large-v3-turbo"]
    assert "chorus" in words_lower
    assert len(result.words) > len(primary_entries)


def test_generate_transcript_accepts_gap_fill_when_added_words_are_meaningful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "groq")
    monkeypatch.setenv("TRANSCRIBE_GROQ_MODEL", "whisper-large-v3")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MIN_DURATION_SEC", "0")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT", "false")
    monkeypatch.setenv("TRANSCRIBE_MAX_WORD_GAP_SEC", "8")
    monkeypatch.setenv("TRANSCRIBE_GAP_FILL_MIN_SEC", "6")
    monkeypatch.setenv("TRANSCRIBE_GAP_CHECK_MIN_WORDS", "2")
    monkeypatch.setenv("TRANSCRIBE_MIN_GAP_FILL_WORDS", "3")
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_FILTER", "false")

    calls: list[str] = []
    primary_entries: list[tuple[float, float, str]] = []
    for idx in range(20):
        start = 1.0 + (idx * 0.6)
        primary_entries.append((start, start + 0.14, f"a{idx}"))
    for idx in range(20):
        start = 46.0 + (idx * 0.6)
        primary_entries.append((start, start + 0.14, f"b{idx}"))
    retry_entries = list(primary_entries) + [
        (24.0, 24.2, "chorus"),
        (24.2, 24.4, "line"),
        (25.0, 25.2, "chorus"),
        (25.2, 25.4, "line"),
    ]

    def fake_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str = "whisper-large-v3",
        prompt: str | None = None,
    ) -> TranscriptPayload | None:
        calls.append(model_name)
        if model_name == "whisper-large-v3":
            return _payload_with_entries(primary_entries, source="groq")
        return _payload_with_entries(retry_entries, source="groq")

    monkeypatch.setattr(ts, "_build_from_groq", fake_groq)
    monkeypatch.setattr(
        ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3)
    )

    result = ts.generate_transcript("sample.mp4", 70.0)
    words_lower = [word.text.lower() for word in result.words]
    assert calls == ["whisper-large-v3", "whisper-large-v3-turbo"]
    assert result.source.endswith("gapfill")
    assert "chorus" in words_lower
    assert len(result.words) > len(primary_entries)


def test_generate_transcript_song_mode_skips_groq_rescue_gap_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "groq")
    monkeypatch.setenv("TRANSCRIBE_ALLOW_MOCK_FALLBACK", "false")
    monkeypatch.setenv("TRANSCRIBE_GROQ_MODEL", "whisper-large-v3")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_GAP_RESCUE", "true")
    monkeypatch.setenv("TRANSCRIBE_RESCUE_ON_LOW_COVERAGE", "true")
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_FILTER", "false")

    primary = _payload_with_entries(
        [
            (1.0, 1.2, "As"),
            (1.2, 1.4, "I"),
            (1.4, 1.7, "walk"),
            (20.0, 20.2, "you"),
            (20.2, 20.4, "and"),
            (20.4, 20.7, "me"),
        ],
        source="groq",
    )

    rescue_called: list[bool] = []

    monkeypatch.setattr(ts, "_build_from_groq", lambda *_args, **_kwargs: primary)
    monkeypatch.setattr(
        ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3)
    )

    def fake_rescue(*_args, **_kwargs):
        rescue_called.append(True)
        return _payload_with_entries(
            [(22.0, 22.4, "hallucinated")],
            source="groq_rescue",
        )

    monkeypatch.setattr(ts, "_call_rescue_groq_gaps", fake_rescue)

    result = ts.generate_transcript("sample.mp4", 40.0, mode="song")

    assert rescue_called == []
    assert [word.text for word in result.words] == [
        "As",
        "I",
        "walk",
        "you",
        "and",
        "me",
    ]


def test_resolve_transcription_profile_auto_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_PROFILE", "auto")
    monkeypatch.setenv("TRANSCRIBE_PROFILE_ANALYZE_SEC", "100")
    monkeypatch.setenv("TRANSCRIBE_PROFILE_MIN_ANALYZE_SEC", "20")
    monkeypatch.setenv("TRANSCRIBE_PROFILE_SPEECH_MIN_SILENCE_RATIO", "0.10")
    monkeypatch.setenv("TRANSCRIBE_PROFILE_MUSIC_MAX_SILENCE_RATIO", "0.04")
    monkeypatch.setattr(
        ts,
        "detect_silence_ranges",
        lambda *_args, **_kwargs: [(0.0, 8.0), (20.0, 28.0), (40.0, 48.0)],
    )

    profile = ts._resolve_transcription_profile(__file__, 180.0)
    assert profile == "speech"


def test_resolve_transcription_profile_auto_music(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_PROFILE", "auto")
    monkeypatch.setenv("TRANSCRIBE_PROFILE_ANALYZE_SEC", "100")
    monkeypatch.setenv("TRANSCRIBE_PROFILE_MIN_ANALYZE_SEC", "20")
    monkeypatch.setenv("TRANSCRIBE_PROFILE_SPEECH_MIN_SILENCE_RATIO", "0.10")
    monkeypatch.setenv("TRANSCRIBE_PROFILE_MUSIC_MAX_SILENCE_RATIO", "0.04")
    monkeypatch.setattr(ts, "detect_silence_ranges", lambda *_args, **_kwargs: [])

    profile = ts._resolve_transcription_profile(__file__, 180.0)
    assert profile == "music"


def test_generate_transcript_speech_profile_disables_lyric_retry_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "groq")
    monkeypatch.setenv("TRANSCRIBE_PROFILE", "speech")
    monkeypatch.setenv("TRANSCRIBE_GROQ_MODEL", "whisper-large-v3")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MIN_DURATION_SEC", "0")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_PROMPT", "lyrics retry prompt")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_PROMPT_SPEECH", "")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT_SPEECH", "false")
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_FILTER", "false")

    seen_calls: list[tuple[str, str | None]] = []

    def fake_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str = "whisper-large-v3",
        prompt: str | None = None,
    ) -> TranscriptPayload | None:
        seen_calls.append((model_name, prompt))
        if model_name == "whisper-large-v3":
            return _payload_with_entries(
                [
                    (1.0, 1.2, "hello"),
                    (2.0, 2.2, "world"),
                    (12.0, 12.2, "again"),
                    (18.0, 18.2, "done"),
                ],
                source="groq",
            )
        return _payload_with_entries(
            [
                (1.0, 1.2, "hello"),
                (2.0, 2.2, "world"),
                (6.0, 6.2, "extra"),
                (8.0, 8.2, "speech"),
                (12.0, 12.2, "again"),
                (18.0, 18.2, "done"),
            ],
            source="groq",
        )

    monkeypatch.setattr(ts, "_build_from_groq", fake_groq)
    monkeypatch.setattr(
        ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3)
    )

    ts.generate_transcript("sample.mp4", 20.0)
    assert seen_calls == [
        ("whisper-large-v3", None),
        ("whisper-large-v3-turbo", None),
    ]


def test_generate_transcript_music_profile_uses_lyric_prompt_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "groq")
    monkeypatch.setenv("TRANSCRIBE_PROFILE", "music")
    monkeypatch.setenv("TRANSCRIBE_GROQ_MODEL", "whisper-large-v3")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MIN_DURATION_SEC", "0")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_PROMPT", "")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_PROMPT_MUSIC", "")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT_MUSIC", "false")
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_FILTER", "false")

    seen_calls: list[tuple[str, str | None]] = []

    def fake_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str = "whisper-large-v3",
        prompt: str | None = None,
    ) -> TranscriptPayload | None:
        seen_calls.append((model_name, prompt))
        if model_name == "whisper-large-v3":
            return _payload_with_entries(
                [
                    (1.0, 1.2, "plan"),
                    (2.0, 2.2, "c"),
                    (3.0, 3.2, "combat"),
                    (18.0, 18.2, "done"),
                ],
                source="groq",
            )
        return _payload_with_entries(
            [
                (1.0, 1.2, "plan"),
                (2.0, 2.2, "a"),
                (7.0, 7.2, "chorus"),
                (8.0, 8.2, "line"),
                (18.0, 18.2, "done"),
            ],
            source="groq",
        )

    monkeypatch.setattr(ts, "_build_from_groq", fake_groq)
    monkeypatch.setattr(
        ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3)
    )

    ts.generate_transcript("sample.mp4", 20.0)
    assert seen_calls[0] == ("whisper-large-v3", ts.DEFAULT_MUSIC_RETRY_PROMPT)
    assert seen_calls[1] == ("whisper-large-v3-turbo", ts.DEFAULT_MUSIC_RETRY_PROMPT)


def test_generate_transcript_uses_gap_rescue_when_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "groq")
    monkeypatch.setenv("TRANSCRIBE_PROFILE", "music")
    monkeypatch.setenv("TRANSCRIBE_GROQ_MODEL", "whisper-large-v3")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MIN_DURATION_SEC", "0")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT", "false")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT_MUSIC", "false")
    monkeypatch.setenv("TRANSCRIBE_MAX_WORD_GAP_SEC", "4")
    monkeypatch.setenv("TRANSCRIBE_GAP_CHECK_MIN_WORDS", "2")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_GAP_RESCUE", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RESCUE_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RESCUE_PROMPT", "rescue prompt")
    monkeypatch.setenv("TRANSCRIBE_MIN_RESCUE_ADDED_WORDS", "1")

    calls: list[str] = []
    rescue_calls: list[tuple[str, str | None]] = []
    primary = _payload_with_entries(
        [
            (1.0, 1.2, "plan"),
            (1.2, 1.4, "c"),
            (2.0, 2.2, "for"),
            (2.2, 2.4, "combat"),
            (10.0, 10.2, "again"),
            (10.2, 10.4, "and"),
            (11.0, 11.2, "again"),
            (12.0, 12.2, "radio"),
            (13.0, 13.2, "check"),
            (14.0, 14.2, "status"),
            (15.0, 15.2, "final"),
            (18.0, 18.2, "done"),
        ],
        source="groq",
    )
    rescue_payload = _payload_with_entries(
        [
            (1.0, 1.2, "plan"),
            (1.2, 1.4, "c"),
            (2.0, 2.2, "for"),
            (2.2, 2.4, "combat"),
            (5.0, 5.2, "chorus"),
            (6.0, 6.2, "line"),
            (10.0, 10.2, "again"),
            (10.2, 10.4, "and"),
            (11.0, 11.2, "again"),
            (12.0, 12.2, "radio"),
            (13.0, 13.2, "check"),
            (14.0, 14.2, "status"),
            (15.0, 15.2, "final"),
            (18.0, 18.2, "done"),
        ],
        source="groq_gapfill",
    )

    def fake_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str = "whisper-large-v3",
        prompt: str | None = None,
    ) -> TranscriptPayload | None:
        calls.append(model_name)
        return primary

    def fake_rescue(
        _path: str,
        _duration_sec: float,
        _primary: TranscriptPayload,
        *,
        profile: str,
        model_name: str,
        prompt: str | None,
    ) -> TranscriptPayload | None:
        assert profile == "music"
        rescue_calls.append((model_name, prompt))
        return rescue_payload

    monkeypatch.setattr(ts, "_build_from_groq", fake_groq)
    monkeypatch.setattr(ts, "_rescue_groq_gaps", fake_rescue)
    monkeypatch.setattr(
        ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3)
    )

    result = ts.generate_transcript("sample.mp4", 20.0)
    assert calls == ["whisper-large-v3", "whisper-large-v3-turbo"]
    assert rescue_calls == [("whisper-large-v3-turbo", "rescue prompt")]
    assert result.source == "groq_gapfill"
    assert any(word.text.lower() == "chorus" for word in result.words)


def test_generate_transcript_skips_gap_rescue_for_speech_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "groq")
    monkeypatch.setenv("TRANSCRIBE_PROFILE", "speech")
    monkeypatch.setenv("TRANSCRIBE_GROQ_MODEL", "whisper-large-v3")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MIN_DURATION_SEC", "0")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT", "false")
    monkeypatch.setenv("TRANSCRIBE_MAX_WORD_GAP_SEC", "4")
    monkeypatch.setenv("TRANSCRIBE_GAP_CHECK_MIN_WORDS", "2")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_GAP_RESCUE", "true")

    rescue_called = False
    primary = _payload_with_entries(
        [
            (1.0, 1.2, "plan"),
            (1.2, 1.4, "c"),
            (10.0, 10.2, "again"),
            (18.0, 18.2, "done"),
        ],
        source="groq",
    )

    def fake_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str = "whisper-large-v3",
        prompt: str | None = None,
    ) -> TranscriptPayload | None:
        return primary

    def fake_rescue(
        _path: str,
        _duration_sec: float,
        _primary: TranscriptPayload,
        *,
        profile: str,
        model_name: str,
        prompt: str | None,
    ) -> TranscriptPayload | None:
        nonlocal rescue_called
        rescue_called = True
        return None

    monkeypatch.setattr(ts, "_build_from_groq", fake_groq)
    monkeypatch.setattr(ts, "_rescue_groq_gaps", fake_rescue)
    monkeypatch.setattr(
        ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3)
    )

    ts.generate_transcript("sample.mp4", 20.0)
    assert rescue_called is False


def test_generate_transcript_language_hint_retries_with_auto_and_fallbacks_on_low_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "groq")
    monkeypatch.setenv("TRANSCRIBE_GROQ_MODEL", "whisper-large-v3")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_GAP_RESCUE", "false")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE_GUARD_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_MIN_WORDS_PER_SEC", "0.45")
    monkeypatch.setenv("TRANSCRIBE_GROQ_LANGUAGE_FALLBACKS", "kn,ta,te,hi")
    monkeypatch.setenv("TRANSCRIBE_GROQ_LANGUAGE_FALLBACK_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(
        ts, "_resolve_transcription_profile", lambda *_args, **_kwargs: "mixed"
    )

    calls: list[str | None] = []

    def fake_call_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str,
        prompt: str | None,
        language_hint: str | None = None,
    ) -> TranscriptPayload | None:
        del model_name, prompt
        calls.append(language_hint)
        if language_hint is None:
            return _payload(120)
        return _payload(1)

    monkeypatch.setattr(ts, "_call_groq", fake_call_groq)
    monkeypatch.setattr(ts, "_start_groq_audio_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ts, "_finish_groq_audio_session", lambda: None)

    result = ts.generate_transcript("sample.mp4", 120.0, language_hint="kn")
    assert len(result.words) == 120
    # Primary: kn, then auto-detect retry. Explicit language mode avoids
    # cross-language fallback by default.
    assert calls == ["kn", None]


def test_pick_better_transcript_with_language_blocks_cross_script_drift() -> None:
    primary = _payload_with_entries(
        [
            (0.0, 0.2, "ನಮಸ್ಕಾರ"),
            (0.2, 0.4, "ಕನ್ನಡ"),
            (0.4, 0.6, "ಹಾಡಿನ"),
            (0.6, 0.8, "ಸಾಲುಗಳು"),
            (0.8, 1.0, "ಚೆನ್ನಾಗಿದೆ"),
            (1.0, 1.2, "ಸ್ವರಗಳು"),
            (1.2, 1.4, "ಸ್ಪಷ್ಟವಾಗಿವೆ"),
            (1.4, 1.6, "ಇಲ್ಲಿ"),
        ],
        source="groq",
    )
    secondary = _payload_with_entries(
        [
            (0.0, 0.2, "வணக்கம்"),
            (0.2, 0.4, "தமிழ்"),
            (0.4, 0.6, "பாடல்"),
            (0.6, 0.8, "வரிகள்"),
            (0.8, 1.0, "மேலும்"),
            (1.0, 1.2, "சொற்கள்"),
        ],
        source="groq",
    )

    chosen = ts._pick_better_transcript_with_language(primary, secondary, 20.0, "kn")
    assert chosen is primary


def test_build_language_retry_candidates_respects_explicit_language_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TRANSCRIBE_GROQ_EXPLICIT_LANGUAGE_ALLOW_CROSS_FALLBACK", "false"
    )
    assert ts._build_language_retry_candidates("kn", ["ta", "te", "hi"]) == [None]
    monkeypatch.setenv("TRANSCRIBE_GROQ_EXPLICIT_LANGUAGE_ALLOW_CROSS_FALLBACK", "true")
    assert ts._build_language_retry_candidates("kn", ["ta", "te", "hi"]) == [
        None,
        "ta",
        "te",
        "hi",
    ]


def test_generate_transcript_language_guard_retry_on_script_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "groq")
    monkeypatch.setenv("TRANSCRIBE_ALLOW_MOCK_FALLBACK", "false")
    monkeypatch.setenv("TRANSCRIBE_GROQ_MODEL", "whisper-large-v3")
    monkeypatch.setenv("TRANSCRIBE_GROQ_RETRY_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_GAP_RESCUE", "false")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE_GUARD_RETRY", "true")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE_SCRIPT_MIN_ALPHA", "1")
    monkeypatch.setenv("TRANSCRIBE_GROQ_LANGUAGE_FALLBACK_MAX_ATTEMPTS", "0")
    monkeypatch.setenv("TRANSCRIBE_MIN_WORDS_PER_SEC", "0.05")
    monkeypatch.setattr(
        ts, "_resolve_transcription_profile", lambda *_args, **_kwargs: "mixed"
    )

    calls: list[tuple[str | None, str | None]] = []
    mismatched = _payload_with_entries(
        [
            (0.0, 0.2, "வணக்கம்"),
            (0.2, 0.4, "தமிழ்"),
            (0.4, 0.6, "பாடல்"),
            (0.6, 0.8, "வரிகள்"),
            (0.8, 1.0, "மேலும்"),
            (1.0, 1.2, "சொற்கள்"),
        ],
        source="groq",
    )
    recovered = _payload_with_entries(
        [
            (0.0, 0.2, "ನಮಸ್ಕಾರ"),
            (0.2, 0.4, "ಕನ್ನಡ"),
            (0.4, 0.6, "ಹಾಡಿನ"),
            (0.6, 0.8, "ಸಾಲುಗಳು"),
            (0.8, 1.0, "ಇಲ್ಲಿ"),
            (1.0, 1.2, "ಸ್ಪಷ್ಟ"),
        ],
        source="groq",
    )

    def fake_call_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str,
        prompt: str | None,
        language_hint: str | None = None,
    ) -> TranscriptPayload | None:
        del model_name
        calls.append((language_hint, prompt))
        if prompt and "strictly in Kannada" in prompt:
            return recovered
        return mismatched

    monkeypatch.setattr(ts, "_call_groq", fake_call_groq)
    monkeypatch.setattr(ts, "_start_groq_audio_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ts, "_finish_groq_audio_session", lambda: None)

    result = ts.generate_transcript("sample.mp4", 20.0, language_hint="kn")
    assert result is recovered
    assert calls[0][0] == "kn"
    assert calls[1][0] == "kn"
    assert calls[1][1] is not None and "strictly in Kannada" in calls[1][1]


def test_generate_transcript_routes_indic_language_to_sarvam_in_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "auto")
    monkeypatch.setenv("TRANSCRIBE_ROUTER_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIBE_ROUTER_INDIAN_TO_SARVAM", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_GAP_RESCUE", "false")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE_GUARD_RETRY", "false")

    seen = {"sarvam": 0, "groq": 0}

    def fake_sarvam(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str,
        mode: str,
        language_hint: str | None,
        prompt: str | None,
        use_vocal_isolation: bool,
    ) -> TranscriptPayload | None:
        del model_name, mode, prompt, use_vocal_isolation
        seen["sarvam"] += 1
        assert language_hint == "kn"
        entries = [(i * 0.1, i * 0.1 + 0.08, f"ಪದ{i}") for i in range(10)]
        return _payload_with_entries(entries, source="sarvam")

    def fake_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str,
        prompt: str | None,
        language_hint: str | None = None,
    ) -> TranscriptPayload | None:
        del model_name, prompt, language_hint
        seen["groq"] += 1
        return None

    monkeypatch.setattr(ts, "_call_sarvam", fake_sarvam)
    monkeypatch.setattr(ts, "_call_groq", fake_groq)
    monkeypatch.setattr(ts, "_start_groq_audio_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ts, "_finish_groq_audio_session", lambda: None)

    result = ts.generate_transcript("sample.mp4", 6.0, language_hint="kn")
    assert result.source == "sarvam"
    assert seen["sarvam"] == 1
    assert seen["groq"] == 0


def test_generate_transcript_routes_english_to_groq_in_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "auto")
    monkeypatch.setenv("TRANSCRIBE_ROUTER_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIBE_ROUTER_INDIAN_TO_SARVAM", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_GAP_RESCUE", "false")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE_GUARD_RETRY", "false")

    seen = {"sarvam": 0, "groq": 0}

    def fake_sarvam(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str,
        mode: str,
        language_hint: str | None,
        prompt: str | None,
        use_vocal_isolation: bool,
    ) -> TranscriptPayload | None:
        del model_name, mode, language_hint, prompt, use_vocal_isolation
        seen["sarvam"] += 1
        return None

    def fake_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str,
        prompt: str | None,
        language_hint: str | None = None,
    ) -> TranscriptPayload | None:
        del model_name, prompt
        seen["groq"] += 1
        assert language_hint == "en"
        entries = [(i * 0.1, i * 0.1 + 0.08, f"en{i}") for i in range(10)]
        return _payload_with_entries(entries, source="groq")

    monkeypatch.setattr(ts, "_call_sarvam", fake_sarvam)
    monkeypatch.setattr(ts, "_call_groq", fake_groq)
    monkeypatch.setattr(ts, "_start_groq_audio_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ts, "_finish_groq_audio_session", lambda: None)

    result = ts.generate_transcript("sample.mp4", 6.0, language_hint="en")
    assert result.source == "groq"
    assert seen["sarvam"] == 0
    assert seen["groq"] == 1


def test_generate_transcript_auto_probes_detected_indic_language_with_sarvam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRANSCRIBE_LANGUAGE", raising=False)
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "auto")
    monkeypatch.setenv("TRANSCRIBE_ROUTER_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIBE_ROUTER_INDIAN_TO_SARVAM", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_GAP_RESCUE", "false")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE_GUARD_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_AFTER_GROQ", "true")
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("TRANSCRIBE_AUTO_UNKNOWN_TO_SARVAM_FIRST", "false")

    seen = {"sarvam": 0, "groq": 0}

    def fake_sarvam(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str,
        mode: str,
        language_hint: str | None,
        prompt: str | None,
        use_vocal_isolation: bool,
    ) -> TranscriptPayload | None:
        del model_name, mode, prompt, use_vocal_isolation
        seen["sarvam"] += 1
        assert language_hint is None
        entries = [(i * 0.1, i * 0.1 + 0.08, f"ಪದ{i}") for i in range(12)]
        payload = _payload_with_entries(entries, source="sarvam")
        return TranscriptPayload(
            source=payload.source,
            language="kn",
            text=payload.text,
            words=payload.words,
            is_mock=payload.is_mock,
        )

    def fake_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str,
        prompt: str | None,
        language_hint: str | None = None,
    ) -> TranscriptPayload | None:
        del model_name, prompt, language_hint
        seen["groq"] += 1
        payload = _payload_with_entries(
            [(0.0, 0.15, "hello"), (0.15, 0.3, "there")], source="groq"
        )
        return TranscriptPayload(
            source=payload.source,
            language="kn",
            text=payload.text,
            words=payload.words,
            is_mock=payload.is_mock,
        )

    monkeypatch.setattr(
        ts, "_resolve_transcription_profile", lambda *_args, **_kwargs: "music"
    )
    monkeypatch.setattr(ts, "_call_sarvam", fake_sarvam)
    monkeypatch.setattr(ts, "_call_groq", fake_groq)
    monkeypatch.setattr(ts, "_start_groq_audio_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ts, "_finish_groq_audio_session", lambda: None)

    result = ts.generate_transcript("sample.mp4", 6.0)
    assert result.source == "sarvam"
    assert seen["groq"] == 1
    assert seen["sarvam"] == 1


def test_build_auto_sarvam_probe_languages_skips_latin_music_without_indic_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_AFTER_GROQ", "true")
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_LANGUAGES", "hi,kn,ta,te,ml")
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_MAX_LATIN_RATIO", "0.92")

    payload = TranscriptPayload(
        source="groq",
        language="en",
        text="we keep spending most our lives living in gangstas paradise",
        words=_payload(40).words,
        is_mock=False,
    )

    result = ts._build_auto_sarvam_probe_languages(payload, 20.0, "music", None)
    assert result == []


def test_build_auto_sarvam_probe_languages_tries_unknown_before_detected_indic_song(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_AFTER_GROQ", "true")
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_PROBE_UNKNOWN_FIRST", "true")

    payload = TranscriptPayload(
        source="groq",
        language="ta",
        text="வணக்கம் தமிழ் பாடல் வரிகள்",
        words=_payload(40).words,
        is_mock=False,
    )

    result = ts._build_auto_sarvam_probe_languages(payload, 20.0, "music", None)
    assert result[:2] == [None, "ta"]
    assert "kn" in result


def test_normalize_detected_language_maps_provider_language_names_to_codes() -> None:
    assert ts._normalize_detected_language("Tamil") == "ta"
    assert ts._normalize_detected_language("Kannada") == "kn"
    assert ts._normalize_detected_language("kn") == "kn"


def test_generate_transcript_fast_mode_music_auto_prefers_sarvam_over_groq_tamil_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRANSCRIBE_LANGUAGE", raising=False)
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "auto")
    monkeypatch.setenv("TRANSCRIBE_PROFILE", "auto")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_GAP_RESCUE", "false")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE_GUARD_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_AFTER_GROQ", "true")
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("TRANSCRIBE_AUTO_UNKNOWN_TO_SARVAM_FIRST", "false")
    monkeypatch.setenv("TRANSCRIBE_ALLOW_MOCK_FALLBACK", "false")
    monkeypatch.setattr(
        ts,
        "_resolve_transcription_profile",
        lambda _path, _duration: "music",
    )

    groq_payload = TranscriptPayload(
        source="groq",
        language="Tamil",
        text="ஓ பிசய குதுரையுந்து எதை இந்த ஓடிதந்தே",
        words=_payload_with_entries(
            [(idx * 0.4, (idx * 0.4) + 0.35, "வார்த்தை") for idx in range(80)]
        ).words,
        is_mock=False,
    )
    sarvam_payload = TranscriptPayload(
        source="sarvam",
        language="kn",
        text="ಓ ಬಿಸಿಲು ಕುದುರೆಯೊಂದು ಎದೆಯಿಂದ ಓಡಿದಂತೆ",
        words=_payload_with_entries(
            [(idx * 0.4, (idx * 0.4) + 0.35, "ಪದ") for idx in range(80)]
        ).words,
        is_mock=False,
    )

    def fake_groq(*_args, **_kwargs) -> TranscriptPayload:
        return groq_payload

    def fake_sarvam(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str,
        mode: str,
        language_hint: str | None,
        prompt: str | None,
        use_vocal_isolation: bool,
    ) -> TranscriptPayload | None:
        del model_name, mode, prompt, use_vocal_isolation
        if language_hint in {None, "kn"}:
            return sarvam_payload
        return None

    monkeypatch.setattr(ts, "_call_groq", fake_groq)
    monkeypatch.setattr(ts, "_call_sarvam", fake_sarvam)
    monkeypatch.setattr(ts, "_start_groq_audio_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ts, "_finish_groq_audio_session", lambda: None)

    result = ts.generate_transcript(
        "sample.mp4",
        45.0,
        language_hint=None,
        fast_mode=True,
        mode="speech",
    )
    assert result.source == "sarvam"
    assert ts._normalize_detected_language(result.language) == "kn"


def test_detect_indic_script_languages_prefers_kannada_for_mixed_english_text() -> None:
    result = ts._detect_indic_script_languages("hello team ನಮಸ್ಕಾರ ಹೇಗಿದ್ದೀರ welcome back")
    assert result[:1] == ["kn"]


def test_generate_transcript_auto_probes_mixed_script_indic_language_with_sarvam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRANSCRIBE_LANGUAGE", raising=False)
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "auto")
    monkeypatch.setenv("TRANSCRIBE_ROUTER_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIBE_ROUTER_INDIAN_TO_SARVAM", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_GAP_RESCUE", "false")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE_GUARD_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_AFTER_GROQ", "true")
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_ON_MIXED_SCRIPT", "true")
    monkeypatch.setenv("TRANSCRIBE_AUTO_UNKNOWN_TO_SARVAM_FIRST", "false")

    seen = {"sarvam": 0, "groq": 0}

    def fake_sarvam(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str,
        mode: str,
        language_hint: str | None,
        prompt: str | None,
        use_vocal_isolation: bool,
    ) -> TranscriptPayload | None:
        del model_name, mode, prompt, use_vocal_isolation
        seen["sarvam"] += 1
        assert language_hint == "kn"
        entries = [(i * 0.1, i * 0.1 + 0.08, f"ಪದ{i}") for i in range(12)]
        return _payload_with_entries(entries, source="sarvam")

    def fake_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str,
        prompt: str | None,
        language_hint: str | None = None,
    ) -> TranscriptPayload | None:
        del model_name, prompt, language_hint
        seen["groq"] += 1
        payload = _payload_with_entries(
            [
                (0.0, 0.15, "hello"),
                (0.15, 0.30, "team"),
                (0.30, 0.48, "ನಮಸ್ಕಾರ"),
                (0.48, 0.66, "ಮತ್ತೆ"),
            ],
            source="groq",
        )
        return TranscriptPayload(
            source=payload.source,
            language="en",
            text=payload.text,
            words=payload.words,
            is_mock=payload.is_mock,
        )

    monkeypatch.setattr(
        ts, "_resolve_transcription_profile", lambda *_args, **_kwargs: "speech"
    )
    monkeypatch.setattr(ts, "_call_sarvam", fake_sarvam)
    monkeypatch.setattr(ts, "_call_groq", fake_groq)
    monkeypatch.setattr(ts, "_start_groq_audio_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ts, "_finish_groq_audio_session", lambda: None)

    result = ts.generate_transcript("sample.mp4", 6.0)
    assert result.source == "sarvam"
    assert seen["groq"] == 1
    assert seen["sarvam"] == 1


def test_generate_transcript_auto_uses_sarvam_unknown_first_for_unpinned_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRANSCRIBE_LANGUAGE", raising=False)
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "auto")
    monkeypatch.setenv("TRANSCRIBE_ROUTER_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIBE_ROUTER_INDIAN_TO_SARVAM", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_GAP_RESCUE", "false")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE_GUARD_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_ALLOW_MOCK_FALLBACK", "false")
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_AFTER_GROQ", "true")
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_PROBE_UNKNOWN_FIRST", "true")
    monkeypatch.setenv("TRANSCRIBE_MIN_WORDS_PER_SEC", "0.05")

    seen = {"sarvam": 0, "groq": 0}

    def fake_sarvam(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str,
        mode: str,
        language_hint: str | None,
        prompt: str | None,
        use_vocal_isolation: bool,
    ) -> TranscriptPayload | None:
        del model_name, mode, prompt, use_vocal_isolation
        seen["sarvam"] += 1
        assert language_hint is None
        payload = _payload_with_entries(
            [
                (0.0, 0.2, "ನಮಸ್ಕಾರ"),
                (0.2, 0.4, "ಕನ್ನಡ"),
                (0.4, 0.6, "ಹಾಡಿನ"),
                (0.6, 0.8, "ಸಾಲುಗಳು"),
                (0.8, 1.0, "ಇಲ್ಲಿ"),
                (1.0, 1.2, "ಸ್ಪಷ್ಟ"),
                (1.2, 1.4, "ಸ್ವರ"),
                (1.4, 1.6, "ಇದೆ"),
                (1.6, 1.8, "ಮತ್ತೆ"),
                (1.8, 2.0, "ಬರಲಿ"),
            ],
            source="sarvam",
        )
        return TranscriptPayload(
            source=payload.source,
            language="kn",
            text=payload.text,
            words=payload.words,
            is_mock=payload.is_mock,
        )

    def fake_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str,
        prompt: str | None,
        language_hint: str | None = None,
    ) -> TranscriptPayload | None:
        del model_name, prompt, language_hint
        seen["groq"] += 1
        payload = _payload_with_entries(
            [
                (0.0, 0.2, "வணக்கம்"),
                (0.2, 0.4, "தமிழ்"),
                (0.4, 0.6, "பாடல்"),
                (0.6, 0.8, "வரிகள்"),
                (0.8, 1.0, "மேலும்"),
                (1.0, 1.2, "சொற்கள்"),
                (1.2, 1.4, "இங்கே"),
                (1.4, 1.6, "தான்"),
                (1.6, 1.8, "மீண்டும்"),
                (1.8, 2.0, "வரும்"),
            ],
            source="groq",
        )
        return TranscriptPayload(
            source=payload.source,
            language="ta",
            text=payload.text,
            words=payload.words,
            is_mock=payload.is_mock,
        )

    monkeypatch.setattr(
        ts, "_resolve_transcription_profile", lambda *_args, **_kwargs: "music"
    )
    monkeypatch.setattr(ts, "_call_sarvam", fake_sarvam)
    monkeypatch.setattr(ts, "_call_groq", fake_groq)
    monkeypatch.setattr(ts, "_start_groq_audio_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ts, "_finish_groq_audio_session", lambda: None)

    result = ts.generate_transcript("sample.mp4", 6.0)
    assert result.source == "sarvam"
    assert result.language == "kn"
    assert seen["groq"] == 0
    assert seen["sarvam"] == 1


def test_generate_transcript_sarvam_low_coverage_falls_back_to_groq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "auto")
    monkeypatch.setenv("TRANSCRIBE_ROUTER_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIBE_ROUTER_INDIAN_TO_SARVAM", "true")
    monkeypatch.setenv("TRANSCRIBE_GROQ_ENABLE_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_GAP_RESCUE", "false")
    monkeypatch.setenv("TRANSCRIBE_GROQ_LANGUAGE_FALLBACK_MAX_ATTEMPTS", "0")
    monkeypatch.setenv("TRANSCRIBE_LANGUAGE_GUARD_RETRY", "false")

    call_order: list[str] = []

    def fake_sarvam(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str,
        mode: str,
        language_hint: str | None,
        prompt: str | None,
        use_vocal_isolation: bool,
    ) -> TranscriptPayload | None:
        del model_name, mode, language_hint, prompt, use_vocal_isolation
        call_order.append("sarvam")
        return _payload_with_entries([(0.0, 0.2, "ಪದ")], source="sarvam")

    def fake_groq(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str,
        prompt: str | None,
        language_hint: str | None = None,
    ) -> TranscriptPayload | None:
        del model_name, prompt, language_hint
        call_order.append("groq")
        return _payload(20)

    monkeypatch.setattr(ts, "_call_sarvam", fake_sarvam)
    monkeypatch.setattr(ts, "_call_groq", fake_groq)
    monkeypatch.setattr(ts, "_start_groq_audio_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ts, "_finish_groq_audio_session", lambda: None)

    result = ts.generate_transcript("sample.mp4", 20.0, language_hint="kn")
    assert len(result.words) == 20
    assert call_order[:2] == ["sarvam", "groq"]


def test_extract_sarvam_word_timestamps_supports_list_payload() -> None:
    payload = {
        "timestamps": [
            {"word": "ನಮಸ್ಕಾರ", "start_time": 0.0, "end_time": 0.4},
            {"word": "ಎಲ್ಲರಿಗೂ", "start_time": 0.4, "end_time": 0.9},
        ]
    }
    words = ts._extract_sarvam_word_timestamps(payload, 2.0)
    assert [item.text for item in words] == ["ನಮಸ್ಕಾರ", "ಎಲ್ಲರಿಗೂ"]
    assert words[0].start_sec == pytest.approx(0.0)
    assert words[1].end_sec == pytest.approx(0.9)


def test_should_fallback_to_text_words_when_timestamps_sparse() -> None:
    sparse_words = [
        TranscriptWordPayload(
            id="w1",
            text="ಓ",
            start_sec=0.0,
            end_sec=0.6,
        )
    ]
    transcript_text = "ಓ ಬಿಸಿಲು ಕುದುರೆಯೊಂದು ಎದೆಯಿಂದ ಓಡಿದಂತೆ"
    assert ts._should_fallback_to_text_words(sparse_words, transcript_text) is True

    dense_words = [
        TranscriptWordPayload(id="w1", text="ಓ", start_sec=0.0, end_sec=0.1),
        TranscriptWordPayload(id="w2", text="ಬಿಸಿಲು", start_sec=0.1, end_sec=0.2),
        TranscriptWordPayload(id="w3", text="ಕುದುರೆಯೊಂದು", start_sec=0.2, end_sec=0.35),
        TranscriptWordPayload(id="w4", text="ಎದೆಯಿಂದ", start_sec=0.35, end_sec=0.5),
        TranscriptWordPayload(id="w5", text="ಓಡಿದಂತೆ", start_sec=0.5, end_sec=0.7),
    ]
    assert ts._should_fallback_to_text_words(dense_words, transcript_text) is False


def test_rescue_groq_gaps_uses_music_profile_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_RESCUE_MIN_GAP_SEC", "10")
    monkeypatch.setenv("TRANSCRIBE_RESCUE_MAX_CHUNKS", "3")
    monkeypatch.setenv("TRANSCRIBE_RESCUE_MAX_WINDOW_SEC", "45")
    monkeypatch.setenv("TRANSCRIBE_RESCUE_PAD_SEC", "0.35")
    monkeypatch.setenv("TRANSCRIBE_RESCUE_MIN_GAP_SEC_MUSIC", "6")
    monkeypatch.setenv("TRANSCRIBE_RESCUE_MAX_CHUNKS_MUSIC", "8")
    monkeypatch.setenv("TRANSCRIBE_RESCUE_MAX_WINDOW_SEC_MUSIC", "12")
    monkeypatch.setenv("TRANSCRIBE_RESCUE_PAD_SEC_MUSIC", "0.4")
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_FILTER", "false")

    primary = _payload_with_entries(
        [
            (0.5, 0.8, "a"),
            (1.0, 1.2, "b"),
            (18.2, 18.5, "c"),
        ],
        source="groq",
    )
    made_windows: list[tuple[float, float]] = []

    def fake_extract(
        _path: str, start_sec: float, end_sec: float
    ) -> tuple[str | None, object | None]:
        made_windows.append((start_sec, end_sec))
        return ("fake-window.mp3", None)

    def fake_build(
        _path: str,
        _duration_sec: float,
        *,
        model_name: str = "whisper-large-v3-turbo",
        prompt: str | None = None,
    ) -> TranscriptPayload | None:
        return _payload_with_entries([(0.1, 0.2, "x"), (0.3, 0.4, "y")], source="groq")

    monkeypatch.setattr(ts, "_extract_audio_window_for_cloud", fake_extract)
    monkeypatch.setattr(ts, "_build_from_groq", fake_build)

    rescued = ts._rescue_groq_gaps(
        "sample.mp4",
        20.0,
        primary,
        profile="music",
        model_name="whisper-large-v3-turbo",
        prompt="p",
    )
    assert rescued is not None
    assert len(made_windows) <= 8
    # With 12s music window, first chunk should be near 0s-12s, not a single 20s span.
    assert made_windows[0][1] - made_windows[0][0] <= 12.0001


def test_rescue_groq_gaps_filters_non_latin_tokens_for_latin_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_RESCUE_SCRIPT_FILTER", "true")
    monkeypatch.setenv("TRANSCRIBE_RESCUE_PRIMARY_MIN_ALPHA", "5")
    monkeypatch.setenv("TRANSCRIBE_RESCUE_PRIMARY_LATIN_RATIO", "0.6")
    monkeypatch.setenv("TRANSCRIBE_RESCUE_TOKEN_LATIN_MIN_RATIO", "0.35")
    monkeypatch.setenv("TRANSCRIBE_RESCUE_MIN_GAP_SEC", "6")
    monkeypatch.setenv("TRANSCRIBE_RESCUE_MAX_CHUNKS", "2")
    monkeypatch.setenv("TRANSCRIBE_RESCUE_MAX_WINDOW_SEC", "12")
    monkeypatch.setenv("TRANSCRIBE_RESCUE_PAD_SEC", "0.35")
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_FILTER", "false")

    primary = _payload_with_entries(
        [
            (1.0, 1.2, "Plan"),
            (1.2, 1.4, "C"),
            (18.0, 18.2, "combat"),
            (18.2, 18.4, "done"),
        ],
        source="groq",
    )

    monkeypatch.setattr(
        ts,
        "_extract_audio_window_for_cloud",
        lambda *_args, **_kwargs: ("fake-window.mp3", None),
    )
    monkeypatch.setattr(
        ts,
        "_build_from_groq",
        lambda *_args, **_kwargs: _payload_with_entries(
            [(0.2, 0.4, "Đăng"), (0.5, 0.7, "chorus"), (0.8, 1.0, "line")],
            source="groq",
        ),
    )

    rescued = ts._rescue_groq_gaps(
        "sample.mp4",
        20.0,
        primary,
        profile="music",
        model_name="whisper-large-v3-turbo",
        prompt="p",
    )
    assert rescued is not None
    lowered = [word.text.lower() for word in rescued.words]
    assert "đăng" not in lowered
    assert "chorus" in lowered
    assert "line" in lowered


def test_generate_transcript_keeps_primary_if_retry_not_better(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_MODEL", "base.en")
    monkeypatch.setenv("TRANSCRIBE_RETRY_MODEL", "medium")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_QUALITY_RETRY", "true")
    monkeypatch.setenv("TRANSCRIBE_MIN_WORDS_PER_SEC", "0.8")
    monkeypatch.setenv("TRANSCRIBE_RETRY_MIN_DURATION_SEC", "0")

    def fake_build(
        _path: str,
        _duration: float,
        *,
        model_name: str | None = None,
        beam_size: int | None = None,
        force_vad_filter: bool | None = None,
    ) -> TranscriptPayload | None:
        if model_name == "base.en":
            return _payload(150)
        return _payload(120)

    monkeypatch.setattr(ts, "_build_from_faster_whisper", fake_build)

    result = ts.generate_transcript("sample.mp4", 240.0)
    assert len(result.words) == 150


def test_generate_transcript_skips_retry_for_short_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_MODEL", "base.en")
    monkeypatch.setenv("TRANSCRIBE_RETRY_MODEL", "medium")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_QUALITY_RETRY", "true")
    monkeypatch.setenv("TRANSCRIBE_MIN_WORDS_PER_SEC", "0.45")
    monkeypatch.setenv("TRANSCRIBE_RETRY_MIN_DURATION_SEC", "120")

    calls: list[str | None] = []

    def fake_build(
        _path: str,
        _duration: float,
        *,
        model_name: str | None = None,
        beam_size: int | None = None,
        force_vad_filter: bool | None = None,
    ) -> TranscriptPayload | None:
        calls.append(model_name)
        return _payload(3)

    monkeypatch.setattr(ts, "_build_from_faster_whisper", fake_build)

    result = ts.generate_transcript("sample.mp4", 6.0)
    assert len(result.words) == 3
    assert calls == ["base.en"]


def test_generate_transcript_song_mode_skips_profile_resolution_and_disables_vad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_MODEL", "base.en")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_QUALITY_RETRY", "false")

    seen_force_vad: list[bool | None] = []

    def fail_profile(*_args, **_kwargs):
        raise AssertionError("song mode should not run auto profile resolution")

    def fake_build(
        _path: str,
        _duration: float,
        *,
        model_name: str | None = None,
        beam_size: int | None = None,
        force_vad_filter: bool | None = None,
    ) -> TranscriptPayload | None:
        del _path, _duration, model_name, beam_size
        seen_force_vad.append(force_vad_filter)
        return _payload(24)

    monkeypatch.setattr(ts, "_resolve_transcription_profile", fail_profile)
    monkeypatch.setattr(ts, "_build_from_faster_whisper", fake_build)

    result = ts.generate_transcript("sample.mp4", 18.0, mode="song")
    assert len(result.words) == 24
    assert seen_force_vad == [False]


def test_generate_transcript_speed_optimized_skips_local_quality_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_MODEL", "base.en")
    monkeypatch.setenv("TRANSCRIBE_RETRY_MODEL", "medium")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_QUALITY_RETRY", "true")
    monkeypatch.setenv("TRANSCRIBE_MIN_WORDS_PER_SEC", "0.8")
    monkeypatch.setenv("TRANSCRIBE_RETRY_MIN_DURATION_SEC", "0")

    calls: list[str | None] = []

    def fake_build(
        _path: str,
        _duration: float,
        *,
        model_name: str | None = None,
        beam_size: int | None = None,
        force_vad_filter: bool | None = None,
    ) -> TranscriptPayload | None:
        del _path, _duration, beam_size, force_vad_filter
        calls.append(model_name)
        return _payload(3)

    monkeypatch.setattr(ts, "_build_from_faster_whisper", fake_build)

    result = ts.generate_transcript(
        "sample.mp4",
        30.0,
        mode="auto",
        optimize_for_speed=True,
    )
    assert len(result.words) == 3
    assert calls == ["base.en"]


def test_generate_transcript_raises_when_mock_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_MODEL", "large-v3")
    monkeypatch.setenv("TRANSCRIBE_RETRY_MODEL", "large-v3")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_QUALITY_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_ALLOW_MOCK_FALLBACK", "false")
    monkeypatch.setattr(
        ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: None
    )

    with pytest.raises(RuntimeError, match="Transcription failed"):
        ts.generate_transcript("missing.mp4", 12.0)


def test_generate_transcript_disables_mock_fallback_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_MODEL", "large-v3")
    monkeypatch.setenv("TRANSCRIBE_RETRY_MODEL", "large-v3")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_QUALITY_RETRY", "false")
    monkeypatch.delenv("TRANSCRIBE_ALLOW_MOCK_FALLBACK", raising=False)
    monkeypatch.setattr(
        ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: None
    )

    with pytest.raises(RuntimeError, match="Transcription failed"):
        ts.generate_transcript("missing.mp4", 12.0)


def test_normalize_words_clamps_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_WORD_MIN_CONFIDENCE", "0")
    words = [
        TranscriptWordPayload(
            id="a", text="hello", start_sec=0.0, end_sec=0.5, confidence=1.5
        ),
        TranscriptWordPayload(
            id="b", text="world", start_sec=0.5, end_sec=1.0, confidence=-0.3
        ),
        TranscriptWordPayload(
            id="c", text="skip", start_sec=1.0, end_sec=1.4, confidence=math.nan
        ),
    ]
    normalized = ts._normalize_words(words, 2.0)
    assert normalized[0].confidence == 1.0
    assert normalized[1].confidence == 0.0
    assert normalized[2].confidence is None


def test_normalize_words_clamps_pathological_word_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_WORD_MIN_CONFIDENCE", "0")
    monkeypatch.setenv("TRANSCRIBE_MAX_WORD_DURATION_SEC", "1.2")
    monkeypatch.setenv("TRANSCRIBE_WORD_NEXT_GUARD_SEC", "0.01")
    monkeypatch.setenv("TRANSCRIBE_TIMESTAMP_OFFSET_SEC", "0")
    words = [
        TranscriptWordPayload(id="a", text="As", start_sec=0.02, end_sec=7.82),
        TranscriptWordPayload(id="b", text="I", start_sec=7.82, end_sec=26.44),
        TranscriptWordPayload(id="c", text="walk", start_sec=26.4, end_sec=26.72),
    ]
    normalized = ts._normalize_words(words, 30.0)
    assert len(normalized) == 3
    assert normalized[0].end_sec == pytest.approx(1.22, abs=0.001)
    assert normalized[1].end_sec == pytest.approx(9.02, abs=0.001)
    assert (normalized[0].end_sec - normalized[0].start_sec) <= 1.2 + 1e-6
    assert (normalized[1].end_sec - normalized[1].start_sec) <= 1.2 + 1e-6


def test_resolve_device_and_compute_type_auto_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_DEVICE", "auto")
    monkeypatch.setenv("TRANSCRIBE_COMPUTE_TYPE", "auto")
    monkeypatch.setenv("TRANSCRIBE_COMPUTE_TYPE_CPU", "int8")
    monkeypatch.setattr(ts, "_gpu_available", lambda: False)
    device, compute_type = ts._resolve_device_and_compute_type()
    assert device == "cpu"
    assert compute_type == "int8"


def test_resolve_device_and_compute_type_auto_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_DEVICE", "auto")
    monkeypatch.setenv("TRANSCRIBE_COMPUTE_TYPE", "auto")
    monkeypatch.setenv("TRANSCRIBE_COMPUTE_TYPE_CUDA", "float16")
    monkeypatch.setattr(ts, "_gpu_available", lambda: True)
    device, compute_type = ts._resolve_device_and_compute_type()
    assert device == "cuda"
    assert compute_type == "float16"


def test_extract_audio_for_cloud_uses_isolated_vocals_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    isolated = tmp_path / "isolated-vocals.wav"
    isolated.write_bytes(b"vocals")
    cleanup_root = tmp_path / "vocal-work"
    cleanup_root.mkdir(parents=True, exist_ok=True)
    (cleanup_root / "marker.txt").write_text("x")
    monkeypatch.setenv("TMP_DIR", str(tmp_path))
    monkeypatch.setattr(
        ts,
        "_prepare_vocal_isolation_source",
        lambda _path: (str(isolated), cleanup_root),
    )

    seen_inputs: list[str] = []

    def fake_run(
        cmd: list[str],
        capture_output: bool,
        text: bool,
        check: bool = False,
        timeout: int | None = None,
    ):
        del capture_output, text, check, timeout
        input_index = cmd.index("-i") + 1
        seen_inputs.append(cmd[input_index])
        Path(cmd[-1]).write_bytes(b"mp3")
        return type("Proc", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(ts.subprocess, "run", fake_run)

    out_path, cleanup = ts._extract_audio_for_cloud(str(source))
    assert seen_inputs == [str(isolated)]
    assert cleanup is not None
    assert out_path == str(cleanup)
    assert Path(out_path).exists()
    assert not cleanup_root.exists()


def test_extract_audio_for_cloud_falls_back_to_original_when_isolated_source_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    missing_isolated = tmp_path / "missing-vocals.wav"
    cleanup_root = tmp_path / "vocal-work"
    cleanup_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TMP_DIR", str(tmp_path))
    monkeypatch.setattr(
        ts,
        "_prepare_vocal_isolation_source",
        lambda _path: (str(missing_isolated), cleanup_root),
    )

    seen_inputs: list[str] = []

    def fake_run(
        cmd: list[str],
        capture_output: bool,
        text: bool,
        check: bool = False,
        timeout: int | None = None,
    ):
        del capture_output, text, check, timeout
        input_index = cmd.index("-i") + 1
        seen_inputs.append(cmd[input_index])
        Path(cmd[-1]).write_bytes(b"mp3")
        return type("Proc", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(ts.subprocess, "run", fake_run)

    out_path, cleanup = ts._extract_audio_for_cloud(str(source))
    assert seen_inputs == [str(source)]
    assert cleanup is not None
    assert out_path == str(cleanup)
    assert Path(out_path).exists()
    assert not cleanup_root.exists()


def test_extract_audio_for_cloud_skips_reprocessing_for_groq_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    window_mp3 = tmp_path / "groq-window-test.mp3"
    window_mp3.write_bytes(b"window-audio")
    called = False

    def fake_prepare(_path: str) -> tuple[str, Path | None]:
        nonlocal called
        called = True
        return (_path, None)

    monkeypatch.setattr(ts, "_prepare_vocal_isolation_source", fake_prepare)

    out_path, cleanup = ts._extract_audio_for_cloud(str(window_mp3))
    assert out_path == str(window_mp3)
    assert cleanup is None
    assert called is False


def test_resolve_groq_input_source_reuses_shared_audio_in_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    prepared = tmp_path / "prepared.mp3"
    prepared.write_bytes(b"audio")

    calls = 0

    def fake_extract(
        _path: str, *, use_vocal_isolation: bool = True
    ) -> tuple[str, Path | None]:
        del use_vocal_isolation
        nonlocal calls
        calls += 1
        return (str(prepared), prepared)

    monkeypatch.setattr(ts, "_extract_audio_for_cloud", fake_extract)

    ts._start_groq_audio_session(str(source))
    try:
        first_path, first_cleanup = ts._resolve_groq_input_source(str(source))
        second_path, second_cleanup = ts._resolve_groq_input_source(str(source))
        assert first_path == str(prepared)
        assert second_path == str(prepared)
        assert first_cleanup is None
        assert second_cleanup is None
        assert calls == 1
    finally:
        ts._finish_groq_audio_session()

    assert not prepared.exists()


def test_resolve_groq_input_source_passes_session_vocal_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    prepared = tmp_path / "prepared.mp3"
    prepared.write_bytes(b"audio")
    seen_flags: list[bool] = []

    def fake_extract(
        _path: str, *, use_vocal_isolation: bool = True
    ) -> tuple[str, Path | None]:
        seen_flags.append(use_vocal_isolation)
        return (str(prepared), prepared)

    monkeypatch.setattr(ts, "_extract_audio_for_cloud", fake_extract)

    ts._start_groq_audio_session(str(source), use_vocal_isolation=False)
    try:
        path, cleanup = ts._resolve_groq_input_source(str(source))
        assert path == str(prepared)
        assert cleanup is None
    finally:
        ts._finish_groq_audio_session()

    assert seen_flags == [False]


def test_vocal_isolation_allowed_for_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_PROFILES", "music,mixed")
    assert ts._vocal_isolation_allowed_for_profile("music") is True
    assert ts._vocal_isolation_allowed_for_profile("mixed") is True
    assert ts._vocal_isolation_allowed_for_profile("speech") is False


def test_prepare_vocal_isolation_source_dispatches_alias_backends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_HIGH_QUALITY", "false")
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    command_out = tmp_path / "command.wav"
    command_out.write_bytes(b"vocals")
    api_out = tmp_path / "api.wav"
    api_out.write_bytes(b"vocals")

    seen: list[tuple[str, str]] = []

    def fake_command(_path: str, *, backend: str) -> tuple[str, Path | None]:
        seen.append(("command", backend))
        return (str(command_out), None)

    def fake_api(_path: str, *, backend: str) -> tuple[str, Path | None]:
        seen.append(("api", backend))
        return (str(api_out), None)

    monkeypatch.setattr(ts, "_prepare_vocal_stem_with_command", fake_command)
    monkeypatch.setattr(ts, "_prepare_vocal_stem_with_api", fake_api)

    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_BACKEND", "bs-roformer")
    assert ts._prepare_vocal_isolation_source(str(source))[0] == str(command_out)

    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_BACKEND", "mdx23c_api")
    assert ts._prepare_vocal_isolation_source(str(source))[0] == str(api_out)

    assert seen == [("command", "bs_roformer"), ("api", "mdx23c_api")]


def test_prepare_vocal_stem_with_command_uses_template_and_output_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    monkeypatch.setenv("TMP_DIR", str(tmp_path))
    monkeypatch.setenv(
        "TRANSCRIBE_VOCAL_ISOLATION_COMMAND_BS_ROFORMER",
        "separator --in {input} --out {output_dir} --model {model} --device {device} --stem {stem}",
    )
    monkeypatch.setenv(
        "TRANSCRIBE_VOCAL_ISOLATION_COMMAND_OUTPUT_BS_ROFORMER",
        "vocals.wav",
    )

    seen_cmds: list[list[str]] = []

    def fake_run(
        cmd: list[str],
        capture_output: bool,
        text: bool,
        check: bool = False,
        timeout: int | None = None,
        cwd: str | None = None,
    ):
        del capture_output, text, check, timeout
        seen_cmds.append(cmd)
        assert cwd is not None
        output_path = Path(cwd) / "out" / "vocals.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"isolated")
        return type("Proc", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(ts.subprocess, "run", fake_run)

    isolated_path, cleanup = ts._prepare_vocal_stem_with_command(
        str(source), backend="bs_roformer"
    )
    assert cleanup is not None
    assert Path(isolated_path).exists()
    assert Path(isolated_path).read_bytes() == b"isolated"
    assert seen_cmds
    joined = " ".join(seen_cmds[0])
    assert "bs_roformer" in joined
    assert "vocals" in joined


def test_prepare_vocal_stem_with_command_uses_absolute_paths_for_relative_tmp_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")

    # Simulate default project config where TMP_DIR is relative ("./tmp").
    monkeypatch.setenv("TMP_DIR", "tmp")
    monkeypatch.setenv(
        "TRANSCRIBE_VOCAL_ISOLATION_COMMAND_BS_ROFORMER",
        "separator --in {input} --out {output_dir} --model {model}",
    )
    monkeypatch.setenv(
        "TRANSCRIBE_VOCAL_ISOLATION_COMMAND_OUTPUT_BS_ROFORMER",
        "vocals.wav",
    )

    seen_cmds: list[list[str]] = []

    def fake_run(
        cmd: list[str],
        capture_output: bool,
        text: bool,
        check: bool = False,
        timeout: int | None = None,
        cwd: str | None = None,
    ):
        del capture_output, text, check, timeout
        seen_cmds.append(cmd)
        assert cwd is not None
        out_flag_index = cmd.index("--out") + 1
        input_flag_index = cmd.index("--in") + 1
        out_path = Path(cmd[out_flag_index])
        in_path = Path(cmd[input_flag_index])
        assert out_path.is_absolute()
        assert in_path.is_absolute()
        output_path = out_path / "vocals.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"isolated")
        return type("Proc", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(ts.subprocess, "run", fake_run)

    isolated_path, cleanup = ts._prepare_vocal_stem_with_command(
        str(source), backend="bs_roformer"
    )
    assert cleanup is not None
    assert Path(isolated_path).exists()
    assert Path(isolated_path).read_bytes() == b"isolated"
    assert seen_cmds


def test_prepare_vocal_stem_with_api_accepts_base64_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    monkeypatch.setenv("TMP_DIR", str(tmp_path))
    monkeypatch.setenv(
        "TRANSCRIBE_VOCAL_ISOLATION_API_URL", "https://example.test/isolate"
    )
    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_API_BASE64_FIELD", "audio_base64")

    class FakeResponse:
        status_code = 200

        def __init__(self) -> None:
            self.headers = {"content-type": "application/json"}
            self.content = b""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            encoded = base64.b64encode(b"voice-only").decode("ascii")
            return {"audio_base64": encoded}

    class FakeClient:
        def __init__(self, timeout: float, follow_redirects: bool) -> None:
            assert timeout >= 5.0
            assert follow_redirects is True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        def post(self, url: str, data=None, files=None, headers=None) -> FakeResponse:
            del data, files, headers
            assert url == "https://example.test/isolate"
            return FakeResponse()

        def get(self, *_args, **_kwargs):
            raise AssertionError("get should not be called for base64 response")

    fake_httpx = type("FakeHttpx", (), {"Client": FakeClient})
    monkeypatch.setitem(ts.sys.modules, "httpx", fake_httpx)

    isolated_path, cleanup = ts._prepare_vocal_stem_with_api(str(source), backend="api")
    assert cleanup is not None
    assert Path(isolated_path).exists()
    assert Path(isolated_path).read_bytes() == b"voice-only"


def test_prepare_vocal_stem_with_api_skips_placeholder_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_API_URL_MDX23C", "YOUR_ENDPOINT")

    isolated_path, cleanup = ts._prepare_vocal_stem_with_api(
        str(source), backend="mdx23c_api"
    )
    assert isolated_path == str(source)
    assert cleanup is None


def test_prepare_vocal_isolation_source_auto_prefers_configured_command_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    isolated = tmp_path / "bs-vocals.wav"
    isolated.write_bytes(b"vocals")

    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_HIGH_QUALITY", "false")
    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_BACKEND", "auto")
    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_COMMAND_MELBAND_ROFORMER", "")
    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_COMMAND_HTDEMUCS_FT", "")
    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_COMMAND_MDX23C", "")
    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_COMMAND", "")
    monkeypatch.setenv(
        "TRANSCRIBE_VOCAL_ISOLATION_COMMAND_BS_ROFORMER", "separator {input}"
    )

    seen_backends: list[str] = []

    def fake_command(_path: str, *, backend: str) -> tuple[str, Path | None]:
        seen_backends.append(backend)
        return (str(isolated), None)

    monkeypatch.setattr(ts, "_prepare_vocal_stem_with_command", fake_command)

    prepared, cleanup = ts._prepare_vocal_isolation_source(str(source))
    assert prepared == str(isolated)
    assert cleanup is None
    assert seen_backends == ["bs_roformer"]


def test_prepare_vocal_isolation_source_uses_fallback_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    isolated = tmp_path / "fallback-vocals.wav"
    isolated.write_bytes(b"vocals")

    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_HIGH_QUALITY", "false")
    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_BACKEND", "unsupported_backend")
    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_FALLBACKS", "command,api")

    seen: list[str] = []

    def fake_command(_path: str, *, backend: str) -> tuple[str, Path | None]:
        seen.append(backend)
        return (str(isolated), None)

    monkeypatch.setattr(ts, "_prepare_vocal_stem_with_command", fake_command)

    prepared, cleanup = ts._prepare_vocal_isolation_source(str(source))
    assert prepared == str(isolated)
    assert cleanup is None
    assert seen == ["command"]


def test_prepare_vocal_isolation_source_uses_precomputed_stem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When a pre-computed vocal stem exists, use it instead of running isolation."""
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")

    # Create pre-computed vocal stem with expected naming convention
    precomputed = tmp_path / "video_vocals.wav"
    precomputed.write_bytes(b"precomputed vocals")

    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_ENABLED", "true")

    # Track if backend isolation was called (it should NOT be)
    backend_called = []
    original_prepare = ts._prepare_with_vocal_backend

    def track_backend_call(*args, **kwargs):
        backend_called.append(True)
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(ts, "_prepare_with_vocal_backend", track_backend_call)

    prepared, cleanup = ts._prepare_vocal_isolation_source(str(source))

    # Should use pre-computed stem
    assert prepared == str(precomputed)
    assert cleanup is None
    # Backend should NOT have been called
    assert len(backend_called) == 0


def test_precompute_vocal_isolation_saves_to_output_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """precompute_vocal_isolation should save the vocal stem to the specified output dir."""
    source = tmp_path / "input" / "video.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"video")

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a fake isolated stem in a temp location
    isolated_stem = tmp_path / "temp" / "vocals.wav"
    isolated_stem.parent.mkdir(parents=True, exist_ok=True)
    isolated_stem.write_bytes(b"isolated vocals")

    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_BACKEND", "command")

    def fake_backend(_path: str, backend: str) -> tuple[str, Path | None]:
        return (str(isolated_stem), isolated_stem.parent)

    monkeypatch.setattr(ts, "_prepare_with_vocal_backend", fake_backend)

    result = ts.precompute_vocal_isolation(str(source), str(output_dir))

    assert result == "video_vocals.wav"
    final_path = output_dir / "video_vocals.wav"
    assert final_path.exists()
    assert final_path.read_bytes() == b"isolated vocals"


def test_get_precomputed_vocal_path_returns_path_if_exists(tmp_path: Path) -> None:
    """get_precomputed_vocal_path should return the path if the pre-computed stem exists."""
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")

    precomputed = tmp_path / "video_vocals.wav"
    precomputed.write_bytes(b"vocals")

    result = ts.get_precomputed_vocal_path(str(source), str(tmp_path))
    assert result == str(precomputed)


def test_get_precomputed_vocal_path_returns_none_if_not_exists(tmp_path: Path) -> None:
    """get_precomputed_vocal_path should return None if the stem does not exist."""
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")

    result = ts.get_precomputed_vocal_path(str(source), str(tmp_path))
    assert result is None
