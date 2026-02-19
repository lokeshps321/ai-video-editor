import os
import math

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")
os.environ["TRANSCRIBE_BACKEND"] = "local"

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


def _payload_with_times(times: list[tuple[float, float]], *, source: str = "groq") -> TranscriptPayload:
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


def _payload_with_entries(entries: list[tuple[float, float, str]], *, source: str = "groq") -> TranscriptPayload:
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


def test_generate_transcript_retries_low_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_generate_transcript_retry_when_primary_fails(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_generate_transcript_retries_groq_on_long_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "groq")
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
            return _payload_with_times([(1.0, 2.0), (10.0, 11.0), (20.0, 21.0), (45.0, 46.0)], source="groq")
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
    monkeypatch.setattr(ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3))

    result = ts.generate_transcript("sample.mp4", 60.0)
    assert len(result.words) == 6
    assert calls == ["whisper-large-v3-turbo", "whisper-large-v3"]


def test_generate_transcript_groq_gap_fill_preserves_primary_dialogue(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3))

    result = ts.generate_transcript("sample.mp4", 20.0)
    words_lower = [word.text.lower() for word in result.words]
    assert calls == ["whisper-large-v3", "whisper-large-v3-turbo"]
    assert "plan" in words_lower
    assert "c" in words_lower
    assert "siege" not in words_lower
    assert "chorus" in words_lower
    assert len(result.words) > len(primary_entries)


def test_generate_transcript_uses_retry_prompt_for_groq_retry(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3))

    ts.generate_transcript("sample.mp4", 20.0)
    assert seen_calls[0] == ("whisper-large-v3", None)
    assert seen_calls[1] == ("whisper-large-v3-turbo", "lyrics retry prompt")


def test_generate_transcript_groq_retry_falls_back_to_no_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3))

    result = ts.generate_transcript("sample.mp4", 20.0)
    words_lower = [word.text.lower() for word in result.words]
    assert seen_calls == [
        ("whisper-large-v3", None),
        ("whisper-large-v3-turbo", "lyrics retry prompt"),
        ("whisper-large-v3-turbo", None),
    ]
    assert "chorus" in words_lower


def test_generate_transcript_retries_groq_on_sparse_window(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3))

    result = ts.generate_transcript("sample.mp4", 80.0)
    words_lower = [word.text.lower() for word in result.words]
    assert calls == ["whisper-large-v3", "whisper-large-v3-turbo"]
    assert "chorus" in words_lower
    assert len(result.words) > len(primary_entries)


def test_generate_transcript_accepts_gap_fill_when_added_words_are_meaningful(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3))

    result = ts.generate_transcript("sample.mp4", 70.0)
    words_lower = [word.text.lower() for word in result.words]
    assert calls == ["whisper-large-v3", "whisper-large-v3-turbo"]
    assert result.source.endswith("gapfill")
    assert "chorus" in words_lower
    assert len(result.words) > len(primary_entries)


def test_resolve_transcription_profile_auto_speech(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_resolve_transcription_profile_auto_music(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_PROFILE", "auto")
    monkeypatch.setenv("TRANSCRIBE_PROFILE_ANALYZE_SEC", "100")
    monkeypatch.setenv("TRANSCRIBE_PROFILE_MIN_ANALYZE_SEC", "20")
    monkeypatch.setenv("TRANSCRIBE_PROFILE_SPEECH_MIN_SILENCE_RATIO", "0.10")
    monkeypatch.setenv("TRANSCRIBE_PROFILE_MUSIC_MAX_SILENCE_RATIO", "0.04")
    monkeypatch.setattr(ts, "detect_silence_ranges", lambda *_args, **_kwargs: [])

    profile = ts._resolve_transcription_profile(__file__, 180.0)
    assert profile == "music"


def test_generate_transcript_speech_profile_disables_lyric_retry_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
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
                [(1.0, 1.2, "hello"), (2.0, 2.2, "world"), (12.0, 12.2, "again"), (18.0, 18.2, "done")],
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
    monkeypatch.setattr(ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3))

    ts.generate_transcript("sample.mp4", 20.0)
    assert seen_calls == [
        ("whisper-large-v3", None),
        ("whisper-large-v3-turbo", None),
    ]


def test_generate_transcript_music_profile_uses_default_music_retry_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
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
                [(1.0, 1.2, "plan"), (2.0, 2.2, "c"), (3.0, 3.2, "combat"), (18.0, 18.2, "done")],
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
    monkeypatch.setattr(ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3))

    ts.generate_transcript("sample.mp4", 20.0)
    assert seen_calls[0] == ("whisper-large-v3", None)
    assert seen_calls[1] == ("whisper-large-v3-turbo", ts.DEFAULT_MUSIC_RETRY_PROMPT)


def test_generate_transcript_uses_gap_rescue_when_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3))

    result = ts.generate_transcript("sample.mp4", 20.0)
    assert calls == ["whisper-large-v3", "whisper-large-v3-turbo"]
    assert rescue_calls == [("whisper-large-v3-turbo", "rescue prompt")]
    assert result.source == "groq_gapfill"
    assert any(word.text.lower() == "chorus" for word in result.words)


def test_generate_transcript_skips_gap_rescue_for_speech_profile(monkeypatch: pytest.MonkeyPatch) -> None:
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
        [(1.0, 1.2, "plan"), (1.2, 1.4, "c"), (10.0, 10.2, "again"), (18.0, 18.2, "done")],
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
    monkeypatch.setattr(ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: _payload(3))

    ts.generate_transcript("sample.mp4", 20.0)
    assert rescue_called is False


def test_rescue_groq_gaps_uses_music_profile_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def fake_extract(_path: str, start_sec: float, end_sec: float) -> tuple[str | None, object | None]:
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


def test_rescue_groq_gaps_filters_non_latin_tokens_for_latin_primary(monkeypatch: pytest.MonkeyPatch) -> None:
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
        [(1.0, 1.2, "Plan"), (1.2, 1.4, "C"), (18.0, 18.2, "combat"), (18.2, 18.4, "done")],
        source="groq",
    )

    monkeypatch.setattr(ts, "_extract_audio_window_for_cloud", lambda *_args, **_kwargs: ("fake-window.mp3", None))
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


def test_generate_transcript_keeps_primary_if_retry_not_better(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_generate_transcript_skips_retry_for_short_duration(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_generate_transcript_raises_when_mock_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_MODEL", "large-v3")
    monkeypatch.setenv("TRANSCRIBE_RETRY_MODEL", "large-v3")
    monkeypatch.setenv("TRANSCRIBE_ENABLE_QUALITY_RETRY", "false")
    monkeypatch.setenv("TRANSCRIBE_ALLOW_MOCK_FALLBACK", "false")
    monkeypatch.setattr(ts, "_build_from_faster_whisper", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="Transcription failed"):
        ts.generate_transcript("missing.mp4", 12.0)


def test_normalize_words_clamps_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_WORD_MIN_CONFIDENCE", "0")
    words = [
        TranscriptWordPayload(id="a", text="hello", start_sec=0.0, end_sec=0.5, confidence=1.5),
        TranscriptWordPayload(id="b", text="world", start_sec=0.5, end_sec=1.0, confidence=-0.3),
        TranscriptWordPayload(id="c", text="skip", start_sec=1.0, end_sec=1.4, confidence=math.nan),
    ]
    normalized = ts._normalize_words(words, 2.0)
    assert normalized[0].confidence == 1.0
    assert normalized[1].confidence == 0.0
    assert normalized[2].confidence is None


def test_resolve_device_and_compute_type_auto_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_DEVICE", "auto")
    monkeypatch.setenv("TRANSCRIBE_COMPUTE_TYPE", "auto")
    monkeypatch.setenv("TRANSCRIBE_COMPUTE_TYPE_CPU", "int8")
    monkeypatch.setattr(ts, "_gpu_available", lambda: False)
    device, compute_type = ts._resolve_device_and_compute_type()
    assert device == "cpu"
    assert compute_type == "int8"


def test_resolve_device_and_compute_type_auto_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_DEVICE", "auto")
    monkeypatch.setenv("TRANSCRIBE_COMPUTE_TYPE", "auto")
    monkeypatch.setenv("TRANSCRIBE_COMPUTE_TYPE_CUDA", "float16")
    monkeypatch.setattr(ts, "_gpu_available", lambda: True)
    device, compute_type = ts._resolve_device_and_compute_type()
    assert device == "cuda"
    assert compute_type == "float16"
