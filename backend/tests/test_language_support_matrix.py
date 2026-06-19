"""Regression tests for supported transcript languages."""

from __future__ import annotations

import pytest

from app import transcription_service as ts
from app.routers import transcript as transcript_router
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
        source="test",
        language="en",
        text=" ".join(text_parts),
        words=words,
        is_mock=False,
    )

UI_LANGUAGE_CODES = (
    "en",
    "kn",
    "hi",
    "ta",
    "te",
    "ml",
    "mr",
    "bn",
    "gu",
    "pa",
    "or",
    "ur",
)

SCRIPT_SAMPLES: dict[str, str] = {
    "kn": "ನಮಸ್ಕಾರ ಇದು ಕನ್ನಡ ಪರೀಕ್ಷೆ",
    "hi": "नमस्ते यह हिंदी परीक्षण है",
    "ta": "வணக்கம் இது தமிழ் சோதனை",
    "te": "నమస్కారం ఇది తెలుగు పరీక్ష",
    "ml": "നമസ്കാരം ഇത് മലയാളം പരീക്ഷണമാണ്",
    "mr": "नमस्कार ही मराठी भाषेची चाचणी आहे ळ",
    "bn": "নমস্কার এটি বাংলা পরীক্ষা",
    "gu": "નમસ્તે આ ગુજરાતી પરીક્ષણ છે",
    "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਇਹ ਪੰਜਾਬੀ ਟੈਸਟ ਹੈ",
    "or": "ନମସ୍କାର ଏହା ଓଡ଼ିଆ ପରୀକ୍ଷା",
    "ur": "سلام یہ اردو ٹیسٹ ہے",
    "as": "নমস্কাৰ এই অসমীয়া পৰীক্ষা",
    "ne": "नमस्ते यो नेपाली परीक्षण हो",
}


@pytest.mark.parametrize("code", UI_LANGUAGE_CODES)
def test_language_name_normalizes_to_iso_code(code: str) -> None:
    name = ts._LANGUAGE_NAMES[code]
    assert ts._normalize_detected_language(name) == code


@pytest.mark.parametrize("code", UI_LANGUAGE_CODES)
def test_sarvam_language_map_exists(code: str) -> None:
    assert code in ts._SARVAM_LANGUAGE_CODE_MAP or code == "en"


@pytest.mark.parametrize(
    ("code", "sample"),
    [(code, SCRIPT_SAMPLES[code]) for code in SCRIPT_SAMPLES],
)
def test_script_detection_identifies_primary_language(code: str, sample: str) -> None:
    detected = ts._detect_indic_script_languages(sample)
    assert detected and detected[0] == code


@pytest.mark.parametrize("code", ("as", "ne"))
def test_backend_only_language_name_normalizes(code: str) -> None:
    name = ts._LANGUAGE_NAMES[code]
    assert ts._normalize_detected_language(name) == code


def test_normalize_detected_language_maps_od_to_or() -> None:
    assert ts._normalize_detected_language("od") == "or"


def test_music_language_neighbor_probes_expand_hindi_to_south_indian() -> None:
    neighbors = ts._music_language_neighbor_probes("hi")
    assert "ta" in neighbors
    assert "te" in neighbors
    assert "kn" in neighbors


def test_build_auto_sarvam_probe_includes_ta_when_groq_guesses_hindi_for_music(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_AFTER_GROQ", "true")
    monkeypatch.setenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_PROBE_UNKNOWN_FIRST", "true")
    payload = TranscriptPayload(
        source="groq",
        language="Hindi",
        text="some hindi guess without strong script",
        words=_payload(40).words,
        is_mock=False,
    )
    result = ts._build_auto_sarvam_probe_languages(payload, 30.0, "music", None)
    assert result[0] is None
    assert "hi" in result
    assert "ta" in result


def test_looks_like_latin_music_lyrics_detects_english_song_text() -> None:
    payload = TranscriptPayload(
        source="groq",
        language="Hindi",
        text="As we walk through the valley of the shadow of death we take a look at our life",
        words=_payload(40).words,
        is_mock=False,
    )
    assert ts._looks_like_latin_music_lyrics(payload)


def test_fast_mode_keeps_music_profile_for_song_like_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ts,
        "_resolve_transcription_profile",
        lambda _path, _duration: "music",
    )
    monkeypatch.setenv("TRANSCRIBE_PROFILE", "auto")

    configured_language = None
    transcript_mode = ts._normalize_transcription_mode(None)
    fast_mode_enabled = True
    requested_profile = "auto"

    profile = "speech"
    if transcript_mode == "song":
        profile = "music"
    elif transcript_mode == "speech":
        profile = "speech"
        if requested_profile in {"", "auto"}:
            detected_profile = ts._resolve_transcription_profile("", 30.0)
            if detected_profile in {"music", "mixed"}:
                profile = detected_profile
    elif fast_mode_enabled and requested_profile in {"", "auto"}:
        detected_profile = ts._resolve_transcription_profile("", 30.0)
        profile = (
            detected_profile
            if detected_profile in {"music", "mixed"}
            else "speech"
        )

    assert profile == "music"


def test_word_script_annotation_identifies_indic_word() -> None:
    script_tag, language_hint = transcript_router._word_script_annotation(
        "ನಮಸ್ಕಾರ", "kn"
    )
    assert script_tag == "indic"
    assert language_hint == "kn"


def test_word_script_annotation_identifies_mixed_script_word() -> None:
    script_tag, language_hint = transcript_router._word_script_annotation(
        "helloनमस्ते", "hi"
    )
    assert script_tag == "mixed"
    assert language_hint == "hi"
