from __future__ import annotations

import logging as _logging_module
import time as _time_module

_perf_logger = _logging_module.getLogger(__name__)

import base64
import ctypes
import gc
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from pathlib import Path
from uuid import uuid4

from .config import get_settings
from .media_utils import detect_silence_ranges

settings = get_settings()
_GROQ_AUDIO_SESSION = threading.local()
_TRANSCRIPTION_RUNTIME = threading.local()

DEFAULT_MUSIC_RETRY_PROMPT = (
    "Transcribe only clearly audible speech and sung lyrics verbatim in the original "
    "language. Preserve repeated chorus lines and ad-libs. If a word is unclear, omit "
    "it instead of guessing. Do not paraphrase. Do not translate. Do not add "
    "commentary, summaries, or intro/outro phrases that are not present in the audio."
)
_PROMPT_LEAKAGE_PHRASES: tuple[tuple[str, ...], ...] = (
    ("transcribe", "speech", "and", "sung", "lyrics"),
    ("transcribe", "speech", "and", "analysis"),
    ("preserve", "repeated", "chorus", "lines"),
    ("do", "not", "paraphrase"),
    ("do", "not", "translate"),
    ("sung", "lyrics"),
    ("ad", "libs"),
)
_PROMPT_LEAKAGE_SINGLETONS: set[str] = {"transcribe", "paraphrase", "translate"}

# ---------------------------------------------------------------------------
# Filler words (used by vibe auto-cut and hallucination heuristic)
# ---------------------------------------------------------------------------
FILLER_WORDS: set[str] = {
    "um",
    "uh",
    "uhm",
    "umm",
    "hmm",
    "hm",
    "ah",
    "er",
    "eh",
    "like",
    "basically",
    "literally",
    "actually",
    "right",
    "you know",
    "i mean",
    "sort of",
    "kind of",
    "so yeah",
}

_HALLUCINATION_STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "i'm",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "we",
    "you",
    # Common hallucination words — frequently appear in silence/music gaps
    "thank",
    "thanks",
    "watching",
    "listen",
    "listening",
    "bye",
    "goodbye",
    "subscribe",
    "subscribed",
    "end",
}

_TAIL_HALLUCINATION_PHRASES: set[tuple[str, ...]] = {
    ("thank", "you"),
    ("thank", "you", "so", "much"),
    ("thank", "you", "for", "watching"),
    ("thank", "you", "for", "listening"),
    ("thank", "you", "very", "much"),
    ("thanks", "for", "watching"),
    ("thanks", "for", "listening"),
    ("the", "end"),
    ("the", "end", "thank", "you"),
    ("bye", "bye"),
    ("good", "bye"),
    ("see", "you", "next", "time"),
    ("see", "you", "later"),
    ("see", "you", "in", "the", "next", "video"),
    ("that's", "all", "for", "today"),
    ("that's", "it", "for", "today"),
    ("please", "subscribe"),
    ("don't", "forget", "to", "subscribe"),
    ("like", "and", "subscribe"),
    ("like", "comment", "and", "subscribe"),
    ("if", "you", "enjoyed", "this", "video"),
}

# Head hallucination phrases - common YouTube intro/outro phrases that appear at START
_HEAD_HALLUCINATION_PHRASES: set[tuple[str, ...]] = {
    ("thank", "you"),
    ("thank", "you", "for", "watching"),
    ("thank", "you", "for", "listening"),
    ("thanks", "for", "watching"),
    ("thank", "you", "so", "much"),
    ("hey", "guys"),
    ("hey", "everyone"),
    ("hello", "everyone"),
    ("hi", "everyone"),
    ("hi", "guys"),
    ("what's", "up"),
    ("what's", "up", "guys"),
    ("welcome", "back"),
    ("welcome", "back", "everyone"),
    ("welcome", "to", "the", "channel"),
    ("subscribe",),
    ("please", "subscribe"),
    ("don't", "forget", "to", "subscribe"),
    ("like", "and", "subscribe"),
    ("like", "comment", "and", "subscribe"),
    ("good", "morning", "everyone"),
    ("good", "evening", "everyone"),
}

# Phrases that are almost certainly hallucinations *anywhere* in the transcript
# when they appear surrounded by sufficient silence gaps.
_ANYWHERE_HALLUCINATION_PHRASES: set[tuple[str, ...]] = {
    ("thank", "you", "for", "watching"),
    ("thank", "you", "for", "listening"),
    ("thank", "you", "very", "much", "for", "watching"),
    ("thanks", "for", "watching"),
    ("thanks", "for", "listening"),
    ("don't", "forget", "to", "subscribe"),
    ("please", "subscribe"),
    ("like", "and", "subscribe"),
    ("like", "comment", "and", "subscribe"),
    ("hit", "the", "subscribe", "button"),
    ("see", "you", "in", "the", "next", "video"),
    ("see", "you", "next", "time"),
    ("see", "you", "guys", "next", "time"),
    ("that's", "all", "for", "today"),
    ("that's", "it", "for", "today"),
    ("if", "you", "enjoyed", "this", "video"),
    ("thank", "you", "so", "much"),
}

# Single tokens that are almost always hallucinated when isolated inside silence gaps.
_HALLUCINATION_SINGLETONS_IN_GAPS: set[str] = {
    "subscribe",
    "subscribed",
}


@dataclass(frozen=True)
class TranscriptWordPayload:
    id: str
    text: str
    start_sec: float
    end_sec: float
    confidence: float | None = None
    quality_score: float | None = None
    quality_label: str | None = None
    source_pass: str | None = None
    speaker_id: str | None = None
    speaker_label: str | None = None


@dataclass(frozen=True)
class TranscriptPayload:
    source: str
    language: str | None
    text: str
    words: list[TranscriptWordPayload]
    is_mock: bool


def _runtime_profile() -> str | None:
    value = getattr(_TRANSCRIPTION_RUNTIME, "profile", None)
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _runtime_mode() -> str | None:
    value = getattr(_TRANSCRIPTION_RUNTIME, "mode", None)
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _normalize_transcription_mode(mode: str | None) -> str:
    normalized = (mode or "auto").strip().lower()
    if normalized in {"speech", "song"}:
        return normalized
    return "auto"


def _normalize_source_pass(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"main", "original"}:
        return "primary"
    if normalized in {"gapfill", "gap_fill", "retry_fill"}:
        return "retry"
    if normalized in {"rescue_gap", "gap_rescue"}:
        return "rescue"
    return normalized


def infer_source_pass(source: str | None) -> str:
    normalized = (source or "").strip().lower()
    if "rescue" in normalized:
        return "rescue"
    if "retry" in normalized or "gapfill" in normalized:
        return "retry"
    if "manual" in normalized:
        return "manual"
    return "primary"


def _copy_word_payload(
    item: TranscriptWordPayload,
    *,
    id: str | None = None,
    text: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    confidence: float | None | object = ...,
    quality_score: float | None | object = ...,
    quality_label: str | None | object = ...,
    source_pass: str | None | object = ...,
) -> TranscriptWordPayload:
    return TranscriptWordPayload(
        id=item.id if id is None else id,
        text=item.text if text is None else text,
        start_sec=float(item.start_sec) if start_sec is None else float(start_sec),
        end_sec=float(item.end_sec) if end_sec is None else float(end_sec),
        confidence=item.confidence if confidence is ... else confidence,
        quality_score=item.quality_score if quality_score is ... else quality_score,
        quality_label=item.quality_label if quality_label is ... else quality_label,
        source_pass=item.source_pass
        if source_pass is ...
        else _normalize_source_pass(source_pass),
    )


def _clean_word(value: str) -> str:
    return " ".join(value.strip().split())


def _normalize_token(value: str) -> str:
    return re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", value.lower()).strip()


def _ascii_latin_ratio(value: str) -> float:
    alpha = [char for char in value if char.isalpha()]
    if not alpha:
        return 1.0
    latin = [char for char in alpha if ("A" <= char <= "Z") or ("a" <= char <= "z")]
    return len(latin) / len(alpha)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)
    return max(minimum, value)


def _is_placeholder_config_value(value: str | None) -> bool:
    if value is None:
        return True
    raw = value.strip()
    if not raw:
        return True
    lowered = raw.lower()
    if lowered in {
        "your_endpoint",
        "your_key",
        "changeme",
        "change_me",
        "replace_me",
        "none",
        "null",
    }:
        return True
    if lowered.startswith("your_"):
        return True
    if lowered.startswith("<") and lowered.endswith(">") and len(lowered) > 2:
        return True
    return False


def _normalize_language_code(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw or raw in {"auto", "detect", "default"}:
        return None
    normalized = re.sub(r"[^a-z-]+", "", raw)
    if len(normalized) < 2 or len(normalized) > 8:
        return None
    return normalized


def _resolve_transcribe_language() -> str | None:
    return _normalize_language_code(os.getenv("TRANSCRIBE_LANGUAGE", ""))


def _parse_language_fallbacks(raw: str) -> list[str]:
    if not raw:
        return []
    items: list[str] = []
    seen: set[str] = set()
    for chunk in raw.split(","):
        code = _normalize_language_code(chunk)
        if not code:
            continue
        if code in seen:
            continue
        seen.add(code)
        items.append(code)
    return items


def _build_language_retry_candidates(
    configured_language: str | None,
    fallback_languages: list[str],
) -> list[str | None]:
    candidates: list[str | None] = []
    seen: set[str] = set()
    if configured_language:
        # When a fixed language hint under-recognizes song/mixed audio,
        # first give Groq one chance to auto-detect.
        candidates.append(None)
        seen.add(configured_language)
        # By default, avoid cross-language drift when the user explicitly picks
        # a language (e.g., Kannada forcing Tamil fallback candidates).
        if not _env_bool(
            "TRANSCRIBE_GROQ_EXPLICIT_LANGUAGE_ALLOW_CROSS_FALLBACK", False
        ):
            return candidates
    for language in fallback_languages:
        if language in seen:
            continue
        seen.add(language)
        candidates.append(language)
    return candidates


_LANGUAGE_SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "as": ((0x0980, 0x09FF),),
    "bn": ((0x0980, 0x09FF),),
    "gu": ((0x0A80, 0x0AFF),),
    "hi": ((0x0900, 0x097F),),
    "kn": ((0x0C80, 0x0CFF),),
    "ml": ((0x0D00, 0x0D7F),),
    "mr": ((0x0900, 0x097F),),
    "ne": ((0x0900, 0x097F),),
    "or": ((0x0B00, 0x0B7F),),
    "od": ((0x0B00, 0x0B7F),),
    "pa": ((0x0A00, 0x0A7F),),
    "te": ((0x0C00, 0x0C7F),),
    "ta": ((0x0B80, 0x0BFF),),
    "ur": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)),
}


_LANGUAGE_NAMES: dict[str, str] = {
    "as": "Assamese",
    "bn": "Bengali",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "od": "Odia",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}

_LANGUAGE_NAME_TO_CODE: dict[str, str] = {
    name.lower(): code for code, name in _LANGUAGE_NAMES.items()
}

# UI uses "or" for Odia; Sarvam/backend may return "od".
_LANGUAGE_CODE_ALIASES: dict[str, str] = {
    "od": "or",
}

_SCRIPT_LANGUAGE_PREFERENCE: tuple[str, ...] = (
    "kn",
    "ta",
    "te",
    "ml",
    "hi",
    "mr",
    "ne",
    "bn",
    "as",
    "gu",
    "pa",
    "or",
    "od",
    "ur",
)


def _language_script_match_metrics(
    value: str, language_code: str | None
) -> tuple[int, float]:
    code = _normalize_language_code(language_code)
    if not code:
        return 0, 1.0
    alpha_chars = [char for char in value if char.isalpha()]
    if not alpha_chars:
        return 0, 1.0

    if code in {"en"}:
        matches = sum(
            1 for char in alpha_chars if ("A" <= char <= "Z") or ("a" <= char <= "z")
        )
        return len(alpha_chars), (matches / len(alpha_chars))

    ranges = _LANGUAGE_SCRIPT_RANGES.get(code)
    if not ranges:
        return 0, 1.0

    matches = 0
    for char in alpha_chars:
        codepoint = ord(char)
        for start, end in ranges:
            if start <= codepoint <= end:
                matches += 1
                break
    return len(alpha_chars), (matches / len(alpha_chars))


def _payload_language_match_metrics(
    payload: TranscriptPayload, language_code: str | None
) -> tuple[int, float]:
    sample = " ".join(word.text for word in payload.words[:800])
    return _language_script_match_metrics(sample, language_code)


def _detect_indic_script_languages(value: str) -> list[str]:
    sample = str(value or "").strip()
    if not sample:
        return []
    min_alpha = _env_int("TRANSCRIBE_AUTO_ROUTE_SARVAM_MIXED_SCRIPT_MIN_ALPHA", 6, 0)
    min_ratio = _env_float(
        "TRANSCRIBE_AUTO_ROUTE_SARVAM_MIXED_SCRIPT_MIN_RATIO", 0.08, 0.0
    )
    preference_rank = {
        code: index for index, code in enumerate(_SCRIPT_LANGUAGE_PREFERENCE)
    }
    best_by_script: dict[tuple[tuple[int, int], ...], tuple[str, int, float]] = {}
    for code in _INDIC_LANGUAGE_CODES:
        ranges = _LANGUAGE_SCRIPT_RANGES.get(code)
        if not ranges:
            continue
        alpha_count, script_ratio = _language_script_match_metrics(sample, code)
        if alpha_count < min_alpha or script_ratio < min_ratio:
            continue
        current = best_by_script.get(ranges)
        candidate = (code, alpha_count, script_ratio)
        if current is None:
            best_by_script[ranges] = candidate
            continue
        current_rank = preference_rank.get(current[0], len(preference_rank))
        candidate_rank = preference_rank.get(code, len(preference_rank))
        if (
            script_ratio > current[2]
            or (script_ratio == current[2] and alpha_count > current[1])
            or (
                script_ratio == current[2]
                and alpha_count == current[1]
                and candidate_rank < current_rank
            )
        ):
            best_by_script[ranges] = candidate
    ranked = sorted(
        best_by_script.values(),
        key=lambda item: (
            -item[2],
            -item[1],
            preference_rank.get(item[0], len(preference_rank)),
        ),
    )
    result = [code for code, _alpha_count, _script_ratio in ranked]
    marker_boost = _script_marker_language_boost(sample)
    if marker_boost is None:
        return result
    if marker_boost in result:
        return [marker_boost] + [code for code in result if code != marker_boost]
    return [marker_boost, *result]


_SCRIPT_LANGUAGE_MARKERS: dict[str, tuple[str, ...]] = {
    "mr": ("मराठी",),
    "ne": ("नेपाल", "नेपाली"),
    "as": ("অসম", "অসমীয়া"),
}


def _script_marker_language_boost(text: str) -> str | None:
    for code, markers in _SCRIPT_LANGUAGE_MARKERS.items():
        if any(marker in text for marker in markers):
            return code
    return None


def _build_language_guard_prompt(language_code: str) -> str:
    normalized = _normalize_language_code(language_code) or language_code
    language_name = _LANGUAGE_NAMES.get(normalized, normalized)
    return (
        f"Transcribe strictly in {language_name} ({normalized}). "
        "Do not translate. Do not switch to another language. "
        "Output only words spoken/sung in this language."
    )


def _needs_language_guard_retry(
    payload: TranscriptPayload, language_code: str | None
) -> bool:
    normalized = _normalize_language_code(language_code)
    if not normalized:
        return False
    min_alpha = _env_int("TRANSCRIBE_LANGUAGE_GUARD_MIN_ALPHA", 16, 0)
    min_ratio = _env_float("TRANSCRIBE_LANGUAGE_GUARD_MIN_RATIO", 0.22, 0.0)
    alpha_count, script_ratio = _payload_language_match_metrics(payload, normalized)
    if alpha_count >= min_alpha and script_ratio < min_ratio:
        return True
    detected_language = _normalize_language_code(payload.language)
    if (
        detected_language
        and detected_language != normalized
        and script_ratio < (min_ratio + 0.10)
    ):
        return True
    return False


_INDIC_LANGUAGE_CODES: set[str] = {
    "as",
    "bn",
    "gu",
    "hi",
    "kn",
    "ml",
    "mr",
    "ne",
    "od",
    "or",
    "pa",
    "ta",
    "te",
    "ur",
}

_SARVAM_LANGUAGE_CODE_MAP: dict[str, str] = {
    "as": "as-IN",
    "bn": "bn-IN",
    "en": "en-IN",
    "gu": "gu-IN",
    "hi": "hi-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "ne": "ne-IN",
    "od": "od-IN",
    "or": "od-IN",
    "pa": "pa-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "ur": "ur-IN",
}


def _is_indic_language(language_code: str | None) -> bool:
    normalized = _normalize_language_code(language_code)
    return normalized in _INDIC_LANGUAGE_CODES


def _should_route_to_sarvam(backend: str, language_code: str | None) -> bool:
    if backend == "sarvam":
        return True
    if backend != "auto":
        return False
    if not _env_bool("TRANSCRIBE_ROUTER_ENABLED", True):
        return False
    if not _env_bool("TRANSCRIBE_ROUTER_INDIAN_TO_SARVAM", True):
        return False
    return _is_indic_language(language_code)


_SOUTH_INDIAN_MUSIC_NEIGHBORS: tuple[str, ...] = ("kn", "ta", "te", "ml")


def _south_indic_music_neighbor_languages(detected_language: str | None) -> list[str]:
    normalized = _normalize_language_code(detected_language)
    if normalized not in set(_SOUTH_INDIAN_MUSIC_NEIGHBORS):
        return []
    return [
        code
        for code in _SOUTH_INDIAN_MUSIC_NEIGHBORS
        if code != normalized and _is_indic_language(code)
    ]


def _music_language_neighbor_probes(detected_language: str | None) -> list[str]:
    normalized = _normalize_detected_language(detected_language)
    if not normalized:
        return []
    if normalized in set(_SOUTH_INDIAN_MUSIC_NEIGHBORS):
        return _south_indic_music_neighbor_languages(normalized)
    if normalized == "hi":
        return [
            code
            for code in ("ta", "te", "kn", "ml", "mr", "bn")
            if _is_indic_language(code)
        ]
    return []


def _looks_like_latin_music_lyrics(payload: TranscriptPayload | None) -> bool:
    if payload is None or not payload.text:
        return False
    min_ratio = _env_float("TRANSCRIBE_MUSIC_LATIN_LYRICS_MIN_RATIO", 0.80, 0.5)
    min_alpha = _env_int("TRANSCRIBE_MUSIC_LATIN_LYRICS_MIN_ALPHA", 24, 8)
    alpha = sum(1 for char in payload.text if char.isalpha())
    if alpha < min_alpha:
        return False
    return _ascii_latin_ratio(payload.text) >= min_ratio


def _sarvam_script_disagrees_with_language(payload: TranscriptPayload) -> bool:
    detected = _normalize_detected_language(payload.language)
    if not _is_indic_language(detected):
        return False
    script_languages = _detect_indic_script_languages(payload.text)
    if not script_languages:
        return False
    return script_languages[0] != detected


def _should_skip_sarvam_first_return(
    payload: TranscriptPayload,
    *,
    configured_language: str | None,
    music_auto_routing: bool,
) -> bool:
    if configured_language is not None or not music_auto_routing:
        return False
    if _looks_like_latin_music_lyrics(payload):
        return True
    return _sarvam_script_disagrees_with_language(payload)


def _auto_sarvam_music_routing_allowed(
    *,
    profile: str,
    configured_language: str | None,
) -> bool:
    return profile in {"music", "mixed"} and configured_language is None


def _resolve_auto_sarvam_probe_attempts(
    *,
    fast_mode_enabled: bool,
    speed_optimized: bool,
    profile: str,
    configured_language: str | None,
) -> int:
    max_attempts = _env_int("TRANSCRIBE_AUTO_ROUTE_SARVAM_MAX_ATTEMPTS", 4, 0)
    if max_attempts <= 0:
        return 0
    if not fast_mode_enabled and not speed_optimized:
        return max_attempts
    if not _auto_sarvam_music_routing_allowed(
        profile=profile,
        configured_language=configured_language,
    ):
        return 0
    fast_min = _env_int("TRANSCRIBE_AUTO_ROUTE_SARVAM_FAST_MIN_ATTEMPTS", 3, 0)
    return min(max_attempts, max(fast_min, 1))


def _build_auto_sarvam_probe_languages(
    payload: TranscriptPayload,
    duration_sec: float,
    profile: str,
    configured_language: str | None,
) -> list[str | None]:
    if configured_language:
        return []
    if not _env_bool("TRANSCRIBE_AUTO_ROUTE_SARVAM_AFTER_GROQ", True):
        return []

    candidates: list[str | None] = []
    detected_language = _normalize_detected_language(payload.language)
    script_languages = (
        _detect_indic_script_languages(payload.text)
        if _env_bool("TRANSCRIBE_AUTO_ROUTE_SARVAM_ON_MIXED_SCRIPT", True)
        else []
    )
    if _is_indic_language(detected_language):
        candidates.append(detected_language or "")
    candidates.extend(script_languages)

    max_latin_ratio = _env_float(
        "TRANSCRIBE_AUTO_ROUTE_SARVAM_MAX_LATIN_RATIO", 0.92, 0.0
    )
    latin_ratio = _ascii_latin_ratio(payload.text)
    should_probe_fallbacks = not candidates and (
        _is_low_coverage(payload, duration_sec) and latin_ratio < max_latin_ratio
    )
    if should_probe_fallbacks:
        fallback_languages = _parse_language_fallbacks(
            os.getenv("TRANSCRIBE_AUTO_ROUTE_SARVAM_LANGUAGES", "hi,kn,ta,te,ml")
        )
        candidates.extend(
            language for language in fallback_languages if _is_indic_language(language)
        )

    if (
        profile in {"music", "mixed"}
        and _is_indic_language(detected_language)
        and _env_bool("TRANSCRIBE_AUTO_ROUTE_SARVAM_MUSIC_NEIGHBOR_PROBE", True)
    ):
        for neighbor in _music_language_neighbor_probes(detected_language):
            if neighbor not in candidates:
                candidates.append(neighbor)

    if (
        candidates
        and profile in {"music", "mixed"}
        and _env_bool("TRANSCRIBE_AUTO_ROUTE_SARVAM_PROBE_UNKNOWN_FIRST", True)
    ):
        # Songs often confuse Whisper between neighboring Indian languages
        # (Kannada/Tamil/Telugu especially). Let Sarvam run its own detector once
        # before forcing the language that Whisper guessed.
        candidates.insert(0, None)

    unique_candidates: list[str | None] = []
    seen: set[str] = set()
    for language in candidates:
        normalized = _normalize_language_code(language)
        if normalized is None:
            if "unknown" in seen:
                continue
            seen.add("unknown")
            unique_candidates.append(None)
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(normalized)
    return unique_candidates


def _sarvam_language_code(language_hint: str | None) -> str:
    normalized = _normalize_language_code(language_hint)
    if not normalized:
        return "unknown"
    return _SARVAM_LANGUAGE_CODE_MAP.get(normalized, "unknown")


def _normalize_detected_language(value: object | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    if "-" in raw:
        raw = raw.split("-", 1)[0]
    if raw in _LANGUAGE_NAME_TO_CODE:
        return _LANGUAGE_NAME_TO_CODE[raw]
    code = _normalize_language_code(raw)
    if code in _LANGUAGE_CODE_ALIASES:
        return _LANGUAGE_CODE_ALIASES[code]
    if code in _INDIC_LANGUAGE_CODES or code in _LANGUAGE_NAMES:
        return code
    return code


def _parse_timestamp_value(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed):
        return None
    return parsed


def _build_word_payloads_from_series(
    words_series: list[object],
    starts_series: list[object],
    ends_series: list[object],
    duration_sec: float,
) -> list[TranscriptWordPayload]:
    words: list[TranscriptWordPayload] = []
    count = min(len(words_series), len(starts_series), len(ends_series))
    for idx in range(count):
        token_text = _clean_word(str(words_series[idx] or ""))
        if not token_text:
            continue
        start = _parse_timestamp_value(starts_series[idx])
        end = _parse_timestamp_value(ends_series[idx])
        if start is None or end is None:
            continue
        if end <= start:
            end = start + 0.05
        parts = token_text.split()
        if len(parts) <= 1:
            words.append(
                TranscriptWordPayload(
                    id=str(uuid4()),
                    text=token_text,
                    start_sec=start,
                    end_sec=end,
                    confidence=None,
                )
            )
            continue
        span = max(end - start, 0.05)
        step = span / len(parts)
        for part_idx, part in enumerate(parts):
            word_start = start + (part_idx * step)
            word_end = start + ((part_idx + 1) * step)
            words.append(
                TranscriptWordPayload(
                    id=str(uuid4()),
                    text=part,
                    start_sec=word_start,
                    end_sec=word_end,
                    confidence=None,
                )
            )
    return _normalize_words(words, duration_sec, apply_offset=False)


def _fallback_words_from_text(
    text: str, duration_sec: float
) -> list[TranscriptWordPayload]:
    parts = [part for part in text.split() if part.strip()]
    if not parts:
        return []
    step = max(duration_sec / max(len(parts), 1), 0.05)
    words: list[TranscriptWordPayload] = []
    for idx, token in enumerate(parts):
        start = idx * step
        end = min(duration_sec, start + max(0.05, step * 0.9))
        words.append(
            TranscriptWordPayload(
                id=str(uuid4()),
                text=token,
                start_sec=start,
                end_sec=end,
                confidence=None,
            )
        )
    return _normalize_words(words, duration_sec, apply_offset=False)


def _should_fallback_to_text_words(
    words: list[TranscriptWordPayload],
    transcript_text: str,
) -> bool:
    tokens = [part for part in transcript_text.split() if part.strip()]
    if not tokens:
        return False
    if not words:
        return True
    if len(tokens) >= 4 and len(words) <= 1:
        return True
    if len(tokens) >= 8 and (len(words) * 4) < len(tokens):
        return True
    return False


def _extract_sarvam_word_timestamps(
    payload: object, duration_sec: float
) -> list[TranscriptWordPayload]:
    if not isinstance(payload, dict):
        return []
    timestamps = payload.get("timestamps")
    if isinstance(timestamps, list):
        words_series: list[object] = []
        starts_series: list[object] = []
        ends_series: list[object] = []
        for item in timestamps:
            if not isinstance(item, dict):
                continue
            words_series.append(item.get("word") or item.get("text") or "")
            starts_series.append(
                item.get("start_time_seconds")
                if item.get("start_time_seconds") is not None
                else item.get("start_time")
                if item.get("start_time") is not None
                else item.get("start")
            )
            ends_series.append(
                item.get("end_time_seconds")
                if item.get("end_time_seconds") is not None
                else item.get("end_time")
                if item.get("end_time") is not None
                else item.get("end")
            )
        if words_series and starts_series and ends_series:
            return _build_word_payloads_from_series(
                words_series, starts_series, ends_series, duration_sec
            )
        return []
    if isinstance(timestamps, dict) and isinstance(timestamps.get("timestamps"), dict):
        timestamps = timestamps.get("timestamps")
    if not isinstance(timestamps, dict):
        return []

    words_series = timestamps.get("words")
    starts_series = timestamps.get("start_time_seconds")
    ends_series = timestamps.get("end_time_seconds")
    if (
        not isinstance(words_series, list)
        or not isinstance(starts_series, list)
        or not isinstance(ends_series, list)
    ):
        return []
    return _build_word_payloads_from_series(
        words_series, starts_series, ends_series, duration_sec
    )


def _session_source_key(path: str) -> str:
    try:
        return str(Path(path).resolve(strict=False))
    except OSError:
        return path


def _start_groq_audio_session(path: str, *, use_vocal_isolation: bool = True) -> None:
    _GROQ_AUDIO_SESSION.source_key = _session_source_key(path)
    _GROQ_AUDIO_SESSION.prepared_path = None
    _GROQ_AUDIO_SESSION.cleanup_path = None
    _GROQ_AUDIO_SESSION.use_vocal_isolation = bool(use_vocal_isolation)


def _finish_groq_audio_session() -> None:
    cleanup_path = getattr(_GROQ_AUDIO_SESSION, "cleanup_path", None)
    if isinstance(cleanup_path, Path):
        cleanup_path.unlink(missing_ok=True)
    _GROQ_AUDIO_SESSION.source_key = None
    _GROQ_AUDIO_SESSION.prepared_path = None
    _GROQ_AUDIO_SESSION.cleanup_path = None
    _GROQ_AUDIO_SESSION.use_vocal_isolation = None


def _resolve_groq_input_source(path: str) -> tuple[str, Path | None]:
    source_key = _session_source_key(path)
    session_source_key = getattr(_GROQ_AUDIO_SESSION, "source_key", None)
    use_vocal_isolation = bool(
        getattr(_GROQ_AUDIO_SESSION, "use_vocal_isolation", True)
    )
    if session_source_key != source_key:
        return _extract_audio_for_cloud(path, use_vocal_isolation=use_vocal_isolation)

    prepared_path = getattr(_GROQ_AUDIO_SESSION, "prepared_path", None)
    if isinstance(prepared_path, str) and Path(prepared_path).exists():
        return prepared_path, None

    prepared_path, cleanup_path = _extract_audio_for_cloud(
        path, use_vocal_isolation=use_vocal_isolation
    )
    _GROQ_AUDIO_SESSION.prepared_path = prepared_path
    _GROQ_AUDIO_SESSION.cleanup_path = cleanup_path
    return prepared_path, None


def _clamp_time(value: float, duration_sec: float) -> float:
    return max(0.0, min(value, duration_sec))


def _normalize_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed):
        return None
    return max(0.0, min(parsed, 1.0))


def _normalize_words(
    words: list[TranscriptWordPayload],
    duration_sec: float,
    *,
    apply_offset: bool = True,
) -> list[TranscriptWordPayload]:
    min_confidence = _env_float("TRANSCRIBE_WORD_MIN_CONFIDENCE", 0.15, 0.0)
    min_word_duration_sec = _env_float("TRANSCRIBE_MIN_WORD_DURATION_SEC", 0.05, 0.01)
    max_word_duration_sec = _env_float(
        "TRANSCRIBE_MAX_WORD_DURATION_SEC", 1.2, min_word_duration_sec
    )
    next_word_guard_sec = _env_float("TRANSCRIBE_WORD_NEXT_GUARD_SEC", 0.01, 0.0)
    # Global timestamp offset: only applied during initial generation (apply_offset=True)
    # NOT during storage/reading to avoid triple-application
    timestamp_offset_sec = (
        _env_float("TRANSCRIBE_TIMESTAMP_OFFSET_SEC", 0.0, -5.0)
        if apply_offset
        else 0.0
    )

    prelim: list[TranscriptWordPayload] = []
    for item in sorted(words, key=lambda entry: entry.start_sec):
        # Apply global timestamp offset (only on initial generation)
        raw_start = float(item.start_sec) + timestamp_offset_sec
        raw_end = float(item.end_sec) + timestamp_offset_sec
        start_sec = round(_clamp_time(raw_start, duration_sec), 3)
        end_sec = round(_clamp_time(raw_end, duration_sec), 3)
        if end_sec <= start_sec:
            end_sec = round(min(duration_sec, start_sec + min_word_duration_sec), 3)
        text = _clean_word(item.text)
        if not text:
            continue
        confidence = _normalize_confidence(item.confidence)
        # Filter out words with very low confidence (likely hallucinations)
        if confidence is not None and confidence < min_confidence:
            continue
        prelim.append(
            TranscriptWordPayload(
                id=item.id,
                text=text,
                start_sec=start_sec,
                end_sec=end_sec,
                confidence=confidence,
                quality_score=item.quality_score,
                quality_label=item.quality_label,
                source_pass=_normalize_source_pass(item.source_pass),
                speaker_id=item.speaker_id,
                speaker_label=item.speaker_label,
            )
        )

    normalized: list[TranscriptWordPayload] = []
    for index, item in enumerate(prelim):
        start_sec = float(item.start_sec)
        end_sec = min(float(item.end_sec), start_sec + max_word_duration_sec)

        if index + 1 < len(prelim):
            next_start = float(prelim[index + 1].start_sec)
            if next_start > start_sec:
                end_sec = min(end_sec, next_start - next_word_guard_sec)

        if end_sec <= start_sec:
            fallback_end = min(duration_sec, start_sec + min_word_duration_sec)
            if index + 1 < len(prelim):
                next_start = float(prelim[index + 1].start_sec)
                if next_start > start_sec:
                    fallback_end = min(
                        fallback_end,
                        max(start_sec + 0.001, next_start - next_word_guard_sec),
                    )
            end_sec = fallback_end

        if end_sec <= start_sec:
            continue

        normalized.append(
            TranscriptWordPayload(
                id=item.id,
                text=item.text,
                start_sec=round(start_sec, 3),
                end_sec=round(min(end_sec, duration_sec), 3),
                confidence=item.confidence,
                quality_score=item.quality_score,
                quality_label=item.quality_label,
                source_pass=_normalize_source_pass(item.source_pass),
                speaker_id=item.speaker_id,
                speaker_label=item.speaker_label,
            )
        )
    return normalized


def sanitize_transcript_words(
    words: list[TranscriptWordPayload],
    duration_sec: float,
    *,
    apply_filters: bool = False,
    apply_offset: bool = True,
) -> list[TranscriptWordPayload]:
    normalized = _normalize_words(
        words, max(float(duration_sec), 0.0), apply_offset=apply_offset
    )
    if apply_filters:
        normalized = _trim_known_tail_hallucination(
            normalized, max(float(duration_sec), 0.0)
        )
    return normalized


def _detect_hallucinations(
    words: list[TranscriptWordPayload],
) -> list[TranscriptWordPayload]:
    """Remove repeated phrase loops — a common Whisper failure mode.

    Detects when the same short phrase is repeated 3+ times consecutively
    and collapses it to a single occurrence.
    """
    if len(words) < 6:
        return words

    # Collapse quick duplicate bursts first (e.g., "I'm I'm worried worried").
    # These are common in music-heavy or noisy sources and should not survive
    # to editing UX as separate words.
    dup_gap_sec = _env_float("TRANSCRIBE_HALLUCINATION_DUP_GAP_SEC", 0.35, 0.0)
    deduped: list[TranscriptWordPayload] = []
    for item in words:
        token = _normalize_token(item.text)
        if not token:
            deduped.append(item)
            continue
        if deduped:
            prev = deduped[-1]
            prev_token = _normalize_token(prev.text)
            if prev_token and prev_token == token:
                gap_sec = float(item.start_sec) - float(prev.end_sec)
                if gap_sec <= dup_gap_sec:
                    deduped[-1] = _copy_word_payload(
                        prev,
                        start_sec=float(prev.start_sec),
                        end_sec=max(float(prev.end_sec), float(item.end_sec)),
                        confidence=prev.confidence
                        if prev.confidence is not None
                        else item.confidence,
                        quality_score=prev.quality_score
                        if prev.quality_score is not None
                        else item.quality_score,
                        quality_label=prev.quality_label or item.quality_label,
                        source_pass=_normalize_source_pass(
                            prev.source_pass or item.source_pass
                        ),
                    )
                    continue
        deduped.append(item)

    words = deduped
    if len(words) < 6:
        return words

    profile = _runtime_profile()
    mode = _runtime_mode()
    if mode == "song" and not _env_bool("TRANSCRIBE_COLLAPSE_REPEATS_IN_SONG", False):
        return words
    if profile in {"music", "mixed"}:
        if _env_bool("TRANSCRIBE_PRESERVE_MUSIC_REPEATS", True):
            return words
        if not _env_bool("TRANSCRIBE_COLLAPSE_REPEATS_IN_MUSIC", False):
            return words

    cleaned: list[TranscriptWordPayload] = []
    i = 0
    while i < len(words):
        # Try phrase lengths 1-4 words
        found_repeat = False
        for phrase_len in range(1, min(5, (len(words) - i) // 2 + 1)):
            phrase_texts = tuple(
                _normalize_token(w.text) for w in words[i : i + phrase_len]
            )
            if any(not token for token in phrase_texts):
                continue
            repeat_count = 1
            j = i + phrase_len
            while j + phrase_len <= len(words):
                next_texts = tuple(
                    _normalize_token(w.text) for w in words[j : j + phrase_len]
                )
                if next_texts == phrase_texts:
                    repeat_count += 1
                    j += phrase_len
                else:
                    break
            if repeat_count >= 3:
                # Keep only the first occurrence
                cleaned.extend(words[i : i + phrase_len])
                i = j
                found_repeat = True
                break
        if not found_repeat:
            cleaned.append(words[i])
            i += 1

    # Keep default behavior simple to avoid over-pruning valid words.
    # Strict mode enables stronger phrase/backtrack suppression for noisy music-heavy audio.
    if not _env_bool("TRANSCRIBE_HALLUCINATION_STRICT", False):
        return cleaned

    # Collapse repeated short phrases that recur many times in a tight window
    # (e.g. lyric loops) while preserving the first occurrence.
    min_occurrences = _env_int("TRANSCRIBE_HALLUCINATION_REPEAT_MIN_OCCURRENCES", 3, 2)
    repeat_window_sec = _env_float(
        "TRANSCRIBE_HALLUCINATION_REPEAT_WINDOW_SEC", 20.0, 2.0
    )
    if len(cleaned) < 8 or min_occurrences <= 1:
        return cleaned

    tokens = [_normalize_token(item.text) for item in cleaned]
    drop_indices: set[int] = set()

    def _mark_cluster(indices: list[int], phrase_len: int) -> None:
        if len(indices) < min_occurrences:
            return
        # Keep first occurrence, drop subsequent repeats.
        for idx in indices[1:]:
            for offset in range(phrase_len):
                drop_indices.add(idx + offset)

    for phrase_len in (4, 3):
        if len(tokens) < phrase_len:
            continue
        phrase_map: dict[tuple[str, ...], list[int]] = {}
        for idx in range(0, len(tokens) - phrase_len + 1):
            phrase = tuple(tokens[idx : idx + phrase_len])
            if any(not token for token in phrase):
                continue
            if all(token in _HALLUCINATION_STOPWORDS for token in phrase):
                continue
            phrase_map.setdefault(phrase, []).append(idx)

        for indices in phrase_map.values():
            if len(indices) < min_occurrences:
                continue
            cluster: list[int] = []
            for idx in indices:
                if not cluster:
                    cluster.append(idx)
                    continue
                first_start = float(cleaned[cluster[0]].start_sec)
                current_start = float(cleaned[idx].start_sec)
                if current_start - first_start <= repeat_window_sec:
                    cluster.append(idx)
                    continue
                _mark_cluster(cluster, phrase_len)
                cluster = [idx]
            _mark_cluster(cluster, phrase_len)

    if not drop_indices:
        return cleaned

    filtered = [item for idx, item in enumerate(cleaned) if idx not in drop_indices]
    if not filtered:
        filtered = cleaned

    # Collapse short "backtrack" glitches (A B A or A B A B) that appear in
    # noisy mixes after ASR decoding.
    backtrack_gap_sec = _env_float(
        "TRANSCRIBE_HALLUCINATION_BACKTRACK_GAP_SEC", 1.2, 0.2
    )
    if len(filtered) < 3:
        return filtered
    normalized = [_normalize_token(item.text) for item in filtered]
    remove_backtrack: set[int] = set()

    for idx in range(0, len(filtered) - 2):
        a = normalized[idx]
        c = normalized[idx + 2]
        if not a or not c or a != c:
            continue
        span = float(filtered[idx + 2].start_sec) - float(filtered[idx].start_sec)
        if span <= backtrack_gap_sec:
            remove_backtrack.add(idx + 2)

    for idx in range(0, len(filtered) - 3):
        a = normalized[idx]
        b = normalized[idx + 1]
        c = normalized[idx + 2]
        d = normalized[idx + 3]
        if not a or not b:
            continue
        if a == c and b == d:
            span = float(filtered[idx + 3].start_sec) - float(filtered[idx].start_sec)
            if span <= backtrack_gap_sec * 1.8:
                remove_backtrack.add(idx + 2)
                remove_backtrack.add(idx + 3)

    if not remove_backtrack:
        return filtered
    collapsed = [
        item for idx, item in enumerate(filtered) if idx not in remove_backtrack
    ]
    return collapsed if collapsed else filtered


def _detect_sparse_hallucinations(
    words: list[TranscriptWordPayload],
    duration_sec: float,
) -> list[TranscriptWordPayload]:
    """Remove isolated low-confidence words in music/silence gaps.

    Whisper often hallucinates a handful of random words during long stretches
    of instrumental audio or silence.  This filter scans for sparse windows
    (few words spread across a long time span) with low average confidence and
    removes them.

    For song/music profiles, the filter is more aggressive when confidence data
    is unavailable (e.g. Groq backend): any sparse cluster surrounded by large
    gaps is likely a hallucination in an instrumental section.
    """
    if len(words) < 3 or duration_sec <= 0:
        return words

    sparse_window_sec = _env_float(
        "TRANSCRIBE_SPARSE_HALLUCINATION_WINDOW_SEC", 8.0, 3.0
    )
    sparse_max_words = _env_int("TRANSCRIBE_SPARSE_HALLUCINATION_MAX_WORDS", 5, 1)
    sparse_max_confidence = _env_float(
        "TRANSCRIBE_SPARSE_HALLUCINATION_MAX_CONFIDENCE", 0.55, 0.0
    )
    sparse_min_gap_before = _env_float(
        "TRANSCRIBE_SPARSE_HALLUCINATION_MIN_GAP_SEC", 3.0, 1.0
    )

    if not _env_bool("TRANSCRIBE_SPARSE_HALLUCINATION_FILTER", True):
        return words

    # In song/music profiles, be more aggressive with gap detection and
    # treat missing-confidence clusters as hallucinations.
    profile = _runtime_profile()
    mode = _runtime_mode()
    is_music_context = profile in {"music", "mixed"} or mode == "song"
    music_min_gap = _env_float(
        "TRANSCRIBE_SPARSE_HALLUCINATION_MUSIC_MIN_GAP_SEC", 2.0, 0.5
    )
    effective_min_gap = music_min_gap if is_music_context else sparse_min_gap_before

    drop_indices: set[int] = set()

    # Build a sorted list with indices for efficient scanning
    indexed = list(enumerate(words))

    for i, word_i in indexed:
        # Find runs of words that are sparse (few words in a wide time window)
        # Look for a gap before or after this word that indicates a non-speech region
        gap_before = (
            float(word_i.start_sec)
            if i == 0
            else (float(word_i.start_sec) - float(words[i - 1].end_sec))
        )
        if gap_before < effective_min_gap and i > 0:
            continue

        # Collect consecutive words that fall within the sparse window
        cluster_indices: list[int] = [i]
        cluster_end = float(word_i.end_sec)
        for j in range(i + 1, len(words)):
            if float(words[j].start_sec) - float(word_i.start_sec) > sparse_window_sec:
                break
            cluster_indices.append(j)
            cluster_end = float(words[j].end_sec)

        # Check if there's also a gap after the cluster
        last_idx = cluster_indices[-1]
        gap_after = (
            (duration_sec - cluster_end)
            if last_idx == len(words) - 1
            else (float(words[last_idx + 1].start_sec) - cluster_end)
        )
        if gap_after < effective_min_gap and last_idx < len(words) - 1:
            continue

        # This cluster is sparse (surrounded by gaps) - check word count and confidence
        if len(cluster_indices) > sparse_max_words:
            continue

        # Check average confidence of the cluster
        confidences = [
            float(words[idx].confidence)
            for idx in cluster_indices
            if words[idx].confidence is not None
        ]
        if not confidences:
            # No confidence data available (common with Groq backend).
            # In music/song context, sparse clusters surrounded by large gaps
            # during instrumental sections are almost certainly hallucinations.
            if is_music_context:
                drop_indices.update(cluster_indices)
            else:
                # Fallback: only drop if all words are stopwords
                all_stopwords = all(
                    _normalize_token(words[idx].text) in _HALLUCINATION_STOPWORDS
                    for idx in cluster_indices
                )
                if all_stopwords:
                    drop_indices.update(cluster_indices)
            continue

        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence <= sparse_max_confidence:
            drop_indices.update(cluster_indices)

    if not drop_indices:
        return words

    _perf_logger.debug(
        "Sparse hallucination filter removed %d word(s) in %s context: %s",
        len(drop_indices),
        "music/song" if is_music_context else "speech",
        [words[idx].text for idx in sorted(drop_indices)],
    )
    filtered = [word for idx, word in enumerate(words) if idx not in drop_indices]
    return filtered if filtered else words


def _drop_words_in_nonvocal_regions(
    words: list[TranscriptWordPayload],
    duration_sec: float,
    *,
    audio_path: str | None,
) -> list[TranscriptWordPayload]:
    """Remove words that land inside long no-vocal spans.

    For music-heavy content, ASR can hallucinate isolated words during
    instrumental sections. Detect these spans on the prepared audio input and
    drop words whose midpoint clearly falls inside them.
    """
    if len(words) < 2 or duration_sec <= 0 or not audio_path:
        return words
    if _runtime_profile() not in {"music", "mixed"}:
        return words
    if not _env_bool("TRANSCRIBE_NONVOCAL_REGION_FILTER", True):
        return words

    min_nonvocal_sec = _env_float("TRANSCRIBE_NONVOCAL_REGION_MIN_SEC", 1.6, 0.4)
    noise_db = _env_float("TRANSCRIBE_NONVOCAL_REGION_NOISE_DB", -34.0, -80.0)
    guard_sec = _env_float("TRANSCRIBE_NONVOCAL_REGION_GUARD_SEC", 0.12, 0.0)
    max_regions = _env_int("TRANSCRIBE_NONVOCAL_REGION_MAX_SPANS", 48, 1)

    try:
        silence_ranges = detect_silence_ranges(
            audio_path,
            noise_db=noise_db,
            min_silence_sec=min_nonvocal_sec,
            max_duration_sec=duration_sec,
        )
    except Exception:  # noqa: BLE001
        return words

    if not silence_ranges:
        return words

    protected_ranges: list[tuple[float, float]] = []
    for start_sec, end_sec in silence_ranges[:max_regions]:
        start = max(0.0, min(float(start_sec), duration_sec))
        end = max(0.0, min(float(end_sec), duration_sec))
        if end - start < min_nonvocal_sec:
            continue
        inner_start = min(end, start + guard_sec)
        inner_end = max(inner_start, end - guard_sec)
        if inner_end - inner_start < max(0.2, min_nonvocal_sec * 0.35):
            continue
        protected_ranges.append((inner_start, inner_end))

    if not protected_ranges:
        return words

    filtered: list[TranscriptWordPayload] = []
    for word in words:
        midpoint = _word_midpoint_sec(word)
        if any(start <= midpoint <= end for start, end in protected_ranges):
            continue
        filtered.append(word)
    return filtered if filtered else words


def _trim_known_tail_hallucination(
    words: list[TranscriptWordPayload],
    duration_sec: float,
) -> list[TranscriptWordPayload]:
    if len(words) < 3 or duration_sec <= 0:
        return words
    if not _env_bool("TRANSCRIBE_TAIL_PHRASE_FILTER", True):
        return words

    min_gap_before_sec = _env_float("TRANSCRIBE_TAIL_PHRASE_MIN_GAP_SEC", 0.3, 0.0)
    max_phrase_span_sec = _env_float("TRANSCRIBE_TAIL_PHRASE_MAX_SPAN_SEC", 3.0, 0.1)
    min_remaining_words = _env_int("TRANSCRIBE_TAIL_PHRASE_MIN_REMAINING_WORDS", 3, 0)
    max_phrase_words = max(
        2, min(_env_int("TRANSCRIBE_TAIL_PHRASE_MAX_WORDS", 6, 2), 8)
    )

    normalized_tokens = [_normalize_token(item.text) for item in words]
    total = len(words)
    max_len = min(total, max_phrase_words)
    for phrase_len in range(max_len, 1, -1):
        start_idx = total - phrase_len
        phrase_tokens = tuple(normalized_tokens[start_idx:])
        if any(not token for token in phrase_tokens):
            continue
        if phrase_tokens not in _TAIL_HALLUCINATION_PHRASES:
            continue
        if start_idx < min_remaining_words:
            continue

        phrase_start = float(words[start_idx].start_sec)
        phrase_end = float(words[-1].end_sec)
        phrase_span = max(0.0, phrase_end - phrase_start)
        if phrase_span > max_phrase_span_sec:
            continue

        previous_end = float(words[start_idx - 1].end_sec) if start_idx > 0 else 0.0
        gap_before = max(0.0, phrase_start - previous_end)
        if gap_before < min_gap_before_sec:
            continue

        trimmed = words[:start_idx]
        return trimmed if trimmed else words

    return words


def _trim_sparse_music_tail(
    words: list[TranscriptWordPayload],
    duration_sec: float,
    *,
    force: bool = False,
) -> list[TranscriptWordPayload]:
    if len(words) < 3 or duration_sec <= 0:
        return words
    if not force and _runtime_profile() not in {"music", "mixed"}:
        return words
    if not _env_bool("TRANSCRIBE_SPARSE_MUSIC_TAIL_FILTER", True):
        return words

    max_tail_words = _env_int("TRANSCRIBE_SPARSE_MUSIC_TAIL_MAX_WORDS", 6, 1)
    max_tail_span_sec = _env_float(
        "TRANSCRIBE_SPARSE_MUSIC_TAIL_MAX_SPAN_SEC", 4.0, 0.2
    )
    default_tail_gap_sec = 4.0 if duration_sec <= 30.0 else 2.5
    min_tail_gap_sec = _env_float(
        "TRANSCRIBE_SPARSE_MUSIC_TAIL_MIN_GAP_SEC", default_tail_gap_sec, 0.5
    )
    if duration_sec <= 30.0:
        min_tail_gap_sec = max(min_tail_gap_sec, default_tail_gap_sec)
    min_prev_gap_sec = _env_float("TRANSCRIBE_SPARSE_MUSIC_TAIL_PREV_GAP_SEC", 1.2, 0.0)
    cluster_gap_sec = _env_float(
        "TRANSCRIBE_SPARSE_MUSIC_TAIL_CLUSTER_GAP_SEC", 0.85, 0.05
    )
    max_avg_confidence = _env_float(
        "TRANSCRIBE_SPARSE_MUSIC_TAIL_MAX_CONFIDENCE", 0.72, 0.0
    )

    tail_gap_sec = max(0.0, duration_sec - float(words[-1].end_sec))
    if tail_gap_sec < min_tail_gap_sec:
        return words

    start_idx = len(words) - 1
    while start_idx > 0:
        gap_sec = max(
            0.0, float(words[start_idx].start_sec) - float(words[start_idx - 1].end_sec)
        )
        if gap_sec > cluster_gap_sec:
            break
        start_idx -= 1

    tail_cluster = words[start_idx:]
    if len(tail_cluster) > max_tail_words:
        return words

    cluster_start = float(tail_cluster[0].start_sec)
    cluster_end = float(tail_cluster[-1].end_sec)
    if (cluster_end - cluster_start) > max_tail_span_sec:
        return words

    prev_gap_sec = (
        cluster_start
        if start_idx == 0
        else max(0.0, cluster_start - float(words[start_idx - 1].end_sec))
    )
    if prev_gap_sec < min_prev_gap_sec:
        return words

    confidences = [
        float(item.confidence) for item in tail_cluster if item.confidence is not None
    ]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence > max_avg_confidence:
            return words

    trimmed = words[:start_idx]
    return trimmed if trimmed else words


def trim_songlike_tail_hallucination(
    payload: TranscriptPayload,
    *,
    duration_sec: float,
) -> TranscriptPayload:
    if payload.is_mock or not payload.words:
        return payload

    trimmed_words = _trim_sparse_music_tail(
        payload.words,
        duration_sec,
        force=True,
    )
    if len(trimmed_words) == len(payload.words):
        return payload

    return TranscriptPayload(
        source=payload.source,
        language=payload.language,
        text=" ".join(word.text for word in trimmed_words).strip(),
        words=trimmed_words,
        is_mock=payload.is_mock,
    )


def trim_song_mode_to_manual_lyrics_span(
    payload: TranscriptPayload,
) -> TranscriptPayload:
    if payload.is_mock or not payload.words:
        return payload
    if not payload.source.endswith("_lyrics_ref"):
        return payload

    manual_indices = [
        index
        for index, word in enumerate(payload.words)
        if _normalize_source_pass(word.source_pass) == "manual"
    ]
    if not manual_indices:
        return payload

    start_index = manual_indices[0]
    end_index = manual_indices[-1] + 1
    trimmed_words = payload.words[start_index:end_index]
    if len(trimmed_words) == len(payload.words):
        return payload

    return TranscriptPayload(
        source=payload.source,
        language=payload.language,
        text=" ".join(word.text for word in trimmed_words).strip(),
        words=trimmed_words,
        is_mock=payload.is_mock,
    )


def _trim_known_head_hallucination(
    words: list[TranscriptWordPayload],
) -> list[TranscriptWordPayload]:
    """Remove known hallucination phrases from the START of transcript.

    Common YouTube intro/outro phrases like "Thank you for watching" often appear
    as hallucinations at the beginning of music transcripts. This function removes
    them if they appear within the first few seconds and are followed by a gap.
    """
    if len(words) < 3:
        return words
    if not _env_bool("TRANSCRIBE_HEAD_PHRASE_FILTER", True):
        return words

    min_gap_after_sec = _env_float("TRANSCRIBE_HEAD_PHRASE_MIN_GAP_SEC", 0.2, 0.0)
    max_phrase_span_sec = _env_float("TRANSCRIBE_HEAD_PHRASE_MAX_SPAN_SEC", 15.0, 0.1)
    max_phrase_start_sec = _env_float("TRANSCRIBE_HEAD_PHRASE_MAX_START_SEC", 20.0, 0.0)
    min_remaining_words = _env_int("TRANSCRIBE_HEAD_PHRASE_MIN_REMAINING_WORDS", 5, 0)
    max_phrase_words = max(
        2, min(_env_int("TRANSCRIBE_HEAD_PHRASE_MAX_WORDS", 7, 2), 10)
    )

    # Only process if first word starts within the max start window
    first_start = float(words[0].start_sec)
    if first_start > max_phrase_start_sec:
        return words

    normalized_tokens = [_normalize_token(item.text) for item in words]
    total = len(words)

    # Try matching phrases of different lengths starting from longest
    for phrase_len in range(min(max_phrase_words, total), 1, -1):
        phrase_tokens = tuple(normalized_tokens[:phrase_len])
        if any(not token for token in phrase_tokens):
            continue
        if phrase_tokens not in _HEAD_HALLUCINATION_PHRASES:
            continue

        remaining = total - phrase_len
        if remaining < min_remaining_words:
            continue

        phrase_start = float(words[0].start_sec)
        phrase_end = float(words[phrase_len - 1].end_sec)
        phrase_span = max(0.0, phrase_end - phrase_start)
        if phrase_span > max_phrase_span_sec:
            continue

        # Check for gap after the phrase
        next_start = float(words[phrase_len].start_sec)
        gap_after = max(0.0, next_start - phrase_end)
        if gap_after < min_gap_after_sec:
            continue

        # Remove the head hallucination phrase
        trimmed = words[phrase_len:]
        return trimmed if trimmed else words

    return words


def _trim_prompt_leakage_phrases(
    words: list[TranscriptWordPayload],
) -> list[TranscriptWordPayload]:
    if len(words) < 2:
        return words
    if not _env_bool("TRANSCRIBE_PROMPT_LEAKAGE_FILTER", True):
        return words
    if _runtime_profile() not in {"music", "mixed"}:
        return words

    normalized_tokens = [_normalize_token(item.text) for item in words]
    kept: list[TranscriptWordPayload] = []
    index = 0
    while index < len(words):
        matched = False
        for phrase in _PROMPT_LEAKAGE_PHRASES:
            phrase_len = len(phrase)
            if index + phrase_len > len(words):
                continue
            if tuple(normalized_tokens[index : index + phrase_len]) == phrase:
                index += phrase_len
                matched = True
                break
        if matched:
            continue
        token = normalized_tokens[index]
        if token in _PROMPT_LEAKAGE_SINGLETONS:
            index += 1
            continue
        kept.append(words[index])
        index += 1

    return kept if kept else words


def _remove_hallucination_phrases_in_silence_gaps(
    words: list[TranscriptWordPayload],
    duration_sec: float,
) -> list[TranscriptWordPayload]:
    """Remove known hallucination phrases that appear *anywhere* in the transcript.

    Whisper / Groq commonly hallucinates YouTube-style sign-off phrases like
    "thank you for watching" or "please subscribe" during silence or music
    transitions in the *middle* of a recording — not just at the head / tail.
    This function scans every position and removes a match when the phrase is
    clearly isolated by silence gaps on both sides, making it very unlikely to
    be real speech.

    The filter is conservative by default:
      • requires a gap of at least ``TRANSCRIBE_ANYWHERE_PHRASE_MIN_GAP_SEC``
        (default 0.8 s) before the phrase AND after it (or near the
        start/end of the audio).
      • the entire phrase must fit within ``TRANSCRIBE_ANYWHERE_PHRASE_MAX_SPAN_SEC``
        (default 3.5 s).
      • at least ``TRANSCRIBE_ANYWHERE_PHRASE_MIN_REMAINING_WORDS`` (default 3)
        real words must remain outside the detected phrase cluster.
    """
    if len(words) < 3 or duration_sec <= 0:
        return words
    if not _env_bool("TRANSCRIBE_ANYWHERE_PHRASE_FILTER", True):
        return words

    min_gap_sec = _env_float("TRANSCRIBE_ANYWHERE_PHRASE_MIN_GAP_SEC", 0.8, 0.2)
    max_phrase_span_sec = _env_float(
        "TRANSCRIBE_ANYWHERE_PHRASE_MAX_SPAN_SEC", 3.5, 0.5
    )
    max_phrase_words = max(
        2, min(_env_int("TRANSCRIBE_ANYWHERE_PHRASE_MAX_WORDS", 8, 2), 12)
    )
    min_remaining_words = _env_int(
        "TRANSCRIBE_ANYWHERE_PHRASE_MIN_REMAINING_WORDS", 3, 0
    )

    normalized_tokens = [_normalize_token(item.text) for item in words]
    drop_indices: set[int] = set()
    total = len(words)

    for i in range(total):
        if i in drop_indices:
            continue

        # Gap before this word
        if i == 0:
            gap_before = float(words[0].start_sec)
        else:
            gap_before = max(
                0.0, float(words[i].start_sec) - float(words[i - 1].end_sec)
            )

        # Require a meaningful silence before the phrase so we don’t accidentally
        # clip real speech that happens to match (e.g., genuine gratitude mid-talk).
        if gap_before < min_gap_sec:
            continue

        # Check single-word singletons first (e.g., isolated "subscribe")
        singleton_token = normalized_tokens[i]
        if singleton_token in _HALLUCINATION_SINGLETONS_IN_GAPS:
            # Also need a gap after
            if i + 1 < total:
                gap_after = max(
                    0.0, float(words[i + 1].start_sec) - float(words[i].end_sec)
                )
            else:
                gap_after = max(0.0, duration_sec - float(words[i].end_sec))
            if gap_after >= min_gap_sec:
                remaining = total - 1  # would remove 1 word
                if remaining >= min_remaining_words:
                    drop_indices.add(i)
                    continue

        # Try matching multi-word phrases from longest to shortest
        matched_phrase_len = 0
        for phrase_len in range(min(max_phrase_words, total - i), 1, -1):
            phrase_tokens = tuple(normalized_tokens[i : i + phrase_len])
            if any(not token for token in phrase_tokens):
                continue
            if phrase_tokens not in _ANYWHERE_HALLUCINATION_PHRASES:
                continue

            phrase_start = float(words[i].start_sec)
            phrase_end = float(words[i + phrase_len - 1].end_sec)
            if (phrase_end - phrase_start) > max_phrase_span_sec:
                continue

            last_idx = i + phrase_len - 1
            if last_idx + 1 < total:
                gap_after = max(
                    0.0,
                    float(words[last_idx + 1].start_sec) - phrase_end,
                )
                near_end = (duration_sec - phrase_end) <= min_gap_sec * 1.5
            else:
                gap_after = max(0.0, duration_sec - phrase_end)
                near_end = True

            # Both sides must be silent (or phrase is near start/end of file)
            if gap_after < min_gap_sec and not near_end:
                continue

            remaining = total - phrase_len
            if remaining < min_remaining_words:
                continue

            matched_phrase_len = phrase_len
            break

        if matched_phrase_len:
            for offset in range(matched_phrase_len):
                drop_indices.add(i + offset)

    if not drop_indices:
        return words

    dropped_texts = [words[idx].text for idx in sorted(drop_indices)]
    _perf_logger.debug(
        "Hallucination anywhere-filter removed %d word(s): %s",
        len(drop_indices),
        dropped_texts,
    )
    filtered = [word for idx, word in enumerate(words) if idx not in drop_indices]
    return filtered if filtered else words


def _drop_sparse_words_in_music_gaps(
    words: list[TranscriptWordPayload],
    duration_sec: float,
) -> list[TranscriptWordPayload]:
    """Remove sparse isolated words that appear in low-density regions of songs.

    Songs typically have dense clusters of lyrics (verses, chorus) separated by
    instrumental breaks. Whisper/Groq often hallucinates random words during
    these breaks. This filter computes the local word density and drops words
    in regions that are dramatically sparser than the densest sections — those
    words are almost certainly hallucinations in instrumental gaps.
    """
    if len(words) < 6 or duration_sec <= 0:
        return words
    profile = _runtime_profile()
    mode = _runtime_mode()
    if profile not in {"music", "mixed"} and mode != "song":
        return words
    if not _env_bool("TRANSCRIBE_MUSIC_GAP_DENSITY_FILTER", True):
        return words

    density_window_sec = _env_float(
        "TRANSCRIBE_MUSIC_GAP_DENSITY_WINDOW_SEC", 6.0, 2.0
    )
    density_ratio_threshold = _env_float(
        "TRANSCRIBE_MUSIC_GAP_DENSITY_RATIO", 0.15, 0.01
    )
    min_dense_cluster_words = _env_int(
        "TRANSCRIBE_MUSIC_GAP_MIN_DENSE_WORDS", 6, 2
    )
    max_sparse_cluster_words = _env_int(
        "TRANSCRIBE_MUSIC_GAP_MAX_SPARSE_WORDS", 4, 1
    )
    min_gap_around_sec = _env_float(
        "TRANSCRIBE_MUSIC_GAP_MIN_SURROUNDING_GAP_SEC", 1.5, 0.3
    )

    # Step 1: compute per-word local density (words per second in surrounding window)
    word_densities: list[float] = []
    for i, word in enumerate(words):
        center = (float(word.start_sec) + float(word.end_sec)) * 0.5
        window_start = max(0.0, center - density_window_sec * 0.5)
        window_end = min(duration_sec, center + density_window_sec * 0.5)
        window_span = max(window_end - window_start, 0.5)
        count_in_window = sum(
            1 for w in words
            if float(w.end_sec) > window_start and float(w.start_sec) < window_end
        )
        word_densities.append(count_in_window / window_span)

    # Step 2: find the peak density (representative of actual lyrics sections)
    if not word_densities:
        return words
    peak_density = max(word_densities)
    if peak_density < 0.5:  # If peak is very low, entire track may be sparse — skip
        return words

    # Step 3: identify words in sparse regions below the threshold
    drop_indices: set[int] = set()
    i = 0
    while i < len(words):
        if word_densities[i] >= peak_density * density_ratio_threshold:
            i += 1
            continue

        # Start a sparse cluster
        cluster = [i]
        j = i + 1
        while j < len(words) and word_densities[j] < peak_density * density_ratio_threshold:
            cluster.append(j)
            j += 1

        # Only drop small clusters (large ones might be actual quiet vocal sections)
        if len(cluster) <= max_sparse_cluster_words:
            cluster_start = float(words[cluster[0]].start_sec)
            cluster_end = float(words[cluster[-1]].end_sec)

            gap_before = (
                cluster_start
                if cluster[0] == 0
                else max(0.0, cluster_start - float(words[cluster[0] - 1].end_sec))
            )
            gap_after = (
                max(0.0, duration_sec - cluster_end)
                if cluster[-1] == len(words) - 1
                else max(0.0, float(words[cluster[-1] + 1].start_sec) - cluster_end)
            )

            if gap_before >= min_gap_around_sec or gap_after >= min_gap_around_sec:
                drop_indices.update(cluster)

        i = j

    if not drop_indices:
        return words

    _perf_logger.debug(
        "Music gap density filter removed %d word(s): %s",
        len(drop_indices),
        [words[idx].text for idx in sorted(drop_indices)],
    )
    filtered = [word for idx, word in enumerate(words) if idx not in drop_indices]
    return filtered if filtered else words


def _apply_word_filters(
    words: list[TranscriptWordPayload],
    duration_sec: float,
    *,
    audio_path: str | None = None,
) -> list[TranscriptWordPayload]:
    filtered = list(words)
    if _env_bool("TRANSCRIBE_HALLUCINATION_FILTER", True):
        filtered = _detect_hallucinations(filtered)
        filtered = _detect_sparse_hallucinations(filtered, duration_sec)
        filtered = _drop_words_in_nonvocal_regions(
            filtered, duration_sec, audio_path=audio_path
        )
        # Song/music-specific: remove isolated words in low-density instrumental gaps
        filtered = _drop_sparse_words_in_music_gaps(filtered, duration_sec)
    filtered = _trim_prompt_leakage_phrases(filtered)
    # Remove hallucination phrases from both head and tail of transcript
    filtered = _trim_known_head_hallucination(filtered)
    filtered = _trim_known_tail_hallucination(filtered, duration_sec)
    filtered = _trim_sparse_music_tail(filtered, duration_sec)
    # Remove known hallucination phrases that appear anywhere in the transcript
    # (e.g., "thank you for watching" mid-video during a silence gap).
    filtered = _remove_hallucination_phrases_in_silence_gaps(filtered, duration_sec)
    return filtered


def _word_midpoint_sec(word: TranscriptWordPayload) -> float:
    return (float(word.start_sec) + float(word.end_sec)) * 0.5


def _song_validator_support_flags(
    words: list[TranscriptWordPayload],
    validator_words: list[TranscriptWordPayload],
    *,
    time_pad_sec: float,
) -> list[bool]:
    validator_index: dict[str, list[TranscriptWordPayload]] = {}
    for validator_word in validator_words:
        token = _normalize_token(validator_word.text)
        if not token:
            continue
        validator_index.setdefault(token, []).append(validator_word)

    supported: list[bool] = []
    for word in words:
        token = _normalize_token(word.text)
        if not token:
            supported.append(True)
            continue
        midpoint = _word_midpoint_sec(word)
        matches = validator_index.get(token, [])
        is_supported = any(
            abs(_word_midpoint_sec(candidate) - midpoint) <= time_pad_sec
            or (
                float(candidate.start_sec) <= (float(word.end_sec) + time_pad_sec)
                and float(candidate.end_sec) >= (float(word.start_sec) - time_pad_sec)
            )
            for candidate in matches
        )
        supported.append(is_supported)
    return supported


def _stabilize_song_words_with_validator(
    words: list[TranscriptWordPayload],
    validator_words: list[TranscriptWordPayload],
    duration_sec: float,
) -> list[TranscriptWordPayload]:
    if len(words) < 4 or len(validator_words) < 2:
        return words

    time_pad_sec = _env_float("TRANSCRIBE_SONG_VALIDATION_TIME_PAD_SEC", 0.55, 0.1)
    cluster_gap_sec = _env_float(
        "TRANSCRIBE_SONG_VALIDATION_CLUSTER_GAP_SEC", 0.55, 0.05
    )
    boundary_gap_sec = _env_float(
        "TRANSCRIBE_SONG_VALIDATION_BOUNDARY_GAP_SEC", 0.9, 0.1
    )
    max_cluster_words = _env_int("TRANSCRIBE_SONG_VALIDATION_MAX_CLUSTER_WORDS", 8, 1)
    max_cluster_span_sec = _env_float(
        "TRANSCRIBE_SONG_VALIDATION_MAX_CLUSTER_SPAN_SEC", 2.4, 0.2
    )
    edge_window_sec = _env_float(
        "TRANSCRIBE_SONG_VALIDATION_EDGE_WINDOW_SEC", 18.0, 1.0
    )

    exact_supported = _song_validator_support_flags(
        words,
        validator_words,
        time_pad_sec=time_pad_sec,
    )

    soft_supported = list(exact_supported)
    for index, word in enumerate(words):
        if soft_supported[index]:
            continue
        token = _normalize_token(word.text)
        if not token or token not in _HALLUCINATION_STOPWORDS:
            continue
        if (index > 0 and exact_supported[index - 1]) or (
            index + 1 < len(words) and exact_supported[index + 1]
        ):
            soft_supported[index] = True

    validator_midpoints = [_word_midpoint_sec(word) for word in validator_words]
    drop_indices: set[int] = set()
    index = 0
    while index < len(words):
        if soft_supported[index]:
            index += 1
            continue
        cluster = [index]
        next_index = index + 1
        while next_index < len(words):
            gap_sec = max(
                0.0,
                float(words[next_index].start_sec)
                - float(words[next_index - 1].end_sec),
            )
            if soft_supported[next_index] or gap_sec > cluster_gap_sec:
                break
            cluster.append(next_index)
            next_index += 1

        cluster_start = float(words[cluster[0]].start_sec)
        cluster_end = float(words[cluster[-1]].end_sec)
        cluster_span = max(0.0, cluster_end - cluster_start)
        prev_gap = (
            cluster_start
            if cluster[0] == 0
            else max(0.0, cluster_start - float(words[cluster[0] - 1].end_sec))
        )
        next_gap = (
            max(0.0, duration_sec - cluster_end)
            if cluster[-1] == len(words) - 1
            else max(0.0, float(words[cluster[-1] + 1].start_sec) - cluster_end)
        )
        at_edge = (
            cluster_start <= edge_window_sec
            or (duration_sec - cluster_end) <= edge_window_sec
        )
        surrounded_by_gap = prev_gap >= boundary_gap_sec or next_gap >= boundary_gap_sec
        validator_nearby = any(
            (cluster_start - time_pad_sec) <= midpoint <= (cluster_end + time_pad_sec)
            for midpoint in validator_midpoints
        )

        if (
            len(cluster) <= max_cluster_words
            and cluster_span <= max_cluster_span_sec
            and (at_edge or surrounded_by_gap or not validator_nearby)
        ):
            drop_indices.update(cluster)

        index = next_index

    if not drop_indices:
        return words

    filtered = [word for idx, word in enumerate(words) if idx not in drop_indices]
    return filtered if filtered else words


def stabilize_song_transcript(
    payload: TranscriptPayload,
    *,
    path: str,
    duration_sec: float,
) -> TranscriptPayload:
    if not payload.words or payload.is_mock:
        return payload
    if not _env_bool("TRANSCRIBE_SONG_VALIDATION_ENABLED", True):
        return payload
    if _runtime_mode() != "song":
        return payload
    max_duration_sec = _env_float(
        "TRANSCRIBE_SONG_VALIDATION_MAX_DURATION_SEC", 120.0, 5.0
    )
    if float(duration_sec) > max_duration_sec:
        return payload
    if payload.source.endswith("_lyrics_ref"):
        return payload

    validator_model = (
        os.getenv("TRANSCRIBE_SONG_VALIDATION_MODEL", "base.en") or "base.en"
    ).strip() or "base.en"
    validator_payload = _build_from_faster_whisper(
        path,
        duration_sec,
        model_name=validator_model,
        force_vad_filter=False,
    )
    if validator_payload is None or not validator_payload.words:
        return payload

    stabilized_words = _stabilize_song_words_with_validator(
        payload.words,
        validator_payload.words,
        max(float(duration_sec), 0.0),
    )
    min_retained_words = _env_int("TRANSCRIBE_SONG_VALIDATION_MIN_RETAINED_WORDS", 3, 1)
    min_retained_ratio = _env_float(
        "TRANSCRIBE_SONG_VALIDATION_MIN_RETAINED_RATIO", 0.3, 0.0
    )
    if len(stabilized_words) >= max(
        min_retained_words,
        int(round(len(payload.words) * min_retained_ratio)),
    ):
        return TranscriptPayload(
            source=f"{payload.source}_validated",
            language=payload.language,
            text=" ".join(word.text for word in stabilized_words).strip(),
            words=stabilized_words,
            is_mock=payload.is_mock,
        )
    return payload


def _build_preprocess_filter_chain() -> str:
    default_chain = "pan=mono|c0=0.5*c0+0.5*c1"
    return (
        os.getenv("TRANSCRIBE_PREPROCESS_FILTER_CHAIN", default_chain) or default_chain
    ).strip()


def _vocal_isolation_allowed_for_profile(profile: str) -> bool:
    raw = (
        os.getenv("TRANSCRIBE_VOCAL_ISOLATION_PROFILES", "speech,mixed,music") or ""
    ).strip()
    if not raw:
        return True
    tokens = {token.strip().lower() for token in raw.split(",") if token.strip()}
    if not tokens:
        return True
    if "all" in tokens:
        return True
    return profile.strip().lower() in tokens


def _prepare_transcription_source(
    path: str, format: str = "wav"
) -> tuple[str, Path | None]:
    if not _env_bool("TRANSCRIBE_PREPROCESS_AUDIO", True):
        return path, None

    source_path = Path(path)
    if not source_path.exists():
        return path, None

    tmp_dir = Path(os.getenv("TMP_DIR", settings.tmp_dir))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path = tmp_dir / f"transcribe-pre-{uuid4()}.{format}"
    sample_rate = _env_int("TRANSCRIBE_PREPROCESS_SAMPLE_RATE", 16000, 8000)
    filter_chain = _build_preprocess_filter_chain()
    cmd = [
        settings.ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-af",
        filter_chain,
    ]
    # Add codec for compressed formats
    if format == "mp3":
        cmd.extend(["-codec:a", "libmp3lame", "-b:a", "64k"])
    cmd.append(str(output_path))

    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return path, None

    if process.returncode != 0 or not output_path.exists():
        _perf_logger.warning(
            "[transcribe] ffmpeg preprocess failed (rc=%d): %s",
            process.returncode, process.stderr,
        )
        output_path.unlink(missing_ok=True)
        return path, None
    if output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        return path, None
    return str(output_path), output_path


@lru_cache(maxsize=1)
def _prime_cuda_runtime_libraries() -> None:
    # In venv installs, CUDA user-space libs can live under site-packages/nvidia/*/lib.
    # Preload them so CTranslate2 can resolve CUDA 12 symbols without system-wide installs.
    search_roots: list[Path] = []
    site_dir = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    search_roots.append(site_dir)

    loaded_any = False
    lib_names = [
        "libcudart.so.12",
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcudnn.so.9",
        "libcudnn_ops_infer.so.9",
        "libcudnn_cnn_infer.so.9",
    ]
    subdirs = [
        ("nvidia", "cuda_runtime", "lib"),
        ("nvidia", "cublas", "lib"),
        ("nvidia", "cudnn", "lib"),
    ]

    for root in search_roots:
        for subdir in subdirs:
            lib_dir = root.joinpath(*subdir)
            if not lib_dir.exists():
                continue
            for name in lib_names:
                lib_path = lib_dir / name
                if not lib_path.exists():
                    continue
                try:
                    ctypes.CDLL(str(lib_path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
                    loaded_any = True
                except OSError:
                    continue

    if loaded_any:
        ld_library_path = os.getenv("LD_LIBRARY_PATH", "")
        paths = [
            str(search_roots[0] / "nvidia" / "cuda_runtime" / "lib"),
            str(search_roots[0] / "nvidia" / "cublas" / "lib"),
            str(search_roots[0] / "nvidia" / "cudnn" / "lib"),
        ]
        merged = [path for path in paths if Path(path).exists()]
        if ld_library_path:
            merged.append(ld_library_path)
        os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


# maxsize=1: keep only one Whisper model in memory at a time to prevent OOM.
# Previously maxsize=4, which could cache 4 different models (~600MB-6GB total).
@lru_cache(maxsize=1)
def _load_faster_whisper_model(
    model_name: str, device: str, compute_type: str
) -> object:
    _prime_cuda_runtime_libraries()
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]

    return WhisperModel(model_name, device=device, compute_type=compute_type)


@lru_cache(maxsize=1)
def _gpu_available() -> bool:
    _prime_cuda_runtime_libraries()
    try:
        import ctranslate2  # type: ignore[import-not-found]

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def _resolve_device_and_compute_type() -> tuple[str, str]:
    raw_device = (os.getenv("TRANSCRIBE_DEVICE", "auto") or "auto").strip().lower()
    if raw_device in {"", "auto"}:
        device = "cuda" if _gpu_available() else "cpu"
    else:
        device = raw_device

    raw_compute = (
        (os.getenv("TRANSCRIBE_COMPUTE_TYPE", "auto") or "auto").strip().lower()
    )
    if raw_compute in {"", "auto"}:
        if device == "cuda":
            compute_type = (
                os.getenv("TRANSCRIBE_COMPUTE_TYPE_CUDA", "float16") or "float16"
            ).strip() or "float16"
        else:
            compute_type = (
                os.getenv("TRANSCRIBE_COMPUTE_TYPE_CPU", "int8") or "int8"
            ).strip() or "int8"
    else:
        compute_type = raw_compute
    return device, compute_type


def _build_from_faster_whisper(
    path: str,
    duration_sec: float,
    *,
    model_name: str | None = None,
    beam_size: int | None = None,
    force_vad_filter: bool | None = None,
    language_hint: str | None = None,
) -> TranscriptPayload | None:
    transcribe_path, cleanup_path = _prepare_transcription_source(path)
    resolved_model_name = (
        model_name or os.getenv("TRANSCRIBE_MODEL", "base.en")
    ).strip() or "base.en"
    device, compute_type = _resolve_device_and_compute_type()

    try:
        try:
            model = _load_faster_whisper_model(
                resolved_model_name, device, compute_type
            )
        except Exception:  # noqa: BLE001
            if device == "cuda":
                # CUDA may be unavailable despite configuration; retry on CPU.
                try:
                    model = _load_faster_whisper_model(
                        resolved_model_name,
                        "cpu",
                        (
                            os.getenv("TRANSCRIBE_COMPUTE_TYPE_CPU", "int8") or "int8"
                        ).strip()
                        or "int8",
                    )
                except Exception:  # noqa: BLE001
                    return None
            else:
                return None

        transcribe_kwargs: dict[str, object] = {
            "beam_size": int(beam_size)
            if beam_size is not None
            else _env_int("TRANSCRIBE_BEAM_SIZE", 5, 1),
            "word_timestamps": True,
            "condition_on_previous_text": _env_bool(
                "TRANSCRIBE_CONDITION_ON_PREVIOUS_TEXT", False
            ),
            "no_speech_threshold": _env_float(
                "TRANSCRIBE_NO_SPEECH_THRESHOLD", 0.6, 0.0
            ),
            "log_prob_threshold": _env_float(
                "TRANSCRIBE_LOG_PROB_THRESHOLD", -1.0, -10.0
            ),
        }
        # Temperature fallback: start deterministic, retry with higher temperature on failure
        raw_temps = (os.getenv("TRANSCRIBE_TEMPERATURE", "") or "").strip()
        if raw_temps:
            try:
                temps = [float(t.strip()) for t in raw_temps.split(",") if t.strip()]
                if len(temps) == 1:
                    transcribe_kwargs["temperature"] = temps[0]
                elif temps:
                    transcribe_kwargs["temperature"] = temps
            except ValueError:
                pass  # Use faster-whisper default
        language = (
            _normalize_language_code(language_hint)
            or (os.getenv("TRANSCRIBE_LANGUAGE", "") or "").strip()
        )
        if language:
            transcribe_kwargs["language"] = language
        initial_prompt = (os.getenv("TRANSCRIBE_INITIAL_PROMPT", "") or "").strip()
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt
        vad_filter_enabled = (
            _env_bool("TRANSCRIBE_VAD_FILTER", False)
            if force_vad_filter is None
            else force_vad_filter
        )
        if vad_filter_enabled:
            transcribe_kwargs["vad_filter"] = True
            transcribe_kwargs["vad_parameters"] = {"min_silence_duration_ms": 250}

        try:
            segments, info = model.transcribe(transcribe_path, **transcribe_kwargs)
        except Exception:  # noqa: BLE001
            if device == "cuda":
                # CUDA may partially initialize but fail during decode (e.g., missing user-space CUDA libs).
                try:
                    cpu_compute_type = (
                        os.getenv("TRANSCRIBE_COMPUTE_TYPE_CPU", "int8") or "int8"
                    ).strip() or "int8"
                    cpu_model = _load_faster_whisper_model(
                        resolved_model_name, "cpu", cpu_compute_type
                    )
                    segments, info = cpu_model.transcribe(
                        transcribe_path, **transcribe_kwargs
                    )
                except Exception:  # noqa: BLE001
                    return None
            else:
                return None

        words: list[TranscriptWordPayload] = []
        for segment in segments:
            segment_text = _clean_word(str(getattr(segment, "text", "") or ""))
            segment_start = float(getattr(segment, "start", 0.0) or 0.0)
            segment_end = float(
                getattr(segment, "end", segment_start + 0.2) or (segment_start + 0.2)
            )

            segment_words = list(getattr(segment, "words", []) or [])
            if segment_words:
                for word in segment_words:
                    token = _clean_word(str(getattr(word, "word", "") or ""))
                    if not token:
                        continue
                    start_sec = float(
                        getattr(word, "start", segment_start) or segment_start
                    )
                    end_sec = float(
                        getattr(word, "end", max(start_sec + 0.05, segment_end))
                        or max(start_sec + 0.05, segment_end)
                    )
                    confidence = _normalize_confidence(
                        getattr(word, "probability", None)
                    )
                    words.append(
                        TranscriptWordPayload(
                            id=str(uuid4()),
                            text=token,
                            start_sec=start_sec,
                            end_sec=end_sec,
                            confidence=confidence,
                        )
                    )
                continue

            if not segment_text:
                continue
            parts = segment_text.split()
            if not parts:
                continue
            span = max(segment_end - segment_start, 0.1)
            step = span / len(parts)
            for idx, token in enumerate(parts):
                start_sec = segment_start + (idx * step)
                end_sec = segment_start + ((idx + 1) * step)
                words.append(
                    TranscriptWordPayload(
                        id=str(uuid4()),
                        text=token,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        confidence=None,
                    )
                )

        normalized = _apply_word_filters(
            _normalize_words(words, duration_sec),
            duration_sec,
            audio_path=transcribe_path,
        )
        if len(normalized) < 2:
            return None
        text = " ".join(item.text for item in normalized)
        language = getattr(info, "language", None)
        return TranscriptPayload(
            source="faster_whisper",
            language=str(language) if language else None,
            text=text,
            words=normalized,
            is_mock=False,
        )
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)
        # Release memory from model inference tensors
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _min_expected_word_count(duration_sec: float) -> int:
    words_per_sec = _env_float("TRANSCRIBE_MIN_WORDS_PER_SEC", 0.45, 0.05)
    return max(8, int(round(max(duration_sec, 1.0) * words_per_sec)))


def _is_low_coverage(payload: TranscriptPayload, duration_sec: float) -> bool:
    return len(payload.words) < _min_expected_word_count(duration_sec)


def _end_coverage_ratio(payload: TranscriptPayload, duration_sec: float) -> float:
    if duration_sec <= 0 or not payload.words:
        return 0.0
    last_end = max(float(word.end_sec) for word in payload.words)
    return max(0.0, min(last_end / max(duration_sec, 0.1), 1.0))


def _max_word_gap_sec(payload: TranscriptPayload, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    if not payload.words:
        return duration_sec
    ordered = sorted(
        payload.words, key=lambda item: (float(item.start_sec), float(item.end_sec))
    )
    max_gap = max(float(ordered[0].start_sec), 0.0)
    for idx in range(1, len(ordered)):
        prev_end = float(ordered[idx - 1].end_sec)
        start = float(ordered[idx].start_sec)
        max_gap = max(max_gap, max(0.0, start - prev_end))
    max_gap = max(max_gap, max(0.0, duration_sec - float(ordered[-1].end_sec)))
    return max_gap


def _has_suspicious_long_gap(payload: TranscriptPayload, duration_sec: float) -> bool:
    threshold_sec = _env_float("TRANSCRIBE_MAX_WORD_GAP_SEC", 24.0, 2.0)
    min_words = _env_int("TRANSCRIBE_GAP_CHECK_MIN_WORDS", 20, 0)
    if len(payload.words) < min_words:
        return False
    if duration_sec < threshold_sec * 1.5:
        return False
    return _max_word_gap_sec(payload, duration_sec) >= threshold_sec


def _has_sparse_window(payload: TranscriptPayload, duration_sec: float) -> bool:
    min_words_for_check = _env_int("TRANSCRIBE_GAP_CHECK_MIN_WORDS", 20, 0)
    if len(payload.words) < min_words_for_check:
        return False

    window_sec = _env_float("TRANSCRIBE_SPARSE_WINDOW_SEC", 20.0, 5.0)
    min_words = _env_int("TRANSCRIBE_SPARSE_WINDOW_MIN_WORDS", 4, 0)
    if min_words <= 0:
        return False
    if duration_sec < window_sec * 1.5:
        return False

    step_sec = _env_float(
        "TRANSCRIBE_SPARSE_WINDOW_STEP_SEC", max(5.0, window_sec / 2.0), 1.0
    )
    start_at_sec = _env_float("TRANSCRIBE_SPARSE_WINDOW_START_SEC", 20.0, 0.0)
    starts = sorted(float(word.start_sec) for word in payload.words)
    if not starts:
        return True

    left = 0
    right = 0
    total = len(starts)
    cursor = max(0.0, min(start_at_sec, max(0.0, duration_sec - window_sec)))
    while cursor + window_sec <= duration_sec + 1e-6:
        while left < total and starts[left] < cursor:
            left += 1
        if right < left:
            right = left
        window_end = cursor + window_sec
        while right < total and starts[right] < window_end:
            right += 1
        if (right - left) < min_words:
            return True
        cursor += step_sec
    return False


def _find_long_gaps(
    payload: TranscriptPayload,
    duration_sec: float,
    *,
    min_gap_sec: float,
) -> list[tuple[float, float]]:
    if duration_sec <= 0 or min_gap_sec <= 0:
        return []
    if not payload.words:
        return [(0.0, duration_sec)] if duration_sec >= min_gap_sec else []

    ordered = sorted(
        payload.words, key=lambda item: (float(item.start_sec), float(item.end_sec))
    )
    gaps: list[tuple[float, float]] = []
    cursor = max(0.0, float(ordered[0].start_sec))
    if cursor >= min_gap_sec:
        gaps.append((0.0, cursor))

    for idx in range(1, len(ordered)):
        prev_end = float(ordered[idx - 1].end_sec)
        start = float(ordered[idx].start_sec)
        if start - prev_end >= min_gap_sec:
            gaps.append((max(0.0, prev_end), min(duration_sec, start)))
    tail_gap = max(0.0, duration_sec - float(ordered[-1].end_sec))
    if tail_gap >= min_gap_sec:
        gaps.append((max(0.0, float(ordered[-1].end_sec)), duration_sec))
    return gaps


def _source_pass_priority(value: str | None) -> int:
    normalized = _normalize_source_pass(value)
    if normalized == "manual":
        return 0
    if normalized == "primary":
        return 1
    if normalized == "retry":
        return 2
    if normalized == "rescue":
        return 3
    return 4


def _merge_gap_fill_transcript(
    primary: TranscriptPayload,
    secondary: TranscriptPayload | None,
    duration_sec: float,
) -> TranscriptPayload | None:
    if secondary is None or not secondary.words:
        return None

    default_gap_fill_sec = max(
        4.0, _env_float("TRANSCRIBE_MAX_WORD_GAP_SEC", 24.0, 2.0) * 0.75
    )
    min_gap_sec = _env_float("TRANSCRIBE_GAP_FILL_MIN_SEC", default_gap_fill_sec, 1.0)
    pad_sec = _env_float("TRANSCRIBE_GAP_FILL_PAD_SEC", 0.18, 0.0)
    gaps = _find_long_gaps(primary, duration_sec, min_gap_sec=min_gap_sec)
    if not gaps:
        return None

    primary_source_pass = infer_source_pass(primary.source)
    secondary_source_pass = infer_source_pass(secondary.source)
    primary_words = [
        _copy_word_payload(
            word,
            source_pass=_normalize_source_pass(word.source_pass) or primary_source_pass,
        )
        for word in primary.words
    ]
    additions: list[TranscriptWordPayload] = []
    for word in sorted(
        secondary.words, key=lambda item: (float(item.start_sec), float(item.end_sec))
    ):
        center = (float(word.start_sec) + float(word.end_sec)) / 2.0
        for start_sec, end_sec in gaps:
            if (start_sec - pad_sec) <= center <= (end_sec + pad_sec):
                additions.append(
                    _copy_word_payload(
                        word,
                        source_pass=_normalize_source_pass(word.source_pass)
                        or secondary_source_pass,
                    )
                )
                break
    if not additions:
        return None

    merged = sorted(
        primary_words + additions,
        key=lambda item: (
            float(item.start_sec),
            float(item.end_sec),
            _source_pass_priority(item.source_pass),
            item.text.lower(),
        ),
    )
    deduped: list[TranscriptWordPayload] = []
    for word in merged:
        if deduped:
            prev = deduped[-1]
            if (
                word.text.strip().lower() == prev.text.strip().lower()
                and abs(float(word.start_sec) - float(prev.start_sec)) <= 0.05
                and abs(float(word.end_sec) - float(prev.end_sec)) <= 0.05
            ):
                continue
        deduped.append(word)

    normalized = _apply_word_filters(
        _normalize_words(deduped, duration_sec), duration_sec
    )
    if len(normalized) <= len(primary.words):
        return None
    return TranscriptPayload(
        source=f"{primary.source}_gapfill",
        language=primary.language or secondary.language,
        text=" ".join(word.text for word in normalized),
        words=normalized,
        is_mock=False,
    )


def _gap_fill_word_count(
    primary: TranscriptPayload,
    secondary: TranscriptPayload,
    duration_sec: float,
) -> int:
    default_gap_fill_sec = max(
        4.0, _env_float("TRANSCRIBE_MAX_WORD_GAP_SEC", 24.0, 2.0) * 0.75
    )
    min_gap_sec = _env_float("TRANSCRIBE_GAP_FILL_MIN_SEC", default_gap_fill_sec, 1.0)
    pad_sec = _env_float("TRANSCRIBE_GAP_FILL_PAD_SEC", 0.18, 0.0)
    gaps = _find_long_gaps(primary, duration_sec, min_gap_sec=min_gap_sec)
    if not gaps:
        return 0
    filled = 0
    for word in secondary.words:
        center = (float(word.start_sec) + float(word.end_sec)) / 2.0
        for start_sec, end_sec in gaps:
            if (start_sec - pad_sec) <= center <= (end_sec + pad_sec):
                filled += 1
                break
    return filled


def _pick_best_gap_fill_candidate(
    primary: TranscriptPayload,
    candidates: list[TranscriptPayload],
    duration_sec: float,
) -> TranscriptPayload | None:
    best: TranscriptPayload | None = None
    best_score: tuple[int, float, int] | None = None
    for candidate in candidates:
        score = (
            _gap_fill_word_count(primary, candidate, duration_sec),
            -_max_word_gap_sec(candidate, duration_sec),
            len(candidate.words),
        )
        if best is None or best_score is None or score > best_score:
            best = candidate
            best_score = score
    return best


def _confidence_stats(payload: TranscriptPayload) -> tuple[int, int, float]:
    threshold = _env_float("TRANSCRIBE_LOW_CONFIDENCE_THRESHOLD", 0.6, 0.0)
    values: list[float] = []
    for word in payload.words:
        confidence = _normalize_confidence(word.confidence)
        if confidence is None:
            continue
        values.append(confidence)
    if not values:
        return 0, 0, 1.0
    low_count = sum(1 for value in values if value < threshold)
    avg_confidence = sum(values) / len(values)
    return len(values), low_count, avg_confidence


def _is_low_confidence_quality(payload: TranscriptPayload) -> bool:
    min_words = _env_int("TRANSCRIBE_LOW_CONFIDENCE_MIN_WORDS", 30, 0)
    trigger_ratio = _env_float("TRANSCRIBE_LOW_CONFIDENCE_RATIO_TRIGGER", 0.18, 0.0)
    total, low_count, _ = _confidence_stats(payload)
    if total < min_words:
        return False
    return (low_count / total) >= trigger_ratio


def _pick_better_transcript(
    primary: TranscriptPayload | None,
    secondary: TranscriptPayload | None,
    duration_sec: float,
) -> TranscriptPayload | None:
    if primary is None:
        return secondary
    if secondary is None:
        return primary

    primary_count = len(primary.words)
    secondary_count = len(secondary.words)
    primary_low = _is_low_coverage(primary, duration_sec)
    secondary_low = _is_low_coverage(secondary, duration_sec)

    if primary_low and not secondary_low:
        return secondary
    if secondary_count >= primary_count + max(12, int(round(primary_count * 0.15))):
        return secondary

    primary_coverage = _end_coverage_ratio(primary, duration_sec)
    secondary_coverage = _end_coverage_ratio(secondary, duration_sec)
    if secondary_coverage >= primary_coverage + 0.08:
        return secondary

    primary_gap = _max_word_gap_sec(primary, duration_sec)
    secondary_gap = _max_word_gap_sec(secondary, duration_sec)
    if secondary_gap + 5.0 <= primary_gap:
        return secondary

    min_words = _env_int("TRANSCRIBE_LOW_CONFIDENCE_MIN_WORDS", 30, 0)
    primary_total, primary_low_count, primary_avg = _confidence_stats(primary)
    secondary_total, secondary_low_count, secondary_avg = _confidence_stats(secondary)
    if primary_total >= min_words and secondary_total >= min_words:
        primary_low_ratio = primary_low_count / primary_total
        secondary_low_ratio = secondary_low_count / secondary_total
        # Prefer a meaningful drop in risky words.
        if secondary_low_ratio + 0.05 <= primary_low_ratio:
            return secondary
        # If risk ratios are similar, pick the one with better average confidence.
        if (
            secondary_low_ratio <= primary_low_ratio + 0.01
            and secondary_avg >= primary_avg + 0.03
        ):
            return secondary

    return primary


def _pick_better_transcript_with_language(
    primary: TranscriptPayload | None,
    secondary: TranscriptPayload | None,
    duration_sec: float,
    language_code: str | None,
) -> TranscriptPayload | None:
    preferred = _pick_better_transcript(primary, secondary, duration_sec)
    resolved_language = _normalize_language_code(language_code)
    if resolved_language is None:
        return preferred
    if primary is None or secondary is None:
        return preferred

    min_alpha = _env_int("TRANSCRIBE_LANGUAGE_SCRIPT_MIN_ALPHA", 24, 0)
    min_ratio = _env_float("TRANSCRIBE_LANGUAGE_SCRIPT_MIN_RATIO", 0.28, 0.0)
    protect_gap = _env_float("TRANSCRIBE_LANGUAGE_SCRIPT_PROTECT_GAP", 0.18, 0.0)
    sparse_words = _env_int("TRANSCRIBE_LANGUAGE_SCRIPT_SPARSE_WORDS", 4, 1)

    primary_alpha, primary_ratio = _payload_language_match_metrics(
        primary, resolved_language
    )
    secondary_alpha, secondary_ratio = _payload_language_match_metrics(
        secondary, resolved_language
    )
    primary_reliable = primary_alpha >= min_alpha
    secondary_reliable = secondary_alpha >= min_alpha

    # Prevent strong cross-language drift when the current transcript already
    # matches the requested language script.
    if (
        primary_reliable
        and primary_ratio >= min_ratio
        and secondary_reliable
        and (secondary_ratio + protect_gap) < primary_ratio
    ):
        if len(primary.words) <= sparse_words and len(secondary.words) >= (
            len(primary.words) + 20
        ):
            return preferred
        return primary

    # If secondary has clearly stronger script consistency for the requested
    # language, prefer it even when base heuristics are close.
    if (
        secondary_reliable
        and secondary_ratio >= min_ratio
        and secondary_ratio >= (primary_ratio + 0.10)
    ):
        return secondary

    return preferred


def _reliable_detected_indic_script(
    payload: TranscriptPayload | None,
    duration_sec: float,
) -> tuple[str | None, int, float]:
    if payload is None:
        return None, 0, 0.0
    language_code = _normalize_detected_language(payload.language)
    if not _is_indic_language(language_code):
        return language_code, 0, 0.0
    if _is_low_coverage(payload, duration_sec):
        return language_code, 0, 0.0
    min_alpha = _env_int("TRANSCRIBE_AUTO_ROUTE_SARVAM_MIN_ALPHA", 18, 0)
    min_ratio = _env_float("TRANSCRIBE_AUTO_ROUTE_SARVAM_MIN_RATIO", 0.30, 0.0)
    alpha_count, script_ratio = _payload_language_match_metrics(payload, language_code)
    if alpha_count >= min_alpha and script_ratio >= min_ratio:
        return language_code, alpha_count, script_ratio
    return language_code, alpha_count, script_ratio


def _pick_better_auto_indic_transcript(
    primary: TranscriptPayload | None,
    secondary: TranscriptPayload | None,
    duration_sec: float,
) -> TranscriptPayload | None:
    preferred = _pick_better_transcript(primary, secondary, duration_sec)
    if primary is None or secondary is None:
        return preferred

    secondary_language, secondary_alpha, secondary_ratio = (
        _reliable_detected_indic_script(secondary, duration_sec)
    )
    if not _is_indic_language(secondary_language):
        return preferred

    min_alpha = _env_int("TRANSCRIBE_AUTO_ROUTE_SARVAM_MIN_ALPHA", 18, 0)
    min_ratio = _env_float("TRANSCRIBE_AUTO_ROUTE_SARVAM_MIN_RATIO", 0.30, 0.0)
    secondary_reliable = secondary_alpha >= min_alpha and secondary_ratio >= min_ratio
    if not secondary_reliable:
        return preferred

    primary_language, primary_alpha, primary_ratio = _reliable_detected_indic_script(
        primary, duration_sec
    )
    primary_reliable = (
        _is_indic_language(primary_language)
        and primary_alpha >= min_alpha
        and primary_ratio >= min_ratio
    )
    prefer_reliable_sarvam = _env_bool(
        "TRANSCRIBE_AUTO_ROUTE_SARVAM_PREFER_RELIABLE_UNKNOWN", True
    )
    if (
        prefer_reliable_sarvam
        and secondary.source.startswith("sarvam")
        and (
            primary.source.startswith("groq")
            or not primary_reliable
            or primary_language != secondary_language
        )
    ):
        return secondary
    if not primary_reliable:
        return secondary
    return preferred


def _silence_ratio_for_profile(path: str, duration_sec: float) -> float | None:
    if duration_sec <= 0:
        return None
    if not Path(path).exists():
        return None

    min_analyze_sec = _env_float("TRANSCRIBE_PROFILE_MIN_ANALYZE_SEC", 25.0, 5.0)
    analyze_sec = min(
        duration_sec,
        _env_float("TRANSCRIBE_PROFILE_ANALYZE_SEC", 120.0, min_analyze_sec),
    )
    if analyze_sec < min_analyze_sec:
        return None

    noise_db = _env_float("TRANSCRIBE_PROFILE_SILENCE_NOISE_DB", -35.0, -80.0)
    min_silence_sec = _env_float("TRANSCRIBE_PROFILE_MIN_SILENCE_SEC", 0.35, 0.05)
    silences = detect_silence_ranges(
        path,
        noise_db=noise_db,
        min_silence_sec=min_silence_sec,
        max_duration_sec=analyze_sec,
    )
    if not silences:
        return 0.0

    silence_total = 0.0
    for start_sec, end_sec in silences:
        start = max(0.0, min(float(start_sec), analyze_sec))
        end = max(0.0, min(float(end_sec), analyze_sec))
        if end > start:
            silence_total += end - start
    return max(0.0, min(silence_total / analyze_sec, 1.0))


def _resolve_transcription_profile(path: str, duration_sec: float) -> str:
    requested = (os.getenv("TRANSCRIBE_PROFILE", "auto") or "auto").strip().lower()
    if requested in {"speech", "music", "mixed"}:
        return requested
    if requested not in {"", "auto"}:
        return "mixed"

    silence_ratio = _silence_ratio_for_profile(path, duration_sec)
    if silence_ratio is None:
        return "mixed"

    speech_min_ratio = _env_float(
        "TRANSCRIBE_PROFILE_SPEECH_MIN_SILENCE_RATIO", 0.10, 0.0
    )
    music_max_ratio = _env_float(
        "TRANSCRIBE_PROFILE_MUSIC_MAX_SILENCE_RATIO", 0.04, 0.0
    )
    if silence_ratio >= speech_min_ratio:
        return "speech"
    if silence_ratio <= music_max_ratio:
        return "music"
    return "mixed"


def _resolve_groq_prompt_strategy(
    profile: str,
    primary_prompt: str | None,
    retry_prompt: str | None,
    retry_try_no_prompt: bool,
) -> tuple[str | None, str | None, bool]:
    if profile == "speech":
        speech_primary_prompt = (
            os.getenv("TRANSCRIBE_GROQ_PROMPT_SPEECH", "") or ""
        ).strip() or primary_prompt
        speech_retry_prompt = (
            os.getenv("TRANSCRIBE_GROQ_RETRY_PROMPT_SPEECH", "") or ""
        ).strip() or None
        speech_retry_try_no_prompt = _env_bool(
            "TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT_SPEECH", False
        )
        return speech_primary_prompt, speech_retry_prompt, speech_retry_try_no_prompt
    if profile == "music":
        music_primary_prompt = (
            (os.getenv("TRANSCRIBE_GROQ_PROMPT_MUSIC", "") or "").strip()
            or primary_prompt
            or DEFAULT_MUSIC_RETRY_PROMPT
        )
        music_retry_prompt = (
            (os.getenv("TRANSCRIBE_GROQ_RETRY_PROMPT_MUSIC", "") or "").strip()
            or retry_prompt
            or DEFAULT_MUSIC_RETRY_PROMPT
        )
        music_retry_try_no_prompt = _env_bool(
            "TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT_MUSIC", False
        )
        return music_primary_prompt, music_retry_prompt, music_retry_try_no_prompt
    return primary_prompt, retry_prompt, retry_try_no_prompt


def _should_retry_groq_for_profile(
    profile: str, payload: TranscriptPayload, duration_sec: float
) -> bool:
    common_retry = _is_low_coverage(payload, duration_sec) or _has_suspicious_long_gap(
        payload, duration_sec
    )
    if common_retry:
        return True
    if profile == "speech":
        return False
    return _has_sparse_window(payload, duration_sec)


def _build_mock_transcript(duration_sec: float) -> TranscriptPayload:
    safe_duration = max(duration_sec, 3.0)
    base_words = [
        "this",
        "is",
        "a",
        "generated",
        "transcript",
        "preview",
        "edit",
        "the",
        "text",
        "to",
        "cut",
        "the",
        "video",
        "automatically",
    ]
    target_count = max(8, int(round(safe_duration * 2.2)))
    step = safe_duration / target_count
    words: list[TranscriptWordPayload] = []
    for idx in range(target_count):
        token = base_words[idx % len(base_words)]
        start_sec = idx * step
        end_sec = min(safe_duration, start_sec + max(0.08, step * 0.82))
        words.append(
            TranscriptWordPayload(
                id=str(uuid4()),
                text=token,
                start_sec=start_sec,
                end_sec=end_sec,
                confidence=None,
            )
        )
    normalized = _normalize_words(words, safe_duration)
    text = " ".join(item.text for item in normalized)
    return TranscriptPayload(
        source="mock",
        language="en",
        text=text,
        words=normalized,
        is_mock=True,
    )


# ---------------------------------------------------------------------------
# Groq Cloud API backend (much faster than local, uses Whisper Large V3)
# ---------------------------------------------------------------------------
def _cleanup_temp_path(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass


def _resolve_vocal_isolation_device() -> str:
    raw_device = (
        (os.getenv("TRANSCRIBE_VOCAL_ISOLATION_DEVICE", "auto") or "auto")
        .strip()
        .lower()
    )
    if raw_device in {"", "auto"}:
        return "cuda" if _gpu_available() else "cpu"
    if raw_device in {"cpu", "cuda"}:
        return raw_device
    return "cpu"


def _apply_vocal_isolation_priority(cmd: list[str]) -> list[str]:
    """Lower process scheduling priority to keep desktop/UI responsive."""
    nice_level = _env_int("TRANSCRIBE_VOCAL_ISOLATION_NICE", 10, 0)
    if nice_level <= 0:
        return cmd
    if os.name != "posix":
        return cmd
    if shutil.which("nice") is None:
        return cmd
    return ["nice", "-n", str(nice_level), *cmd]


def _backend_env_suffix(backend: str) -> str | None:
    normalized = backend.strip().lower().replace("-", "_")
    if normalized.startswith("melband_roformer"):
        return "MELBAND_ROFORMER"
    if normalized.startswith("htdemucs_ft"):
        return "HTDEMUCS_FT"
    if normalized.startswith("bs_roformer"):
        return "BS_ROFORMER"
    if normalized.startswith("mdx23c"):
        return "MDX23C"
    return None


def _normalize_vocal_backend(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if not normalized:
        return "none"
    if normalized in {"off", "disabled"}:
        return "none"
    return normalized


def _parse_backend_list(raw: str) -> list[str]:
    if not raw:
        return []
    items: list[str] = []
    for chunk in raw.split(","):
        backend = _normalize_vocal_backend(chunk)
        if backend and backend != "none":
            items.append(backend)
    return items


def _backend_env(name: str, backend: str) -> str:
    suffix = _backend_env_suffix(backend)
    if suffix:
        override = (os.getenv(f"{name}_{suffix}", "") or "").strip()
        if override:
            return override
    return (os.getenv(name, "") or "").strip()


def _resolve_vocal_isolation_model(backend: str) -> str:
    model_name = _backend_env("TRANSCRIBE_VOCAL_ISOLATION_MODEL", backend)
    if model_name:
        return model_name
    normalized = backend.strip().lower().replace("-", "_")
    if normalized.startswith("melband_roformer"):
        return "vocals_mel_band_roformer.ckpt"
    if normalized.startswith("htdemucs_ft"):
        return "htdemucs_ft.yaml"
    if normalized.startswith("bs_roformer"):
        return "bs_roformer"
    return "mdx23c"


def _resolve_vocal_stem_name(backend: str) -> str:
    stem_name = _backend_env("TRANSCRIBE_VOCAL_ISOLATION_TARGET_STEM", backend)
    return stem_name or "vocals"


def _is_distinct_isolated_source(candidate: str, original: str) -> bool:
    candidate_path = Path(candidate)
    original_path = Path(original)
    try:
        return candidate_path.resolve(strict=False) != original_path.resolve(
            strict=False
        )
    except OSError:
        return candidate != original


def _auto_vocal_isolation_backends() -> list[str]:
    candidates: list[str] = []

    # Prefer explicit task-specific command/API overrides before generic ones.
    # Order by quality: melband_roformer > htdemucs_ft > bs_roformer > mdx23c
    command_candidates = (
        ("melband_roformer", "TRANSCRIBE_VOCAL_ISOLATION_COMMAND_MELBAND_ROFORMER"),
        ("htdemucs_ft", "TRANSCRIBE_VOCAL_ISOLATION_COMMAND_HTDEMUCS_FT"),
        ("bs_roformer", "TRANSCRIBE_VOCAL_ISOLATION_COMMAND_BS_ROFORMER"),
        ("mdx23c", "TRANSCRIBE_VOCAL_ISOLATION_COMMAND_MDX23C"),
        ("command", "TRANSCRIBE_VOCAL_ISOLATION_COMMAND"),
    )
    for backend, env_name in command_candidates:
        if (os.getenv(env_name, "") or "").strip():
            candidates.append(backend)

    api_candidates = (
        ("melband_roformer_api", "TRANSCRIBE_VOCAL_ISOLATION_API_URL_MELBAND_ROFORMER"),
        ("htdemucs_ft_api", "TRANSCRIBE_VOCAL_ISOLATION_API_URL_HTDEMUCS_FT"),
        ("bs_roformer_api", "TRANSCRIBE_VOCAL_ISOLATION_API_URL_BS_ROFORMER"),
        ("mdx23c_api", "TRANSCRIBE_VOCAL_ISOLATION_API_URL_MDX23C"),
        ("api", "TRANSCRIBE_VOCAL_ISOLATION_API_URL"),
    )
    for backend, env_name in api_candidates:
        if (os.getenv(env_name, "") or "").strip():
            candidates.append(backend)
    return candidates


def _find_isolated_stem(root: Path, stem_name: str) -> Path | None:
    candidates: list[Path] = []
    for ext in ("wav", "flac", "mp3", "m4a", "ogg"):
        candidates.extend(root.rglob(f"{stem_name}.{ext}"))
    if not candidates:
        return None

    scored: list[tuple[int, float, Path]] = []
    for candidate in candidates:
        try:
            stat = candidate.stat()
        except OSError:
            continue
        if stat.st_size <= 0:
            continue
        scored.append((int(stat.st_size), float(stat.st_mtime), candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def _render_command_template_tokens(
    template: str, mapping: dict[str, str]
) -> list[str] | None:
    try:
        raw_tokens = shlex.split(template)
    except ValueError as exc:
        _perf_logger.warning("[transcribe] invalid vocal isolation command template: %s", exc)
        return None

    rendered: list[str] = []
    for token in raw_tokens:
        try:
            rendered.append(token.format(**mapping))
        except KeyError as exc:
            _perf_logger.warning("[transcribe] vocal isolation command has unknown placeholder: %s", exc)
            return None
    return rendered


def _resolve_command_output_hint(
    output_hint_template: str,
    *,
    output_dir: Path,
    mapping: dict[str, str],
) -> Path | None:
    if not output_hint_template:
        return None
    try:
        rendered = output_hint_template.format(**mapping)
    except KeyError as exc:
        _perf_logger.warning("[transcribe] invalid command output template placeholder: %s", exc)
        return None
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = output_dir / candidate
    return candidate


def _prepare_vocal_stem_with_command(
    path: str, *, backend: str
) -> tuple[str, Path | None]:
    source_path = Path(path).expanduser().resolve(strict=False)
    if not source_path.exists():
        return path, None

    command_template = _backend_env("TRANSCRIBE_VOCAL_ISOLATION_COMMAND", backend)
    if not command_template:
        _perf_logger.warning("[transcribe] vocal isolation command backend selected but no command configured")
        return path, None

    model_name = _resolve_vocal_isolation_model(backend)
    stem_name = _resolve_vocal_stem_name(backend)
    device = _resolve_vocal_isolation_device()
    timeout_sec = _env_int("TRANSCRIBE_VOCAL_ISOLATION_TIMEOUT_SEC", 1200, 30)

    tmp_dir = Path(os.getenv("TMP_DIR", settings.tmp_dir)).expanduser()
    if not tmp_dir.is_absolute():
        tmp_dir = (Path.cwd() / tmp_dir).resolve(strict=False)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    work_dir = (tmp_dir / f"vocal-command-{uuid4()}").resolve(strict=False)
    output_dir = work_dir / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract audio to WAV first if source is video (audio-separator can't read MP4)
    input_for_separator = source_path
    audio_formats = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
    if source_path.suffix.lower() not in audio_formats:
        extracted_audio = work_dir / "input_audio.wav"
        extract_cmd = [
            settings.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-acodec",
            "pcm_s16le",
            str(extracted_audio),
        ]
        try:
            extract_process = subprocess.run(
                extract_cmd, capture_output=True, text=True, check=False, timeout=120
            )
            if extract_process.returncode == 0 and extracted_audio.exists():
                input_for_separator = extracted_audio
            else:
                _perf_logger.warning("[transcribe] audio extraction for vocal isolation failed, trying with original")
        except (OSError, subprocess.TimeoutExpired) as exc:
            _perf_logger.warning("[transcribe] audio extraction timed out or failed: %s", exc)

    output_path_hint = output_dir / f"{stem_name}.wav"
    mapping = {
        "input": str(input_for_separator.resolve(strict=False)),
        "output_dir": str(output_dir),
        "work_dir": str(work_dir),
        "output_path": str(output_path_hint),
        "stem": stem_name,
        "model": model_name,
        "device": device,
    }
    cmd = _render_command_template_tokens(command_template, mapping)
    if not cmd:
        _cleanup_temp_path(work_dir)
        return path, None
    cmd = _apply_vocal_isolation_priority(cmd)

    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
            cwd=str(work_dir),
        )
    except (OSError, subprocess.TimeoutExpired):
        _cleanup_temp_path(work_dir)
        return path, None

    if process.returncode != 0:
        stderr_tail = (process.stderr or "").strip()[-240:]
        _perf_logger.warning(
            "[transcribe] command vocal isolation failed (rc=%d): %s",
            process.returncode, stderr_tail,
        )
        _cleanup_temp_path(work_dir)
        return path, None

    output_hint_template = _backend_env(
        "TRANSCRIBE_VOCAL_ISOLATION_COMMAND_OUTPUT", backend
    )
    hinted_path = _resolve_command_output_hint(
        output_hint_template, output_dir=output_dir, mapping=mapping
    )
    if (
        hinted_path is not None
        and hinted_path.exists()
        and hinted_path.stat().st_size > 0
    ):
        return str(hinted_path), work_dir

    stem_path = _find_isolated_stem(output_dir, stem_name) or _find_isolated_stem(
        work_dir, stem_name
    )
    if stem_path is None:
        _perf_logger.warning("[transcribe] command vocal isolation produced no usable stem")
        _cleanup_temp_path(work_dir)
        return path, None
    return str(stem_path), work_dir


def _guess_audio_extension(content_type: str | None, default_ext: str = "wav") -> str:
    if not content_type:
        return default_ext
    normalized = content_type.lower().split(";", 1)[0].strip()
    mapping = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/flac": "flac",
        "audio/x-flac": "flac",
        "audio/ogg": "ogg",
        "audio/webm": "webm",
        "audio/mp4": "m4a",
        "application/octet-stream": default_ext,
    }
    return mapping.get(normalized, default_ext)


def _write_vocal_stem_bytes(
    data: bytes, *, work_dir: Path, stem_name: str, ext: str
) -> Path | None:
    if not data:
        return None
    safe_ext = re.sub(r"[^a-z0-9]+", "", ext.lower()) or "wav"
    output_path = work_dir / f"{stem_name}.{safe_ext}"
    try:
        output_path.write_bytes(data)
    except OSError:
        return None
    if output_path.stat().st_size <= 0:
        output_path.unlink(missing_ok=True)
        return None
    return output_path


def _parse_extra_fields(raw: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not raw:
        return parsed
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            parsed[key] = value
    return parsed


def _prepare_vocal_stem_with_api(path: str, *, backend: str) -> tuple[str, Path | None]:
    source_path = Path(path)
    if not source_path.exists():
        return path, None

    api_url = _backend_env("TRANSCRIBE_VOCAL_ISOLATION_API_URL", backend)
    if _is_placeholder_config_value(api_url):
        _perf_logger.warning("[transcribe] vocal isolation API backend selected but API URL is missing/placeholder")
        return path, None
    if not re.match(r"^https?://", api_url, flags=re.IGNORECASE):
        _perf_logger.warning("[transcribe] vocal isolation API URL must start with http:// or https://")
        return path, None

    try:
        import httpx  # type: ignore[import-not-found]
    except Exception:
        _perf_logger.warning("[transcribe] vocal isolation API backend unavailable: httpx missing")
        return path, None

    stem_name = _resolve_vocal_stem_name(backend)
    model_name = _backend_env(
        "TRANSCRIBE_VOCAL_ISOLATION_API_MODEL", backend
    ) or _resolve_vocal_isolation_model(backend)
    device = _resolve_vocal_isolation_device()
    timeout_sec = _env_float("TRANSCRIBE_VOCAL_ISOLATION_API_TIMEOUT_SEC", 300.0, 5.0)
    output_ext_default = (
        (os.getenv("TRANSCRIBE_VOCAL_ISOLATION_API_OUTPUT_EXT", "wav") or "wav")
        .strip()
        .lstrip(".")
    )
    output_ext_default = output_ext_default or "wav"

    file_field = (
        os.getenv("TRANSCRIBE_VOCAL_ISOLATION_API_FILE_FIELD", "file") or "file"
    ).strip() or "file"
    model_field = (
        os.getenv("TRANSCRIBE_VOCAL_ISOLATION_API_MODEL_FIELD", "model") or "model"
    ).strip()
    stem_field = (
        os.getenv("TRANSCRIBE_VOCAL_ISOLATION_API_STEM_FIELD", "stem") or "stem"
    ).strip()
    device_field = (
        os.getenv("TRANSCRIBE_VOCAL_ISOLATION_API_DEVICE_FIELD", "device") or "device"
    ).strip()
    input_mime = (
        os.getenv("TRANSCRIBE_VOCAL_ISOLATION_API_INPUT_MIME", "").strip()
        or "application/octet-stream"
    )

    headers: dict[str, str] = {}
    api_key = _backend_env("TRANSCRIBE_VOCAL_ISOLATION_API_KEY", backend)
    if _is_placeholder_config_value(api_key):
        api_key = ""
    if api_key:
        key_header = (
            os.getenv("TRANSCRIBE_VOCAL_ISOLATION_API_KEY_HEADER", "Authorization")
            or "Authorization"
        ).strip() or "Authorization"
        if key_header.lower() == "authorization" and not api_key.lower().startswith(
            "bearer "
        ):
            headers[key_header] = f"Bearer {api_key}"
        else:
            headers[key_header] = api_key

    form_fields: dict[str, str] = {}
    if model_field and model_name:
        form_fields[model_field] = model_name
    if stem_field:
        form_fields[stem_field] = stem_name
    if device_field:
        form_fields[device_field] = device
    form_fields.update(
        _parse_extra_fields(
            _backend_env("TRANSCRIBE_VOCAL_ISOLATION_API_EXTRA_FIELDS", backend)
        )
    )

    tmp_dir = Path(os.getenv("TMP_DIR", settings.tmp_dir))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    work_dir = tmp_dir / f"vocal-api-{uuid4()}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(source_path, "rb") as audio_file:
            files = {file_field: (source_path.name, audio_file, input_mime)}
            with httpx.Client(timeout=timeout_sec, follow_redirects=True) as client:
                response = client.post(
                    api_url,
                    data=form_fields or None,
                    files=files,
                    headers=headers or None,
                )
                response.raise_for_status()
                content_type = str(response.headers.get("content-type", "") or "")
                if response.content and (
                    content_type.lower().startswith("audio/")
                    or content_type.lower().startswith("application/octet-stream")
                ):
                    stem_path = _write_vocal_stem_bytes(
                        response.content,
                        work_dir=work_dir,
                        stem_name=stem_name,
                        ext=_guess_audio_extension(content_type, output_ext_default),
                    )
                    if stem_path is not None:
                        return str(stem_path), work_dir

                payload: object | None = None
                try:
                    payload = response.json()
                except ValueError:
                    payload = None

                if isinstance(payload, dict):
                    path_fields = [
                        (
                            os.getenv("TRANSCRIBE_VOCAL_ISOLATION_API_PATH_FIELD", "")
                            or ""
                        ).strip(),
                        "path",
                        "file_path",
                        "output_path",
                        "vocals_path",
                    ]
                    for field in path_fields:
                        if not field:
                            continue
                        value = payload.get(field)
                        if not isinstance(value, str) or not value.strip():
                            continue
                        candidate = Path(value.strip())
                        if not candidate.exists() or not candidate.is_file():
                            continue
                        ext = candidate.suffix.lstrip(".") or output_ext_default
                        try:
                            stem_data = candidate.read_bytes()
                        except OSError:
                            continue
                        stem_path = _write_vocal_stem_bytes(
                            stem_data,
                            work_dir=work_dir,
                            stem_name=stem_name,
                            ext=ext,
                        )
                        if stem_path is not None:
                            return str(stem_path), work_dir

                    base64_fields = [
                        (
                            os.getenv("TRANSCRIBE_VOCAL_ISOLATION_API_BASE64_FIELD", "")
                            or ""
                        ).strip(),
                        "audio_base64",
                        "vocals_base64",
                        "base64",
                        "data",
                    ]
                    for field in base64_fields:
                        if not field:
                            continue
                        value = payload.get(field)
                        if not isinstance(value, str) or not value.strip():
                            continue
                        encoded = value.strip()
                        if "base64," in encoded:
                            encoded = encoded.split("base64,", 1)[1]
                        try:
                            blob = base64.b64decode(encoded, validate=False)
                        except Exception:  # noqa: BLE001
                            continue
                        stem_path = _write_vocal_stem_bytes(
                            blob,
                            work_dir=work_dir,
                            stem_name=stem_name,
                            ext=output_ext_default,
                        )
                        if stem_path is not None:
                            return str(stem_path), work_dir

                    url_fields = [
                        (
                            os.getenv("TRANSCRIBE_VOCAL_ISOLATION_API_URL_FIELD", "")
                            or ""
                        ).strip(),
                        "audio_url",
                        "vocals_url",
                        "file_url",
                        "url",
                    ]
                    forward_auth = _env_bool(
                        "TRANSCRIBE_VOCAL_ISOLATION_API_FORWARD_AUTH_ON_DOWNLOAD", False
                    )
                    download_headers = headers if forward_auth else None
                    for field in url_fields:
                        if not field:
                            continue
                        value = payload.get(field)
                        if not isinstance(value, str) or not value.strip():
                            continue
                        try:
                            download_response = client.get(
                                value.strip(), headers=download_headers
                            )
                            download_response.raise_for_status()
                        except Exception:  # noqa: BLE001
                            continue
                        stem_path = _write_vocal_stem_bytes(
                            download_response.content,
                            work_dir=work_dir,
                            stem_name=stem_name,
                            ext=_guess_audio_extension(
                                str(
                                    download_response.headers.get("content-type", "")
                                    or ""
                                ),
                                output_ext_default,
                            ),
                        )
                        if stem_path is not None:
                            return str(stem_path), work_dir
    except Exception as exc:  # noqa: BLE001
        _perf_logger.warning("[transcribe] vocal isolation API call failed: %s", exc)
        _cleanup_temp_path(work_dir)
        return path, None

    _cleanup_temp_path(work_dir)
    _perf_logger.warning("[transcribe] vocal isolation API produced no usable stem")
    return path, None


def _prepare_with_vocal_backend(path: str, backend: str) -> tuple[str, Path | None]:
    # Command-based backends (local processing)
    command_backends = {
        "command",
        "bs_roformer",
        "mdx23c",
        "melband_roformer",
        "htdemucs_ft",
    }
    if backend in command_backends:
        return _prepare_vocal_stem_with_command(path, backend=backend)

    # API-based backends (remote processing)
    api_backends = {
        "api",
        "bs_roformer_api",
        "mdx23c_api",
        "melband_roformer_api",
        "htdemucs_ft_api",
    }
    if backend in api_backends:
        return _prepare_vocal_stem_with_api(path, backend=backend)

    _perf_logger.warning("[transcribe] unsupported vocal isolation backend %r; skipping", backend)
    return path, None


def _prepare_vocal_isolation_source(path: str) -> tuple[str, Path | None]:
    if not _env_bool("TRANSCRIBE_VOCAL_ISOLATION_ENABLED", True):
        return path, None

    duet_mode = bool(getattr(_TRANSCRIPTION_RUNTIME, "duet_mode", False))
    # Check for pre-computed vocal stem first (from upload-time processing)
    source_path = Path(path)
    if source_path.exists():
        source_dir = source_path.parent
        stem_filename = f"{source_path.stem}_vocals.wav"
        precomputed_path = source_dir / stem_filename
        if precomputed_path.exists() and precomputed_path.stat().st_size > 0:
            _perf_logger.info("[transcribe] using pre-computed vocal stem: %s", precomputed_path)
            return str(precomputed_path), None

    # Check if high-quality mode is enabled - uses the best available model
    high_quality_enabled = _env_bool("TRANSCRIBE_VOCAL_ISOLATION_HIGH_QUALITY", False) or duet_mode
    if high_quality_enabled:
        hq_backend = (
            os.getenv("TRANSCRIBE_VOCAL_ISOLATION_HIGH_QUALITY_BACKEND", "")
            or "melband_roformer"
        ).strip()
        requested_backend = _normalize_vocal_backend(hq_backend)
        _perf_logger.info("[transcribe] using high-quality vocal isolation: %s", requested_backend)
    else:
        requested_backend = _normalize_vocal_backend(
            os.getenv("TRANSCRIBE_VOCAL_ISOLATION_BACKEND", "auto") or "auto"
        )
    if requested_backend == "none":
        return path, None

    candidates: list[str] = []
    if requested_backend == "auto":
        candidates.extend(_auto_vocal_isolation_backends())
    else:
        candidates.append(requested_backend)

    fallback_candidates = _parse_backend_list(
        os.getenv("TRANSCRIBE_VOCAL_ISOLATION_FALLBACKS", "")
    )
    candidates.extend(fallback_candidates)
    if not candidates:
        _perf_logger.warning("[transcribe] no vocal isolation backends available; using original audio")
        return path, None

    seen: set[str] = set()
    for backend in candidates:
        normalized_backend = _normalize_vocal_backend(backend)
        if normalized_backend in {"", "none"} or normalized_backend in seen:
            continue
        seen.add(normalized_backend)
        prepared_source, cleanup_path = _prepare_with_vocal_backend(
            path, normalized_backend
        )
        if (
            _is_distinct_isolated_source(prepared_source, path)
            and Path(prepared_source).exists()
        ):
            return prepared_source, cleanup_path
        _cleanup_temp_path(cleanup_path)

    _perf_logger.warning("[transcribe] vocal isolation produced no valid stem; using original audio")
    return path, None


def _extract_audio_for_cloud(
    path: str, *, use_vocal_isolation: bool = True
) -> tuple[str, Path | None]:
    """Prepare cloud ASR input audio.

    When enabled, this first isolates a vocal stem (command/api) and
    then exports a compact mono 16kHz MP3 suitable for Groq upload (< 25MB).
    """
    source_path = Path(path)
    if not source_path.exists():
        return path, None

    # Rescue windows are already mono 16k MP3 chunks; avoid re-encoding/re-separating.
    if source_path.suffix.lower() == ".mp3" and source_path.name.startswith(
        "groq-window-"
    ):
        return str(source_path), None

    prepared_cleanup: Path | None = None
    prepared_path = source_path
    if use_vocal_isolation:
        prepared_source, prepared_cleanup = _prepare_vocal_isolation_source(path)
        prepared_path = Path(prepared_source)
        if not prepared_path.exists():
            prepared_path = source_path
            _cleanup_temp_path(prepared_cleanup)
            prepared_cleanup = None

    tmp_dir = Path(os.getenv("TMP_DIR", settings.tmp_dir))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path = tmp_dir / f"groq-audio-{uuid4()}.mp3"

    cmd = [
        settings.ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(prepared_path),
        "-vn",  # no video
        "-ac",
        "1",  # mono
        "-ar",
        "16000",  # 16kHz is Whisper's native rate
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "64k",  # 64kbps mono = ~0.5MB/min
        str(output_path),
    ]

    try:
        process = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        output_path.unlink(missing_ok=True)
        _cleanup_temp_path(prepared_cleanup)
        return path, None

    _cleanup_temp_path(prepared_cleanup)
    if process.returncode != 0 or not output_path.exists():
        output_path.unlink(missing_ok=True)
        return path, None
    if output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        return path, None
    return str(output_path), output_path


def _call_sarvam_rest(
    audio_path: str,
    duration_sec: float,
    *,
    model_name: str,
    mode: str,
    language_hint: str | None,
    prompt: str | None,
    timeout_sec: float,
) -> TranscriptPayload | None:
    api_key = (
        os.getenv("SARVAM_API_KEY", "")
        or os.getenv("TRANSCRIBE_SARVAM_API_KEY", "")
        or ""
    ).strip()
    if not api_key:
        return None
    api_url = (
        os.getenv("TRANSCRIBE_SARVAM_API_URL", "")
        or os.getenv("SARVAM_SPEECH_TO_TEXT_API_URL", "")
        or "https://api.sarvam.ai/speech-to-text"
    ).strip()
    if _is_placeholder_config_value(api_url):
        return None
    if not re.match(r"^https?://", api_url, flags=re.IGNORECASE):
        return None

    try:
        import httpx  # type: ignore[import-not-found]
    except Exception:
        return None

    headers = {
        "api-subscription-key": api_key,
    }
    language_code = _sarvam_language_code(language_hint)
    with_timestamps = _env_bool("TRANSCRIBE_SARVAM_WITH_TIMESTAMPS", True)
    form_data: dict[str, str] = {
        "model": model_name,
        "language_code": language_code,
        "with_timestamps": "true" if with_timestamps else "false",
        "with_diarization": "false",
        "num_speakers": "1",
        "mode": mode,
    }
    if prompt:
        form_data["prompt"] = prompt

    source = Path(audio_path)
    mime = (
        "audio/mpeg" if source.suffix.lower() == ".mp3" else "application/octet-stream"
    )
    try:
        with open(source, "rb") as audio_file:
            files = {"file": (source.name, audio_file, mime)}
            with httpx.Client(timeout=timeout_sec, follow_redirects=True) as client:
                response = client.post(
                    api_url, headers=headers, data=form_data, files=files
                )
                response.raise_for_status()
                payload = response.json()
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    transcript_text = _clean_word(
        str(payload.get("transcript") or payload.get("text") or "")
    )
    words = _extract_sarvam_word_timestamps(payload, duration_sec)
    if _should_fallback_to_text_words(words, transcript_text):
        words = _fallback_words_from_text(transcript_text, duration_sec)
    # Sarvam timestamps are already accurate (no Whisper drift),
    # so we must NOT apply the global TRANSCRIBE_TIMESTAMP_OFFSET_SEC offset.
    words = _apply_word_filters(
        _normalize_words(words, duration_sec, apply_offset=False),
        duration_sec,
        audio_path=audio_path,
    )
    if _should_fallback_to_text_words(words, transcript_text):
        words = _fallback_words_from_text(transcript_text, duration_sec)
        words = _apply_word_filters(words, duration_sec, audio_path=audio_path)
    if not words:
        return None
    if not transcript_text:
        transcript_text = " ".join(word.text for word in words)

    detected_language = _normalize_detected_language(payload.get("language_code"))
    return TranscriptPayload(
        source="sarvam",
        language=detected_language or _normalize_language_code(language_hint),
        text=transcript_text.strip(),
        words=words,
        is_mock=False,
    )


def _build_from_sarvam(
    path: str,
    duration_sec: float,
    *,
    model_name: str,
    mode: str,
    language_hint: str | None,
    prompt: str | None,
    use_vocal_isolation: bool,
) -> TranscriptPayload | None:
    prepared_path, cleanup_path = _extract_audio_for_cloud(
        path, use_vocal_isolation=use_vocal_isolation
    )
    timeout_sec = _env_float("TRANSCRIBE_SARVAM_TIMEOUT_SEC", 120.0, 5.0)
    max_window_sec = _env_float("TRANSCRIBE_SARVAM_MAX_WINDOW_SEC", 25.0, 5.0)
    overlap_sec = _env_float("TRANSCRIBE_SARVAM_WINDOW_OVERLAP_SEC", 0.25, 0.0)
    try:
        if duration_sec <= max_window_sec + 0.05:
            return _call_sarvam_rest(
                prepared_path,
                duration_sec,
                model_name=model_name,
                mode=mode,
                language_hint=language_hint,
                prompt=prompt,
                timeout_sec=timeout_sec,
            )

        step_sec = max(1.0, max_window_sec - overlap_sec)
        cursor = 0.0
        merged_words: list[TranscriptWordPayload] = []
        detected_language: str | None = None
        # Track the last absolute time we have accepted so words in the
        # overlap zone of consecutive chunks are not duplicated.
        last_accepted_end_sec = 0.0
        while cursor < duration_sec:
            window_end = min(duration_sec, cursor + max_window_sec)
            window_path, window_cleanup = _extract_audio_window_for_cloud(
                prepared_path, cursor, window_end
            )
            if not window_path:
                cursor += step_sec
                continue
            try:
                window_payload = _call_sarvam_rest(
                    window_path,
                    max(window_end - cursor, 0.1),
                    model_name=model_name,
                    mode=mode,
                    language_hint=language_hint,
                    prompt=prompt,
                    timeout_sec=timeout_sec,
                )
            finally:
                if window_cleanup is not None:
                    window_cleanup.unlink(missing_ok=True)
            if window_payload is not None:
                if detected_language is None:
                    detected_language = window_payload.language
                # The safe zone for this chunk starts after the overlap region.
                # Words whose absolute start is before `last_accepted_end_sec`
                # were already produced by the previous chunk — skip them.
                safe_start_abs = max(cursor, last_accepted_end_sec)
                for word in window_payload.words:
                    abs_start = float(word.start_sec) + cursor
                    abs_end = float(word.end_sec) + cursor
                    if abs_start < safe_start_abs - 0.05:
                        # Word falls entirely inside the overlap zone already
                        # covered by the previous chunk — discard to avoid dupe.
                        continue
                    merged_words.append(
                        TranscriptWordPayload(
                            id=str(uuid4()),
                            text=word.text,
                            start_sec=abs_start,
                            end_sec=abs_end,
                            confidence=word.confidence,
                        )
                    )
                    last_accepted_end_sec = max(last_accepted_end_sec, abs_end)
            cursor += step_sec

        if not merged_words:
            return None
        # Sarvam timestamps are already accurate — do NOT apply the Whisper-specific offset.
        normalized_words = _apply_word_filters(
            _normalize_words(merged_words, duration_sec, apply_offset=False),
            duration_sec,
            audio_path=prepared_path,
        )
        if not normalized_words:
            return None
        return TranscriptPayload(
            source="sarvam",
            language=detected_language or _normalize_language_code(language_hint),
            text=" ".join(word.text for word in normalized_words),
            words=normalized_words,
            is_mock=False,
        )
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)


def _call_sarvam(
    path: str,
    duration_sec: float,
    *,
    model_name: str,
    mode: str,
    language_hint: str | None,
    prompt: str | None,
    use_vocal_isolation: bool,
) -> TranscriptPayload | None:
    if language_hint:
        try:
            return _build_from_sarvam(
                path,
                duration_sec,
                model_name=model_name,
                mode=mode,
                language_hint=language_hint,
                prompt=prompt,
                use_vocal_isolation=use_vocal_isolation,
            )
        except TypeError as exc:
            if "language_hint" not in str(exc):
                raise
    return _build_from_sarvam(
        path,
        duration_sec,
        model_name=model_name,
        mode=mode,
        language_hint=None,
        prompt=prompt,
        use_vocal_isolation=use_vocal_isolation,
    )


def _build_from_groq(
    path: str,
    duration_sec: float,
    *,
    model_name: str = "whisper-large-v3-turbo",
    prompt: str | None = None,
    language_hint: str | None = None,
    translate_to_english: bool | None = None,
) -> TranscriptPayload | None:
    """Transcribe via Groq's cloud API. Returns None on failure.

    If translate_to_english is True, uses Groq's translation endpoint to
    translate any language to English. If None, checks TRANSCRIBE_TRANSLATE_TO_ENGLISH env var.
    """
    api_key = (os.getenv("GROQ_API_KEY", "") or "").strip()
    if not api_key:
        return None

    try:
        from groq import Groq
    except ImportError:
        return None

    # Determine if translation mode is enabled
    if translate_to_english is None:
        translate_to_english = _env_bool("TRANSCRIBE_TRANSLATE_TO_ENGLISH", False)

    # Fast lightweight extraction (no heavy filters — Groq handles normalization)
    _t_audio_prep_start = _time_module.monotonic()
    source_path, cleanup_path = _resolve_groq_input_source(path)
    _perf_logger.info(
        "⏱️ INNER TIMING: Audio prep (vocal iso + ffmpeg) took %.1fs for %s",
        _time_module.monotonic() - _t_audio_prep_start,
        path,
    )
    try:
        client = Groq(api_key=api_key)

        language = (
            language_hint or _resolve_transcribe_language() or ""
        ).strip() or None
        resolved_prompt = (
            prompt.strip()
            if prompt is not None
            else (
                (os.getenv("TRANSCRIBE_GROQ_PROMPT", "") or "").strip()
                or (os.getenv("TRANSCRIBE_INITIAL_PROMPT", "") or "").strip()
            )
        ) or None
        request_kwargs: dict[str, object] = {
            "model": model_name,
            "response_format": "verbose_json",
            "timestamp_granularities": ["word", "segment"],
        }
        if language:
            request_kwargs["language"] = language
        if resolved_prompt:
            request_kwargs["prompt"] = resolved_prompt
        with open(source_path, "rb") as audio_file:
            request_payload = {
                **request_kwargs,
                "file": (Path(source_path).name, audio_file),
            }
            try:
                if translate_to_english:
                    # Use translation endpoint - translates any language to English
                    # Note: translations endpoint doesn't support language parameter
                    request_payload.pop("language", None)
                    response = client.audio.translations.create(**request_payload)
                else:
                    response = client.audio.transcriptions.create(**request_payload)
            except TypeError:
                # Some SDK versions may not expose `prompt` yet.
                request_payload.pop("prompt", None)
                if translate_to_english:
                    response = client.audio.translations.create(**request_payload)
                else:
                    response = client.audio.transcriptions.create(**request_payload)

        # Parse words from response.
        # Groq SDK may return `words` as dict entries instead of typed objects.
        words: list[TranscriptWordPayload] = []
        raw_words = getattr(response, "words", None) or []
        for item in raw_words:
            if isinstance(item, dict):
                word_text = str(item.get("word") or item.get("text") or "")
                start = float(item.get("start", 0.0) or 0.0)
                end = float(item.get("end", 0.0) or 0.0)
            else:
                word_text = getattr(item, "word", "") or getattr(item, "text", "") or ""
                start = float(getattr(item, "start", 0.0) or 0.0)
                end = float(getattr(item, "end", 0.0) or 0.0)
            if not word_text.strip():
                continue
            words.append(
                TranscriptWordPayload(
                    id=str(uuid4()),
                    text=word_text.strip(),
                    start_sec=start,
                    end_sec=end,
                    confidence=None,  # Groq doesn't return per-word confidence
                )
            )

        normalized = _normalize_words(words, duration_sec)
        if not normalized:
            # Fallback: some responses omit granular words but include timed segments.
            raw_segments = getattr(response, "segments", None) or []
            recovered_words: list[TranscriptWordPayload] = []
            for segment in raw_segments:
                if isinstance(segment, dict):
                    segment_text = str(segment.get("text") or "")
                    segment_start = float(segment.get("start", 0.0) or 0.0)
                    segment_end = float(
                        segment.get("end", segment_start + 0.2) or (segment_start + 0.2)
                    )
                else:
                    segment_text = str(getattr(segment, "text", "") or "")
                    segment_start = float(getattr(segment, "start", 0.0) or 0.0)
                    segment_end = float(
                        getattr(segment, "end", segment_start + 0.2)
                        or (segment_start + 0.2)
                    )
                parts = segment_text.strip().split()
                if not parts:
                    continue
                span = max(segment_end - segment_start, 0.1)
                step = span / len(parts)
                for idx, token in enumerate(parts):
                    recovered_words.append(
                        TranscriptWordPayload(
                            id=str(uuid4()),
                            text=token,
                            start_sec=segment_start + (idx * step),
                            end_sec=segment_start + ((idx + 1) * step),
                            confidence=None,
                        )
                    )
            normalized = _normalize_words(recovered_words, duration_sec)
        normalized = _apply_word_filters(
            normalized,
            duration_sec,
            audio_path=source_path,
        )
        if not normalized:
            return None

        # Keep transcript text aligned with post-processed word tokens so UI text
        # does not reintroduce repeats that were removed from `normalized`.
        text = " ".join(w.text for w in normalized)
        source = "groq_translated" if translate_to_english else "groq"
        return TranscriptPayload(
            source=source,
            language="en"
            if translate_to_english
            else (getattr(response, "language", None) or language),
            text=text.strip(),
            words=normalized,
            is_mock=False,
        )
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return None
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)


def _call_groq(
    path: str,
    duration_sec: float,
    *,
    model_name: str,
    prompt: str | None,
    language_hint: str | None = None,
    translate_to_english: bool | None = None,
) -> TranscriptPayload | None:
    call_kwargs: dict[str, object] = {
        "model_name": model_name,
        "prompt": prompt,
    }
    if language_hint:
        call_kwargs["language_hint"] = language_hint
    if translate_to_english is not None:
        call_kwargs["translate_to_english"] = translate_to_english
    if language_hint:
        try:
            return _build_from_groq(
                path,
                duration_sec,
                **call_kwargs,
            )
        except TypeError as exc:
            # Backward-compatible shim for tests or patched call-sites that still
            # expose the old signature without language hints or translation flags.
            if "language_hint" not in str(exc) and "translate_to_english" not in str(
                exc
            ):
                raise
    fallback_kwargs: dict[str, object] = {
        "model_name": model_name,
        "prompt": prompt,
    }
    if translate_to_english is not None:
        fallback_kwargs["translate_to_english"] = translate_to_english
    try:
        return _build_from_groq(
            path,
            duration_sec,
            **fallback_kwargs,
        )
    except TypeError as exc:
        if "translate_to_english" not in str(exc):
            raise
    return _build_from_groq(
        path,
        duration_sec,
        model_name=model_name,
        prompt=prompt,
    )


def _extract_audio_window_for_cloud(
    path: str, start_sec: float, end_sec: float
) -> tuple[str | None, Path | None]:
    source_path = Path(path)
    if not source_path.exists():
        return None, None
    clip_start = max(0.0, float(start_sec))
    clip_end = max(clip_start, float(end_sec))
    clip_duration = clip_end - clip_start
    if clip_duration < 0.4:
        return None, None

    tmp_dir = Path(os.getenv("TMP_DIR", settings.tmp_dir))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path = tmp_dir / f"groq-window-{uuid4()}.mp3"
    cmd = [
        settings.ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{clip_start:.3f}",
        "-t",
        f"{clip_duration:.3f}",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "64k",
        str(output_path),
    ]
    try:
        process = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        output_path.unlink(missing_ok=True)
        return None, None
    if (
        process.returncode != 0
        or not output_path.exists()
        or output_path.stat().st_size == 0
    ):
        output_path.unlink(missing_ok=True)
        return None, None
    return str(output_path), output_path


def _rescue_groq_gaps(
    path: str,
    duration_sec: float,
    primary: TranscriptPayload,
    *,
    profile: str,
    model_name: str,
    prompt: str | None,
    language_hint: str | None = None,
) -> TranscriptPayload | None:
    if profile == "music":
        default_min_gap_sec = 6.0
        default_max_chunks = 8
        default_max_window_sec = 12.0
        default_pad_sec = 0.35
    elif profile == "mixed":
        default_min_gap_sec = 8.0
        default_max_chunks = 5
        default_max_window_sec = 20.0
        default_pad_sec = 0.35
    else:
        default_min_gap_sec = 10.0
        default_max_chunks = 3
        default_max_window_sec = 45.0
        default_pad_sec = 0.35

    min_gap_sec = _env_float(
        f"TRANSCRIBE_RESCUE_MIN_GAP_SEC_{profile.upper()}",
        _env_float("TRANSCRIBE_RESCUE_MIN_GAP_SEC", default_min_gap_sec, 2.0),
        2.0,
    )
    max_chunks = _env_int(
        f"TRANSCRIBE_RESCUE_MAX_CHUNKS_{profile.upper()}",
        _env_int("TRANSCRIBE_RESCUE_MAX_CHUNKS", default_max_chunks, 0),
        0,
    )
    max_window_sec = _env_float(
        f"TRANSCRIBE_RESCUE_MAX_WINDOW_SEC_{profile.upper()}",
        _env_float("TRANSCRIBE_RESCUE_MAX_WINDOW_SEC", default_max_window_sec, 5.0),
        5.0,
    )
    pad_sec = _env_float(
        f"TRANSCRIBE_RESCUE_PAD_SEC_{profile.upper()}",
        _env_float("TRANSCRIBE_RESCUE_PAD_SEC", default_pad_sec, 0.0),
        0.0,
    )
    gaps = _find_long_gaps(primary, duration_sec, min_gap_sec=min_gap_sec)
    if not gaps or max_chunks <= 0:
        return None

    ordered_gaps = sorted(
        gaps, key=lambda gap: float(gap[1]) - float(gap[0]), reverse=True
    )
    collected_words: list[TranscriptWordPayload] = []
    used_chunks = 0

    for gap_start, gap_end in ordered_gaps:
        if used_chunks >= max_chunks:
            break
        window_start = max(0.0, float(gap_start) - pad_sec)
        window_end = min(duration_sec, float(gap_end) + pad_sec)
        cursor = window_start
        while cursor < window_end and used_chunks < max_chunks:
            chunk_end = min(window_end, cursor + max_window_sec)
            window_path, cleanup_path = _extract_audio_window_for_cloud(
                path, cursor, chunk_end
            )
            used_chunks += 1
            if not window_path:
                cursor = chunk_end
                continue
            try:
                window_groq_kwargs: dict[str, object] = {
                    "model_name": model_name,
                    "prompt": prompt,
                }
                if language_hint is not None:
                    window_groq_kwargs["language_hint"] = language_hint
                window_payload = _call_groq(
                    window_path,
                    chunk_end - cursor,
                    **window_groq_kwargs,
                )
            finally:
                if cleanup_path is not None:
                    cleanup_path.unlink(missing_ok=True)
            if window_payload is not None:
                for word in window_payload.words:
                    collected_words.append(
                        TranscriptWordPayload(
                            id=str(uuid4()),
                            text=word.text,
                            start_sec=float(word.start_sec) + cursor,
                            end_sec=float(word.end_sec) + cursor,
                            confidence=word.confidence,
                        )
                    )
            cursor = chunk_end

    if _env_bool("TRANSCRIBE_RESCUE_SCRIPT_FILTER", True) and collected_words:
        primary_text = " ".join(word.text for word in primary.words[:600])
        primary_alpha_count = sum(1 for char in primary_text if char.isalpha())
        primary_latin_ratio = _ascii_latin_ratio(primary_text)
        primary_min_alpha = _env_int("TRANSCRIBE_RESCUE_PRIMARY_MIN_ALPHA", 40, 0)
        primary_latin_min = _env_float(
            "TRANSCRIBE_RESCUE_PRIMARY_LATIN_RATIO", 0.65, 0.0
        )
        rescue_token_latin_min = _env_float(
            "TRANSCRIBE_RESCUE_TOKEN_LATIN_MIN_RATIO", 0.35, 0.0
        )
        if (
            primary_alpha_count >= primary_min_alpha
            and primary_latin_ratio >= primary_latin_min
        ):
            filtered_words: list[TranscriptWordPayload] = []
            drop_non_ascii_tokens = _env_bool(
                "TRANSCRIBE_RESCUE_DROP_NON_ASCII_TOKENS", True
            )
            for word in collected_words:
                alpha_count = sum(1 for char in word.text if char.isalpha())
                if alpha_count < 2:
                    filtered_words.append(word)
                    continue
                if drop_non_ascii_tokens and any(
                    char.isalpha() and ord(char) > 127 for char in word.text
                ):
                    continue
                if _ascii_latin_ratio(word.text) >= rescue_token_latin_min:
                    filtered_words.append(word)
            collected_words = filtered_words

    if not collected_words:
        return None
    secondary = TranscriptPayload(
        source="groq_rescue",
        language=primary.language,
        text=" ".join(word.text for word in collected_words),
        words=_normalize_words(collected_words, duration_sec),
        is_mock=False,
    )
    return _merge_gap_fill_transcript(primary, secondary, duration_sec)


def _call_rescue_groq_gaps(
    path: str,
    duration_sec: float,
    primary: TranscriptPayload,
    *,
    profile: str,
    model_name: str,
    prompt: str | None,
    language_hint: str | None = None,
) -> TranscriptPayload | None:
    if language_hint:
        try:
            return _rescue_groq_gaps(
                path,
                duration_sec,
                primary,
                profile=profile,
                model_name=model_name,
                prompt=prompt,
                language_hint=language_hint,
            )
        except TypeError as exc:
            if "language_hint" not in str(exc):
                raise
    return _rescue_groq_gaps(
        path,
        duration_sec,
        primary,
        profile=profile,
        model_name=model_name,
        prompt=prompt,
    )


def generate_transcript(
    path: str,
    duration_sec: float,
    *,
    language_hint: str | None = None,
    allow_mock_fallback: bool | None = None,
    fast_mode: bool | None = None,
    prompt: str | None = None,
    translate_to_english: bool | None = None,
    mode: str | None = None,
    optimize_for_speed: bool | None = None,
    filename: str | None = None,
) -> TranscriptPayload:
    safe_duration = max(float(duration_sec), 0.1)
    allow_mock = (
        _env_bool("TRANSCRIBE_ALLOW_MOCK_FALLBACK", False)
        if allow_mock_fallback is None
        else bool(allow_mock_fallback)
    )
    configured_language = (
        _normalize_language_code(language_hint) or _resolve_transcribe_language()
    )
    transcript_mode = _normalize_transcription_mode(mode)
    speed_optimized = (
        bool(optimize_for_speed) if optimize_for_speed is not None else False
    )
    backend = (os.getenv("TRANSCRIBE_BACKEND", "auto") or "auto").strip().lower()
    fast_mode_enabled = bool(fast_mode) if fast_mode is not None else False
    requested_profile = (
        (os.getenv("TRANSCRIBE_PROFILE", "auto") or "auto").strip().lower()
    )
    if transcript_mode == "song":
        profile = "music"
    elif transcript_mode == "speech":
        profile = "speech"
        if requested_profile in {"", "auto"}:
            detected_profile = _resolve_transcription_profile(path, safe_duration)
            if detected_profile in {"music", "mixed"}:
                profile = detected_profile
    elif fast_mode_enabled and requested_profile in {"", "auto"}:
        detected_profile = _resolve_transcription_profile(path, safe_duration)
        profile = (
            detected_profile
            if detected_profile in {"music", "mixed"}
            else "speech"
        )
    else:
        _t_profile_start = _time_module.monotonic()
        profile = _resolve_transcription_profile(path, safe_duration)
        _perf_logger.info(
            "⏱️ INNER TIMING: Profile analysis took %.1fs → %s",
            _time_module.monotonic() - _t_profile_start,
            profile,
        )
    previous_runtime_profile = getattr(_TRANSCRIPTION_RUNTIME, "profile", None)
    previous_runtime_mode = getattr(_TRANSCRIPTION_RUNTIME, "mode", None)
    previous_runtime_duet_mode = getattr(_TRANSCRIPTION_RUNTIME, "duet_mode", False)
    _TRANSCRIPTION_RUNTIME.profile = profile
    _TRANSCRIPTION_RUNTIME.mode = transcript_mode
    duet_detected = False
    if filename:
        from .lyrics_reference_service import looks_like_duet_media

        duet_detected = looks_like_duet_media(filename)
    _TRANSCRIPTION_RUNTIME.duet_mode = duet_detected
    try:
        groq_primary_model = (
            os.getenv("TRANSCRIBE_GROQ_MODEL", "whisper-large-v3-turbo")
            or "whisper-large-v3-turbo"
        ).strip()
        groq_retry_model = (
            os.getenv("TRANSCRIBE_GROQ_RETRY_MODEL", "whisper-large-v3")
            or "whisper-large-v3"
        ).strip()
        groq_primary_prompt = (
            (prompt.strip() if prompt else None)
            or (os.getenv("TRANSCRIBE_GROQ_PROMPT", "") or "").strip()
            or None
        )
        groq_retry_prompt = (
            (prompt.strip() if prompt else None)
            or (os.getenv("TRANSCRIBE_GROQ_RETRY_PROMPT", "") or "").strip()
            or groq_primary_prompt
        )
        groq_retry_try_no_prompt = _env_bool(
            "TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT", True
        )
        groq_primary_prompt, groq_retry_prompt, groq_retry_try_no_prompt = (
            _resolve_groq_prompt_strategy(
                profile,
                groq_primary_prompt,
                groq_retry_prompt,
                groq_retry_try_no_prompt,
            )
        )
        groq_retry_enabled = (
            _env_bool("TRANSCRIBE_GROQ_ENABLE_RETRY", True)
            and not fast_mode_enabled
            and not speed_optimized
        )
        groq_retry_min_duration_sec = _env_float(
            "TRANSCRIBE_GROQ_RETRY_MIN_DURATION_SEC", 60.0, 0.0
        )
        sarvam_model = (
            os.getenv("TRANSCRIBE_SARVAM_MODEL", "saaras:v3") or "saaras:v3"
        ).strip() or "saaras:v3"
        sarvam_mode = (
            os.getenv("TRANSCRIBE_SARVAM_MODE", "transcribe") or "transcribe"
        ).strip() or "transcribe"
        sarvam_prompt = (
            os.getenv("TRANSCRIBE_SARVAM_PROMPT", "") or ""
        ).strip() or None
        use_vocal_isolation = _vocal_isolation_allowed_for_profile(profile)
        music_auto_routing = _auto_sarvam_music_routing_allowed(
            profile=profile,
            configured_language=configured_language,
        )
        route_to_sarvam = _should_route_to_sarvam(backend, configured_language)
        route_unknown_auto_to_sarvam = (
            backend == "auto"
            and configured_language is None
            and (
                (not fast_mode_enabled and not speed_optimized)
                or music_auto_routing
            )
            and translate_to_english is not True
            and _env_bool("TRANSCRIBE_AUTO_UNKNOWN_TO_SARVAM_FIRST", True)
        )
        if route_unknown_auto_to_sarvam:
            route_to_sarvam = True
        if (
            transcript_mode == "song"
            and configured_language is not None
            and not _is_indic_language(configured_language)
        ):
            route_to_sarvam = False
        elif (
            speed_optimized
            and backend == "auto"
            and not _is_indic_language(configured_language)
            and not music_auto_routing
        ):
            route_to_sarvam = False
        if fast_mode_enabled and backend == "auto" and not music_auto_routing:
            route_to_sarvam = False
        sarvam_allow_groq_fallback = _env_bool(
            "TRANSCRIBE_SARVAM_ALLOW_GROQ_FALLBACK", True
        )
        sarvam_result: TranscriptPayload | None = None

        if route_to_sarvam:
            sarvam_result = _call_sarvam(
                path,
                safe_duration,
                model_name=sarvam_model,
                mode=sarvam_mode,
                language_hint=configured_language,
                prompt=sarvam_prompt,
                use_vocal_isolation=use_vocal_isolation,
            )
            if sarvam_result is not None and not _is_low_coverage(
                sarvam_result, safe_duration
            ):
                if not _should_skip_sarvam_first_return(
                    sarvam_result,
                    configured_language=configured_language,
                    music_auto_routing=music_auto_routing,
                ):
                    return sarvam_result
            if backend == "sarvam" and not sarvam_allow_groq_fallback:
                if sarvam_result is not None:
                    return sarvam_result
                if allow_mock:
                    return _build_mock_transcript(safe_duration)
                raise RuntimeError(
                    "Sarvam transcription failed. Check SARVAM API key and network."
                )

        # ------------------------------------------------------------------
        # Groq cloud backend: fast, no local GPU needed
        # ------------------------------------------------------------------
        if backend in {"groq", "auto"} or (
            backend == "sarvam" and sarvam_allow_groq_fallback
        ):
            _start_groq_audio_session(path, use_vocal_isolation=use_vocal_isolation)
            try:
                _t_groq_primary_start = _time_module.monotonic()
                primary_groq_kwargs: dict[str, object] = {
                    "model_name": groq_primary_model,
                    "prompt": groq_primary_prompt,
                }
                if configured_language is not None:
                    primary_groq_kwargs["language_hint"] = configured_language
                if translate_to_english is not None:
                    primary_groq_kwargs["translate_to_english"] = translate_to_english
                groq_result = _call_groq(
                    path,
                    safe_duration,
                    **primary_groq_kwargs,
                )
                _perf_logger.info(
                    "⏱️ INNER TIMING: Groq primary call took %.1fs (model=%s, vocal_iso=%s)",
                    _time_module.monotonic() - _t_groq_primary_start,
                    groq_primary_model,
                    use_vocal_isolation,
                )
                if groq_result is not None:
                    should_retry_groq = (
                        groq_retry_enabled
                        and safe_duration >= groq_retry_min_duration_sec
                        and _should_retry_groq_for_profile(
                            profile, groq_result, safe_duration
                        )
                    )
                    if (
                        should_retry_groq
                        and groq_retry_model
                        and groq_retry_model != groq_primary_model
                    ):
                        groq_retry_candidates: list[TranscriptPayload] = []
                        retry_groq_kwargs: dict[str, object] = {
                            "model_name": groq_retry_model,
                            "prompt": groq_retry_prompt,
                        }
                        if configured_language is not None:
                            retry_groq_kwargs["language_hint"] = configured_language
                        if translate_to_english is not None:
                            retry_groq_kwargs["translate_to_english"] = (
                                translate_to_english
                            )
                        groq_retry = _call_groq(
                            path,
                            safe_duration,
                            **retry_groq_kwargs,
                        )
                        if groq_retry is not None:
                            groq_retry_candidates.append(groq_retry)
                        if groq_retry_try_no_prompt and groq_retry_prompt:
                            retry_no_prompt_kwargs: dict[str, object] = {
                                "model_name": groq_retry_model,
                                "prompt": None,
                            }
                            if configured_language is not None:
                                retry_no_prompt_kwargs["language_hint"] = (
                                    configured_language
                                )
                            if translate_to_english is not None:
                                retry_no_prompt_kwargs["translate_to_english"] = (
                                    translate_to_english
                                )
                            groq_retry_no_prompt = _call_groq(
                                path,
                                safe_duration,
                                **retry_no_prompt_kwargs,
                            )
                            if groq_retry_no_prompt is not None:
                                groq_retry_candidates.append(groq_retry_no_prompt)
                        groq_retry = _pick_best_gap_fill_candidate(
                            groq_result, groq_retry_candidates, safe_duration
                        )
                        merged = _merge_gap_fill_transcript(
                            groq_result, groq_retry, safe_duration
                        )
                        if merged is not None:
                            default_min_gap_fill_words = (
                                2 if safe_duration <= 30.0 else 3
                            )
                            min_gap_fill_words = _env_int(
                                "TRANSCRIBE_MIN_GAP_FILL_WORDS",
                                default_min_gap_fill_words,
                                1,
                            )
                            if safe_duration <= 30.0:
                                min_gap_fill_words = min(
                                    min_gap_fill_words,
                                    _env_int(
                                        "TRANSCRIBE_MIN_GAP_FILL_WORDS_SHORT", 2, 1
                                    ),
                                )
                            added_words = max(
                                len(merged.words) - len(groq_result.words), 0
                            )
                            if added_words >= min_gap_fill_words:
                                groq_result = merged
                            else:
                                preferred_gap_fill = (
                                    _pick_better_transcript_with_language(
                                        groq_result,
                                        merged,
                                        safe_duration,
                                        configured_language,
                                    )
                                )
                                if preferred_gap_fill is not None:
                                    groq_result = preferred_gap_fill
                        # Only replace the full transcript with retry when primary remains low coverage.
                        if _is_low_coverage(groq_result, safe_duration):
                            preferred_groq = _pick_better_transcript_with_language(
                                groq_result,
                                groq_retry,
                                safe_duration,
                                configured_language,
                            )
                            if preferred_groq is not None:
                                groq_result = preferred_groq
                    low_coverage_now = _is_low_coverage(groq_result, safe_duration)
                    unresolved_gaps = _has_suspicious_long_gap(
                        groq_result, safe_duration
                    ) or (
                        profile != "speech"
                        and _has_sparse_window(groq_result, safe_duration)
                    )
                    rescue_enabled = (
                        _env_bool("TRANSCRIBE_ENABLE_GAP_RESCUE", True)
                        and not fast_mode_enabled
                        and not speed_optimized
                        and transcript_mode != "song"
                    )
                    rescue_on_low_coverage = _env_bool(
                        "TRANSCRIBE_RESCUE_ON_LOW_COVERAGE", True
                    )
                    should_run_rescue = unresolved_gaps or (
                        rescue_on_low_coverage and low_coverage_now
                    )
                    if should_run_rescue and rescue_enabled and profile != "speech":
                        rescue_model = (
                            os.getenv("TRANSCRIBE_GROQ_RESCUE_MODEL", "") or ""
                        ).strip() or groq_retry_model
                        rescue_prompt = (
                            os.getenv("TRANSCRIBE_GROQ_RESCUE_PROMPT", "") or ""
                        ).strip() or groq_retry_prompt
                        rescue = _call_rescue_groq_gaps(
                            path,
                            safe_duration,
                            groq_result,
                            profile=profile,
                            model_name=rescue_model,
                            prompt=rescue_prompt,
                            language_hint=configured_language,
                        )
                        if rescue is not None:
                            min_added_rescue_words = _env_int(
                                "TRANSCRIBE_MIN_RESCUE_ADDED_WORDS", 2, 1
                            )
                            rescue_added = max(
                                len(rescue.words) - len(groq_result.words), 0
                            )
                            if rescue_added >= min_added_rescue_words:
                                groq_result = rescue

                    # When language choice is weak (common in music-heavy clips), try
                    # auto detect + configured language fallbacks and keep the best transcript.
                    if _is_low_coverage(groq_result, safe_duration):
                        language_fallbacks = _parse_language_fallbacks(
                            os.getenv("TRANSCRIBE_GROQ_LANGUAGE_FALLBACKS", "")
                        )
                        max_lang_attempts = (
                            0
                            if fast_mode_enabled or speed_optimized
                            else _env_int(
                                "TRANSCRIBE_GROQ_LANGUAGE_FALLBACK_MAX_ATTEMPTS", 2, 0
                            )
                        )
                        retry_languages = _build_language_retry_candidates(
                            configured_language, language_fallbacks
                        )
                        if retry_languages and max_lang_attempts > 0:
                            fallback_model = groq_retry_model or groq_primary_model
                            attempted = 0
                            best_result = groq_result
                            for language_code in retry_languages:
                                if attempted >= max_lang_attempts:
                                    break
                                attempted += 1
                                fallback_groq_kwargs: dict[str, object] = {
                                    "model_name": fallback_model,
                                    "prompt": None,
                                    "language_hint": language_code,
                                }
                                if translate_to_english is not None:
                                    fallback_groq_kwargs["translate_to_english"] = (
                                        translate_to_english
                                    )
                                fallback_result = _call_groq(
                                    path,
                                    safe_duration,
                                    **fallback_groq_kwargs,
                                )
                                candidate = _pick_better_transcript_with_language(
                                    best_result,
                                    fallback_result,
                                    safe_duration,
                                    configured_language,
                                )
                                if candidate is not None:
                                    best_result = candidate
                                if not _is_low_coverage(best_result, safe_duration):
                                    break
                            groq_result = best_result
                    guard_retry_enabled = (
                        _env_bool("TRANSCRIBE_LANGUAGE_GUARD_RETRY", True)
                        and not fast_mode_enabled
                        and not speed_optimized
                    )
                    if guard_retry_enabled and _needs_language_guard_retry(
                        groq_result, configured_language
                    ):
                        guard_model = groq_retry_model or groq_primary_model
                        guard_prompt = _build_language_guard_prompt(
                            configured_language or ""
                        )
                        guard_groq_kwargs: dict[str, object] = {
                            "model_name": guard_model,
                            "prompt": guard_prompt,
                        }
                        if configured_language is not None:
                            guard_groq_kwargs["language_hint"] = configured_language
                        if translate_to_english is not None:
                            guard_groq_kwargs["translate_to_english"] = (
                                translate_to_english
                            )
                        guard_result = _call_groq(
                            path,
                            safe_duration,
                            **guard_groq_kwargs,
                        )
                        preferred_guard = _pick_better_transcript_with_language(
                            groq_result,
                            guard_result,
                            safe_duration,
                            configured_language,
                        )
                        if preferred_guard is not None:
                            groq_result = preferred_guard
                    if sarvam_result is not None:
                        preferred_provider = _pick_better_transcript_with_language(
                            groq_result,
                            sarvam_result,
                            safe_duration,
                            configured_language,
                        )
                        if preferred_provider is not None:
                            groq_result = preferred_provider
                    if (
                        configured_language is None
                        and music_auto_routing
                        and groq_result is not None
                        and _looks_like_latin_music_lyrics(groq_result)
                        and _normalize_detected_language(groq_result.language) != "en"
                    ):
                        en_groq_kwargs: dict[str, object] = {
                            "model_name": groq_primary_model,
                            "prompt": groq_primary_prompt,
                            "language_hint": "en",
                        }
                        if translate_to_english is not None:
                            en_groq_kwargs["translate_to_english"] = (
                                translate_to_english
                            )
                        en_result = _call_groq(
                            path,
                            safe_duration,
                            **en_groq_kwargs,
                        )
                        preferred_en = _pick_better_transcript_with_language(
                            groq_result,
                            en_result,
                            safe_duration,
                            "en",
                        )
                        if preferred_en is not None:
                            groq_result = preferred_en
                    if (
                        fast_mode_enabled
                        and music_auto_routing
                        and groq_result is not None
                        and _normalize_detected_language(groq_result.language) == "en"
                        and not _looks_like_latin_music_lyrics(groq_result)
                    ):
                        sarvam_guess = _call_sarvam(
                            path,
                            safe_duration,
                            model_name=sarvam_model,
                            mode=sarvam_mode,
                            language_hint=None,
                            prompt=sarvam_prompt,
                            use_vocal_isolation=use_vocal_isolation,
                        )
                        preferred_guess = _pick_better_auto_indic_transcript(
                            groq_result,
                            sarvam_guess,
                            safe_duration,
                        )
                        if preferred_guess is not None:
                            groq_result = preferred_guess
                    auto_sarvam_attempts = _resolve_auto_sarvam_probe_attempts(
                        fast_mode_enabled=fast_mode_enabled,
                        speed_optimized=speed_optimized,
                        profile=profile,
                        configured_language=configured_language,
                    )
                    auto_probe_languages = (
                        _build_auto_sarvam_probe_languages(
                            groq_result,
                            safe_duration,
                            profile,
                            configured_language,
                        )
                        if (
                            backend == "auto"
                            and auto_sarvam_attempts > 0
                            and configured_language is None
                        )
                        else []
                    )
                    if auto_probe_languages:
                        best_result = groq_result
                        attempts = 0
                        min_auto_alpha = _env_int(
                            "TRANSCRIBE_AUTO_ROUTE_SARVAM_MIN_ALPHA", 18, 0
                        )
                        min_auto_ratio = _env_float(
                            "TRANSCRIBE_AUTO_ROUTE_SARVAM_MIN_RATIO", 0.30, 0.0
                        )
                        for language_code in auto_probe_languages:
                            if attempts >= auto_sarvam_attempts:
                                break
                            attempts += 1
                            sarvam_candidate = _call_sarvam(
                                path,
                                safe_duration,
                                model_name=sarvam_model,
                                mode=sarvam_mode,
                                language_hint=language_code,
                                prompt=sarvam_prompt,
                                use_vocal_isolation=use_vocal_isolation,
                            )
                            if language_code is None:
                                candidate = _pick_better_auto_indic_transcript(
                                    best_result,
                                    sarvam_candidate,
                                    safe_duration,
                                )
                            else:
                                candidate = _pick_better_transcript_with_language(
                                    best_result,
                                    sarvam_candidate,
                                    safe_duration,
                                    language_code,
                                )
                            if candidate is not None:
                                best_result = candidate
                            if (
                                candidate is sarvam_candidate
                                and sarvam_candidate is not None
                            ):
                                candidate_language = (
                                    language_code
                                    or _normalize_detected_language(
                                        sarvam_candidate.language
                                    )
                                )
                                alpha_count, script_ratio = (
                                    _payload_language_match_metrics(
                                        sarvam_candidate, candidate_language
                                    )
                                )
                                if (
                                    not _is_low_coverage(
                                        sarvam_candidate, safe_duration
                                    )
                                    and alpha_count >= min_auto_alpha
                                    and script_ratio >= min_auto_ratio
                                ):
                                    break
                        groq_result = best_result
                    if (
                        allow_mock
                        and _env_bool("TRANSCRIBE_MOCK_ON_LOW_COVERAGE", True)
                        and _is_low_coverage(
                            groq_result,
                            safe_duration,
                        )
                    ):
                        return _build_mock_transcript(safe_duration)
                    return groq_result
                if backend == "groq":
                    # Avoid unexpectedly slow local fallback when explicit Groq mode is selected.
                    groq_local_fallback = _env_bool(
                        "TRANSCRIBE_GROQ_LOCAL_FALLBACK", backend != "groq"
                    )
                    if fast_mode_enabled:
                        groq_local_fallback = False
                    if (
                        _env_bool("TRANSCRIBE_GROQ_STRICT_ONLY", False)
                        or not groq_local_fallback
                    ):
                        if allow_mock:
                            return _build_mock_transcript(safe_duration)
                        raise RuntimeError(
                            "Groq transcription failed. Check GROQ_API_KEY and network."
                        )
                if backend == "sarvam":
                    if sarvam_result is not None:
                        return sarvam_result
                    if allow_mock:
                        return _build_mock_transcript(safe_duration)
                    raise RuntimeError(
                        "Transcription failed for Sarvam and Groq fallback."
                    )
            finally:
                _finish_groq_audio_session()
        if sarvam_result is not None:
            if (
                allow_mock
                and _env_bool("TRANSCRIBE_MOCK_ON_LOW_COVERAGE", True)
                and _is_low_coverage(
                    sarvam_result,
                    safe_duration,
                )
            ):
                return _build_mock_transcript(safe_duration)
            return sarvam_result
        if backend == "sarvam":
            if allow_mock:
                return _build_mock_transcript(safe_duration)
            raise RuntimeError(
                "Sarvam transcription failed. Check SARVAM API key and network."
            )

        # Cloud-only mode: no local offline model fallback.
        # Always raise a clear error when cloud transcription fails.
        if allow_mock:
            return _build_mock_transcript(safe_duration)
        raise RuntimeError(
            "Cloud transcription failed (Groq/Sarvam). "
            "Check your GROQ_API_KEY, SARVAM_API_KEY, and network connectivity. "
            "Ensure the API keys are valid and the services are reachable."
        )
    finally:
        _TRANSCRIPTION_RUNTIME.profile = previous_runtime_profile
        _TRANSCRIPTION_RUNTIME.mode = previous_runtime_mode
        _TRANSCRIPTION_RUNTIME.duet_mode = previous_runtime_duet_mode
        # Aggressive cleanup after each transcription to prevent OOM on low-memory systems
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def generate_transcript_with_refinement(
    path: str,
    duration_sec: float,
    *,
    language_hint: str | None = None,
    allow_mock_fallback: bool | None = None,
    fast_mode: bool | None = None,
    prompt: str | None = None,
    refine_timestamps: bool | None = None,
    mode: str | None = None,
    optimize_for_speed: bool | None = None,
) -> TranscriptPayload:
    """
    Generate transcript with optional per-word timestamp refinement.

    This wrapper calls generate_transcript() and then applies energy-based
    timestamp refinement to improve word-level alignment with audio.

    Args:
        path: Path to audio/video file
        duration_sec: Duration in seconds
        language_hint: Optional language code
        allow_mock_fallback: Whether to allow mock transcript on failure
        fast_mode: Fast mode skips retries and refinement
        prompt: Optional transcription prompt
        refine_timestamps: Whether to refine timestamps (default: auto from env)

    Returns:
        TranscriptPayload with optionally refined timestamps
    """
    result = generate_transcript(
        path,
        duration_sec,
        language_hint=language_hint,
        allow_mock_fallback=allow_mock_fallback,
        fast_mode=fast_mode,
        prompt=prompt,
        mode=mode,
        optimize_for_speed=optimize_for_speed,
    )

    # Skip refinement for mock transcripts or fast mode
    if result.is_mock:
        return result

    # Check if refinement is enabled
    refine_enabled = (
        _env_bool("TRANSCRIBE_TIMESTAMP_REFINEMENT_ENABLED", False)
        if refine_timestamps is None
        else bool(refine_timestamps)
    )

    # Fast mode disables refinement
    if fast_mode:
        refine_enabled = False

    if not refine_enabled:
        return result

    try:
        from app.timestamp_refiner import refine_word_timestamps_batch

        refined_words = refine_word_timestamps_batch(
            result.words,
            path,
            max_words=_env_int("TRANSCRIBE_TIMESTAMP_REFINEMENT_MAX_WORDS", 1000, 100),
        )

        if refined_words is not result.words:
            return TranscriptPayload(
                source=result.source,
                language=result.language,
                text=result.text,
                words=refined_words,
                is_mock=result.is_mock,
            )
    except Exception as exc:  # noqa: BLE001
        # Log but don't fail - timestamp refinement is optional
        import logging

        logging.getLogger(__name__).warning(
            f"Timestamp refinement failed, using original timestamps: {exc}"
        )

    return result


# ---------------------------------------------------------------------------
# Pre-computed vocal isolation for upload-time processing
# ---------------------------------------------------------------------------


def precompute_vocal_isolation(
    source_path: str,
    output_dir: str,
    *,
    force_backend: str | None = None,
) -> str | None:
    """Run vocal isolation and store the result for later transcription.

    This function is called at upload time to pre-compute the vocal stem,
    so that transcription can use it directly without waiting.

    Args:
        source_path: Absolute path to the source video/audio file
        output_dir: Directory to store the output vocal stem (typically same as upload dir)
        force_backend: If provided, use this backend instead of config-based selection

    Returns:
        Relative path to the vocal stem file if successful, None otherwise
    """
    from pathlib import Path as P

    source = P(source_path)
    if not source.exists():
        _perf_logger.warning("[precompute] source does not exist: %s", source_path)
        return None

    # Check if vocal isolation is enabled at all
    if not _env_bool("TRANSCRIBE_VOCAL_ISOLATION_ENABLED", True):
        _perf_logger.debug("[precompute] vocal isolation disabled, skipping")
        return None

    # Determine the backend to use (same logic as _prepare_vocal_isolation_source)
    high_quality_enabled = _env_bool("TRANSCRIBE_VOCAL_ISOLATION_HIGH_QUALITY", False)
    if force_backend:
        requested_backend = _normalize_vocal_backend(force_backend)
    elif high_quality_enabled:
        hq_backend = (
            os.getenv("TRANSCRIBE_VOCAL_ISOLATION_HIGH_QUALITY_BACKEND", "")
            or "melband_roformer"
        ).strip()
        requested_backend = _normalize_vocal_backend(hq_backend)
    else:
        requested_backend = _normalize_vocal_backend(
            os.getenv("TRANSCRIBE_VOCAL_ISOLATION_BACKEND", "auto") or "auto"
        )

    if requested_backend == "none":
        _perf_logger.debug("[precompute] backend is 'none', skipping")
        return None

    # Build candidate list
    candidates: list[str] = []
    if requested_backend == "auto":
        candidates.extend(_auto_vocal_isolation_backends())
    else:
        candidates.append(requested_backend)

    fallback_candidates = _parse_backend_list(
        os.getenv("TRANSCRIBE_VOCAL_ISOLATION_FALLBACKS", "")
    )
    candidates.extend(fallback_candidates)

    if not candidates:
        _perf_logger.warning("[precompute] no vocal isolation backends available")
        return None

    # Try each backend until one succeeds
    seen: set[str] = set()
    for backend in candidates:
        normalized_backend = _normalize_vocal_backend(backend)
        if normalized_backend in {"", "none"} or normalized_backend in seen:
            continue
        seen.add(normalized_backend)

        _perf_logger.info("[precompute] trying backend: %s", normalized_backend)

        # Run isolation (this goes to a temp directory first)
        prepared_source, cleanup_path = _prepare_with_vocal_backend(
            source_path, normalized_backend
        )

        if (
            _is_distinct_isolated_source(prepared_source, source_path)
            and P(prepared_source).exists()
        ):
            # Success! Move to permanent location
            out_dir = P(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Name the vocal stem based on source file
            stem_filename = f"{source.stem}_vocals.wav"
            final_path = out_dir / stem_filename

            try:
                # Copy to permanent location
                shutil.copy2(prepared_source, final_path)
                _perf_logger.info("[precompute] saved vocal stem: %s", final_path)

                # Clean up temp files
                _cleanup_temp_path(cleanup_path)

                # Return just the filename (relative to output_dir)
                return stem_filename

            except (OSError, IOError) as exc:
                _perf_logger.warning("[precompute] failed to save vocal stem: %s", exc)
                _cleanup_temp_path(cleanup_path)
                continue

        _cleanup_temp_path(cleanup_path)

    _perf_logger.warning("[precompute] all backends failed")
    return None


def get_precomputed_vocal_path(
    source_path: str,
    upload_dir: str,
) -> str | None:
    """Check if a pre-computed vocal stem exists for the given source file.

    Args:
        source_path: Absolute path to the source video/audio file
        upload_dir: Directory where vocal stems are stored

    Returns:
        Absolute path to the vocal stem if it exists, None otherwise
    """
    from pathlib import Path as P

    source = P(source_path)
    stem_filename = f"{source.stem}_vocals.wav"
    vocal_path = P(upload_dir) / stem_filename

    if vocal_path.exists() and vocal_path.stat().st_size > 0:
        return str(vocal_path)

    return None
