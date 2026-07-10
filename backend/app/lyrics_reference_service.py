from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import isfinite
from statistics import median
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .transcription_service import TranscriptPayload, TranscriptWordPayload


@dataclass
class LyricsReference:
    track_name: str
    artist_name: str | None
    plain_lyrics: str
    duration_sec: float | None
    score: float
    synced_lyrics: str | None = None


@dataclass(frozen=True)
class _LineWindowMatch:
    start_token_idx: int
    end_token_idx: int
    score: float


@dataclass(frozen=True)
class _SyncedLineMatch:
    start_word_idx: int
    end_word_idx: int
    score: float


_FILENAME_NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bofficial\b", re.IGNORECASE),
    re.compile(r"\bmusic\s+video\b", re.IGNORECASE),
    re.compile(r"\bofficial\s+video\b", re.IGNORECASE),
    re.compile(r"\bofficial\s+audio\b", re.IGNORECASE),
    re.compile(r"\blyrics?\b", re.IGNORECASE),
    re.compile(r"\bvideo\b", re.IGNORECASE),
    re.compile(r"\baudio\b", re.IGNORECASE),
    re.compile(r"\bhd\b", re.IGNORECASE),
    re.compile(r"\b4k\b", re.IGNORECASE),
    re.compile(r"\b\d{3,4}p\b", re.IGNORECASE),
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_SYNCED_LYRIC_LINE_RE = re.compile(
    r"^\[(?P<minutes>\d{2,}):(?P<seconds>\d{2})(?:\.(?P<fraction>\d{2,3}))?\]\s*(?P<text>.*)$"
)
_BRACKETED_NOISE_RE = re.compile(r"[\[(](.*?)[\])]")
_FEATURE_RE = re.compile(r"\(\s*(feat(?:uring)?\.?\s+[^)]+)\)", re.IGNORECASE)
_MUSIC_FILENAME_HINT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bmusic\s+video\b", re.IGNORECASE),
    re.compile(r"\bofficial\s+audio\b", re.IGNORECASE),
    re.compile(r"\bofficial\s+lyric(?:s)?\b", re.IGNORECASE),
    re.compile(r"\blyric(?:s)?\s+video\b", re.IGNORECASE),
    re.compile(r"\blyrical\b", re.IGNORECASE),
    re.compile(r"\bfull\s+song\b", re.IGNORECASE),
    re.compile(r"\baudio\s+jukebox\b", re.IGNORECASE),
)
_COMMON_LYRIC_TOKENS = {
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
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "there",
    "they",
    "to",
    "we",
    "you",
    "your",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)
    return max(minimum, value)


def _normalize_hint_text(value: str) -> str:
    text = str(value or "").strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    for pattern in _FILENAME_NOISE_PATTERNS:
        text = pattern.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -_.")


def _normalize_match_text(value: str) -> str:
    text = _normalize_hint_text(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\bfeat(?:uring)?\b.*$", "", text)
    text = re.sub(r"\bft\b.*$", "", text)
    text = re.sub(r"[^a-z0-9']+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _sequence_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _duration_score(reference_duration: float | None, duration_sec: float) -> float:
    if reference_duration is None or duration_sec <= 0:
        return 0.35
    delta = abs(float(reference_duration) - float(duration_sec))
    if delta <= 1.0:
        return 1.0
    if delta <= 4.0:
        return 0.9
    if delta <= 8.0:
        return 0.75
    if delta <= 15.0:
        return 0.55
    if delta <= 30.0:
        return 0.35
    return 0.0


def parse_track_hints(filename: str) -> tuple[str | None, str | None, str]:
    stem = Path(filename).stem
    raw = str(stem or "").strip().replace("_", " ")
    raw = re.sub(r"\s+", " ", raw).strip()

    featured_artist = None
    featured_match = _FEATURE_RE.search(raw)
    if featured_match:
        featured_artist = re.sub(r"\s+", " ", featured_match.group(1)).strip()

    def _strip_bracketed_noise(text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            inner = _normalize_hint_text(match.group(1))
            inner_match = _normalize_match_text(inner)
            if (
                not inner_match
                or "feat" in inner_match
                or "ft" == inner_match
            ):
                return f" {match.group(0)} "
            return " "

        return _BRACKETED_NOISE_RE.sub(replace, text)

    raw = _strip_bracketed_noise(raw)
    raw = re.sub(r"#\S+", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    cleaned = _normalize_hint_text(raw)
    artist_hint: str | None = None
    track_hint: str | None = None
    if " - " in cleaned:
        artist_hint, track_hint = (
            part.strip() or None for part in cleaned.split(" - ", 1)
        )
    elif featured_artist:
        base_without_feat = _normalize_hint_text(_FEATURE_RE.sub(" ", raw))
        tokens = [token for token in base_without_feat.split() if token]
        for artist_len in (2, 3, 1, 4):
            if len(tokens) <= artist_len + 1:
                continue
            track_candidate = " ".join(tokens[:-artist_len]).strip()
            artist_candidate = " ".join(tokens[-artist_len:]).strip()
            if not track_candidate or not artist_candidate:
                continue
            artist_hint = f"{artist_candidate} ({featured_artist})".strip()
            track_hint = track_candidate
            break
    query_hint = cleaned
    if artist_hint and track_hint:
        query_hint = f"{artist_hint} {track_hint}".strip()
    return artist_hint, track_hint, query_hint


_DUET_FILENAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bfeat(?:uring)?\.?\b", re.IGNORECASE),
    re.compile(r"\bft\.?\b", re.IGNORECASE),
    re.compile(r"\s&\s"),
    re.compile(r"\s+x\s+", re.IGNORECASE),
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bduet\b", re.IGNORECASE),
)


def looks_like_duet_media(filename: str) -> bool:
    normalized = _normalize_hint_text(Path(filename).stem)
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _DUET_FILENAME_PATTERNS)


def parse_duet_artists(filename: str) -> tuple[str | None, str | None]:
    """Return (primary_artist, featured_artist) parsed from a duet-style filename."""
    stem = Path(filename).stem
    raw = str(stem or "").strip().replace("_", " ")
    raw = re.sub(r"\s+", " ", raw).strip()

    featured_artist: str | None = None
    featured_match = _FEATURE_RE.search(raw)
    if featured_match:
        featured_text = re.sub(r"\s+", " ", featured_match.group(1)).strip()
        featured_text = re.sub(
            r"^feat(?:uring)?\.?\s*",
            "",
            featured_text,
            flags=re.IGNORECASE,
        ).strip()
        featured_artist = featured_text or None

    primary_artist: str | None = None
    if featured_match:
        before_feat = raw[: featured_match.start()].strip()
        before_feat = re.sub(r"\([^)]*\)", " ", before_feat)
        before_feat = re.sub(r"\[[^\]]*\]", " ", before_feat)
        before_feat = re.sub(r"#\S+", " ", before_feat)
        before_feat = re.sub(r"\s+", " ", before_feat).strip()
        if " - " in before_feat:
            primary_artist = before_feat.split(" - ", 1)[0].strip() or None
        else:
            artist_hint, track_hint, _query = parse_track_hints(filename)
            if artist_hint and track_hint:
                primary_artist = (
                    artist_hint.split("(", 1)[0].strip() if artist_hint else None
                )
            elif before_feat:
                tokens = before_feat.split()
                if len(tokens) >= 2:
                    primary_artist = " ".join(tokens[-2:])
                elif tokens:
                    primary_artist = tokens[-1]
    elif " & " in raw or " x " in raw.lower():
        separator = " & " if " & " in raw else " x "
        parts = [part.strip() for part in re.split(re.escape(separator), raw, maxsplit=1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            primary_artist = parts[0].split(" - ", 1)[-1].strip() or parts[0]
            featured_artist = parts[1].split("[", 1)[0].strip()

    return primary_artist, featured_artist


def looks_like_song_media(filename: str) -> bool:
    normalized = _normalize_hint_text(Path(filename).stem)
    if not normalized:
        return False
    if any(pattern.search(normalized) for pattern in _MUSIC_FILENAME_HINT_PATTERNS):
        return True
    artist_hint, track_hint, _query_hint = parse_track_hints(filename)
    if not artist_hint or not track_hint:
        return False
    artist_words = len(artist_hint.split())
    track_words = len(track_hint.split())
    return artist_words <= 6 and 1 <= track_words <= 10


def _request_json(url: str, timeout_sec: float) -> object:
    request = Request(url, headers={"User-Agent": "clipmind-transcript/1.0"})
    with urlopen(request, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_candidate_duration(item: dict[str, object]) -> float | None:
    raw = item.get("duration")
    try:
        value = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
    if value is None or not isfinite(value) or value <= 0:
        return None
    return value


def _candidate_score(
    item: dict[str, object],
    *,
    artist_hint: str | None,
    track_hint: str | None,
    query_hint: str,
    duration_sec: float,
) -> float:
    candidate_track = _normalize_match_text(
        str(item.get("trackName") or item.get("name") or "")
    )
    candidate_artist = _normalize_match_text(str(item.get("artistName") or ""))
    if not candidate_track:
        return 0.0
    title_score = _sequence_ratio(
        candidate_track,
        _normalize_match_text(track_hint or query_hint),
    )
    if artist_hint:
        artist_score = _sequence_ratio(candidate_artist, _normalize_match_text(artist_hint))
    else:
        artist_score = 0.45 if candidate_artist else 0.0
    duration_score = _duration_score(_extract_candidate_duration(item), duration_sec)
    lyrics = str(item.get("plainLyrics") or "").strip()
    lyrics_score = 0.25 if len(lyrics.split()) >= 40 else 0.0
    return round(
        (title_score * 0.5)
        + (artist_score * 0.2)
        + (duration_score * 0.25)
        + lyrics_score,
        4,
    )


def fetch_lyrics_reference(filename: str, duration_sec: float) -> LyricsReference | None:
    if not _env_bool("TRANSCRIBE_LYRICS_REFERENCE_ENABLED", True):
        return None
    artist_hint, track_hint, query_hint = parse_track_hints(filename)
    if not track_hint and not query_hint:
        return None

    timeout_sec = _env_float("TRANSCRIBE_LYRICS_REFERENCE_TIMEOUT_SEC", 12.0, 0.5)
    base_url = (
        os.getenv("TRANSCRIBE_LYRICS_REFERENCE_API_URL", "https://lrclib.net/api/search").strip()
        or "https://lrclib.net/api/search"
    )
    urls: list[str] = []
    if track_hint and artist_hint:
        urls.append(
            f"{base_url}?{urlencode({'track_name': track_hint, 'artist_name': artist_hint})}"
        )
    urls.append(f"{base_url}?{urlencode({'q': query_hint})}")

    best: LyricsReference | None = None
    best_score = _env_float("TRANSCRIBE_LYRICS_REFERENCE_MIN_SCORE", 0.84, 0.0)
    for url in urls:
        try:
            payload = _request_json(url, timeout_sec)
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            if bool(item.get("instrumental")):
                continue
            plain_lyrics = str(item.get("plainLyrics") or "").strip()
            if not plain_lyrics:
                continue
            score = _candidate_score(
                item,
                artist_hint=artist_hint,
                track_hint=track_hint,
                query_hint=query_hint,
                duration_sec=duration_sec,
            )
            if score < best_score:
                continue
            candidate = LyricsReference(
                track_name=str(
                    item.get("trackName") or item.get("name") or track_hint or query_hint
                ).strip(),
                artist_name=str(item.get("artistName") or artist_hint or "").strip() or None,
                plain_lyrics=plain_lyrics,
                duration_sec=_extract_candidate_duration(item),
                score=score,
                synced_lyrics=str(item.get("syncedLyrics") or "").strip() or None,
            )
            if best is None or candidate.score > best.score:
                best = candidate
                best_score = candidate.score
    return best


def _tokenize_lyrics(lyrics: str) -> list[str]:
    tokens: list[str] = []
    for raw in lyrics.split():
        cleaned = raw.strip()
        token = cleaned.strip('"“”‘’()[]{}')
        if not _TOKEN_RE.search(token):
            continue
        tokens.append(token)
    return _sanitize_reference_tokens(tokens)


def _sanitize_reference_tokens(ref_tokens: list[str]) -> list[str]:
    tokens = list(ref_tokens)
    if len(tokens) >= 2 and tokens[0].lower() == "me" and tokens[1].lower() == "be":
        tokens[0] = "To"

    for idx in range(len(tokens) - 1):
        current = tokens[idx]
        nxt = tokens[idx + 1]
        if current.lower() in {"a", "an"} and nxt and nxt[0].lower() in {"a", "e", "i", "o", "u"}:
            tokens[idx] = "an" if current.islower() else "An"
    return tokens


def _normalize_token(value: str) -> str:
    match = _TOKEN_RE.search(str(value or ""))
    if not match:
        return ""
    return match.group(0).lower()


def _build_interpolated_words(
    ref_tokens: list[str],
    *,
    start_sec: float,
    end_sec: float,
    id_prefix: str = "lyrics-ref",
    confidence: float = 1.0,
    quality_score: float = 0.82,
    quality_label: str = "trusted",
    source_pass: str = "manual",
) -> list[TranscriptWordPayload]:
    if not ref_tokens:
        return []
    span = max(float(end_sec) - float(start_sec), 0.05 * len(ref_tokens))
    step = span / max(len(ref_tokens), 1)
    words: list[TranscriptWordPayload] = []
    for idx, token in enumerate(ref_tokens):
        word_start = float(start_sec) + (idx * step)
        word_end = min(float(end_sec), word_start + max(0.05, step * 0.92))
        if word_end <= word_start:
            word_end = word_start + 0.05
        words.append(
            TranscriptWordPayload(
                id=f"{id_prefix}-{idx}",
                text=token,
                start_sec=round(word_start, 3),
                end_sec=round(word_end, 3),
                confidence=max(0.0, min(float(confidence), 1.0)),
                quality_score=max(0.0, min(float(quality_score), 1.0)),
                quality_label="trusted" if str(quality_label) == "trusted" else "weak",
                source_pass=source_pass,
            )
        )
    return words


def _build_reference_words_with_asr_timing(
    ref_tokens: list[str],
    asr_words: list[TranscriptWordPayload],
    *,
    id_prefix: str,
    confidence: float = 1.0,
    quality_score: float = 0.82,
    quality_label: str = "trusted",
    source_pass: str = "manual",
) -> list[TranscriptWordPayload]:
    """Apply corrected lyric text without flattening the ASR timing rhythm.

    Synced lyric providers normally provide a timestamp for an entire line,
    not every word. Splitting that line into equal word windows makes fast and
    slow words visibly lead or lag the vocal. A matched ASR word window has
    useful per-word timing, so it remains the timing source.
    """
    if not ref_tokens or not asr_words:
        return []

    source_count = len(asr_words)
    token_count = len(ref_tokens)
    boundaries = [float(asr_words[0].start_sec)]
    for word in asr_words:
        boundaries.append(max(boundaries[-1], float(word.end_sec)))

    def time_at(position: float) -> float:
        clamped = max(0.0, min(float(source_count), position))
        lower = int(clamped)
        if lower >= source_count or abs(clamped - lower) < 1e-9:
            return boundaries[lower]
        upper = lower + 1
        fraction = clamped - lower
        return boundaries[lower] + (boundaries[upper] - boundaries[lower]) * fraction

    words: list[TranscriptWordPayload] = []
    for idx, token in enumerate(ref_tokens):
        start_sec = time_at((idx * source_count) / token_count)
        end_sec = time_at(((idx + 1) * source_count) / token_count)
        if end_sec <= start_sec:
            end_sec = start_sec + 0.001
        source_idx = min(
            source_count - 1,
            int(((idx + 0.5) * source_count) / token_count),
        )
        source_word = asr_words[source_idx]
        words.append(
            TranscriptWordPayload(
                id=f"{id_prefix}-{idx}",
                text=token,
                start_sec=round(start_sec, 3),
                end_sec=round(end_sec, 3),
                confidence=max(0.0, min(float(confidence), 1.0)),
                quality_score=max(0.0, min(float(quality_score), 1.0)),
                quality_label="trusted"
                if str(quality_label) == "trusted"
                else "weak",
                source_pass=source_pass,
                speaker_id=source_word.speaker_id,
                speaker_label=source_word.speaker_label,
            )
        )
    return words


def _lyrics_line_quality(score: float) -> tuple[float, str]:
    bounded = max(0.0, min(float(score), 1.0))
    trusted_threshold = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_TRUSTED_LINE_SCORE",
        0.86,
        0.0,
    )
    weak_floor = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_WEAK_LINE_SCORE",
        0.62,
        0.0,
    )
    if bounded >= trusted_threshold:
        return round(max(0.86, bounded), 3), "trusted"
    if bounded >= weak_floor:
        return round(max(0.45, bounded), 3), "weak"
    return round(max(0.25, bounded), 3), "weak"


def _window_tokens_from_asr_words(
    words: list[TranscriptWordPayload],
    *,
    start_sec: float,
    end_sec: float,
) -> list[str]:
    tokens: list[str] = []
    for word in words:
        word_start = float(word.start_sec)
        word_end = float(word.end_sec)
        if word_end <= start_sec or word_start >= end_sec:
            continue
        normalized = _normalize_token(word.text)
        if normalized:
            tokens.append(normalized)
    return tokens


def _content_token_overlap_ratio(window_tokens: list[str], reference_tokens: list[str]) -> float:
    filtered_window = [token for token in window_tokens if token not in _COMMON_LYRIC_TOKENS]
    filtered_reference = [
        token for token in reference_tokens if token not in _COMMON_LYRIC_TOKENS
    ]
    if not filtered_reference:
        return _token_overlap_ratio(window_tokens, reference_tokens)
    return _token_overlap_ratio(filtered_window, filtered_reference)


def _region_span(
    words: list[TranscriptWordPayload],
    start_idx: int,
    end_idx: int,
    *,
    fallback_start: float,
    fallback_end: float,
    duration_sec: float,
) -> tuple[float, float]:
    if start_idx < end_idx:
        start_sec = float(words[start_idx].start_sec)
        end_sec = float(words[end_idx - 1].end_sec)
        return start_sec, max(start_sec + 0.05, end_sec)
    start_sec = max(0.0, min(float(fallback_start), duration_sec))
    end_sec = max(start_sec + 0.05, min(float(fallback_end), duration_sec))
    return start_sec, end_sec


def _reference_lines(lyrics: str) -> list[list[str]]:
    lines: list[list[str]] = []
    for raw_line in str(lyrics or "").splitlines():
        tokens = _tokenize_lyrics(raw_line)
        if len(tokens) >= 4:
            lines.append(tokens)
    return lines if len(lines) >= 2 else []


def _parse_synced_lyric_lines(synced_lyrics: str) -> list[tuple[float, list[str]]]:
    lines: list[tuple[float, list[str]]] = []
    for raw_line in str(synced_lyrics or "").splitlines():
        match = _SYNCED_LYRIC_LINE_RE.match(raw_line.strip())
        if not match:
            continue
        lyric_text = match.group("text")
        tokens = _tokenize_lyrics(lyric_text)
        if not tokens:
            continue
        fraction = match.group("fraction") or ""
        if fraction:
            divisor = 1000.0 if len(fraction) == 3 else 100.0
            fraction_sec = int(fraction) / divisor
        else:
            fraction_sec = 0.0
        start_sec = (int(match.group("minutes")) * 60) + int(match.group("seconds")) + fraction_sec
        lines.append((round(start_sec, 3), tokens))
    lines.sort(key=lambda item: item[0])
    return lines


def _align_reference_lyrics_by_synced_lines(
    payload: TranscriptPayload,
    reference: LyricsReference,
    *,
    duration_sec: float,
) -> TranscriptPayload | None:
    synced_lines = _parse_synced_lyric_lines(reference.synced_lyrics or "")
    if len(synced_lines) < 6:
        return None
    if reference.duration_sec is not None and abs(float(reference.duration_sec) - duration_sec) > 20.0:
        return None

    asr_words = list(payload.words)
    if len(asr_words) < 20:
        return None
    synced_time_offset_sec = _estimate_synced_time_offset(
        asr_words,
        synced_lines,
        duration_sec=duration_sec,
    )
    synced_anchor_scores = _collect_synced_anchor_scores(
        asr_words,
        synced_lines,
    )
    rebuilt_payload = _rebuild_payload_from_synced_reference(
        payload,
        synced_lines,
        duration_sec=duration_sec,
        time_offset_sec=synced_time_offset_sec,
        anchor_scores=synced_anchor_scores,
    )
    if rebuilt_payload is not None:
        return rebuilt_payload

    min_replace_score = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_MIN_SYNCED_REPLACE_SCORE",
        0.82,
        0.0,
    )
    min_average_score = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_MIN_SYNCED_AVG_SCORE",
        0.7,
        0.0,
    )
    min_rescue_score = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_MIN_SYNCED_RESCUE_SCORE",
        0.64,
        0.0,
    )
    min_rescue_content_overlap = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_MIN_SYNCED_RESCUE_CONTENT_OVERLAP",
        0.5,
        0.0,
    )

    corrected: list[TranscriptWordPayload] = []
    cursor_word_idx = 0
    adopted_line_scores: list[float] = []
    adopted_token_count = 0
    total_synced_token_count = sum(len(tokens) for _, tokens in synced_lines)
    for idx, (start_sec, tokens) in enumerate(synced_lines):
        adjusted_start_sec = max(0.0, min(duration_sec, start_sec + synced_time_offset_sec))
        next_start = (
            max(0.0, min(duration_sec, synced_lines[idx + 1][0] + synced_time_offset_sec))
            if idx + 1 < len(synced_lines)
            else duration_sec
        )
        natural_span = max(0.7, 0.24 * len(tokens))
        gap_to_next = max(0.0, next_start - adjusted_start_sec - 0.02)
        if 0.0 < gap_to_next <= 4.0:
            span = gap_to_next
        else:
            span = min(max(natural_span, 1.0), 4.0)
        end_sec = min(duration_sec, adjusted_start_sec + span)
        if end_sec <= adjusted_start_sec:
            end_sec = min(duration_sec, adjusted_start_sec + max(0.6, natural_span))

        match = _find_best_synced_line_window(
            asr_words,
            [_normalize_token(token) for token in tokens if _normalize_token(token)],
            expected_start_sec=adjusted_start_sec,
            expected_end_sec=end_sec,
            start_word_idx=cursor_word_idx,
        )
        if match is not None:
            replace_start_idx = match.start_word_idx
            replace_end_idx = match.end_word_idx
            line_score = match.score
        else:
            replace_start_idx = cursor_word_idx
            while replace_start_idx < len(asr_words) and float(asr_words[replace_start_idx].end_sec) <= adjusted_start_sec:
                replace_start_idx += 1
            replace_end_idx = replace_start_idx
            while replace_end_idx < len(asr_words) and float(asr_words[replace_end_idx].start_sec) < end_sec:
                replace_end_idx += 1
            overlapping_words = asr_words[replace_start_idx:replace_end_idx]
            window_tokens = _window_tokens_from_asr_words(
                overlapping_words,
                start_sec=adjusted_start_sec,
                end_sec=end_sec,
            )
            line_score = _line_similarity_score(
                window_tokens,
                [_normalize_token(token) for token in tokens if _normalize_token(token)],
            )

        overlapping_words = asr_words[replace_start_idx:replace_end_idx]
        if match is not None and replace_start_idx > cursor_word_idx:
            corrected.extend(asr_words[cursor_word_idx:replace_start_idx])

        reference_norm = [_normalize_token(token) for token in tokens if _normalize_token(token)]
        content_overlap = _content_token_overlap_ratio(
            [_normalize_token(word.text) for word in overlapping_words if _normalize_token(word.text)],
            reference_norm,
        )

        if line_score >= min_replace_score:
            adopted_line_scores.append(line_score)
            adopted_token_count += len(tokens)
            quality_score, quality_label = _lyrics_line_quality(line_score)
            corrected.extend(
                _build_reference_words_with_asr_timing(
                    tokens,
                    asr_words[replace_start_idx:replace_end_idx],
                    id_prefix=f"lyrics-sync-{idx}",
                    confidence=quality_score,
                    quality_score=quality_score,
                    quality_label=quality_label,
                )
            )
            cursor_word_idx = replace_end_idx
        elif (
            match is not None
            and line_score >= min_rescue_score
            and content_overlap >= min_rescue_content_overlap
        ):
            adopted_line_scores.append(line_score)
            adopted_token_count += len(tokens)
            quality_score, quality_label = _lyrics_line_quality(line_score)
            corrected.extend(
                _build_reference_words_with_asr_timing(
                    tokens,
                    asr_words[replace_start_idx:replace_end_idx],
                    id_prefix=f"lyrics-sync-rescue-{idx}",
                    confidence=quality_score,
                    quality_score=quality_score,
                    quality_label=quality_label,
                )
            )
            cursor_word_idx = replace_end_idx
        elif overlapping_words:
            corrected.extend(overlapping_words)
            cursor_word_idx = replace_end_idx

    if cursor_word_idx < len(asr_words):
        corrected.extend(asr_words[cursor_word_idx:])

    min_adopted_lines = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_MIN_SYNCED_ADOPTED_LINES",
        2.0,
        0.0,
    )
    min_adopted_ratio = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_MIN_SYNCED_ADOPTED_TOKEN_RATIO",
        0.18,
        0.0,
    )
    adopted_ratio = adopted_token_count / max(total_synced_token_count, 1)

    if len(adopted_line_scores) < int(round(min_adopted_lines)) or adopted_ratio < min_adopted_ratio:
        return None
    average_score = sum(adopted_line_scores) / max(len(adopted_line_scores), 1)
    if average_score < min_average_score:
        return None

    corrected = _trim_synced_reference_edge_words(
        corrected,
        synced_lines=synced_lines,
        duration_sec=duration_sec,
        time_offset_sec=synced_time_offset_sec,
    )

    return _finalize_corrected_payload(payload, corrected)


def _token_overlap_ratio(window_tokens: list[str], reference_tokens: list[str]) -> float:
    if not window_tokens or not reference_tokens:
        return 0.0
    window_counts = Counter(window_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum(
        min(window_counts[token], count) for token, count in reference_counts.items()
    )
    return overlap / max(len(reference_tokens), 1)


def _line_similarity_score(window_tokens: list[str], reference_tokens: list[str]) -> float:
    if not window_tokens or not reference_tokens:
        return 0.0
    sequence_score = SequenceMatcher(
        None,
        window_tokens,
        reference_tokens,
        autojunk=False,
    ).ratio()
    overlap_score = _token_overlap_ratio(window_tokens, reference_tokens)
    anchor_score = 0.0
    if window_tokens[0] == reference_tokens[0]:
        anchor_score += 0.06
    elif len(reference_tokens) >= 5:
        anchor_score -= 0.12
    if window_tokens[-1] == reference_tokens[-1]:
        anchor_score += 0.06
    elif len(reference_tokens) >= 5:
        anchor_score -= 0.08
    if len(window_tokens) >= 2 and len(reference_tokens) >= 2:
        if window_tokens[:2] == reference_tokens[:2]:
            anchor_score += 0.04
        elif len(reference_tokens) >= 6:
            anchor_score -= 0.08
        if window_tokens[-2:] == reference_tokens[-2:]:
            anchor_score += 0.04
        elif len(reference_tokens) >= 6:
            anchor_score -= 0.05
    length_penalty = min(
        0.18,
        (abs(len(window_tokens) - len(reference_tokens)) / max(len(reference_tokens), 1))
        * 0.12,
    )
    return (sequence_score * 0.62) + (overlap_score * 0.28) + anchor_score - length_penalty


def _find_best_synced_line_window(
    asr_words: list[TranscriptWordPayload],
    reference_tokens: list[str],
    *,
    expected_start_sec: float,
    expected_end_sec: float,
    start_word_idx: int,
) -> _SyncedLineMatch | None:
    if not reference_tokens or start_word_idx >= len(asr_words):
        return None

    pre_pad_sec = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_SYNCED_SEARCH_PRE_PAD_SEC",
        1.75,
        0.0,
    )
    post_pad_sec = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_SYNCED_SEARCH_POST_PAD_SEC",
        2.75,
        0.0,
    )
    max_start_drift_sec = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_SYNCED_MAX_START_DRIFT_SEC",
        2.6,
        0.0,
    )
    min_score = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_MIN_SYNCED_SEARCH_SCORE",
        0.56,
        0.0,
    )

    search_start_sec = max(0.0, expected_start_sec - pre_pad_sec)
    search_end_sec = max(expected_end_sec, expected_start_sec) + post_pad_sec
    candidate_start_indices: list[int] = []
    for word_idx in range(start_word_idx, len(asr_words)):
        word = asr_words[word_idx]
        word_start = float(word.start_sec)
        if word_start > search_end_sec:
            break
        if float(word.end_sec) < search_start_sec:
            continue
        candidate_start_indices.append(word_idx)
    if not candidate_start_indices:
        return None

    length_pad = int(
        round(
            _env_float(
                "TRANSCRIBE_LYRICS_REFERENCE_SYNCED_LENGTH_PAD",
                max(2.0, len(reference_tokens) * 0.3),
                1.0,
            )
        )
    )
    min_len = max(4, len(reference_tokens) - length_pad)
    best: _SyncedLineMatch | None = None
    for word_start_idx in candidate_start_indices:
        start_drift = abs(float(asr_words[word_start_idx].start_sec) - expected_start_sec)
        if start_drift > max_start_drift_sec:
            continue
        max_len = min(len(asr_words) - word_start_idx, len(reference_tokens) + length_pad)
        for window_len in range(min_len, max_len + 1):
            word_end_idx = word_start_idx + window_len
            if word_end_idx > len(asr_words):
                break
            window_tokens = [
                normalized
                for normalized in (
                    _normalize_token(word.text) for word in asr_words[word_start_idx:word_end_idx]
                )
                if normalized
            ]
            if len(window_tokens) < 4:
                continue
            base_score = _line_similarity_score(window_tokens, reference_tokens)
            content_overlap = _content_token_overlap_ratio(window_tokens, reference_tokens)
            drift_penalty = min(0.18, (start_drift / max(max_start_drift_sec, 0.1)) * 0.16)
            score = base_score + (content_overlap * 0.08) - drift_penalty
            if best is None or score > best.score + 1e-9:
                best = _SyncedLineMatch(word_start_idx, word_end_idx, score)
                continue
            if best is not None and abs(score - best.score) <= 1e-9:
                current_delta = abs(window_len - len(reference_tokens))
                best_delta = abs(
                    (best.end_word_idx - best.start_word_idx) - len(reference_tokens)
                )
                if current_delta < best_delta or (
                    current_delta == best_delta and word_start_idx < best.start_word_idx
                ):
                    best = _SyncedLineMatch(word_start_idx, word_end_idx, score)

    if best is None or best.score < min_score:
        return None
    return best


def _find_best_reference_window_anywhere(
    asr_words: list[TranscriptWordPayload],
    reference_tokens: list[str],
    *,
    start_word_idx: int = 0,
) -> _SyncedLineMatch | None:
    if not reference_tokens or start_word_idx >= len(asr_words):
        return None

    length_pad = int(
        round(
            _env_float(
                "TRANSCRIBE_LYRICS_REFERENCE_ANYWHERE_LENGTH_PAD",
                max(2.0, len(reference_tokens) * 0.3),
                1.0,
            )
        )
    )
    min_len = max(4, len(reference_tokens) - length_pad)
    best: _SyncedLineMatch | None = None
    for word_start_idx in range(start_word_idx, len(asr_words) - min_len + 1):
        max_len = min(len(asr_words) - word_start_idx, len(reference_tokens) + length_pad)
        for window_len in range(min_len, max_len + 1):
            word_end_idx = word_start_idx + window_len
            window_tokens = [
                normalized
                for normalized in (
                    _normalize_token(word.text) for word in asr_words[word_start_idx:word_end_idx]
                )
                if normalized
            ]
            if len(window_tokens) < 4:
                continue
            base_score = _line_similarity_score(window_tokens, reference_tokens)
            content_overlap = _content_token_overlap_ratio(window_tokens, reference_tokens)
            score = base_score + (content_overlap * 0.08)
            if best is None or score > best.score + 1e-9:
                best = _SyncedLineMatch(word_start_idx, word_end_idx, score)
                continue
            if best is not None and abs(score - best.score) <= 1e-9:
                current_delta = abs(window_len - len(reference_tokens))
                best_delta = abs(
                    (best.end_word_idx - best.start_word_idx) - len(reference_tokens)
                )
                if current_delta < best_delta or (
                    current_delta == best_delta and word_start_idx < best.start_word_idx
                ):
                    best = _SyncedLineMatch(word_start_idx, word_end_idx, score)
    return best


def _estimate_synced_time_offset(
    asr_words: list[TranscriptWordPayload],
    synced_lines: list[tuple[float, list[str]]],
    *,
    duration_sec: float,
) -> float:
    anchor_matches = _sample_synced_anchor_matches(
        asr_words,
        synced_lines,
        duration_sec=duration_sec,
    )
    offsets = [offset_sec for offset_sec, _score in anchor_matches]
    if len(offsets) < 2:
        return 0.0

    min_detectable_offset_sec = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_SYNCED_MIN_DETECTABLE_OFFSET_SEC",
        1.2,
        0.0,
    )
    estimated = float(median(offsets))
    if abs(estimated) < min_detectable_offset_sec:
        return 0.0
    return round(estimated, 3)


def _sample_synced_anchor_matches(
    asr_words: list[TranscriptWordPayload],
    synced_lines: list[tuple[float, list[str]]],
    *,
    duration_sec: float,
) -> list[tuple[float, float]]:
    if len(asr_words) < 20 or len(synced_lines) < 4:
        return []

    sample_line_count = int(
        round(_env_float("TRANSCRIBE_LYRICS_REFERENCE_SYNCED_OFFSET_SAMPLE_LINES", 8.0, 2.0))
    )
    min_anchor_score = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_SYNCED_OFFSET_MIN_SCORE",
        0.74,
        0.0,
    )
    max_abs_offset_sec = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_SYNCED_MAX_ABS_OFFSET_SEC",
        min(30.0, max(duration_sec * 0.35, 12.0)),
        0.0,
    )

    matches: list[tuple[float, float]] = []
    cursor_word_idx = 0
    for start_sec, tokens in synced_lines[:sample_line_count]:
        reference_tokens = [_normalize_token(token) for token in tokens if _normalize_token(token)]
        if len(reference_tokens) < 4:
            continue
        match = _find_best_reference_window_anywhere(
            asr_words,
            reference_tokens,
            start_word_idx=cursor_word_idx,
        )
        if match is None or match.score < min_anchor_score:
            continue
        matched_start_sec = float(asr_words[match.start_word_idx].start_sec)
        offset_sec = matched_start_sec - float(start_sec)
        if abs(offset_sec) > max_abs_offset_sec:
            continue
        matches.append((offset_sec, match.score))
        cursor_word_idx = match.end_word_idx
        if len(matches) >= 3:
            break

    return matches


def _collect_synced_anchor_scores(
    asr_words: list[TranscriptWordPayload],
    synced_lines: list[tuple[float, list[str]]],
) -> list[float]:
    return [score for _offset_sec, score in _sample_synced_anchor_matches(asr_words, synced_lines, duration_sec=max(float(asr_words[-1].end_sec), 1.0))]


def _rebuild_payload_from_synced_reference(
    payload: TranscriptPayload,
    synced_lines: list[tuple[float, list[str]]],
    *,
    duration_sec: float,
    time_offset_sec: float,
    anchor_scores: list[float],
) -> TranscriptPayload | None:
    min_anchor_count = int(
        round(_env_float("TRANSCRIBE_LYRICS_REFERENCE_SYNCED_REBUILD_MIN_ANCHORS", 2.0, 1.0))
    )
    min_anchor_average = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_SYNCED_REBUILD_MIN_AVG_SCORE",
        0.82,
        0.0,
    )
    if len(anchor_scores) < min_anchor_count:
        return None
    anchor_average = sum(anchor_scores) / max(len(anchor_scores), 1)
    if anchor_average < min_anchor_average:
        return None

    quality_score, quality_label = _lyrics_line_quality(anchor_average)
    asr_words = list(payload.words)
    if not synced_lines or not asr_words:
        return None

    # Line timestamps tell us *where* to search, but are not precise enough to
    # create new word timing. Reuse the matched ASR window so every corrected
    # lyric keeps the singer's actual, non-uniform cadence.
    corrected: list[TranscriptWordPayload] = []
    cursor_word_idx = 0
    matched_line_count = 0
    for idx, (start_sec, tokens) in enumerate(synced_lines):
        adjusted_start_sec = max(
            0.0, min(duration_sec, start_sec + time_offset_sec)
        )
        next_start_sec = (
            max(
                0.0,
                min(duration_sec, synced_lines[idx + 1][0] + time_offset_sec),
            )
            if idx + 1 < len(synced_lines)
            else duration_sec
        )
        natural_span = max(0.7, 0.24 * len(tokens))
        gap_to_next = max(0.0, next_start_sec - adjusted_start_sec - 0.02)
        span = (
            gap_to_next
            if 0.0 < gap_to_next <= 4.0
            else min(max(natural_span, 1.0), 4.0)
        )
        end_sec = min(duration_sec, adjusted_start_sec + span)
        if end_sec <= adjusted_start_sec:
            end_sec = min(duration_sec, adjusted_start_sec + max(0.6, natural_span))

        reference_tokens = [
            _normalize_token(token) for token in tokens if _normalize_token(token)
        ]
        match = _find_best_synced_line_window(
            asr_words,
            reference_tokens,
            expected_start_sec=adjusted_start_sec,
            expected_end_sec=end_sec,
            start_word_idx=cursor_word_idx,
        )
        if match is None:
            continue
        # The first synced lyric line is the content anchor. Anything before
        # its matched ASR window is commonly spoken/video-intro noise rather
        # than part of the lyric reference, so preserve gaps only after that
        # first anchor.
        if match.start_word_idx > cursor_word_idx and matched_line_count > 0:
            corrected.extend(asr_words[cursor_word_idx : match.start_word_idx])
        corrected.extend(
            _build_reference_words_with_asr_timing(
                tokens,
                asr_words[match.start_word_idx : match.end_word_idx],
                id_prefix=f"lyrics-sync-rebuild-{idx}",
                confidence=quality_score,
                quality_score=quality_score,
                quality_label=quality_label,
            )
        )
        cursor_word_idx = match.end_word_idx
        matched_line_count += 1

    if matched_line_count < min_anchor_count:
        return None
    if cursor_word_idx < len(asr_words):
        corrected.extend(asr_words[cursor_word_idx:])
    return _finalize_corrected_payload(payload, corrected)


def _should_keep_synced_edge_words(
    edge_words: list[TranscriptWordPayload],
    reference_tokens: list[str],
    *,
    side: str,
    gap_to_synced_sec: float,
) -> bool:
    if not edge_words:
        return False
    edge_tokens = [
        normalized
        for normalized in (_normalize_token(word.text) for word in edge_words)
        if normalized
    ]
    reference_norm = [_normalize_token(token) for token in reference_tokens if _normalize_token(token)]
    if not edge_tokens or not reference_norm:
        return False

    near_sync_gap_sec = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_SYNCED_EDGE_NEAR_GAP_SEC",
        1.6,
        0.0,
    )
    max_edge_span_sec = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_SYNCED_EDGE_MAX_SPAN_SEC",
        12.0,
        0.0,
    )
    max_edge_words = max(
        1,
        int(
            round(
                _env_float(
                    "TRANSCRIBE_LYRICS_REFERENCE_SYNCED_EDGE_MAX_WORDS",
                    24.0,
                    1.0,
                )
            )
        ),
    )
    min_content_overlap = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_SYNCED_EDGE_MIN_OVERLAP",
        0.45,
        0.0,
    )

    edge_span_sec = max(
        0.0, float(edge_words[-1].end_sec) - float(edge_words[0].start_sec)
    )
    content_overlap = _content_token_overlap_ratio(edge_tokens, reference_norm)
    matched_prefix = False
    if side == "leading" and len(edge_tokens) <= len(reference_norm):
        matched_prefix = edge_tokens == reference_norm[: len(edge_tokens)]
    elif side == "trailing" and len(edge_tokens) <= len(reference_norm):
        matched_prefix = edge_tokens == reference_norm[-len(edge_tokens) :]

    if gap_to_synced_sec <= near_sync_gap_sec and matched_prefix and len(edge_tokens) >= 2:
        return False
    if matched_prefix and len(edge_tokens) == 1:
        return True
    if len(edge_tokens) <= max_edge_words and edge_span_sec <= max_edge_span_sec:
        return content_overlap >= min_content_overlap
    return True


def _trim_synced_reference_edge_words(
    words: list[TranscriptWordPayload],
    *,
    synced_lines: list[tuple[float, list[str]]],
    duration_sec: float,
    time_offset_sec: float,
) -> list[TranscriptWordPayload]:
    if not words or not synced_lines:
        return words

    trimmed = list(words)
    first_start_sec = max(
        0.0, min(duration_sec, float(synced_lines[0][0]) + float(time_offset_sec))
    )
    last_start_sec = max(
        0.0, min(duration_sec, float(synced_lines[-1][0]) + float(time_offset_sec))
    )
    last_tokens = synced_lines[-1][1]
    last_natural_span = max(0.7, 0.24 * len(last_tokens))
    last_end_sec = min(
        duration_sec, last_start_sec + min(max(last_natural_span, 1.0), 4.0)
    )

    leading_count = 0
    for word in trimmed:
        if float(word.end_sec) <= first_start_sec:
            leading_count += 1
            continue
        break
    if leading_count:
        leading_words = trimmed[:leading_count]
        leading_gap_sec = max(
            0.0, first_start_sec - float(leading_words[-1].end_sec)
        )
        if not _should_keep_synced_edge_words(
            leading_words,
            synced_lines[0][1],
            side="leading",
            gap_to_synced_sec=leading_gap_sec,
        ):
            trimmed = trimmed[leading_count:]

    trailing_count = 0
    for word in reversed(trimmed):
        if float(word.start_sec) >= last_end_sec:
            trailing_count += 1
            continue
        break
    if trailing_count:
        trailing_words = trimmed[-trailing_count:]
        trailing_gap_sec = max(
            0.0, float(trailing_words[0].start_sec) - last_end_sec
        )
        if not _should_keep_synced_edge_words(
            trailing_words,
            synced_lines[-1][1],
            side="trailing",
            gap_to_synced_sec=trailing_gap_sec,
        ):
            trimmed = trimmed[:-trailing_count]

    return trimmed if trimmed else words


def _find_best_line_window(
    asr_norm_tokens: list[str],
    reference_tokens: list[str],
    *,
    start_token_idx: int,
) -> _LineWindowMatch | None:
    if len(asr_norm_tokens) - start_token_idx < 4:
        return None

    length_pad = int(
        round(
            _env_float(
                "TRANSCRIBE_LYRICS_REFERENCE_LINE_LENGTH_PAD",
                max(3.0, len(reference_tokens) * 0.35),
                1.0,
            )
        )
    )
    lookahead = int(
        round(
            _env_float(
                "TRANSCRIBE_LYRICS_REFERENCE_LINE_LOOKAHEAD_WORDS",
                max(20.0, len(reference_tokens) * 2.5),
                8.0,
            )
        )
    )
    min_line_ratio = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_MIN_LINE_RATIO",
        0.6,
        0.0,
    )

    min_len = max(4, len(reference_tokens) - length_pad)
    best: _LineWindowMatch | None = None
    max_start = min(len(asr_norm_tokens) - min_len, start_token_idx + lookahead)
    for token_start in range(start_token_idx, max_start + 1):
        remaining = len(asr_norm_tokens) - token_start
        if remaining < min_len:
            break
        max_len = min(remaining, len(reference_tokens) + length_pad)
        for window_len in range(min_len, max_len + 1):
            token_end = token_start + window_len
            score = _line_similarity_score(
                asr_norm_tokens[token_start:token_end],
                reference_tokens,
            )
            if best is None or score > best.score + 1e-9:
                best = _LineWindowMatch(token_start, token_end, score)
                continue
            if best is not None and abs(score - best.score) <= 1e-9:
                current_delta = abs(window_len - len(reference_tokens))
                best_delta = abs(
                    (best.end_token_idx - best.start_token_idx) - len(reference_tokens)
                )
                if current_delta < best_delta or (
                    current_delta == best_delta and token_start < best.start_token_idx
                ):
                    best = _LineWindowMatch(token_start, token_end, score)

    if best is None or best.score < min_line_ratio:
        return None
    return best


def _finalize_corrected_payload(
    payload: TranscriptPayload,
    corrected: list[TranscriptWordPayload],
) -> TranscriptPayload:
    if len(corrected) < 20:
        return payload
    corrected.sort(key=lambda item: (float(item.start_sec), float(item.end_sec), item.id))
    return TranscriptPayload(
        source=f"{payload.source}_lyrics_ref",
        language=payload.language,
        text=" ".join(word.text for word in corrected).strip(),
        words=corrected,
        is_mock=payload.is_mock,
    )


def _merge_matched_line_regions(
    matched_regions: list[tuple[int, int, list[str], float]],
    asr_words: list[TranscriptWordPayload],
) -> list[tuple[int, int, list[str]]]:
    if not matched_regions:
        return []

    max_gap_words = int(
        round(_env_float("TRANSCRIBE_LYRICS_REFERENCE_BLOCK_GAP_WORDS", 4.0, 0.0))
    )
    max_gap_sec = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_BLOCK_GAP_SEC",
        1.0,
        0.0,
    )

    block_start_idx, block_end_idx, block_tokens, _score = matched_regions[0]
    merged: list[tuple[int, int, list[str]]] = []
    current_tokens = list(block_tokens)
    current_end_sec = float(asr_words[block_end_idx - 1].end_sec)
    for word_start_idx, word_end_idx, reference_line, _score in matched_regions[1:]:
        next_start_sec = float(asr_words[word_start_idx].start_sec)
        gap_words = max(0, word_start_idx - block_end_idx)
        gap_sec = max(0.0, next_start_sec - current_end_sec)
        if gap_words <= max_gap_words and gap_sec <= max_gap_sec:
            block_end_idx = word_end_idx
            current_tokens.extend(reference_line)
            current_end_sec = float(asr_words[block_end_idx - 1].end_sec)
            continue
        merged.append((block_start_idx, block_end_idx, current_tokens))
        block_start_idx = word_start_idx
        block_end_idx = word_end_idx
        current_tokens = list(reference_line)
        current_end_sec = float(asr_words[block_end_idx - 1].end_sec)

    merged.append((block_start_idx, block_end_idx, current_tokens))
    return merged


def _align_reference_lyrics_by_lines(
    payload: TranscriptPayload,
    reference: LyricsReference,
    *,
    ref_tokens: list[str],
) -> TranscriptPayload | None:
    reference_lines = _reference_lines(reference.plain_lyrics)
    if not reference_lines:
        return None

    asr_words = list(payload.words)
    asr_token_word_indices: list[int] = []
    asr_norm_tokens: list[str] = []
    for word_idx, word in enumerate(asr_words):
        normalized = _normalize_token(word.text)
        if not normalized:
            continue
        asr_token_word_indices.append(word_idx)
        asr_norm_tokens.append(normalized)
    if len(asr_norm_tokens) < 20:
        return None

    matched_regions: list[tuple[int, int, list[str], float]] = []
    cursor_token_idx = 0
    matched_ref_tokens = 0
    line_scores: list[float] = []
    for reference_line in reference_lines:
        reference_norm = [_normalize_token(token) for token in reference_line]
        reference_norm = [token for token in reference_norm if token]
        if len(reference_norm) < 4:
            continue
        match = _find_best_line_window(
            asr_norm_tokens,
            reference_norm,
            start_token_idx=cursor_token_idx,
        )
        if match is None:
            continue
        word_start_idx = asr_token_word_indices[match.start_token_idx]
        word_end_idx = asr_token_word_indices[match.end_token_idx - 1] + 1
        if matched_regions and word_start_idx < matched_regions[-1][1]:
            continue
        matched_regions.append((word_start_idx, word_end_idx, reference_line, match.score))
        matched_ref_tokens += len(reference_line)
        line_scores.append(match.score)
        cursor_token_idx = match.end_token_idx

    min_line_count = max(3, min(len(reference_lines) // 5, 8))
    if len(matched_regions) < min_line_count:
        return None

    coverage_ratio = matched_ref_tokens / max(len(ref_tokens), 1)
    min_coverage_ratio = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_MIN_LINE_COVERAGE_RATIO",
        0.45,
        0.0,
    )
    if coverage_ratio < min_coverage_ratio:
        return None

    average_line_score = sum(line_scores) / max(len(line_scores), 1)
    min_average_line_score = _env_float(
        "TRANSCRIBE_LYRICS_REFERENCE_MIN_AVG_LINE_SCORE",
        0.67,
        0.0,
    )
    if average_line_score < min_average_line_score:
        return None

    merged_regions = _merge_matched_line_regions(matched_regions, asr_words)
    corrected: list[TranscriptWordPayload] = []
    cursor_word_idx = 0
    for match_idx, (word_start_idx, word_end_idx, reference_tokens) in enumerate(
        merged_regions
    ):
        if word_start_idx > cursor_word_idx:
            corrected.extend(asr_words[cursor_word_idx:word_start_idx])
        corrected.extend(
            _build_reference_words_with_asr_timing(
                reference_tokens,
                asr_words[word_start_idx:word_end_idx],
                id_prefix=f"lyrics-ref-block-{match_idx}",
            )
        )
        cursor_word_idx = word_end_idx
    if cursor_word_idx < len(asr_words):
        corrected.extend(asr_words[cursor_word_idx:])

    return _finalize_corrected_payload(payload, corrected)

def _align_reference_lyrics_by_sequence(
    payload: TranscriptPayload,
    *,
    duration_sec: float,
    ref_tokens: list[str],
) -> TranscriptPayload:
    asr_words = list(payload.words)
    ref_norm = [_normalize_token(token) for token in ref_tokens]
    asr_norm = [_normalize_token(word.text) for word in asr_words]
    asr_norm = [token for token in asr_norm if token]
    if len(asr_norm) < 20:
        return payload

    ratio = SequenceMatcher(None, asr_norm, ref_norm, autojunk=False).ratio()
    min_ratio = _env_float("TRANSCRIBE_LYRICS_REFERENCE_MIN_ALIGNMENT_RATIO", 0.38, 0.0)
    if ratio < min_ratio:
        return payload

    matcher = SequenceMatcher(
        None,
        [_normalize_token(word.text) for word in asr_words],
        ref_norm,
        autojunk=False,
    )
    blocks = [block for block in matcher.get_matching_blocks() if block.size > 0]
    if not blocks:
        return payload

    corrected: list[TranscriptWordPayload] = []
    prev_asr = 0
    prev_ref = 0
    previous_end = 0.0
    for block in blocks:
        next_start = (
            float(asr_words[block.a].start_sec) if block.a < len(asr_words) else duration_sec
        )
        if block.b > prev_ref:
            span_start, span_end = _region_span(
                asr_words,
                prev_asr,
                block.a,
                fallback_start=previous_end,
                fallback_end=next_start,
                duration_sec=duration_sec,
            )
            corrected.extend(
                _build_interpolated_words(
                    ref_tokens[prev_ref:block.b],
                    start_sec=span_start,
                    end_sec=span_end,
                    id_prefix=f"lyrics-ref-gap-{prev_ref}",
                )
            )
        for offset in range(block.size):
            asr_word = asr_words[block.a + offset]
            quality_score, quality_label = _lyrics_line_quality(ratio)
            corrected.append(
                TranscriptWordPayload(
                    id=asr_word.id,
                    text=ref_tokens[block.b + offset],
                    start_sec=float(asr_word.start_sec),
                    end_sec=float(asr_word.end_sec),
                    confidence=quality_score,
                    quality_score=quality_score,
                    quality_label=quality_label,
                    source_pass="manual",
                    speaker_id=asr_word.speaker_id,
                    speaker_label=asr_word.speaker_label,
                )
            )
        prev_asr = block.a + block.size
        prev_ref = block.b + block.size
        previous_end = float(asr_words[block.a + block.size - 1].end_sec)

    if prev_ref < len(ref_tokens):
        span_start, span_end = _region_span(
            asr_words,
            prev_asr,
            len(asr_words),
            fallback_start=previous_end,
            fallback_end=duration_sec,
            duration_sec=duration_sec,
        )
        corrected.extend(
            _build_interpolated_words(
                ref_tokens[prev_ref:],
                start_sec=span_start,
                end_sec=span_end,
                id_prefix=f"lyrics-ref-tail-{prev_ref}",
            )
        )

    return _finalize_corrected_payload(payload, corrected)


def align_reference_lyrics(
    payload: TranscriptPayload,
    reference: LyricsReference,
    *,
    duration_sec: float,
) -> TranscriptPayload:
    if not payload.words:
        return payload
    ref_tokens = _tokenize_lyrics(reference.plain_lyrics)
    if len(ref_tokens) < 20:
        return payload

    has_synced_reference = len(_parse_synced_lyric_lines(reference.synced_lyrics or "")) >= 6
    synced_aligned = _align_reference_lyrics_by_synced_lines(
        payload,
        reference,
        duration_sec=duration_sec,
    )
    if synced_aligned is not None:
        return synced_aligned
    if has_synced_reference:
        return payload

    line_aligned = _align_reference_lyrics_by_lines(
        payload,
        reference,
        ref_tokens=ref_tokens,
    )
    if line_aligned is not None:
        return line_aligned
    return _align_reference_lyrics_by_sequence(
        payload,
        duration_sec=duration_sec,
        ref_tokens=ref_tokens,
    )


def maybe_apply_reference_lyrics(
    payload: TranscriptPayload,
    *,
    filename: str,
    duration_sec: float,
) -> TranscriptPayload:
    reference = fetch_lyrics_reference(filename, duration_sec)
    if reference is None:
        return payload
    return align_reference_lyrics(payload, reference, duration_sec=duration_sec)
