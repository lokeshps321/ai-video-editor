from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .lyrics_reference_service import looks_like_duet_media, parse_duet_artists
from .transcription_service import (
    TranscriptPayload,
    TranscriptWordPayload,
    _env_bool,
    _env_float,
    _env_int,
    _normalize_detected_language,
    _normalize_language_code,
    _normalize_words,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None] | None

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


@dataclass(frozen=True)
class DiarizedEntry:
    transcript: str
    start_sec: float
    end_sec: float
    speaker_id: str


def diarization_enabled() -> bool:
    return _env_bool("TRANSCRIBE_DIARIZATION_ENABLED", True)


def _resolve_speaker_count(filename: str | None) -> int:
    default = _env_int("TRANSCRIBE_DUET_DEFAULT_SPEAKERS", 2, 1)
    if filename and looks_like_duet_media(filename):
        return max(default, 2)
    return max(default, 1)


def _resolve_sarvam_api_key() -> str | None:
    for key in (
        "TRANSCRIBE_SARVAM_API_KEY",
        "SARVAM_API_KEY",
        "SARVAM_SPEECH_TO_TEXT_API_KEY",
    ):
        value = (os.getenv(key, "") or "").strip()
        if value and not value.startswith("your_"):
            return value
    return None


def _resolve_sarvam_language_code(language_hint: str | None) -> str:
    normalized = _normalize_language_code(language_hint)
    mapping = {
        "en": "en-IN",
        "hi": "hi-IN",
        "kn": "kn-IN",
        "ta": "ta-IN",
        "te": "te-IN",
        "ml": "ml-IN",
        "mr": "mr-IN",
        "bn": "bn-IN",
        "gu": "gu-IN",
        "pa": "pa-IN",
        "or": "od-IN",
        "ur": "ur-IN",
    }
    if normalized and normalized in mapping:
        return mapping[normalized]
    return "en-IN"


def _parse_diarized_entries(payload: dict[str, object]) -> list[DiarizedEntry]:
    diarized = payload.get("diarized_transcript")
    if not isinstance(diarized, dict):
        return []
    entries_raw = diarized.get("entries")
    if not isinstance(entries_raw, list):
        return []
    entries: list[DiarizedEntry] = []
    for item in entries_raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("transcript") or "").strip()
        if not text:
            continue
        try:
            start_sec = float(item.get("start_time_seconds") or 0.0)
            end_sec = float(item.get("end_time_seconds") or start_sec)
        except (TypeError, ValueError):
            continue
        if end_sec <= start_sec:
            end_sec = start_sec + 0.05
        speaker_id = str(item.get("speaker_id") or "0").strip() or "0"
        entries.append(
            DiarizedEntry(
                transcript=text,
                start_sec=start_sec,
                end_sec=end_sec,
                speaker_id=f"speaker_{speaker_id}",
            )
        )
    return entries


def _segment_words(
    text: str,
    start_sec: float,
    end_sec: float,
    *,
    speaker_id: str,
) -> list[TranscriptWordPayload]:
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return []
    span = max(end_sec - start_sec, 0.05)
    step = max(span / max(len(tokens), 1), 0.05)
    words: list[TranscriptWordPayload] = []
    for idx, token in enumerate(tokens):
        word_start = start_sec + (idx * step)
        word_end = min(end_sec, word_start + max(0.05, step * 0.9))
        words.append(
            TranscriptWordPayload(
                id=str(uuid4()),
                text=token,
                start_sec=word_start,
                end_sec=word_end,
                confidence=None,
                speaker_id=speaker_id,
            )
        )
    return words


def diarized_entries_to_words(
    entries: list[DiarizedEntry],
    duration_sec: float,
) -> list[TranscriptWordPayload]:
    words: list[TranscriptWordPayload] = []
    for entry in entries:
        words.extend(
            _segment_words(
                entry.transcript,
                entry.start_sec,
                entry.end_sec,
                speaker_id=entry.speaker_id,
            )
        )
    return _normalize_words(words, duration_sec, apply_offset=False)


def _overlap_sec(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def assign_speaker_ids_to_words(
    words: list[TranscriptWordPayload],
    entries: list[DiarizedEntry],
) -> list[TranscriptWordPayload]:
    if not words or not entries:
        return words
    assigned: list[TranscriptWordPayload] = []
    for word in words:
        best_speaker: str | None = None
        best_overlap = 0.0
        for entry in entries:
            overlap = _overlap_sec(
                float(word.start_sec),
                float(word.end_sec),
                entry.start_sec,
                entry.end_sec,
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = entry.speaker_id
        if best_speaker is None:
            midpoint = (float(word.start_sec) + float(word.end_sec)) / 2.0
            for entry in entries:
                if entry.start_sec <= midpoint <= entry.end_sec:
                    best_speaker = entry.speaker_id
                    break
        assigned.append(
            TranscriptWordPayload(
                id=word.id,
                text=word.text,
                start_sec=word.start_sec,
                end_sec=word.end_sec,
                confidence=word.confidence,
                quality_score=word.quality_score,
                quality_label=word.quality_label,
                source_pass=word.source_pass,
                speaker_id=best_speaker or word.speaker_id,
                speaker_label=word.speaker_label,
            )
        )
    return assigned


def apply_speaker_labels(
    words: list[TranscriptWordPayload],
    *,
    primary_artist: str | None,
    featured_artist: str | None,
) -> list[TranscriptWordPayload]:
    speaker_ids = sorted({word.speaker_id for word in words if word.speaker_id})
    if not speaker_ids:
        return words
    label_map: dict[str, str] = {}
    if primary_artist and featured_artist:
        label_map[speaker_ids[0]] = primary_artist
        if len(speaker_ids) > 1:
            label_map[speaker_ids[1]] = featured_artist
    elif primary_artist:
        label_map[speaker_ids[0]] = primary_artist
    for idx, speaker_id in enumerate(speaker_ids):
        label_map.setdefault(speaker_id, f"Speaker {idx + 1}")

    labeled: list[TranscriptWordPayload] = []
    for word in words:
        speaker_id = word.speaker_id
        labeled.append(
            TranscriptWordPayload(
                id=word.id,
                text=word.text,
                start_sec=word.start_sec,
                end_sec=word.end_sec,
                confidence=word.confidence,
                quality_score=word.quality_score,
                quality_label=word.quality_label,
                source_pass=word.source_pass,
                speaker_id=speaker_id,
                speaker_label=label_map.get(speaker_id or "", word.speaker_label)
                if speaker_id
                else word.speaker_label,
            )
        )
    return labeled


def run_sarvam_batch_diarization(
    audio_path: str,
    duration_sec: float,
    *,
    language_hint: str | None = None,
    num_speakers: int | None = None,
    progress_callback: ProgressCallback = None,
) -> tuple[list[DiarizedEntry], str | None] | None:
    api_key = _resolve_sarvam_api_key()
    if not api_key:
        return None
    source = Path(audio_path).expanduser().resolve(strict=False)
    if not source.exists():
        return None

    try:
        from sarvamai import SarvamAI
    except Exception:
        logger.warning("[diarization] sarvamai SDK unavailable")
        return None

    model = (
        os.getenv("TRANSCRIBE_SARVAM_MODEL", "saaras:v3") or "saaras:v3"
    ).strip()
    mode = (os.getenv("TRANSCRIBE_SARVAM_MODE", "transcribe") or "transcribe").strip()
    timeout_sec = _env_float("TRANSCRIBE_DIARIZATION_TIMEOUT_SEC", 60.0, 10.0)
    poll_interval_sec = _env_float("TRANSCRIBE_DIARIZATION_POLL_SEC", 2.0, 0.5)
    speaker_count = num_speakers or _resolve_speaker_count(None)

    if progress_callback:
        progress_callback(32, "Identifying speakers in multi-voice audio")

    client = SarvamAI(api_subscription_key=api_key)
    try:
        job = client.speech_to_text_job.create_job(
            model=model,
            language_code=_resolve_sarvam_language_code(language_hint),
            mode=mode,
            with_diarization=True,
            num_speakers=speaker_count,
            with_timestamps=True,
        )
        job.upload_files(file_paths=[str(source)])
        job.start()
        job.wait_until_complete(
            timeout=int(timeout_sec),
            poll_interval=max(1, int(poll_interval_sec)),
        )

        with tempfile.TemporaryDirectory(prefix="sarvam-diar-") as tmp_dir:
            job.download_outputs(output_dir=tmp_dir)
            json_files = sorted(Path(tmp_dir).glob("*.json"))
            if not json_files:
                return None
            payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[diarization] Sarvam batch diarization failed: %s", exc)
        return None

    if not isinstance(payload, dict):
        return None
    entries = _parse_diarized_entries(payload)
    if not entries:
        return None
    language = _normalize_detected_language(payload.get("language_code"))
    if progress_callback:
        progress_callback(52, f"Detected {len({e.speaker_id for e in entries})} speakers")
    return entries, language


def maybe_enhance_duet_transcript(
    payload: TranscriptPayload,
    *,
    audio_path: str,
    duration_sec: float,
    filename: str | None,
    language_hint: str | None = None,
    progress_callback: ProgressCallback = None,
) -> TranscriptPayload:
    if not diarization_enabled():
        return payload
    if not filename or not looks_like_duet_media(filename):
        return payload
    if payload.is_mock or not payload.words:
        pass

    batch_result = run_sarvam_batch_diarization(
        audio_path,
        duration_sec,
        language_hint=language_hint or payload.language,
        num_speakers=_resolve_speaker_count(filename),
        progress_callback=progress_callback,
    )
    if batch_result is None:
        return payload

    entries, detected_language = batch_result
    primary_artist, featured_artist = parse_duet_artists(filename)

    diarized_words = diarized_entries_to_words(entries, duration_sec)
    if diarized_words:
        words = apply_speaker_labels(
            diarized_words,
            primary_artist=primary_artist,
            featured_artist=featured_artist,
        )
        text = " ".join(word.text for word in words)
        return TranscriptPayload(
            source="sarvam_diarized",
            language=detected_language or payload.language,
            text=text,
            words=words,
            is_mock=False,
        )

    if not payload.words:
        return payload

    tagged_words = assign_speaker_ids_to_words(payload.words, entries)
    tagged_words = apply_speaker_labels(
        tagged_words,
        primary_artist=primary_artist,
        featured_artist=featured_artist,
    )
    return TranscriptPayload(
        source=f"{payload.source}+diarized",
        language=payload.language or detected_language,
        text=" ".join(word.text for word in tagged_words),
        words=tagged_words,
        is_mock=False,
    )
