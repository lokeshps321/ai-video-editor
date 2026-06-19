from __future__ import annotations

import gc
import hashlib
import inspect
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concurrency gate – prevent multiple transcript generations from competing
# for memory at the same time. Unlike render jobs (which have a
# BoundedSemaphore), transcript generation previously had no gate at all.
# ---------------------------------------------------------------------------
_MAX_CONCURRENT_TRANSCRIPTS = max(
    1, int(os.getenv("MAX_CONCURRENT_TRANSCRIPT_JOBS", "1"))
)
_transcript_semaphore = threading.BoundedSemaphore(_MAX_CONCURRENT_TRANSCRIPTS)

from ..config import get_settings
from ..database import engine, get_session
from ..deps import get_current_user
from ..diarization_service import maybe_enhance_duet_transcript
from ..jobs import (
    create_job,
    find_recent_active_job,
    get_latest_job_event,
    set_job_status,
)
from ..lyrics_reference_service import (
    looks_like_duet_media,
    looks_like_song_media,
    maybe_apply_reference_lyrics,
)
from ..media_utils import probe_duration_seconds
from ..models import Job, MediaAsset, Project, Transcript
from ..schemas import (
    JobResponse,
    OperationPayload,
    TimelineState,
    TranscriptCutRequest,
    TranscriptCutResponse,
    TranscriptEditResponse,
    TranscriptGenerateRequest,
    TranscriptGenerateResponse,
    TranscriptRangeUpdateRequest,
    TranscriptRegion,
    TranscriptResponse,
    TranscriptWord,
    TranscriptWordPageResponse,
)
from ..storage import storage
from ..timeline_service import (
    apply_operation,
    get_timeline_row,
    load_timeline_state,
    save_timeline_state,
)
from ..transcription_service import (
    TranscriptPayload,
    TranscriptWordPayload,
    _detect_indic_script_languages,
    _normalize_detected_language,
    generate_transcript,
    infer_source_pass,
    sanitize_transcript_words,
    stabilize_song_transcript,
    trim_song_mode_to_manual_lyrics_span,
    trim_songlike_tail_hallucination,
)
from ..transliteration_service import contains_indic_script, transliterate_words

router = APIRouter(prefix="/api/v1/transcript", tags=["transcript"])
settings = get_settings()
_OPAQUE_MEDIA_FILENAME_RE = re.compile(
    r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_RELATED_MEDIA_DURATION_TOLERANCE_SEC = 1.5
_RELATED_MEDIA_MAX_CANDIDATES = 120
_RELATED_MEDIA_HASH_SAMPLE_BYTES = 256 * 1024
_ARABIC_SCRIPT_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
)


# ---------------------------------------------------------------------------
# Memory safety helpers
# ---------------------------------------------------------------------------
_MIN_AVAILABLE_MB = max(50, int(os.getenv("TRANSCRIBE_MIN_AVAILABLE_MB", "200")))


def _get_available_memory_mb() -> float | None:
    """Return available system memory in MB, or None if undetectable."""
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0  # kB -> MB
    except Exception:  # noqa: BLE001
        pass
    return None


def _check_memory_before_transcription() -> None:
    """Raise HTTPException if system memory is critically low."""
    available_mb = _get_available_memory_mb()
    if available_mb is not None and available_mb < _MIN_AVAILABLE_MB:
        gc.collect()
        # Try once more after GC
        available_mb = _get_available_memory_mb()
        if available_mb is not None and available_mb < _MIN_AVAILABLE_MB:
            logger.warning(
                "Transcript generation blocked: only %.0f MB available (minimum %d MB)",
                available_mb,
                _MIN_AVAILABLE_MB,
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Server memory is critically low ({available_mb:.0f} MB available). "
                    "Close other applications or wait for current tasks to finish before "
                    "generating a transcript."
                ),
            )


def _force_gc() -> None:
    """Aggressive garbage collection after transcription to release memory."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()


def _to_job_response(session: Session, job: Job) -> JobResponse:
    latest_event = get_latest_job_event(session, job.id)
    return JobResponse(
        id=job.id,
        project_id=job.project_id,
        kind=job.kind,
        status=job.status,
        progress=job.progress,
        stage=latest_event.stage if latest_event else None,
        message=latest_event.message if latest_event else None,
        output_path=job.output_path,
        error=job.error,
    )


def _transcript_generate_result_path(job_id: str) -> Path:
    folder = storage.tmp_root / "transcript-generate-jobs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{job_id}.json"


def _resolve_transcribe_chunk_seconds() -> float:
    raw = os.getenv("TRANSCRIBE_CHUNK_DURATION_SEC")
    try:
        value = float(raw) if raw is not None else 45.0
    except (TypeError, ValueError):
        value = 45.0
    return max(30.0, min(60.0, value))


def _resolve_transcribe_chunk_retries() -> int:
    raw = os.getenv("TRANSCRIBE_CHUNK_RETRIES")
    try:
        value = int(raw) if raw is not None else 2
    except (TypeError, ValueError):
        value = 2
    return max(0, min(5, value))


def _resolve_transcribe_chunk_bypass_max_duration_sec() -> float:
    raw = os.getenv("TRANSCRIBE_CHUNK_BYPASS_MAX_DURATION_SEC")
    try:
        value = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, value)


def _resolve_transcribe_chunk_overlap_sec() -> float:
    raw = os.getenv("TRANSCRIBE_CHUNK_OVERLAP_SEC")
    try:
        value = float(raw) if raw is not None else 3.0
    except (TypeError, ValueError):
        value = 3.0
    return max(0.0, min(8.0, value))


def _resolve_transcribe_chunk_parallelism() -> int:
    raw = os.getenv("TRANSCRIBE_CHUNK_PARALLELISM")
    try:
        value = int(raw) if raw is not None else 1
    except (TypeError, ValueError):
        value = 1
    return max(1, min(6, value))


def _normalize_transcript_mode(value: str | None) -> str:
    normalized = (value or "auto").strip().lower()
    if normalized in {"speech", "song"}:
        return normalized
    return "auto"


def _resolve_requested_transcript_mode(
    requested_mode: str | None,
    *,
    filename: str,
) -> str:
    normalized_mode = _normalize_transcript_mode(requested_mode)
    if normalized_mode != "auto":
        return normalized_mode
    if looks_like_song_media(filename):
        # Song uploads benefit more from fast speech-first ASR followed by lyrics
        # alignment than from the slower music-specific retry path.
        return "speech"
    return normalized_mode


@dataclass(frozen=True)
class _TranscriptGenerationStrategy:
    mode: str
    optimize_for_speed: bool
    bypass_max_duration_sec: float | None = None
    chunk_duration_sec: float | None = None
    chunk_overlap_sec: float | None = None
    chunk_parallelism: int | None = None
    skip_timestamp_refinement: bool = False
    skip_weak_region_retry: bool = False


def _resolve_transcript_generation_strategy(
    duration_sec: float, mode: str, *, song_like_media: bool = False
) -> _TranscriptGenerationStrategy:
    resolved_mode = _normalize_transcript_mode(mode)
    safe_duration = max(float(duration_sec), 0.0)
    optimize_for_speed = safe_duration <= 90.0
    skip_songlike_retry = resolved_mode == "song" or song_like_media

    if safe_duration <= 30.0:
        return _TranscriptGenerationStrategy(
            mode=resolved_mode,
            optimize_for_speed=optimize_for_speed,
            bypass_max_duration_sec=max(30.0, safe_duration),
            skip_timestamp_refinement=resolved_mode == "song",
            skip_weak_region_retry=skip_songlike_retry,
        )

    if safe_duration <= 90.0:
        return _TranscriptGenerationStrategy(
            mode=resolved_mode,
            optimize_for_speed=optimize_for_speed,
            bypass_max_duration_sec=0.0,
            chunk_duration_sec=30.0 if safe_duration <= 60.0 else 45.0,
            chunk_overlap_sec=2.5,
            chunk_parallelism=min(2, _resolve_transcribe_chunk_parallelism()),
            skip_timestamp_refinement=resolved_mode == "song",
            skip_weak_region_retry=skip_songlike_retry,
        )

    return _TranscriptGenerationStrategy(
        mode=resolved_mode,
        optimize_for_speed=False,
        skip_timestamp_refinement=resolved_mode == "song",
        skip_weak_region_retry=skip_songlike_retry,
    )


def _resolve_transcript_word_response_limit() -> int:
    raw = os.getenv("TRANSCRIPT_WORD_RESPONSE_LIMIT")
    try:
        value = int(raw) if raw is not None else 1200
    except (TypeError, ValueError):
        value = 1200
    return max(0, min(10000, value))


def _signal_number_from_returncode(returncode: int) -> int | None:
    if returncode < 0:
        return abs(returncode)
    if returncode > 128:
        return returncode - 128
    return None


def _looks_opaque_media_filename(filename: str) -> bool:
    stem = Path(str(filename or "")).stem.strip()
    if not stem:
        return False
    normalized = stem.replace("_", "-")
    if _OPAQUE_MEDIA_FILENAME_RE.fullmatch(normalized):
        return True
    compact = normalized.replace("-", "")
    return len(compact) >= 24 and compact.isalnum() and compact.lower() == compact


def _media_file_identity_signature(path: Path) -> tuple[int, str] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    size = int(stat.st_size)
    if size <= 0:
        return None

    sample_bytes = min(_RELATED_MEDIA_HASH_SAMPLE_BYTES, size)
    digest = hashlib.sha1()
    digest.update(str(size).encode("ascii"))

    try:
        with path.open("rb") as stream:
            if size <= sample_bytes * 3:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            else:
                digest.update(stream.read(sample_bytes))

                middle_offset = max(0, (size // 2) - (sample_bytes // 2))
                stream.seek(middle_offset)
                digest.update(stream.read(sample_bytes))

                stream.seek(max(0, size - sample_bytes))
                digest.update(stream.read(sample_bytes))
    except OSError:
        return None

    return size, digest.hexdigest()


def _related_media_assets(session: Session, asset: MediaAsset) -> list[MediaAsset]:
    filename = str(asset.filename or "").strip()
    if not filename or not _looks_opaque_media_filename(filename):
        return []

    candidates_by_id: dict[str, MediaAsset] = {}

    basename = Path(filename).name
    basename_matches = session.exec(
        select(MediaAsset)
        .where(MediaAsset.id != asset.id)
        .where(MediaAsset.storage_path.like(f"%/{basename}"))
        .order_by(MediaAsset.created_at.desc())
    ).all()
    for candidate in basename_matches:
        candidates_by_id[candidate.id] = candidate

    source_path = Path(storage.resolve_upload_asset(asset.storage_path))
    source_signature = _media_file_identity_signature(source_path)
    if source_signature is None:
        return list(candidates_by_id.values())

    source_size, _source_hash = source_signature
    query = (
        select(MediaAsset)
        .where(MediaAsset.id != asset.id)
        .where(MediaAsset.media_type == asset.media_type)
        .order_by(MediaAsset.created_at.desc())
    )
    duration_sec = float(asset.duration_sec or 0.0)
    if duration_sec > 0:
        lower = max(0.0, duration_sec - _RELATED_MEDIA_DURATION_TOLERANCE_SEC)
        upper = duration_sec + _RELATED_MEDIA_DURATION_TOLERANCE_SEC
        query = query.where(
            MediaAsset.duration_sec >= lower, MediaAsset.duration_sec <= upper
        )

    signature_cache: dict[str, tuple[int, str] | None] = {
        str(source_path): source_signature
    }
    checked = 0
    for candidate in session.exec(query):
        if checked >= _RELATED_MEDIA_MAX_CANDIDATES:
            break
        checked += 1
        candidate_path = Path(storage.resolve_upload_asset(candidate.storage_path))
        try:
            if int(candidate_path.stat().st_size) != source_size:
                continue
        except OSError:
            continue

        cache_key = str(candidate_path)
        candidate_signature = signature_cache.get(cache_key)
        if candidate_signature is None:
            candidate_signature = _media_file_identity_signature(candidate_path)
            signature_cache[cache_key] = candidate_signature
        if candidate_signature != source_signature:
            continue
        candidates_by_id[candidate.id] = candidate

    return list(candidates_by_id.values())


def _lyrics_reference_filename_hint(session: Session, asset: MediaAsset) -> str:
    filename = str(asset.filename or "").strip()
    if not filename or not _looks_opaque_media_filename(filename):
        return filename

    related_assets = _related_media_assets(session, asset)
    best_filename = filename
    best_score = (-1, -1, -1)
    for candidate in related_assets:
        candidate_name = str(candidate.filename or "").strip()
        if not candidate_name or _looks_opaque_media_filename(candidate_name):
            continue
        stem_length = len(Path(candidate_name).stem)
        score = (
            1 if candidate.project_id == asset.project_id else 0,
            stem_length,
            len(candidate_name),
        )
        if score > best_score:
            best_filename = candidate_name
            best_score = score
    return best_filename


def _related_library_transcript(
    session: Session,
    asset: MediaAsset,
    *,
    requested_language: str | None,
) -> Transcript | None:
    # Allow disabling cross-library transcript sharing
    if not _env_bool("TRANSCRIBE_REUSE_CROSS_LIBRARY", False):
        return None
    filename = str(asset.filename or "").strip()
    if not filename or not _looks_opaque_media_filename(filename):
        return None

    asset_duration = float(asset.duration_sec or 0.0)
    related_assets = _related_media_assets(session, asset)

    if asset_duration > 0:
        related_assets = [
            candidate
            for candidate in related_assets
            if candidate.duration_sec is None
            or abs(float(candidate.duration_sec) - asset_duration) <= 1.5
        ]
    if not related_assets:
        return None

    candidate_asset_ids = [candidate.id for candidate in related_assets]
    candidates = session.exec(
        select(Transcript)
        .where(Transcript.asset_id.in_(candidate_asset_ids))
        .order_by(Transcript.created_at.desc())
    ).all()

    best: Transcript | None = None
    best_score = (-1, -1, -1, -1)
    for candidate in candidates:
        if candidate.is_mock:
            continue
        existing_language = _normalize_requested_language(candidate.language)
        if requested_language is not None and requested_language != existing_language:
            continue
        items = _load_raw_items(candidate)
        if not items:
            continue
        if _transcript_has_library_blocking_edits(candidate, items=items):
            continue
        score = _transcript_reuse_score(candidate, items=items, filename=asset.filename)
        if score is None:
            continue
        if score > best_score:
            best = candidate
            best_score = score
    return best


def _allow_cached_transcript_reuse(requested_language: str | None) -> bool:
    if requested_language is not None:
        return True
    return _env_bool("TRANSCRIBE_REUSE_AUTO_LANGUAGE_EXISTING", False)


def _manual_transcript_source(source: str | None) -> str:
    source_text = str(source or "").strip()
    if source_text.startswith("manual_edit"):
        return source_text
    if not source_text:
        return "manual_edit"
    return f"manual_edit:{source_text}"


def _transcript_has_library_blocking_edits(
    transcript: Transcript, *, items: list[dict[str, object]]
) -> bool:
    source_text = str(transcript.source or "").strip().lower()
    if source_text.startswith("manual_edit"):
        return True

    created_at = transcript.created_at
    updated_at = transcript.updated_at
    if (
        created_at is not None
        and updated_at is not None
        and (updated_at - created_at).total_seconds() > 1.0
    ):
        return True

    for item in items:
        if str(item.get("source_pass") or "").strip().lower() == "manual":
            return True
    return False


def _transcript_reuse_score(
    transcript: Transcript,
    *,
    items: list[dict[str, object]],
    filename: str,
) -> tuple[int, int, int, int] | None:
    source_text = str(transcript.source or "").lower()
    lyrics_ref = "lyrics_ref" in source_text
    manual_or_synced = "manual" in source_text or "sync" in source_text
    quality_values: list[float] = []
    weak_count = 0
    for item in items:
        raw_score = item.get("quality_score")
        try:
            if raw_score is not None:
                quality_values.append(float(raw_score))
        except (TypeError, ValueError):
            pass
        if str(item.get("quality_label") or "").strip().lower() == "weak":
            weak_count += 1
    avg_quality = sum(quality_values) / len(quality_values) if quality_values else None
    weak_ratio = weak_count / max(len(items), 1)

    if looks_like_song_media(filename):
        if len(items) < 20 and not lyrics_ref:
            return None
        if avg_quality is None and not lyrics_ref:
            return None
        if avg_quality is not None and avg_quality < 0.72:
            return None
        if weak_ratio > 0.18 and not lyrics_ref:
            return None

    return (
        1 if lyrics_ref else 0,
        1 if manual_or_synced else 0,
        int(round((avg_quality or 0.0) * 1000)),
        len(items),
    )


def _looks_like_sigkill_error(message: str) -> bool:
    text = message.lower()
    markers = (
        "sigkill",
        "signal 9",
        "return code -9",
        "returncode -9",
        "exit status 137",
        "killed by the os",
    )
    return any(marker in text for marker in markers)


def _format_ffmpeg_failure(
    command: list[str], returncode: int, stderr_output: str
) -> str:
    rendered_command = " ".join(shlex.quote(part) for part in command)
    signal_number = _signal_number_from_returncode(returncode)
    sigkill_number = int(getattr(signal, "SIGKILL", 9))
    if signal_number == sigkill_number:
        return (
            "ffmpeg was killed by the OS (SIGKILL / likely out-of-memory) while preparing transcript audio chunks. "
            "Try shorter clips, fewer concurrent jobs, or set TRANSCRIBE_BACKEND=groq.\n"
            f"command: {rendered_command}\n"
            f"stderr: {stderr_output.strip()}"
        )
    return (
        f"ffmpeg failed ({returncode}) while preparing transcript audio chunks.\n"
        f"command: {rendered_command}\n"
        f"stderr: {stderr_output.strip()}"
    )


def _humanize_transcript_runtime_error(exc: RuntimeError) -> str:
    raw = str(exc)
    if _looks_like_sigkill_error(raw):
        return (
            "Transcript generation was killed by the OS (SIGKILL / likely out-of-memory). "
            "The service retried with lower-overhead settings when possible. "
            "Try shorter clips, reduce concurrent jobs, or set TRANSCRIBE_BACKEND=groq."
        )
    return raw


def _effective_word_limit(requested_limit: int | None) -> int | None:
    if requested_limit is not None:
        return (
            None if requested_limit <= 0 else max(1, min(10000, int(requested_limit)))
        )
    configured = _resolve_transcript_word_response_limit()
    return None if configured <= 0 else configured


def _resolve_transcript_weak_retry_max_regions() -> int:
    raw = os.getenv("TRANSCRIPT_WEAK_RETRY_MAX_REGIONS")
    try:
        value = int(raw) if raw is not None else 6
    except (TypeError, ValueError):
        value = 6
    return max(0, min(12, value))


def _resolve_transcript_weak_retry_pad_sec() -> float:
    raw = os.getenv("TRANSCRIPT_WEAK_RETRY_PAD_SEC")
    try:
        value = float(raw) if raw is not None else 1.2
    except (TypeError, ValueError):
        value = 1.2
    return max(0.0, min(4.0, value))


def _resolve_transcript_cut_weak_region_safety_sec() -> float:
    raw = os.getenv("TRANSCRIPT_CUT_WEAK_REGION_SAFETY_SEC")
    try:
        value = float(raw) if raw is not None else 0.12
    except (TypeError, ValueError):
        value = 0.12
    return max(0.0, min(0.5, value))


def _transcript_progress_stage(progress: int, message: str | None) -> str:
    normalized = (message or "").strip().lower()
    if not normalized:
        return "transcribe"
    if "reusing" in normalized:
        return "reuse"
    if "timeline" in normalized:
        return "timeline"
    if "weak transcript region" in normalized:
        return "weak_retry"
    if "refining word timestamps" in normalized:
        return "refine"
    if "matching reference lyrics" in normalized:
        return "lyrics"
    if "speech recognition" in normalized or "completed chunk" in normalized:
        return "recognize"
    if "audio chunk" in normalized or "preparing audio" in normalized:
        return "prepare_audio"
    if "strategy" in normalized or "mode" in normalized or progress <= 12:
        return "prepare"
    return "transcribe"


def _payload_items(payload: TranscriptPayload) -> list[dict[str, object]]:
    default_source_pass = infer_source_pass(payload.source)
    return [
        {
            "id": word.id,
            "text": word.text,
            "start_sec": float(word.start_sec),
            "end_sec": float(word.end_sec),
            "confidence": word.confidence,
            "quality_score": word.quality_score,
            "quality_label": word.quality_label,
            "source_pass": word.source_pass or default_source_pass,
        }
        for word in payload.words
    ]


def _words_in_region_by_ids(
    items: list[dict[str, object]], word_ids: set[str]
) -> list[TranscriptWordPayload]:
    if not word_ids:
        return []
    words = []
    for item in items:
        word = _word_payload_from_item(item)
        if word is not None and word.id in word_ids:
            words.append(word)
    return sorted(words, key=lambda item: (float(item.start_sec), float(item.end_sec)))


def _overlaps_region(
    word: TranscriptWordPayload, start_sec: float, end_sec: float
) -> bool:
    return float(word.end_sec) > start_sec and float(word.start_sec) < end_sec


def _rebase_words_for_quality(
    words: list[TranscriptWordPayload],
    *,
    start_sec: float,
    end_sec: float,
) -> list[TranscriptWordPayload]:
    span_sec = max(float(end_sec) - float(start_sec), 0.1)
    rebased: list[TranscriptWordPayload] = []
    for word in words:
        local_start = max(0.0, min(float(word.start_sec) - float(start_sec), span_sec))
        local_end = max(0.0, min(float(word.end_sec) - float(start_sec), span_sec))
        if local_end <= local_start:
            local_end = min(span_sec, local_start + 0.05)
        rebased.append(
            TranscriptWordPayload(
                id=word.id,
                text=word.text,
                start_sec=round(local_start, 3),
                end_sec=round(local_end, 3),
                confidence=word.confidence,
                quality_score=word.quality_score,
                quality_label=word.quality_label,
                source_pass=word.source_pass,
            )
        )
    return rebased


def _score_retry_region(
    words: list[TranscriptWordPayload],
    *,
    start_sec: float,
    end_sec: float,
) -> tuple[int, int, float, int]:
    if not words:
        return (-1, -1, -1.0, -1)
    annotated = _annotate_word_quality(
        _rebase_words_for_quality(words, start_sec=start_sec, end_sec=end_sec),
        max(float(end_sec) - float(start_sec), 0.1),
    )
    trusted_count = sum(1 for word in annotated if word.quality_label == "trusted")
    weak_count = sum(1 for word in annotated if word.quality_label == "weak")
    avg_score = sum(float(word.quality_score or 0.0) for word in annotated) / max(
        len(annotated), 1
    )
    return (trusted_count, -weak_count, round(avg_score, 3), len(annotated))


def _merge_retried_region_items(
    items: list[dict[str, object]],
    *,
    region_word_ids: set[str],
    replacement_words: list[TranscriptWordPayload],
) -> list[dict[str, object]]:
    blank_items = [item for item in items if _is_blank_region(item)]
    retained_words = [
        item
        for item in items
        if not _is_blank_region(item)
        and str(item.get("id") or "") not in region_word_ids
    ]
    return _sort_items(
        [
            *retained_words,
            *[_serialize_word(word) for word in replacement_words],
            *blank_items,
        ]
    )


def _retry_weak_regions_in_items(
    source_path: str,
    duration_sec: float,
    *,
    items: list[dict[str, object]],
    language_hint: str | None,
    prompt: str | None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> list[dict[str, object]]:
    if not _env_bool("TRANSCRIPT_WEAK_REGION_RETRY_ENABLED", True):
        return items

    max_regions = _resolve_transcript_weak_retry_max_regions()
    if max_regions <= 0:
        return items

    current_items = _sort_items(list(items))
    _stored_items, _words, _text, regions = _materialize_transcript_items(
        current_items, duration_sec
    )
    weak_regions = [region for region in regions if region.status == "weak"]
    if not weak_regions:
        return current_items

    retry_pad_sec = _resolve_transcript_weak_retry_pad_sec()
    temp_root = Path(settings.tmp_dir) / f"transcript-weak-retry-{uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        for region_index, region in enumerate(weak_regions[:max_regions], start=1):
            region_word_ids = {word_id for word_id in region.word_ids if word_id}
            current_region_words = _words_in_region_by_ids(
                current_items, region_word_ids
            )
            if not current_region_words:
                continue

            region_start = float(region.start_sec)
            region_end = float(region.end_sec)
            window_start = max(0.0, region_start - retry_pad_sec)
            window_end = min(float(duration_sec), region_end + retry_pad_sec)
            window_duration = max(window_end - window_start, 0.1)
            chunk_path = temp_root / f"weak_region_{region_index:02d}.wav"

            if progress_callback:
                progress_callback(
                    min(95, 91 + region_index),
                    f"Retrying weak transcript region {region_index}/{min(len(weak_regions), max_regions)}",
                )

            try:
                _extract_audio_chunk(
                    source_path, window_start, window_duration, chunk_path
                )
                retry_payload = _call_with_signature_compat(
                    _call_generate_transcript_compatible,
                    str(chunk_path),
                    window_duration,
                    language_hint=language_hint,
                    allow_mock_fallback=False,
                    fast_mode=False,
                    prompt=prompt,
                )
            except Exception as exc:
                logger.warning(
                    "Weak-region transcript retry failed for region %s: %s",
                    region_index,
                    exc,
                )
                continue
            finally:
                chunk_path.unlink(missing_ok=True)

            retry_region_words: list[TranscriptWordPayload] = []
            for retry_word in retry_payload.words:
                shifted = TranscriptWordPayload(
                    id=str(uuid4()),
                    text=retry_word.text,
                    start_sec=round(float(retry_word.start_sec) + window_start, 3),
                    end_sec=round(float(retry_word.end_sec) + window_start, 3),
                    confidence=retry_word.confidence,
                    quality_score=retry_word.quality_score,
                    quality_label=retry_word.quality_label,
                    source_pass="retry",
                )
                if _overlaps_region(shifted, region_start, region_end):
                    retry_region_words.append(shifted)

            if not retry_region_words:
                continue

            if _score_retry_region(
                retry_region_words, start_sec=region_start, end_sec=region_end
            ) <= _score_retry_region(
                current_region_words, start_sec=region_start, end_sec=region_end
            ):
                continue

            current_items = _merge_retried_region_items(
                current_items,
                region_word_ids=region_word_ids,
                replacement_words=retry_region_words,
            )

        return current_items
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _extract_audio_chunk(
    source_path: str, start_sec: float, duration_sec: float, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        settings.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{max(0.0, start_sec):.3f}",
        "-t",
        f"{max(0.02, duration_sec):.3f}",
        "-i",
        source_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            _format_ffmpeg_failure(command, process.returncode, process.stderr or "")
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            "ffmpeg produced no audio output while preparing transcript chunks"
        )


def _kwargs_matching_signature(
    func: Callable[..., object],
    kwargs: dict[str, object],
) -> dict[str, object]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return kwargs
    allowed = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in allowed}


def _call_with_signature_compat(
    func: Callable[..., TranscriptPayload],
    *args: object,
    **kwargs: object,
) -> TranscriptPayload:
    try:
        return func(*args, **kwargs)
    except TypeError:
        filtered = _kwargs_matching_signature(func, kwargs)
        if filtered == kwargs:
            raise
        return func(*args, **filtered)


def _call_generate_transcript_compatible(
    source_path: str,
    duration_sec: float,
    *,
    language_hint: str | None,
    allow_mock_fallback: bool,
    fast_mode: bool,
    prompt: str | None,
    translate_to_english: bool | None = None,
    mode: str | None = None,
    optimize_for_speed: bool | None = None,
    filename: str | None = None,
) -> TranscriptPayload:
    full_kwargs: dict[str, object] = {
        "language_hint": language_hint,
        "allow_mock_fallback": allow_mock_fallback,
        "fast_mode": fast_mode,
        "prompt": prompt,
        "translate_to_english": translate_to_english,
    }
    if mode is not None:
        full_kwargs["mode"] = mode
    if optimize_for_speed is not None:
        full_kwargs["optimize_for_speed"] = optimize_for_speed
    if filename is not None:
        full_kwargs["filename"] = filename
    return _call_with_signature_compat(
        generate_transcript,
        source_path,
        duration_sec,
        **full_kwargs,
    )


def _transcribe_chunk_payload(
    chunk_path: str,
    chunk_duration: float,
    *,
    retry_count: int,
    language_hint: str | None,
    fast_mode: bool,
    prompt: str | None,
    chunk_index: int,
    chunk_count: int,
    translate_to_english: bool | None = None,
    mode: str | None = None,
    optimize_for_speed: bool | None = None,
) -> TranscriptPayload:
    chunk_payload: TranscriptPayload | None = None
    last_error: Exception | None = None
    max_attempts = retry_count + 1
    for attempt in range(1, max_attempts + 1):
        try:
            chunk_call_kwargs: dict[str, object] = {
                "language_hint": language_hint,
                "allow_mock_fallback": False,
                "fast_mode": fast_mode,
                "prompt": prompt,
            }
            if translate_to_english is not None:
                chunk_call_kwargs["translate_to_english"] = translate_to_english
            if mode is not None:
                chunk_call_kwargs["mode"] = mode
            if optimize_for_speed is not None:
                chunk_call_kwargs["optimize_for_speed"] = optimize_for_speed
            chunk_payload = _call_with_signature_compat(
                _call_generate_transcript_compatible,
                chunk_path,
                chunk_duration,
                **chunk_call_kwargs,
            )
            break
        except RuntimeError as exc:
            last_error = exc
            if attempt >= max_attempts:
                break

    if chunk_payload is None:
        if last_error is not None:
            raise RuntimeError(
                f"Chunk {chunk_index + 1}/{chunk_count} failed after {max_attempts} attempts: {last_error}"
            ) from last_error
        raise RuntimeError(
            f"Chunk {chunk_index + 1}/{chunk_count} failed with unknown error"
        )
    return chunk_payload


def _generate_transcript_payload_chunked(
    source_path: str,
    duration_sec: float,
    *,
    language_hint: str | None,
    allow_mock_fallback: bool | None = None,
    fast_mode: bool,
    prompt: str | None,
    progress_callback: Callable[[int, str], None] | None = None,
    translate_to_english: bool | None = None,
    mode: str | None = None,
    optimize_for_speed: bool | None = None,
    bypass_max_duration_sec_override: float | None = None,
    chunk_duration_sec_override: float | None = None,
    chunk_overlap_sec_override: float | None = None,
    chunk_parallelism_override: int | None = None,
    filename: str | None = None,
) -> TranscriptPayload:
    safe_duration = max(float(duration_sec), 0.1)
    allow_mock = (
        _env_bool("TRANSCRIBE_ALLOW_MOCK_FALLBACK", False)
        if allow_mock_fallback is None
        else bool(allow_mock_fallback)
    )
    bypass_max_duration_sec = (
        _resolve_transcribe_chunk_bypass_max_duration_sec()
        if bypass_max_duration_sec_override is None
        else max(0.0, float(bypass_max_duration_sec_override))
    )
    if bypass_max_duration_sec > 0 and safe_duration <= bypass_max_duration_sec:
        if progress_callback:
            progress_callback(14, "Preparing audio for full-file transcription")
            progress_callback(28, "Running speech recognition on full clip")
        full_file_kwargs: dict[str, object] = {
            "language_hint": language_hint,
            "allow_mock_fallback": allow_mock,
            "fast_mode": fast_mode,
            "prompt": prompt,
        }
        if translate_to_english is not None:
            full_file_kwargs["translate_to_english"] = translate_to_english
        if mode is not None:
            full_file_kwargs["mode"] = mode
        if optimize_for_speed is not None:
            full_file_kwargs["optimize_for_speed"] = optimize_for_speed
        if filename is not None:
            full_file_kwargs["filename"] = filename
        return _call_with_signature_compat(
            _call_generate_transcript_compatible,
            source_path,
            safe_duration,
            **full_file_kwargs,
        )

    chunk_sec = (
        _resolve_transcribe_chunk_seconds()
        if chunk_duration_sec_override is None
        else max(30.0, min(60.0, float(chunk_duration_sec_override)))
    )
    overlap_sec = (
        _resolve_transcribe_chunk_overlap_sec()
        if chunk_overlap_sec_override is None
        else max(0.0, min(8.0, float(chunk_overlap_sec_override)))
    )
    chunk_count = max(1, int(math.ceil(safe_duration / chunk_sec)))
    retry_count = _resolve_transcribe_chunk_retries()
    temp_root = Path(settings.tmp_dir) / f"transcript_chunks_{uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=True)

    all_words: list[TranscriptWordPayload] = []
    source_values: list[str] = []
    detected_language: str | None = None
    chunk_languages: list[
        str
    ] = []  # collect all chunk language detections for majority vote
    mock_chunks = 0
    chunk_specs: list[dict[str, object]] = []

    try:
        for chunk_index in range(chunk_count):
            core_start = float(chunk_index) * chunk_sec
            core_end = min(safe_duration, core_start + chunk_sec)
            if core_end - core_start <= 0.02:
                continue
            chunk_start = max(0.0, core_start - overlap_sec)
            chunk_end = min(safe_duration, core_end + overlap_sec)
            chunk_duration = max(0.0, chunk_end - chunk_start)
            chunk_path = temp_root / f"chunk_{chunk_index:04d}.wav"
            if progress_callback:
                progress_callback(
                    max(
                        14,
                        int(round(14 + ((chunk_index + 1) / max(chunk_count, 1)) * 14)),
                    ),
                    f"Preparing audio chunk {chunk_index + 1}/{chunk_count}",
                )

            try:
                _extract_audio_chunk(
                    source_path, chunk_start, chunk_duration, chunk_path
                )
            except RuntimeError as exc:
                if progress_callback:
                    progress_callback(
                        18,
                        "Chunk extraction unavailable, falling back to full-file transcription",
                    )
                try:
                    if progress_callback:
                        progress_callback(28, "Running speech recognition on full clip")
                    fallback_kwargs: dict[str, object] = {
                        "language_hint": language_hint,
                        "allow_mock_fallback": allow_mock,
                        "fast_mode": fast_mode,
                        "prompt": prompt,
                    }
                    if translate_to_english is not None:
                        fallback_kwargs["translate_to_english"] = translate_to_english
                    if mode is not None:
                        fallback_kwargs["mode"] = mode
                    if optimize_for_speed is not None:
                        fallback_kwargs["optimize_for_speed"] = optimize_for_speed
                    return _call_with_signature_compat(
                        _call_generate_transcript_compatible,
                        source_path,
                        safe_duration,
                        **fallback_kwargs,
                    )
                except RuntimeError as full_exc:
                    if _looks_like_sigkill_error(str(exc)) or _looks_like_sigkill_error(
                        str(full_exc)
                    ):
                        raise RuntimeError(
                            "Transcript generation hit OS memory limits (SIGKILL) during chunk extraction and fallback."
                        ) from full_exc
                    raise RuntimeError(
                        f"Failed to extract transcript chunk {chunk_index + 1} and full-file fallback failed: {full_exc}"
                    ) from exc

            chunk_specs.append(
                {
                    "chunk_index": chunk_index,
                    "chunk_path": chunk_path,
                    "chunk_start": chunk_start,
                    "chunk_duration": chunk_duration,
                    "core_start": core_start,
                    "core_end": core_end,
                }
            )

        if not chunk_specs:
            raise RuntimeError("Transcript generation returned no chunks")

        parallelism = min(
            chunk_parallelism_override
            if chunk_parallelism_override is not None
            else _resolve_transcribe_chunk_parallelism(),
            len(chunk_specs),
        )
        completed_payloads: dict[int, TranscriptPayload] = {}

        if progress_callback:
            progress_callback(
                32,
                f"Running speech recognition on {len(chunk_specs)} chunk{'s' if len(chunk_specs) != 1 else ''}",
            )

        if parallelism <= 1:
            for completed, spec in enumerate(chunk_specs, start=1):
                chunk_index = int(spec["chunk_index"])
                payload = _transcribe_chunk_payload(
                    str(spec["chunk_path"]),
                    float(spec["chunk_duration"]),
                    retry_count=retry_count,
                    language_hint=language_hint,
                    fast_mode=fast_mode,
                    prompt=prompt,
                    chunk_index=chunk_index,
                    chunk_count=len(chunk_specs),
                    translate_to_english=translate_to_english,
                    mode=mode,
                    optimize_for_speed=optimize_for_speed,
                )
                completed_payloads[chunk_index] = payload
                if progress_callback:
                    progress_callback(
                        max(
                            32,
                            int(
                                round(32 + (completed / max(len(chunk_specs), 1)) * 42)
                            ),
                        ),
                        f"Completed chunk {completed}/{len(chunk_specs)}",
                    )
        else:
            with ThreadPoolExecutor(
                max_workers=parallelism,
                thread_name_prefix="transcript-chunk",
            ) as executor:
                futures = {
                    executor.submit(
                        _transcribe_chunk_payload,
                        str(spec["chunk_path"]),
                        float(spec["chunk_duration"]),
                        retry_count=retry_count,
                        language_hint=language_hint,
                        fast_mode=fast_mode,
                        prompt=prompt,
                        chunk_index=int(spec["chunk_index"]),
                        chunk_count=len(chunk_specs),
                        translate_to_english=translate_to_english,
                        mode=mode,
                        optimize_for_speed=optimize_for_speed,
                    ): spec
                    for spec in chunk_specs
                }
                completed = 0
                while futures:
                    done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        spec = futures.pop(future)
                        try:
                            payload = future.result()
                        except Exception:
                            for pending in futures:
                                pending.cancel()
                            raise
                        chunk_index = int(spec["chunk_index"])
                        completed_payloads[chunk_index] = payload
                        completed += 1
                        if progress_callback:
                            progress_callback(
                                max(
                                    32,
                                    int(
                                        round(
                                            32
                                            + (completed / max(len(chunk_specs), 1))
                                            * 42
                                        )
                                    ),
                                ),
                                f"Completed chunk {completed}/{len(chunk_specs)}",
                            )

        for spec in chunk_specs:
            chunk_index = int(spec["chunk_index"])
            chunk_start = float(spec["chunk_start"])
            core_start = float(spec["core_start"])
            core_end = float(spec["core_end"])
            chunk_path = Path(spec["chunk_path"])
            chunk_payload = completed_payloads[chunk_index]

            if chunk_payload.language:
                chunk_languages.append(chunk_payload.language)
            source_values.append(chunk_payload.source)
            if chunk_payload.is_mock:
                mock_chunks += 1

            for word in chunk_payload.words:
                start_sec = max(
                    0.0, min(float(word.start_sec) + chunk_start, safe_duration)
                )
                end_sec = max(
                    start_sec + 0.001,
                    min(float(word.end_sec) + chunk_start, safe_duration),
                )
                center_sec = (start_sec + end_sec) / 2.0
                core_min = core_start if chunk_index == 0 else core_start - 0.001
                core_max = (
                    core_end if chunk_index == chunk_count - 1 else core_end + 0.001
                )
                if center_sec < core_min or center_sec > core_max:
                    continue
                word_id = (
                    word.id
                    if chunk_count == 1
                    else f"chunk{chunk_index + 1}-{uuid4().hex}"
                )
                all_words.append(
                    TranscriptWordPayload(
                        id=word_id,
                        text=word.text,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        confidence=word.confidence,
                        quality_score=word.quality_score,
                        quality_label=word.quality_label,
                        source_pass=word.source_pass,
                    )
                )

            if chunk_path.exists():
                chunk_path.unlink()
            gc.collect()

        if not all_words:
            raise RuntimeError("Transcript generation returned no words")

        all_words.sort(
            key=lambda item: (float(item.start_sec), float(item.end_sec), item.id)
        )
        transcript_text = " ".join(word.text for word in all_words).strip()
        source_value = source_values[0] if source_values else "chunked"
        is_mock = mock_chunks > 0 and mock_chunks == chunk_count

        # Majority-vote across all chunk language detections.
        # "First wins" caused the first chunk (often a silent/ambiguous intro)
        # to lock in a wrong language (e.g. Tamil instead of Kannada).
        if chunk_languages:
            from collections import Counter

            lang_counts = Counter(chunk_languages)
            detected_language = lang_counts.most_common(1)[0][0]
            logger.debug(
                "Language detection vote: %s -> '%s'",
                dict(lang_counts),
                detected_language,
            )

        return TranscriptPayload(
            source=f"chunked:{source_value}",
            language=detected_language,
            text=transcript_text,
            words=all_words,
            is_mock=is_mock,
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        gc.collect()


def _call_generate_transcript_payload_chunked_compatible(
    source_path: str,
    duration_sec: float,
    *,
    language_hint: str | None,
    allow_mock_fallback: bool | None = None,
    fast_mode: bool,
    prompt: str | None,
    progress_callback: Callable[[int, str], None] | None = None,
    translate_to_english: bool | None = None,
    mode: str | None = None,
    optimize_for_speed: bool | None = None,
    bypass_max_duration_sec_override: float | None = None,
    chunk_duration_sec_override: float | None = None,
    chunk_overlap_sec_override: float | None = None,
    chunk_parallelism_override: int | None = None,
    filename: str | None = None,
) -> TranscriptPayload:
    full_kwargs: dict[str, object] = {
        "language_hint": language_hint,
        "allow_mock_fallback": allow_mock_fallback,
        "fast_mode": fast_mode,
        "prompt": prompt,
        "progress_callback": progress_callback,
    }
    if translate_to_english is not None:
        full_kwargs["translate_to_english"] = translate_to_english
    if mode is not None:
        full_kwargs["mode"] = mode
    if optimize_for_speed is not None:
        full_kwargs["optimize_for_speed"] = optimize_for_speed
    if filename is not None:
        full_kwargs["filename"] = filename
    if bypass_max_duration_sec_override is not None:
        full_kwargs["bypass_max_duration_sec_override"] = (
            bypass_max_duration_sec_override
        )
    if chunk_duration_sec_override is not None:
        full_kwargs["chunk_duration_sec_override"] = chunk_duration_sec_override
    if chunk_overlap_sec_override is not None:
        full_kwargs["chunk_overlap_sec_override"] = chunk_overlap_sec_override
    if chunk_parallelism_override is not None:
        full_kwargs["chunk_parallelism_override"] = chunk_parallelism_override
    return _call_with_signature_compat(
        _generate_transcript_payload_chunked,
        source_path,
        duration_sec,
        **full_kwargs,
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)
    return max(minimum, value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_requested_language(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "auto", "detect", "default"}:
        return None
    return normalized


def _load_raw_items(row: Transcript) -> list[dict[str, object]]:
    try:
        payload = json.loads(row.words_json or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail="Stored transcript words are invalid"
        ) from exc
    if not isinstance(payload, list):
        raise HTTPException(
            status_code=500, detail="Stored transcript words are invalid"
        )
    return [item for item in payload if isinstance(item, dict)]


def _parse_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _is_blank_region(item: dict[str, object]) -> bool:
    return (
        bool(item.get("blanked"))
        or str(item.get("kind") or "").strip().lower() == "blank_region"
    )


def _sort_items(items: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        items,
        key=lambda item: (
            _parse_float(item.get("start_sec"), 0.0),
            _parse_float(item.get("end_sec"), 0.0),
            1 if _is_blank_region(item) else 0,
            str(item.get("id") or ""),
        ),
    )


def _normalize_blank_item(
    item: dict[str, object], duration_sec: float
) -> dict[str, object] | None:
    start_sec = max(0.0, min(_parse_float(item.get("start_sec"), 0.0), duration_sec))
    end_sec = max(
        0.0, min(_parse_float(item.get("end_sec"), duration_sec), duration_sec)
    )
    if end_sec <= start_sec + 0.02:
        return None
    return {
        "id": str(item.get("id") or f"blank-{uuid4()}"),
        "text": "",
        "start_sec": round(start_sec, 3),
        "end_sec": round(end_sec, 3),
        "blanked": True,
        "kind": "blank_region",
        "quality_score": 1.0,
        "quality_label": "blanked",
        "source_pass": "manual",
    }


def _serialize_word(item: TranscriptWordPayload) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": item.id,
        "text": item.text,
        "start_sec": round(float(item.start_sec), 3),
        "end_sec": round(float(item.end_sec), 3),
    }
    if item.confidence is not None:
        payload["confidence"] = round(float(item.confidence), 3)
    if item.quality_score is not None:
        payload["quality_score"] = round(float(item.quality_score), 3)
    if item.quality_label:
        payload["quality_label"] = item.quality_label
    if item.source_pass:
        payload["source_pass"] = item.source_pass
    if item.speaker_id:
        payload["speaker_id"] = item.speaker_id
    if item.speaker_label:
        payload["speaker_label"] = item.speaker_label
    return payload


def _word_payload_from_item(item: dict[str, object]) -> TranscriptWordPayload | None:
    if _is_blank_region(item):
        return None
    text = str(item.get("text") or "").strip()
    if not text:
        return None
    start_sec = _parse_float(item.get("start_sec"), 0.0)
    end_sec = _parse_float(item.get("end_sec"), 0.0)
    if end_sec <= start_sec:
        end_sec = start_sec + 0.05
    confidence_raw = item.get("confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else None
    except (TypeError, ValueError):
        confidence = None
    quality_score_raw = item.get("quality_score")
    try:
        quality_score = (
            float(quality_score_raw) if quality_score_raw is not None else None
        )
    except (TypeError, ValueError):
        quality_score = None
    quality_label = str(item.get("quality_label") or "").strip().lower() or None
    if quality_label not in {"trusted", "weak"}:
        quality_label = None
    source_pass = str(item.get("source_pass") or "").strip().lower() or None
    if source_pass not in {"primary", "retry", "rescue", "manual"}:
        source_pass = None
    speaker_id = str(item.get("speaker_id") or "").strip() or None
    speaker_label = str(item.get("speaker_label") or "").strip() or None
    return TranscriptWordPayload(
        id=str(item.get("id") or uuid4()),
        text=text,
        start_sec=start_sec,
        end_sec=end_sec,
        confidence=confidence,
        quality_score=quality_score,
        quality_label=quality_label,
        source_pass=source_pass,
        speaker_id=speaker_id,
        speaker_label=speaker_label,
    )


def _word_models_from_payloads(
    words: list[TranscriptWordPayload],
) -> list[TranscriptWord]:
    return [
        TranscriptWord(
            id=item.id,
            text=item.text,
            display_text=None,
            start_sec=float(item.start_sec),
            end_sec=float(item.end_sec),
            confidence=item.confidence,
            quality_score=item.quality_score,
            quality_label=item.quality_label,
            source_pass=item.source_pass,
            speaker_id=item.speaker_id,
            speaker_label=item.speaker_label,
        )
        for item in words
    ]


def _contains_chars_in_ranges(text: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    for char in text:
        codepoint = ord(char)
        for start, end in ranges:
            if start <= codepoint <= end:
                return True
    return False


def _word_script_annotation(
    text: str,
    transcript_language: str | None,
) -> tuple[str | None, str | None]:
    sample = str(text or "").strip()
    if not sample:
        return None, None

    has_latin = any(("A" <= char <= "Z") or ("a" <= char <= "z") for char in sample)
    has_arabic = _contains_chars_in_ranges(sample, _ARABIC_SCRIPT_RANGES)
    indic_languages = _detect_indic_script_languages(sample)
    has_indic = contains_indic_script(sample)
    active_scripts = int(has_latin) + int(has_arabic) + int(has_indic)
    normalized_language = _normalize_detected_language(transcript_language)

    if active_scripts >= 2:
        preferred_language = indic_languages[0] if indic_languages else None
        if preferred_language is None and has_arabic and normalized_language == "ur":
            preferred_language = "ur"
        return "mixed", preferred_language
    if has_indic:
        fallback_language = normalized_language
        if fallback_language in {"en", "ur"}:
            fallback_language = None
        return "indic", indic_languages[0] if indic_languages else fallback_language
    if has_arabic:
        return "arabic", "ur" if normalized_language == "ur" else None
    if has_latin:
        return "latin", "en" if normalized_language == "en" else None
    return "other", None


def _annotate_word_script_metadata(
    words: list[TranscriptWord],
    transcript_language: str | None,
) -> list[TranscriptWord]:
    if not words:
        return words
    enriched: list[TranscriptWord] = []
    for word in words:
        script_tag, language_hint = _word_script_annotation(
            word.text, transcript_language
        )
        if script_tag is None and language_hint is None:
            enriched.append(word)
            continue
        enriched.append(
            word.model_copy(
                update={
                    "script_tag": script_tag,
                    "language_hint": language_hint,
                }
            )
        )
    return enriched


def _with_transliteration_display(words: list[TranscriptWord]) -> list[TranscriptWord]:
    if not words:
        return words
    sample_text = " ".join(word.text for word in words[:40])
    if not contains_indic_script(sample_text):
        return words
    transliterated_words = transliterate_words(
        [word.model_dump(mode="json") for word in words]
    )
    if not transliterated_words:
        return words

    enriched: list[TranscriptWord] = []
    for word, transliterated in zip(words, transliterated_words):
        display_text = str(transliterated.get("text") or "").strip()
        if not display_text or display_text == word.text:
            enriched.append(word)
            continue
        enriched.append(word.model_copy(update={"display_text": display_text}))
    return enriched


def _annotate_word_quality(
    words: list[TranscriptWordPayload], duration_sec: float
) -> list[TranscriptWordPayload]:
    if not words:
        return []

    trusted_min_score = _env_float("TRANSCRIPT_TRUSTED_MIN_SCORE", 0.72, 0.0)
    weak_gap_sec = _env_float(
        "TRANSCRIPT_WEAK_REGION_GAP_SEC",
        max(1.4, min(2.8, duration_sec * 0.035 + 0.6)),
        0.4,
    )
    ordered = sorted(
        words, key=lambda item: (float(item.start_sec), float(item.end_sec), item.id)
    )
    annotated: list[TranscriptWordPayload] = []
    for index, item in enumerate(ordered):
        source_pass = (
            item.source_pass
            if item.source_pass in {"primary", "retry", "rescue", "manual"}
            else "primary"
        )
        if source_pass == "manual" and item.quality_label in {"trusted", "weak"}:
            score = item.quality_score
            if score is None:
                score = 1.0 if item.quality_label == "trusted" else 0.45
            annotated.append(
                TranscriptWordPayload(
                    id=item.id,
                    text=item.text,
                    start_sec=float(item.start_sec),
                    end_sec=float(item.end_sec),
                    confidence=item.confidence,
                    quality_score=round(max(0.0, min(float(score), 1.0)), 3),
                    quality_label=item.quality_label,
                    source_pass=source_pass,
                )
            )
            continue

        score = 0.88
        confidence = item.confidence
        if confidence is not None:
            if confidence < 0.30:
                score = min(score, 0.28)
            elif confidence < 0.50:
                score = min(score, 0.45)
            elif confidence < 0.68:
                score = min(score, 0.63)
            elif confidence < 0.82:
                score = min(score, 0.78)

        if source_pass == "retry":
            score = min(score, 0.74)
        elif source_pass == "rescue":
            score = min(score, 0.58)

        prev_gap = (
            float(item.start_sec)
            if index == 0
            else max(0.0, float(item.start_sec) - float(ordered[index - 1].end_sec))
        )
        next_gap = (
            max(0.0, duration_sec - float(item.end_sec))
            if index == len(ordered) - 1
            else max(0.0, float(ordered[index + 1].start_sec) - float(item.end_sec))
        )
        local_gap = max(prev_gap, next_gap)
        surrounded_by_gaps = prev_gap >= (weak_gap_sec * 0.6) and next_gap >= (
            weak_gap_sec * 0.6
        )
        if surrounded_by_gaps or (
            (source_pass in {"retry", "rescue"}) and local_gap >= weak_gap_sec
        ):
            score = min(score, 0.56)

        quality_label = "trusted" if score >= trusted_min_score else "weak"
        annotated.append(
            TranscriptWordPayload(
                id=item.id,
                text=item.text,
                start_sec=float(item.start_sec),
                end_sec=float(item.end_sec),
                confidence=item.confidence,
                quality_score=round(score, 3),
                quality_label=quality_label,
                source_pass=source_pass,
            )
        )
    return annotated


def _materialize_transcript_items(
    items: list[dict[str, object]],
    duration_sec: float,
) -> tuple[list[dict[str, object]], list[TranscriptWord], str, list[TranscriptRegion]]:
    visible_payloads = [
        payload
        for payload in (_word_payload_from_item(item) for item in items)
        if payload is not None
    ]
    normalized_words = sanitize_transcript_words(
        visible_payloads,
        max(float(duration_sec), 0.0),
        apply_filters=False,
        apply_offset=False,  # Timestamps already have the offset from initial generation
    )
    annotated_words = _annotate_word_quality(
        normalized_words, max(float(duration_sec), 0.0)
    )
    blank_items = [
        normalized
        for normalized in (
            _normalize_blank_item(item, max(float(duration_sec), 0.0))
            for item in items
            if _is_blank_region(item)
        )
        if normalized is not None
    ]
    stored_items = _sort_items(
        [*[_serialize_word(word) for word in annotated_words], *blank_items]
    )
    text = " ".join(word.text for word in annotated_words)
    regions = _build_regions(
        annotated_words, blank_items, max(float(duration_sec), 0.0)
    )
    return stored_items, _word_models_from_payloads(annotated_words), text, regions


def _build_regions(
    words: list[TranscriptWordPayload],
    blank_items: list[dict[str, object]],
    duration_sec: float,
) -> list[TranscriptRegion]:
    del duration_sec
    weak_regions: list[TranscriptRegion] = []
    weak_group_gap_sec = _env_float("TRANSCRIPT_WEAK_GROUP_GAP_SEC", 0.65, 0.05)
    weak_words = [word for word in words if word.quality_label == "weak"]
    if weak_words:
        current_group: list[TranscriptWordPayload] = [weak_words[0]]
        for word in weak_words[1:]:
            gap_sec = max(0.0, float(word.start_sec) - float(current_group[-1].end_sec))
            if gap_sec <= weak_group_gap_sec:
                current_group.append(word)
                continue
            weak_regions.append(_make_weak_region(current_group))
            current_group = [word]
        weak_regions.append(_make_weak_region(current_group))

    blank_regions = [
        TranscriptRegion(
            start_sec=round(_parse_float(item.get("start_sec"), 0.0), 3),
            end_sec=round(_parse_float(item.get("end_sec"), 0.0), 3),
            status="blanked",
            reason="manual_blank",
            word_ids=[],
        )
        for item in blank_items
    ]

    return sorted(
        [*weak_regions, *blank_regions],
        key=lambda region: (
            float(region.start_sec),
            float(region.end_sec),
            region.status,
        ),
    )


def _make_weak_region(words: list[TranscriptWordPayload]) -> TranscriptRegion:
    reason = "uncertain_audio"
    if any(word.source_pass == "rescue" for word in words):
        reason = "rescue_fill"
    elif any(word.source_pass == "retry" for word in words):
        reason = "retry_fill"
    elif any(
        word.confidence is not None and float(word.confidence) < 0.55 for word in words
    ):
        reason = "low_confidence"
    return TranscriptRegion(
        start_sec=round(float(words[0].start_sec), 3),
        end_sec=round(float(words[-1].end_sec), 3),
        status="weak",
        reason=reason,
        word_ids=[word.id for word in words],
    )


def _summarize_transcript_quality(
    words: list[TranscriptWord],
    regions: list[TranscriptRegion],
) -> tuple[float, str, int, float, int]:
    if not words:
        issue_region_count = sum(1 for region in regions if region.status != "trusted")
        label = "needs_review" if issue_region_count > 0 else "trusted"
        return 0.0, label, 0, 0.0, issue_region_count

    weak_word_count = sum(1 for word in words if word.quality_label == "weak")
    weak_word_ratio = weak_word_count / max(len(words), 1)
    issue_region_count = sum(1 for region in regions if region.status != "trusted")

    quality_values: list[float] = []
    for word in words:
        if word.quality_score is not None:
            quality_values.append(float(word.quality_score))
        elif word.quality_label == "trusted":
            quality_values.append(0.92)
        elif word.quality_label == "weak":
            quality_values.append(0.45)
        elif word.confidence is not None:
            quality_values.append(max(0.0, min(float(word.confidence), 1.0)))
        else:
            quality_values.append(0.8)

    avg_quality = sum(quality_values) / max(len(quality_values), 1)
    quality_score = round(max(0.0, min(avg_quality, 1.0)), 3)
    label = (
        "needs_review"
        if weak_word_ratio >= 0.08 or issue_region_count > 0 or quality_score < 0.86
        else "trusted"
    )
    return (
        quality_score,
        label,
        weak_word_count,
        round(weak_word_ratio, 3),
        issue_region_count,
    )


def _load_words(row: Transcript) -> list[TranscriptWord]:
    _stored_items, words, _text, _regions = _materialize_transcript_items(
        _load_raw_items(row), float(row.duration_sec or 0.0)
    )
    return words


def _to_response(
    row: Transcript,
    *,
    word_offset: int = 0,
    word_limit: int | None = None,
) -> TranscriptResponse:
    _stored_items, words, text, regions = _materialize_transcript_items(
        _load_raw_items(row), float(row.duration_sec or 0.0)
    )
    words = _annotate_word_script_metadata(words, row.language)
    (
        quality_score,
        quality_label,
        weak_word_count,
        weak_word_ratio,
        issue_region_count,
    ) = _summarize_transcript_quality(words, regions)
    total_words = len(words)
    safe_offset = max(0, min(int(word_offset), total_words))
    if word_limit is None:
        response_words = words[safe_offset:]
    else:
        response_words = words[safe_offset : safe_offset + max(1, int(word_limit))]
    response_words = _with_transliteration_display(response_words)
    script_tags = sorted(
        {
            str(word.script_tag).strip()
            for word in words
            if str(word.script_tag or "").strip()
        }
    )
    # Only flag mixed_script when genuinely different script families are
    # present (e.g. latin + indic).  Ignore "other" (numbers, punctuation)
    # since it's noise and causes false positives for English-only videos.
    _meaningful_tags = [t for t in script_tags if t not in {"other"}]
    words_truncated = safe_offset > 0 or len(response_words) < total_words
    return TranscriptResponse(
        id=row.id,
        project_id=row.project_id,
        asset_id=row.asset_id,
        source=row.source,
        language=row.language,
        text=text,
        words=response_words,
        word_count=total_words,
        words_truncated=words_truncated,
        regions=regions,
        quality_score=quality_score,
        quality_label=quality_label,
        weak_word_count=weak_word_count,
        weak_word_ratio=weak_word_ratio,
        issue_region_count=issue_region_count,
        script_tags=script_tags,
        mixed_script=len(_meaningful_tags) > 1,
        duration_sec=row.duration_sec,
        is_mock=row.is_mock,
        created_at=row.created_at.isoformat(),
    )


def _store_transcript_items(
    session: Session,
    *,
    project_id: str,
    asset_id: str,
    duration_sec: float,
    source: str,
    language: str | None,
    is_mock: bool,
    items: list[dict[str, object]],
) -> Transcript:
    stored_items, _words, text, _regions = _materialize_transcript_items(
        items, duration_sec
    )
    row = Transcript(
        project_id=project_id,
        asset_id=asset_id,
        source=source,
        language=language,
        text=text,
        words_json=_json_dumps(stored_items),
        duration_sec=round(duration_sec, 3),
        is_mock=is_mock,
        updated_at=_utcnow(),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def _store_transcript(
    session: Session,
    *,
    project_id: str,
    asset_id: str,
    duration_sec: float,
    payload: TranscriptPayload,
) -> Transcript:
    items = []
    default_source_pass = infer_source_pass(payload.source)
    for word in payload.words:
        items.append(
            {
                "id": word.id,
                "text": word.text,
                "start_sec": float(word.start_sec),
                "end_sec": float(word.end_sec),
                "confidence": word.confidence,
                "quality_score": word.quality_score,
                "quality_label": word.quality_label,
                "source_pass": word.source_pass or default_source_pass,
            }
        )
    return _store_transcript_items(
        session,
        project_id=project_id,
        asset_id=asset_id,
        duration_sec=duration_sec,
        source=payload.source,
        language=payload.language,
        is_mock=payload.is_mock,
        items=items,
    )


def _transcript_row_to_payload(row: Transcript) -> TranscriptPayload | None:
    items = _load_raw_items(row)
    words: list[TranscriptWordPayload] = []
    for item in items:
        word = _word_payload_from_item(item)
        if word is not None:
            words.append(word)
    if not words:
        return None
    return TranscriptPayload(
        source=str(row.source or ""),
        language=row.language,
        text=str(row.text or "").strip() or " ".join(word.text for word in words),
        words=words,
        is_mock=bool(row.is_mock),
    )


def _transcript_payload_changed(
    before: TranscriptPayload, after: TranscriptPayload
) -> bool:
    if before.source != after.source or before.text != after.text:
        return True
    if len(before.words) != len(after.words):
        return True
    return any(
        left.text != right.text
        for left, right in zip(before.words, after.words, strict=False)
    )


def _refresh_reference_lyrics_on_reused_transcript(
    session: Session,
    *,
    row: Transcript,
    asset: MediaAsset,
    duration_sec: float,
    transcript_mode: str | None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> Transcript:
    if row.is_mock:
        return row
    song_like = looks_like_song_media(asset.filename)
    if not song_like and transcript_mode != "song":
        return row

    payload = _transcript_row_to_payload(row)
    if payload is None:
        return row

    if progress_callback:
        progress_callback(78, "Matching reference lyrics")
    _t_lyrics_start = time.monotonic()
    updated = maybe_apply_reference_lyrics(
        payload,
        filename=_lyrics_reference_filename_hint(session, asset),
        duration_sec=duration_sec,
        transcript_mode=transcript_mode,
    )
    logger.info(
        "⏱️ TIMING: Reused transcript lyrics matching took %.1fs",
        time.monotonic() - _t_lyrics_start,
    )
    if not _transcript_payload_changed(payload, updated):
        return row

    _persist_transcript_items(
        row,
        session=session,
        items=_payload_items(updated),
        source=updated.source,
    )
    return row


def _persist_transcript_items(
    row: Transcript,
    *,
    session: Session,
    items: list[dict[str, object]],
    source: str | None = None,
) -> None:
    stored_items, _words, text, _regions = _materialize_transcript_items(
        items, float(row.duration_sec or 0.0)
    )
    row.words_json = _json_dumps(stored_items)
    row.text = text
    if source is not None:
        row.source = source
    row.updated_at = _utcnow()
    session.add(row)
    session.commit()
    session.refresh(row)


def _sync_existing_subtitles(
    session: Session,
    *,
    project_id: str,
    asset_id: str,
    words: list[TranscriptWord],
) -> tuple[object, bool]:
    timeline_row = get_timeline_row(session, project_id)
    timeline_state = load_timeline_state(timeline_row)

    matching_styles: list[str] = []
    for track in timeline_state.tracks:
        if track.kind != "video":
            continue
        for clip in track.clips:
            if clip.asset_id != asset_id:
                continue
            matching_styles.extend(
                overlay.style
                for overlay in clip.text_overlays
                if isinstance(overlay.style, str) and overlay.style.strip()
            )

    if not matching_styles:
        return timeline_state, False

    style = Counter(matching_styles).most_common(1)[0][0]
    updated_state = apply_operation(
        timeline_state,
        OperationPayload(
            op_type="set_subtitles",
            source="ui",
            params={
                "asset_id": asset_id,
                "style": style,
                "words": [word.model_dump() for word in words],
                "clear_existing": True,
            },
        ),
    )
    save_timeline_state(session, timeline_row, updated_state, source="transcript_sync")
    return updated_state, True


def _keep_ranges_from_deleted_words(
    words: list[TranscriptWord],
    duration_sec: float,
    kept_ids: set[str],
    *,
    context_sec_override: float | None = None,
    merge_gap_sec_override: float | None = None,
    min_removed_sec_override: float | None = None,
) -> list[dict[str, float]]:
    all_ids = {word.id for word in words}
    deleted_ids = all_ids - kept_ids
    if not deleted_ids:
        return [{"start_sec": 0.0, "end_sec": round(duration_sec, 3)}]

    logger = logging.getLogger(__name__)
    logger.info(
        "transcript_cut: %d total words, %d kept, %d deleted, duration=%.3fs",
        len(words),
        len(kept_ids),
        len(deleted_ids),
        duration_sec,
    )

    context_sec = (
        max(0.0, float(context_sec_override))
        if context_sec_override is not None
        else _env_float("TRANSCRIPT_CUT_CONTEXT_SEC", 0.0, 0.0)
    )
    merge_gap_sec = (
        max(0.0, float(merge_gap_sec_override))
        if merge_gap_sec_override is not None
        else _env_float("TRANSCRIPT_CUT_MERGE_GAP_SEC", 0.08, 0.0)
    )
    min_removed_sec = (
        max(0.0, float(min_removed_sec_override))
        if min_removed_sec_override is not None
        else _env_float("TRANSCRIPT_CUT_MIN_REMOVAL_SEC", 0.0, 0.0)
    )
    # ASR timestamps are typically 50–150 ms imprecise.  A deleted word's
    # voice often starts a little before `word.start_sec` and trails a little
    # after `word.end_sec`.  These pads extend the cut window to capture the
    # full voice without cutting into the adjacent *kept* words.
    asr_head_pad = _env_float("TRANSCRIPT_CUT_ASR_HEAD_PAD_SEC", 0.08, 0.0)
    asr_tail_pad = _env_float("TRANSCRIPT_CUT_ASR_TAIL_PAD_SEC", 0.10, 0.0)
    weak_region_safety_sec = _resolve_transcript_cut_weak_region_safety_sec()

    ordered_words = sorted(
        words, key=lambda item: (float(item.start_sec), float(item.end_sec))
    )
    kept_words = [word for word in ordered_words if word.id in kept_ids]
    if not kept_words:
        raise HTTPException(
            status_code=400,
            detail="No valid words were kept; cannot render an empty timeline",
        )

    delete_runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for idx, word in enumerate(ordered_words):
        if word.id not in kept_ids:
            if run_start is None:
                run_start = idx
            continue
        if run_start is not None:
            delete_runs.append((run_start, idx - 1))
            run_start = None
    if run_start is not None:
        delete_runs.append((run_start, len(ordered_words) - 1))

    cut_ranges: list[tuple[float, float]] = []
    for start_idx, end_idx in delete_runs:
        deleted_run_words = ordered_words[start_idx : end_idx + 1]
        deleted_start = min(
            float(ordered_words[i].start_sec) for i in range(start_idx, end_idx + 1)
        )
        deleted_end = max(
            float(ordered_words[i].end_sec) for i in range(start_idx, end_idx + 1)
        )
        run_has_weak_boundary = any(
            _word_needs_cut_safety(word) for word in deleted_run_words
        )
        left_safety = 0.0
        right_safety = 0.0
        prev_kept = next(
            (
                ordered_words[idx]
                for idx in range(start_idx - 1, -1, -1)
                if ordered_words[idx].id in kept_ids
            ),
            None,
        )
        next_kept = next(
            (
                ordered_words[idx]
                for idx in range(end_idx + 1, len(ordered_words))
                if ordered_words[idx].id in kept_ids
            ),
            None,
        )

        if prev_kept is None:
            raw_cut_start = max(0.0, deleted_start - context_sec)
        else:
            left_safety = (
                weak_region_safety_sec
                if run_has_weak_boundary or _word_needs_cut_safety(prev_kept)
                else 0.0
            )
            raw_cut_start = float(prev_kept.end_sec) + context_sec + left_safety

        if next_kept is None:
            raw_cut_end = min(duration_sec, deleted_end + context_sec)
        else:
            right_safety = (
                weak_region_safety_sec
                if run_has_weak_boundary or _word_needs_cut_safety(next_kept)
                else 0.0
            )
            raw_cut_end = (
                float(next_kept.start_sec) - right_safety
                if prev_kept is None
                else float(next_kept.start_sec) - context_sec - right_safety
            )

        if raw_cut_start >= raw_cut_end:
            # Context/safety padding collapsed the cut range.  Use midpoints
            # between adjacent kept words and the deleted span so we still
            # remove a meaningful chunk of audio.
            if prev_kept is not None:
                cut_start = (float(prev_kept.end_sec) + deleted_start) / 2.0
            else:
                cut_start = max(0.0, deleted_start)
            if next_kept is not None:
                cut_end = (deleted_end + float(next_kept.start_sec)) / 2.0
            else:
                cut_end = min(duration_sec, deleted_end)
            # Guarantee we at least cover the deleted word span.
            cut_start = min(cut_start, deleted_start)
            cut_end = max(cut_end, deleted_end)
        else:
            if left_safety > 0 or right_safety > 0:
                cut_start = max(raw_cut_start, deleted_start)
                cut_end = max(raw_cut_end, deleted_end)
            else:
                cut_start = min(raw_cut_start, deleted_start)
                cut_end = max(raw_cut_end, deleted_end)

        # ── ASR imprecision padding ──────────────────────────────────────────
        # ASR word timestamps are imprecise by ~50–150 ms. Extend ordinary cuts
        # to swallow voice onset/tail, but do not let this padding erase the
        # explicit weak-region safety margins. Also preserve leading silence when
        # the first word is deleted; creators often rely on that gap for pacing.
        padded_start = cut_start
        padded_end = cut_end
        if left_safety <= 0 and right_safety <= 0:
            # head_pad: push cut_start earlier to catch the voice onset, but only
            # when there is a previous kept word to clamp against.
            if prev_kept is not None:
                padded_start = max(cut_start - asr_head_pad, float(prev_kept.end_sec))
                padded_start = max(0.0, padded_start)
                cut_start = min(cut_start, padded_start)

            # tail_pad: push cut_end later to catch the voice tail.
            padded_end = cut_end + asr_tail_pad
            if next_kept is not None:
                # Never cut past where the next kept word starts
                padded_end = min(padded_end, float(next_kept.start_sec))
            padded_end = min(duration_sec, padded_end)
            cut_end = max(cut_end, padded_end)

        cut_start = max(0.0, min(cut_start, duration_sec))
        cut_end = max(0.0, min(cut_end, duration_sec))
        if cut_end > cut_start:
            cut_ranges.append((cut_start, cut_end))

        logger.info(
            "transcript_cut: run [%d-%d] deleted_span=[%.3f, %.3f] "
            "prev_kept=%s next_kept=%s raw=[%.3f, %.3f] padded=[%.3f, %.3f]",
            start_idx,
            end_idx,
            deleted_start,
            deleted_end,
            f"{prev_kept.text}@{prev_kept.end_sec:.3f}" if prev_kept else "None",
            f"{next_kept.text}@{next_kept.start_sec:.3f}" if next_kept else "None",
            cut_start,
            cut_end,
            padded_start,
            padded_end,
        )

    merged_cuts: list[list[float]] = []
    for start, end in sorted(cut_ranges, key=lambda item: item[0]):
        if not merged_cuts:
            merged_cuts.append([start, end])
            continue
        prev_start, prev_end = merged_cuts[-1]
        if start <= prev_end + merge_gap_sec:
            merged_cuts[-1] = [prev_start, max(prev_end, end)]
        else:
            merged_cuts.append([start, end])

    effective_cuts = [
        (start, end) for start, end in merged_cuts if (end - start) >= min_removed_sec
    ]

    keep_ranges: list[dict[str, float]] = []
    cursor = 0.0
    for start, end in effective_cuts:
        if start > cursor:
            keep_ranges.append({"start_sec": cursor, "end_sec": start})
        cursor = max(cursor, end)
    if cursor < duration_sec:
        keep_ranges.append({"start_sec": cursor, "end_sec": duration_sec})

    normalized_ranges: list[dict[str, float]] = []
    for item in keep_ranges:
        start = round(float(item["start_sec"]), 3)
        end = round(float(item["end_sec"]), 3)
        if end <= start + 0.02:
            continue
        normalized_ranges.append({"start_sec": start, "end_sec": end})

    logger.info(
        "transcript_cut: cut_ranges=%s keep_ranges=%s",
        [(round(s, 3), round(e, 3)) for s, e in effective_cuts],
        normalized_ranges,
    )
    return normalized_ranges


def _word_needs_cut_safety(word: TranscriptWord | None) -> bool:
    if word is None:
        return False
    if word.quality_label == "weak":
        return True
    return (word.source_pass or "") in {"retry", "rescue"}


def _apply_video_ranges(
    session: Session,
    *,
    project_id: str,
    asset_id: str,
    ranges: list[dict[str, float]],
) -> tuple[list[dict[str, float]], object]:
    if not ranges:
        raise HTTPException(
            status_code=400,
            detail="Deleting all transcript words would remove the entire video",
        )

    timeline = get_timeline_row(session, project_id)
    state = load_timeline_state(timeline)
    operation = OperationPayload(
        op_type="replace_video_track_clips",
        source="ui",
        params={
            "asset_id": asset_id,
            "ranges": ranges,
            "clear_audio_tracks": True,
        },
    )
    try:
        apply_operation(state, operation)
        timeline = save_timeline_state(
            session,
            timeline,
            state,
            source="ui",
            operation=operation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ranges, load_timeline_state(timeline)


def _current_video_source_ranges(
    session: Session,
    *,
    project_id: str,
    asset_id: str,
    duration_sec: float,
) -> list[dict[str, float]]:
    timeline = get_timeline_row(session, project_id)
    state = load_timeline_state(timeline)
    ranges: list[dict[str, float]] = []
    for track in state.tracks:
        if track.kind != "video":
            continue
        for clip in sorted(
            track.clips, key=lambda item: float(item.timeline_start_sec)
        ):
            if clip.asset_id != asset_id:
                continue
            start = max(0.0, min(float(clip.start_sec), duration_sec))
            end = max(0.0, min(float(clip.end_sec), duration_sec))
            if end > start + 0.02:
                ranges.append({"start_sec": round(start, 3), "end_sec": round(end, 3)})
    if ranges:
        return ranges
    return [{"start_sec": 0.0, "end_sec": round(duration_sec, 3)}]


def _intersect_source_ranges(
    current_ranges: list[dict[str, float]],
    keep_ranges: list[dict[str, float]],
) -> list[dict[str, float]]:
    intersected: list[dict[str, float]] = []
    sorted_keep_ranges = sorted(
        keep_ranges,
        key=lambda item: (float(item["start_sec"]), float(item["end_sec"])),
    )
    for current in current_ranges:
        current_start = float(current["start_sec"])
        current_end = float(current["end_sec"])
        for keep in sorted_keep_ranges:
            keep_start = float(keep["start_sec"])
            keep_end = float(keep["end_sec"])
            start = max(current_start, keep_start)
            end = min(current_end, keep_end)
            if end <= start + 0.02:
                continue
            if intersected and abs(float(intersected[-1]["end_sec"]) - start) <= 0.001:
                intersected[-1]["end_sec"] = round(end, 3)
            else:
                intersected.append(
                    {"start_sec": round(start, 3), "end_sec": round(end, 3)}
                )
    return intersected


def _apply_transcript_keep_ranges(
    session: Session,
    *,
    project_id: str,
    asset_id: str,
    duration_sec: float,
    ranges: list[dict[str, float]],
) -> tuple[list[dict[str, float]], object]:
    current_ranges = _current_video_source_ranges(
        session,
        project_id=project_id,
        asset_id=asset_id,
        duration_sec=duration_sec,
    )
    next_ranges = _intersect_source_ranges(current_ranges, ranges)
    return _apply_video_ranges(
        session,
        project_id=project_id,
        asset_id=asset_id,
        ranges=next_ranges,
    )


def _build_replacement_words(
    selected_words: list[TranscriptWordPayload], replacement_text: str
) -> list[TranscriptWordPayload]:
    tokens = [token for token in replacement_text.strip().split() if token.strip()]
    if not tokens:
        return []

    old_tokens = [word.text for word in selected_words]
    prefix_len = 0
    max_prefix = min(len(old_tokens), len(tokens))
    while (
        prefix_len < max_prefix
        and old_tokens[prefix_len].strip().lower() == tokens[prefix_len].strip().lower()
    ):
        prefix_len += 1

    suffix_len = 0
    max_suffix = min(len(old_tokens) - prefix_len, len(tokens) - prefix_len)
    while suffix_len < max_suffix:
        old_token = old_tokens[len(old_tokens) - 1 - suffix_len].strip().lower()
        new_token = tokens[len(tokens) - 1 - suffix_len].strip().lower()
        if old_token != new_token:
            break
        suffix_len += 1

    kept_prefix = selected_words[:prefix_len]
    kept_suffix = (
        selected_words[len(selected_words) - suffix_len :] if suffix_len else []
    )
    old_middle = selected_words[
        prefix_len : len(selected_words) - suffix_len
        if suffix_len
        else len(selected_words)
    ]
    new_middle_tokens = tokens[
        prefix_len : len(tokens) - suffix_len if suffix_len else len(tokens)
    ]

    replacement: list[TranscriptWordPayload] = [*kept_prefix]
    if new_middle_tokens:
        if old_middle:
            span_start = float(old_middle[0].start_sec)
            span_end = float(old_middle[-1].end_sec)
        else:
            span_start = float(selected_words[0].start_sec)
            span_end = float(selected_words[-1].end_sec)
        span_end = max(span_end, span_start + 0.05)
        step = max((span_end - span_start) / max(len(new_middle_tokens), 1), 0.05)
        for index, token in enumerate(new_middle_tokens):
            source_word = (
                old_middle[index]
                if index < len(old_middle)
                else selected_words[min(prefix_len, len(selected_words) - 1)]
            )
            word_start = span_start + (index * step)
            word_end = span_start + ((index + 1) * step)
            if index == len(new_middle_tokens) - 1:
                word_end = span_end
            replacement.append(
                TranscriptWordPayload(
                    id=source_word.id if index < len(old_middle) else str(uuid4()),
                    text=token,
                    start_sec=word_start,
                    end_sec=word_end,
                    confidence=None,
                    quality_score=1.0,
                    quality_label="trusted",
                    source_pass="manual",
                )
            )
    replacement.extend(kept_suffix)
    return replacement


def _apply_range_update_items(
    items: list[dict[str, object]],
    *,
    duration_sec: float,
    start_word_id: str,
    end_word_id: str,
    mode: str,
    text: str | None,
) -> list[dict[str, object]]:
    visible_words = [
        TranscriptWordPayload(
            id=word.id,
            text=word.text,
            start_sec=word.start_sec,
            end_sec=word.end_sec,
            confidence=word.confidence,
            quality_score=word.quality_score,
            quality_label=word.quality_label,
            source_pass=word.source_pass,
        )
        for word in _load_word_models_from_items(items, duration_sec)
    ]
    if not visible_words:
        raise HTTPException(status_code=400, detail="Transcript has no editable words")

    index_by_id = {word.id: idx for idx, word in enumerate(visible_words)}
    if start_word_id not in index_by_id or end_word_id not in index_by_id:
        raise HTTPException(
            status_code=404, detail="Selected transcript range was not found"
        )
    start_idx = min(index_by_id[start_word_id], index_by_id[end_word_id])
    end_idx = max(index_by_id[start_word_id], index_by_id[end_word_id])
    selected_words = visible_words[start_idx : end_idx + 1]

    replacement_words: list[TranscriptWordPayload]
    blank_items = [
        normalized
        for normalized in (
            _normalize_blank_item(item, duration_sec)
            for item in items
            if _is_blank_region(item)
        )
        if normalized is not None
    ]
    if mode == "blank":
        replacement_words = []
        blank_items.append(
            {
                "id": f"blank-{uuid4()}",
                "text": "",
                "start_sec": round(float(selected_words[0].start_sec), 3),
                "end_sec": round(float(selected_words[-1].end_sec), 3),
                "blanked": True,
                "kind": "blank_region",
                "quality_score": 1.0,
                "quality_label": "blanked",
                "source_pass": "manual",
            }
        )
    elif mode == "delete":
        replacement_words = []
    elif mode == "preserve":
        replacement_words = [
            TranscriptWordPayload(
                id=word.id,
                text=word.text,
                start_sec=word.start_sec,
                end_sec=word.end_sec,
                confidence=word.confidence,
                quality_score=1.0,
                quality_label="trusted",
                source_pass="manual",
            )
            for word in selected_words
        ]
    else:
        replacement_words = _build_replacement_words(selected_words, text or "")

    merged_words = [
        *visible_words[:start_idx],
        *replacement_words,
        *visible_words[end_idx + 1 :],
    ]
    merged_items = [
        *_sort_items([_serialize_word(word) for word in merged_words]),
        *blank_items,
    ]
    return merged_items


def _load_word_models_from_items(
    items: list[dict[str, object]], duration_sec: float
) -> list[TranscriptWord]:
    _stored_items, words, _text, _regions = _materialize_transcript_items(
        items, duration_sec
    )
    return words


def _get_project_and_transcript(
    session: Session,
    *,
    project_id: str,
    transcript_id: str,
) -> tuple[Project, Transcript]:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    row = session.exec(
        select(Transcript).where(
            Transcript.id == transcript_id, Transcript.project_id == project_id
        )
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return project, row


def _generate_transcript_response(
    session: Session,
    *,
    project_id: str,
    payload: TranscriptGenerateRequest,
    progress_callback: Callable[[int, str], None] | None = None,
    word_limit: int | None = None,
) -> TranscriptGenerateResponse:
    # --- Memory safety: reject early if the system is starved ---
    _check_memory_before_transcription()

    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    asset = session.exec(
        select(MediaAsset).where(
            MediaAsset.id == payload.asset_id, MediaAsset.project_id == project_id
        )
    ).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Media asset not found")
    if asset.media_type != "video":
        raise HTTPException(
            status_code=400, detail="Transcript generation requires a video asset"
        )

    source_path = storage.resolve_upload_asset(asset.storage_path)
    if not Path(source_path).exists():
        raise HTTPException(
            status_code=404,
            detail="Uploaded media file is missing. Re-upload the asset and try again.",
        )
    duration_sec = (
        float(asset.duration_sec)
        if asset.duration_sec is not None
        else (probe_duration_seconds(source_path) or 0.0)
    )
    if duration_sec <= 0:
        raise HTTPException(
            status_code=400,
            detail="Could not determine video duration for transcript generation",
        )
    if (
        settings.max_transcribe_duration_sec > 0
        and duration_sec > settings.max_transcribe_duration_sec
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Video exceeds configured transcription limit "
                f"({settings.max_transcribe_duration_sec:.0f} seconds)"
            ),
        )

    requested_language = _normalize_requested_language(payload.language)
    transcript_mode = _resolve_requested_transcript_mode(
        payload.mode,
        filename=asset.filename,
    )
    row: Transcript | None = None
    reused_transcript = False
    can_reuse_existing = (
        not payload.force_regenerate
        and _env_bool("TRANSCRIBE_REUSE_EXISTING_ON_GENERATE", False)
        and _allow_cached_transcript_reuse(requested_language)
    )
    if can_reuse_existing:
        existing = session.exec(
            select(Transcript)
            .where(Transcript.project_id == project_id, Transcript.asset_id == asset.id)
            .order_by(Transcript.created_at.desc())
        ).first()
        if existing and not existing.is_mock:
            existing_language = _normalize_requested_language(existing.language)
            if requested_language is None or requested_language == existing_language:
                try:
                    existing_items = _load_raw_items(existing)
                    if _load_words(existing) and (
                        _transcript_reuse_score(
                            existing,
                            items=existing_items,
                            filename=asset.filename,
                        )
                        is not None
                    ):
                        retry_items = _retry_weak_regions_in_items(
                            source_path,
                            duration_sec,
                            items=existing_items,
                            language_hint=payload.language or existing.language,
                            prompt=payload.prompt,
                            progress_callback=progress_callback,
                        )
                        if retry_items != existing_items:
                            _persist_transcript_items(
                                existing, session=session, items=retry_items
                            )
                        row = existing
                        reused_transcript = True
                        if progress_callback:
                            progress_callback(90, "Reusing existing transcript")
                except HTTPException:
                    row = None
        if row is None:
            related = _related_library_transcript(
                session,
                asset,
                requested_language=requested_language,
            )
            if related is not None:
                try:
                    row = _store_transcript_items(
                        session,
                        project_id=project_id,
                        asset_id=asset.id,
                        duration_sec=duration_sec,
                        source=related.source,
                        language=related.language,
                        is_mock=related.is_mock,
                        items=_load_raw_items(related),
                    )
                    reused_transcript = True
                    if progress_callback:
                        progress_callback(
                            90, "Reusing matching transcript from your library"
                        )
                except HTTPException:
                    row = None

    if reused_transcript and row is not None:
        row = _refresh_reference_lyrics_on_reused_transcript(
            session,
            row=row,
            asset=asset,
            duration_sec=duration_sec,
            transcript_mode=transcript_mode,
            progress_callback=progress_callback,
        )

    if row is None:
        _t_total_start = time.monotonic()
        song_like_media = looks_like_song_media(asset.filename)
        strategy = _resolve_transcript_generation_strategy(
            duration_sec, transcript_mode, song_like_media=song_like_media
        )
        configured_fast_mode = _env_bool("TRANSCRIBE_FAST_MODE", False)
        fast_mode = configured_fast_mode
        allow_fast_retry = not configured_fast_mode
        if progress_callback:
            progress_callback(
                8,
                f"Using {strategy.mode} transcript mode for a {duration_sec:.0f}s clip",
            )
        # Acquire semaphore to limit concurrent transcript jobs and prevent OOM
        acquired = _transcript_semaphore.acquire(timeout=300)
        if not acquired:
            raise HTTPException(
                status_code=503,
                detail="Another transcript generation is in progress. Please wait and retry.",
            )
        try:
            # Re-check memory after acquiring the semaphore
            _check_memory_before_transcription()
            while True:
                try:
                    _t_chunked_start = time.monotonic()
                    transcript_payload = _call_generate_transcript_payload_chunked_compatible(
                        source_path,
                        duration_sec,
                        language_hint=payload.language,
                        allow_mock_fallback=_env_bool(
                            "TRANSCRIBE_ALLOW_MOCK_FALLBACK", False
                        ),
                        fast_mode=fast_mode,
                        prompt=payload.prompt,
                        progress_callback=progress_callback,
                        translate_to_english=payload.translate_to_english,
                        mode=strategy.mode,
                        optimize_for_speed=strategy.optimize_for_speed,
                        bypass_max_duration_sec_override=strategy.bypass_max_duration_sec,
                        chunk_duration_sec_override=strategy.chunk_duration_sec,
                        chunk_overlap_sec_override=strategy.chunk_overlap_sec,
                        chunk_parallelism_override=strategy.chunk_parallelism,
                        filename=asset.filename,
                    )
                    _t_chunked_elapsed = time.monotonic() - _t_chunked_start
                    logger.info(
                        "⏱️ TIMING: Chunked transcription took %.1fs (%.1fs audio)",
                        _t_chunked_elapsed,
                        duration_sec,
                    )
                    break
                except RuntimeError as exc:
                    if allow_fast_retry and _looks_like_sigkill_error(str(exc)):
                        allow_fast_retry = False
                        fast_mode = True
                        if progress_callback:
                            progress_callback(
                                12,
                                "Transcript process hit memory limits, retrying in fast mode",
                            )
                        continue
                    raise HTTPException(
                        status_code=500, detail=_humanize_transcript_runtime_error(exc)
                    ) from exc
        finally:
            _force_gc()
            _transcript_semaphore.release()

        if (
            transcript_payload.words
            and not transcript_payload.is_mock
            and looks_like_duet_media(asset.filename)
        ):
            if progress_callback:
                progress_callback(72, "Detecting speakers in multi-voice song")
            transcript_payload = maybe_enhance_duet_transcript(
                transcript_payload,
                audio_path=source_path,
                duration_sec=duration_sec,
                filename=asset.filename,
                language_hint=payload.language or transcript_payload.language,
                progress_callback=progress_callback,
            )

        _t_lyrics_start = time.monotonic()
        if (
            transcript_payload.words
            and not transcript_payload.is_mock
            and (song_like_media or strategy.mode == "song")
        ):
            if progress_callback:
                progress_callback(78, "Matching reference lyrics")
            transcript_payload = maybe_apply_reference_lyrics(
                transcript_payload,
                filename=_lyrics_reference_filename_hint(session, asset),
                duration_sec=duration_sec,
            )
        _t_lyrics_elapsed = time.monotonic() - _t_lyrics_start
        logger.info("⏱️ TIMING: Lyrics reference matching took %.1fs", _t_lyrics_elapsed)

        if not transcript_payload.words:
            raise HTTPException(
                status_code=500, detail="Transcript generation returned no words"
            )

        if (
            strategy.mode == "song"
            and transcript_payload.words
            and not transcript_payload.is_mock
        ):
            transcript_payload = trim_song_mode_to_manual_lyrics_span(
                transcript_payload
            )
            if progress_callback:
                progress_callback(84, "Validating sung words")
            transcript_payload = stabilize_song_transcript(
                transcript_payload,
                path=source_path,
                duration_sec=duration_sec,
            )

        # --- Per-word timestamp refinement (energy-based onset detection) ---
        if (
            transcript_payload.words
            and not transcript_payload.is_mock
            and not strategy.skip_timestamp_refinement
            and _env_bool("TRANSCRIBE_TIMESTAMP_REFINEMENT_ENABLED", True)
        ):
            if progress_callback:
                progress_callback(88, "Refining word timestamps")
            _t_refine_start = time.monotonic()
            try:
                from ..timestamp_refiner import refine_word_timestamps_batch

                refined_words = refine_word_timestamps_batch(
                    transcript_payload.words,
                    source_path,
                    max_words=int(
                        os.getenv("TRANSCRIBE_TIMESTAMP_REFINEMENT_MAX_WORDS", "2000")
                    ),
                )
                if refined_words is not transcript_payload.words:
                    transcript_payload = TranscriptPayload(
                        source=transcript_payload.source,
                        language=transcript_payload.language,
                        text=transcript_payload.text,
                        words=refined_words,
                        is_mock=transcript_payload.is_mock,
                    )
            except Exception as refine_exc:  # noqa: BLE001
                logger.warning(
                    "Timestamp refinement failed, using original timestamps: %s",
                    refine_exc,
                )
            _t_refine_elapsed = time.monotonic() - _t_refine_start
            logger.info("⏱️ TIMING: Timestamp refinement took %.1fs", _t_refine_elapsed)

        if (
            song_like_media
            and transcript_payload.words
            and not transcript_payload.is_mock
        ):
            transcript_payload = trim_songlike_tail_hallucination(
                transcript_payload,
                duration_sec=duration_sec,
            )

        _t_weak_start = time.monotonic()
        transcript_items = _payload_items(transcript_payload)
        if not strategy.skip_weak_region_retry:
            transcript_items = _retry_weak_regions_in_items(
                source_path,
                duration_sec,
                items=transcript_items,
                language_hint=payload.language or transcript_payload.language,
                prompt=payload.prompt,
                progress_callback=progress_callback,
            )
        _t_weak_elapsed = time.monotonic() - _t_weak_start
        logger.info("⏱️ TIMING: Weak region retry took %.1fs", _t_weak_elapsed)

        _t_store_start = time.monotonic()
        row = _store_transcript_items(
            session,
            project_id=project_id,
            asset_id=asset.id,
            duration_sec=duration_sec,
            source=transcript_payload.source,
            language=transcript_payload.language,
            is_mock=transcript_payload.is_mock,
            items=transcript_items,
        )
        _t_store_elapsed = time.monotonic() - _t_store_start
        logger.info("⏱️ TIMING: DB storage took %.1fs", _t_store_elapsed)

        _t_total_elapsed = time.monotonic() - _t_total_start
        logger.info(
            "⏱️ TIMING: TOTAL transcript generation took %.1fs for %.1fs audio (%d words)",
            _t_total_elapsed,
            duration_sec,
            len(transcript_payload.words),
        )

    if progress_callback:
        progress_callback(97, "Applying transcript ranges to timeline")
    _ranges, timeline_state = _apply_video_ranges(
        session,
        project_id=project_id,
        asset_id=asset.id,
        ranges=[{"start_sec": 0.0, "end_sec": round(duration_sec, 3)}],
    )

    return TranscriptGenerateResponse(
        transcript=_to_response(row, word_limit=word_limit),
        timeline=timeline_state,
        reused_transcript=reused_transcript,
    )


def _process_generate_transcript_job(
    *,
    job_id: str,
    project_id: str,
    payload_json: dict[str, object],
) -> None:
    with Session(engine) as session:
        job = session.exec(select(Job).where(Job.id == job_id)).first()
        if not job:
            return
        try:
            set_job_status(
                session,
                job,
                status="running",
                progress=5,
                stage="prepare",
                message="Preparing transcript generation",
            )
            payload = TranscriptGenerateRequest.model_validate(payload_json)

            def on_progress(progress: int, message: str) -> None:
                set_job_status(
                    session,
                    job,
                    status="running",
                    progress=max(5, min(99, int(progress))),
                    stage=_transcript_progress_stage(progress, message),
                    message=message,
                )

            response = _generate_transcript_response(
                session,
                project_id=project_id,
                payload=payload,
                progress_callback=on_progress,
                word_limit=_effective_word_limit(None),
            )
            result_path = _transcript_generate_result_path(job.id)
            result_path.write_text(
                _json_dumps(response.model_dump(mode="json")), encoding="utf-8"
            )
            set_job_status(
                session,
                job,
                status="completed",
                progress=100,
                stage="complete",
                message="Transcript generation completed",
                output_path=str(result_path),
            )
        except Exception as exc:  # noqa: BLE001
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            message = detail if isinstance(detail, str) else str(detail)
            set_job_status(
                session,
                job,
                status="failed",
                progress=100,
                stage="failed",
                message=message,
                error=message,
            )


@router.post("/generate", response_model=TranscriptGenerateResponse)
def generate(
    payload: TranscriptGenerateRequest,
    project_id: str,
    word_limit: int | None = Query(default=None, ge=0, le=10000),
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> TranscriptGenerateResponse:
    return _generate_transcript_response(
        session,
        project_id=project_id,
        payload=payload,
        word_limit=_effective_word_limit(word_limit),
    )


@router.post("/generate/async", response_model=JobResponse)
def generate_async(
    payload: TranscriptGenerateRequest,
    project_id: str,
    force: bool = False,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> JobResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not force:
        active = find_recent_active_job(
            session, project_id, kind="transcript_generate", within_seconds=0
        )
        if active:
            return _to_job_response(session, active)

    job = create_job(session, project_id, kind="transcript_generate")
    threading.Thread(
        target=_process_generate_transcript_job,
        kwargs={
            "job_id": job.id,
            "project_id": project_id,
            "payload_json": payload.model_dump(mode="json"),
        },
        name=f"transcript-generate-{job.id[:8]}",
        daemon=True,
    ).start()
    return _to_job_response(session, job)


@router.get("/generate/results/{job_id}", response_model=TranscriptGenerateResponse)
def get_generate_result(
    job_id: str,
    project_id: str,
    word_limit: int | None = Query(default=None, ge=0, le=10000),
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> TranscriptGenerateResponse:
    job = session.exec(
        select(Job).where(
            Job.id == job_id,
            Job.project_id == project_id,
            Job.kind == "transcript_generate",
        )
    ).first()
    if not job:
        raise HTTPException(
            status_code=404, detail="Transcript generation job not found"
        )
    if job.status == "failed":
        raise HTTPException(
            status_code=409, detail=job.error or "Transcript generation failed"
        )
    if job.status != "completed":
        raise HTTPException(
            status_code=409, detail="Transcript generation job not completed"
        )

    result_path = _transcript_generate_result_path(job.id)
    if not result_path.exists():
        raise HTTPException(
            status_code=500, detail="Transcript generation result missing"
        )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail="Transcript generation result payload invalid"
        ) from exc
    response = TranscriptGenerateResponse.model_validate(payload)
    effective_limit = _effective_word_limit(word_limit)
    if effective_limit is None:
        return response
    row = session.exec(
        select(Transcript).where(
            Transcript.id == response.transcript.id, Transcript.project_id == project_id
        )
    ).first()
    if not row:
        return response
    return TranscriptGenerateResponse(
        transcript=_to_response(row, word_limit=effective_limit),
        timeline=response.timeline,
        reused_transcript=response.reused_transcript,
    )


@router.get("", response_model=TranscriptResponse)
def get_latest(
    project_id: str,
    transcript_id: str | None = None,
    word_offset: int = Query(default=0, ge=0),
    word_limit: int | None = Query(default=None, ge=0, le=10000),
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> TranscriptResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if transcript_id:
        row = session.exec(
            select(Transcript).where(
                Transcript.project_id == project_id,
                Transcript.id == transcript_id,
            )
        ).first()
    else:
        row = session.exec(
            select(Transcript)
            .where(Transcript.project_id == project_id, Transcript.is_mock == False)  # noqa: E712
            .order_by(Transcript.created_at.desc())
        ).first()
        if row is None:
            row = session.exec(
                select(Transcript)
                .where(Transcript.project_id == project_id)
                .order_by(Transcript.created_at.desc())
            ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return _to_response(
        row,
        word_offset=word_offset,
        word_limit=_effective_word_limit(word_limit),
    )


@router.get("/{transcript_id}/words", response_model=TranscriptWordPageResponse)
def get_words_page(
    transcript_id: str,
    project_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=800, ge=1, le=5000),
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> TranscriptWordPageResponse:
    row = session.exec(
        select(Transcript).where(
            Transcript.id == transcript_id, Transcript.project_id == project_id
        )
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Transcript not found")
    words = _load_words(row)
    total_words = len(words)
    page_words = _with_transliteration_display(words[offset : offset + limit])
    return TranscriptWordPageResponse(
        transcript_id=row.id,
        project_id=row.project_id,
        offset=offset,
        limit=limit,
        total_words=total_words,
        words=page_words,
    )


@router.post("/cut", response_model=TranscriptCutResponse)
def apply_text_cut(
    payload: TranscriptCutRequest,
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> TranscriptCutResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    row = session.exec(
        select(Transcript).where(
            Transcript.id == payload.transcript_id, Transcript.project_id == project_id
        )
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Transcript not found")

    raw_items = _load_raw_items(row)
    words = _load_words(row)
    if not words:
        raise HTTPException(status_code=400, detail="Transcript has no words")

    all_ids = {word.id for word in words}
    kept_ids = {word_id for word_id in payload.kept_word_ids if word_id in all_ids}
    if not kept_ids:
        raise HTTPException(
            status_code=400,
            detail="No valid words were kept; cannot render an empty timeline",
        )

    keep_ranges = _keep_ranges_from_deleted_words(
        words,
        row.duration_sec,
        kept_ids,
        context_sec_override=payload.context_sec,
        merge_gap_sec_override=payload.merge_gap_sec,
        min_removed_sec_override=payload.min_removed_sec,
    )
    _ranges, timeline_state = _apply_transcript_keep_ranges(
        session,
        project_id=project_id,
        asset_id=row.asset_id,
        duration_sec=float(row.duration_sec or 0.0),
        ranges=keep_ranges,
    )
    kept_words = [word for word in words if word.id in kept_ids]
    timeline_state, _captions_synced = _sync_existing_subtitles(
        session,
        project_id=project_id,
        asset_id=row.asset_id,
        words=kept_words,
    )
    kept_count = len(kept_ids)
    removed_count = max(len(words) - kept_count, 0)
    if removed_count:
        kept_items = [
            item
            for item in raw_items
            if _is_blank_region(item) or str(item.get("id") or "") in kept_ids
        ]
        _persist_transcript_items(
            row,
            session=session,
            items=kept_items,
            source=_manual_transcript_source(row.source),
        )
    else:
        row.updated_at = _utcnow()
        session.add(row)
        session.commit()

    return TranscriptCutResponse(
        project_id=project_id,
        transcript_id=row.id,
        kept_word_count=kept_count,
        removed_word_count=removed_count,
        timeline=timeline_state,
    )


@router.patch("/{transcript_id}/words/{word_id}", response_model=TranscriptEditResponse)
def update_word_text(
    transcript_id: str,
    word_id: str,
    payload: dict,
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> TranscriptEditResponse:
    new_text = str(payload.get("text") or "").strip()
    if not new_text:
        raise HTTPException(status_code=400, detail="Word text cannot be empty")

    _project, row = _get_project_and_transcript(
        session, project_id=project_id, transcript_id=transcript_id
    )
    updated_items = _apply_range_update_items(
        _load_raw_items(row),
        duration_sec=float(row.duration_sec or 0.0),
        start_word_id=word_id,
        end_word_id=word_id,
        mode="replace",
        text=new_text,
    )
    _persist_transcript_items(
        row,
        session=session,
        items=updated_items,
        source=_manual_transcript_source(row.source),
    )
    full_transcript = _to_response(row, word_limit=None)
    transcript_response = _to_response(row, word_limit=_effective_word_limit(None))
    timeline_state, captions_synced = _sync_existing_subtitles(
        session,
        project_id=project_id,
        asset_id=row.asset_id,
        words=full_transcript.words,
    )
    return TranscriptEditResponse(
        transcript=transcript_response,
        timeline=timeline_state,
        captions_synced=captions_synced,
    )


@router.patch("/{transcript_id}/range", response_model=TranscriptEditResponse)
def update_transcript_range(
    transcript_id: str,
    payload: TranscriptRangeUpdateRequest,
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> TranscriptEditResponse:
    _project, row = _get_project_and_transcript(
        session, project_id=project_id, transcript_id=transcript_id
    )
    raw_items = _load_raw_items(row)
    duration_sec = float(row.duration_sec or 0.0)

    # Capture the full word list BEFORE deletion so we can compute cut ranges.
    original_words = _load_words(row) if payload.mode == "delete" else []

    updated_items = _apply_range_update_items(
        raw_items,
        duration_sec=duration_sec,
        start_word_id=payload.start_word_id,
        end_word_id=payload.end_word_id,
        mode=payload.mode,
        text=payload.text,
    )
    _persist_transcript_items(
        row,
        session=session,
        items=updated_items,
        source=_manual_transcript_source(row.source),
    )
    full_transcript = _to_response(row, word_limit=None)
    transcript_response = _to_response(row, word_limit=_effective_word_limit(None))

    # When deleting words, also update the video track clips so the video
    # timeline reflects the removal — not just subtitles.
    if payload.mode == "delete" and original_words and full_transcript.words:
        remaining_ids = {w.id for w in full_transcript.words}
        if remaining_ids and len(remaining_ids) < len(original_words):
            keep_ranges = _keep_ranges_from_deleted_words(
                original_words,
                duration_sec,
                remaining_ids,
                context_sec_override=0.0,
                merge_gap_sec_override=0.08,
                min_removed_sec_override=0.0,
            )
            _apply_transcript_keep_ranges(
                session,
                project_id=project_id,
                asset_id=row.asset_id,
                duration_sec=duration_sec,
                ranges=keep_ranges,
            )

    timeline_state, captions_synced = _sync_existing_subtitles(
        session,
        project_id=project_id,
        asset_id=row.asset_id,
        words=full_transcript.words,
    )
    return TranscriptEditResponse(
        transcript=transcript_response,
        timeline=timeline_state,
        captions_synced=captions_synced,
    )


@router.post("/{transcript_id}/restore", response_model=TranscriptEditResponse)
def restore_transcript_snapshot(
    transcript_id: str,
    payload: dict,
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> TranscriptEditResponse:
    _project, row = _get_project_and_transcript(
        session, project_id=project_id, transcript_id=transcript_id
    )
    raw_words = payload.get("words")
    if not isinstance(raw_words, list) or not raw_words:
        raise HTTPException(
            status_code=400, detail="Restore payload must include words"
        )

    restored_words: list[TranscriptWordPayload] = []
    for item in raw_words:
        try:
            word = TranscriptWord.model_validate(item)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="Restore words are invalid"
            ) from exc
        restored_words.append(
            TranscriptWordPayload(
                id=word.id,
                text=word.text,
                start_sec=word.start_sec,
                end_sec=word.end_sec,
                confidence=word.confidence,
                quality_score=word.quality_score,
                quality_label=word.quality_label,
                source_pass=word.source_pass,
            )
        )

    _persist_transcript_items(
        row,
        session=session,
        items=[_serialize_word(word) for word in restored_words],
        source=_manual_transcript_source(row.source),
    )

    timeline_payload = payload.get("timeline")
    timeline_row = get_timeline_row(session, project_id)
    if timeline_payload is not None:
        try:
            restored_timeline = TimelineState.model_validate(timeline_payload)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="Restore timeline is invalid"
            ) from exc
        timeline_row = save_timeline_state(
            session,
            timeline_row,
            restored_timeline,
            source="transcript_undo",
        )

    return TranscriptEditResponse(
        transcript=_to_response(row, word_limit=_effective_word_limit(None)),
        timeline=load_timeline_state(timeline_row),
        captions_synced=False,
    )
