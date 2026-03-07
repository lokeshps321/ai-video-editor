from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..database import get_session
from ..jobs import create_job, enqueue_render_job
from ..media_utils import detect_silence_ranges, probe_duration_seconds
from ..models import Job, MediaAsset, Project, Transcript
from ..schemas import (
    ExportSettings,
    JobResponse,
    OperationPayload,
    TranscriptWord,
    VibeActionRequest,
    VibeActionResponse,
)
from ..storage import storage
from ..timeline_service import apply_operation, get_timeline_row, load_timeline_state, save_timeline_state
from ..transcription_service import (
    TranscriptPayload,
    TranscriptWordPayload,
    generate_transcript,
    sanitize_transcript_words,
)

router = APIRouter(prefix="/api/v1/vibe", tags=["vibe"])

_TRIM_EDGE_PUNCT_RE = re.compile(r"(^[^\w']+|[^\w']+$)")
_FILLER_SINGLE_WORDS_CONSERVATIVE: set[str] = {"um", "uh", "uhm", "umm", "hmm", "hm", "ah", "er", "eh"}
_FILLER_SINGLE_WORDS_AGGRESSIVE: set[str] = {"like", "basically", "literally", "actually", "right"}
_FILLER_MULTI_WORD_PHRASES: tuple[tuple[str, ...], ...] = (
    ("you", "know"),
    ("i", "mean"),
    ("sort", "of"),
    ("kind", "of"),
    ("so", "yeah"),
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_word_token(text: str) -> str:
    lowered = text.strip().lower()
    return _TRIM_EDGE_PUNCT_RE.sub("", lowered)


def _detect_filler_ranges(words: list[TranscriptWord]) -> list[tuple[float, float]]:
    if not words:
        return []

    tokens = [_normalize_word_token(word.text) for word in words]
    ranges: list[tuple[float, float]] = []
    consumed_indices: set[int] = set()

    # Prefer phrase-level matches ("you know", "i mean", etc.) to avoid double counting.
    idx = 0
    while idx < len(words):
        matched = False
        for phrase in _FILLER_MULTI_WORD_PHRASES:
            phrase_len = len(phrase)
            if idx + phrase_len > len(words):
                continue
            if tuple(tokens[idx : idx + phrase_len]) != phrase:
                continue
            ranges.append((float(words[idx].start_sec), float(words[idx + phrase_len - 1].end_sec)))
            for phrase_index in range(idx, idx + phrase_len):
                consumed_indices.add(phrase_index)
            idx += phrase_len
            matched = True
            break
        if not matched:
            idx += 1

    filler_single_words = set(_FILLER_SINGLE_WORDS_CONSERVATIVE)
    if _env_bool("TRANSCRIBE_FILLER_AGGRESSIVE_SINGLE_WORDS", False):
        filler_single_words.update(_FILLER_SINGLE_WORDS_AGGRESSIVE)

    for idx, token in enumerate(tokens):
        if not token or idx in consumed_indices:
            continue
        if token in filler_single_words:
            ranges.append((float(words[idx].start_sec), float(words[idx].end_sec)))

    if not ranges:
        return []

    ranges.sort(key=lambda item: item[0])
    merged: list[tuple[float, float]] = []
    for start_sec, end_sec in ranges:
        if not merged:
            merged.append((start_sec, end_sec))
            continue
        prev_start, prev_end = merged[-1]
        if start_sec <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end_sec))
            continue
        merged.append((start_sec, end_sec))
    return merged


def _min_expected_words(duration_sec: float) -> int:
    raw = os.getenv("TRANSCRIBE_MIN_WORDS_PER_SEC", "0.45")
    try:
        words_per_sec = max(float(raw), 0.05)
    except (TypeError, ValueError):
        words_per_sec = 0.45
    return max(8, int(round(max(duration_sec, 1.0) * words_per_sec)))


def _to_job_response(job: Job) -> JobResponse:
    return JobResponse(
        id=job.id,
        project_id=job.project_id,
        kind=job.kind,
        status=job.status,
        progress=job.progress,
        output_path=job.output_path,
        error=job.error,
    )


def _pick_video_asset(session: Session, *, project_id: str, asset_id: str | None) -> MediaAsset:
    query = select(MediaAsset).where(MediaAsset.project_id == project_id, MediaAsset.media_type == "video")
    if asset_id:
        row = session.exec(query.where(MediaAsset.id == asset_id)).first()
    else:
        row = session.exec(query.order_by(MediaAsset.created_at.desc())).first()
    if not row:
        raise HTTPException(status_code=404, detail="No video asset found for this project")
    return row


def _load_words(row: Transcript) -> list[TranscriptWord]:
    try:
        payload = json.loads(row.words_json or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Stored transcript words are invalid") from exc
    words = [TranscriptWord.model_validate(item) for item in payload]
    normalized = sanitize_transcript_words(
        [
            TranscriptWordPayload(
                id=item.id,
                text=item.text,
                start_sec=float(item.start_sec),
                end_sec=float(item.end_sec),
                confidence=item.confidence,
            )
            for item in words
        ],
        float(row.duration_sec or 0.0),
        apply_filters=True,
    )
    return [
        TranscriptWord(
            id=item.id,
            text=item.text,
            start_sec=item.start_sec,
            end_sec=item.end_sec,
            confidence=item.confidence,
        )
        for item in normalized
    ]


def _coerce_client_words(raw_words: object, duration_sec: float) -> list[TranscriptWord]:
    if not isinstance(raw_words, list):
        return []
    incoming: list[TranscriptWordPayload] = []
    for item in raw_words:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        try:
            start_sec = float(item.get("start_sec", 0.0))
            end_sec = float(item.get("end_sec", 0.0))
        except (TypeError, ValueError):
            continue
        if end_sec <= start_sec:
            end_sec = start_sec + 0.05
        confidence_raw = item.get("confidence")
        confidence: float | None
        try:
            confidence = float(confidence_raw) if confidence_raw is not None else None
        except (TypeError, ValueError):
            confidence = None
        incoming.append(
            TranscriptWordPayload(
                id=str(item.get("id") or uuid4()),
                text=text,
                start_sec=start_sec,
                end_sec=end_sec,
                confidence=confidence,
            )
        )
    if not incoming:
        return []
    normalized = sanitize_transcript_words(incoming, duration_sec)
    return [
        TranscriptWord(
            id=item.id,
            text=item.text,
            start_sec=item.start_sec,
            end_sec=item.end_sec,
            confidence=item.confidence,
        )
        for item in normalized
    ]


def _duration_from_asset(asset: MediaAsset, source_path: str) -> float:
    duration_sec = float(asset.duration_sec) if asset.duration_sec is not None else (probe_duration_seconds(source_path) or 0.0)
    if duration_sec <= 0:
        raise HTTPException(status_code=400, detail="Could not determine video duration")
    return duration_sec


def _store_transcript(
    session: Session,
    *,
    project_id: str,
    asset_id: str,
    duration_sec: float,
    payload: TranscriptPayload,
) -> Transcript:
    row = Transcript(
        project_id=project_id,
        asset_id=asset_id,
        source=payload.source,
        language=payload.language,
        text=payload.text,
        words_json=_json_dumps([word.__dict__ for word in payload.words]),
        duration_sec=round(duration_sec, 3),
        is_mock=payload.is_mock,
        updated_at=_utcnow(),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def _latest_transcript(session: Session, *, project_id: str, asset_id: str) -> Transcript | None:
    return session.exec(
        select(Transcript)
        .where(Transcript.project_id == project_id, Transcript.asset_id == asset_id)
        .order_by(Transcript.created_at.desc())
    ).first()


def _get_or_create_transcript(
    session: Session,
    *,
    project_id: str,
    asset: MediaAsset,
    source_path: str,
    duration_sec: float,
) -> Transcript:
    allow_mock_fallback = _env_bool("TRANSCRIBE_ALLOW_MOCK_FALLBACK", True)
    existing = _latest_transcript(session, project_id=project_id, asset_id=asset.id)
    if existing and existing.is_mock and not allow_mock_fallback:
        existing = None
    if existing:
        existing_words = _load_words(existing)
        if existing_words:
            regenerate_low_quality = _env_bool("TRANSCRIBE_REGENERATE_LOW_QUALITY", True)
            if not regenerate_low_quality or len(existing_words) >= _min_expected_words(duration_sec):
                return existing
    try:
        # Caption workflow should remain usable even if a cloud ASR provider is down.
        fast_mode = _env_bool("TRANSCRIBE_FAST_MODE", False)
        try:
            payload = generate_transcript(
                source_path,
                duration_sec,
                allow_mock_fallback=allow_mock_fallback,
                fast_mode=fast_mode,
            )
        except TypeError as exc:
            if "allow_mock_fallback" not in str(exc) and "fast_mode" not in str(exc):
                raise
            payload = generate_transcript(source_path, duration_sec)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if payload.is_mock and not allow_mock_fallback:
        raise HTTPException(
            status_code=503,
            detail="Transcription provider unavailable and mock fallback is disabled. Retry after restoring ASR connectivity.",
        )
    if not payload.words:
        raise HTTPException(status_code=500, detail="Transcript generation returned no words")
    return _store_transcript(
        session,
        project_id=project_id,
        asset_id=asset.id,
        duration_sec=duration_sec,
        payload=payload,
    )


def _apply_single_operation(session: Session, *, project_id: str, operation: OperationPayload):
    timeline = get_timeline_row(session, project_id)
    state = load_timeline_state(timeline)
    try:
        apply_operation(state, operation)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    timeline = save_timeline_state(
        session,
        timeline,
        state,
        source=operation.source,
        operation=operation,
    )
    return load_timeline_state(timeline)


def _ensure_selected_asset_on_timeline(session: Session, *, project_id: str, asset_id: str, duration_sec: float) -> None:
    timeline = get_timeline_row(session, project_id)
    state = load_timeline_state(timeline)
    video_track = next((track for track in state.tracks if track.kind == "video"), None)
    has_selected_clip = bool(video_track and any(clip.asset_id == asset_id for clip in video_track.clips))
    if has_selected_clip:
        return
    op = OperationPayload(
        op_type="replace_video_track_clips",
        source="ui",
        params={
            "asset_id": asset_id,
            "ranges": [{"start_sec": 0.0, "end_sec": round(duration_sec, 3)}],
            "clear_audio_tracks": True,
        },
    )
    _apply_single_operation(session, project_id=project_id, operation=op)


def _queue_preview(session: Session, *, project_id: str) -> Job:
    # Always queue a fresh preview for vibe actions because timeline state has
    # just changed (e.g., subtitles). Reusing an older in-flight preview can
    # return a render built from stale pre-action timeline data.
    job = create_job(session, project_id, kind="preview")
    request = ExportSettings(format="mp4", aspect_ratio="16:9", resolution="720p", fps=24, quality="low")
    enqueue_render_job(job.id, request)
    return job


def _keep_ranges_from_silence(duration_sec: float, silences: list[tuple[float, float]], *, min_silence_sec: float) -> list[dict[str, float]]:
    if not silences:
        return [{"start_sec": 0.0, "end_sec": round(duration_sec, 3)}]

    merged: list[tuple[float, float]] = []
    for start_sec, end_sec in sorted(silences):
        if end_sec <= start_sec:
            continue
        if (end_sec - start_sec) < min_silence_sec:
            continue
        if not merged:
            merged.append((start_sec, end_sec))
            continue
        prev_start, prev_end = merged[-1]
        if start_sec <= prev_end + 0.05:
            merged[-1] = (prev_start, max(prev_end, end_sec))
        else:
            merged.append((start_sec, end_sec))

    if not merged:
        return [{"start_sec": 0.0, "end_sec": round(duration_sec, 3)}]

    ranges: list[dict[str, float]] = []
    cursor = 0.0
    for start_sec, end_sec in merged:
        start = max(0.0, min(start_sec, duration_sec))
        end = max(0.0, min(end_sec, duration_sec))
        if start > cursor + 0.08:
            ranges.append({"start_sec": round(cursor, 3), "end_sec": round(start, 3)})
        cursor = max(cursor, end)
    if duration_sec > cursor + 0.08:
        ranges.append({"start_sec": round(cursor, 3), "end_sec": round(duration_sec, 3)})
    return ranges or [{"start_sec": 0.0, "end_sec": round(duration_sec, 3)}]


@router.post("/apply", response_model=VibeActionResponse)
def apply_vibe_action(
    payload: VibeActionRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> VibeActionResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    asset = _pick_video_asset(session, project_id=project_id, asset_id=payload.asset_id)
    source_path = storage.resolve_upload_asset(asset.storage_path)
    duration_sec = _duration_from_asset(asset, source_path)

    transcript_row: Transcript | None = None
    details: str | None = None

    if payload.action == "add_subtitles":
        _ensure_selected_asset_on_timeline(
            session,
            project_id=project_id,
            asset_id=asset.id,
            duration_sec=duration_sec,
        )
        client_words = _coerce_client_words(payload.options.get("words"), duration_sec)
        if client_words:
            words = client_words
            transcript_row = _latest_transcript(session, project_id=project_id, asset_id=asset.id)
        else:
            transcript_row = _get_or_create_transcript(
                session,
                project_id=project_id,
                asset=asset,
                source_path=source_path,
                duration_sec=duration_sec,
            )
            words = _load_words(transcript_row)
        subtitle_style = str(payload.options.get("style", "hormozi_bold"))
        op_params: dict[str, object] = {
            "asset_id": asset.id,
            "words": [word.model_dump() for word in words],
            "style": subtitle_style,
            "clear_existing": True,
        }
        if "caption_styles" in payload.options:
            op_params["caption_styles"] = payload.options.get("caption_styles")
        for key in (
            "max_words_per_caption",
            "max_chars_per_caption",
            "max_gap_sec",
            "max_caption_duration_sec",
            "max_caption_display_sec",
            "caption_guard_sec",
            "max_caption_display_hard_cap_factor",
        ):
            if key in payload.options:
                op_params[key] = payload.options.get(key)
        if "max_caption_overlays" in payload.options:
            op_params["max_caption_overlays"] = payload.options.get("max_caption_overlays")
        else:
            # Safety cap for long renders: too many drawtext filters can be OOM-killed.
            op_params["max_caption_overlays"] = max(60, min(140, int(duration_sec * 0.45)))
        op = OperationPayload(
            op_type="set_subtitles",
            source="ui",
            params=op_params,
        )
        timeline_state = _apply_single_operation(session, project_id=project_id, operation=op)
        caption_count = 0
        first_caption_at: float | None = None
        for track in timeline_state.tracks:
            if track.kind != "video":
                continue
            for clip in track.clips:
                overlays = clip.text_overlays
                caption_count += len(overlays)
                for overlay in overlays:
                    absolute_start = max(float(clip.timeline_start_sec) + float(overlay.start_sec), 0.0)
                    if first_caption_at is None or absolute_start < first_caption_at:
                        first_caption_at = absolute_start
        if caption_count <= 0:
            raise HTTPException(
                status_code=400,
                detail="No captions were generated from transcript words. Regenerate transcript and try Add Captions again.",
            )
        if first_caption_at is not None:
            details = f"Added {caption_count} caption overlay blocks. First caption at {first_caption_at:.2f}s."
        else:
            details = f"Added {caption_count} caption overlay blocks."
    elif payload.action == "auto_cut_pauses":
        threshold = float(payload.options.get("silence_threshold_db", -35.0))
        min_silence_sec = float(payload.options.get("min_silence_sec", 0.35))
        min_pause_sec = float(payload.options.get("min_pause_sec", 0.4))
        remove_fillers = _coerce_bool(payload.options.get("remove_filler_words"), True)

        # --- Hybrid: transcript gaps + silence detection + filler words ---
        transcript_row = _get_or_create_transcript(
            session,
            project_id=project_id,
            asset=asset,
            source_path=source_path,
            duration_sec=duration_sec,
        )
        words = _load_words(transcript_row)

        # Detect pauses from transcript word gaps
        transcript_pauses: list[tuple[float, float]] = []
        filler_ranges: list[tuple[float, float]] = []
        if words:
            for i in range(len(words) - 1):
                gap_start = float(words[i].end_sec)
                gap_end = float(words[i + 1].start_sec)
                if (gap_end - gap_start) >= min_pause_sec:
                    transcript_pauses.append((gap_start, gap_end))
            # Detect filler words
            if remove_fillers:
                filler_ranges = _detect_filler_ranges(words)

        # Also get silence-detected pauses (catches non-speech silence)
        silences = detect_silence_ranges(source_path, noise_db=threshold, min_silence_sec=min_silence_sec)

        # Merge all pause sources
        all_pauses: list[tuple[float, float]] = sorted(
            set(silences + transcript_pauses + filler_ranges),
            key=lambda r: r[0],
        )

        ranges = _keep_ranges_from_silence(duration_sec, all_pauses, min_silence_sec=min(min_silence_sec, 0.1))
        op = OperationPayload(
            op_type="replace_video_track_clips",
            source="ui",
            params={
                "asset_id": asset.id,
                "ranges": ranges,
                "clear_audio_tracks": True,
            },
        )
        timeline_state = _apply_single_operation(session, project_id=project_id, operation=op)
        removed_sec = max(duration_sec - timeline_state.duration_sec, 0.0)
        filler_note = f" Removed {len(filler_ranges)} filler word(s)." if filler_ranges else ""
        details = f"Removed {removed_sec:.2f}s of pauses/silence.{filler_note}"
    elif payload.action == "trim_start_end":
        threshold = float(payload.options.get("silence_threshold_db", -35.0))
        min_silence_sec = float(payload.options.get("min_silence_sec", 0.25))
        silences = detect_silence_ranges(source_path, noise_db=threshold, min_silence_sec=min_silence_sec)
        start_sec = 0.0
        end_sec = duration_sec
        if silences and silences[0][0] <= 0.08:
            start_sec = min(max(silences[0][1], 0.0), duration_sec)
        if silences and silences[-1][1] >= (duration_sec - 0.08):
            end_sec = max(min(silences[-1][0], duration_sec), start_sec + 0.08)

        transcript_row = _latest_transcript(session, project_id=project_id, asset_id=asset.id)
        if transcript_row is None:
            transcript_row = _get_or_create_transcript(
                session,
                project_id=project_id,
                asset=asset,
                source_path=source_path,
                duration_sec=duration_sec,
            )
        words = _load_words(transcript_row)
        if words:
            first_word = words[0]
            last_word = words[-1]
            start_sec = max(start_sec, max(first_word.start_sec - 0.06, 0.0))
            end_sec = min(end_sec, min(last_word.end_sec + 0.06, duration_sec))
        if end_sec <= start_sec + 0.15:
            start_sec = 0.0
            end_sec = duration_sec

        op = OperationPayload(
            op_type="replace_video_track_clips",
            source="ui",
            params={
                "asset_id": asset.id,
                "ranges": [{"start_sec": round(start_sec, 3), "end_sec": round(end_sec, 3)}],
                "clear_audio_tracks": True,
            },
        )
        timeline_state = _apply_single_operation(session, project_id=project_id, operation=op)
        trimmed = max(duration_sec - (end_sec - start_sec), 0.0)
        details = f"Trimmed {trimmed:.2f}s from start/end dead space."
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported action: {payload.action}")

    preview_job = _queue_preview(session, project_id=project_id)
    return VibeActionResponse(
        project_id=project_id,
        action=payload.action,
        transcript_id=transcript_row.id if transcript_row else None,
        details=details,
        timeline=timeline_state,
        preview_job=_to_job_response(preview_job),
    )
