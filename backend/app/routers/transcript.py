from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..database import get_session
from ..media_utils import probe_duration_seconds
from ..config import get_settings
from ..models import MediaAsset, Project, Transcript
from ..schemas import (
    OperationPayload,
    TranscriptCutRequest,
    TranscriptCutResponse,
    TranscriptGenerateRequest,
    TranscriptGenerateResponse,
    TranscriptResponse,
    TranscriptWord,
)
from ..storage import storage
from ..timeline_service import apply_operation, get_timeline_row, load_timeline_state, save_timeline_state
from ..transcription_service import TranscriptPayload, generate_transcript

router = APIRouter(prefix="/api/v1/transcript", tags=["transcript"])
settings = get_settings()


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


def _load_words(row: Transcript) -> list[TranscriptWord]:
    try:
        payload = json.loads(row.words_json or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Stored transcript words are invalid") from exc
    words: list[TranscriptWord] = []
    for item in payload:
        words.append(TranscriptWord.model_validate(item))
    return words


def _to_response(row: Transcript) -> TranscriptResponse:
    return TranscriptResponse(
        id=row.id,
        project_id=row.project_id,
        asset_id=row.asset_id,
        source=row.source,
        language=row.language,
        text=row.text,
        words=_load_words(row),
        duration_sec=row.duration_sec,
        is_mock=row.is_mock,
        created_at=row.created_at.isoformat(),
    )


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

    ordered_words = sorted(words, key=lambda item: (float(item.start_sec), float(item.end_sec)))
    kept_words = [word for word in ordered_words if word.id in kept_ids]
    if not kept_words:
        raise HTTPException(status_code=400, detail="No valid words were kept; cannot render an empty timeline")

    # Build delete intervals from contiguous deleted word runs. This avoids
    # collapsing untouched gaps across the full transcript after a small edit.
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

    # Post-speech trailing content: the transcript may not cover the full
    # video (music, silence, outro after speech ends). Keep this untranscribed
    # tail unless the final transcript word itself is kept.
    last_transcript_word = max(ordered_words, key=lambda w: float(w.end_sec))
    last_word_kept = last_transcript_word.id in kept_ids
    last_word_end = float(last_transcript_word.end_sec)
    trailing_gap = duration_sec - last_word_end
    cut_ranges: list[tuple[float, float]] = []
    for start_idx, end_idx in delete_runs:
        prev_kept = next(
            (ordered_words[idx] for idx in range(start_idx - 1, -1, -1) if ordered_words[idx].id in kept_ids),
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
            cut_start = 0.0
        else:
            cut_start = float(prev_kept.end_sec) + context_sec

        if next_kept is None:
            cut_end = duration_sec
            if not last_word_kept and trailing_gap > 1.0:
                cut_end = last_word_end
        else:
            cut_end = float(next_kept.start_sec) - context_sec

        cut_start = max(0.0, min(cut_start, duration_sec))
        cut_end = max(0.0, min(cut_end, duration_sec))
        if cut_end > cut_start:
            cut_ranges.append((cut_start, cut_end))

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
        (start, end)
        for start, end in merged_cuts
        if (end - start) >= min_removed_sec
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
    return normalized_ranges


def _apply_video_ranges(
    session: Session,
    *,
    project_id: str,
    asset_id: str,
    ranges: list[dict[str, float]],
) -> tuple[list[dict[str, float]], object]:
    if not ranges:
        raise HTTPException(status_code=400, detail="Deleting all transcript words would remove the entire video")

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


@router.post("/generate", response_model=TranscriptGenerateResponse)
def generate(
    payload: TranscriptGenerateRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> TranscriptGenerateResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    asset = session.exec(
        select(MediaAsset).where(MediaAsset.id == payload.asset_id, MediaAsset.project_id == project_id)
    ).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Media asset not found")
    if asset.media_type != "video":
        raise HTTPException(status_code=400, detail="Transcript generation requires a video asset")

    source_path = storage.resolve_upload_asset(asset.storage_path)
    duration_sec = float(asset.duration_sec) if asset.duration_sec is not None else (probe_duration_seconds(source_path) or 0.0)
    if duration_sec <= 0:
        raise HTTPException(status_code=400, detail="Could not determine video duration for transcript generation")
    if settings.max_transcribe_duration_sec > 0 and duration_sec > settings.max_transcribe_duration_sec:
        raise HTTPException(
            status_code=400,
            detail=(
                "Video exceeds configured transcription limit "
                f"({settings.max_transcribe_duration_sec:.0f} seconds)"
            ),
        )

    try:
        transcript_payload = generate_transcript(source_path, duration_sec)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not transcript_payload.words:
        raise HTTPException(status_code=500, detail="Transcript generation returned no words")

    row = _store_transcript(
        session,
        project_id=project_id,
        asset_id=asset.id,
        duration_sec=duration_sec,
        payload=transcript_payload,
    )

    _ranges, timeline_state = _apply_video_ranges(
        session,
        project_id=project_id,
        asset_id=asset.id,
        ranges=[{"start_sec": 0.0, "end_sec": round(duration_sec, 3)}],
    )

    return TranscriptGenerateResponse(
        transcript=_to_response(row),
        timeline=timeline_state,
    )


@router.get("", response_model=TranscriptResponse)
def get_latest(
    project_id: str,
    transcript_id: str | None = None,
    session: Session = Depends(get_session),
) -> TranscriptResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    query = select(Transcript).where(Transcript.project_id == project_id)
    if transcript_id:
        query = query.where(Transcript.id == transcript_id)
    row = session.exec(query.order_by(Transcript.created_at.desc())).first()
    if not row:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return _to_response(row)


@router.post("/cut", response_model=TranscriptCutResponse)
def apply_text_cut(
    payload: TranscriptCutRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> TranscriptCutResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    row = session.exec(
        select(Transcript).where(Transcript.id == payload.transcript_id, Transcript.project_id == project_id)
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Transcript not found")

    words = _load_words(row)
    if not words:
        raise HTTPException(status_code=400, detail="Transcript has no words")

    all_ids = {word.id for word in words}
    kept_ids = {word_id for word_id in payload.kept_word_ids if word_id in all_ids}
    if not kept_ids:
        raise HTTPException(status_code=400, detail="No valid words were kept; cannot render an empty timeline")

    keep_ranges = _keep_ranges_from_deleted_words(
        words,
        row.duration_sec,
        kept_ids,
        context_sec_override=payload.context_sec,
        merge_gap_sec_override=payload.merge_gap_sec,
        min_removed_sec_override=payload.min_removed_sec,
    )
    _ranges, timeline_state = _apply_video_ranges(
        session,
        project_id=project_id,
        asset_id=row.asset_id,
        ranges=keep_ranges,
    )
    kept_count = len(kept_ids)
    removed_count = max(len(words) - kept_count, 0)
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


@router.patch("/{transcript_id}/words/{word_id}")
def update_word_text(
    transcript_id: str,
    word_id: str,
    payload: dict,
    project_id: str,
    session: Session = Depends(get_session),
) -> dict:
    """Update the text of a single word in a transcript (for inline editing)."""
    new_text = payload.get("text", "").strip()
    if not new_text:
        raise HTTPException(status_code=400, detail="Word text cannot be empty")

    row = session.exec(
        select(Transcript).where(Transcript.id == transcript_id, Transcript.project_id == project_id)
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Transcript not found")

    try:
        words_data = json.loads(row.words_json or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Stored transcript words are invalid") from exc

    updated = False
    for word in words_data:
        if word.get("id") == word_id:
            word["text"] = new_text
            updated = True
            break

    if not updated:
        raise HTTPException(status_code=404, detail="Word not found in transcript")

    row.words_json = _json_dumps(words_data)
    row.text = " ".join(w.get("text", "") for w in words_data)
    row.updated_at = _utcnow()
    session.add(row)
    session.commit()

    return {"ok": True}
