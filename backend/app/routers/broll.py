from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import subprocess
import threading
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, delete, select

from ..broll_ai_service import expand_broll_queries, rerank_broll_candidates
from ..broll_external_service import (
    ExternalBrollCandidate,
    search_external_broll_candidates,
)
from ..broll_generative_service import generate_generative_broll_candidates
from ..broll_llm_service import (
    build_broll_search_strategy,
    infer_broll_domain_context,
    llm_rerank_broll_candidates,
)
from ..broll_planner_service import plan_broll
from ..config import get_settings

logger = logging.getLogger(__name__)
from ..database import engine, get_session
from ..deps import get_current_user
from ..jobs import create_job, find_recent_active_job, set_job_status
from ..media_utils import probe_duration_seconds, probe_stream_flags
from ..models import (
    BrollCandidate,
    BrollChoice,
    BrollPlan,
    BrollPlanBeat,
    BrollSlot,
    Job,
    MediaAsset,
    Project,
    Transcript,
)
from ..schemas import (
    BrollAutoApplyRequest,
    BrollAutoApplyResponse,
    BrollAutoApplySkipSummary,
    BrollCandidateResponse,
    BrollChooseRequest,
    BrollConfigResponse,
    BrollCoverageSectionResponse,
    BrollPlanBeatResponse,
    BrollPlanRequest,
    BrollPlanResponse,
    BrollRejectRequest,
    BrollRerollRequest,
    BrollSlotResponse,
    BrollSuggestRequest,
    BrollSuggestResponse,
    BrollSyncRequest,
    BrollSyncResponse,
    BrollUndoResponse,
    Clip,
    JobResponse,
    OperationPayload,
)
from ..storage import storage
from ..timeline_service import (
    apply_operation,
    get_timeline_row,
    load_timeline_state,
    save_timeline_state,
)

router = APIRouter(prefix="/api/v1/broll", tags=["broll"])
settings = get_settings()

# --- split-out helper modules (re-exported for a stable surface) ---
from ._broll_constants import (  # noqa: F401
    _BROLL_TX_SLOT_ID,
    _LOCAL_SHOT_STYLE_CUES,
    _LOCAL_VISUAL_INTENT_CUES,
    _NEGATIVE_ENERGY_WORDS,
    _POSITIVE_ENERGY_WORDS,
    _SENTENCE_END_RE,
    _SEQUENCE_MEMORY,
    _STOP_WORDS,
    _THREE_WAYS_SHOTS,
    _VISUAL_INTENT_QUERY_MODE,
    _WORD_RE,
)
from ._broll_util import (  # noqa: F401
    _clamp,
    _filename_tokens,
    _is_vertical_project,
    _json_dumps,
    _parse_anchor_word_ids,
    _parse_asset_metadata,
    _parse_reason_json,
    _utcnow,
)
from ._broll_media import (  # noqa: F401
    _analyze_center_visual_risk,
    _build_vertical_crop,
    _build_vertical_crop_keyframes,
    _detect_focus_track,
    _detect_focus_x_ratio,
    _download_external_video,
    _ensure_asset_focus_metadata,
    _extract_audio_transients,
    _find_existing_asset_for_source_url,
    _materialize_candidate_asset,
    _probe_video_dimensions,
    _resolve_asset_video_path,
    _safe_filename_from_url,
    _snap_chunks_to_audio_grid,
    _snap_time_to_transient,
    _text_safety_preset_from_metadata,
)


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


def _broll_suggest_result_path(job_id: str) -> Path:
    folder = storage.tmp_root / "broll-suggest-jobs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{job_id}.json"


def _suggest_error_message(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        detail = exc.detail
        if isinstance(detail, str):
            return detail
        return str(detail)
    return str(exc)


def _require_project(session: Session, project_id: str) -> Project:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _load_transcript_words(row: Transcript) -> list[dict[str, object]]:
    try:
        payload = json.loads(row.words_json or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail="Stored transcript words are invalid"
        ) from exc

    words: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            word_id = str(item["id"])
            text = str(item["text"]).strip()
            start_sec = float(item["start_sec"])
            end_sec = float(item["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if not text or end_sec <= start_sec:
            continue
        words.append(
            {
                "id": word_id,
                "text": text,
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )
    words.sort(key=lambda item: float(item["start_sec"]))
    return words


def _asset_metadata_text(asset: MediaAsset) -> str:
    metadata = _parse_asset_metadata(asset)
    parts = [asset.filename]
    for key in ("title", "description", "tags", "keywords"):
        value = metadata.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    return " ".join(part for part in parts if part).strip()


def _to_plan_beat_response(row: BrollPlanBeat) -> BrollPlanBeatResponse:
    try:
        anchor_word_ids = json.loads(row.anchor_word_ids_json or "[]")
    except json.JSONDecodeError:
        anchor_word_ids = []
    try:
        query_hints = json.loads(row.query_hints_json or "[]")
    except json.JSONDecodeError:
        query_hints = []
    try:
        metadata = json.loads(row.metadata_json or "{}")
    except json.JSONDecodeError:
        metadata = {}
    return BrollPlanBeatResponse(
        id=row.id,
        beat_index=row.beat_index,
        start_sec=row.start_sec,
        end_sec=row.end_sec,
        timeline_start_sec=row.timeline_start_sec,
        timeline_end_sec=row.timeline_end_sec,
        section_label=row.section_label,
        intent_label=row.intent_label,
        source_strategy=row.source_strategy,
        shot_style=row.shot_style,
        should_place=row.should_place,
        confidence=row.confidence,
        rationale=row.rationale,
        concept_text=row.concept_text,
        segment_text=row.segment_text,
        anchor_word_ids=anchor_word_ids if isinstance(anchor_word_ids, list) else [],
        query_hints=query_hints if isinstance(query_hints, list) else [],
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _load_plan_response(
    session: Session, plan_id: str, *, project_id: str
) -> BrollPlanResponse:
    plan = session.exec(
        select(BrollPlan).where(
            BrollPlan.id == plan_id, BrollPlan.project_id == project_id
        )
    ).first()
    if not plan:
        raise HTTPException(status_code=404, detail="B-roll plan not found")
    beats = list(
        session.exec(
            select(BrollPlanBeat)
            .where(
                BrollPlanBeat.plan_id == plan.id, BrollPlanBeat.project_id == project_id
            )
            .order_by(BrollPlanBeat.beat_index.asc(), BrollPlanBeat.created_at.asc())
        ).all()
    )
    try:
        coverage = json.loads(plan.coverage_json or "{}")
    except json.JSONDecodeError:
        coverage = {}
    uncovered_ranges = (
        coverage.get("uncovered_ranges", []) if isinstance(coverage, dict) else []
    )
    coverage_sections = (
        coverage.get("coverage_sections", []) if isinstance(coverage, dict) else []
    )
    return BrollPlanResponse(
        id=plan.id,
        project_id=plan.project_id,
        transcript_id=plan.transcript_id,
        plan_version=plan.plan_version,
        fallback_used=plan.fallback_used,
        planner_model=plan.planner_model,
        created_at=plan.created_at.isoformat(),
        beats=[_to_plan_beat_response(row) for row in beats],
        uncovered_ranges=uncovered_ranges if isinstance(uncovered_ranges, list) else [],
        coverage_sections=[
            BrollCoverageSectionResponse.model_validate(item)
            for item in coverage_sections
            if isinstance(item, dict)
        ],
    )


def _build_broll_plan(
    session: Session,
    *,
    project: Project,
    transcript: Transcript,
    payload: BrollPlanRequest,
) -> BrollPlanResponse:
    words = _load_transcript_words(transcript)
    if not words:
        raise HTTPException(status_code=400, detail="Transcript has no words")

    assets = list(
        session.exec(
            select(MediaAsset)
            .where(
                MediaAsset.project_id == project.id, MediaAsset.media_type == "video"
            )
            .order_by(MediaAsset.created_at.desc())
        ).all()
    )
    asset_payload = []
    if payload.include_project_assets:
        asset_payload = [
            {
                "id": asset.id,
                "filename": asset.filename,
                "metadata_text": _asset_metadata_text(asset),
            }
            for asset in assets
        ]
    timeline_state = load_timeline_state(get_timeline_row(session, project.id))
    video_clips = _video_track_clips_sorted(timeline_state)

    planner_result = plan_broll(
        words=words,
        transcript_text=transcript.text,
        transcript_duration_sec=float(transcript.duration_sec),
        max_slots=payload.max_slots,
        min_chunk_words=payload.min_chunk_words,
        assets=asset_payload,
        include_external_sources=payload.include_external_sources,
    )
    beats = planner_result.get("beats")
    if not isinstance(beats, list) or not beats:
        raise HTTPException(
            status_code=400, detail="Planner produced no usable B-roll beats"
        )

    plan = BrollPlan(
        project_id=project.id,
        transcript_id=transcript.id,
        plan_version=str(planner_result.get("plan_version") or "v1"),
        fallback_used=bool(planner_result.get("fallback_used", True)),
        planner_model=str(planner_result.get("planner_model"))
        if planner_result.get("planner_model")
        else None,
        request_json=_json_dumps(payload.model_dump(mode="json")),
        coverage_json=_json_dumps(planner_result.get("coverage") or {}),
    )
    session.add(plan)

    for idx, beat in enumerate(beats):
        if not isinstance(beat, dict):
            continue
        start_sec = round(float(beat.get("start_sec", 0.0)), 3)
        end_sec = round(float(beat.get("end_sec", start_sec + 0.5)), 3)
        timeline_window = _resolve_slot_timeline_window(
            start_sec, end_sec, video_clips=video_clips
        )
        timeline_start_sec = timeline_window[0] if timeline_window else None
        timeline_end_sec = timeline_window[1] if timeline_window else None
        session.add(
            BrollPlanBeat(
                plan_id=plan.id,
                project_id=project.id,
                transcript_id=transcript.id,
                beat_index=idx,
                start_sec=start_sec,
                end_sec=end_sec,
                timeline_start_sec=timeline_start_sec,
                timeline_end_sec=timeline_end_sec,
                section_label=str(beat.get("section_label") or "body"),
                intent_label=str(beat.get("intent_label") or "supporting_visual"),
                source_strategy=str(beat.get("source_strategy") or "local_first"),
                shot_style=str(beat.get("shot_style") or "medium"),
                should_place=bool(beat.get("should_place", True)),
                confidence=round(float(beat.get("confidence", 0.0)), 3),
                rationale=str(beat.get("rationale") or ""),
                concept_text=str(beat.get("concept_text") or ""),
                segment_text=str(beat.get("segment_text") or ""),
                anchor_word_ids_json=_json_dumps(beat.get("anchor_word_ids") or []),
                query_hints_json=_json_dumps(beat.get("query_hints") or []),
                metadata_json=_json_dumps(beat.get("metadata") or {}),
            )
        )
    session.commit()
    return _load_plan_response(session, plan.id, project_id=project.id)


def _resolve_broll_transcript(
    session: Session,
    *,
    project_id: str,
    transcript_id: str | None,
) -> Transcript:
    transcript_query = select(Transcript).where(Transcript.project_id == project_id)
    if transcript_id:
        transcript_query = transcript_query.where(Transcript.id == transcript_id)
    transcript = session.exec(
        transcript_query.order_by(Transcript.created_at.desc())
    ).first()
    if not transcript:
        raise HTTPException(
            status_code=404,
            detail="Transcript not found. Generate transcript before requesting B-roll.",
        )
    return transcript


def _plan_request_from_suggest(payload: BrollSuggestRequest) -> BrollPlanRequest:
    return BrollPlanRequest(
        transcript_id=payload.transcript_id,
        max_slots=payload.max_slots,
        min_chunk_words=payload.min_chunk_words,
        include_project_assets=payload.include_project_assets,
        include_external_sources=payload.include_external_sources,
    )


def _chunk_words(
    words: list[dict[str, object]],
    min_chunk_words: int,
    max_slots: int,
    *,
    min_chunk_duration_sec: float = 0.0,
    max_chunk_duration_sec: float = 4.0,
) -> list[dict[str, object]]:
    if not words:
        return []

    chunks: list[dict[str, object]] = []
    current: list[dict[str, object]] = []

    def flush(*, force_short: bool = False) -> None:
        nonlocal current
        if not current:
            return
        duration = float(current[-1]["end_sec"]) - float(current[0]["start_sec"])
        if len(current) < max(1, min_chunk_words):
            current = []
            return
        if (
            not force_short
            and min_chunk_duration_sec > 0
            and duration < min_chunk_duration_sec
        ):
            current = []
            return
        chunks.append(
            {
                "word_ids": [str(item["id"]) for item in current],
                "text": " ".join(str(item["text"]) for item in current),
                "start_sec": float(current[0]["start_sec"]),
                "end_sec": float(current[-1]["end_sec"]),
            }
        )
        current = []

    for idx, word in enumerate(words):
        prev = words[idx - 1] if idx > 0 else None
        if prev is not None:
            gap = float(word["start_sec"]) - float(prev["end_sec"])
            if gap > 1.1 and current:
                flush(force_short=True)

        current.append(word)
        token = str(word["text"]).strip()
        sentence_end = bool(_SENTENCE_END_RE.search(token))
        cap_reached = len(current) >= 16
        duration = float(current[-1]["end_sec"]) - float(current[0]["start_sec"])
        duration_reached = (
            max_chunk_duration_sec > 0 and duration >= max_chunk_duration_sec
        )
        min_duration_reached = (
            min_chunk_duration_sec <= 0 or duration >= min_chunk_duration_sec
        )
        if duration_reached or cap_reached or (sentence_end and min_duration_reached):
            flush(force_short=duration_reached or cap_reached)
            if len(chunks) >= max_slots:
                break

    if len(chunks) < max_slots:
        flush(force_short=True)
    return chunks[:max_slots]


def _extract_concepts(text: str) -> tuple[str, list[str]]:
    tokens = [token.lower() for token in _WORD_RE.findall(text)]
    filtered = [
        token for token in tokens if len(token) >= 3 and token not in _STOP_WORDS
    ]
    if not filtered:
        fallback = [
            token for token in tokens if len(token) >= 4 and token not in _STOP_WORDS
        ]
        filtered = fallback[:3]
    if not filtered:
        return ("general scene", ["general"])

    counts = Counter(filtered)
    seen: set[str] = set()
    ordered_unique: list[str] = []
    for token in filtered:
        if token in seen:
            continue
        seen.add(token)
        ordered_unique.append(token)

    ordered_unique.sort(key=lambda token: (-counts[token], filtered.index(token)))
    selected = ordered_unique[:4]
    return (" ".join(selected), selected)


def _focus_terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in _WORD_RE.findall(text)
        if len(token) >= 3 and token.lower() not in _STOP_WORDS
    }


def _local_duration_fit(
    candidate_duration: float | None, slot_duration: float
) -> float:
    if not candidate_duration or candidate_duration <= 0:
        return 0.45
    baseline = max(slot_duration, 0.6)
    delta = abs(candidate_duration - baseline)
    ratio = max(0.0, min(1.0, 1.0 - (delta / max(baseline * 2.2, 1.0))))
    return 0.25 + (ratio * 0.75)


def _local_intent_score(asset_terms: set[str], visual_intent: str | None) -> float:
    cues = _LOCAL_VISUAL_INTENT_CUES.get((visual_intent or "").strip().lower(), set())
    if not cues:
        return 0.5
    hits = len(asset_terms.intersection(cues))
    return max(0.0, min(1.0, 0.42 + min(hits * 0.18, 0.42)))


def _local_shot_score(asset_terms: set[str], shot_style: str | None) -> float:
    cues = _LOCAL_SHOT_STYLE_CUES.get((shot_style or "").strip().lower(), set())
    if not cues:
        return 0.52
    hits = len(asset_terms.intersection(cues))
    return max(0.0, min(1.0, 0.4 + min(hits * 0.2, 0.5)))


def _empty_sequence_state() -> dict[str, list[object]]:
    return {
        "asset_ids": [],
        "source_urls": [],
        "labels": [],
        "query_modes": [],
        "source_types": [],
        "signatures": [],
    }


def _push_recent_item(
    items: list[object], value: object, *, limit: int = _SEQUENCE_MEMORY
) -> None:
    if value in (None, "", []):
        return
    items.insert(0, value)
    del items[limit:]


def _candidate_label_key(
    source_label: str | None, asset_id: str | None, source_url: str | None
) -> str:
    base = source_label or asset_id or source_url or ""
    return " ".join(base.strip().lower().split())


def _candidate_signature_terms(
    *,
    source_label: str | None,
    reason: dict[str, object],
) -> set[str]:
    parts: list[str] = []
    if source_label:
        parts.append(source_label)
    for key in ("search_concept", "query", "page_url"):
        value = reason.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    keyword_hits = reason.get("keyword_hits")
    if isinstance(keyword_hits, list):
        parts.extend(str(item) for item in keyword_hits if str(item).strip())
    return _focus_terms(" ".join(parts))


def _sequence_diversify_candidates(
    candidates: list[
        tuple[str, str | None, str | None, str | None, float, dict[str, object]]
    ],
    *,
    sequence_state: dict[str, list[object]],
) -> list[tuple[str, str | None, str | None, str | None, float, dict[str, object]]]:
    if not candidates:
        return []

    diversified: list[
        tuple[str, str | None, str | None, str | None, float, dict[str, object]]
    ] = []
    for source_type, asset_id, source_url, source_label, score, reason in candidates:
        reason_payload = dict(reason) if isinstance(reason, dict) else {}
        breakdown = (
            dict(reason_payload.get("score_breakdown") or {})
            if isinstance(reason_payload.get("score_breakdown"), dict)
            else {}
        )
        diversity_multiplier = 1.0
        query_mode = str(reason_payload.get("query_mode") or "").strip().lower()
        label_key = _candidate_label_key(source_label, asset_id, source_url)
        signature = _candidate_signature_terms(
            source_label=source_label, reason=reason_payload
        )

        if asset_id and asset_id in sequence_state["asset_ids"]:
            diversity_multiplier *= 0.68
        elif source_url and source_url in sequence_state["source_urls"]:
            diversity_multiplier *= 0.72
        elif label_key and label_key in sequence_state["labels"]:
            diversity_multiplier *= 0.84

        if query_mode and sequence_state["query_modes"].count(query_mode) >= 2:
            diversity_multiplier *= 0.92
        if source_type and sequence_state["source_types"].count(source_type) >= 3:
            diversity_multiplier *= 0.95

        recent_overlap = 0.0
        if signature:
            for prior in sequence_state["signatures"]:
                if not isinstance(prior, set) or not prior:
                    continue
                overlap = len(signature.intersection(prior)) / max(
                    min(len(signature), 4), 1
                )
                recent_overlap = max(recent_overlap, overlap)
            if recent_overlap >= 0.75:
                diversity_multiplier *= 0.86
            elif recent_overlap >= 0.5:
                diversity_multiplier *= 0.93

        breakdown["diversity"] = round(_clamp(diversity_multiplier, 0.0, 1.0), 3)
        reason_payload["score_breakdown"] = breakdown
        if "confidence" in reason_payload:
            try:
                reason_payload["confidence"] = round(
                    _clamp(
                        float(reason_payload["confidence"])
                        * max(diversity_multiplier, 0.88),
                        0.0,
                        1.0,
                    ),
                    3,
                )
            except (TypeError, ValueError):
                pass
        diversified.append(
            (
                source_type,
                asset_id,
                source_url,
                source_label,
                round(_clamp(float(score) * diversity_multiplier, 0.0, 0.99), 3),
                reason_payload,
            )
        )

    diversified.sort(key=lambda item: item[4], reverse=True)
    return diversified


def _remember_sequence_candidate(
    sequence_state: dict[str, list[object]],
    candidate_row: tuple[
        str, str | None, str | None, str | None, float, dict[str, object]
    ]
    | None,
) -> None:
    if candidate_row is None:
        return
    source_type, asset_id, source_url, source_label, _score, reason = candidate_row
    if asset_id:
        _push_recent_item(sequence_state["asset_ids"], asset_id)
    if source_url:
        _push_recent_item(sequence_state["source_urls"], source_url)
    label_key = _candidate_label_key(source_label, asset_id, source_url)
    if label_key:
        _push_recent_item(sequence_state["labels"], label_key)
    query_mode = str(reason.get("query_mode") or "").strip().lower()
    if query_mode:
        _push_recent_item(sequence_state["query_modes"], query_mode)
    if source_type:
        _push_recent_item(sequence_state["source_types"], source_type)
    signature = _candidate_signature_terms(source_label=source_label, reason=reason)
    if signature:
        _push_recent_item(sequence_state["signatures"], signature)


def _resolve_slot_pacing(project: Project) -> tuple[float, float]:
    if _is_vertical_project(project):
        minimum = max(
            0.25,
            min(settings.broll_shortform_min_sec, settings.broll_shortform_max_sec),
        )
        maximum = max(minimum + 0.05, settings.broll_shortform_max_sec)
        return (minimum, maximum)
    return (0.35, 4.5)


def _clip_timeline_duration_sec(clip: Clip) -> float:
    return max(
        (float(clip.end_sec) - float(clip.start_sec)) / max(float(clip.speed), 0.01),
        0.0,
    )


def _video_track_clips_sorted(timeline_state: object) -> list[Clip]:
    tracks = getattr(timeline_state, "tracks", [])
    for track in tracks:
        if getattr(track, "kind", "") != "video":
            continue
        clips = [
            clip
            for clip in getattr(track, "clips", [])
            if float(clip.end_sec) > float(clip.start_sec)
        ]
        clips.sort(key=lambda item: float(item.timeline_start_sec))
        return clips
    return []


def _resolve_slot_timeline_window(
    slot_start_sec: float,
    slot_end_sec: float,
    *,
    video_clips: list[Clip],
) -> tuple[float, float] | None:
    if slot_end_sec <= slot_start_sec or not video_clips:
        return None

    best_overlap = 0.0
    best_timeline_start = 0.0
    best_timeline_end = 0.0

    for clip in video_clips:
        source_start = float(clip.start_sec)
        source_end = float(clip.end_sec)
        overlap_start = max(slot_start_sec, source_start)
        overlap_end = min(slot_end_sec, source_end)
        if overlap_end <= overlap_start:
            continue

        speed = max(float(clip.speed), 0.01)
        timeline_start = float(clip.timeline_start_sec) + (
            (overlap_start - source_start) / speed
        )
        timeline_end = float(clip.timeline_start_sec) + (
            (overlap_end - source_start) / speed
        )
        overlap = overlap_end - overlap_start

        if overlap > best_overlap + 1e-6:
            best_overlap = overlap
            best_timeline_start = timeline_start
            best_timeline_end = timeline_end
            continue
        if abs(overlap - best_overlap) <= 1e-6 and timeline_start < best_timeline_start:
            best_timeline_start = timeline_start
            best_timeline_end = timeline_end

    if best_overlap <= 0.0 or best_timeline_end <= best_timeline_start:
        return None
    return (
        round(max(0.0, best_timeline_start), 3),
        round(max(best_timeline_start + 0.06, best_timeline_end), 3),
    )


def _shot_variant_for_index(index: int) -> tuple[str, str]:
    if not _THREE_WAYS_SHOTS:
        return ("general", "general shot")
    return _THREE_WAYS_SHOTS[index % len(_THREE_WAYS_SHOTS)]


def _emotion_signal(text: str) -> float:
    tokens = [token.lower() for token in _WORD_RE.findall(text)]
    if not tokens:
        return 0.45
    pos = sum(1 for token in tokens if token in _POSITIVE_ENERGY_WORDS)
    neg = sum(1 for token in tokens if token in _NEGATIVE_ENERGY_WORDS)
    exclam = text.count("!") * 0.35
    question = text.count("?") * 0.15
    density = (pos + neg) / max(len(tokens), 1)
    base = 0.5 + (0.28 * density) + (0.05 * exclam) + (0.03 * question)
    # Negative content often wants slower, heavier pacing.
    base -= 0.12 * (neg / max(len(tokens), 1))
    return max(0.0, min(1.0, base))


def _n_arc_energy(position: float) -> float:
    pos = max(0.0, min(1.0, position))
    if pos < 0.20:
        return 0.95
    if pos < 0.42:
        return 0.55
    if pos < 0.68:
        return 0.35
    if pos < 0.88:
        return 0.72
    return 0.92


def _apply_emotional_pacing(
    chunks: list[dict[str, object]],
    *,
    min_chunk_sec: float,
    max_chunk_sec: float,
) -> list[dict[str, object]]:
    if not chunks:
        return []
    if max_chunk_sec <= min_chunk_sec:
        return chunks

    paced: list[dict[str, object]] = []
    total = max(len(chunks) - 1, 1)
    previous_end = 0.0
    for idx, chunk in enumerate(chunks):
        text = str(chunk.get("text") or "")
        arc_energy = _n_arc_energy(idx / total)
        emotion = _emotion_signal(text)
        energy = max(0.0, min(1.0, (0.58 * arc_energy) + (0.42 * emotion)))

        start_sec = max(previous_end, float(chunk["start_sec"]))
        raw_end_sec = max(start_sec + 0.06, float(chunk["end_sec"]))
        raw_duration = raw_end_sec - start_sec
        target_duration = max_chunk_sec - (energy * (max_chunk_sec - min_chunk_sec))
        duration = max(min_chunk_sec, min(max_chunk_sec, target_duration))
        duration = min(duration, max(raw_duration, min_chunk_sec))
        end_sec = start_sec + duration

        updated = dict(chunk)
        updated["start_sec"] = round(start_sec, 3)
        updated["end_sec"] = round(end_sec, 3)
        updated["emotion_energy"] = round(energy, 3)
        paced.append(updated)
        previous_end = end_sec
    return paced


def _rank_candidates(
    *,
    assets: list[MediaAsset],
    transcript_asset_id: str,
    concept_tokens: list[str],
    candidates_per_slot: int,
    slot_duration: float,
    shot_style: str | None = None,
    visual_intent: str | None = None,
) -> list[tuple[MediaAsset, float, dict[str, object]]]:
    if not assets:
        return []

    ranked: list[tuple[MediaAsset, float, dict[str, object]]] = []
    total = max(len(assets), 1)

    for idx, asset in enumerate(assets):
        metadata_text = _asset_metadata_text(asset)
        asset_terms = _focus_terms(metadata_text)
        filename_terms = _filename_tokens(asset.filename)
        concept_hits = [
            token
            for token in concept_tokens
            if token in asset_terms or token in filename_terms
        ]
        semantic_match = (
            (len(concept_hits) / max(len(concept_tokens), 1)) if concept_tokens else 0.0
        )
        semantic_score = min(semantic_match * 0.42, 0.42)

        diversity_score = 0.16 if asset.id != transcript_asset_id else 0.02
        duration_score = _local_duration_fit(asset.duration_sec, slot_duration) * 0.16
        intent_score = _local_intent_score(asset_terms, visual_intent)
        shot_score = _local_shot_score(asset_terms, shot_style)
        metadata_density = min(len(asset_terms), 8) / 8 if asset_terms else 0.0
        recency_ratio = 1.0 - (idx / total)
        recency_score = recency_ratio * 0.1
        primary_penalty = (
            0.84 if asset.id == transcript_asset_id and semantic_match < 0.5 else 1.0
        )

        score = max(
            0.0,
            min(
                1.0,
                round(
                    (
                        0.12
                        + semantic_score
                        + diversity_score
                        + duration_score
                        + (intent_score * 0.14)
                        + (shot_score * 0.1)
                        + (metadata_density * 0.08)
                        + recency_score
                    )
                    * primary_penalty,
                    3,
                ),
            ),
        )

        tags = ["project_asset", "visual_variety"]
        if concept_hits:
            tags.append("keyword_match")
            tags.append("metadata_match")
        if asset.id != transcript_asset_id:
            tags.append("not_primary_asset")
        else:
            tags.append("primary_asset")
        if shot_style:
            tags.append(f"shot_{shot_style}")
        if visual_intent:
            tags.append(f"intent_{visual_intent}")

        ranked.append(
            (
                asset,
                score,
                {
                    "tags": tags,
                    "breakdown": {
                        "semantic_score": round(semantic_score, 3),
                        "diversity_score": round(diversity_score, 3),
                        "duration_score": round(duration_score, 3),
                        "intent_score": round(intent_score, 3),
                        "shot_score": round(shot_score, 3),
                        "metadata_density": round(metadata_density, 3),
                        "recency_score": round(recency_score, 3),
                    },
                    "keyword_hits": concept_hits,
                    "shot_type": shot_style or "",
                    "visual_intent": visual_intent or "",
                    "query_mode": _VISUAL_INTENT_QUERY_MODE.get(
                        (visual_intent or "").strip().lower(), "literal"
                    ),
                    "crop_score": 0.9,
                },
            )
        )

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:candidates_per_slot]


def _mix_candidates(
    *,
    local_candidates: list[tuple[MediaAsset, float, dict[str, object]]],
    external_candidates: list[ExternalBrollCandidate],
    limit: int,
) -> list[tuple[str, str | None, str | None, str | None, float, dict[str, object]]]:
    if limit <= 0:
        return []

    merged: list[
        tuple[str, str | None, str | None, str | None, float, dict[str, object]]
    ] = []

    local_target = 0
    external_target = 0
    if local_candidates and external_candidates:
        local_target = max(1, limit // 2)
        external_target = max(1, limit - local_target)
    elif local_candidates:
        local_target = limit
    elif external_candidates:
        external_target = limit

    for asset, score, reason in local_candidates[:local_target]:
        merged.append(
            (
                "project_asset",
                asset.id,
                None,
                asset.filename,
                score,
                reason,
            )
        )
    for candidate in external_candidates[:external_target]:
        merged.append(
            (
                candidate.source_type,
                None,
                candidate.source_url,
                candidate.source_label,
                candidate.score,
                candidate.reason,
            )
        )

    if len(merged) < limit:
        remaining_local = local_candidates[local_target:]
        remaining_external = external_candidates[external_target:]
        leftovers: list[
            tuple[str, str | None, str | None, str | None, float, dict[str, object]]
        ] = []
        for asset, score, reason in remaining_local:
            leftovers.append(
                ("project_asset", asset.id, None, asset.filename, score, reason)
            )
        for candidate in remaining_external:
            leftovers.append(
                (
                    candidate.source_type,
                    None,
                    candidate.source_url,
                    candidate.source_label,
                    candidate.score,
                    candidate.reason,
                )
            )
        leftovers.sort(key=lambda item: item[4], reverse=True)
        merged.extend(leftovers[: max(0, limit - len(merged))])

    merged.sort(key=lambda item: item[4], reverse=True)
    deduped: list[
        tuple[str, str | None, str | None, str | None, float, dict[str, object]]
    ] = []
    seen_asset_ids: set[str] = set()
    seen_urls: set[str] = set()
    for entry in merged:
        _source_type, asset_id, source_url, _source_label, _score, _reason = entry
        if asset_id:
            if asset_id in seen_asset_ids:
                continue
            seen_asset_ids.add(asset_id)
        if source_url:
            if source_url in seen_urls:
                continue
            seen_urls.add(source_url)
        deduped.append(entry)
        if len(deduped) >= limit:
            break
    return deduped


def _snapshot_overlay_clips(timeline_state: object) -> list[dict[str, object]]:
    tracks = getattr(timeline_state, "tracks", [])
    for track in tracks:
        if getattr(track, "kind", "") != "overlay":
            continue
        return [clip.model_dump(mode="json") for clip in getattr(track, "clips", [])]
    return []


def _restore_overlay_clips_from_snapshot(
    timeline_state: object, snapshot: list[dict[str, object]]
) -> int:
    tracks = getattr(timeline_state, "tracks", [])
    overlay_track = next(
        (track for track in tracks if getattr(track, "kind", "") == "overlay"), None
    )
    if overlay_track is None:
        return 0
    restored: list[Clip] = []
    for raw in snapshot:
        try:
            restored.append(Clip.model_validate(raw))
        except Exception:
            continue
    restored.sort(key=lambda item: item.timeline_start_sec)
    overlay_track.clips = restored
    return len(restored)


def _record_broll_transaction(
    session: Session,
    *,
    project_id: str,
    action: str,
    previous_overlay_clips: list[dict[str, object]],
    payload: dict[str, object] | None = None,
) -> None:
    extra = payload or {}
    session.add(
        BrollChoice(
            project_id=project_id,
            slot_id=_BROLL_TX_SLOT_ID,
            candidate_id=None,
            action=action,
            payload_json=_json_dumps(
                {
                    "action": action,
                    "previous_overlay_clips": previous_overlay_clips,
                    **extra,
                }
            ),
        )
    )


def _confidence_from_reason(reason: dict[str, object], score: float) -> float | None:
    raw = reason.get("confidence")
    try:
        if raw is not None:
            value = float(raw)
        else:
            value = float(score)
    except (TypeError, ValueError):
        return None
    return round(max(0.0, min(1.0, value)), 3)


def _breakdown_from_reason(reason: dict[str, object]) -> dict[str, float]:
    raw = reason.get("score_breakdown")
    if not isinstance(raw, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in raw.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        result[str(key)] = round(max(0.0, min(1.0, numeric)), 3)
    return result


def _entities_from_reason(reason: dict[str, object]) -> list[str]:
    raw = reason.get("entities")
    if not isinstance(raw, list):
        return []
    entities: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text:
            entities.append(text)
    return entities[:8]


def _weak_reason_codes_from_reason(reason: dict[str, object]) -> list[str]:
    raw = reason.get("weak_reason_codes")
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()][:6]


def _visual_intent_from_reason(reason: dict[str, object]) -> str | None:
    raw = str(reason.get("visual_intent") or "").strip()
    return raw or None


def _review_status_for_slot(
    row: BrollSlot, ordered_candidates: list[BrollCandidate]
) -> tuple[str, list[str], str | None, str | None]:
    if row.status == "rejected":
        return ("rejected", [], None, "Rejected by user")
    if row.status == "chosen":
        return ("approved", [], None, "Approved for timeline")
    if not ordered_candidates:
        return ("unfilled", ["no_candidates"], None, "No candidates available")

    top_reason = _parse_reason_json(ordered_candidates[0])
    confidence = (
        _confidence_from_reason(top_reason, float(ordered_candidates[0].score)) or 0.0
    )
    weak_reason_codes = _weak_reason_codes_from_reason(top_reason)
    visual_intent = _visual_intent_from_reason(top_reason)
    if (
        confidence >= settings.broll_confidence_autopick_threshold
        and not weak_reason_codes
    ):
        return ("ready", [], visual_intent, "Ready for auto-apply")
    summary = "Review recommended before syncing"
    if weak_reason_codes:
        summary = f"Review needed: {', '.join(code.replace('_', ' ') for code in weak_reason_codes[:2])}"
    return ("needs_review", weak_reason_codes, visual_intent, summary)


def _visual_intent_for_beat(beat: BrollPlanBeatResponse, beat_text: str) -> str:
    intent = beat.intent_label.strip().lower()
    text = beat_text.lower()
    if intent == "process_visual":
        return "process_step"
    if intent == "payoff_visual":
        return "reaction_payoff"
    if intent == "problem_visual":
        return "environment_context"
    if "screen" in text or "demo" in text or "product" in text:
        return "literal_demo"
    if any(
        token in text
        for token in ("office", "studio", "warehouse", "street", "factory")
    ):
        return "environment_context"
    return (
        "abstract_support"
        if beat.section_label in {"hook", "outro"}
        else "literal_demo"
    )


def _domain_context_for_retrieval(
    transcript_text: str, assets: list[MediaAsset]
) -> dict[str, object]:
    asset_descriptors = [
        " ".join(
            part
            for part in (asset.filename, str(asset.metadata_json or "")[:200])
            if part
        ).strip()
        for asset in assets[:8]
    ]
    return infer_broll_domain_context(
        transcript_text=transcript_text,
        asset_filenames=asset_descriptors,
    )


def _prepare_search_strategy(
    *,
    chunk_text: str,
    concept_text: str,
    visual_intent: str,
    expanded_queries: list[str],
    domain_context: dict[str, object],
    language_hint: str | None = None,
    english_gloss_override: str | None = None,
) -> dict[str, object]:
    strategy = build_broll_search_strategy(
        chunk_text=chunk_text,
        concept_text=concept_text,
        visual_intent=visual_intent,
        query_hints=expanded_queries,
        max_queries=max(4, min(len(expanded_queries) + 2, 8)),
        domain_context=dict(domain_context),
        language_hint=language_hint,
        english_gloss_override=english_gloss_override,
    )
    search_concept = (
        " ".join(str(strategy.get("search_concept") or concept_text).split()).strip()
        or concept_text
    )
    search_visual_intent = (
        str(strategy.get("visual_intent") or visual_intent).strip().lower()
        or visual_intent
    )
    raw_packets = strategy.get("queries")
    query_packets: list[dict[str, str]] = []
    if isinstance(raw_packets, list):
        for item in raw_packets:
            if not isinstance(item, dict):
                continue
            query = " ".join(str(item.get("query") or "").split()).strip()
            mode = str(item.get("mode") or "literal").strip().lower() or "literal"
            if not query:
                continue
            query_packets.append({"query": query, "mode": mode})
    if not query_packets:
        query_packets = [{"query": search_concept, "mode": "literal"}]
    search_tokens = _extract_concepts(f"{search_concept} {chunk_text}".strip())[1]
    blocked_terms = strategy.get("blocked_terms")
    return {
        "search_concept": search_concept,
        "search_tokens": search_tokens,
        "visual_intent": search_visual_intent,
        "query_packets": query_packets,
        "blocked_terms": list(blocked_terms) if isinstance(blocked_terms, list) else [],
        "stockability": str(strategy.get("stockability") or "medium").strip().lower()
        or "medium",
        "rationale": str(strategy.get("rationale") or "").strip(),
        "domain_context": dict(domain_context),
        "raw_strategy": strategy,
    }


def _query_packet_labels(search_strategy: dict[str, object]) -> list[str]:
    raw_packets = search_strategy.get("query_packets")
    if not isinstance(raw_packets, list):
        return []
    queries: list[str] = []
    seen: set[str] = set()
    for item in raw_packets:
        if not isinstance(item, dict):
            continue
        query = " ".join(str(item.get("query") or "").split()).strip()
        if not query:
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return queries[:8]


def _strategy_debug_fields(
    *,
    search_strategy: dict[str, object],
    fallback_chunk_text: str,
) -> dict[str, object]:
    raw_strategy = search_strategy.get("raw_strategy")
    raw_dict = dict(raw_strategy) if isinstance(raw_strategy, dict) else {}
    return {
        "search_queries": _query_packet_labels(search_strategy),
        "original_chunk_text": str(
            raw_dict.get("original_chunk_text") or fallback_chunk_text
        ),
        "english_gloss": str(raw_dict.get("english_gloss") or ""),
        "gloss_override_used": str(raw_dict.get("gloss_override_used") or ""),
    }


def _to_candidate_response(row: BrollCandidate) -> BrollCandidateResponse:
    reason = _parse_reason_json(row)
    return BrollCandidateResponse(
        id=row.id,
        project_id=row.project_id,
        slot_id=row.slot_id,
        asset_id=row.asset_id,
        source_type=row.source_type,
        source_url=row.source_url,
        source_label=row.source_label,
        score=round(float(row.score), 3),
        confidence=_confidence_from_reason(reason, float(row.score)),
        score_breakdown=_breakdown_from_reason(reason),
        entities=_entities_from_reason(reason),
        visual_intent=_visual_intent_from_reason(reason),
        weak_reason_codes=_weak_reason_codes_from_reason(reason),
        reason=reason,
        created_at=row.created_at.isoformat(),
    )


def _to_slot_response(
    row: BrollSlot, candidates: list[BrollCandidate]
) -> BrollSlotResponse:
    ordered_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    review_status, weak_reason_codes, visual_intent, review_summary = (
        _review_status_for_slot(row, ordered_candidates)
    )
    return BrollSlotResponse(
        id=row.id,
        project_id=row.project_id,
        transcript_id=row.transcript_id,
        start_sec=round(float(row.start_sec), 3),
        end_sec=round(float(row.end_sec), 3),
        anchor_word_ids=_parse_anchor_word_ids(row),
        concept_text=row.concept_text,
        locked=bool(row.locked),
        status=row.status,
        review_status=review_status,
        visual_intent=visual_intent,
        review_summary=review_summary,
        weak_reason_codes=weak_reason_codes,
        chosen_candidate_id=row.chosen_candidate_id,
        created_at=row.created_at.isoformat(),
        updated_at=row.updated_at.isoformat(),
        candidates=[
            _to_candidate_response(candidate) for candidate in ordered_candidates
        ],
    )


def _load_slots_with_candidates(
    session: Session,
    *,
    project_id: str,
    transcript_id: str | None,
    slot_ids: list[str] | None = None,
) -> list[BrollSlotResponse]:
    slot_query = select(BrollSlot).where(BrollSlot.project_id == project_id)
    if transcript_id:
        slot_query = slot_query.where(BrollSlot.transcript_id == transcript_id)
    if slot_ids:
        slot_query = slot_query.where(BrollSlot.id.in_(slot_ids))

    slots = list(
        session.exec(
            slot_query.order_by(BrollSlot.start_sec.asc(), BrollSlot.created_at.asc())
        ).all()
    )
    if not slots:
        return []

    ids = [slot.id for slot in slots]
    candidates = list(
        session.exec(
            select(BrollCandidate)
            .where(
                BrollCandidate.project_id == project_id, BrollCandidate.slot_id.in_(ids)
            )
            .order_by(BrollCandidate.score.desc(), BrollCandidate.created_at.asc())
        ).all()
    )

    by_slot: dict[str, list[BrollCandidate]] = {slot_id: [] for slot_id in ids}
    for candidate in candidates:
        by_slot.setdefault(candidate.slot_id, []).append(candidate)

    return [_to_slot_response(slot, by_slot.get(slot.id, [])) for slot in slots]


def _to_suggest_request(payload: BrollAutoApplyRequest) -> BrollSuggestRequest:
    return BrollSuggestRequest(
        transcript_id=payload.transcript_id,
        max_slots=payload.max_slots,
        candidates_per_slot=payload.candidates_per_slot,
        min_chunk_words=payload.min_chunk_words,
        replace_existing=payload.replace_existing,
        include_project_assets=payload.include_project_assets,
        include_external_sources=payload.include_external_sources,
        ai_rerank=payload.ai_rerank,
    )


def _resolve_slot_chunk_text(slot: BrollSlot, transcript: Transcript | None) -> str:
    if transcript is None:
        return slot.concept_text.strip()

    words = _load_transcript_words(transcript)
    if not words:
        return slot.concept_text.strip()

    by_id = {str(item["id"]): str(item["text"]).strip() for item in words}
    anchor_ids = _parse_anchor_word_ids(slot)
    anchored_tokens = [by_id.get(word_id, "").strip() for word_id in anchor_ids]
    anchored_text = " ".join(token for token in anchored_tokens if token).strip()
    if anchored_text:
        return anchored_text

    overlap_tokens = [
        str(item["text"]).strip()
        for item in words
        if float(item["start_sec"]) < float(slot.end_sec)
        and float(item["end_sec"]) > float(slot.start_sec)
    ]
    overlap_text = " ".join(token for token in overlap_tokens if token).strip()
    if overlap_text:
        return overlap_text
    return slot.concept_text.strip()


@router.get("/config", response_model=BrollConfigResponse)
def broll_config(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> BrollConfigResponse:
    current = get_settings()
    pexels_configured = bool(current.pexels_api_key)
    pixabay_configured = bool(current.pixabay_api_key)
    stock_search_available = current.broll_external_enabled and (
        pexels_configured or pixabay_configured
    )
    llm_raw = (os.getenv("BROLL_LLM_ENABLED", "true") or "true").strip().lower()
    llm_rerank_available = llm_raw in {"1", "true", "yes", "on"}
    return BrollConfigResponse(
        external_enabled=current.broll_external_enabled,
        pexels_configured=pexels_configured,
        pixabay_configured=pixabay_configured,
        stock_search_available=stock_search_available,
        generative_enabled=current.broll_generative_enabled,
        llm_rerank_available=llm_rerank_available,
    )


@router.post("/plan", response_model=BrollPlanResponse)
def create_broll_plan(
    payload: BrollPlanRequest,
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> BrollPlanResponse:
    project = _require_project(session, project_id)
    transcript = _resolve_broll_transcript(
        session, project_id=project_id, transcript_id=payload.transcript_id
    )
    return _build_broll_plan(
        session, project=project, transcript=transcript, payload=payload
    )


@router.get("/plans/{plan_id}", response_model=BrollPlanResponse)
def get_broll_plan(
    plan_id: str,
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> BrollPlanResponse:
    _require_project(session, project_id)
    return _load_plan_response(session, plan_id, project_id=project_id)


def _suggest_broll_for_selection(
    *,
    session: Session,
    project: Project,
    transcript: Transcript,
    payload: BrollSuggestRequest,
) -> BrollSuggestResponse:
    """Create a single B-roll slot from user-selected transcript words."""
    project_id = project.id
    word_ids = payload.anchor_word_ids or []
    if not word_ids:
        raise HTTPException(
            status_code=400, detail="anchor_word_ids must not be empty"
        )

    words = _load_transcript_words(transcript)
    words_by_id = {str(w["id"]): w for w in words}
    matched = [words_by_id[wid] for wid in word_ids if wid in words_by_id]
    if not matched:
        raise HTTPException(
            status_code=400,
            detail="None of the provided word IDs found in transcript",
        )

    matched.sort(key=lambda w: float(w["start_sec"]))
    start_sec = round(float(matched[0]["start_sec"]), 3)
    end_sec = round(float(matched[-1]["end_sec"]), 3)
    if end_sec <= start_sec:
        end_sec = start_sec + 0.5
    chunk_text = " ".join(str(w["text"]) for w in matched)

    # Use override text or extract concepts from the selected words
    if payload.concept_override and payload.concept_override.strip():
        concept_text = payload.concept_override.strip()
        _ignored, concept_tokens = _extract_concepts(concept_text)
    else:
        concept_text, concept_tokens = _extract_concepts(chunk_text)

    # Gather project assets
    assets = list(
        session.exec(
            select(MediaAsset)
            .where(
                MediaAsset.project_id == project_id,
                MediaAsset.media_type == "video",
            )
            .order_by(MediaAsset.created_at.desc())
        ).all()
    )
    assets_by_id: dict[str, MediaAsset] = {a.id: a for a in assets}
    domain_context = _domain_context_for_retrieval(transcript.text, assets)

    shot_style, shot_hint = _shot_variant_for_index(0)
    visual_intent = _local_visual_intent_from_text(chunk_text)
    expanded_queries = expand_broll_queries(
        chunk_text=chunk_text,
        concept_text=concept_text,
        concept_tokens=concept_tokens,
    )
    expanded_queries.extend(
        q
        for q in (shot_hint, f"{concept_text} {shot_hint}".strip())
        if q and q not in expanded_queries
    )

    search_strategy = _prepare_search_strategy(
        chunk_text=chunk_text,
        concept_text=concept_text,
        visual_intent=visual_intent,
        expanded_queries=expanded_queries,
        domain_context=domain_context,
        language_hint=transcript.language,
    )
    search_concept_text = str(search_strategy["search_concept"])
    search_concept_tokens = list(search_strategy["search_tokens"])  # type: ignore[arg-type]
    search_visual_intent = str(search_strategy["visual_intent"] or visual_intent)
    query_packets = list(search_strategy["query_packets"])  # type: ignore[arg-type]
    slot_duration_sec = max(end_sec - start_sec, 0.1)
    candidate_limit = payload.candidates_per_slot
    retrieval_limit = max(candidate_limit * 6, 18)

    # Local ranking
    ranked_candidates: list[tuple[MediaAsset, float, dict[str, object]]] = []
    if payload.include_project_assets and assets:
        ranked_candidates = _rank_candidates(
            assets=assets,
            transcript_asset_id=transcript.asset_id,
            concept_tokens=search_concept_tokens,
            candidates_per_slot=retrieval_limit,
            slot_duration=slot_duration_sec,
            shot_style=shot_style,
            visual_intent=search_visual_intent,
        )

    # External candidates
    external_candidates: list[ExternalBrollCandidate] = []
    if payload.include_external_sources:
        try:
            external_candidates = search_external_broll_candidates(
                chunk_text=chunk_text,
                concept_text=search_concept_text,
                concept_tokens=search_concept_tokens,
                slot_duration_sec=slot_duration_sec,
                limit=retrieval_limit,
                query_hints=expanded_queries,
                query_packets=query_packets,
                visual_intent=search_visual_intent,
                domain_context=dict(domain_context),
            )
            external_candidates = [
                ExternalBrollCandidate(
                    source_type=c.source_type,
                    source_url=c.source_url,
                    source_label=c.source_label,
                    score=c.score,
                    reason={
                        **c.reason,
                        "shot_type": shot_style,
                        "visual_intent": search_visual_intent,
                        "search_concept": search_concept_text,
                        "domain_label": str(domain_context.get("domain") or ""),
                    },
                )
                for c in external_candidates
            ]
            top_score = max((c.score for c in external_candidates), default=0.0)
            if top_score < settings.broll_generative_min_external_score:
                generated = generate_generative_broll_candidates(
                    concept_text=search_concept_text,
                    concept_tokens=search_concept_tokens,
                    shot_hint=shot_hint,
                    slot_duration_sec=slot_duration_sec,
                    limit=max(2, retrieval_limit // 3),
                )
                if generated:
                    external_candidates.extend(generated)
        except Exception:
            logger.warning("External B-roll search failed for selection slot")

    # Mix, rerank, store
    merged_candidates = _mix_candidates(
        local_candidates=ranked_candidates,
        external_candidates=external_candidates,
        limit=retrieval_limit,
    )
    if payload.ai_rerank and merged_candidates:
        try:
            merged_candidates = rerank_broll_candidates(
                chunk_text=chunk_text,
                concept_text=search_concept_text,
                concept_tokens=search_concept_tokens,
                slot_duration_sec=slot_duration_sec,
                candidates=merged_candidates,
                assets_by_id=assets_by_id,
                visual_intent=search_visual_intent,
            )
        except Exception as exc:
            logger.warning("B-roll AI rerank failed for selection slot: %s", exc)
        try:
            llm_ranked = llm_rerank_broll_candidates(
                chunk_text=chunk_text,
                concept_text=search_concept_text,
                visual_intent=search_visual_intent,
                candidates=merged_candidates,
                assets_by_id=assets_by_id,
                domain_context=dict(domain_context),
            )
            if llm_ranked:
                merged_candidates = llm_ranked
        except Exception as exc:
            logger.warning("B-roll LLM rerank failed for selection slot: %s", exc)
    merged_candidates = merged_candidates[:candidate_limit]

    if not merged_candidates:
        raise HTTPException(
            status_code=400,
            detail="No B-roll candidates found for the selected transcript segment",
        )

    now = _utcnow()
    slot = BrollSlot(
        project_id=project_id,
        transcript_id=transcript.id,
        start_sec=start_sec,
        end_sec=end_sec,
        anchor_word_ids_json=_json_dumps(word_ids),
        concept_text=concept_text,
        locked=False,
        status="pending",
        updated_at=now,
    )
    session.add(slot)

    for source_type, asset_id, source_url, source_label, score, reason in merged_candidates:
        session.add(
            BrollCandidate(
                project_id=project_id,
                slot_id=slot.id,
                asset_id=asset_id,
                source_type=source_type,
                source_url=source_url,
                source_label=source_label,
                score=score,
                reason_json=_json_dumps(
                    {
                        **reason,
                        "visual_intent": search_visual_intent,
                        "search_concept": search_concept_text,
                        "original_concept_text": concept_text,
                        "shot_style": shot_style,
                        "source_strategy": "local_first",
                        "query_hints": expanded_queries[:8],
                        **_strategy_debug_fields(
                            search_strategy=search_strategy,
                            fallback_chunk_text=chunk_text,
                        ),
                        "domain_label": str(domain_context.get("domain") or ""),
                    }
                ),
            )
        )

    session.commit()
    responses = _load_slots_with_candidates(
        session,
        project_id=project_id,
        transcript_id=transcript.id,
        slot_ids=[slot.id],
    )
    return BrollSuggestResponse(
        project_id=project_id,
        transcript_id=transcript.id,
        created_slots=1,
        slots=responses,
    )


def _local_visual_intent_from_text(text: str) -> str:
    """Infer visual intent from raw text using keyword cues."""
    terms = _focus_terms(text)
    best_intent = "abstract_support"
    best_hits = 0
    for intent, cue_set in _LOCAL_VISUAL_INTENT_CUES.items():
        hits = len(terms.intersection(cue_set))
        if hits > best_hits:
            best_hits = hits
            best_intent = intent
    return best_intent


@router.post("/suggest", response_model=BrollSuggestResponse)
def suggest_broll(
    payload: BrollSuggestRequest,
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> BrollSuggestResponse:
    project = _require_project(session, project_id)
    transcript = _resolve_broll_transcript(
        session, project_id=project_id, transcript_id=payload.transcript_id
    )

    # ── Single-slot selection mode ────────────────────────────────────
    if payload.anchor_word_ids:
        return _suggest_broll_for_selection(
            session=session,
            project=project,
            transcript=transcript,
            payload=payload,
        )

    # ── Full-transcript mode (existing behaviour) ─────────────────────
    plan_response = _build_broll_plan(
        session,
        project=project,
        transcript=transcript,
        payload=_plan_request_from_suggest(payload),
    )
    beats = [
        beat
        for beat in plan_response.beats
        if beat.should_place and beat.end_sec > beat.start_sec
    ]
    if not beats:
        raise HTTPException(
            status_code=400, detail="Planner produced no eligible B-roll beats"
        )

    assets = list(
        session.exec(
            select(MediaAsset)
            .where(
                MediaAsset.project_id == project_id, MediaAsset.media_type == "video"
            )
            .order_by(MediaAsset.created_at.desc())
        ).all()
    )
    if payload.include_project_assets and not assets:
        raise HTTPException(status_code=400, detail="No video assets found in project")
    assets_by_id: dict[str, MediaAsset] = {asset.id: asset for asset in assets}
    domain_context = _domain_context_for_retrieval(transcript.text, assets)

    if payload.replace_existing:
        existing_slots = list(
            session.exec(
                select(BrollSlot).where(
                    BrollSlot.project_id == project_id,
                    BrollSlot.transcript_id == transcript.id,
                )
            ).all()
        )
        existing_slot_ids = [row.id for row in existing_slots]
        if existing_slot_ids:
            session.exec(
                delete(BrollChoice).where(
                    BrollChoice.project_id == project_id,
                    BrollChoice.slot_id.in_(existing_slot_ids),
                )
            )
            session.exec(
                delete(BrollCandidate).where(
                    BrollCandidate.project_id == project_id,
                    BrollCandidate.slot_id.in_(existing_slot_ids),
                )
            )
            session.exec(
                delete(BrollSlot).where(
                    BrollSlot.project_id == project_id,
                    BrollSlot.id.in_(existing_slot_ids),
                )
            )

    now = _utcnow()
    created_slot_ids: list[str] = []
    beat_metas: list[dict[str, object]] = []
    for idx, beat in enumerate(beats):
        beat_text = beat.segment_text.strip() or beat.concept_text.strip()
        concept_text = beat.concept_text.strip() or _extract_concepts(beat_text)[0]
        _ignored, concept_tokens = _extract_concepts(
            f"{concept_text} {beat_text}".strip()
        )
        shot_style = beat.shot_style.strip() or _shot_variant_for_index(idx)[0]
        shot_hint = f"{shot_style} shot".strip()
        visual_intent = _visual_intent_for_beat(beat, beat_text)
        expanded_queries = expand_broll_queries(
            chunk_text=beat_text,
            concept_text=concept_text,
            concept_tokens=concept_tokens,
        )
        expanded_queries.extend(
            query
            for query in (
                shot_hint,
                f"{concept_text} {shot_hint}".strip(),
            )
            if query and query not in expanded_queries
        )
        search_strategy = _prepare_search_strategy(
            chunk_text=beat_text,
            concept_text=concept_text,
            visual_intent=visual_intent,
            expanded_queries=expanded_queries,
            domain_context=domain_context,
            language_hint=transcript.language,
        )
        search_concept_text = str(search_strategy["search_concept"])
        search_concept_tokens = list(search_strategy["search_tokens"])  # type: ignore[arg-type]
        search_visual_intent = str(search_strategy["visual_intent"] or visual_intent)
        query_packets = list(search_strategy["query_packets"])  # type: ignore[arg-type]
        slot_duration_sec = max(float(beat.end_sec) - float(beat.start_sec), 0.1)
        candidate_limit = payload.candidates_per_slot
        retrieval_limit = max(candidate_limit * 6, 18)

        ranked_candidates: list[tuple[MediaAsset, float, dict[str, object]]] = []
        if payload.include_project_assets:
            ranked_candidates = _rank_candidates(
                assets=assets,
                transcript_asset_id=transcript.asset_id,
                concept_tokens=search_concept_tokens,
                candidates_per_slot=retrieval_limit,
                slot_duration=slot_duration_sec,
                shot_style=shot_style,
                visual_intent=search_visual_intent,
            )

        beat_metas.append(
            {
                "idx": idx,
                "beat": beat,
                "beat_text": beat_text,
                "concept_text": search_concept_text,
                "concept_tokens": search_concept_tokens,
                "expanded_queries": expanded_queries,
                "query_packets": query_packets,
                "slot_duration_sec": slot_duration_sec,
                "candidate_limit": candidate_limit,
                "retrieval_limit": retrieval_limit,
                "shot_style": shot_style,
                "shot_hint": shot_hint,
                "visual_intent": search_visual_intent,
                "source_strategy": beat.source_strategy,
                "ranked_candidates": ranked_candidates,
                "search_strategy": search_strategy,
                "original_concept_text": concept_text,
            }
        )

    # ── Phase 2: Fetch external candidates in PARALLEL ────────────────
    def _fetch_external_for_slot(
        meta: dict[str, object],
    ) -> list[ExternalBrollCandidate]:
        """Fetch external + generative candidates for a single slot (thread-safe)."""
        concept_text = str(meta["concept_text"])
        concept_tokens = list(meta["concept_tokens"])  # type: ignore[arg-type]
        slot_duration_sec = float(meta["slot_duration_sec"])  # type: ignore[arg-type]
        candidate_limit = int(meta["candidate_limit"])  # type: ignore[arg-type]
        retrieval_limit = int(meta["retrieval_limit"])  # type: ignore[arg-type]
        expanded_queries = list(meta["expanded_queries"])  # type: ignore[arg-type]
        query_packets = list(meta["query_packets"])  # type: ignore[arg-type]
        shot_style = str(meta["shot_style"])
        shot_hint = str(meta["shot_hint"])
        visual_intent = str(meta["visual_intent"] or "literal_demo")
        search_strategy = dict(meta["search_strategy"])  # type: ignore[arg-type]

        try:
            candidates = search_external_broll_candidates(
                chunk_text=str(meta["beat_text"]),
                concept_text=concept_text,
                concept_tokens=concept_tokens,
                slot_duration_sec=slot_duration_sec,
                limit=retrieval_limit,
                query_hints=expanded_queries,
                query_packets=query_packets,
                visual_intent=visual_intent,
                domain_context=dict(domain_context),
            )
            candidates = [
                ExternalBrollCandidate(
                    source_type=c.source_type,
                    source_url=c.source_url,
                    source_label=c.source_label,
                    score=c.score,
                    reason={
                        **c.reason,
                        "shot_type": shot_style,
                        "visual_intent": visual_intent,
                        "search_concept": concept_text,
                        "search_strategy_rationale": str(
                            search_strategy.get("rationale") or ""
                        ),
                        "stockability": str(search_strategy.get("stockability") or ""),
                        "blocked_terms": list(
                            search_strategy.get("blocked_terms") or []
                        ),
                        "domain_label": str(domain_context.get("domain") or ""),
                    },
                )
                for c in candidates
            ]
            top_score = max((c.score for c in candidates), default=0.0)
            if top_score < settings.broll_generative_min_external_score:
                generated = generate_generative_broll_candidates(
                    concept_text=concept_text,
                    concept_tokens=concept_tokens,
                    shot_hint=shot_hint,
                    slot_duration_sec=slot_duration_sec,
                    limit=max(2, retrieval_limit // 3),
                )
                if generated:
                    candidates.extend(generated)
            return candidates
        except Exception:
            return []

    external_results: list[list[ExternalBrollCandidate]] = [[] for _ in beat_metas]
    if payload.include_external_sources and beat_metas:
        max_workers = min(8, len(beat_metas))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_fetch_external_for_slot, meta): i
                for i, meta in enumerate(beat_metas)
            }
            for future in as_completed(future_to_idx):
                slot_idx = future_to_idx[future]
                try:
                    external_results[slot_idx] = future.result()
                except Exception:
                    external_results[slot_idx] = []

    # ── Phase 3: Mix, rerank, and store results (serial, fast) ────────
    sequence_state = _empty_sequence_state()
    for meta_idx, meta in enumerate(beat_metas):
        beat = meta["beat"]
        chunk_text = str(meta["beat_text"])
        concept_text = str(meta["concept_text"])
        concept_tokens = list(meta["concept_tokens"])  # type: ignore[arg-type]
        slot_duration_sec = float(meta["slot_duration_sec"])  # type: ignore[arg-type]
        candidate_limit = int(meta["candidate_limit"])  # type: ignore[arg-type]
        retrieval_limit = int(meta["retrieval_limit"])  # type: ignore[arg-type]
        visual_intent = str(meta["visual_intent"] or "literal_demo")
        ranked_candidates = meta["ranked_candidates"]  # type: ignore[assignment]
        original_concept_text = str(meta["original_concept_text"] or concept_text)
        search_strategy = dict(meta["search_strategy"])  # type: ignore[arg-type]

        external_candidates = external_results[meta_idx]
        merged_candidates = _mix_candidates(
            local_candidates=ranked_candidates,
            external_candidates=external_candidates,
            limit=retrieval_limit,
        )
        if payload.ai_rerank and merged_candidates:
            try:
                merged_candidates = rerank_broll_candidates(
                    chunk_text=chunk_text,
                    concept_text=concept_text,
                    concept_tokens=concept_tokens,
                    slot_duration_sec=slot_duration_sec,
                    candidates=merged_candidates,
                    assets_by_id=assets_by_id,
                    visual_intent=visual_intent,
                )
            except Exception as exc:
                logger.warning(
                    "B-roll AI rerank failed; using pre-rerank candidates: %s", exc
                )
        if payload.ai_rerank and merged_candidates:
            try:
                llm_ranked = llm_rerank_broll_candidates(
                    chunk_text=chunk_text,
                    concept_text=concept_text,
                    visual_intent=visual_intent,
                    candidates=merged_candidates,
                    assets_by_id=assets_by_id,
                    domain_context=dict(domain_context),
                )
                if llm_ranked:
                    merged_candidates = llm_ranked
            except Exception as exc:
                logger.warning(
                    "B-roll LLM rerank failed; using pre-LLM candidates: %s", exc
                )
        merged_candidates = _sequence_diversify_candidates(
            merged_candidates,
            sequence_state=sequence_state,
        )
        merged_candidates = merged_candidates[:candidate_limit]

        if not merged_candidates:
            continue

        slot = BrollSlot(
            project_id=project_id,
            transcript_id=transcript.id,
            start_sec=round(float(beat.start_sec), 3),
            end_sec=round(float(beat.end_sec), 3),
            anchor_word_ids_json=_json_dumps(beat.anchor_word_ids),
            concept_text=concept_text,
            locked=False,
            status="pending",
            updated_at=now,
        )
        session.add(slot)
        created_slot_ids.append(slot.id)

        for (
            source_type,
            asset_id,
            source_url,
            source_label,
            score,
            reason,
        ) in merged_candidates:
            session.add(
                BrollCandidate(
                    project_id=project_id,
                    slot_id=slot.id,
                    asset_id=asset_id,
                    source_type=source_type,
                    source_url=source_url,
                    source_label=source_label,
                    score=score,
                    reason_json=_json_dumps(
                        {
                            **reason,
                            "visual_intent": visual_intent,
                            "search_concept": concept_text,
                            "original_concept_text": original_concept_text,
                            "section_label": beat.section_label,
                            "shot_style": shot_style,
                            "source_strategy": beat.source_strategy,
                            "planner_confidence": round(float(beat.confidence), 3),
                            "planner_rationale": beat.rationale,
                            "query_hints": expanded_queries[:8],
                            **_strategy_debug_fields(
                                search_strategy=search_strategy,
                                fallback_chunk_text=beat_text,
                            ),
                            "search_strategy_rationale": str(
                                search_strategy.get("rationale") or ""
                            ),
                            "stockability": str(
                                search_strategy.get("stockability") or ""
                            ),
                            "blocked_terms": list(
                                search_strategy.get("blocked_terms") or []
                            ),
                            "domain_label": str(domain_context.get("domain") or ""),
                        }
                    ),
                )
            )
        _remember_sequence_candidate(sequence_state, merged_candidates[0])

    if not created_slot_ids:
        raise HTTPException(
            status_code=400,
            detail="No B-roll candidates available for current settings",
        )

    session.commit()

    responses = _load_slots_with_candidates(
        session,
        project_id=project_id,
        transcript_id=transcript.id,
        slot_ids=created_slot_ids,
    )
    return BrollSuggestResponse(
        project_id=project_id,
        transcript_id=transcript.id,
        created_slots=len(created_slot_ids),
        slots=responses,
    )


def _process_suggest_broll_job(
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
                progress=10,
                stage="running",
                message="Generating B-roll slots and candidates",
            )
            payload = BrollSuggestRequest.model_validate(payload_json)
            response = suggest_broll(payload, project_id=project_id, session=session)

            result_path = _broll_suggest_result_path(job_id)
            result_path.write_text(
                _json_dumps(response.model_dump(mode="json")),
                encoding="utf-8",
            )
            set_job_status(
                session,
                job,
                status="completed",
                progress=100,
                stage="complete",
                message=f"Generated {response.created_slots} B-roll slots",
                output_path=str(result_path),
            )
        except Exception as exc:  # noqa: BLE001
            set_job_status(
                session,
                job,
                status="failed",
                progress=100,
                stage="failed",
                message=_suggest_error_message(exc),
                error=_suggest_error_message(exc),
            )


@router.post("/suggest/async", response_model=JobResponse)
def suggest_broll_async(
    payload: BrollSuggestRequest,
    project_id: str,
    force: bool = False,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> JobResponse:
    _require_project(session, project_id)
    if not force:
        active = find_recent_active_job(
            session, project_id, kind="broll_suggest", within_seconds=0
        )
        if active:
            return _to_job_response(active)

    job = create_job(session, project_id, kind="broll_suggest")
    threading.Thread(
        target=_process_suggest_broll_job,
        kwargs={
            "job_id": job.id,
            "project_id": project_id,
            "payload_json": payload.model_dump(mode="json"),
        },
        name=f"broll-suggest-{job.id[:8]}",
        daemon=True,
    ).start()
    return _to_job_response(job)


@router.get("/suggest/results/{job_id}", response_model=BrollSuggestResponse)
def get_suggest_broll_result(
    job_id: str,
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> BrollSuggestResponse:
    _require_project(session, project_id)
    job = session.exec(
        select(Job).where(
            Job.id == job_id,
            Job.project_id == project_id,
            Job.kind == "broll_suggest",
        )
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="B-roll suggest job not found")
    if job.status == "failed":
        raise HTTPException(
            status_code=409, detail=job.error or "B-roll suggest job failed"
        )
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="B-roll suggest job not completed")

    result_path = _broll_suggest_result_path(job.id)
    if not result_path.exists():
        raise HTTPException(status_code=500, detail="B-roll suggest result missing")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail="B-roll suggest result payload invalid"
        ) from exc
    return BrollSuggestResponse.model_validate(payload)


@router.post("/auto-apply", response_model=BrollAutoApplyResponse)
def auto_apply_broll(
    payload: BrollAutoApplyRequest,
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> BrollAutoApplyResponse:
    project = _require_project(session, project_id)

    suggest_response = suggest_broll(
        _to_suggest_request(payload), project_id=project_id, session=session
    )
    slot_ids = [slot.id for slot in suggest_response.slots]
    if not slot_ids:
        raise HTTPException(
            status_code=400, detail="No B-roll slots available to auto-apply"
        )

    confidence_threshold = payload.min_confidence
    if confidence_threshold is None:
        confidence_threshold = settings.broll_confidence_autopick_threshold
    confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))

    slots = list(
        session.exec(
            select(BrollSlot)
            .where(BrollSlot.project_id == project_id, BrollSlot.id.in_(slot_ids))
            .order_by(BrollSlot.start_sec.asc(), BrollSlot.created_at.asc())
        ).all()
    )
    candidates = list(
        session.exec(
            select(BrollCandidate)
            .where(
                BrollCandidate.project_id == project_id,
                BrollCandidate.slot_id.in_(slot_ids),
            )
            .order_by(BrollCandidate.score.desc(), BrollCandidate.created_at.asc())
        ).all()
    )
    by_slot: dict[str, list[BrollCandidate]] = {slot_id: [] for slot_id in slot_ids}
    for candidate in candidates:
        by_slot.setdefault(candidate.slot_id, []).append(candidate)

    selected_pairs: list[tuple[BrollSlot, BrollCandidate]] = []
    auto_chosen_slots = 0
    skipped_slots = 0
    skipped_slot_summaries: list[BrollAutoApplySkipSummary] = []
    for slot in slots:
        ordered = by_slot.get(slot.id, [])
        selected_candidate: BrollCandidate | None = None
        if not ordered:
            slot.status = "needs_review"
            slot.chosen_candidate_id = None
            slot.updated_at = _utcnow()
            session.add(slot)
            session.add(
                BrollChoice(
                    project_id=project_id,
                    slot_id=slot.id,
                    candidate_id=None,
                    action="auto_skip",
                    payload_json=_json_dumps(
                        {
                            "reason": "no_candidates",
                            "threshold": round(confidence_threshold, 3),
                        }
                    ),
                )
            )
            skipped_slot_summaries.append(
                BrollAutoApplySkipSummary(
                    slot_id=slot.id,
                    concept_text=slot.concept_text or "",
                    reason="no_candidates",
                    detail="No B-roll candidates were found for this slot.",
                )
            )
            skipped_slots += 1
            session.commit()
            continue
        for candidate in ordered:
            reason = _parse_reason_json(candidate)
            confidence = _confidence_from_reason(reason, float(candidate.score))
            weak_reason_codes = _weak_reason_codes_from_reason(reason)
            if (
                confidence is not None
                and confidence >= confidence_threshold
                and not weak_reason_codes
            ):
                selected_candidate = candidate
                break
        if selected_candidate is None and payload.fallback_to_top_candidate and ordered:
            selected_candidate = ordered[0]

        if selected_candidate is None:
            slot.status = "needs_review"
            slot.chosen_candidate_id = None
            slot.updated_at = _utcnow()
            session.add(slot)
            session.add(
                BrollChoice(
                    project_id=project_id,
                    slot_id=slot.id,
                    candidate_id=None,
                    action="auto_skip",
                    payload_json=_json_dumps(
                        {
                            "reason": "needs_review",
                            "threshold": round(confidence_threshold, 3),
                        }
                    ),
                )
            )
            skipped_slot_summaries.append(
                BrollAutoApplySkipSummary(
                    slot_id=slot.id,
                    concept_text=slot.concept_text or "",
                    reason="needs_review",
                    detail=(
                        f"No candidate met the {(confidence_threshold * 100):.0f}% confidence threshold."
                    ),
                )
            )
            skipped_slots += 1
            session.commit()  # Release DB lock so other operations don't block
            continue

        try:
            if not selected_candidate.asset_id:
                _materialize_candidate_asset(session, project_id, selected_candidate)
        except HTTPException as exc:
            slot.status = "needs_review"
            slot.chosen_candidate_id = None
            slot.updated_at = _utcnow()
            session.add(slot)
            session.add(
                BrollChoice(
                    project_id=project_id,
                    slot_id=slot.id,
                    candidate_id=None,
                    action="auto_skip",
                    payload_json=_json_dumps(
                        {
                            "reason": "materialize_failed",
                            "detail": str(exc.detail),
                        }
                    ),
                )
            )
            skipped_slot_summaries.append(
                BrollAutoApplySkipSummary(
                    slot_id=slot.id,
                    concept_text=slot.concept_text or "",
                    reason="materialize_failed",
                    detail=str(exc.detail),
                )
            )
            skipped_slots += 1
            session.commit()  # Release DB lock between downloads
            continue

        slot.status = "chosen"
        slot.chosen_candidate_id = selected_candidate.id
        slot.updated_at = _utcnow()
        session.add(slot)
        session.add(
            BrollChoice(
                project_id=project_id,
                slot_id=slot.id,
                candidate_id=selected_candidate.id,
                action="auto_choose",
                payload_json=_json_dumps(
                    {
                        "candidate_id": selected_candidate.id,
                        "asset_id": selected_candidate.asset_id,
                        "threshold": round(confidence_threshold, 3),
                    }
                ),
            )
        )
        if selected_candidate.asset_id:
            selected_pairs.append((slot, selected_candidate))
            auto_chosen_slots += 1
        else:
            skipped_slots += 1
        session.commit()  # Release DB lock after each slot

    # Final commit for any remaining changes

    timeline = get_timeline_row(session, project_id)
    timeline_state = load_timeline_state(timeline)
    previous_overlay_clips = _snapshot_overlay_clips(timeline_state)
    timeline_changed = False
    if payload.clear_existing_overlay:
        overlay_track = next(
            (track for track in timeline_state.tracks if track.kind == "overlay"), None
        )
        for clip in list(overlay_track.clips) if overlay_track else []:
            try:
                apply_operation(
                    timeline_state,
                    OperationPayload(
                        op_type="delete_broll_clip",
                        params={"clip": clip.id},
                        source="ui",
                    ),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            timeline_changed = True

    selected_asset_ids = [
        candidate.asset_id for _slot, candidate in selected_pairs if candidate.asset_id
    ]
    assets_by_id: dict[str, MediaAsset] = {}
    if selected_asset_ids:
        assets = list(
            session.exec(
                select(MediaAsset).where(
                    MediaAsset.project_id == project_id,
                    MediaAsset.id.in_(selected_asset_ids),
                )
            ).all()
        )
        assets_by_id = {asset.id: asset for asset in assets}

    synced_clip_count = 0
    min_slot_sec, max_slot_sec = _resolve_slot_pacing(project)
    for slot, candidate in selected_pairs:
        if not candidate.asset_id:
            continue
        asset = assets_by_id.get(candidate.asset_id)
        slot_duration = max(float(slot.end_sec) - float(slot.start_sec), 0.2)
        if _is_vertical_project(project):
            slot_duration = max(min_slot_sec, min(max_slot_sec, slot_duration))

        source_duration = slot_duration
        metadata: dict[str, object] = {}
        if asset:
            metadata = _ensure_asset_focus_metadata(session, asset)
            if asset.duration_sec and asset.duration_sec > 0:
                source_duration = min(float(asset.duration_sec), slot_duration)
                if (
                    _is_vertical_project(project)
                    and source_duration < min_slot_sec
                    and float(asset.duration_sec) >= min_slot_sec
                ):
                    source_duration = min(float(asset.duration_sec), min_slot_sec)

        crop_payload: dict[str, int] | None = None
        crop_keyframes: list[dict[str, float | int]] = []
        overlay_opacity = round(float(payload.overlay_opacity), 3)
        preset: str | None = None
        if asset:
            try:
                focus_x = (
                    float(metadata.get("focus_x"))
                    if metadata.get("focus_x") is not None
                    else None
                )
            except (TypeError, ValueError):
                focus_x = None
            crop_payload = _build_vertical_crop(
                project,
                int(metadata.get("width") or 0),
                int(metadata.get("height") or 0),
                focus_x,
            )
            crop_keyframes = _build_vertical_crop_keyframes(
                project,
                int(metadata.get("width") or 0),
                int(metadata.get("height") or 0),
                metadata.get("focus_track"),
                clip_duration_sec=float(source_duration),
            )
            preset, safety_opacity = _text_safety_preset_from_metadata(metadata)
            overlay_opacity = min(overlay_opacity, safety_opacity)
        try:
            params: dict[str, object] = {
                "asset_id": candidate.asset_id,
                "start_sec": 0.0,
                "end_sec": round(source_duration, 3),
                "timeline_start_sec": round(float(slot.start_sec), 3),
                "opacity": round(float(overlay_opacity), 3),
            }
            if crop_payload:
                params["crop"] = crop_payload
            if crop_keyframes:
                params["crop_keyframes"] = crop_keyframes
            if preset:
                params["preset"] = preset
            apply_operation(
                timeline_state,
                OperationPayload(
                    op_type="add_broll_clip",
                    params=params,
                    source="ui",
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        synced_clip_count += 1
        timeline_changed = True

    if timeline_changed:
        timeline = save_timeline_state(
            session,
            timeline,
            timeline_state,
            source="ui",
            operation=OperationPayload(
                op_type="auto_apply_broll",
                source="ui",
                params={
                    "slot_ids": slot_ids,
                    "created_slots": suggest_response.created_slots,
                    "auto_chosen_slots": auto_chosen_slots,
                    "synced_clip_count": synced_clip_count,
                    "skipped_slots": skipped_slots,
                    "confidence_threshold": round(confidence_threshold, 3),
                    "clear_existing_overlay": payload.clear_existing_overlay,
                    "previous_overlay_clip_count": len(previous_overlay_clips),
                },
            ),
        )
        _record_broll_transaction(
            session,
            project_id=project_id,
            action="auto_apply_broll",
            previous_overlay_clips=previous_overlay_clips,
            payload={
                "slot_ids": slot_ids,
                "synced_clip_count": synced_clip_count,
            },
        )
        session.commit()
    else:
        timeline = get_timeline_row(session, project_id)

    refreshed_slots = _load_slots_with_candidates(
        session,
        project_id=project_id,
        transcript_id=suggest_response.transcript_id,
        slot_ids=slot_ids,
    )
    return BrollAutoApplyResponse(
        project_id=project_id,
        transcript_id=suggest_response.transcript_id,
        created_slots=suggest_response.created_slots,
        auto_chosen_slots=auto_chosen_slots,
        synced_clip_count=synced_clip_count,
        skipped_slots=skipped_slots,
        confidence_threshold=round(confidence_threshold, 3),
        skipped_slot_summaries=skipped_slot_summaries,
        timeline=load_timeline_state(timeline),
        slots=refreshed_slots,
    )


@router.post("/sync", response_model=BrollSyncResponse)
def sync_broll_to_timeline(
    payload: BrollSyncRequest,
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> BrollSyncResponse:
    project = _require_project(session, project_id)

    slot_query = select(BrollSlot).where(BrollSlot.project_id == project_id)
    if payload.transcript_id:
        slot_query = slot_query.where(BrollSlot.transcript_id == payload.transcript_id)
    if payload.slot_ids:
        slot_query = slot_query.where(BrollSlot.id.in_(payload.slot_ids))
    slots = list(
        session.exec(
            slot_query.order_by(BrollSlot.start_sec.asc(), BrollSlot.created_at.asc())
        ).all()
    )

    chosen_slots = [slot for slot in slots if slot.chosen_candidate_id]
    if not chosen_slots:
        timeline = get_timeline_row(session, project_id)
        return BrollSyncResponse(
            project_id=project_id,
            transcript_id=payload.transcript_id,
            synced_clip_count=0,
            timeline=load_timeline_state(timeline),
            slots=_load_slots_with_candidates(
                session,
                project_id=project_id,
                transcript_id=payload.transcript_id,
                slot_ids=[slot.id for slot in slots] if slots else None,
            ),
        )

    candidate_ids = [
        slot.chosen_candidate_id for slot in chosen_slots if slot.chosen_candidate_id
    ]
    candidates = list(
        session.exec(
            select(BrollCandidate)
            .where(
                BrollCandidate.project_id == project_id,
                BrollCandidate.id.in_(candidate_ids),
            )
            .order_by(BrollCandidate.created_at.asc())
        ).all()
    )
    by_candidate_id = {candidate.id: candidate for candidate in candidates}

    selected_pairs: list[tuple[BrollSlot, BrollCandidate]] = []
    for slot in chosen_slots:
        if not slot.chosen_candidate_id:
            continue
        candidate = by_candidate_id.get(slot.chosen_candidate_id)
        if not candidate:
            continue
        if not candidate.asset_id:
            _materialize_candidate_asset(session, project_id, candidate)
        if candidate.asset_id:
            selected_pairs.append((slot, candidate))

    if not selected_pairs:
        raise HTTPException(
            status_code=400, detail="No chosen B-roll candidates available to sync"
        )

    session.commit()

    timeline = get_timeline_row(session, project_id)
    timeline_state = load_timeline_state(timeline)
    previous_overlay_clips = _snapshot_overlay_clips(timeline_state)
    timeline_changed = False
    if payload.clear_existing_overlay:
        overlay_track = next(
            (track for track in timeline_state.tracks if track.kind == "overlay"), None
        )
        for clip in list(overlay_track.clips) if overlay_track else []:
            try:
                apply_operation(
                    timeline_state,
                    OperationPayload(
                        op_type="delete_broll_clip",
                        params={"clip": clip.id},
                        source="ui",
                    ),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            timeline_changed = True

    selected_asset_ids = [
        candidate.asset_id for _slot, candidate in selected_pairs if candidate.asset_id
    ]
    assets_by_id: dict[str, MediaAsset] = {}
    if selected_asset_ids:
        assets = list(
            session.exec(
                select(MediaAsset).where(
                    MediaAsset.project_id == project_id,
                    MediaAsset.id.in_(selected_asset_ids),
                )
            ).all()
        )
        assets_by_id = {asset.id: asset for asset in assets}

    synced_clip_count = 0
    min_slot_sec, max_slot_sec = _resolve_slot_pacing(project)
    for slot, candidate in selected_pairs:
        if not candidate.asset_id:
            continue
        asset = assets_by_id.get(candidate.asset_id)
        slot_duration = max(float(slot.end_sec) - float(slot.start_sec), 0.2)
        if _is_vertical_project(project):
            slot_duration = max(min_slot_sec, min(max_slot_sec, slot_duration))

        source_duration = slot_duration
        metadata: dict[str, object] = {}
        if asset:
            metadata = _ensure_asset_focus_metadata(session, asset)
            if asset.duration_sec and asset.duration_sec > 0:
                source_duration = min(float(asset.duration_sec), slot_duration)
                if (
                    _is_vertical_project(project)
                    and source_duration < min_slot_sec
                    and float(asset.duration_sec) >= min_slot_sec
                ):
                    source_duration = min(float(asset.duration_sec), min_slot_sec)

        crop_payload: dict[str, int] | None = None
        crop_keyframes: list[dict[str, float | int]] = []
        overlay_opacity = round(float(payload.overlay_opacity), 3)
        preset: str | None = None
        if asset:
            try:
                focus_x = (
                    float(metadata.get("focus_x"))
                    if metadata.get("focus_x") is not None
                    else None
                )
            except (TypeError, ValueError):
                focus_x = None
            crop_payload = _build_vertical_crop(
                project,
                int(metadata.get("width") or 0),
                int(metadata.get("height") or 0),
                focus_x,
            )
            crop_keyframes = _build_vertical_crop_keyframes(
                project,
                int(metadata.get("width") or 0),
                int(metadata.get("height") or 0),
                metadata.get("focus_track"),
                clip_duration_sec=float(source_duration),
            )
            preset, safety_opacity = _text_safety_preset_from_metadata(metadata)
            overlay_opacity = min(overlay_opacity, safety_opacity)

        params: dict[str, object] = {
            "asset_id": candidate.asset_id,
            "start_sec": 0.0,
            "end_sec": round(source_duration, 3),
            "timeline_start_sec": round(float(slot.start_sec), 3),
            "opacity": round(float(overlay_opacity), 3),
        }
        if crop_payload:
            params["crop"] = crop_payload
        if crop_keyframes:
            params["crop_keyframes"] = crop_keyframes
        if preset:
            params["preset"] = preset

        try:
            apply_operation(
                timeline_state,
                OperationPayload(
                    op_type="add_broll_clip",
                    params=params,
                    source="ui",
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        synced_clip_count += 1
        timeline_changed = True

    if timeline_changed:
        synced_slot_ids = [slot.id for slot, _candidate in selected_pairs]
        timeline = save_timeline_state(
            session,
            timeline,
            timeline_state,
            source="ui",
            operation=OperationPayload(
                op_type="sync_broll_to_timeline",
                source="ui",
                params={
                    "slot_ids": synced_slot_ids,
                    "synced_clip_count": synced_clip_count,
                    "clear_existing_overlay": payload.clear_existing_overlay,
                    "previous_overlay_clip_count": len(previous_overlay_clips),
                },
            ),
        )
        _record_broll_transaction(
            session,
            project_id=project_id,
            action="sync_broll_to_timeline",
            previous_overlay_clips=previous_overlay_clips,
            payload={
                "slot_ids": synced_slot_ids,
                "synced_clip_count": synced_clip_count,
            },
        )
        session.commit()
    else:
        timeline = get_timeline_row(session, project_id)

    return BrollSyncResponse(
        project_id=project_id,
        transcript_id=payload.transcript_id,
        synced_clip_count=synced_clip_count,
        timeline=load_timeline_state(timeline),
        slots=_load_slots_with_candidates(
            session,
            project_id=project_id,
            transcript_id=payload.transcript_id,
            slot_ids=[slot.id for slot in slots] if slots else None,
        ),
    )


@router.post("/undo", response_model=BrollUndoResponse)
def undo_last_broll_transaction(
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> BrollUndoResponse:
    _require_project(session, project_id)
    row = session.exec(
        select(BrollChoice)
        .where(
            BrollChoice.project_id == project_id,
            BrollChoice.slot_id == _BROLL_TX_SLOT_ID,
            BrollChoice.action.in_(["auto_apply_broll", "sync_broll_to_timeline"]),
        )
        .order_by(BrollChoice.id.desc())
    ).first()
    if not row:
        raise HTTPException(
            status_code=404, detail="No B-roll transaction found to undo"
        )

    try:
        payload = json.loads(row.payload_json or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail="Stored B-roll transaction payload is invalid"
        ) from exc
    if not isinstance(payload, dict):
        payload = {}
    snapshot = payload.get("previous_overlay_clips")
    if not isinstance(snapshot, list):
        snapshot = []

    timeline = get_timeline_row(session, project_id)
    timeline_state = load_timeline_state(timeline)
    restored_clip_count = _restore_overlay_clips_from_snapshot(timeline_state, snapshot)
    timeline = save_timeline_state(
        session,
        timeline,
        timeline_state,
        source="ui",
        operation=OperationPayload(
            op_type="undo_broll_transaction",
            source="ui",
            params={
                "transaction_choice_id": row.id,
                "restored_clip_count": restored_clip_count,
            },
        ),
    )
    session.add(
        BrollChoice(
            project_id=project_id,
            slot_id=_BROLL_TX_SLOT_ID,
            candidate_id=None,
            action="undo_broll_transaction",
            payload_json=_json_dumps(
                {
                    "undone_choice_id": row.id,
                    "restored_clip_count": restored_clip_count,
                }
            ),
        )
    )
    session.commit()

    return BrollUndoResponse(
        project_id=project_id,
        restored_clip_count=restored_clip_count,
        timeline=load_timeline_state(timeline),
        transaction_action=str(payload.get("action") or ""),
    )


@router.get("/slots", response_model=list[BrollSlotResponse])
def list_broll_slots(
    project_id: str,
    transcript_id: str | None = None,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> list[BrollSlotResponse]:
    _require_project(session, project_id)
    return _load_slots_with_candidates(
        session, project_id=project_id, transcript_id=transcript_id
    )


@router.post("/slots/{slot_id}/reroll", response_model=BrollSlotResponse)
def reroll_broll_slot(
    slot_id: str,
    payload: BrollRerollRequest,
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> BrollSlotResponse:
    _require_project(session, project_id)

    slot = session.exec(
        select(BrollSlot).where(
            BrollSlot.id == slot_id, BrollSlot.project_id == project_id
        )
    ).first()
    if not slot:
        raise HTTPException(status_code=404, detail="B-roll slot not found")
    if slot.locked:
        raise HTTPException(status_code=409, detail="B-roll slot is locked")

    transcript: Transcript | None = None
    if slot.transcript_id:
        transcript = session.exec(
            select(Transcript).where(
                Transcript.id == slot.transcript_id, Transcript.project_id == project_id
            )
        ).first()

    chunk_text = _resolve_slot_chunk_text(slot, transcript)
    concept_text = slot.concept_text.strip() or _extract_concepts(chunk_text)[0]
    _ignored, concept_tokens = _extract_concepts(f"{concept_text} {chunk_text}".strip())
    existing_slot_rows = list(
        session.exec(
            select(BrollCandidate)
            .where(
                BrollCandidate.project_id == project_id,
                BrollCandidate.slot_id == slot.id,
            )
            .order_by(BrollCandidate.score.desc(), BrollCandidate.created_at.asc())
        ).all()
    )
    existing_top_reason = (
        _parse_reason_json(existing_slot_rows[0]) if existing_slot_rows else {}
    )
    visual_intent = _visual_intent_from_reason(
        existing_top_reason
    ) or _visual_intent_for_beat(
        BrollPlanBeatResponse(
            id=slot.id,
            beat_index=0,
            start_sec=slot.start_sec,
            end_sec=slot.end_sec,
            timeline_start_sec=None,
            timeline_end_sec=None,
            section_label="body",
            intent_label="supporting_visual",
            source_strategy="local_first",
            shot_style="medium",
            should_place=True,
            confidence=0.0,
            rationale="",
            concept_text=concept_text,
            segment_text=chunk_text,
            anchor_word_ids=_parse_anchor_word_ids(slot),
            query_hints=[],
            metadata={},
        ),
        chunk_text,
    )
    slot_duration_sec = max(float(slot.end_sec) - float(slot.start_sec), 0.1)
    ordered_slot_ids = list(
        session.exec(
            select(BrollSlot.id)
            .where(
                BrollSlot.project_id == project_id,
                BrollSlot.transcript_id == slot.transcript_id,
            )
            .order_by(BrollSlot.start_sec.asc(), BrollSlot.created_at.asc())
        ).all()
    )
    slot_index = ordered_slot_ids.index(slot.id) if slot.id in ordered_slot_ids else 0
    shot_style, shot_hint = _shot_variant_for_index(slot_index)
    expanded_queries = expand_broll_queries(
        chunk_text=chunk_text or concept_text,
        concept_text=concept_text,
        concept_tokens=concept_tokens,
    )
    expanded_queries.extend(
        query
        for query in (
            shot_hint,
            f"{concept_text} {shot_hint}".strip(),
        )
        if query and query not in expanded_queries
    )
    if slot_index == 0:
        expanded_queries.extend(
            query
            for query in (
                f"{concept_text} visual hook",
                f"{concept_text} opening hook",
            )
            if query and query not in expanded_queries
        )
    candidate_limit = payload.candidates_per_slot
    retrieval_limit = max(candidate_limit * 6, 18)
    if slot_index == 0:
        retrieval_limit = max(retrieval_limit, 20)

    assets = list(
        session.exec(
            select(MediaAsset)
            .where(
                MediaAsset.project_id == project_id, MediaAsset.media_type == "video"
            )
            .order_by(MediaAsset.created_at.desc())
        ).all()
    )
    assets_by_id: dict[str, MediaAsset] = {asset.id: asset for asset in assets}
    transcript_text = transcript.text if transcript is not None else chunk_text
    domain_context = _domain_context_for_retrieval(transcript_text, assets)
    search_strategy = _prepare_search_strategy(
        chunk_text=chunk_text,
        concept_text=concept_text,
        visual_intent=visual_intent,
        expanded_queries=expanded_queries,
        domain_context=domain_context,
        language_hint=transcript.language if transcript is not None else None,
        english_gloss_override=payload.english_gloss_override,
    )
    search_concept_text = str(search_strategy["search_concept"])
    search_concept_tokens = list(search_strategy["search_tokens"])  # type: ignore[arg-type]
    visual_intent = str(search_strategy["visual_intent"] or visual_intent)
    query_packets = list(search_strategy["query_packets"])  # type: ignore[arg-type]

    ranked_candidates: list[tuple[MediaAsset, float, dict[str, object]]] = []
    if payload.include_project_assets and assets:
        transcript_asset_id = transcript.asset_id if transcript is not None else ""
        ranked_candidates = _rank_candidates(
            assets=assets,
            transcript_asset_id=transcript_asset_id,
            concept_tokens=search_concept_tokens,
            candidates_per_slot=retrieval_limit,
            slot_duration=slot_duration_sec,
            shot_style=shot_style,
            visual_intent=visual_intent,
        )

    external_candidates: list[ExternalBrollCandidate] = []
    if payload.include_external_sources:
        external_candidates = search_external_broll_candidates(
            chunk_text=chunk_text,
            concept_text=search_concept_text,
            concept_tokens=search_concept_tokens,
            slot_duration_sec=slot_duration_sec,
            limit=retrieval_limit,
            query_hints=expanded_queries,
            query_packets=query_packets,
            visual_intent=visual_intent,
            domain_context=dict(domain_context),
        )
        external_candidates = [
            ExternalBrollCandidate(
                source_type=candidate.source_type,
                source_url=candidate.source_url,
                source_label=candidate.source_label,
                score=candidate.score,
                reason={
                    **candidate.reason,
                    "shot_type": shot_style,
                    "visual_intent": visual_intent,
                    "search_concept": search_concept_text,
                    "search_strategy_rationale": str(
                        search_strategy.get("rationale") or ""
                    ),
                    "stockability": str(search_strategy.get("stockability") or ""),
                    "blocked_terms": list(search_strategy.get("blocked_terms") or []),
                    "domain_label": str(domain_context.get("domain") or ""),
                },
            )
            for candidate in external_candidates
        ]
        top_external_score = max(
            (candidate.score for candidate in external_candidates), default=0.0
        )
        if top_external_score < settings.broll_generative_min_external_score:
            generated = generate_generative_broll_candidates(
                concept_text=search_concept_text,
                concept_tokens=search_concept_tokens,
                shot_hint=shot_hint,
                slot_duration_sec=slot_duration_sec,
                limit=max(2, retrieval_limit // 3),
            )
            if generated:
                external_candidates.extend(generated)

    merged_candidates = _mix_candidates(
        local_candidates=ranked_candidates,
        external_candidates=external_candidates,
        limit=retrieval_limit,
    )
    if payload.ai_rerank and merged_candidates:
        try:
            merged_candidates = rerank_broll_candidates(
                chunk_text=chunk_text,
                concept_text=search_concept_text,
                concept_tokens=search_concept_tokens,
                slot_duration_sec=slot_duration_sec,
                candidates=merged_candidates,
                assets_by_id=assets_by_id,
                visual_intent=visual_intent,
            )
        except Exception as exc:
            logger.warning(
                "B-roll AI rerank failed during reroll; using pre-rerank candidates: %s",
                exc,
            )
    if payload.ai_rerank and merged_candidates:
        try:
            llm_ranked = llm_rerank_broll_candidates(
                chunk_text=chunk_text,
                concept_text=search_concept_text,
                visual_intent=visual_intent,
                candidates=merged_candidates,
                assets_by_id=assets_by_id,
                domain_context=dict(domain_context),
            )
            if llm_ranked:
                merged_candidates = llm_ranked
        except Exception as exc:
            logger.warning(
                "B-roll LLM rerank failed during reroll; using pre-LLM candidates: %s",
                exc,
            )
    existing_candidates = list(
        session.exec(
            select(BrollCandidate)
            .where(
                BrollCandidate.project_id == project_id,
                BrollCandidate.slot_id == slot.id,
            )
            .order_by(BrollCandidate.score.desc(), BrollCandidate.created_at.asc())
        ).all()
    )
    reroll_sequence_state = _empty_sequence_state()
    for existing in existing_candidates[:2]:
        _remember_sequence_candidate(
            reroll_sequence_state,
            (
                existing.source_type,
                existing.asset_id,
                existing.source_url,
                existing.source_label,
                float(existing.score),
                _parse_reason_json(existing),
            ),
        )
    merged_candidates = _sequence_diversify_candidates(
        merged_candidates,
        sequence_state=reroll_sequence_state,
    )
    merged_candidates = merged_candidates[:candidate_limit]

    if not merged_candidates:
        raise HTTPException(
            status_code=400, detail="No B-roll candidates available for reroll"
        )

    seen_asset_ids = {
        candidate.asset_id for candidate in existing_candidates if candidate.asset_id
    }
    seen_urls = {
        candidate.source_url
        for candidate in existing_candidates
        if candidate.source_url
    }

    new_candidates: list[
        tuple[str, str | None, str | None, str | None, float, dict[str, object]]
    ] = []
    for (
        source_type,
        asset_id,
        source_url,
        source_label,
        score,
        reason,
    ) in merged_candidates:
        if asset_id and asset_id in seen_asset_ids:
            continue
        if source_url and source_url in seen_urls:
            continue
        new_candidates.append(
            (source_type, asset_id, source_url, source_label, score, reason)
        )

    if not new_candidates:
        raise HTTPException(
            status_code=400, detail="No new B-roll variants found for this slot"
        )

    added_candidate_ids: list[str] = []
    for (
        source_type,
        asset_id,
        source_url,
        source_label,
        score,
        reason,
    ) in new_candidates:
        row = BrollCandidate(
            project_id=project_id,
            slot_id=slot.id,
            asset_id=asset_id,
            source_type=source_type,
            source_url=source_url,
            source_label=source_label,
            score=score,
            reason_json=_json_dumps(
                {
                    **reason,
                    "visual_intent": visual_intent,
                    "search_concept": search_concept_text,
                    "original_concept_text": concept_text,
                    "section_label": str(
                        existing_top_reason.get("section_label") or "body"
                    ),
                    "shot_style": shot_style,
                    "source_strategy": str(
                        existing_top_reason.get("source_strategy") or "local_first"
                    ),
                    "planner_confidence": existing_top_reason.get("planner_confidence"),
                    "planner_rationale": str(
                        existing_top_reason.get("planner_rationale") or ""
                    ),
                    "query_hints": expanded_queries[:8],
                    **_strategy_debug_fields(
                        search_strategy=search_strategy, fallback_chunk_text=chunk_text
                    ),
                    "search_strategy_rationale": str(
                        search_strategy.get("rationale") or ""
                    ),
                    "stockability": str(search_strategy.get("stockability") or ""),
                    "blocked_terms": list(search_strategy.get("blocked_terms") or []),
                    "domain_label": str(domain_context.get("domain") or ""),
                }
            ),
        )
        session.add(row)
        added_candidate_ids.append(row.id)

    slot.concept_text = search_concept_text
    slot.updated_at = _utcnow()
    session.add(slot)
    session.add(
        BrollChoice(
            project_id=project_id,
            slot_id=slot.id,
            candidate_id=None,
            action="reroll",
            payload_json=_json_dumps(
                {
                    "added_candidate_ids": added_candidate_ids,
                    "count": len(added_candidate_ids),
                }
            ),
        )
    )
    session.commit()

    updated = _load_slots_with_candidates(
        session,
        project_id=project_id,
        transcript_id=slot.transcript_id,
        slot_ids=[slot.id],
    )
    if not updated:
        raise HTTPException(
            status_code=500, detail="Failed to load rerolled B-roll slot"
        )
    return updated[0]


@router.post("/slots/{slot_id}/choose", response_model=BrollSlotResponse)
def choose_broll_candidate(
    slot_id: str,
    payload: BrollChooseRequest,
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> BrollSlotResponse:
    _require_project(session, project_id)

    slot = session.exec(
        select(BrollSlot).where(
            BrollSlot.id == slot_id, BrollSlot.project_id == project_id
        )
    ).first()
    if not slot:
        raise HTTPException(status_code=404, detail="B-roll slot not found")
    if slot.locked:
        raise HTTPException(status_code=409, detail="B-roll slot is locked")

    candidate = session.exec(
        select(BrollCandidate).where(
            BrollCandidate.id == payload.candidate_id,
            BrollCandidate.project_id == project_id,
            BrollCandidate.slot_id == slot_id,
        )
    ).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="B-roll candidate not found")

    if not candidate.asset_id:
        _materialize_candidate_asset(session, project_id, candidate)

    slot.status = "chosen"
    slot.chosen_candidate_id = candidate.id
    slot.updated_at = _utcnow()
    session.add(slot)
    session.add(
        BrollChoice(
            project_id=project_id,
            slot_id=slot_id,
            candidate_id=candidate.id,
            action="choose",
            payload_json=_json_dumps(
                {"candidate_id": candidate.id, "asset_id": candidate.asset_id}
            ),
        )
    )
    session.commit()

    updated = _load_slots_with_candidates(
        session,
        project_id=project_id,
        transcript_id=slot.transcript_id,
        slot_ids=[slot_id],
    )
    if not updated:
        raise HTTPException(
            status_code=500, detail="Failed to load updated B-roll slot"
        )
    return updated[0]


@router.post("/slots/{slot_id}/reject", response_model=BrollSlotResponse)
def reject_broll_slot(
    slot_id: str,
    payload: BrollRejectRequest,
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> BrollSlotResponse:
    _require_project(session, project_id)

    slot = session.exec(
        select(BrollSlot).where(
            BrollSlot.id == slot_id, BrollSlot.project_id == project_id
        )
    ).first()
    if not slot:
        raise HTTPException(status_code=404, detail="B-roll slot not found")
    if slot.locked:
        raise HTTPException(status_code=409, detail="B-roll slot is locked")

    slot.status = "rejected"
    slot.chosen_candidate_id = None
    slot.updated_at = _utcnow()
    session.add(slot)
    session.add(
        BrollChoice(
            project_id=project_id,
            slot_id=slot_id,
            candidate_id=None,
            action="reject",
            payload_json=_json_dumps({"reason": payload.reason or ""}),
        )
    )
    session.commit()

    updated = _load_slots_with_candidates(
        session,
        project_id=project_id,
        transcript_id=slot.transcript_id,
        slot_ids=[slot_id],
    )
    if not updated:
        raise HTTPException(
            status_code=500, detail="Failed to load updated B-roll slot"
        )
    return updated[0]
