from __future__ import annotations

from bisect import bisect_left
import json
import mimetypes
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import subprocess
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, delete, select

from ..broll_ai_service import expand_broll_queries, rerank_broll_candidates
from ..broll_external_service import ExternalBrollCandidate, search_external_broll_candidates
from ..broll_generative_service import generate_generative_broll_candidates
from ..broll_llm_service import (
    build_broll_search_strategy,
    infer_broll_domain_context,
    llm_rerank_broll_candidates,
)
from ..broll_planner_service import plan_broll
from ..config import get_settings
from ..database import engine, get_session
from ..jobs import create_job, find_recent_active_job, set_job_status
from ..media_utils import probe_duration_seconds, probe_stream_flags
from ..models import BrollCandidate, BrollChoice, BrollPlan, BrollPlanBeat, BrollSlot, Job, MediaAsset, Project, Transcript
from ..storage import storage
from ..schemas import (
    BrollAutoApplyRequest,
    BrollAutoApplyResponse,
    BrollCandidateResponse,
    BrollCoverageSectionResponse,
    BrollChooseRequest,
    BrollPlanBeatResponse,
    BrollPlanRequest,
    BrollPlanResponse,
    BrollRejectRequest,
    BrollRerollRequest,
    BrollSyncRequest,
    BrollSyncResponse,
    BrollSlotResponse,
    BrollSuggestRequest,
    BrollSuggestResponse,
    BrollUndoResponse,
    Clip,
    JobResponse,
    OperationPayload,
)
from ..timeline_service import apply_operation, get_timeline_row, load_timeline_state, save_timeline_state

router = APIRouter(prefix="/api/v1/broll", tags=["broll"])
settings = get_settings()

_SENTENCE_END_RE = re.compile(r"[.!?]$")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "i", "if",
    "in", "into", "is", "it", "its", "of", "on", "or", "our", "that", "the", "their", "there", "this",
    "to", "was", "we", "were", "with", "you", "your", "about", "after", "before", "during", "then", "than",
    "here", "now", "yeah", "okay", "ok", "just", "really", "very", "got", "going", "go", "back", "last",
    "round", "team", "new", "what", "who", "when", "where", "why", "how", "im", "i'm", "ive", "i've",
}
_THREE_WAYS_SHOTS: tuple[tuple[str, str], ...] = (
    ("wide", "wide shot"),
    ("medium", "medium shot"),
    ("detail", "detail shot"),
)
_BROLL_TX_SLOT_ID = "__broll_transaction__"
_POSITIVE_ENERGY_WORDS = {
    "amazing", "awesome", "boom", "build", "crazy", "fast", "fire", "go", "great",
    "hype", "insane", "massive", "power", "rapid", "rush", "strong", "top", "viral", "win",
}
_NEGATIVE_ENERGY_WORDS = {
    "alone", "broken", "calm", "dark", "death", "empty", "fear", "lost", "pain",
    "sad", "silent", "slow", "soft", "still", "tired", "weak", "worry",
}
_LOCAL_VISUAL_INTENT_CUES: dict[str, set[str]] = {
    "literal_demo": {"app", "camera", "demo", "device", "phone", "product", "screen", "tutorial"},
    "process_step": {"build", "dashboard", "editing", "hands", "keyboard", "packing", "process", "screen", "testing", "workflow"},
    "environment_context": {"city", "crowd", "factory", "meeting", "office", "shop", "street", "studio", "warehouse", "workspace"},
    "reaction_payoff": {"celebration", "crowd", "growth", "launch", "result", "smile", "success", "team", "win"},
    "abstract_support": {"background", "bokeh", "light", "motion", "shadow", "texture"},
}
_LOCAL_SHOT_STYLE_CUES: dict[str, set[str]] = {
    "wide": {"city", "crowd", "factory", "group", "landscape", "office", "room", "shop", "stage", "street", "team", "warehouse", "wide"},
    "medium": {"desk", "host", "meeting", "person", "speaker", "studio", "team"},
    "detail": {"close", "dashboard", "detail", "device", "editing", "hands", "keyboard", "macro", "phone", "screen"},
}
_VISUAL_INTENT_QUERY_MODE = {
    "literal_demo": "literal",
    "process_step": "process",
    "environment_context": "environment",
    "reaction_payoff": "reaction",
    "abstract_support": "abstract",
}
_SEQUENCE_MEMORY = 4


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


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
        raise HTTPException(status_code=500, detail="Stored transcript words are invalid") from exc

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


def _load_plan_response(session: Session, plan_id: str, *, project_id: str) -> BrollPlanResponse:
    plan = session.exec(select(BrollPlan).where(BrollPlan.id == plan_id, BrollPlan.project_id == project_id)).first()
    if not plan:
        raise HTTPException(status_code=404, detail="B-roll plan not found")
    beats = list(
        session.exec(
            select(BrollPlanBeat)
            .where(BrollPlanBeat.plan_id == plan.id, BrollPlanBeat.project_id == project_id)
            .order_by(BrollPlanBeat.beat_index.asc(), BrollPlanBeat.created_at.asc())
        ).all()
    )
    try:
        coverage = json.loads(plan.coverage_json or "{}")
    except json.JSONDecodeError:
        coverage = {}
    uncovered_ranges = coverage.get("uncovered_ranges", []) if isinstance(coverage, dict) else []
    coverage_sections = coverage.get("coverage_sections", []) if isinstance(coverage, dict) else []
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
            .where(MediaAsset.project_id == project.id, MediaAsset.media_type == "video")
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
        raise HTTPException(status_code=400, detail="Planner produced no usable B-roll beats")

    plan = BrollPlan(
        project_id=project.id,
        transcript_id=transcript.id,
        plan_version=str(planner_result.get("plan_version") or "v1"),
        fallback_used=bool(planner_result.get("fallback_used", True)),
        planner_model=str(planner_result.get("planner_model")) if planner_result.get("planner_model") else None,
        request_json=_json_dumps(payload.model_dump(mode="json")),
        coverage_json=_json_dumps(planner_result.get("coverage") or {}),
    )
    session.add(plan)

    for idx, beat in enumerate(beats):
        if not isinstance(beat, dict):
            continue
        start_sec = round(float(beat.get("start_sec", 0.0)), 3)
        end_sec = round(float(beat.get("end_sec", start_sec + 0.5)), 3)
        timeline_window = _resolve_slot_timeline_window(start_sec, end_sec, video_clips=video_clips)
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
    transcript = session.exec(transcript_query.order_by(Transcript.created_at.desc())).first()
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found. Generate transcript before requesting B-roll.")
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
        if not force_short and min_chunk_duration_sec > 0 and duration < min_chunk_duration_sec:
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
        duration_reached = max_chunk_duration_sec > 0 and duration >= max_chunk_duration_sec
        min_duration_reached = min_chunk_duration_sec <= 0 or duration >= min_chunk_duration_sec
        if duration_reached or cap_reached or (sentence_end and min_duration_reached):
            flush(force_short=duration_reached or cap_reached)
            if len(chunks) >= max_slots:
                break

    if len(chunks) < max_slots:
        flush(force_short=True)
    return chunks[:max_slots]


def _extract_concepts(text: str) -> tuple[str, list[str]]:
    tokens = [token.lower() for token in _WORD_RE.findall(text)]
    filtered = [token for token in tokens if len(token) >= 3 and token not in _STOP_WORDS]
    if not filtered:
        fallback = [token for token in tokens if len(token) >= 4 and token not in _STOP_WORDS]
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


def _local_duration_fit(candidate_duration: float | None, slot_duration: float) -> float:
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


def _push_recent_item(items: list[object], value: object, *, limit: int = _SEQUENCE_MEMORY) -> None:
    if value in (None, "", []):
        return
    items.insert(0, value)
    del items[limit:]


def _candidate_label_key(source_label: str | None, asset_id: str | None, source_url: str | None) -> str:
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
    candidates: list[tuple[str, str | None, str | None, str | None, float, dict[str, object]]],
    *,
    sequence_state: dict[str, list[object]],
) -> list[tuple[str, str | None, str | None, str | None, float, dict[str, object]]]:
    if not candidates:
        return []

    diversified: list[tuple[str, str | None, str | None, str | None, float, dict[str, object]]] = []
    for source_type, asset_id, source_url, source_label, score, reason in candidates:
        reason_payload = dict(reason) if isinstance(reason, dict) else {}
        breakdown = dict(reason_payload.get("score_breakdown") or {}) if isinstance(reason_payload.get("score_breakdown"), dict) else {}
        diversity_multiplier = 1.0
        query_mode = str(reason_payload.get("query_mode") or "").strip().lower()
        label_key = _candidate_label_key(source_label, asset_id, source_url)
        signature = _candidate_signature_terms(source_label=source_label, reason=reason_payload)

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
                overlap = len(signature.intersection(prior)) / max(min(len(signature), 4), 1)
                recent_overlap = max(recent_overlap, overlap)
            if recent_overlap >= 0.75:
                diversity_multiplier *= 0.86
            elif recent_overlap >= 0.5:
                diversity_multiplier *= 0.93

        breakdown["diversity"] = round(_clamp(diversity_multiplier, 0.0, 1.0), 3)
        reason_payload["score_breakdown"] = breakdown
        if "confidence" in reason_payload:
            try:
                reason_payload["confidence"] = round(_clamp(float(reason_payload["confidence"]) * max(diversity_multiplier, 0.88), 0.0, 1.0), 3)
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
    candidate_row: tuple[str, str | None, str | None, str | None, float, dict[str, object]] | None,
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


def _filename_tokens(filename: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(filename)}


def _is_vertical_project(project: Project) -> bool:
    return int(project.height) >= int(project.width)


def _resolve_slot_pacing(project: Project) -> tuple[float, float]:
    if _is_vertical_project(project):
        minimum = max(0.25, min(settings.broll_shortform_min_sec, settings.broll_shortform_max_sec))
        maximum = max(minimum + 0.05, settings.broll_shortform_max_sec)
        return (minimum, maximum)
    return (0.35, 4.5)


def _clip_timeline_duration_sec(clip: Clip) -> float:
    return max((float(clip.end_sec) - float(clip.start_sec)) / max(float(clip.speed), 0.01), 0.0)


def _video_track_clips_sorted(timeline_state: object) -> list[Clip]:
    tracks = getattr(timeline_state, "tracks", [])
    for track in tracks:
        if getattr(track, "kind", "") != "video":
            continue
        clips = [clip for clip in getattr(track, "clips", []) if float(clip.end_sec) > float(clip.start_sec)]
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
        timeline_start = float(clip.timeline_start_sec) + ((overlap_start - source_start) / speed)
        timeline_end = float(clip.timeline_start_sec) + ((overlap_end - source_start) / speed)
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
        concept_hits = [token for token in concept_tokens if token in asset_terms or token in filename_terms]
        semantic_match = (len(concept_hits) / max(len(concept_tokens), 1)) if concept_tokens else 0.0
        semantic_score = min(semantic_match * 0.42, 0.42)

        diversity_score = 0.16 if asset.id != transcript_asset_id else 0.02
        duration_score = _local_duration_fit(asset.duration_sec, slot_duration) * 0.16
        intent_score = _local_intent_score(asset_terms, visual_intent)
        shot_score = _local_shot_score(asset_terms, shot_style)
        metadata_density = min(len(asset_terms), 8) / 8 if asset_terms else 0.0
        recency_ratio = 1.0 - (idx / total)
        recency_score = recency_ratio * 0.1
        primary_penalty = 0.84 if asset.id == transcript_asset_id and semantic_match < 0.5 else 1.0

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
                    ) * primary_penalty,
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
                    "query_mode": _VISUAL_INTENT_QUERY_MODE.get((visual_intent or "").strip().lower(), "literal"),
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

    merged: list[tuple[str, str | None, str | None, str | None, float, dict[str, object]]] = []

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
        leftovers: list[tuple[str, str | None, str | None, str | None, float, dict[str, object]]] = []
        for asset, score, reason in remaining_local:
            leftovers.append(("project_asset", asset.id, None, asset.filename, score, reason))
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
    deduped: list[tuple[str, str | None, str | None, str | None, float, dict[str, object]]] = []
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


def _resolve_asset_video_path(asset: MediaAsset) -> str:
    return storage.resolve_upload_asset(asset.storage_path)


@lru_cache(maxsize=64)
def _probe_video_dimensions(path: str, mtime_ns: int) -> tuple[int, int]:
    cmd = [
        settings.ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return (0, 0)
    raw = (proc.stdout or "").strip().splitlines()
    if not raw:
        return (0, 0)
    first = raw[0].strip()
    if "x" not in first:
        return (0, 0)
    width_raw, height_raw = first.split("x", 1)
    try:
        width = max(0, int(float(width_raw)))
        height = max(0, int(float(height_raw)))
    except ValueError:
        return (0, 0)
    return (width, height)


@lru_cache(maxsize=32)
def _extract_audio_transients(
    path: str,
    mtime_ns: int,
    sample_rate: int,
) -> tuple[float, ...]:
    try:
        import numpy as np  # type: ignore
    except Exception:
        return ()

    cmd = [
        settings.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True, timeout=120)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ()

    if not proc.stdout:
        return ()
    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if samples.size < sample_rate // 4:
        return ()

    hop = max(96, int(sample_rate * 0.02))
    frame_count = samples.size // hop
    if frame_count < 8:
        return ()

    trimmed = samples[: frame_count * hop]
    matrix = np.abs(trimmed.reshape(frame_count, hop))
    energy = np.sqrt(np.mean(matrix * matrix, axis=1))
    if energy.size < 4:
        return ()

    delta = np.maximum(energy[1:] - energy[:-1], 0.0)
    if delta.size == 0:
        return ()
    baseline = float(np.median(delta))
    spread = float(np.percentile(delta, 88) - baseline)
    threshold = baseline + max(spread * 0.30, 0.003)
    candidate_idx = np.where(delta >= threshold)[0] + 1
    if candidate_idx.size == 0:
        return ()

    min_step = max(1, int(round(0.16 / 0.02)))
    picked: list[int] = []
    for idx in candidate_idx.tolist():
        if not picked:
            picked.append(idx)
            continue
        if idx - picked[-1] >= min_step:
            picked.append(idx)
            continue
        prev_idx = picked[-1]
        if float(delta[idx - 1]) > float(delta[prev_idx - 1]):
            picked[-1] = idx
    if not picked:
        return ()
    times = tuple(round((idx * hop) / float(sample_rate), 3) for idx in picked[:6000])
    return times


def _snap_time_to_transient(value: float, transients: tuple[float, ...], window_sec: float) -> float:
    if not transients:
        return value
    idx = bisect_left(transients, value)
    candidates: list[float] = []
    if idx < len(transients):
        candidates.append(float(transients[idx]))
    if idx > 0:
        candidates.append(float(transients[idx - 1]))
    if not candidates:
        return value
    best = min(candidates, key=lambda item: abs(item - value))
    if abs(best - value) <= window_sec:
        return best
    return value


def _snap_chunks_to_audio_grid(
    chunks: list[dict[str, object]],
    audio_path: str,
    *,
    min_chunk_sec: float,
    max_chunk_sec: float,
) -> list[dict[str, object]]:
    if not settings.broll_audio_reactive_enabled:
        return chunks
    path = Path(audio_path)
    if not path.exists():
        return chunks
    transients = _extract_audio_transients(
        str(path.resolve()),
        path.stat().st_mtime_ns,
        max(4000, settings.broll_audio_reactive_sample_rate),
    )
    if not transients:
        return chunks

    snapped: list[dict[str, object]] = []
    prev_end = 0.0
    window_sec = max(0.05, settings.broll_audio_reactive_window_sec)
    for chunk in chunks:
        start_sec = float(chunk["start_sec"])
        end_sec = float(chunk["end_sec"])
        start_sec = _snap_time_to_transient(start_sec, transients, window_sec)
        end_sec = _snap_time_to_transient(end_sec, transients, window_sec)

        if max_chunk_sec > 0 and end_sec - start_sec > max_chunk_sec:
            end_sec = start_sec + max_chunk_sec
        if min_chunk_sec > 0 and end_sec - start_sec < min_chunk_sec:
            end_sec = start_sec + min_chunk_sec

        start_sec = max(prev_end, start_sec)
        end_sec = max(start_sec + 0.06, end_sec)
        updated = dict(chunk)
        updated["start_sec"] = round(start_sec, 3)
        updated["end_sec"] = round(end_sec, 3)
        snapped.append(updated)
        prev_end = end_sec
    return snapped


def _detect_focus_track(path: str, *, max_samples: int = 60) -> list[dict[str, float]] | None:
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            return None

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = 30.0

        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        samples: list[tuple[float, float]] = []
        previous_gray = None
        frame_idx = 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(max(frame_count, max_samples * 4) // max_samples))
        while len(samples) < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            frame_idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(36, 36))
            if len(faces) > 0:
                largest = max(faces, key=lambda item: int(item[2]) * int(item[3]))
                focus = float(largest[0] + (largest[2] / 2)) / float(width)
                previous_gray = gray
            else:
                focus = 0.5
                if previous_gray is not None:
                    diff = cv2.absdiff(gray, previous_gray)
                    _, mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
                    moments = cv2.moments(mask)
                    if moments["m00"] > 0:
                        motion_x = float(moments["m10"] / moments["m00"])
                        focus = motion_x / float(width)
            focus = max(0.0, min(1.0, focus))
            time_sec = max(0.0, frame_idx / fps)
            samples.append((round(time_sec, 3), focus))
            previous_gray = gray

        if not samples:
            return None

        smoothed: list[tuple[float, float]] = []
        for idx, (time_sec, focus) in enumerate(samples):
            if idx == 0:
                smoothed.append((time_sec, focus))
                continue
            prev_focus = smoothed[-1][1]
            smooth_focus = (0.68 * prev_focus) + (0.32 * focus)
            smoothed.append((time_sec, smooth_focus))

        keyframes: list[dict[str, float]] = []
        for time_sec, focus in smoothed:
            if not keyframes:
                keyframes.append({"time_sec": 0.0, "x_ratio": round(focus, 4)})
                continue
            prev = keyframes[-1]
            if abs(float(prev["x_ratio"]) - float(focus)) < 0.008 and (time_sec - float(prev["time_sec"])) < 0.20:
                continue
            keyframes.append({"time_sec": round(time_sec, 3), "x_ratio": round(focus, 4)})

        if not keyframes:
            return None
        duration = max(float(samples[-1][0]), 0.0)
        if duration > 0.05 and keyframes[-1]["time_sec"] < duration:
            keyframes.append({"time_sec": round(duration, 3), "x_ratio": float(keyframes[-1]["x_ratio"])})

        if len(keyframes) > 24:
            stride = max(1, len(keyframes) // 24)
            reduced = [keyframes[0]]
            reduced.extend(keyframes[idx] for idx in range(stride, len(keyframes), stride))
            if reduced[-1]["time_sec"] != keyframes[-1]["time_sec"]:
                reduced.append(keyframes[-1])
            keyframes = reduced[:24]
        return keyframes
    finally:
        cap.release()


def _detect_focus_x_ratio(path: str) -> float | None:
    track = _detect_focus_track(path)
    if not track:
        return None
    values = sorted(float(item.get("x_ratio", 0.5)) for item in track)
    if not values:
        return None
    return values[len(values) // 2]


def _analyze_center_visual_risk(path: str, *, max_samples: int = 40) -> tuple[float, float, str]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return (0.5, 0.15, "medium")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return (0.5, 0.15, "medium")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            return (0.5, 0.15, "medium")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(max(frame_count, max_samples * 5) // max_samples))

        brightness_values: list[float] = []
        texture_values: list[float] = []
        frame_idx = 0
        while len(brightness_values) < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            frame_idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crop_w = max(10, int(width * 0.42))
            crop_h = max(10, int(height * 0.34))
            x0 = max(0, min(width - crop_w, (width - crop_w) // 2))
            y0 = max(0, min(height - crop_h, (height - crop_h) // 2))
            roi = gray[y0 : y0 + crop_h, x0 : x0 + crop_w]
            if roi.size == 0:
                continue
            brightness_values.append(float(np.mean(roi)) / 255.0)
            lap = cv2.Laplacian(roi, cv2.CV_64F)
            texture_values.append(float(np.var(lap)) / 3000.0)

        if not brightness_values:
            return (0.5, 0.15, "medium")
        brightness = max(0.0, min(1.0, float(np.mean(brightness_values))))
        texture = max(0.0, min(1.0, float(np.mean(texture_values))))
        if brightness >= 0.64 or texture >= 0.26:
            risk = "high"
        elif brightness >= 0.54 or texture >= 0.17:
            risk = "medium"
        else:
            risk = "low"
        return (round(brightness, 3), round(texture, 3), risk)
    finally:
        cap.release()


def _parse_asset_metadata(asset: MediaAsset | None) -> dict[str, object]:
    if asset is None:
        return {}
    try:
        payload = json.loads(asset.metadata_json or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _ensure_asset_focus_metadata(session: Session, asset: MediaAsset) -> dict[str, object]:
    metadata = _parse_asset_metadata(asset)
    path = Path(_resolve_asset_video_path(asset))
    if not path.exists():
        return metadata

    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)
    if width <= 0 or height <= 0:
        width, height = _probe_video_dimensions(str(path.resolve()), path.stat().st_mtime_ns)
        if width > 0 and height > 0:
            metadata["width"] = width
            metadata["height"] = height

    if settings.broll_auto_reframe_enabled and width > 0 and height > 0 and width > height:
        if "focus_track" not in metadata:
            focus_track = _detect_focus_track(str(path.resolve()))
            if focus_track:
                metadata["focus_track"] = focus_track
        if "focus_x" not in metadata:
            focus_x = _detect_focus_x_ratio(str(path.resolve()))
            if focus_x is not None:
                metadata["focus_x"] = round(float(focus_x), 4)

    if any(key not in metadata for key in ("center_brightness", "center_texture", "text_safety_risk")):
        brightness, texture, risk = _analyze_center_visual_risk(str(path.resolve()))
        metadata["center_brightness"] = brightness
        metadata["center_texture"] = texture
        metadata["text_safety_risk"] = risk

    asset.metadata_json = _json_dumps(metadata)
    session.add(asset)
    return metadata


def _build_vertical_crop(project: Project, width: int, height: int, focus_x: float | None) -> dict[str, int] | None:
    if width <= 0 or height <= 0:
        return None
    if not _is_vertical_project(project):
        return None
    if width <= height:
        return None

    target_ratio = float(project.width) / float(project.height)
    crop_width = int(round(height * target_ratio))
    crop_width = max(2, min(crop_width, width))
    crop_height = height
    focus = 0.5 if focus_x is None else max(0.0, min(1.0, focus_x))
    center_x = int(round(focus * width))
    left = max(0, min(width - crop_width, center_x - (crop_width // 2)))
    return {
        "x": int(left),
        "y": 0,
        "width": int(crop_width),
        "height": int(crop_height),
    }


def _build_vertical_crop_keyframes(
    project: Project,
    width: int,
    height: int,
    focus_track: object,
    *,
    clip_duration_sec: float,
) -> list[dict[str, float | int]]:
    if not isinstance(focus_track, list):
        return []
    if not _is_vertical_project(project):
        return []
    if width <= 0 or height <= 0 or width <= height:
        return []
    if clip_duration_sec <= 0:
        return []

    target_ratio = float(project.width) / float(project.height)
    crop_width = int(round(height * target_ratio))
    crop_width = max(2, min(crop_width, width))

    keyframes: list[dict[str, float | int]] = []
    previous_x: int | None = None
    for raw in focus_track:
        if not isinstance(raw, dict):
            continue
        try:
            time_sec = float(raw.get("time_sec", 0.0))
            x_ratio = float(raw.get("x_ratio", 0.5))
        except (TypeError, ValueError):
            continue
        if time_sec < 0:
            continue
        if time_sec > clip_duration_sec:
            break
        center_x = int(round(max(0.0, min(1.0, x_ratio)) * width))
        x = max(0, min(width - crop_width, center_x - (crop_width // 2)))
        if previous_x is not None:
            x = int(round((0.65 * previous_x) + (0.35 * x)))
        previous_x = x
        keyframes.append(
            {
                "time_sec": round(time_sec, 3),
                "x": int(x),
                "y": 0,
            }
        )
    if not keyframes:
        return []

    first = keyframes[0]
    if float(first["time_sec"]) > 0:
        keyframes.insert(0, {"time_sec": 0.0, "x": int(first["x"]), "y": 0})
    last = keyframes[-1]
    if float(last["time_sec"]) < clip_duration_sec:
        keyframes.append(
            {
                "time_sec": round(clip_duration_sec, 3),
                "x": int(last["x"]),
                "y": 0,
            }
        )

    deduped: list[dict[str, float | int]] = []
    for item in keyframes:
        if deduped and abs(float(item["time_sec"]) - float(deduped[-1]["time_sec"])) < 0.001:
            deduped[-1] = item
            continue
        if deduped and abs(float(item["x"]) - float(deduped[-1]["x"])) < 1 and (float(item["time_sec"]) - float(deduped[-1]["time_sec"])) < 0.12:
            continue
        deduped.append(item)
    if len(deduped) > 24:
        stride = max(1, len(deduped) // 24)
        reduced = [deduped[0]]
        reduced.extend(deduped[idx] for idx in range(stride, len(deduped), stride))
        if reduced[-1]["time_sec"] != deduped[-1]["time_sec"]:
            reduced.append(deduped[-1])
        deduped = reduced[:24]
    return deduped


def _text_safety_preset_from_metadata(metadata: dict[str, object]) -> tuple[str | None, float]:
    risk = str(metadata.get("text_safety_risk") or "").strip().lower()
    if risk == "high":
        return ("text_safe_soft", 0.76)
    if risk == "medium":
        return ("text_safe_mild", 0.82)
    return (None, 1.0)


def _snapshot_overlay_clips(timeline_state: object) -> list[dict[str, object]]:
    tracks = getattr(timeline_state, "tracks", [])
    for track in tracks:
        if getattr(track, "kind", "") != "overlay":
            continue
        return [clip.model_dump(mode="json") for clip in getattr(track, "clips", [])]
    return []


def _restore_overlay_clips_from_snapshot(timeline_state: object, snapshot: list[dict[str, object]]) -> int:
    tracks = getattr(timeline_state, "tracks", [])
    overlay_track = next((track for track in tracks if getattr(track, "kind", "") == "overlay"), None)
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


def _safe_filename_from_url(url: str, fallback_stem: str = "broll") -> str:
    parsed = urlparse(url)
    stem = Path(parsed.path).stem or fallback_stem
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".webm", ".mkv"}:
        suffix = ".mp4"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    if not cleaned:
        cleaned = fallback_stem
    return f"{cleaned}-{uuid4().hex[:8]}{suffix}"


def _download_external_video(project_id: str, source_url: str) -> tuple[str, str, str]:
    parsed = urlparse(source_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=422, detail="B-roll source URL must be http(s)")

    project_dir = storage.upload_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename_from_url(source_url)
    destination = project_dir / filename

    max_bytes = max(5, settings.broll_external_download_max_mb) * 1024 * 1024
    timeout = httpx.Timeout(max(2.0, settings.broll_external_timeout_sec))

    total = 0
    try:
        with httpx.stream("GET", source_url, timeout=timeout, follow_redirects=True) as response:
            response.raise_for_status()
            content_type = (response.headers.get("content-type") or "video/mp4").split(";")[0].strip()
            with destination.open("wb") as stream:
                for chunk in response.iter_bytes(1024 * 256):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"External B-roll file too large (> {settings.broll_external_download_max_mb} MB)",
                        )
                    stream.write(chunk)
    except HTTPException:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail=f"Failed to download external B-roll: {exc}") from exc

    relative = str(destination.resolve().relative_to(storage.upload_root))
    mime_type = mimetypes.guess_type(destination.name)[0] or "video/mp4"
    return (str(destination.resolve()), relative, mime_type)


def _find_existing_asset_for_source_url(
    session: Session,
    *,
    project_id: str,
    source_url: str,
) -> MediaAsset | None:
    normalized_source = source_url.strip()
    if not normalized_source:
        return None

    assets = list(
        session.exec(
            select(MediaAsset)
            .where(MediaAsset.project_id == project_id, MediaAsset.media_type == "video")
            .order_by(MediaAsset.created_at.desc())
        ).all()
    )
    for asset in assets:
        metadata = _parse_asset_metadata(asset)
        if str(metadata.get("source_url") or "").strip() != normalized_source:
            continue
        if Path(_resolve_asset_video_path(asset)).exists():
            return asset
    return None


def _materialize_candidate_asset(session: Session, project_id: str, candidate: BrollCandidate) -> MediaAsset:
    if candidate.asset_id:
        existing = session.exec(
            select(MediaAsset).where(MediaAsset.id == candidate.asset_id, MediaAsset.project_id == project_id)
        ).first()
        if existing:
            return existing
    if not candidate.source_url:
        raise HTTPException(status_code=422, detail="Selected candidate has no importable source URL")

    existing_for_source = _find_existing_asset_for_source_url(
        session,
        project_id=project_id,
        source_url=candidate.source_url,
    )
    if existing_for_source is not None:
        candidate.asset_id = existing_for_source.id
        session.add(candidate)
        return existing_for_source

    # AI-generated clips are already on disk (local file path, not a URL)
    is_local_generated = (
        candidate.source_type == "generated_image_video"
        and not candidate.source_url.startswith("http")
        and Path(candidate.source_url).is_file()
    )

    if is_local_generated:
        # Copy the generated video from tmp to project uploads
        src_path = Path(candidate.source_url)
        upload_dir = Path(settings.upload_dir) / project_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest_name = f"ai_broll_{src_path.stem}.mp4"
        dest_path = upload_dir / dest_name
        import shutil
        shutil.copy2(str(src_path), str(dest_path))
        absolute_path = str(dest_path.resolve())
        relative_path = f"{project_id}/{dest_name}"
        guessed_mime = "video/mp4"
        # Clean up temp file
        try:
            src_path.unlink(missing_ok=True)
        except OSError:
            pass
    else:
        absolute_path, relative_path, guessed_mime = _download_external_video(project_id, candidate.source_url)

    stream_flags = probe_stream_flags(absolute_path)
    if not stream_flags.get("has_video", False):
        Path(absolute_path).unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Selected B-roll source has no video stream")

    reason_payload = _parse_reason_json(candidate)
    path_obj = Path(absolute_path)
    probe_width, probe_height = _probe_video_dimensions(str(path_obj.resolve()), path_obj.stat().st_mtime_ns)
    width = int(reason_payload.get("width") or probe_width or 0)
    height = int(reason_payload.get("height") or probe_height or 0)

    # Skip heavy OpenCV analysis for AI-generated clips (no real faces to track)
    focus_track = None
    focus_x = None
    if not is_local_generated:
        focus_track = _detect_focus_track(str(path_obj.resolve())) if settings.broll_auto_reframe_enabled and width > height else None
        if focus_track:
            ratios = sorted(float(item.get("x_ratio", 0.5)) for item in focus_track if isinstance(item, dict))
            if ratios:
                focus_x = ratios[len(ratios) // 2]
        if focus_x is None and settings.broll_auto_reframe_enabled and width > height:
            focus_x = _detect_focus_x_ratio(str(path_obj.resolve()))

    brightness, texture, risk = _analyze_center_visual_risk(str(path_obj.resolve()))

    duration_sec = probe_duration_seconds(absolute_path)
    source_filename = candidate.source_label or Path(relative_path).name
    metadata = {
        "source_type": candidate.source_type,
        "source_url": candidate.source_url,
        "width": width,
        "height": height,
        "center_brightness": brightness,
        "center_texture": texture,
        "text_safety_risk": risk,
        **stream_flags,
    }
    if focus_x is not None:
        metadata["focus_x"] = round(float(focus_x), 4)
    if focus_track:
        metadata["focus_track"] = focus_track
    if is_local_generated:
        metadata["ai_generated"] = True
    asset = MediaAsset(
        project_id=project_id,
        media_type="video",
        filename=source_filename[:180],
        storage_path=relative_path,
        mime_type=guessed_mime,
        duration_sec=duration_sec,
        metadata_json=_json_dumps(metadata),
    )
    session.add(asset)
    session.flush()
    candidate.asset_id = asset.id
    session.add(candidate)
    return asset


def _parse_anchor_word_ids(row: BrollSlot) -> list[str]:
    try:
        parsed = json.loads(row.anchor_word_ids_json or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def _parse_reason_json(row: BrollCandidate) -> dict[str, object]:
    try:
        parsed = json.loads(row.reason_json or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def _review_status_for_slot(row: BrollSlot, ordered_candidates: list[BrollCandidate]) -> tuple[str, list[str], str | None, str | None]:
    if row.status == "rejected":
        return ("rejected", [], None, "Rejected by user")
    if row.status == "chosen":
        return ("approved", [], None, "Approved for timeline")
    if not ordered_candidates:
        return ("unfilled", ["no_candidates"], None, "No candidates available")

    top_reason = _parse_reason_json(ordered_candidates[0])
    confidence = _confidence_from_reason(top_reason, float(ordered_candidates[0].score)) or 0.0
    weak_reason_codes = _weak_reason_codes_from_reason(top_reason)
    visual_intent = _visual_intent_from_reason(top_reason)
    if confidence >= settings.broll_confidence_autopick_threshold and not weak_reason_codes:
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
    if any(token in text for token in ("office", "studio", "warehouse", "street", "factory")):
        return "environment_context"
    return "abstract_support" if beat.section_label in {"hook", "outro"} else "literal_demo"


def _domain_context_for_retrieval(transcript_text: str, assets: list[MediaAsset]) -> dict[str, object]:
    asset_descriptors = [
        " ".join(part for part in (asset.filename, str(asset.metadata_json or "")[:200]) if part).strip()
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
) -> dict[str, object]:
    strategy = build_broll_search_strategy(
        chunk_text=chunk_text,
        concept_text=concept_text,
        visual_intent=visual_intent,
        query_hints=expanded_queries,
        max_queries=max(4, min(len(expanded_queries) + 2, 8)),
        domain_context=dict(domain_context),
    )
    search_concept = " ".join(str(strategy.get("search_concept") or concept_text).split()).strip() or concept_text
    search_visual_intent = str(strategy.get("visual_intent") or visual_intent).strip().lower() or visual_intent
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
        "stockability": str(strategy.get("stockability") or "medium").strip().lower() or "medium",
        "rationale": str(strategy.get("rationale") or "").strip(),
        "domain_context": dict(domain_context),
        "raw_strategy": strategy,
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


def _to_slot_response(row: BrollSlot, candidates: list[BrollCandidate]) -> BrollSlotResponse:
    ordered_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    review_status, weak_reason_codes, visual_intent, review_summary = _review_status_for_slot(row, ordered_candidates)
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
        candidates=[_to_candidate_response(candidate) for candidate in ordered_candidates],
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

    slots = list(session.exec(slot_query.order_by(BrollSlot.start_sec.asc(), BrollSlot.created_at.asc())).all())
    if not slots:
        return []

    ids = [slot.id for slot in slots]
    candidates = list(
        session.exec(
            select(BrollCandidate)
            .where(BrollCandidate.project_id == project_id, BrollCandidate.slot_id.in_(ids))
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
        if float(item["start_sec"]) < float(slot.end_sec) and float(item["end_sec"]) > float(slot.start_sec)
    ]
    overlap_text = " ".join(token for token in overlap_tokens if token).strip()
    if overlap_text:
        return overlap_text
    return slot.concept_text.strip()


@router.post("/plan", response_model=BrollPlanResponse)
def create_broll_plan(
    payload: BrollPlanRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> BrollPlanResponse:
    project = _require_project(session, project_id)
    transcript = _resolve_broll_transcript(session, project_id=project_id, transcript_id=payload.transcript_id)
    return _build_broll_plan(session, project=project, transcript=transcript, payload=payload)


@router.get("/plans/{plan_id}", response_model=BrollPlanResponse)
def get_broll_plan(
    plan_id: str,
    project_id: str,
    session: Session = Depends(get_session),
) -> BrollPlanResponse:
    _require_project(session, project_id)
    return _load_plan_response(session, plan_id, project_id=project_id)


@router.post("/suggest", response_model=BrollSuggestResponse)
def suggest_broll(
    payload: BrollSuggestRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> BrollSuggestResponse:
    project = _require_project(session, project_id)
    transcript = _resolve_broll_transcript(session, project_id=project_id, transcript_id=payload.transcript_id)
    plan_response = _build_broll_plan(
        session,
        project=project,
        transcript=transcript,
        payload=_plan_request_from_suggest(payload),
    )
    beats = [beat for beat in plan_response.beats if beat.should_place and beat.end_sec > beat.start_sec]
    if not beats:
        raise HTTPException(status_code=400, detail="Planner produced no eligible B-roll beats")

    assets = list(
        session.exec(
            select(MediaAsset)
            .where(MediaAsset.project_id == project_id, MediaAsset.media_type == "video")
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
                select(BrollSlot).where(BrollSlot.project_id == project_id, BrollSlot.transcript_id == transcript.id)
            ).all()
        )
        existing_slot_ids = [row.id for row in existing_slots]
        if existing_slot_ids:
            session.exec(
                delete(BrollChoice).where(BrollChoice.project_id == project_id, BrollChoice.slot_id.in_(existing_slot_ids))
            )
            session.exec(
                delete(BrollCandidate).where(
                    BrollCandidate.project_id == project_id,
                    BrollCandidate.slot_id.in_(existing_slot_ids),
                )
            )
            session.exec(delete(BrollSlot).where(BrollSlot.project_id == project_id, BrollSlot.id.in_(existing_slot_ids)))

    now = _utcnow()
    created_slot_ids: list[str] = []
    beat_metas: list[dict[str, object]] = []
    for idx, beat in enumerate(beats):
        beat_text = beat.segment_text.strip() or beat.concept_text.strip()
        concept_text = beat.concept_text.strip() or _extract_concepts(beat_text)[0]
        _ignored, concept_tokens = _extract_concepts(f"{concept_text} {beat_text}".strip())
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

        beat_metas.append({
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
        })

    # ── Phase 2: Fetch external candidates in PARALLEL ────────────────
    def _fetch_external_for_slot(meta: dict[str, object]) -> list[ExternalBrollCandidate]:
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
                        "search_strategy_rationale": str(search_strategy.get("rationale") or ""),
                        "stockability": str(search_strategy.get("stockability") or ""),
                        "blocked_terms": list(search_strategy.get("blocked_terms") or []),
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
            except Exception:
                pass
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
            except Exception:
                pass
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
                            "visual_intent": visual_intent,
                            "search_concept": concept_text,
                            "original_concept_text": original_concept_text,
                            "section_label": beat.section_label,
                            "shot_style": shot_style,
                            "source_strategy": beat.source_strategy,
                            "planner_confidence": round(float(beat.confidence), 3),
                            "planner_rationale": beat.rationale,
                            "query_hints": expanded_queries[:8],
                            "search_strategy_rationale": str(search_strategy.get("rationale") or ""),
                            "stockability": str(search_strategy.get("stockability") or ""),
                            "blocked_terms": list(search_strategy.get("blocked_terms") or []),
                            "domain_label": str(domain_context.get("domain") or ""),
                        }
                    ),
                )
            )
        _remember_sequence_candidate(sequence_state, merged_candidates[0])

    if not created_slot_ids:
        raise HTTPException(status_code=400, detail="No B-roll candidates available for current settings")

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
) -> JobResponse:
    _require_project(session, project_id)
    if not force:
        active = find_recent_active_job(session, project_id, kind="broll_suggest", within_seconds=0)
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
        raise HTTPException(status_code=409, detail=job.error or "B-roll suggest job failed")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="B-roll suggest job not completed")

    result_path = _broll_suggest_result_path(job.id)
    if not result_path.exists():
        raise HTTPException(status_code=500, detail="B-roll suggest result missing")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="B-roll suggest result payload invalid") from exc
    return BrollSuggestResponse.model_validate(payload)


@router.post("/auto-apply", response_model=BrollAutoApplyResponse)
def auto_apply_broll(
    payload: BrollAutoApplyRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> BrollAutoApplyResponse:
    project = _require_project(session, project_id)

    suggest_response = suggest_broll(_to_suggest_request(payload), project_id=project_id, session=session)
    slot_ids = [slot.id for slot in suggest_response.slots]
    if not slot_ids:
        raise HTTPException(status_code=400, detail="No B-roll slots available to auto-apply")

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
            .where(BrollCandidate.project_id == project_id, BrollCandidate.slot_id.in_(slot_ids))
            .order_by(BrollCandidate.score.desc(), BrollCandidate.created_at.asc())
        ).all()
    )
    by_slot: dict[str, list[BrollCandidate]] = {slot_id: [] for slot_id in slot_ids}
    for candidate in candidates:
        by_slot.setdefault(candidate.slot_id, []).append(candidate)

    selected_pairs: list[tuple[BrollSlot, BrollCandidate]] = []
    auto_chosen_slots = 0
    skipped_slots = 0
    for slot in slots:
        ordered = by_slot.get(slot.id, [])
        selected_candidate: BrollCandidate | None = None
        for candidate in ordered:
            reason = _parse_reason_json(candidate)
            confidence = _confidence_from_reason(reason, float(candidate.score))
            weak_reason_codes = _weak_reason_codes_from_reason(reason)
            if confidence is not None and confidence >= confidence_threshold and not weak_reason_codes:
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
        overlay_track = next((track for track in timeline_state.tracks if track.kind == "overlay"), None)
        for clip in list(overlay_track.clips) if overlay_track else []:
            try:
                apply_operation(
                    timeline_state,
                    OperationPayload(op_type="delete_broll_clip", params={"clip": clip.id}, source="ui"),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            timeline_changed = True

    selected_asset_ids = [candidate.asset_id for _slot, candidate in selected_pairs if candidate.asset_id]
    assets_by_id: dict[str, MediaAsset] = {}
    if selected_asset_ids:
        assets = list(
            session.exec(
                select(MediaAsset)
                .where(MediaAsset.project_id == project_id, MediaAsset.id.in_(selected_asset_ids))
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
                if _is_vertical_project(project) and source_duration < min_slot_sec and float(asset.duration_sec) >= min_slot_sec:
                    source_duration = min(float(asset.duration_sec), min_slot_sec)

        crop_payload: dict[str, int] | None = None
        crop_keyframes: list[dict[str, float | int]] = []
        overlay_opacity = round(float(payload.overlay_opacity), 3)
        preset: str | None = None
        if asset:
            try:
                focus_x = float(metadata.get("focus_x")) if metadata.get("focus_x") is not None else None
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
        timeline=load_timeline_state(timeline),
        slots=refreshed_slots,
    )


@router.post("/sync", response_model=BrollSyncResponse)
def sync_broll_to_timeline(
    payload: BrollSyncRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> BrollSyncResponse:
    project = _require_project(session, project_id)

    slot_query = select(BrollSlot).where(BrollSlot.project_id == project_id)
    if payload.transcript_id:
        slot_query = slot_query.where(BrollSlot.transcript_id == payload.transcript_id)
    if payload.slot_ids:
        slot_query = slot_query.where(BrollSlot.id.in_(payload.slot_ids))
    slots = list(session.exec(slot_query.order_by(BrollSlot.start_sec.asc(), BrollSlot.created_at.asc())).all())

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

    candidate_ids = [slot.chosen_candidate_id for slot in chosen_slots if slot.chosen_candidate_id]
    candidates = list(
        session.exec(
            select(BrollCandidate)
            .where(BrollCandidate.project_id == project_id, BrollCandidate.id.in_(candidate_ids))
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
        raise HTTPException(status_code=400, detail="No chosen B-roll candidates available to sync")

    session.commit()

    timeline = get_timeline_row(session, project_id)
    timeline_state = load_timeline_state(timeline)
    previous_overlay_clips = _snapshot_overlay_clips(timeline_state)
    timeline_changed = False
    if payload.clear_existing_overlay:
        overlay_track = next((track for track in timeline_state.tracks if track.kind == "overlay"), None)
        for clip in list(overlay_track.clips) if overlay_track else []:
            try:
                apply_operation(
                    timeline_state,
                    OperationPayload(op_type="delete_broll_clip", params={"clip": clip.id}, source="ui"),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            timeline_changed = True

    selected_asset_ids = [candidate.asset_id for _slot, candidate in selected_pairs if candidate.asset_id]
    assets_by_id: dict[str, MediaAsset] = {}
    if selected_asset_ids:
        assets = list(
            session.exec(
                select(MediaAsset)
                .where(MediaAsset.project_id == project_id, MediaAsset.id.in_(selected_asset_ids))
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
                if _is_vertical_project(project) and source_duration < min_slot_sec and float(asset.duration_sec) >= min_slot_sec:
                    source_duration = min(float(asset.duration_sec), min_slot_sec)

        crop_payload: dict[str, int] | None = None
        crop_keyframes: list[dict[str, float | int]] = []
        overlay_opacity = round(float(payload.overlay_opacity), 3)
        preset: str | None = None
        if asset:
            try:
                focus_x = float(metadata.get("focus_x")) if metadata.get("focus_x") is not None else None
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
        raise HTTPException(status_code=404, detail="No B-roll transaction found to undo")

    try:
        payload = json.loads(row.payload_json or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Stored B-roll transaction payload is invalid") from exc
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
) -> list[BrollSlotResponse]:
    _require_project(session, project_id)
    return _load_slots_with_candidates(session, project_id=project_id, transcript_id=transcript_id)


@router.post("/slots/{slot_id}/reroll", response_model=BrollSlotResponse)
def reroll_broll_slot(
    slot_id: str,
    payload: BrollRerollRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> BrollSlotResponse:
    _require_project(session, project_id)

    slot = session.exec(select(BrollSlot).where(BrollSlot.id == slot_id, BrollSlot.project_id == project_id)).first()
    if not slot:
        raise HTTPException(status_code=404, detail="B-roll slot not found")
    if slot.locked:
        raise HTTPException(status_code=409, detail="B-roll slot is locked")

    transcript: Transcript | None = None
    if slot.transcript_id:
        transcript = session.exec(
            select(Transcript).where(Transcript.id == slot.transcript_id, Transcript.project_id == project_id)
        ).first()

    chunk_text = _resolve_slot_chunk_text(slot, transcript)
    concept_text = slot.concept_text.strip() or _extract_concepts(chunk_text)[0]
    _ignored, concept_tokens = _extract_concepts(f"{concept_text} {chunk_text}".strip())
    existing_slot_rows = list(
        session.exec(
            select(BrollCandidate)
            .where(BrollCandidate.project_id == project_id, BrollCandidate.slot_id == slot.id)
            .order_by(BrollCandidate.score.desc(), BrollCandidate.created_at.asc())
        ).all()
    )
    existing_top_reason = _parse_reason_json(existing_slot_rows[0]) if existing_slot_rows else {}
    visual_intent = _visual_intent_from_reason(existing_top_reason) or _visual_intent_for_beat(
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
            .where(BrollSlot.project_id == project_id, BrollSlot.transcript_id == slot.transcript_id)
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
            .where(MediaAsset.project_id == project_id, MediaAsset.media_type == "video")
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
                    "search_strategy_rationale": str(search_strategy.get("rationale") or ""),
                    "stockability": str(search_strategy.get("stockability") or ""),
                    "blocked_terms": list(search_strategy.get("blocked_terms") or []),
                    "domain_label": str(domain_context.get("domain") or ""),
                },
            )
            for candidate in external_candidates
        ]
        top_external_score = max((candidate.score for candidate in external_candidates), default=0.0)
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
        except Exception:
            pass
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
        except Exception:
            pass
    existing_candidates = list(
        session.exec(
            select(BrollCandidate)
            .where(BrollCandidate.project_id == project_id, BrollCandidate.slot_id == slot.id)
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
        raise HTTPException(status_code=400, detail="No B-roll candidates available for reroll")

    seen_asset_ids = {candidate.asset_id for candidate in existing_candidates if candidate.asset_id}
    seen_urls = {candidate.source_url for candidate in existing_candidates if candidate.source_url}

    new_candidates: list[tuple[str, str | None, str | None, str | None, float, dict[str, object]]] = []
    for source_type, asset_id, source_url, source_label, score, reason in merged_candidates:
        if asset_id and asset_id in seen_asset_ids:
            continue
        if source_url and source_url in seen_urls:
            continue
        new_candidates.append((source_type, asset_id, source_url, source_label, score, reason))

    if not new_candidates:
        raise HTTPException(status_code=400, detail="No new B-roll variants found for this slot")

    added_candidate_ids: list[str] = []
    for source_type, asset_id, source_url, source_label, score, reason in new_candidates:
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
                    "section_label": str(existing_top_reason.get("section_label") or "body"),
                    "shot_style": shot_style,
                    "source_strategy": str(existing_top_reason.get("source_strategy") or "local_first"),
                    "planner_confidence": existing_top_reason.get("planner_confidence"),
                    "planner_rationale": str(existing_top_reason.get("planner_rationale") or ""),
                    "query_hints": expanded_queries[:8],
                    "search_strategy_rationale": str(search_strategy.get("rationale") or ""),
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
        raise HTTPException(status_code=500, detail="Failed to load rerolled B-roll slot")
    return updated[0]


@router.post("/slots/{slot_id}/choose", response_model=BrollSlotResponse)
def choose_broll_candidate(
    slot_id: str,
    payload: BrollChooseRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> BrollSlotResponse:
    _require_project(session, project_id)

    slot = session.exec(select(BrollSlot).where(BrollSlot.id == slot_id, BrollSlot.project_id == project_id)).first()
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
            payload_json=_json_dumps({"candidate_id": candidate.id, "asset_id": candidate.asset_id}),
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
        raise HTTPException(status_code=500, detail="Failed to load updated B-roll slot")
    return updated[0]


@router.post("/slots/{slot_id}/reject", response_model=BrollSlotResponse)
def reject_broll_slot(
    slot_id: str,
    payload: BrollRejectRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> BrollSlotResponse:
    _require_project(session, project_id)

    slot = session.exec(select(BrollSlot).where(BrollSlot.id == slot_id, BrollSlot.project_id == project_id)).first()
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
        raise HTTPException(status_code=500, detail="Failed to load updated B-roll slot")
    return updated[0]
