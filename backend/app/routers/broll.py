from __future__ import annotations

import json
import mimetypes
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, delete, select

from ..broll_ai_service import expand_broll_queries, rerank_broll_candidates
from ..broll_external_service import ExternalBrollCandidate, search_external_broll_candidates
from ..config import get_settings
from ..database import get_session
from ..media_utils import probe_duration_seconds, probe_stream_flags
from ..models import BrollCandidate, BrollChoice, BrollSlot, MediaAsset, Project, Transcript
from ..storage import storage
from ..schemas import (
    BrollAutoApplyRequest,
    BrollAutoApplyResponse,
    BrollCandidateResponse,
    BrollChooseRequest,
    BrollRejectRequest,
    BrollRerollRequest,
    BrollSlotResponse,
    BrollSuggestRequest,
    BrollSuggestResponse,
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
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


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


def _chunk_words(words: list[dict[str, object]], min_chunk_words: int, max_slots: int) -> list[dict[str, object]]:
    if not words:
        return []

    chunks: list[dict[str, object]] = []
    current: list[dict[str, object]] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        if len(current) < min_chunk_words:
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
                flush()

        current.append(word)
        token = str(word["text"]).strip()
        sentence_end = bool(_SENTENCE_END_RE.search(token))
        cap_reached = len(current) >= 14
        if sentence_end or cap_reached:
            flush()
            if len(chunks) >= max_slots:
                break

    if len(chunks) < max_slots:
        flush()
    return chunks[:max_slots]


def _extract_concepts(text: str) -> tuple[str, list[str]]:
    tokens = [token.lower() for token in _WORD_RE.findall(text)]
    filtered = [token for token in tokens if len(token) >= 3 and token not in _STOP_WORDS]
    if not filtered:
        filtered = [token.lower() for token in tokens[:3] if token]
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


def _filename_tokens(filename: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(filename)}


def _rank_candidates(
    *,
    assets: list[MediaAsset],
    transcript_asset_id: str,
    concept_tokens: list[str],
    candidates_per_slot: int,
    slot_duration: float,
) -> list[tuple[MediaAsset, float, dict[str, object]]]:
    if not assets:
        return []

    ranked: list[tuple[MediaAsset, float, dict[str, object]]] = []
    total = max(len(assets), 1)

    for idx, asset in enumerate(assets):
        filename_terms = _filename_tokens(asset.filename)
        concept_hits = [token for token in concept_tokens if token in filename_terms]
        semantic_match = (len(concept_hits) / max(len(concept_tokens), 1)) if concept_tokens else 0.0
        semantic_score = min(semantic_match * 0.45, 0.45)

        diversity_score = 0.18 if asset.id != transcript_asset_id else 0.06

        duration_score = 0.05
        if asset.duration_sec and asset.duration_sec >= max(slot_duration, 1.0):
            duration_score = 0.12

        recency_ratio = 1.0 - (idx / total)
        recency_score = recency_ratio * 0.15

        score = max(0.0, min(1.0, round(0.20 + semantic_score + diversity_score + duration_score + recency_score, 3)))

        tags = ["project_asset", "visual_variety"]
        if concept_hits:
            tags.append("keyword_match")
        if asset.id != transcript_asset_id:
            tags.append("not_primary_asset")

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
                        "recency_score": round(recency_score, 3),
                    },
                    "keyword_hits": concept_hits,
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


def _materialize_candidate_asset(session: Session, project_id: str, candidate: BrollCandidate) -> MediaAsset:
    if candidate.asset_id:
        existing = session.exec(
            select(MediaAsset).where(MediaAsset.id == candidate.asset_id, MediaAsset.project_id == project_id)
        ).first()
        if existing:
            return existing
    if not candidate.source_url:
        raise HTTPException(status_code=422, detail="Selected candidate has no importable source URL")

    absolute_path, relative_path, guessed_mime = _download_external_video(project_id, candidate.source_url)
    stream_flags = probe_stream_flags(absolute_path)
    if not stream_flags.get("has_video", False):
        Path(absolute_path).unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Selected B-roll source has no video stream")

    duration_sec = probe_duration_seconds(absolute_path)
    source_filename = candidate.source_label or Path(relative_path).name
    metadata = {
        "source_type": candidate.source_type,
        "source_url": candidate.source_url,
        **stream_flags,
    }
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
        reason=reason,
        created_at=row.created_at.isoformat(),
    )


def _to_slot_response(row: BrollSlot, candidates: list[BrollCandidate]) -> BrollSlotResponse:
    ordered_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
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


@router.post("/suggest", response_model=BrollSuggestResponse)
def suggest_broll(
    payload: BrollSuggestRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> BrollSuggestResponse:
    _require_project(session, project_id)

    transcript_query = select(Transcript).where(Transcript.project_id == project_id)
    if payload.transcript_id:
        transcript_query = transcript_query.where(Transcript.id == payload.transcript_id)
    transcript = session.exec(transcript_query.order_by(Transcript.created_at.desc())).first()
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found. Generate transcript before requesting B-roll.")

    words = _load_transcript_words(transcript)
    if not words:
        raise HTTPException(status_code=400, detail="Transcript has no words")

    chunks = _chunk_words(words, min_chunk_words=payload.min_chunk_words, max_slots=payload.max_slots)
    if not chunks:
        raise HTTPException(status_code=400, detail="No eligible transcript chunks found for B-roll suggestions")

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
    for chunk in chunks:
        chunk_text = str(chunk["text"])
        concept_text, concept_tokens = _extract_concepts(chunk_text)
        expanded_queries = expand_broll_queries(
            chunk_text=chunk_text,
            concept_text=concept_text,
            concept_tokens=concept_tokens,
        )
        slot_duration_sec = max(float(chunk["end_sec"]) - float(chunk["start_sec"]), 0.1)
        ranked_candidates: list[tuple[MediaAsset, float, dict[str, object]]] = []
        if payload.include_project_assets:
            ranked_candidates = _rank_candidates(
                assets=assets,
                transcript_asset_id=transcript.asset_id,
                concept_tokens=concept_tokens,
                candidates_per_slot=payload.candidates_per_slot,
                slot_duration=slot_duration_sec,
            )
        external_candidates: list[ExternalBrollCandidate] = []
        if payload.include_external_sources:
            external_candidates = search_external_broll_candidates(
                concept_text=concept_text,
                concept_tokens=concept_tokens,
                slot_duration_sec=slot_duration_sec,
                limit=payload.candidates_per_slot,
                query_hints=expanded_queries,
            )
        merged_candidates = _mix_candidates(
            local_candidates=ranked_candidates,
            external_candidates=external_candidates,
            limit=payload.candidates_per_slot,
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
                )
            except Exception:
                pass

        if not merged_candidates:
            continue

        slot = BrollSlot(
            project_id=project_id,
            transcript_id=transcript.id,
            start_sec=round(float(chunk["start_sec"]), 3),
            end_sec=round(float(chunk["end_sec"]), 3),
            anchor_word_ids_json=_json_dumps(chunk["word_ids"]),
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
                    reason_json=_json_dumps(reason),
                )
            )

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


@router.post("/auto-apply", response_model=BrollAutoApplyResponse)
def auto_apply_broll(
    payload: BrollAutoApplyRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> BrollAutoApplyResponse:
    _require_project(session, project_id)

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
            confidence = _confidence_from_reason(_parse_reason_json(candidate), float(candidate.score))
            if confidence is not None and confidence >= confidence_threshold:
                selected_candidate = candidate
                break
        if selected_candidate is None and payload.fallback_to_top_candidate and ordered:
            selected_candidate = ordered[0]

        if selected_candidate is None:
            slot.status = "rejected"
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
                            "reason": "no_candidate_over_threshold",
                            "threshold": round(confidence_threshold, 3),
                        }
                    ),
                )
            )
            skipped_slots += 1
            continue

        try:
            if not selected_candidate.asset_id:
                _materialize_candidate_asset(session, project_id, selected_candidate)
        except HTTPException as exc:
            slot.status = "rejected"
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

    session.commit()

    timeline = get_timeline_row(session, project_id)
    timeline_state = load_timeline_state(timeline)
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
    for slot, candidate in selected_pairs:
        if not candidate.asset_id:
            continue
        slot_duration = max(float(slot.end_sec) - float(slot.start_sec), 0.2)
        asset = assets_by_id.get(candidate.asset_id)
        source_duration = slot_duration
        if asset and asset.duration_sec and asset.duration_sec > 0:
            source_duration = min(float(asset.duration_sec), slot_duration)
        try:
            apply_operation(
                timeline_state,
                OperationPayload(
                    op_type="add_broll_clip",
                    params={
                        "asset_id": candidate.asset_id,
                        "start_sec": 0.0,
                        "end_sec": round(source_duration, 3),
                        "timeline_start_sec": round(float(slot.start_sec), 3),
                        "opacity": round(float(payload.overlay_opacity), 3),
                    },
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
                },
            ),
        )
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
    slot_duration_sec = max(float(slot.end_sec) - float(slot.start_sec), 0.1)
    expanded_queries = expand_broll_queries(
        chunk_text=chunk_text or concept_text,
        concept_text=concept_text,
        concept_tokens=concept_tokens,
    )

    assets = list(
        session.exec(
            select(MediaAsset)
            .where(MediaAsset.project_id == project_id, MediaAsset.media_type == "video")
            .order_by(MediaAsset.created_at.desc())
        ).all()
    )
    assets_by_id: dict[str, MediaAsset] = {asset.id: asset for asset in assets}

    ranked_candidates: list[tuple[MediaAsset, float, dict[str, object]]] = []
    if payload.include_project_assets and assets:
        transcript_asset_id = transcript.asset_id if transcript is not None else ""
        ranked_candidates = _rank_candidates(
            assets=assets,
            transcript_asset_id=transcript_asset_id,
            concept_tokens=concept_tokens,
            candidates_per_slot=payload.candidates_per_slot,
            slot_duration=slot_duration_sec,
        )

    external_candidates: list[ExternalBrollCandidate] = []
    if payload.include_external_sources:
        external_candidates = search_external_broll_candidates(
            concept_text=concept_text,
            concept_tokens=concept_tokens,
            slot_duration_sec=slot_duration_sec,
            limit=payload.candidates_per_slot,
            query_hints=expanded_queries,
        )

    merged_candidates = _mix_candidates(
        local_candidates=ranked_candidates,
        external_candidates=external_candidates,
        limit=payload.candidates_per_slot,
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
            )
        except Exception:
            pass

    if not merged_candidates:
        raise HTTPException(status_code=400, detail="No B-roll candidates available for reroll")

    existing_candidates = list(
        session.exec(
            select(BrollCandidate)
            .where(BrollCandidate.project_id == project_id, BrollCandidate.slot_id == slot.id)
            .order_by(BrollCandidate.created_at.asc())
        ).all()
    )
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
            reason_json=_json_dumps(reason),
        )
        session.add(row)
        added_candidate_ids.append(row.id)

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
