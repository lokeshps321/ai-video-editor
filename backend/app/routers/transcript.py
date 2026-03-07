from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..config import get_settings
from ..database import get_session
from ..media_utils import probe_duration_seconds
from ..models import MediaAsset, Project, Transcript
from ..schemas import (
    OperationPayload,
    TranscriptCutRequest,
    TranscriptCutResponse,
    TranscriptGenerateRequest,
    TranscriptGenerateResponse,
    TranscriptRangeUpdateRequest,
    TranscriptRegion,
    TranscriptResponse,
    TranscriptWord,
)
from ..storage import storage
from ..timeline_service import apply_operation, get_timeline_row, load_timeline_state, save_timeline_state
from ..transcription_service import (
    TranscriptPayload,
    TranscriptWordPayload,
    generate_transcript,
    infer_source_pass,
    sanitize_transcript_words,
)

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
        raise HTTPException(status_code=500, detail="Stored transcript words are invalid") from exc
    if not isinstance(payload, list):
        raise HTTPException(status_code=500, detail="Stored transcript words are invalid")
    return [item for item in payload if isinstance(item, dict)]


def _parse_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _is_blank_region(item: dict[str, object]) -> bool:
    return bool(item.get("blanked")) or str(item.get("kind") or "").strip().lower() == "blank_region"


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


def _normalize_blank_item(item: dict[str, object], duration_sec: float) -> dict[str, object] | None:
    start_sec = max(0.0, min(_parse_float(item.get("start_sec"), 0.0), duration_sec))
    end_sec = max(0.0, min(_parse_float(item.get("end_sec"), duration_sec), duration_sec))
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
        quality_score = float(quality_score_raw) if quality_score_raw is not None else None
    except (TypeError, ValueError):
        quality_score = None
    quality_label = str(item.get("quality_label") or "").strip().lower() or None
    if quality_label not in {"trusted", "weak"}:
        quality_label = None
    source_pass = str(item.get("source_pass") or "").strip().lower() or None
    if source_pass not in {"primary", "retry", "rescue", "manual"}:
        source_pass = None
    return TranscriptWordPayload(
        id=str(item.get("id") or uuid4()),
        text=text,
        start_sec=start_sec,
        end_sec=end_sec,
        confidence=confidence,
        quality_score=quality_score,
        quality_label=quality_label,
        source_pass=source_pass,
    )


def _word_models_from_payloads(words: list[TranscriptWordPayload]) -> list[TranscriptWord]:
    return [
        TranscriptWord(
            id=item.id,
            text=item.text,
            start_sec=float(item.start_sec),
            end_sec=float(item.end_sec),
            confidence=item.confidence,
            quality_score=item.quality_score,
            quality_label=item.quality_label,
            source_pass=item.source_pass,
        )
        for item in words
    ]


def _annotate_word_quality(words: list[TranscriptWordPayload], duration_sec: float) -> list[TranscriptWordPayload]:
    if not words:
        return []

    trusted_min_score = _env_float("TRANSCRIPT_TRUSTED_MIN_SCORE", 0.72, 0.0)
    weak_gap_sec = _env_float("TRANSCRIPT_WEAK_REGION_GAP_SEC", max(1.4, min(2.8, duration_sec * 0.035 + 0.6)), 0.4)
    ordered = sorted(words, key=lambda item: (float(item.start_sec), float(item.end_sec), item.id))
    annotated: list[TranscriptWordPayload] = []
    for index, item in enumerate(ordered):
        source_pass = item.source_pass if item.source_pass in {"primary", "retry", "rescue", "manual"} else "primary"
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

        prev_gap = float(item.start_sec) if index == 0 else max(0.0, float(item.start_sec) - float(ordered[index - 1].end_sec))
        next_gap = (
            max(0.0, duration_sec - float(item.end_sec))
            if index == len(ordered) - 1
            else max(0.0, float(ordered[index + 1].start_sec) - float(item.end_sec))
        )
        local_gap = max(prev_gap, next_gap)
        surrounded_by_gaps = prev_gap >= (weak_gap_sec * 0.6) and next_gap >= (weak_gap_sec * 0.6)
        if surrounded_by_gaps or ((source_pass in {"retry", "rescue"}) and local_gap >= weak_gap_sec):
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
        for payload in (
            _word_payload_from_item(item)
            for item in items
        )
        if payload is not None
    ]
    normalized_words = sanitize_transcript_words(
        visible_payloads,
        max(float(duration_sec), 0.0),
        apply_filters=False,
    )
    annotated_words = _annotate_word_quality(normalized_words, max(float(duration_sec), 0.0))
    blank_items = [
        normalized
        for normalized in (
            _normalize_blank_item(item, max(float(duration_sec), 0.0))
            for item in items
            if _is_blank_region(item)
        )
        if normalized is not None
    ]
    stored_items = _sort_items([*[_serialize_word(word) for word in annotated_words], *blank_items])
    text = " ".join(word.text for word in annotated_words)
    regions = _build_regions(annotated_words, blank_items, max(float(duration_sec), 0.0))
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
        key=lambda region: (float(region.start_sec), float(region.end_sec), region.status),
    )


def _make_weak_region(words: list[TranscriptWordPayload]) -> TranscriptRegion:
    reason = "uncertain_audio"
    if any(word.source_pass == "rescue" for word in words):
        reason = "rescue_fill"
    elif any(word.source_pass == "retry" for word in words):
        reason = "retry_fill"
    elif any(word.confidence is not None and float(word.confidence) < 0.55 for word in words):
        reason = "low_confidence"
    return TranscriptRegion(
        start_sec=round(float(words[0].start_sec), 3),
        end_sec=round(float(words[-1].end_sec), 3),
        status="weak",
        reason=reason,
        word_ids=[word.id for word in words],
    )


def _load_words(row: Transcript) -> list[TranscriptWord]:
    _stored_items, words, _text, _regions = _materialize_transcript_items(_load_raw_items(row), float(row.duration_sec or 0.0))
    return words


def _to_response(row: Transcript) -> TranscriptResponse:
    _stored_items, words, text, regions = _materialize_transcript_items(_load_raw_items(row), float(row.duration_sec or 0.0))
    return TranscriptResponse(
        id=row.id,
        project_id=row.project_id,
        asset_id=row.asset_id,
        source=row.source,
        language=row.language,
        text=text,
        words=words,
        regions=regions,
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
    stored_items, _words, text, _regions = _materialize_transcript_items(items, duration_sec)
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


def _persist_transcript_items(row: Transcript, *, session: Session, items: list[dict[str, object]]) -> None:
    stored_items, _words, text, _regions = _materialize_transcript_items(items, float(row.duration_sec or 0.0))
    row.words_json = _json_dumps(stored_items)
    row.text = text
    row.updated_at = _utcnow()
    session.add(row)
    session.commit()
    session.refresh(row)


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


def _build_replacement_words(selected_words: list[TranscriptWordPayload], replacement_text: str) -> list[TranscriptWordPayload]:
    tokens = [token for token in replacement_text.strip().split() if token.strip()]
    if not tokens:
        return []

    old_tokens = [word.text for word in selected_words]
    prefix_len = 0
    max_prefix = min(len(old_tokens), len(tokens))
    while prefix_len < max_prefix and old_tokens[prefix_len].strip().lower() == tokens[prefix_len].strip().lower():
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
    kept_suffix = selected_words[len(selected_words) - suffix_len :] if suffix_len else []
    old_middle = selected_words[prefix_len : len(selected_words) - suffix_len if suffix_len else len(selected_words)]
    new_middle_tokens = tokens[prefix_len : len(tokens) - suffix_len if suffix_len else len(tokens)]

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
            source_word = old_middle[index] if index < len(old_middle) else selected_words[min(prefix_len, len(selected_words) - 1)]
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
        raise HTTPException(status_code=404, detail="Selected transcript range was not found")
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
    merged_items = [*_sort_items([_serialize_word(word) for word in merged_words]), *blank_items]
    return merged_items


def _load_word_models_from_items(items: list[dict[str, object]], duration_sec: float) -> list[TranscriptWord]:
    _stored_items, words, _text, _regions = _materialize_transcript_items(items, duration_sec)
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
        select(Transcript).where(Transcript.id == transcript_id, Transcript.project_id == project_id)
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return project, row


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

    requested_language = _normalize_requested_language(payload.language)
    row: Transcript | None = None
    if _env_bool("TRANSCRIBE_REUSE_EXISTING_ON_GENERATE", True):
        existing = session.exec(
            select(Transcript)
            .where(Transcript.project_id == project_id, Transcript.asset_id == asset.id)
            .order_by(Transcript.created_at.desc())
        ).first()
        if existing and not existing.is_mock:
            existing_language = _normalize_requested_language(existing.language)
            if requested_language is None or requested_language == existing_language:
                try:
                    if _load_words(existing):
                        row = existing
                except HTTPException:
                    row = None

    if row is None:
        fast_mode = _env_bool("TRANSCRIBE_FAST_MODE", False)
        try:
            try:
                transcript_payload = generate_transcript(
                    source_path,
                    duration_sec,
                    language_hint=payload.language,
                    fast_mode=fast_mode,
                    prompt=payload.prompt,
                )
            except TypeError as exc:
                if "fast_mode" not in str(exc) and "prompt" not in str(exc):
                    raise
                transcript_payload = generate_transcript(
                    source_path,
                    duration_sec,
                    language_hint=payload.language,
                )
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
    new_text = str(payload.get("text") or "").strip()
    if not new_text:
        raise HTTPException(status_code=400, detail="Word text cannot be empty")

    _project, row = _get_project_and_transcript(session, project_id=project_id, transcript_id=transcript_id)
    updated_items = _apply_range_update_items(
        _load_raw_items(row),
        duration_sec=float(row.duration_sec or 0.0),
        start_word_id=word_id,
        end_word_id=word_id,
        mode="replace",
        text=new_text,
    )
    _persist_transcript_items(row, session=session, items=updated_items)
    return {"ok": True}


@router.patch("/{transcript_id}/range", response_model=TranscriptResponse)
def update_transcript_range(
    transcript_id: str,
    payload: TranscriptRangeUpdateRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> TranscriptResponse:
    _project, row = _get_project_and_transcript(session, project_id=project_id, transcript_id=transcript_id)
    updated_items = _apply_range_update_items(
        _load_raw_items(row),
        duration_sec=float(row.duration_sec or 0.0),
        start_word_id=payload.start_word_id,
        end_word_id=payload.end_word_id,
        mode=payload.mode,
        text=payload.text,
    )
    _persist_transcript_items(row, session=session, items=updated_items)
    return _to_response(row)
