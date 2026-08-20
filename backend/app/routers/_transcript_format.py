from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException

from ..models import Transcript
from ..schemas import TranscriptRegion, TranscriptResponse, TranscriptWord
from ..transcription_service import (
    TranscriptWordPayload,
    _detect_indic_script_languages,
    _normalize_detected_language,
    sanitize_transcript_words,
)
from ..transliteration_service import contains_indic_script, transliterate_words
from ._transcript_constants import _ARABIC_SCRIPT_RANGES


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
    if item.display_text:
        payload["display_text"] = item.display_text
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
    display_text = str(item.get("display_text") or "").strip() or None
    return TranscriptWordPayload(
        id=str(item.get("id") or uuid4()),
        text=text,
        display_text=display_text,
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
            display_text=item.display_text,
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


def _transliteration_display(
    words: list[TranscriptWord],
) -> tuple[list[TranscriptWord], bool]:
    """Fill in `display_text` for words that don't have it cached yet.

    Returns the enriched words plus a flag saying whether anything new was
    computed, so callers can persist the result and skip the (expensive,
    per-word neural) transliteration on every subsequent read.
    """
    if not words:
        return words, False
    sample_text = " ".join(word.text for word in words[:40])
    if not contains_indic_script(sample_text):
        return words, False

    pending_positions = [
        position for position, word in enumerate(words) if not word.display_text
    ]
    if not pending_positions:
        return words, False

    transliterated_words = transliterate_words(
        [words[position].model_dump(mode="json") for position in pending_positions]
    )
    if not transliterated_words or len(transliterated_words) != len(pending_positions):
        return words, False

    enriched = list(words)
    changed = False
    for position, transliterated in zip(pending_positions, transliterated_words):
        word = enriched[position]
        display_text = str(transliterated.get("text") or "").strip()
        if not display_text or display_text == word.text:
            continue
        enriched[position] = word.model_copy(update={"display_text": display_text})
        changed = True
    return enriched, changed


def _with_transliteration_display(words: list[TranscriptWord]) -> list[TranscriptWord]:
    return _transliteration_display(words)[0]


def _local_word_rate(
    ordered: list[TranscriptWordPayload], index: int, window: int = 5
) -> float | None:
    """Fastest words-per-second among the windows containing ``index``.

    Measured over a window rather than a single word because one crushed
    timestamp is a glitch, while a sustained burst is a hallucination. All
    windows containing the word are considered, not just a centred one: a
    centred window at the edge of a burst straddles the silence before it and
    averages the burst away.
    """
    if len(ordered) < window:
        return None
    first = max(0, index - window + 1)
    last = min(index, len(ordered) - window)
    fastest = 0.0
    for start in range(first, last + 1):
        span = float(ordered[start + window - 1].end_sec) - float(
            ordered[start].start_sec
        )
        if span <= 0:
            return float("inf")
        fastest = max(fastest, window / span)
    return fastest or None


def _annotate_word_quality(
    words: list[TranscriptWordPayload], duration_sec: float
) -> list[TranscriptWordPayload]:
    if not words:
        return []

    trusted_min_score = _env_float("TRANSCRIPT_TRUSTED_MIN_SCORE", 0.72, 0.0)
    impossible_rate = _env_float("TRANSCRIBE_MAX_WORDS_PER_SEC", 9.0, 1.0)
    implausible_rate = _env_float("TRANSCRIPT_IMPLAUSIBLE_WORDS_PER_SEC", 7.0, 1.0)
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
                    display_text=item.display_text,
                    confidence=item.confidence,
                    quality_score=round(max(0.0, min(float(score), 1.0)), 3),
                    quality_label=item.quality_label,
                    source_pass=source_pass,
                )
            )
            continue

        score = 0.88
        confidence = item.confidence

        # Groq never returns per-word confidence, so the ladder below is dead on
        # the cloud path and every word used to land on a flat 0.88 "trusted" —
        # including hallucinated bursts of 50 words/sec. Fall back to a signal
        # that is always available: how fast the words claim to arrive.
        local_rate = _local_word_rate(ordered, index)
        if local_rate is not None:
            if local_rate > impossible_rate:
                score = min(score, 0.25)
            elif local_rate > implausible_rate:
                score = min(score, 0.55)

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
                display_text=item.display_text,
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


def _limit_transcript_words(
    response: TranscriptResponse, word_limit: int | None
) -> TranscriptResponse:
    """Narrow an already-built response to a word page.

    Lets callers that need both the full and the paged view build the
    (expensive) response once instead of running the whole materialize +
    transliterate pipeline twice.
    """
    if word_limit is None:
        return response
    limit = max(1, int(word_limit))
    if len(response.words) <= limit:
        return response
    return response.model_copy(
        update={"words": response.words[:limit], "words_truncated": True}
    )


def fill_display_text_in_items(
    items: list[dict[str, object]],
    duration_sec: float,
    *,
    word_offset: int = 0,
    word_limit: int | None = None,
) -> bool:
    """Populate `display_text` on stored word items, in place.

    Returns True when something new was computed. Shared by the write paths
    (so a transcript is romanized once, in the background job that generated
    it) and the read paths (so older transcripts heal on first access).
    """
    _stored_items, words, _text, _regions = _materialize_transcript_items(
        items, duration_sec
    )
    total_words = len(words)
    safe_offset = max(0, min(int(word_offset), total_words))
    if word_limit is None:
        target_words = words[safe_offset:]
    else:
        target_words = words[safe_offset : safe_offset + max(1, int(word_limit))]

    enriched, changed = _transliteration_display(target_words)
    if not changed:
        return False

    display_by_id = {
        word.id: word.display_text for word in enriched if word.display_text
    }
    updated = False
    for item in items:
        display_text = display_by_id.get(str(item.get("id") or ""))
        if display_text and item.get("display_text") != display_text:
            item["display_text"] = display_text
            updated = True
    return updated


def ensure_transliteration_persisted(
    session: object,
    row: Transcript,
    *,
    word_offset: int = 0,
    word_limit: int | None = None,
) -> None:
    """Compute romanization once and cache it in `words_json`.

    Transliteration is by far the most expensive part of a transcript read
    (IndicXlit runs a beam search per word). Without this, every project
    open and every word edit pays the full cost again. We fill in only the
    slice the caller is about to serve, so first-load latency is unchanged
    and every subsequent read short-circuits.
    """
    raw_items = _load_raw_items(row)
    if not fill_display_text_in_items(
        raw_items,
        float(row.duration_sec or 0.0),
        word_offset=word_offset,
        word_limit=word_limit,
    ):
        return

    row.words_json = _json_dumps(raw_items)
    try:
        session.add(row)  # type: ignore[attr-defined]
        session.commit()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        # Caching is best-effort: a failed write must never break the read.
        session.rollback()  # type: ignore[attr-defined]


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
