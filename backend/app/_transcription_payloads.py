from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptWordPayload:
    id: str
    text: str
    start_sec: float
    end_sec: float
    confidence: float | None = None
    quality_score: float | None = None
    quality_label: str | None = None
    source_pass: str | None = None
    speaker_id: str | None = None
    speaker_label: str | None = None


@dataclass(frozen=True)
class TranscriptPayload:
    source: str
    language: str | None
    text: str
    words: list[TranscriptWordPayload]
    is_mock: bool


def _normalize_source_pass(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"main", "original"}:
        return "primary"
    if normalized in {"gapfill", "gap_fill", "retry_fill"}:
        return "retry"
    if normalized in {"rescue_gap", "gap_rescue"}:
        return "rescue"
    return normalized


def infer_source_pass(source: str | None) -> str:
    normalized = (source or "").strip().lower()
    if "rescue" in normalized:
        return "rescue"
    if "retry" in normalized or "gapfill" in normalized:
        return "retry"
    if "manual" in normalized:
        return "manual"
    return "primary"


def _copy_word_payload(
    item: TranscriptWordPayload,
    *,
    id: str | None = None,
    text: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    confidence: float | None | object = ...,
    quality_score: float | None | object = ...,
    quality_label: str | None | object = ...,
    source_pass: str | None | object = ...,
) -> TranscriptWordPayload:
    return TranscriptWordPayload(
        id=item.id if id is None else id,
        text=item.text if text is None else text,
        start_sec=float(item.start_sec) if start_sec is None else float(start_sec),
        end_sec=float(item.end_sec) if end_sec is None else float(end_sec),
        confidence=item.confidence if confidence is ... else confidence,
        quality_score=item.quality_score if quality_score is ... else quality_score,
        quality_label=item.quality_label if quality_label is ... else quality_label,
        source_pass=item.source_pass
        if source_pass is ...
        else _normalize_source_pass(source_pass),
    )


def _word_midpoint_sec(word: TranscriptWordPayload) -> float:
    return (float(word.start_sec) + float(word.end_sec)) * 0.5
