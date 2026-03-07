from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import httpx

from .broll_ai_service import extract_entities

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_END_RE = re.compile(r"[.!?]$")
_NUMBER_RE = re.compile(r"(?:\b\d+(?:\.\d+)?\b|[%$])")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "how", "i", "if",
    "im", "i'm", "in", "into", "is", "it", "its", "just", "me", "my", "of", "on", "or", "our", "so", "that",
    "the", "their", "them", "there", "they", "this", "to", "up", "was", "we", "were", "what", "when", "where",
    "who", "why", "with", "you", "your", "yeah", "okay", "ok", "really", "very", "like",
}
_VISUAL_WORDS = {
    "build", "camera", "crowd", "demo", "design", "device", "factory", "kitchen", "laptop", "meeting",
    "money", "office", "phone", "product", "screen", "shop", "software", "street", "studio", "team",
    "warehouse", "workflow",
}
_ACTION_WORDS = {
    "build", "click", "create", "cut", "deliver", "design", "edit", "film", "launch", "make",
    "move", "pack", "record", "ship", "show", "start", "test", "trim", "type", "upload", "walk",
    "work", "write",
}
_EMPHASIS_WORDS = {
    "amazing", "big", "crazy", "critical", "fast", "huge", "important", "massive", "must", "powerful",
    "problem", "secret", "simple", "smart", "viral", "wow",
}
_RESULT_WORDS = {
    "conversion", "customers", "growth", "improvement", "lift", "metric", "metrics", "profit",
    "proof", "revenue", "result", "results", "retention", "sales", "spike", "success", "win",
}
_CONTRAST_WORDS = {
    "after", "before", "but", "instead", "now", "versus", "vs", "without", "yet",
}
_FILLER_LEAD_WORDS = {"and", "but", "just", "like", "okay", "ok", "really", "so", "well", "yeah"}
_SECTION_ORDER = ("hook", "setup", "body", "payoff", "outro")
_SHOT_SEQUENCE = ("wide", "medium", "detail")
_DETAIL_SHOT_CUES = {
    "app", "camera", "dashboard", "demo", "detail", "device", "editing", "hands", "keyboard",
    "laptop", "metrics", "money", "phone", "screen", "software", "testing", "workflow",
}
_WIDE_SHOT_CUES = {
    "celebration", "city", "crowd", "delivery", "factory", "office", "shop", "stage", "street",
    "studio", "team", "warehouse", "workspace",
}


@dataclass(frozen=True)
class PlannerSegment:
    start_sec: float
    end_sec: float
    text: str
    word_ids: list[str]
    tokens: list[str]
    section_label: str
    score: float
    local_signal: float
    intent_label: str
    concept_text: str
    query_hints: list[str]
    rationale: str
    should_place: bool


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def _focus_tokens(text: str) -> list[str]:
    tokens = [token.lower() for token in _WORD_RE.findall(text)]
    return [token for token in tokens if len(token) >= 3 and token not in _STOP_WORDS]


def _extract_concepts(text: str) -> tuple[str, list[str]]:
    focus = _focus_tokens(text)
    if not focus:
        return ("general scene", ["general"])
    counts = Counter(focus)
    ordered = sorted(set(focus), key=lambda token: (-counts[token], focus.index(token)))
    chosen = ordered[:4]
    return (" ".join(chosen), chosen)


def _section_label(position_ratio: float) -> str:
    if position_ratio < 0.14:
        return "hook"
    if position_ratio < 0.30:
        return "setup"
    if position_ratio < 0.76:
        return "body"
    if position_ratio < 0.92:
        return "payoff"
    return "outro"


def _intent_label(text: str, tokens: list[str]) -> str:
    lower = text.lower()
    if any(token in {"problem", "mistake", "issue", "fail"} for token in tokens):
        return "problem_visual"
    if _NUMBER_RE.search(text) or any(token in _RESULT_WORDS for token in tokens):
        return "payoff_visual"
    if any(token in _ACTION_WORDS for token in tokens) or any(token in {"step", "how", "workflow", "process"} for token in tokens) or "how to" in lower:
        return "process_visual"
    return "supporting_visual"


def _sentence_segments(words: list[dict[str, object]], min_chunk_words: int) -> list[dict[str, object]]:
    if not words:
        return []
    segments: list[dict[str, object]] = []
    current: list[dict[str, object]] = []

    def flush(force: bool = False) -> None:
        nonlocal current
        if not current:
            return
        if len(current) < min_chunk_words and not force:
            current = []
            return
        segments.append(
            {
                "start_sec": float(current[0]["start_sec"]),
                "end_sec": float(current[-1]["end_sec"]),
                "text": " ".join(str(item["text"]).strip() for item in current).strip(),
                "word_ids": [str(item["id"]) for item in current],
            }
        )
        current = []

    for idx, word in enumerate(words):
        current.append(word)
        text = str(word["text"]).strip()
        prev = words[idx - 1] if idx > 0 else None
        if prev is not None:
            gap = float(word["start_sec"]) - float(prev["end_sec"])
            if gap > 1.35 and current:
                flush(force=True)
                current.append(word)
                text = str(word["text"]).strip()
        duration = float(current[-1]["end_sec"]) - float(current[0]["start_sec"])
        if _SENTENCE_END_RE.search(text) or len(current) >= 18 or duration >= 6.5:
            flush(force=True)
    flush(force=True)
    return [segment for segment in segments if segment["text"]]


def _local_asset_signal(text: str, assets: list[dict[str, str]]) -> float:
    focus = set(_focus_tokens(text))
    if not focus or not assets:
        return 0.0
    best = 0.0
    for asset in assets:
        haystack = f"{asset.get('filename', '')} {asset.get('metadata_text', '')}".strip().lower()
        asset_terms = set(_focus_tokens(haystack))
        if not asset_terms:
            continue
        overlap = len(focus.intersection(asset_terms)) / max(len(focus), 1)
        best = max(best, overlap)
    return round(_clamp(best, 0.0, 1.0), 3)


def _score_segment(
    text: str,
    *,
    position_ratio: float,
    local_signal: float,
    duration_sec: float,
) -> tuple[float, str]:
    tokens = _focus_tokens(text)
    token_set = set(tokens)
    lower = text.lower()
    entity_score = min(len(extract_entities(text)) * 0.16, 0.32)
    visual_score = min(sum(1 for token in token_set if token in _VISUAL_WORDS) * 0.08, 0.24)
    action_score = min(sum(1 for token in token_set if token in _ACTION_WORDS) * 0.06, 0.18)
    result_score = min(sum(1 for token in token_set if token in _RESULT_WORDS) * 0.08, 0.24)
    emphasis_score = min(sum(1 for token in token_set if token in _EMPHASIS_WORDS) * 0.08, 0.24)
    numeric_score = 0.1 if _NUMBER_RE.search(text) else 0.0
    contrast_score = 0.08 if any(token in token_set for token in _CONTRAST_WORDS) or ("before" in lower and "after" in lower) else 0.0
    duration_score = 0.14 if 1.1 <= duration_sec <= 4.2 else (0.08 if duration_sec <= 5.4 else 0.04)
    section_boost = 0.18 if position_ratio < 0.14 or position_ratio > 0.76 else 0.08
    score = _clamp(
        0.12
        + entity_score
        + visual_score
        + action_score
        + result_score
        + emphasis_score
        + numeric_score
        + contrast_score
        + duration_score
        + section_boost
        + (local_signal * 0.16),
        0.0,
        1.0,
    )
    if tokens and tokens[0] in _FILLER_LEAD_WORDS and max(entity_score, visual_score, action_score, result_score, numeric_score) < 0.12:
        score *= 0.82
    if len(tokens) < 2 and entity_score < 0.1 and visual_score < 0.08:
        return (round(score * 0.55, 3), "Low-information speech segment")
    if result_score > 0.12 or numeric_score > 0.0:
        return (round(score, 3), "Proof, payoff, or metric callout worth visual reinforcement")
    if action_score > 0.12:
        return (round(score, 3), "Action-heavy workflow moment")
    if emphasis_score > 0.12:
        return (round(score, 3), "High-emphasis statement worth visual support")
    if visual_score > 0.12:
        return (round(score, 3), "Concrete visual nouns detected")
    if entity_score > 0.12:
        return (round(score, 3), "Named entities or specific references detected")
    if contrast_score > 0.0:
        return (round(score, 3), "Contrast or transition callout suggests a supporting cutaway")
    return (round(score, 3), "General supporting visual opportunity")


def _preferred_shot_style(segment: PlannerSegment) -> str:
    token_set = set(segment.tokens)
    if segment.intent_label == "process_visual" or token_set.intersection(_DETAIL_SHOT_CUES):
        return "detail"
    if segment.section_label in {"hook", "payoff", "outro"} or token_set.intersection(_WIDE_SHOT_CUES):
        return "wide"
    return "medium"


def _resolve_shot_style(segment: PlannerSegment, previous_style: str | None) -> str:
    preferred = _preferred_shot_style(segment)
    if preferred == "detail":
        return preferred
    if preferred != previous_style:
        return preferred
    index = _SHOT_SEQUENCE.index(preferred)
    return _SHOT_SEQUENCE[(index + 1) % len(_SHOT_SEQUENCE)]


def _target_beats_by_section(max_slots: int) -> dict[str, int]:
    if max_slots <= 0:
        return {section: 0 for section in _SECTION_ORDER}
    if max_slots < len(_SECTION_ORDER):
        priorities = ("hook", "body", "payoff", "setup", "outro")
        return {
            section: 1 if index < max_slots else 0
            for index, section in enumerate(priorities)
        }
    base = {
        "hook": max(1, round(max_slots * 0.20)),
        "setup": max(1, round(max_slots * 0.18)),
        "body": max(2, round(max_slots * 0.34)),
        "payoff": max(1, round(max_slots * 0.18)),
        "outro": max(1, round(max_slots * 0.10)),
    }
    total = sum(base.values())
    while total > max_slots:
        for section in ("body", "setup", "payoff", "hook", "outro"):
            if base[section] > 1 and total > max_slots:
                base[section] -= 1
                total -= 1
    while total < max_slots:
        base["body"] += 1
        total += 1
    return base


def _build_query_hints(concept_text: str, section_label: str, intent_label: str) -> list[str]:
    hints = [concept_text]
    if section_label == "hook":
        hints.append(f"{concept_text} opening hook")
    if intent_label == "process_visual":
        hints.append(f"{concept_text} workflow")
    if intent_label == "payoff_visual":
        hints.append(f"{concept_text} success result")
    deduped: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        normalized = " ".join(hint.strip().split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped[:4]


def _deterministic_plan(
    *,
    words: list[dict[str, object]],
    transcript_duration_sec: float,
    max_slots: int,
    min_chunk_words: int,
    assets: list[dict[str, str]],
    include_external_sources: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    segments = _sentence_segments(words, min_chunk_words=min_chunk_words)
    if not segments:
        return ([], {"coverage_sections": [], "uncovered_ranges": []})

    annotated: list[PlannerSegment] = []
    total_duration = max(transcript_duration_sec, float(segments[-1]["end_sec"]) if segments else 1.0, 1.0)
    for segment in segments:
        mid = (float(segment["start_sec"]) + float(segment["end_sec"])) * 0.5
        ratio = mid / total_duration
        section = _section_label(ratio)
        text = str(segment["text"])
        duration = max(0.1, float(segment["end_sec"]) - float(segment["start_sec"]))
        local_signal = _local_asset_signal(text, assets)
        score, rationale = _score_segment(text, position_ratio=ratio, local_signal=local_signal, duration_sec=duration)
        concept_text, _concept_tokens = _extract_concepts(text)
        tokens = _focus_tokens(text)
        intent = _intent_label(text, tokens)
        should_place = score >= 0.34 and len(tokens) >= 2
        annotated.append(
            PlannerSegment(
                start_sec=float(segment["start_sec"]),
                end_sec=float(segment["end_sec"]),
                text=text,
                word_ids=list(segment["word_ids"]),
                tokens=tokens,
                section_label=section,
                score=score,
                local_signal=local_signal,
                intent_label=intent,
                concept_text=concept_text,
                query_hints=_build_query_hints(concept_text, section, intent),
                rationale=rationale,
                should_place=should_place,
            )
        )

    budgets = _target_beats_by_section(max_slots)
    picked: list[PlannerSegment] = []
    used_per_section = {section: 0 for section in _SECTION_ORDER}
    min_gap_sec = 0.8
    for segment in sorted(annotated, key=lambda item: (item.score, item.local_signal, -item.start_sec), reverse=True):
        if not segment.should_place:
            continue
        if used_per_section[segment.section_label] >= budgets[segment.section_label]:
            continue
        if any(abs(segment.start_sec - item.start_sec) < min_gap_sec or (segment.start_sec < item.end_sec and segment.end_sec > item.start_sec) for item in picked):
            continue
        used_per_section[segment.section_label] += 1
        picked.append(segment)
        if len(picked) >= max_slots:
            break

    for section in _SECTION_ORDER:
        if len(picked) >= max_slots:
            break
        if used_per_section[section] > 0:
            continue
        section_candidates = [
            segment
            for segment in annotated
            if segment.section_label == section
            and not any(
                abs(segment.start_sec - item.start_sec) < min_gap_sec
                or (segment.start_sec < item.end_sec and segment.end_sec > item.start_sec)
                for item in picked
            )
        ]
        if not section_candidates:
            continue
        fallback_segment = max(section_candidates, key=lambda item: (item.score, item.local_signal, -item.start_sec))
        used_per_section[section] += 1
        picked.append(fallback_segment)

    late_boundary = total_duration * 0.84
    if picked and max(item.start_sec for item in picked) < late_boundary:
        late_candidates = [
            segment
            for segment in annotated
            if segment.start_sec >= late_boundary
            and not any(
                abs(segment.start_sec - item.start_sec) < min_gap_sec
                or (segment.start_sec < item.end_sec and segment.end_sec > item.start_sec)
                for item in picked
            )
        ]
        if late_candidates:
            late_pick = max(late_candidates, key=lambda item: (item.score, item.local_signal, item.start_sec))
            if len(picked) >= max_slots:
                replaceable = [
                    idx
                    for idx, item in enumerate(picked)
                    if item.section_label not in {"payoff", "outro"}
                ]
                if not replaceable:
                    replaceable = list(range(len(picked)))
                replacement_idx = min(replaceable, key=lambda idx: (picked[idx].score, picked[idx].start_sec))
                picked[replacement_idx] = late_pick
            else:
                picked.append(late_pick)

    picked.sort(key=lambda item: item.start_sec)
    beats: list[dict[str, object]] = []
    previous_shot_style: str | None = None
    for idx, segment in enumerate(picked):
        shot_style = _resolve_shot_style(segment, previous_shot_style)
        previous_shot_style = shot_style
        source_strategy = "local_first"
        if segment.local_signal < 0.18 and include_external_sources:
            source_strategy = "external_fallback"
        confidence = _clamp(segment.score * (0.86 if source_strategy == "external_fallback" else 0.94), 0.0, 1.0)
        beats.append(
            {
                "beat_index": idx,
                "start_sec": round(segment.start_sec, 3),
                "end_sec": round(segment.end_sec, 3),
                "section_label": segment.section_label,
                "intent_label": segment.intent_label,
                "source_strategy": source_strategy,
                "shot_style": shot_style,
                "should_place": True,
                "confidence": round(confidence, 3),
                "rationale": segment.rationale,
                "concept_text": segment.concept_text,
                "segment_text": segment.text,
                "anchor_word_ids": segment.word_ids,
                "query_hints": segment.query_hints,
                "metadata": {
                    "local_signal": segment.local_signal,
                    "score": segment.score,
                    "signal_type": segment.intent_label,
                },
            }
        )

    uncovered_ranges: list[dict[str, float]] = []
    cursor = 0.0
    for beat in beats:
        if float(beat["start_sec"]) - cursor > 12.0:
            uncovered_ranges.append({"start_sec": round(cursor, 3), "end_sec": round(float(beat["start_sec"]), 3)})
        cursor = max(cursor, float(beat["end_sec"]))
    if total_duration - cursor > 12.0:
        uncovered_ranges.append({"start_sec": round(cursor, 3), "end_sec": round(total_duration, 3)})

    coverage_sections: list[dict[str, object]] = []
    boundaries = {
        "hook": (0.0, total_duration * 0.14),
        "setup": (total_duration * 0.14, total_duration * 0.30),
        "body": (total_duration * 0.30, total_duration * 0.76),
        "payoff": (total_duration * 0.76, total_duration * 0.92),
        "outro": (total_duration * 0.92, total_duration),
    }
    for section in _SECTION_ORDER:
        start_sec, end_sec = boundaries[section]
        coverage_sections.append(
            {
                "section_label": section,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "beat_count": sum(1 for beat in beats if beat["section_label"] == section),
                "target_beats": budgets[section],
            }
        )
    return (beats, {"coverage_sections": coverage_sections, "uncovered_ranges": uncovered_ranges})


def _cloud_plan(
    *,
    transcript_excerpt: str,
    beats: list[dict[str, object]],
    max_slots: int,
) -> tuple[list[dict[str, object]] | None, str | None]:
    groq_api_key = (os.getenv("GROQ_API_KEY", "") or "").strip()
    openai_api_key = (os.getenv("OPENAI_API_KEY", "") or "").strip()
    api_key = groq_api_key or openai_api_key
    if not api_key:
        return (None, None)
    default_base_url = "https://api.groq.com/openai/v1" if groq_api_key else "https://api.openai.com/v1"
    default_model = "openai/gpt-oss-120b" if groq_api_key else "gpt-4.1-mini"
    model = (os.getenv("BROLL_PLANNER_MODEL", default_model) or default_model).strip()
    base_url = (os.getenv("OPENAI_BASE_URL", default_base_url) or default_base_url).rstrip("/")
    timeout_sec = max(5.0, float(os.getenv("BROLL_PLANNER_TIMEOUT_SEC", "25") or "25"))

    prompt = {
        "goal": "Improve B-roll placement quality for a talking-head video. Keep the plan balanced across the entire transcript.",
        "max_slots": max_slots,
        "candidate_beats": beats[: max(6, min(len(beats), 18))],
        "transcript_excerpt": transcript_excerpt[:8000],
        "rules": [
            "Do not front-load all B-roll into the opening section.",
            "Keep only beats that deserve visual support.",
            "Prefer local_first unless the beat obviously needs external_fallback.",
            "Return strict JSON with a beats array.",
        ],
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a video editor planning B-roll beats. Return only valid JSON."},
            {"role": "user", "content": json.dumps(prompt, separators=(",", ":"))},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_sec), headers={"Authorization": f"Bearer {api_key}"}) as client:
            response = client.post(f"{base_url}/chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
    except Exception:
        return (None, model)

    try:
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        raw_beats = parsed.get("beats")
        if not isinstance(raw_beats, list):
            return (None, model)
    except Exception:
        return (None, model)
    validated: list[dict[str, object]] = []
    for idx, item in enumerate(raw_beats):
        if not isinstance(item, dict):
            continue
        try:
            start_sec = float(item.get("start_sec"))
            end_sec = float(item.get("end_sec"))
        except (TypeError, ValueError):
            continue
        if end_sec <= start_sec:
            continue
        validated.append(
            {
                "beat_index": idx,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "section_label": str(item.get("section_label") or "body"),
                "intent_label": str(item.get("intent_label") or "supporting_visual"),
                "source_strategy": str(item.get("source_strategy") or "local_first"),
                "shot_style": str(item.get("shot_style") or _SHOT_SEQUENCE[idx % len(_SHOT_SEQUENCE)]),
                "should_place": bool(item.get("should_place", True)),
                "confidence": round(_clamp(float(item.get("confidence", 0.65)), 0.0, 1.0), 3),
                "rationale": str(item.get("rationale") or "Cloud planner refinement"),
                "concept_text": str(item.get("concept_text") or ""),
                "segment_text": str(item.get("segment_text") or ""),
                "anchor_word_ids": item.get("anchor_word_ids") if isinstance(item.get("anchor_word_ids"), list) else [],
                "query_hints": item.get("query_hints") if isinstance(item.get("query_hints"), list) else [],
                "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
            }
        )
    return (validated or None, model)


def plan_broll(
    *,
    words: list[dict[str, object]],
    transcript_text: str,
    transcript_duration_sec: float,
    max_slots: int,
    min_chunk_words: int,
    assets: list[dict[str, str]],
    include_external_sources: bool,
) -> dict[str, object]:
    beats, coverage = _deterministic_plan(
        words=words,
        transcript_duration_sec=transcript_duration_sec,
        max_slots=max_slots,
        min_chunk_words=min_chunk_words,
        assets=assets,
        include_external_sources=include_external_sources,
    )
    fallback_used = True
    planner_model: str | None = None
    cloud_beats, planner_model = _cloud_plan(
        transcript_excerpt=transcript_text,
        beats=beats,
        max_slots=max_slots,
    )
    if cloud_beats:
        beats = cloud_beats[:max_slots]
        fallback_used = False
    return {
        "plan_version": "v2",
        "fallback_used": fallback_used,
        "planner_model": planner_model,
        "beats": beats,
        "coverage": coverage,
    }
