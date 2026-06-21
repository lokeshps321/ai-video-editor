from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlparse

from .config import get_settings
from .models import MediaAsset

_logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
_FOCUS_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "like",
    "of",
    "on",
    "or",
    "our",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "up",
    "use",
    "using",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "you",
    "your",
    "here",
    "now",
    "yeah",
    "ok",
    "okay",
    "got",
    "going",
    "go",
    "back",
    "last",
    "round",
    "team",
    "new",
}
_GENERIC_SOURCE_TERMS = {
    "clip",
    "clips",
    "footage",
    "hd",
    "library",
    "media",
    "pexels",
    "pixabay",
    "project",
    "stock",
    "video",
    "videos",
}
_VISUAL_INTENT_QUERY_MODES: dict[str, dict[str, float]] = {
    "literal_demo": {
        "literal": 1.0,
        "process": 0.76,
        "environment": 0.5,
        "reaction": 0.42,
        "abstract": 0.3,
    },
    "process_step": {
        "process": 1.0,
        "literal": 0.82,
        "environment": 0.56,
        "reaction": 0.36,
        "abstract": 0.28,
    },
    "environment_context": {
        "environment": 1.0,
        "literal": 0.62,
        "reaction": 0.58,
        "process": 0.48,
        "abstract": 0.42,
    },
    "reaction_payoff": {
        "reaction": 1.0,
        "literal": 0.72,
        "environment": 0.54,
        "process": 0.38,
        "abstract": 0.34,
    },
    "abstract_support": {
        "abstract": 1.0,
        "environment": 0.72,
        "reaction": 0.48,
        "literal": 0.34,
        "process": 0.28,
    },
}

CandidateRow = tuple[str, str | None, str | None, str | None, float, dict[str, object]]


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


def _focus_tokens(text: str) -> set[str]:
    return {
        token
        for token in _tokenize(text)
        if len(token) >= 3 and token not in _FOCUS_STOPWORDS
    }


def _safe_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _duration_fit(candidate_duration: float | None, slot_duration: float) -> float:
    if not candidate_duration or candidate_duration <= 0:
        return 0.45
    baseline = max(slot_duration, 0.6)
    delta = abs(candidate_duration - baseline)
    ratio = _clamp(1.0 - (delta / max(baseline * 2.2, 1.0)), 0.0, 1.0)
    return 0.25 + (ratio * 0.75)


def _token_overlap_score(base_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not base_tokens:
        return 0.0
    return len(base_tokens.intersection(candidate_tokens)) / max(len(base_tokens), 1)


def _contains_blocked_term(text: str, blocked_terms: list[str]) -> bool:
    lower = text.lower()
    for raw_term in blocked_terms:
        term = raw_term.strip().lower()
        if not term:
            continue
        if " " in term:
            if term in lower:
                return True
            continue
        if re.search(rf"\b{re.escape(term)}\b", lower):
            return True
    return False


def _normalize_weights(raw_weights: list[float]) -> list[float]:
    non_negative = [max(weight, 0.0) for weight in raw_weights]
    total = sum(non_negative)
    if total <= 0.0:
        return [0.25, 0.25, 0.25, 0.25]
    return [weight / total for weight in non_negative]


def _json_as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_json_as_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_json_as_text(item) for item in value.values())
    return ""


def _copy_reason(reason: object) -> dict[str, object]:
    return dict(reason) if isinstance(reason, dict) else {}


def _weak_reason_codes(
    *,
    semantic_score: float,
    entity_score: float,
    crop_score: float,
    talking_head_risk: float,
    intent_alignment: float,
    specificity_score: float,
    confidence: float,
    source_type: str,
) -> list[str]:
    codes: list[str] = []
    if semantic_score < 0.33 and entity_score <= 0.0:
        codes.append("semantic_weak")
    if crop_score < 0.45:
        codes.append("crop_weak")
    if talking_head_risk >= 0.6:
        codes.append("talking_head_risk")
    if intent_alignment < 0.42:
        codes.append("intent_weak")
    if specificity_score < 0.36:
        codes.append("specificity_low")
    if confidence < 0.68:
        codes.append("confidence_low")
    if source_type in {"generated_video", "generated_image_video"}:
        codes.append("generated_fallback")
    return codes


def _talking_head_risk(text: str) -> float:
    lower = text.lower()
    risk_terms = (
        "interview",
        "podcast",
        "speaker",
        "microphone",
        "conference",
        "talking",
        "presentation",
        "meeting room",
        "webcam",
        "host",
    )
    hits = sum(1 for term in risk_terms if term in lower)
    if hits <= 0:
        return 0.0
    return _clamp(0.25 + (hits * 0.18), 0.0, 1.0)


def _infer_visual_intent(chunk_text: str, concept_text: str) -> str:
    lower = f"{chunk_text} {concept_text}".lower()
    if any(
        term in lower
        for term in (
            "result",
            "results",
            "success",
            "growth",
            "revenue",
            "celebration",
            "win",
        )
    ):
        return "reaction_payoff"
    if any(
        term in lower
        for term in (
            "how to",
            "workflow",
            "process",
            "step",
            "screen",
            "dashboard",
            "tutorial",
        )
    ):
        return "process_step"
    if any(
        term in lower
        for term in (
            "office",
            "studio",
            "street",
            "warehouse",
            "factory",
            "meeting",
            "workspace",
        )
    ):
        return "environment_context"
    if any(term in lower for term in ("hook", "opening", "intro", "outro")):
        return "abstract_support"
    return "literal_demo"


def _intent_alignment_score(
    *,
    visual_intent: str,
    query_mode: str,
    semantic_score: float,
    content_overlap: float,
    talking_head_risk: float,
) -> float:
    mode_scores = _VISUAL_INTENT_QUERY_MODES.get(
        visual_intent, _VISUAL_INTENT_QUERY_MODES["literal_demo"]
    )
    normalized_mode = query_mode.strip().lower()
    base = mode_scores.get(normalized_mode, 0.48 if not normalized_mode else 0.34)
    if normalized_mode in {"literal", "process"} and content_overlap >= 0.16:
        base += 0.12
    if normalized_mode == "reaction" and semantic_score >= 0.34:
        base += 0.08
    if (
        normalized_mode == "abstract"
        and visual_intent == "abstract_support"
        and semantic_score >= 0.32
    ):
        base += 0.1
    if (
        talking_head_risk >= 0.6
        and normalized_mode in {"literal", "process"}
        and visual_intent != "reaction_payoff"
    ):
        base -= 0.12
    return _clamp(base, 0.0, 1.0)


def _specificity_score(
    *,
    slot_token_set: set[str],
    concept_token_set: set[str],
    content_tokens: set[str],
    query_tokens: set[str],
    keyword_hits: list[object],
) -> float:
    if not content_tokens and not query_tokens:
        return 0.0
    descriptive_terms = {
        token for token in content_tokens if token not in _GENERIC_SOURCE_TERMS
    }
    slot_hit_ratio = (
        len(slot_token_set.intersection(content_tokens))
        / max(min(len(slot_token_set), 4), 1)
        if slot_token_set
        else 0.0
    )
    concept_hit_ratio = (
        len(concept_token_set.intersection(content_tokens))
        / max(min(len(concept_token_set), 3), 1)
        if concept_token_set
        else slot_hit_ratio
    )
    query_hit_ratio = (
        len(slot_token_set.intersection(query_tokens))
        / max(min(len(slot_token_set), 4), 1)
        if slot_token_set and query_tokens
        else 0.0
    )
    keyword_ratio = min(len(keyword_hits), 3) / 3 if keyword_hits else 0.0
    descriptor_ratio = len(descriptive_terms) / max(len(content_tokens), 1)
    return _clamp(
        (0.38 * slot_hit_ratio)
        + (0.18 * concept_hit_ratio)
        + (0.22 * query_hit_ratio)
        + (0.14 * keyword_ratio)
        + (0.08 * descriptor_ratio),
        0.0,
        1.0,
    )


def _source_quality_score(
    *,
    source_type: str,
    crop_score: float,
    talking_head_risk: float,
    specificity_score: float,
    keyword_hits: list[object],
) -> float:
    base = {
        "project_asset": 0.72,
        "pexels_video": 0.66,
        "pixabay_video": 0.64,
        "generated_video": 0.54,
        "generated_image_video": 0.5,
    }.get(source_type, 0.6)
    base += crop_score * 0.18
    if keyword_hits:
        base += 0.08
    if talking_head_risk >= 0.6:
        base -= 0.16
    if (
        source_type in {"generated_video", "generated_image_video"}
        and specificity_score < 0.45
    ):
        base -= 0.12
    return _clamp(base, 0.0, 1.0)


@lru_cache(maxsize=1)
def _load_spacy_nlp() -> object | None:
    try:
        import spacy  # type: ignore
    except Exception:
        return None
    for model_name in ("en_core_web_sm", "xx_ent_wiki_sm"):
        try:
            return spacy.load(
                model_name,
                disable=["tagger", "parser", "lemmatizer", "attribute_ruler"],
            )
        except Exception:
            continue
    return None


def _fallback_entities(text: str) -> list[str]:
    candidates: list[str] = []
    for match in _CAP_PHRASE_RE.finditer(text):
        phrase = match.group(1).strip()
        if len(phrase) < 3:
            continue
        candidates.append(phrase)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:8]


def extract_entities(text: str) -> list[str]:
    settings = get_settings()
    if not settings.broll_entity_enabled:
        return []

    nlp = _load_spacy_nlp()
    if nlp is None:
        return _fallback_entities(text)

    entities: list[str] = []
    try:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ not in {
                "PERSON",
                "ORG",
                "GPE",
                "LOC",
                "EVENT",
                "PRODUCT",
                "WORK_OF_ART",
            }:
                continue
            value = ent.text.strip()
            if len(value) >= 2:
                entities.append(value)
    except Exception:
        return _fallback_entities(text)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in entities:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    if deduped:
        return deduped[:10]
    return _fallback_entities(text)


def _gemini_expand_queries(chunk_text: str, concept_text: str) -> list[str]:
    """Call Gemini Flash to produce 3 semantically rich stock-video search queries.

    This is Option B (Semantic Query Expansion) — instead of literal keyword
    matching the AI understands the *context and mood* of the transcript segment
    and suggests visually descriptive queries suitable for Pexels / Pixabay.
    Falls back silently to an empty list on any error so the caller always gets
    at least the classic keyword-based queries.
    """
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return []
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        return []

    prompt = (
        "You are a stock-video search expert. Given a transcript segment and its "
        "visual concept, output EXACTLY 3 short stock-video search queries (2-5 words "
        "each) that would retrieve visually compelling B-roll matching the MOOD and "
        "CONTEXT — not just the literal words.\n\n"
        f"Transcript segment: \"{chunk_text.strip()}\"\n"
        f"Visual concept: \"{concept_text.strip()}\"\n\n"
        "Rules:\n"
        "- Each query must be a concrete, visualisable scene (e.g. 'team celebrating milestone').\n"
        "- No abstract nouns alone (e.g. NOT just 'success').\n"
        "- No speaker, microphone, or talking-head imagery.\n"
        "- Return ONLY a JSON array of 3 strings, nothing else.\n"
        "Example output: [\"engineers reviewing code on screens\", \"rocket launch countdown\", \"data analytics dashboard close-up\"]"
    )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
                max_output_tokens=128,
            ),
        )
        raw = (response.text or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
            if "```" in raw:
                raw = raw[: raw.index("```")]
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()][:3]
    except Exception as exc:  # noqa: BLE001
        _logger.debug("[broll] Gemini query expansion failed: %s", exc)
    return []


def expand_broll_queries(
    *,
    chunk_text: str,
    concept_text: str,
    concept_tokens: list[str],
    max_queries: int = 6,
) -> list[str]:
    entities = extract_entities(chunk_text)

    # --- Gemini semantic expansion (prepended so they rank first) ---
    gemini_queries = _gemini_expand_queries(chunk_text, concept_text)

    queries: list[str] = list(gemini_queries)

    if concept_text.strip():
        queries.append(concept_text.strip())
    if concept_tokens:
        normalized_tokens = [
            token.strip().lower() for token in concept_tokens if token.strip()
        ]
        if normalized_tokens:
            queries.append(" ".join(normalized_tokens[:4]))
            queries.append(" ".join(normalized_tokens[:3]))
            queries.append(" ".join(normalized_tokens[:2]))
    for entity in entities[:3]:
        queries.append(entity)
        if concept_tokens:
            queries.append(f"{entity} {' '.join(concept_tokens[:2])}".strip())

    chunk_focus = list(_focus_tokens(chunk_text))
    if chunk_focus:
        queries.append(" ".join(chunk_focus[:4]))
        if len(chunk_focus) >= 3:
            queries.append(" ".join(chunk_focus[:3]))

    deduped: list[str] = []
    seen: set[str] = set()
    for item in queries:
        trimmed = item.strip()
        if not trimmed:
            continue
        focus_terms = _focus_tokens(trimmed)
        if not focus_terms:
            continue
        key = trimmed.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(trimmed)
        if len(deduped) >= max(1, max_queries + 3):  # allow extra slots for Gemini queries
            break
    return deduped


@lru_cache(maxsize=1)
def _load_embedder(model_name: str, device: str) -> object | None:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception:
        return None


def _encode_embeddings(texts: list[str]) -> list[list[float]] | None:
    if not texts:
        return None
    settings = get_settings()
    embedder = _load_embedder(settings.broll_embed_model, settings.broll_embed_device)
    if embedder is None:
        return None
    try:
        matrix = embedder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
    except Exception:
        return None
    vectors: list[list[float]] = []
    for row in matrix:
        try:
            vectors.append([float(item) for item in row.tolist()])
        except Exception:
            return None
    return vectors


def _cosine_from_normalized(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    for value_a, value_b in zip(vec_a, vec_b, strict=True):
        dot += value_a * value_b
    return _clamp((dot + 1.0) * 0.5, 0.0, 1.0)


def _parse_asset_metadata(asset: MediaAsset | None) -> dict[str, object]:
    if not asset:
        return {}
    try:
        parsed = json.loads(asset.metadata_json or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


@dataclass
class _CandidateDoc:
    row: CandidateRow
    text: str
    content_text: str
    query_text: str
    tokens: set[str]
    content_tokens: set[str]
    query_tokens: set[str]
    duration_sec: float | None
    base_score: float
    reason: dict[str, object]
    entities: list[str]
    embedding_text: str


def _candidate_text(
    *,
    source_type: str,
    source_url: str | None,
    source_label: str | None,
    reason: dict[str, object],
    asset: MediaAsset | None,
) -> tuple[str, str, str, float | None]:
    metadata = _parse_asset_metadata(asset)
    parsed_url = urlparse(source_url or "")
    query_text = str(reason.get("query") or "").strip()
    page_url_text = str(reason.get("page_url") or "").strip()
    tags_text = _json_as_text(reason.get("tags", []))
    keyword_hits_text = _json_as_text(reason.get("keyword_hits", []))
    metadata_text = _json_as_text(metadata)

    # Keep "content_text" focused on what the candidate is (not only what query produced it).
    content_values = [
        source_label or "",
        source_type,
        page_url_text,
        tags_text,
        keyword_hits_text,
        parsed_url.netloc,
        parsed_url.path.replace("/", " "),
        asset.filename if asset else "",
        metadata_text,
    ]
    values = [
        source_label or "",
        source_type,
        query_text,
        page_url_text,
        tags_text,
        keyword_hits_text,
        parsed_url.netloc,
        parsed_url.path.replace("/", " "),
        asset.filename if asset else "",
        metadata_text,
    ]
    duration = _safe_float(reason.get("duration_sec"))
    if duration is None and asset is not None:
        duration = _safe_float(asset.duration_sec)
    text = " ".join(str(item).strip() for item in values if str(item).strip())
    content_text = " ".join(
        str(item).strip() for item in content_values if str(item).strip()
    )
    return text, content_text, query_text, duration


def _with_ai_metadata(
    row: CandidateRow,
    *,
    score: float,
    confidence: float,
    score_breakdown: dict[str, float],
    entity_hits: list[str],
    ai_status: str,
    weak_reason_codes: list[str] | None = None,
) -> CandidateRow:
    source_type, asset_id, source_url, source_label, _old_score, reason = row
    payload = _copy_reason(reason)
    payload["ai_status"] = ai_status
    payload["confidence"] = round(_clamp(confidence, 0.0, 1.0), 3)
    payload["score_breakdown"] = {
        key: round(_clamp(value, 0.0, 1.0), 3) for key, value in score_breakdown.items()
    }
    payload["entities"] = entity_hits[:8]
    payload["weak_reason_codes"] = list(weak_reason_codes or [])
    return (
        source_type,
        asset_id,
        source_url,
        source_label,
        round(_clamp(score, 0.0, 0.99), 3),
        payload,
    )


def _fallback_rows(
    candidates: list[CandidateRow], ai_status: str
) -> list[CandidateRow]:
    prepared: list[CandidateRow] = []
    for row in candidates:
        score = _clamp(float(row[4]), 0.0, 0.99)
        prepared.append(
            _with_ai_metadata(
                row,
                score=score,
                confidence=score,
                score_breakdown={"legacy_score": score},
                entity_hits=[],
                ai_status=ai_status,
                weak_reason_codes=["ai_fallback"],
            )
        )
    return sorted(prepared, key=lambda item: item[4], reverse=True)


def rerank_broll_candidates(
    *,
    chunk_text: str,
    concept_text: str,
    concept_tokens: list[str],
    slot_duration_sec: float,
    candidates: list[CandidateRow],
    assets_by_id: dict[str, MediaAsset],
    visual_intent: str | None = None,
) -> list[CandidateRow]:
    if not candidates:
        return []

    settings = get_settings()
    if not settings.broll_ai_enabled:
        return _fallback_rows(candidates, "disabled")

    slot_entities = extract_entities(chunk_text)
    slot_entities_lower = [item.lower() for item in slot_entities]
    slot_token_set = _focus_tokens(
        f"{chunk_text} {concept_text} {' '.join(concept_tokens)} {' '.join(slot_entities)}"
    )
    if not slot_token_set:
        slot_token_set = set(
            _tokenize(
                f"{chunk_text} {concept_text} {' '.join(concept_tokens)} {' '.join(slot_entities)}"
            )
        )
    resolved_visual_intent = (
        visual_intent or ""
    ).strip().lower() or _infer_visual_intent(chunk_text, concept_text)

    docs: list[_CandidateDoc] = []
    for row in candidates:
        source_type, asset_id, source_url, source_label, base_score, reason = row
        parsed_reason = _copy_reason(reason)
        asset = assets_by_id.get(asset_id or "")
        text, content_text, query_text, duration_sec = _candidate_text(
            source_type=source_type,
            source_url=source_url,
            source_label=source_label,
            reason=parsed_reason,
            asset=asset,
        )
        if settings.broll_blocklist_terms and _contains_blocked_term(
            text, settings.broll_blocklist_terms
        ):
            continue
        docs.append(
            _CandidateDoc(
                row=row,
                text=text,
                content_text=content_text,
                query_text=query_text,
                tokens=set(_tokenize(text)),
                content_tokens=_focus_tokens(content_text),
                query_tokens=_focus_tokens(query_text),
                duration_sec=duration_sec,
                base_score=_clamp(float(base_score), 0.0, 0.99),
                reason=parsed_reason,
                entities=extract_entities(text),
                embedding_text=content_text or text,
            )
        )

    if not docs:
        return []

    weights = _normalize_weights(
        [
            settings.broll_semantic_weight,
            settings.broll_entity_weight,
            settings.broll_metadata_weight,
            settings.broll_duration_weight,
        ]
    )

    slot_embed_text = f"{chunk_text} {concept_text} {' '.join(slot_entities)}".strip()
    embedding_payload = [slot_embed_text] + [doc.embedding_text for doc in docs]
    vectors = _encode_embeddings(embedding_payload)
    has_embeddings = bool(vectors and len(vectors) == len(embedding_payload))
    slot_vector = vectors[0] if has_embeddings and vectors else []
    candidate_vectors = vectors[1:] if has_embeddings and vectors else []

    concept_token_set = {token.lower() for token in concept_tokens}
    ranked: list[CandidateRow] = []
    for idx, doc in enumerate(docs):
        crop_score = _clamp(_safe_float(doc.reason.get("crop_score")) or 0.45, 0.0, 1.0)
        query_mode = str(doc.reason.get("query_mode") or "").strip().lower()
        talking_head_risk = _talking_head_risk(doc.text)
        content_overlap = (
            _token_overlap_score(slot_token_set, doc.content_tokens or doc.tokens)
            if slot_token_set
            else 0.0
        )
        query_overlap = (
            _token_overlap_score(slot_token_set, doc.query_tokens)
            if slot_token_set
            else 0.0
        )
        if has_embeddings:
            semantic_score = _cosine_from_normalized(
                slot_vector, candidate_vectors[idx]
            )
        else:
            semantic_score = max(content_overlap, query_overlap * 0.85)

        entity_hits = [
            entity for entity in doc.entities if entity.lower() in slot_entities_lower
        ]
        if not entity_hits:
            entity_hits = [
                entity for entity in slot_entities if entity.lower() in doc.text.lower()
            ]
        entity_score = (
            len({item.lower() for item in entity_hits}) / max(len(slot_entities), 1)
            if slot_entities
            else 0.0
        )

        concept_overlap = (
            _token_overlap_score(concept_token_set, doc.content_tokens or doc.tokens)
            if concept_token_set
            else 0.0
        )
        keyword_hits = doc.reason.get("keyword_hits", [])
        keyword_hit_score = (
            min(len(keyword_hits), max(len(concept_tokens), 1))
            / max(len(concept_tokens), 1)
            if isinstance(keyword_hits, list) and concept_tokens
            else 0.0
        )
        metadata_score = _clamp(
            (0.5 * concept_overlap)
            + (0.2 * keyword_hit_score)
            + (0.3 * doc.base_score),
            0.0,
            1.0,
        )
        duration_score = _duration_fit(doc.duration_sec, slot_duration_sec)
        mode_alignment = _intent_alignment_score(
            visual_intent=resolved_visual_intent,
            query_mode=query_mode,
            semantic_score=semantic_score,
            content_overlap=content_overlap,
            talking_head_risk=talking_head_risk,
        )
        specificity_score = _specificity_score(
            slot_token_set=slot_token_set,
            concept_token_set=concept_token_set,
            content_tokens=doc.content_tokens or doc.tokens,
            query_tokens=doc.query_tokens,
            keyword_hits=keyword_hits if isinstance(keyword_hits, list) else [],
        )
        source_quality = _source_quality_score(
            source_type=str(doc.row[0]),
            crop_score=crop_score,
            talking_head_risk=talking_head_risk,
            specificity_score=specificity_score,
            keyword_hits=keyword_hits if isinstance(keyword_hits, list) else [],
        )

        weighted = (
            (weights[0] * semantic_score)
            + (weights[1] * entity_score)
            + (weights[2] * metadata_score)
            + (weights[3] * duration_score)
        )
        # Penalize "query echo" results where only the provider query matches transcript tokens.
        query_echo_penalty = 1.0
        if query_overlap >= 0.34 and content_overlap < 0.08:
            query_echo_penalty = 0.72
        low_relevance_penalty = 1.0
        if semantic_score < 0.30 and entity_score <= 0.0 and content_overlap < 0.06:
            low_relevance_penalty = 0.78
        intent_penalty = 1.0
        if mode_alignment < 0.42 and content_overlap < 0.12:
            intent_penalty = 0.82
        specificity_penalty = 1.0
        if specificity_score < 0.35 and content_overlap < 0.12:
            specificity_penalty = 0.82
        talking_head_penalty = 1.0
        if (
            resolved_visual_intent
            in {"process_step", "literal_demo", "environment_context"}
            and talking_head_risk >= 0.6
        ):
            talking_head_penalty = 0.76
        generated_penalty = 1.0
        if (
            str(doc.row[0]) in {"generated_video", "generated_image_video"}
            and semantic_score < 0.46
        ):
            generated_penalty = 0.78

        final_score = _clamp(
            (0.46 * weighted)
            + (0.14 * doc.base_score)
            + (0.10 * crop_score)
            + (0.10 * mode_alignment)
            + (0.08 * specificity_score)
            + (0.12 * source_quality),
            0.0,
            0.99,
        )
        final_score = _clamp(
            final_score
            * query_echo_penalty
            * low_relevance_penalty
            * intent_penalty
            * specificity_penalty
            * talking_head_penalty
            * generated_penalty,
            0.0,
            0.99,
        )
        confidence = _clamp(
            (0.32 * final_score)
            + (0.18 * semantic_score)
            + (0.12 * entity_score)
            + (0.10 * metadata_score)
            + (0.10 * crop_score)
            + (0.10 * mode_alignment)
            + (0.08 * specificity_score),
            0.0,
            1.0,
        )
        if not has_embeddings:
            confidence = _clamp(confidence * 0.92, 0.0, 1.0)
        ai_status = "reranked" if has_embeddings else "fallback_no_embeddings"
        weak_reason_codes = _weak_reason_codes(
            semantic_score=semantic_score,
            entity_score=entity_score,
            crop_score=crop_score,
            talking_head_risk=talking_head_risk,
            intent_alignment=mode_alignment,
            specificity_score=specificity_score,
            confidence=confidence,
            source_type=str(doc.row[0]),
        )

        ranked.append(
            _with_ai_metadata(
                doc.row,
                score=final_score,
                confidence=confidence,
                score_breakdown={
                    "semantic": semantic_score,
                    "entity": entity_score,
                    "content": content_overlap,
                    "query": query_overlap,
                    "metadata": metadata_score,
                    "duration": duration_score,
                    "crop": crop_score,
                    "alignment": mode_alignment,
                    "specificity": specificity_score,
                    "source_quality": source_quality,
                    "legacy": doc.base_score,
                },
                entity_hits=entity_hits,
                ai_status=ai_status,
                weak_reason_codes=weak_reason_codes,
            )
        )

    ranked.sort(key=lambda item: item[4], reverse=True)
    return ranked
