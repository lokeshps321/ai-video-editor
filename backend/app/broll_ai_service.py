from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlparse

from .config import get_settings
from .models import MediaAsset

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")

CandidateRow = tuple[str, str | None, str | None, str | None, float, dict[str, object]]


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


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


@lru_cache(maxsize=1)
def _load_spacy_nlp() -> object | None:
    try:
        import spacy  # type: ignore
    except Exception:
        return None
    for model_name in ("en_core_web_sm", "xx_ent_wiki_sm"):
        try:
            return spacy.load(model_name, disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
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
            if ent.label_ not in {"PERSON", "ORG", "GPE", "LOC", "EVENT", "PRODUCT", "WORK_OF_ART"}:
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


def expand_broll_queries(
    *,
    chunk_text: str,
    concept_text: str,
    concept_tokens: list[str],
    max_queries: int = 6,
) -> list[str]:
    entities = extract_entities(chunk_text)
    queries: list[str] = []

    if concept_text.strip():
        queries.append(concept_text.strip())
    if concept_tokens:
        queries.append(" ".join(concept_tokens[:3]))
    for entity in entities[:3]:
        queries.append(entity)
        if concept_tokens:
            queries.append(f"{entity} {' '.join(concept_tokens[:2])}".strip())

    chunk_tokens = _tokenize(chunk_text)
    if chunk_tokens:
        queries.append(" ".join(chunk_tokens[:4]))

    deduped: list[str] = []
    seen: set[str] = set()
    for item in queries:
        trimmed = item.strip()
        if not trimmed:
            continue
        key = trimmed.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(trimmed)
        if len(deduped) >= max(1, max_queries):
            break
    return deduped


@lru_cache(maxsize=4)
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
        matrix = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
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
    tokens: set[str]
    duration_sec: float | None
    base_score: float
    reason: dict[str, object]
    entities: list[str]


def _candidate_text(
    *,
    source_type: str,
    source_url: str | None,
    source_label: str | None,
    reason: dict[str, object],
    asset: MediaAsset | None,
) -> tuple[str, float | None]:
    metadata = _parse_asset_metadata(asset)
    parsed_url = urlparse(source_url or "")
    values = [
        source_label or "",
        source_type,
        reason.get("query", ""),
        reason.get("page_url", ""),
        reason.get("tags", []),
        reason.get("keyword_hits", []),
        parsed_url.netloc,
        parsed_url.path.replace("/", " "),
        asset.filename if asset else "",
        _json_as_text(metadata),
    ]
    duration = _safe_float(reason.get("duration_sec"))
    if duration is None and asset is not None:
        duration = _safe_float(asset.duration_sec)
    text = " ".join(str(item).strip() for item in values if str(item).strip())
    return text, duration


def _with_ai_metadata(
    row: CandidateRow,
    *,
    score: float,
    confidence: float,
    score_breakdown: dict[str, float],
    entity_hits: list[str],
    ai_status: str,
) -> CandidateRow:
    source_type, asset_id, source_url, source_label, _old_score, reason = row
    payload = _copy_reason(reason)
    payload["ai_status"] = ai_status
    payload["confidence"] = round(_clamp(confidence, 0.0, 1.0), 3)
    payload["score_breakdown"] = {key: round(_clamp(value, 0.0, 1.0), 3) for key, value in score_breakdown.items()}
    payload["entities"] = entity_hits[:8]
    return (source_type, asset_id, source_url, source_label, round(_clamp(score, 0.0, 0.99), 3), payload)


def _fallback_rows(candidates: list[CandidateRow], ai_status: str) -> list[CandidateRow]:
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
) -> list[CandidateRow]:
    if not candidates:
        return []

    settings = get_settings()
    if not settings.broll_ai_enabled:
        return _fallback_rows(candidates, "disabled")

    slot_entities = extract_entities(chunk_text)
    slot_entities_lower = [item.lower() for item in slot_entities]
    slot_token_set = set(_tokenize(f"{chunk_text} {concept_text} {' '.join(concept_tokens)} {' '.join(slot_entities)}"))

    docs: list[_CandidateDoc] = []
    for row in candidates:
        source_type, asset_id, source_url, source_label, base_score, reason = row
        parsed_reason = _copy_reason(reason)
        asset = assets_by_id.get(asset_id or "")
        text, duration_sec = _candidate_text(
            source_type=source_type,
            source_url=source_url,
            source_label=source_label,
            reason=parsed_reason,
            asset=asset,
        )
        if settings.broll_blocklist_terms and _contains_blocked_term(text, settings.broll_blocklist_terms):
            continue
        docs.append(
            _CandidateDoc(
                row=row,
                text=text,
                tokens=set(_tokenize(text)),
                duration_sec=duration_sec,
                base_score=_clamp(float(base_score), 0.0, 0.99),
                reason=parsed_reason,
                entities=extract_entities(text),
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
    embedding_payload = [slot_embed_text] + [doc.text for doc in docs]
    vectors = _encode_embeddings(embedding_payload)
    has_embeddings = bool(vectors and len(vectors) == len(embedding_payload))
    slot_vector = vectors[0] if has_embeddings and vectors else []
    candidate_vectors = vectors[1:] if has_embeddings and vectors else []

    concept_token_set = {token.lower() for token in concept_tokens}
    ranked: list[CandidateRow] = []
    for idx, doc in enumerate(docs):
        if has_embeddings:
            semantic_score = _cosine_from_normalized(slot_vector, candidate_vectors[idx])
        else:
            semantic_score = _token_overlap_score(slot_token_set, doc.tokens)

        entity_hits = [entity for entity in doc.entities if entity.lower() in slot_entities_lower]
        if not entity_hits:
            entity_hits = [entity for entity in slot_entities if entity.lower() in doc.text.lower()]
        entity_score = len({item.lower() for item in entity_hits}) / max(len(slot_entities), 1) if slot_entities else 0.0

        concept_overlap = _token_overlap_score(concept_token_set, doc.tokens) if concept_token_set else 0.0
        keyword_hits = doc.reason.get("keyword_hits", [])
        keyword_hit_score = (
            min(len(keyword_hits), max(len(concept_tokens), 1)) / max(len(concept_tokens), 1)
            if isinstance(keyword_hits, list) and concept_tokens
            else 0.0
        )
        metadata_score = _clamp((0.5 * concept_overlap) + (0.2 * keyword_hit_score) + (0.3 * doc.base_score), 0.0, 1.0)
        duration_score = _duration_fit(doc.duration_sec, slot_duration_sec)

        weighted = (
            (weights[0] * semantic_score)
            + (weights[1] * entity_score)
            + (weights[2] * metadata_score)
            + (weights[3] * duration_score)
        )
        final_score = _clamp((0.75 * weighted) + (0.25 * doc.base_score), 0.0, 0.99)
        confidence = _clamp(
            (0.45 * final_score) + (0.25 * semantic_score) + (0.20 * entity_score) + (0.10 * metadata_score),
            0.0,
            1.0,
        )
        if not has_embeddings:
            confidence = _clamp(confidence * 0.92, 0.0, 1.0)
        ai_status = "reranked" if has_embeddings else "fallback_no_embeddings"

        ranked.append(
            _with_ai_metadata(
                doc.row,
                score=final_score,
                confidence=confidence,
                score_breakdown={
                    "semantic": semantic_score,
                    "entity": entity_score,
                    "metadata": metadata_score,
                    "duration": duration_score,
                    "legacy": doc.base_score,
                },
                entity_hits=entity_hits,
                ai_status=ai_status,
            )
        )

    ranked.sort(key=lambda item: item[4], reverse=True)
    return ranked
