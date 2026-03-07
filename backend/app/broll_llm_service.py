from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from .broll_ai_service import extract_entities
from .models import MediaAsset

CandidateRow = tuple[str, str | None, str | None, str | None, float, dict[str, object]]

_ALLOWED_WEAK_CODES = {"semantic_weak", "crop_weak", "talking_head_risk", "confidence_low", "generated_fallback"}
_ALLOWED_VISUAL_INTENTS = {"literal_demo", "process_step", "environment_context", "reaction_payoff", "abstract_support"}
_ALLOWED_QUERY_MODES = {"literal", "process", "environment", "reaction", "abstract"}
_LOW_SIGNAL_PHRASES = {
    "general scene",
    "support visual",
    "abstract scene",
    "visual background",
    "background scene",
    "wide shot",
    "medium shot",
    "detail shot",
}
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_FOCUS_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "did", "do", "for", "from",
    "get", "go", "going", "got", "had", "has", "have", "here", "how", "i", "if", "im", "i'm", "in",
    "into", "is", "it", "its", "just", "like", "me", "my", "of", "on", "or", "our", "out", "really",
    "so", "than", "that", "the", "their", "them", "there", "they", "this", "those", "to", "up", "very",
    "was", "we", "were", "what", "when", "where", "who", "why", "with", "yeah", "you", "your",
}
_DOMAIN_KEYWORDS = {
    "motorsport": {
        "apex", "car", "cars", "cockpit", "corner", "corners", "driver", "drivers", "f1", "finish", "garage",
        "gp", "grid", "helmet", "lap", "laps", "monza", "motorsport", "overtake", "pit", "pits", "podium",
        "qualifying", "race", "races", "racing", "telemetry", "track", "teammate", "wheel",
    },
    "technology": {
        "app", "apps", "camera", "code", "dashboard", "device", "devices", "interface", "laptop", "mobile",
        "phone", "product", "saas", "screen", "screens", "software", "startup", "tech", "workflow",
    },
    "business": {
        "analytics", "brand", "business", "campaign", "customer", "customers", "finance", "growth", "meeting",
        "office", "revenue", "sales", "startup", "strategy", "team", "teams",
    },
    "fitness": {
        "athlete", "coach", "exercise", "fitness", "gym", "run", "runner", "running", "sport", "sports",
        "training", "workout",
    },
    "music": {
        "album", "beat", "concert", "crowd", "feat", "hiphop", "lyrics", "music", "official", "performance",
        "rap", "rapper", "song", "songs", "stage", "studio", "vocal",
    },
}
_DOMAIN_SUMMARIES = {
    "motorsport": "motorsport race coverage with drivers, pit crew, telemetry, garage tension, and track action",
    "technology": "technology product explainer with devices, software, screens, workflows, and team collaboration",
    "business": "business explainer with offices, meetings, dashboards, customers, and growth moments",
    "fitness": "fitness and sports training with athletes, movement, coaching, and performance visuals",
    "music": "music and performance video with studio sessions, stage moments, and audience reaction",
    "general": "talking-head explainer with supportive environment, process, reaction, and context visuals",
}


def _api_key() -> str:
    return (os.getenv("GROQ_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or "").strip()


def _base_url() -> str:
    if (os.getenv("GROQ_API_KEY", "") or "").strip():
        return (os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1") or "https://api.groq.com/openai/v1").rstrip("/")
    return (os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1").rstrip("/")


def _model() -> str:
    default_model = "openai/gpt-oss-120b" if (os.getenv("GROQ_API_KEY", "") or "").strip() else "gpt-4.1-mini"
    return (os.getenv("BROLL_RETRIEVAL_MODEL", default_model) or default_model).strip()


def _timeout_sec() -> float:
    return max(5.0, float(os.getenv("BROLL_RETRIEVAL_TIMEOUT_SEC", "20") or "20"))


def _llm_enabled() -> bool:
    raw = (os.getenv("BROLL_LLM_ENABLED", "true") or "true").strip().lower()
    return raw in {"1", "true", "yes", "on"} and bool(_api_key())


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


def _focus_terms(text: str, *, min_len: int = 3) -> list[str]:
    tokens = _tokenize(text)
    return [
        token
        for token in tokens
        if (len(token) >= min_len or token in {"f1", "gp"})
        and token not in _FOCUS_STOP_WORDS
    ]


def _normalize_visual_intent(raw: object, fallback: str) -> str:
    value = str(raw or "").strip().lower() or fallback
    return value if value in _ALLOWED_VISUAL_INTENTS else fallback


def _normalize_domain_label(raw: object, fallback: str = "general") -> str:
    value = str(raw or "").strip().lower()
    if not value:
        return fallback
    if any(term in value for term in ("race", "f1", "motor", "pit", "driver", "track")):
        return "motorsport"
    if any(term in value for term in ("tech", "software", "product", "app", "saas")):
        return "technology"
    if any(term in value for term in ("business", "finance", "sales", "marketing", "office")):
        return "business"
    if any(term in value for term in ("fitness", "sport", "training", "workout", "gym")):
        return "fitness"
    if any(term in value for term in ("music", "song", "concert", "stage", "studio")):
        return "music"
    return fallback if value == "unknown" else value


def _dedupe_strings(items: list[str], *, limit: int) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = " ".join(str(item).strip().split())
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
        if len(deduped) >= max(1, limit):
            break
    return deduped


def _normalize_query_packets(raw_queries: object, *, max_queries: int) -> list[dict[str, str]]:
    if not isinstance(raw_queries, list):
        return []
    packets: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_queries:
        if not isinstance(item, dict):
            continue
        query = " ".join(str(item.get("query") or "").split()).strip()
        mode = str(item.get("mode") or "literal").strip().lower() or "literal"
        if not query:
            continue
        if _is_low_signal_phrase(query):
            continue
        if mode not in _ALLOWED_QUERY_MODES:
            mode = "literal"
        key = (query.lower(), mode)
        if key in seen:
            continue
        seen.add(key)
        packets.append({"query": query, "mode": mode})
        if len(packets) >= max(1, max_queries):
            break
    return packets


def _is_low_signal_phrase(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return True
    if normalized in _LOW_SIGNAL_PHRASES:
        return True
    focus = _focus_terms(normalized, min_len=2)
    if len(focus) < 2 and normalized not in {"prayer scene at night", "reflective journey"}:
        return True
    return False


def _strip_blocked_terms(text: str, blocked_terms: list[str]) -> str:
    cleaned = text
    for raw_term in blocked_terms:
        term = " ".join(str(raw_term).strip().split())
        if not term:
            continue
        cleaned = re.sub(rf"\b{re.escape(term)}\b", " ", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def _fallback_domain_context(*, transcript_text: str, asset_filenames: list[str]) -> dict[str, Any]:
    haystack = " ".join([transcript_text[:6000], *asset_filenames[:8]])
    tokens = _tokenize(haystack)
    scores: dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = 0
        for token in tokens:
            if token in keywords:
                score += 2 if token in {"f1", "monza", "gp", "pit", "pitlane", "telemetry", "race", "racing"} else 1
        scores[domain] = score
    domain = max(scores, key=scores.get) if scores else "general"
    if scores.get(domain, 0) < 2:
        domain = "general"
    anchors = _dedupe_strings(
        [token for token in tokens if token in _DOMAIN_KEYWORDS.get(domain, set())],
        limit=6,
    )
    return {
        "domain": domain,
        "summary": _DOMAIN_SUMMARIES.get(domain, _DOMAIN_SUMMARIES["general"]),
        "anchors": anchors,
    }


def infer_broll_domain_context(*, transcript_text: str, asset_filenames: list[str]) -> dict[str, Any]:
    fallback = _fallback_domain_context(
        transcript_text=transcript_text,
        asset_filenames=asset_filenames,
    )
    if not _llm_enabled():
        return fallback
    prompt = {
        "goal": "Infer the source-video domain so stock B-roll queries stay in-context.",
        "transcript_excerpt": transcript_text[:5000],
        "asset_filenames": asset_filenames[:8],
        "fallback_context": fallback,
        "rules": [
            "Choose a generic domain label like motorsport, technology, business, fitness, music, or general.",
            "Avoid specific people, brands, teams, or copyrighted franchises in the summary.",
            "If the transcript has named entities that stock libraries will not have, convert them to generic roles or environments.",
        ],
        "output_schema": {
            "domain": "string",
            "summary": "string",
            "anchors": ["string"],
        },
    }
    parsed = _chat_json(prompt)
    if not parsed:
        return fallback
    domain = _normalize_domain_label(parsed.get("domain"), fallback=str(fallback["domain"]))
    anchors = _dedupe_strings(
        [str(item) for item in parsed.get("anchors", [])] if isinstance(parsed.get("anchors"), list) else [],
        limit=6,
    )
    summary = " ".join(str(parsed.get("summary") or "").split()).strip() or str(fallback["summary"])
    return {
        "domain": domain,
        "summary": summary,
        "anchors": anchors or list(fallback["anchors"]),
    }


def _chat_json(prompt: dict[str, Any]) -> dict[str, Any] | None:
    if not _llm_enabled():
        return None
    payload = {
        "model": _model(),
        "messages": [
            {"role": "system", "content": "You are a B-roll retrieval assistant. Return only valid JSON."},
            {"role": "user", "content": json.dumps(prompt, separators=(",", ":"))},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }
    try:
        with httpx.Client(
            timeout=httpx.Timeout(_timeout_sec()),
            headers={"Authorization": f"Bearer {_api_key()}"},
        ) as client:
            response = client.post(f"{_base_url()}/chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _fallback_motorsport_packets(
    *,
    token_set: set[str],
    visual_intent: str,
    max_queries: int,
) -> tuple[str, str, str, list[dict[str, str]], str]:
    packs: list[tuple[set[str], str, str, list[tuple[str, str]], str]] = [
        (
            {"plan", "combat", "battle", "attack", "penalized", "strategy", "cellar", "monza"},
            "race strategy and team pressure",
            "process_step",
            [
                ("race strategy meeting", "process"),
                ("pit wall strategy", "process"),
                ("race engineer monitors", "process"),
                ("team garage tension", "environment"),
                ("telemetry screen race", "process"),
            ],
            "Converted aggressive race commentary into motorsport strategy visuals.",
        ),
        (
            {"upgraded", "apex", "strength", "tenths", "gain", "corner", "corners", "telemetry"},
            "motorsport telemetry and performance",
            "process_step",
            [
                ("race telemetry screen", "process"),
                ("cockpit controls close up", "literal"),
                ("race engineer monitor", "process"),
                ("motorsport garage tech", "environment"),
                ("pit wall data screen", "process"),
            ],
            "Mapped specific team or car references to stock-friendly telemetry visuals.",
        ),
        (
            {"inside", "line", "pass", "passing", "past", "midfield", "threads", "overtake", "lap"},
            "motorsport overtake and track action",
            "literal_demo",
            [
                ("race car overtaking", "literal"),
                ("onboard race cockpit", "literal"),
                ("motorsport track action", "environment"),
                ("driver hands steering", "literal"),
                ("trackside racing duel", "reaction"),
            ],
            "Turned race-action speech into overtaking and cockpit visuals.",
        ),
        (
            {"saboteur", "forget", "ahead", "tension", "snap", "clear", "close"},
            "team tension in the garage",
            "environment_context",
            [
                ("tense team meeting", "environment"),
                ("garage confrontation", "reaction"),
                ("team members arguing", "reaction"),
                ("suspicious glance workshop", "reaction"),
                ("motorsport garage tension", "environment"),
            ],
            "Specific names were replaced with generic team-conflict visuals.",
        ),
        (
            {"finish", "win", "best", "brilliant", "fantastic", "valiant", "sixth", "love", "payoff"},
            "motorsport celebration and finish",
            "reaction_payoff",
            [
                ("team celebration race", "reaction"),
                ("crowd cheering motorsport", "reaction"),
                ("finish line race", "literal"),
                ("pit wall cheering", "reaction"),
                ("podium celebration", "reaction"),
            ],
            "Converted payoff language into finish-line and celebration visuals.",
        ),
    ]
    for cues, concept, intent, raw_queries, rationale in packs:
        if token_set.intersection(cues):
            return (
                concept,
                intent,
                "medium",
                _normalize_query_packets(
                    [{"query": query, "mode": mode} for query, mode in raw_queries],
                    max_queries=max_queries,
                ),
                rationale,
            )
    default_queries = _normalize_query_packets(
        [
            {"query": "pit crew working", "mode": "process"},
            {"query": "race engineer monitors", "mode": "process"},
            {"query": "racing car cockpit", "mode": "literal"},
            {"query": "trackside crowd race", "mode": "reaction"},
            {"query": "motorsport garage", "mode": "environment"},
        ],
        max_queries=max_queries,
    )
    default_intent = visual_intent if visual_intent in _ALLOWED_VISUAL_INTENTS else "environment_context"
    return (
        "motorsport support visual",
        default_intent,
        "medium",
        default_queries,
        "Used generic motorsport support visuals when exact footage is unlikely to exist in stock.",
    )


def _fallback_music_packets(
    *,
    token_set: set[str],
    visual_intent: str,
    max_queries: int,
) -> tuple[str, str, str, list[dict[str, str]], str]:
    packs: list[tuple[set[str], str, str, list[tuple[str, str]], str]] = [
        (
            {"walk", "valley", "shadow", "death"},
            "reflective night walk",
            "abstract_support",
            [
                ("silhouette walking through dark alley", "abstract"),
                ("lonely figure under streetlights", "environment"),
                ("moody night walk city", "process"),
                ("shadowy urban path at night", "environment"),
                ("cinematic low light street scene", "abstract"),
            ],
            "Mapped the lyric to a moody urban journey instead of a literal placeholder scene.",
        ),
        (
            {"knees", "night", "prayers", "streetlight"},
            "prayer scene at night",
            "abstract_support",
            [
                ("person kneeling under streetlight at night", "literal"),
                ("hands clasped in prayer silhouette", "abstract"),
                ("quiet city street at night", "environment"),
                ("contemplative figure in dark alley", "literal"),
                ("slow motion urban night lights", "abstract"),
            ],
            "Converted the lyric into prayer and streetlight visuals that stock libraries can actually satisfy.",
        ),
        (
            {"gangster", "paradise", "hood", "homies", "pistol", "smoke", "streetlight"},
            "urban rap life",
            "abstract_support",
            [
                ("urban night street scene", "environment"),
                ("rapper performing in city alley", "literal"),
                ("graffiti wall at night", "environment"),
                ("group under streetlights", "reaction"),
                ("moody city skyline night", "abstract"),
            ],
            "Converted the chorus into urban rap atmosphere instead of the unusable phrase itself.",
        ),
        (
            {"money", "power", "minute", "hour", "cash"},
            "money pressure and time",
            "abstract_support",
            [
                ("hands holding cash in low light", "literal"),
                ("close up ticking clock hands", "abstract"),
                ("urban night street with neon lights", "environment"),
                ("time lapse city lights at night", "abstract"),
                ("serious face under streetlamp", "reaction"),
            ],
            "Turned the lyric into time, money, and pressure visuals instead of a vague general scene.",
        ),
        (
            {"blind", "hurt", "life", "luck", "front"},
            "street struggle and reflection",
            "abstract_support",
            [
                ("person alone on city sidewalk at night", "literal"),
                ("serious face in shadow", "reaction"),
                ("empty urban street after dark", "environment"),
                ("slow motion neighborhood lights", "abstract"),
                ("silhouette looking over city at night", "reaction"),
            ],
            "Mapped introspective struggle lyrics to reflective urban visuals.",
        ),
    ]
    for cues, concept, intent, raw_queries, rationale in packs:
        if token_set.intersection(cues):
            return (
                concept,
                intent,
                "medium",
                _normalize_query_packets(
                    [{"query": query, "mode": mode} for query, mode in raw_queries],
                    max_queries=max_queries,
                ),
                rationale,
            )
    default_queries = _normalize_query_packets(
        [
            {"query": "urban rap performance", "mode": "literal"},
            {"query": "urban night street with neon lights", "mode": "environment"},
            {"query": "graffiti covered wall with vibrant colors", "mode": "environment"},
            {"query": "moody city skyline night", "mode": "abstract"},
            {"query": "crowd silhouette under stage lights", "mode": "reaction"},
        ],
        max_queries=max_queries,
    )
    default_intent = visual_intent if visual_intent in _ALLOWED_VISUAL_INTENTS else "abstract_support"
    return (
        "urban rap performance",
        default_intent,
        "medium",
        default_queries,
        "Used music-video fallback visuals instead of the low-signal general-scene placeholder.",
    )


def _fallback_generic_packets(
    *,
    chunk_text: str,
    concept_text: str,
    visual_intent: str,
    query_hints: list[str],
    blocked_terms: list[str],
    domain_context: dict[str, Any],
    max_queries: int,
) -> tuple[str, str, str, list[dict[str, str]], str]:
    cleaned_hints = [
        _strip_blocked_terms(item, blocked_terms)
        for item in [concept_text, *query_hints[:6], chunk_text]
    ]
    compact_queries: list[dict[str, str]] = []
    mode_map = {
        "literal_demo": "literal",
        "process_step": "process",
        "environment_context": "environment",
        "reaction_payoff": "reaction",
        "abstract_support": "abstract",
    }
    for hint in cleaned_hints:
        focus = _focus_terms(hint)
        if not focus:
            continue
        query = " ".join(focus[:4]).strip()
        if query:
            compact_queries.append({"query": query, "mode": mode_map.get(visual_intent, "literal")})
    domain_summary = str(domain_context.get("summary") or _DOMAIN_SUMMARIES["general"]).strip()
    summary_focus = _focus_terms(domain_summary)
    if summary_focus:
        compact_queries.append({"query": " ".join(summary_focus[:4]), "mode": "environment"})
    packets = _normalize_query_packets(compact_queries, max_queries=max_queries)
    if not packets:
        packets = _normalize_query_packets(
            [{"query": "supportive background motion", "mode": "abstract"}],
            max_queries=max_queries,
        )
    cleaned_concept = _strip_blocked_terms(concept_text, blocked_terms)
    concept_focus = _focus_terms(cleaned_concept) or _focus_terms(chunk_text)
    search_concept = " ".join(concept_focus[:4]).strip() or str(domain_context.get("summary") or "support visual")
    stockability = "medium" if concept_focus else "low"
    return (
        search_concept,
        _normalize_visual_intent(visual_intent, "abstract_support"),
        stockability,
        packets,
        "Removed named entities and kept only stock-searchable visual nouns and actions.",
    )


def _fallback_search_strategy(
    *,
    chunk_text: str,
    concept_text: str,
    visual_intent: str,
    query_hints: list[str],
    domain_context: dict[str, Any] | None,
    max_queries: int,
) -> dict[str, Any]:
    resolved_domain = dict(domain_context or {})
    if not resolved_domain:
        resolved_domain = _fallback_domain_context(transcript_text=chunk_text, asset_filenames=[])
    domain = _normalize_domain_label(resolved_domain.get("domain"), fallback="general")
    entity_terms = _dedupe_strings(extract_entities(chunk_text)[:6], limit=6)
    token_set = set(_focus_terms(f"{chunk_text} {concept_text}", min_len=2))
    if domain == "motorsport":
        search_concept, resolved_intent, stockability, queries, rationale = _fallback_motorsport_packets(
            token_set=token_set,
            visual_intent=visual_intent,
            max_queries=max_queries,
        )
    elif domain == "music":
        search_concept, resolved_intent, stockability, queries, rationale = _fallback_music_packets(
            token_set=token_set,
            visual_intent=visual_intent,
            max_queries=max_queries,
        )
    else:
        search_concept, resolved_intent, stockability, queries, rationale = _fallback_generic_packets(
            chunk_text=chunk_text,
            concept_text=concept_text,
            visual_intent=visual_intent,
            query_hints=query_hints,
            blocked_terms=entity_terms,
            domain_context=resolved_domain,
            max_queries=max_queries,
        )
    return {
        "search_concept": search_concept,
        "visual_intent": resolved_intent,
        "stockability": stockability,
        "blocked_terms": entity_terms,
        "queries": queries,
        "rationale": rationale,
    }


def build_broll_search_strategy(
    *,
    chunk_text: str,
    concept_text: str,
    visual_intent: str,
    query_hints: list[str],
    max_queries: int,
    domain_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_domain = dict(domain_context or {})
    if not resolved_domain:
        resolved_domain = _fallback_domain_context(transcript_text=chunk_text, asset_filenames=[])
    fallback = _fallback_search_strategy(
        chunk_text=chunk_text,
        concept_text=concept_text,
        visual_intent=visual_intent,
        query_hints=query_hints,
        domain_context=resolved_domain,
        max_queries=max_queries,
    )
    if not _llm_enabled():
        return fallback
    prompt = {
        "goal": "Convert one noisy transcript beat into strong stock-video retrieval queries.",
        "beat_text": chunk_text[:700],
        "concept_text": concept_text[:240],
        "visual_intent": visual_intent,
        "domain_context": resolved_domain,
        "query_hints": query_hints[:8],
        "fallback_strategy": fallback,
        "rules": [
            "Stock libraries will not have exact movie scenes, copyrighted footage, public figures, fictional teams, or branded IP shots.",
            "Replace specific names with generic roles, actions, props, or environments in the same domain.",
            "Use domain context to disambiguate words. In motorsport, combat or attack means race battle, not military footage.",
            "If the line is lyrical, noisy, or not visually literal, choose a faithful supporting visual rather than a literal but wrong query.",
            "Avoid person-only or brand-only queries.",
            "Return 4 to 8 concise stock-search queries.",
        ],
        "output_schema": {
            "search_concept": "string",
            "visual_intent": "literal_demo|process_step|environment_context|reaction_payoff|abstract_support",
            "stockability": "high|medium|low",
            "blocked_terms": ["string"],
            "queries": [
                {"query": "string", "mode": "literal|process|environment|reaction|abstract"}
            ],
            "rationale": "string",
        },
    }
    parsed = _chat_json(prompt)
    if not parsed:
        return fallback
    queries = _normalize_query_packets(parsed.get("queries"), max_queries=max_queries)
    parsed_search_concept = " ".join(str(parsed.get("search_concept") or "").split()).strip()
    if not queries or _is_low_signal_phrase(parsed_search_concept):
        return fallback
    blocked_terms = _dedupe_strings(
        [str(item) for item in parsed.get("blocked_terms", [])] if isinstance(parsed.get("blocked_terms"), list) else [],
        limit=8,
    )
    fallback_blocked = list(fallback["blocked_terms"])
    for item in fallback_blocked:
        if item not in blocked_terms:
            blocked_terms.append(item)
    return {
        "search_concept": parsed_search_concept or str(fallback["search_concept"]),
        "visual_intent": _normalize_visual_intent(parsed.get("visual_intent"), str(fallback["visual_intent"])),
        "stockability": str(parsed.get("stockability") or fallback["stockability"]).strip().lower() or str(fallback["stockability"]),
        "blocked_terms": blocked_terms[:8],
        "queries": queries,
        "rationale": " ".join(str(parsed.get("rationale") or "").split()).strip() or str(fallback["rationale"]),
    }


def build_broll_search_packets(
    *,
    chunk_text: str,
    concept_text: str,
    visual_intent: str,
    query_hints: list[str],
    max_queries: int,
    domain_context: dict[str, Any] | None = None,
) -> list[dict[str, str]] | None:
    strategy = build_broll_search_strategy(
        chunk_text=chunk_text,
        concept_text=concept_text,
        visual_intent=visual_intent,
        query_hints=query_hints,
        max_queries=max_queries,
        domain_context=domain_context,
    )
    packets = strategy.get("queries")
    return list(packets) if isinstance(packets, list) and packets else None


def _candidate_doc(row: CandidateRow, assets_by_id: dict[str, MediaAsset]) -> dict[str, Any]:
    source_type, asset_id, source_url, source_label, score, reason = row
    asset = assets_by_id.get(asset_id or "")
    metadata_text = ""
    if asset is not None:
        metadata_text = f"{asset.filename} {asset.metadata_json}"
    return {
        "source_type": source_type,
        "label": source_label or asset_id or source_type,
        "query": str(reason.get("query") or ""),
        "query_mode": str(reason.get("query_mode") or ""),
        "tags": reason.get("tags") if isinstance(reason.get("tags"), list) else [],
        "keyword_hits": reason.get("keyword_hits") if isinstance(reason.get("keyword_hits"), list) else [],
        "crop_score": reason.get("crop_score"),
        "width": reason.get("width"),
        "height": reason.get("height"),
        "url_host": urlparse(source_url or "").netloc,
        "legacy_score": score,
        "metadata_text": metadata_text[:400],
    }


def llm_rerank_broll_candidates(
    *,
    chunk_text: str,
    concept_text: str,
    visual_intent: str,
    candidates: list[CandidateRow],
    assets_by_id: dict[str, MediaAsset],
    domain_context: dict[str, Any] | None = None,
) -> list[CandidateRow] | None:
    if not _llm_enabled() or len(candidates) < 2:
        return None
    indexed_docs = [
        {"candidate_index": idx, **_candidate_doc(row, assets_by_id)}
        for idx, row in enumerate(candidates[:12])
    ]
    prompt = {
        "goal": "Pick the best B-roll candidates for a transcript beat.",
        "beat_text": chunk_text[:500],
        "concept_text": concept_text[:200],
        "visual_intent": visual_intent,
        "domain_context": domain_context or {},
        "candidates": indexed_docs,
        "rules": [
            "Prefer visually precise matches to the beat over generic stock.",
            "Penalize talking-head clips unless the beat explicitly wants reaction or payoff.",
            "Penalize clips that will crop badly to portrait.",
            "Respect the domain context. Do not choose military footage for sports conflict language, or literal brand/name matches that are not visually faithful.",
            "Use weak_reason_codes only from: semantic_weak, crop_weak, talking_head_risk, confidence_low, generated_fallback.",
        ],
        "output_schema": {
            "ranking": [
                {
                    "candidate_index": 0,
                    "score": 0.0,
                    "rationale": "string",
                    "weak_reason_codes": ["semantic_weak"],
                }
            ]
        },
    }
    parsed = _chat_json(prompt)
    if not parsed:
        return None
    raw_ranking = parsed.get("ranking")
    if not isinstance(raw_ranking, list):
        return None
    ranking_map: dict[int, dict[str, Any]] = {}
    for item in raw_ranking:
        if not isinstance(item, dict):
            continue
        try:
            candidate_index = int(item.get("candidate_index"))
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if candidate_index < 0 or candidate_index >= len(candidates):
            continue
        weak_codes = item.get("weak_reason_codes")
        normalized_codes: list[str] = []
        if isinstance(weak_codes, list):
            for code in weak_codes:
                text = str(code).strip()
                if text in _ALLOWED_WEAK_CODES and text not in normalized_codes:
                    normalized_codes.append(text)
        ranking_map[candidate_index] = {
            "score": max(0.0, min(score, 0.99)),
            "rationale": str(item.get("rationale") or "").strip(),
            "weak_reason_codes": normalized_codes,
        }
    if not ranking_map:
        return None

    reranked: list[CandidateRow] = []
    for idx, row in enumerate(candidates):
        source_type, asset_id, source_url, source_label, base_score, reason = row
        extra = ranking_map.get(idx)
        if not extra:
            reranked.append(row)
            continue
        merged_reason = dict(reason)
        merged_reason["llm_retrieval_rationale"] = extra["rationale"]
        existing_weak = merged_reason.get("weak_reason_codes")
        weak_codes = list(existing_weak) if isinstance(existing_weak, list) else []
        for code in extra["weak_reason_codes"]:
            if code not in weak_codes:
                weak_codes.append(code)
        merged_reason["weak_reason_codes"] = weak_codes
        merged_reason["llm_retrieval_score"] = round(float(extra["score"]), 3)
        blended_score = max(0.0, min((0.55 * float(base_score)) + (0.45 * float(extra["score"])), 0.99))
        reranked.append((source_type, asset_id, source_url, source_label, round(blended_score, 3), merged_reason))
    reranked.sort(key=lambda item: item[4], reverse=True)
    return reranked
