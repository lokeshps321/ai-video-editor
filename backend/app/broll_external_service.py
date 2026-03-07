from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

from .config import get_settings
from .broll_llm_service import build_broll_search_packets

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "do", "for", "from", "go", "going", "got",
    "have", "here", "i", "im", "i'm", "in", "into", "is", "it", "its", "just", "last", "me", "my", "new", "now",
    "of", "on", "or", "our", "out", "past", "really", "round", "so", "team", "that", "the", "their", "them",
    "there", "these", "they", "this", "those", "to", "up", "very", "was", "we", "were", "what", "when", "where",
    "who", "why", "win", "with", "yeah", "you", "your",
}


@dataclass(frozen=True)
class ExternalBrollCandidate:
    source_type: str
    source_url: str
    source_label: str
    score: float
    reason: dict[str, Any]


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def _duration_score(candidate_duration_sec: float | None, slot_duration_sec: float) -> float:
    if not candidate_duration_sec or candidate_duration_sec <= 0:
        return 0.08
    baseline = max(slot_duration_sec, 0.8)
    delta = abs(candidate_duration_sec - baseline)
    ratio = _clamp(1.0 - (delta / max(baseline * 2.0, 1.0)), 0.0, 1.0)
    return 0.06 + (ratio * 0.14)


def _orientation_score(width: int | None, height: int | None) -> float:
    if not width or not height or width <= 0 or height <= 0:
        return 0.03
    return 0.09 if height >= width else 0.05


def _crop_score(width: int | None, height: int | None, target_orientation: str) -> float:
    if not width or not height or width <= 0 or height <= 0:
        return 0.42
    if target_orientation != "portrait":
        return 0.7 if width >= height else 0.5
    ratio = height / max(width, 1)
    if ratio >= 1.45:
        return 0.92
    if ratio >= 1.0:
        return 0.78
    if ratio >= 0.75:
        return 0.58
    return 0.38


def _resolution_score(width: int | None, height: int | None) -> float:
    if not width or not height or width <= 0 or height <= 0:
        return 0.03
    pixels = float(width * height)
    return 0.03 + (_clamp(pixels / (1920.0 * 1080.0), 0.0, 1.0) * 0.07)


def _rank_bonus(index: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return _clamp((1.0 - (index / total)) * 0.12, 0.0, 0.12)


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


def _focus_terms(text: str) -> list[str]:
    return [
        token
        for token in _tokenize(text)
        if len(token) >= 3 and token not in _QUERY_STOP_WORDS
    ]


def _query_term_count(text: str) -> int:
    return len(set(_focus_terms(text)))


def _query_relevance(
    *,
    query: str,
    source_label: str,
    page_url: str,
    tags: list[str],
) -> tuple[float, list[str]]:
    query_terms = set(_focus_terms(query))
    if not query_terms:
        return 0.0, []

    parsed = urlparse(page_url)
    url_terms = _focus_terms(unquote(parsed.path.replace("/", " ")))
    candidate_terms = set(_focus_terms(f"{source_label} {' '.join(tags)} {' '.join(url_terms)}"))
    if not candidate_terms:
        return 0.0, []

    hits = sorted(query_terms.intersection(candidate_terms))
    if not hits:
        return 0.0, []

    coverage = len(hits) / max(len(query_terms), 1)
    precision = len(hits) / max(len(candidate_terms), 1)
    score = _clamp((0.23 * coverage) + (0.09 * precision), 0.0, 0.30)
    return score, hits


def _is_low_information_query(query: str) -> bool:
    terms = _focus_terms(query)
    if len(terms) >= 2:
        return False
    if not terms:
        return True
    return terms[0] in _QUERY_STOP_WORDS


def _build_queries(concept_text: str, concept_tokens: list[str]) -> list[str]:
    queries: list[str] = []
    stripped = concept_text.strip()
    if stripped:
        queries.append(stripped)
    if concept_tokens:
        compact = [token.strip().lower() for token in concept_tokens if token.strip()]
        if compact:
            queries.append(" ".join(compact[:4]))
            queries.append(" ".join(compact[:3]))
            queries.append(" ".join(compact[:2]))
            if compact[0] not in _QUERY_STOP_WORDS and len(compact[0]) >= 4:
                queries.append(compact[0])

    deduped: list[str] = []
    seen: set[str] = set()
    for item in queries:
        trimmed = " ".join(item.strip().split())
        key = trimmed.lower()
        if not key or key in seen:
            continue
        if _is_low_information_query(trimmed):
            continue
        seen.add(key)
        deduped.append(trimmed)
        if len(deduped) >= 4:
            break
    if deduped:
        return deduped
    if stripped and not _is_low_information_query(stripped):
        return [stripped]
    if concept_tokens:
        fallback = " ".join(token.strip().lower() for token in concept_tokens[:2] if token.strip())
        if fallback and not _is_low_information_query(fallback):
            return [fallback]
    return []


def _visual_intent_suffixes(visual_intent: str) -> list[tuple[str, str]]:
    mapping = {
        "literal_demo": [("", "literal"), ("close up", "literal"), ("detail", "literal")],
        "process_step": [("workflow", "process"), ("hands", "process"), ("screen", "process")],
        "environment_context": [("workspace", "environment"), ("office", "environment"), ("studio", "environment")],
        "reaction_payoff": [("success", "reaction"), ("celebration", "reaction"), ("team reaction", "reaction")],
        "abstract_support": [("background", "abstract"), ("motion", "abstract"), ("texture", "abstract")],
    }
    return mapping.get(visual_intent, [("", "literal"), ("environment", "environment")])


def _build_query_packets(
    chunk_text: str,
    concept_text: str,
    concept_tokens: list[str],
    query_hints: list[str] | None,
    visual_intent: str,
    max_queries: int,
    domain_context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    target_queries = max(4, min(max_queries, 8))
    llm_chunk_text = chunk_text or (" ".join(query_hints[:4]) if query_hints else concept_text)
    llm_packets = build_broll_search_packets(
        chunk_text=llm_chunk_text,
        concept_text=concept_text,
        visual_intent=visual_intent,
        query_hints=query_hints or [],
        max_queries=target_queries,
        domain_context=domain_context,
    )
    if llm_packets:
        return llm_packets
    base_queries = _build_queries(concept_text, concept_tokens)
    if query_hints:
        seen_base = {item.lower() for item in base_queries}
        for item in query_hints:
            trimmed = " ".join(item.strip().split())
            key = trimmed.lower()
            if not trimmed or key in seen_base or _is_low_information_query(trimmed):
                continue
            seen_base.add(key)
            base_queries.append(trimmed)
    packets: list[dict[str, str]] = []
    seen_packets: set[tuple[str, str]] = set()
    for query in base_queries:
        for suffix, mode in _visual_intent_suffixes(visual_intent):
            expanded = " ".join(part for part in (query, suffix) if part).strip()
            if not expanded or _is_low_information_query(expanded):
                continue
            key = (expanded.lower(), mode)
            if key in seen_packets:
                continue
            seen_packets.add(key)
            packets.append({"query": expanded, "mode": mode})
            if len(packets) >= target_queries:
                return packets
    return packets


def _diversify_candidates(candidates: list[ExternalBrollCandidate], limit: int) -> list[ExternalBrollCandidate]:
    if limit <= 0:
        return []
    by_mode: dict[str, list[ExternalBrollCandidate]] = {}
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        mode = str(candidate.reason.get("query_mode") or "literal").strip() or "literal"
        by_mode.setdefault(mode, []).append(candidate)
    ordered_modes = sorted(by_mode.keys(), key=lambda key: by_mode[key][0].score if by_mode[key] else 0.0, reverse=True)
    diversified: list[ExternalBrollCandidate] = []
    while ordered_modes and len(diversified) < limit:
        next_modes: list[str] = []
        for mode in ordered_modes:
            bucket = by_mode.get(mode) or []
            if not bucket:
                continue
            diversified.append(bucket.pop(0))
            if len(diversified) >= limit:
                break
            if bucket:
                next_modes.append(mode)
        ordered_modes = next_modes
    return diversified[:limit]


def _preferred_stream_score(width: int, height: int, target_orientation: str) -> float:
    if width <= 0 or height <= 0:
        return -1.0
    pixels = float(width * height)
    target_pixels = float(1080 * 1920 if target_orientation == "portrait" else 1920 * 1080)
    min_pixels = float(720 * 1280 if target_orientation == "portrait" else 1280 * 720)
    max_pixels = float(2560 * 1440)
    proximity = 1.0 - min(abs(pixels - target_pixels) / max(target_pixels, 1.0), 1.4)
    orientation_bonus = 0.18 if (target_orientation == "portrait" and height >= width) or (target_orientation != "portrait" and width >= height) else 0.0
    oversize_penalty = 0.42 if pixels > max_pixels else 0.0
    undersize_penalty = 0.25 if pixels < min_pixels else 0.0
    return proximity + orientation_bonus - oversize_penalty - undersize_penalty


def _pick_pexels_file(video_files: list[dict[str, Any]], *, target_orientation: str) -> dict[str, Any] | None:
    if not video_files:
        return None

    def _score(item: dict[str, Any]) -> float:
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        if width <= 0 or height <= 0:
            return 0.0
        score = _preferred_stream_score(width, height, target_orientation)
        file_type_bonus = 0.03 if str(item.get("file_type") or "").lower() == "video/mp4" else 0.0
        return score + file_type_bonus

    candidates = [
        item for item in video_files
        if str(item.get("link", "")).startswith("http")
    ]
    if not candidates:
        return None
    return max(candidates, key=_score)


def _search_pexels(
    *,
    query_packets: list[dict[str, str]],
    slot_duration_sec: float,
    per_query: int,
    timeout_sec: float,
    limit: int,
    api_key: str,
    target_orientation: str,
    allow_landscape: bool,
    visual_intent: str,
) -> list[ExternalBrollCandidate]:
    headers = {"Authorization": api_key}
    timeout = httpx.Timeout(timeout_sec)
    results: list[ExternalBrollCandidate] = []
    seen_urls: set[str] = set()

    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        for packet in query_packets:
            if len(results) >= limit:
                break
            query = str(packet.get("query") or "").strip()
            query_mode = str(packet.get("mode") or "literal").strip()
            orientations = ["portrait"]
            if allow_landscape:
                orientations.append("landscape")
            for requested_orientation in orientations:
                if len(results) >= limit:
                    break
                try:
                    response = client.get(
                        "https://api.pexels.com/videos/search",
                        params={
                            "query": query,
                            "per_page": per_query,
                            "orientation": requested_orientation,
                            "size": "medium",
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                except Exception:
                    continue

                videos = payload.get("videos")
                if not isinstance(videos, list):
                    continue

                total = len(videos)
                for idx, item in enumerate(videos):
                    if not isinstance(item, dict):
                        continue
                    file_info = _pick_pexels_file(item.get("video_files") or [], target_orientation=target_orientation)
                    if not file_info:
                        continue
                    source_url = str(file_info.get("link") or "").strip()
                    if not source_url or source_url in seen_urls:
                        continue
                    seen_urls.add(source_url)

                    width = int(file_info.get("width") or item.get("width") or 0)
                    height = int(file_info.get("height") or item.get("height") or 0)
                    duration = float(item.get("duration") or 0.0) if item.get("duration") is not None else None
                    photographer = str((item.get("user") or {}).get("name") or "").strip()
                    page_url = str(item.get("url") or "").strip()
                    clip_id = str(item.get("id") or "").strip()
                    pexels_tags = item.get("tags")
                    thumbnail_url = str(item.get("image") or "").strip()
                    label = f"Pexels {clip_id}".strip()
                    if photographer:
                        label = f"Pexels {clip_id} - {photographer}".strip()
                    tag_items = [str(tag).strip() for tag in pexels_tags] if isinstance(pexels_tags, list) else []
                    relevance_score, keyword_hits = _query_relevance(
                        query=query,
                        source_label=label,
                        page_url=page_url,
                        tags=tag_items,
                    )
                    query_terms_count = _query_term_count(query)
                    metadata_miss_penalty = 0.92 if query_terms_count >= 4 and not keyword_hits and idx > max(2, total // 5) else 1.0
                    low_relevance_penalty = 0.88 if query_terms_count >= 2 and relevance_score <= 0.02 else 1.0
                    crop_score = _crop_score(width, height, target_orientation)

                    score = _clamp(
                        0.18
                        + relevance_score
                        + _duration_score(duration, slot_duration_sec)
                        + _orientation_score(width, height)
                        + _resolution_score(width, height)
                        + (crop_score * 0.18)
                        + _rank_bonus(idx, total),
                        0.0,
                        0.99,
                    )
                    score = _clamp(score * low_relevance_penalty * metadata_miss_penalty, 0.0, 0.99)
                    results.append(
                        ExternalBrollCandidate(
                            source_type="pexels_video",
                            source_url=source_url,
                            source_label=label,
                            score=round(score, 3),
                            reason={
                                "provider": "pexels",
                                "query": query,
                                "query_mode": query_mode,
                                "query_terms": query.split(),
                                "page_url": page_url,
                                "tags": tag_items,
                                "keyword_hits": keyword_hits,
                                "relevance_score": round(relevance_score, 3),
                                "photographer": photographer,
                                "video_id": clip_id,
                                "duration_sec": duration,
                                "width": width,
                                "height": height,
                                "thumbnail_url": thumbnail_url,
                                "requested_orientation": requested_orientation,
                                "crop_score": round(crop_score, 3),
                                "visual_intent": visual_intent,
                            },
                        )
                    )
                    if len(results) >= limit:
                        break
    return results


def _pick_pixabay_video(videos: dict[str, Any], *, target_orientation: str) -> tuple[str, int, int] | None:
    if not isinstance(videos, dict):
        return None
    candidates: list[tuple[str, int, int, float]] = []
    for key in ("large", "medium", "small", "tiny"):
        item = videos.get(key)
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url.startswith("http"):
            continue
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        candidates.append((url, width, height, _preferred_stream_score(width, height, target_orientation)))
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item[3])
    return (best[0], best[1], best[2])


def _search_pixabay(
    *,
    query_packets: list[dict[str, str]],
    slot_duration_sec: float,
    per_query: int,
    timeout_sec: float,
    limit: int,
    api_key: str,
    target_orientation: str,
    allow_landscape: bool,
    visual_intent: str,
) -> list[ExternalBrollCandidate]:
    timeout = httpx.Timeout(timeout_sec)
    results: list[ExternalBrollCandidate] = []
    seen_urls: set[str] = set()

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for packet in query_packets:
            if len(results) >= limit:
                break
            query = str(packet.get("query") or "").strip()
            query_mode = str(packet.get("mode") or "literal").strip()
            orientations = ["vertical"]
            if allow_landscape:
                orientations.append("horizontal")
            for requested_orientation in orientations:
                if len(results) >= limit:
                    break
                try:
                    response = client.get(
                        "https://pixabay.com/api/videos/",
                        params={
                            "key": api_key,
                            "q": query,
                            "per_page": per_query,
                            "orientation": requested_orientation,
                            "safesearch": "true",
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                except Exception:
                    continue

                hits = payload.get("hits")
                if not isinstance(hits, list):
                    continue

                total = len(hits)
                for idx, item in enumerate(hits):
                    if not isinstance(item, dict):
                        continue
                    picked = _pick_pixabay_video(item.get("videos") or {}, target_orientation=target_orientation)
                    if not picked:
                        continue
                    source_url, width, height = picked
                    if source_url in seen_urls:
                        continue
                    seen_urls.add(source_url)

                    duration = float(item.get("duration") or 0.0) if item.get("duration") is not None else None
                    user = str(item.get("user") or "").strip()
                    page_url = str(item.get("pageURL") or "").strip()
                    clip_id = str(item.get("id") or "").strip()
                    tags = str(item.get("tags") or "").strip()
                    picture_id = str(item.get("picture_id") or "").strip()
                    thumbnail_url = f"https://i.vimeocdn.com/video/{picture_id}_295x166.jpg" if picture_id else ""

                    label = f"Pixabay {clip_id}".strip()
                    if user:
                        label = f"Pixabay {clip_id} - {user}".strip()
                    tag_items = [part.strip() for part in tags.split(",") if part.strip()]
                    relevance_score, keyword_hits = _query_relevance(
                        query=query,
                        source_label=label,
                        page_url=page_url,
                        tags=tag_items,
                    )
                    query_terms_count = _query_term_count(query)
                    metadata_miss_penalty = 0.92 if query_terms_count >= 4 and not keyword_hits and idx > max(2, total // 5) else 1.0
                    low_relevance_penalty = 0.88 if query_terms_count >= 2 and relevance_score <= 0.02 else 1.0
                    crop_score = _crop_score(width, height, target_orientation)

                    score = _clamp(
                        0.16
                        + relevance_score
                        + _duration_score(duration, slot_duration_sec)
                        + _orientation_score(width, height)
                        + _resolution_score(width, height)
                        + (crop_score * 0.2)
                        + _rank_bonus(idx, total),
                        0.0,
                        0.99,
                    )
                    score = _clamp(score * low_relevance_penalty * metadata_miss_penalty, 0.0, 0.99)
                    results.append(
                        ExternalBrollCandidate(
                            source_type="pixabay_video",
                            source_url=source_url,
                            source_label=label,
                            score=round(score, 3),
                            reason={
                                "provider": "pixabay",
                                "query": query,
                                "query_mode": query_mode,
                                "query_terms": query.split(),
                                "page_url": page_url,
                                "tags": tag_items,
                                "keyword_hits": keyword_hits,
                                "relevance_score": round(relevance_score, 3),
                                "author": user,
                                "video_id": clip_id,
                                "duration_sec": duration,
                                "width": width,
                                "height": height,
                                "thumbnail_url": thumbnail_url,
                                "requested_orientation": requested_orientation,
                                "crop_score": round(crop_score, 3),
                                "visual_intent": visual_intent,
                            },
                        )
                    )
                    if len(results) >= limit:
                        break
    return results


def search_external_broll_candidates(
    *,
    concept_text: str,
    concept_tokens: list[str],
    slot_duration_sec: float,
    limit: int,
    chunk_text: str | None = None,
    query_hints: list[str] | None = None,
    query_packets: list[dict[str, str]] | None = None,
    visual_intent: str = "literal_demo",
    target_orientation: str = "portrait",
    allow_landscape: bool = True,
    domain_context: dict[str, Any] | None = None,
) -> list[ExternalBrollCandidate]:
    if limit <= 0:
        return []
    settings = get_settings()
    if not settings.broll_external_enabled:
        return []

    resolved_query_packets = query_packets or _build_query_packets(
        chunk_text or concept_text,
        concept_text,
        concept_tokens,
        query_hints,
        visual_intent,
        max(settings.broll_external_max_queries, min(8, max(4, limit // 3))),
        domain_context,
    )
    if not resolved_query_packets:
        return []

    per_query = max(settings.broll_external_per_query, min(20, max(8, limit // 2)))
    timeout_sec = max(2.0, settings.broll_external_timeout_sec)
    provider_limit = max(limit, min(limit * 2, 40))

    candidates: list[ExternalBrollCandidate] = []
    if settings.pexels_api_key:
        candidates.extend(
            _search_pexels(
                query_packets=resolved_query_packets,
                slot_duration_sec=slot_duration_sec,
                per_query=per_query,
                timeout_sec=timeout_sec,
                limit=provider_limit,
                api_key=settings.pexels_api_key,
                target_orientation=target_orientation,
                allow_landscape=allow_landscape,
                visual_intent=visual_intent,
            )
        )
    if settings.pixabay_api_key:
        candidates.extend(
            _search_pixabay(
                query_packets=resolved_query_packets,
                slot_duration_sec=slot_duration_sec,
                per_query=per_query,
                timeout_sec=timeout_sec,
                limit=provider_limit,
                api_key=settings.pixabay_api_key,
                target_orientation=target_orientation,
                allow_landscape=allow_landscape,
                visual_intent=visual_intent,
            )
        )

    if not candidates:
        return []

    deduped: list[ExternalBrollCandidate] = []
    seen_urls: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if candidate.source_url in seen_urls:
            continue
        seen_urls.add(candidate.source_url)
        deduped.append(candidate)
        if len(deduped) >= provider_limit:
            break
    return _diversify_candidates(deduped, limit)
