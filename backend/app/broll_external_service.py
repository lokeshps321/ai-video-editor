from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .config import get_settings


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


def _resolution_score(width: int | None, height: int | None) -> float:
    if not width or not height or width <= 0 or height <= 0:
        return 0.03
    pixels = float(width * height)
    return 0.03 + (_clamp(pixels / (1920.0 * 1080.0), 0.0, 1.0) * 0.07)


def _rank_bonus(index: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return _clamp((1.0 - (index / total)) * 0.12, 0.0, 0.12)


def _build_queries(concept_text: str, concept_tokens: list[str]) -> list[str]:
    queries: list[str] = []
    stripped = concept_text.strip()
    if stripped:
        queries.append(stripped)
    if concept_tokens:
        queries.append(" ".join(concept_tokens[:3]))
        queries.append(" ".join(concept_tokens[:2]))
        queries.append(concept_tokens[0])

    deduped: list[str] = []
    seen: set[str] = set()
    for item in queries:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item.strip())
    return deduped[:3]


def _pick_pexels_file(video_files: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not video_files:
        return None

    def _score(item: dict[str, Any]) -> float:
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        if width <= 0 or height <= 0:
            return 0.0
        pixels = width * height
        portrait_bonus = 0.2 if height >= width else 0.0
        return pixels + (portrait_bonus * 1_000_000.0)

    candidates = [
        item for item in video_files
        if str(item.get("link", "")).startswith("http")
    ]
    if not candidates:
        return None
    return max(candidates, key=_score)


def _search_pexels(
    *,
    queries: list[str],
    slot_duration_sec: float,
    per_query: int,
    timeout_sec: float,
    api_key: str,
) -> list[ExternalBrollCandidate]:
    headers = {"Authorization": api_key}
    timeout = httpx.Timeout(timeout_sec)
    results: list[ExternalBrollCandidate] = []
    seen_urls: set[str] = set()

    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        for query in queries:
            try:
                response = client.get(
                    "https://api.pexels.com/videos/search",
                    params={
                        "query": query,
                        "per_page": per_query,
                        "orientation": "portrait",
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
                file_info = _pick_pexels_file(item.get("video_files") or [])
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
                label = f"Pexels {clip_id}".strip()
                if photographer:
                    label = f"Pexels {clip_id} - {photographer}".strip()

                score = _clamp(
                    0.42
                    + _duration_score(duration, slot_duration_sec)
                    + _orientation_score(width, height)
                    + _resolution_score(width, height)
                    + _rank_bonus(idx, total),
                    0.0,
                    0.99,
                )
                results.append(
                    ExternalBrollCandidate(
                        source_type="pexels_video",
                        source_url=source_url,
                        source_label=label,
                        score=round(score, 3),
                        reason={
                            "provider": "pexels",
                            "query": query,
                            "page_url": page_url,
                            "duration_sec": duration,
                            "width": width,
                            "height": height,
                        },
                    )
                )
    return results


def _pick_pixabay_video(videos: dict[str, Any]) -> tuple[str, int, int] | None:
    if not isinstance(videos, dict):
        return None
    order = ["large", "medium", "small", "tiny"]
    for key in order:
        item = videos.get(key)
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url.startswith("http"):
            continue
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        return (url, width, height)
    return None


def _search_pixabay(
    *,
    queries: list[str],
    slot_duration_sec: float,
    per_query: int,
    timeout_sec: float,
    api_key: str,
) -> list[ExternalBrollCandidate]:
    timeout = httpx.Timeout(timeout_sec)
    results: list[ExternalBrollCandidate] = []
    seen_urls: set[str] = set()

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for query in queries:
            try:
                response = client.get(
                    "https://pixabay.com/api/videos/",
                    params={
                        "key": api_key,
                        "q": query,
                        "per_page": per_query,
                        "orientation": "vertical",
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
                picked = _pick_pixabay_video(item.get("videos") or {})
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
                label = f"Pixabay {clip_id}".strip()
                if user:
                    label = f"Pixabay {clip_id} - {user}".strip()

                score = _clamp(
                    0.40
                    + _duration_score(duration, slot_duration_sec)
                    + _orientation_score(width, height)
                    + _resolution_score(width, height)
                    + _rank_bonus(idx, total),
                    0.0,
                    0.99,
                )
                results.append(
                    ExternalBrollCandidate(
                        source_type="pixabay_video",
                        source_url=source_url,
                        source_label=label,
                        score=round(score, 3),
                        reason={
                            "provider": "pixabay",
                            "query": query,
                            "page_url": page_url,
                            "duration_sec": duration,
                            "width": width,
                            "height": height,
                        },
                    )
                )
    return results


def search_external_broll_candidates(
    *,
    concept_text: str,
    concept_tokens: list[str],
    slot_duration_sec: float,
    limit: int,
    query_hints: list[str] | None = None,
) -> list[ExternalBrollCandidate]:
    if limit <= 0:
        return []
    settings = get_settings()
    if not settings.broll_external_enabled:
        return []

    queries = _build_queries(concept_text, concept_tokens)
    if query_hints:
        seen: set[str] = {item.lower() for item in queries}
        for item in query_hints:
            trimmed = item.strip()
            key = trimmed.lower()
            if not trimmed or key in seen:
                continue
            seen.add(key)
            queries.append(trimmed)
            if len(queries) >= 6:
                break
    if not queries:
        return []

    per_query = max(1, min(settings.broll_external_per_query, 40))
    timeout_sec = max(2.0, settings.broll_external_timeout_sec)

    candidates: list[ExternalBrollCandidate] = []
    if settings.pexels_api_key:
        candidates.extend(
            _search_pexels(
                queries=queries,
                slot_duration_sec=slot_duration_sec,
                per_query=per_query,
                timeout_sec=timeout_sec,
                api_key=settings.pexels_api_key,
            )
        )
    if settings.pixabay_api_key and len(candidates) < limit:
        candidates.extend(
            _search_pixabay(
                queries=queries,
                slot_duration_sec=slot_duration_sec,
                per_query=per_query,
                timeout_sec=timeout_sec,
                api_key=settings.pixabay_api_key,
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
        if len(deduped) >= limit:
            break
    return deduped
