from __future__ import annotations

import os
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from .broll_external_service import ExternalBrollCandidate
from .config import get_settings


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _normalize_video_url(item: object) -> str:
    if isinstance(item, str):
        text = item.strip()
        if text.startswith("http://") or text.startswith("https://"):
            return text
        return ""
    if isinstance(item, dict):
        for key in ("url", "video_url", "download_url", "asset_url"):
            raw = item.get(key)
            if isinstance(raw, str) and raw.strip().startswith(("http://", "https://")):
                return raw.strip()
    return ""


def _to_candidates(
    payload: dict[str, Any],
    *,
    concept_text: str,
    shot_hint: str,
    limit: int,
) -> list[ExternalBrollCandidate]:
    candidates: list[ExternalBrollCandidate] = []
    raw_items: list[object] = []

    direct_url = _normalize_video_url(payload)
    if direct_url:
        raw_items.append({"url": direct_url})

    for key in ("videos", "results", "data", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            raw_items.extend(value)

    seen: set[str] = set()
    for idx, item in enumerate(raw_items):
        if len(candidates) >= limit:
            break
        url = _normalize_video_url(item)
        if not url or url in seen:
            continue
        seen.add(url)
        if isinstance(item, dict):
            raw_label = item.get("label") or item.get("title") or item.get("name")
            label = str(raw_label).strip() if raw_label else ""
            prompt = str(item.get("prompt") or item.get("used_prompt") or "").strip()
        else:
            label = ""
            prompt = ""
        score = _clamp(0.86 - (idx * 0.02), 0.62, 0.95)
        candidates.append(
            ExternalBrollCandidate(
                source_type="generated_video",
                source_url=url,
                source_label=label or f"Generated clip {idx + 1}",
                score=round(score, 3),
                reason={
                    "provider": "generative",
                    "query": concept_text,
                    "shot_type": shot_hint,
                    "prompt": prompt,
                },
            )
        )
    return candidates


# ── Ken Burns animation helper ──────────────────────────────────────────

_KEN_BURNS_STYLES: list[dict[str, str]] = [
    # Slow zoom in (most common, feels cinematic)
    {
        "name": "zoom_in",
        "zoompan": "z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
    },
    # Slow zoom out
    {
        "name": "zoom_out",
        "zoompan": "z='if(eq(on,1),1.5,max(zoom-0.0015,1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
    },
    # Pan left to right
    {
        "name": "pan_right",
        "zoompan": "z='1.15':x='min(on*2,iw/zoom-iw)':y='0'",
    },
    # Pan right to left
    {
        "name": "pan_left",
        "zoompan": "z='1.15':x='max(iw/zoom-iw-on*2,0)':y='0'",
    },
    # Zoom in on lower-right (focus on detail)
    {
        "name": "zoom_detail",
        "zoompan": "z='min(zoom+0.002,1.6)':x='iw/4':y='ih/4'",
    },
]


def _build_kenburns_video(
    image_path: str,
    output_path: str,
    *,
    duration_sec: float = 3.0,
    width: int = 1080,
    height: int = 1920,
    fps: int = 24,
    style_index: int = 0,
) -> bool:
    """Convert a static image into a video clip with Ken Burns animation."""
    settings = get_settings()
    style = _KEN_BURNS_STYLES[style_index % len(_KEN_BURNS_STYLES)]
    total_frames = int(duration_sec * fps)

    # zoompan generates the animated frames from the still image
    zoompan_filter = (
        f"zoompan={style['zoompan']}:d={total_frames}:s={width}x{height}:fps={fps}"
    )
    cmd = [
        settings.ffmpeg_bin,
        "-y",
        "-i", image_path,
        "-vf", f"{zoompan_filter},format=yuv420p",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-t", f"{duration_sec:.2f}",
        "-an",  # No audio
        output_path,
    ]
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except Exception:
        return False


# ── Image generation providers ──────────────────────────────────────────

_BROLL_STYLE_PREFIX = (
    "cinematic vertical photograph, dramatic lighting, shallow depth of field, "
    "professional color grading, 9:16 aspect ratio, "
)


def _build_image_prompt(concept_text: str, shot_hint: str) -> str:
    """Build a rich image generation prompt from transcript concepts."""
    parts = [_BROLL_STYLE_PREFIX]
    if concept_text.strip():
        parts.append(concept_text.strip())
    if shot_hint.strip():
        parts.append(f", {shot_hint.strip()} shot")
    parts.append(", highly detailed, no text, no watermark")
    return "".join(parts)


def _generate_pollinations_image(
    prompt: str,
    *,
    output_path: str,
    timeout_sec: float = 45.0,
    width: int = 576,
    height: int = 1024,
) -> bool:
    """Generate an image via Pollinations.ai (completely free, no API key needed)."""
    encoded = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true"
    try:
        timeout = httpx.Timeout(timeout_sec)
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            if response.status_code != 200:
                return False
            content_type = response.headers.get("content-type", "")
            if "image" not in content_type and "octet-stream" not in content_type:
                return False
            if len(response.content) < 1000:
                return False
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(response.content)
            return True
    except Exception:
        return False


def _generate_hf_image(
    prompt: str,
    *,
    output_path: str,
    model: str = "black-forest-labs/FLUX.1-schnell",
    api_key: str = "",
    timeout_sec: float = 30.0,
    width: int = 576,
    height: int = 1024,
) -> bool:
    """Generate an image via HuggingFace Inference API (free with token)."""
    url = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "width": width,
            "height": height,
        },
    }

    try:
        timeout = httpx.Timeout(timeout_sec)
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "image" not in content_type and "octet-stream" not in content_type:
                return False
            if len(response.content) < 1000:
                return False
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(response.content)
            return True
    except Exception:
        return False


def _generate_image(
    prompt: str,
    *,
    output_path: str,
    model: str,
    api_key: str,
    timeout_sec: float,
) -> bool:
    """Try image generation with available providers (falls back gracefully)."""
    # Provider 1: HuggingFace (if API key available)
    if api_key:
        hf_model = model if model != "auto" else "black-forest-labs/FLUX.1-schnell"
        if _generate_hf_image(prompt, output_path=output_path, model=hf_model, api_key=api_key, timeout_sec=timeout_sec):
            return True

    # Provider 2: Pollinations.ai (completely free, no key needed)
    if _generate_pollinations_image(prompt, output_path=output_path, timeout_sec=timeout_sec):
        return True

    return False


# ── Main public API ─────────────────────────────────────────────────────

def generate_generative_broll_candidates(
    *,
    concept_text: str,
    concept_tokens: list[str],
    shot_hint: str,
    slot_duration_sec: float,
    limit: int,
) -> list[ExternalBrollCandidate]:
    settings = get_settings()
    if limit <= 0:
        return []
    if not settings.broll_generative_enabled:
        return []

    # ── Path A: Custom generative API (e.g. self-hosted video gen) ──
    if settings.broll_generative_api_url:
        return _generate_via_custom_api(
            concept_text=concept_text,
            concept_tokens=concept_tokens,
            shot_hint=shot_hint,
            slot_duration_sec=slot_duration_sec,
            limit=limit,
        )

    # ── Path B: AI image → Ken Burns video (FREE) ───────────────────
    api_key = settings.broll_generative_api_key
    model = settings.broll_generative_model

    prompt = _build_image_prompt(concept_text, shot_hint)
    tmp_dir = Path(settings.tmp_dir).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[ExternalBrollCandidate] = []
    # Generate 1 image per slot to balance speed vs quality
    count = min(limit, 1)

    for gen_idx in range(count):
        uid = uuid4().hex[:12]
        image_path = str(tmp_dir / f"hf_broll_{uid}.png")
        video_path = str(tmp_dir / f"hf_broll_{uid}.mp4")

        ok = _generate_image(
            prompt,
            output_path=image_path,
            model=model,
            api_key=api_key,
            timeout_sec=settings.broll_generative_timeout_sec,
        )
        if not ok:
            continue

        duration = max(slot_duration_sec, 1.5)
        style_idx = gen_idx  # Vary Ken Burns style between candidates
        ok = _build_kenburns_video(
            image_path,
            video_path,
            duration_sec=duration,
            style_index=style_idx,
        )
        if not ok:
            try:
                os.unlink(image_path)
            except OSError:
                pass
            continue

        # Clean up image (video is the deliverable)
        try:
            os.unlink(image_path)
        except OSError:
            pass

        score = _clamp(0.82 - (gen_idx * 0.04), 0.60, 0.92)
        candidates.append(
            ExternalBrollCandidate(
                source_type="generated_image_video",
                source_url=video_path,  # Local file path for generated content
                source_label=f"AI Generated B-roll",
                score=round(score, 3),
                reason={
                    "provider": "ai_image",
                    "model": model,
                    "query": concept_text,
                    "shot_type": shot_hint,
                    "prompt": prompt,
                    "ken_burns_style": _KEN_BURNS_STYLES[style_idx % len(_KEN_BURNS_STYLES)]["name"],
                },
            )
        )

    return candidates


def _generate_via_custom_api(
    *,
    concept_text: str,
    concept_tokens: list[str],
    shot_hint: str,
    slot_duration_sec: float,
    limit: int,
) -> list[ExternalBrollCandidate]:
    """Original custom API path for self-hosted video generation services."""
    settings = get_settings()
    timeout = httpx.Timeout(settings.broll_generative_timeout_sec)
    headers = {"Content-Type": "application/json"}
    if settings.broll_generative_api_key:
        headers["Authorization"] = f"Bearer {settings.broll_generative_api_key}"
        headers["X-API-Key"] = settings.broll_generative_api_key

    prompt = " ".join(token for token in [concept_text.strip(), shot_hint.strip()] if token).strip()
    if not prompt:
        prompt = "cinematic broll"

    payload = {
        "prompt": prompt,
        "concept_text": concept_text,
        "concept_tokens": concept_tokens,
        "duration_sec": round(max(0.8, slot_duration_sec), 3),
        "aspect_ratio": "9:16",
        "count": max(1, min(5, limit)),
        "model": settings.broll_generative_model,
    }

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.post(settings.broll_generative_api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception:
        return []

    if not isinstance(data, dict):
        return []
    return _to_candidates(
        data,
        concept_text=concept_text,
        shot_hint=shot_hint,
        limit=limit,
    )
