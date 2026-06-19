#!/usr/bin/env python3
"""Smoke-check B-roll setup scenarios (in-process TestClient by default)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]

for env_file in (BACKEND / ".env", BACKEND / ".env.local"):
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/broll_live_audit.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/broll_live_audit_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/broll_live_audit_renders")
os.environ.setdefault("TMP_DIR", "/tmp/broll_live_audit_tmp")
os.environ.setdefault("BROLL_LLM_ENABLED", "false")

sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.transcription_service import TranscriptPayload, TranscriptWordPayload  # noqa: E402


def _fake_transcript(_path: str, _duration_sec: float, language_hint: str | None = None) -> TranscriptPayload:
    _ = language_hint
    words = [
        TranscriptWordPayload(id="w1", text="building", start_sec=0.0, end_sec=0.2),
        TranscriptWordPayload(id="w2", text="a", start_sec=0.2, end_sec=0.35),
        TranscriptWordPayload(id="w3", text="great", start_sec=0.35, end_sec=0.55),
        TranscriptWordPayload(id="w4", text="product.", start_sec=0.55, end_sec=0.9),
        TranscriptWordPayload(id="w5", text="this", start_sec=1.0, end_sec=1.2),
        TranscriptWordPayload(id="w6", text="needs", start_sec=1.2, end_sec=1.4),
        TranscriptWordPayload(id="w7", text="strong", start_sec=1.4, end_sec=1.6),
        TranscriptWordPayload(id="w8", text="visuals.", start_sec=1.6, end_sec=1.9),
    ]
    return TranscriptPayload(
        source="audit",
        language="en",
        text=" ".join(word.text for word in words),
        words=words,
        is_mock=False,
    )


def _create_project(client: TestClient, name: str) -> str:
    res = client.post("/api/v1/projects", json={"name": name, "fps": 30, "width": 1080, "height": 1920})
    res.raise_for_status()
    return res.json()["id"]


def _upload_video(client: TestClient, project_id: str, filename: str) -> str:
    res = client.post(
        "/api/v1/media/upload",
        data={"project_id": project_id},
        files={"file": (filename, b"fake-video-bytes", "video/mp4")},
    )
    res.raise_for_status()
    return res.json()["id"]


def run_scenario(name: str, *, single_asset: bool, external: bool, min_confidence: float, fallback: bool) -> dict:
    from app.routers import media, transcript

    transcript.generate_transcript = _fake_transcript
    media.probe_duration_seconds = lambda _: 10.0
    media.probe_stream_flags = lambda _: {"has_video": True, "has_audio": True}

    with TestClient(app) as client:
        project_id = _create_project(client, name)
        transcript_asset_id = _upload_video(client, project_id, "main.mp4")
        if not single_asset:
            _upload_video(client, project_id, "cutaway.mp4")

        generate_res = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": transcript_asset_id},
        )
        generate_res.raise_for_status()

        auto_res = client.post(
            f"/api/v1/broll/auto-apply?project_id={project_id}",
            json={
                "max_slots": 2,
                "candidates_per_slot": 2,
                "include_external_sources": external,
                "include_project_assets": True,
                "ai_rerank": False,
                "fallback_to_top_candidate": fallback,
                "min_confidence": min_confidence,
            },
        )
        auto_res.raise_for_status()
        payload = auto_res.json()
        return {
            "scenario": name,
            "synced": payload["synced_clip_count"],
            "skipped": payload["skipped_slots"],
            "summaries": payload.get("skipped_slot_summaries") or [],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run B-roll smoke scenarios.")
    parser.add_argument(
        "--live-keys-check",
        action="store_true",
        help="Only print whether Pexels/Pixabay keys are configured on this machine.",
    )
    args = parser.parse_args()

    settings = get_settings()
    print("B-roll config:")
    print(f"  stock_search_available={settings.broll_external_enabled and bool(settings.pexels_api_key or settings.pixabay_api_key)}")
    print(f"  pexels_configured={bool(settings.pexels_api_key)}")
    print(f"  pixabay_configured={bool(settings.pixabay_api_key)}")
    print(f"  generative_enabled={settings.broll_generative_enabled}")

    if args.live_keys_check:
        return 0

    scenarios = [
        run_scenario("single_asset_no_external", single_asset=True, external=False, min_confidence=0.76, fallback=True),
        run_scenario("dual_asset_no_external", single_asset=False, external=False, min_confidence=0.76, fallback=True),
        run_scenario("dual_asset_strict_confidence", single_asset=False, external=False, min_confidence=0.99, fallback=False),
    ]

    print("\nScenario results:")
    for item in scenarios:
        print(
            f"  - {item['scenario']}: synced={item['synced']} skipped={item['skipped']} "
            f"summaries={len(item['summaries'])}"
        )

    print("\nManual live checks (optional):")
    print("  1. Single video, no API keys -> B-roll Studio should show limited-sources banner.")
    print("  2. Single video + PEXELS_API_KEY -> auto-apply should sync stock overlays.")
    print("  3. Kannada speech transcript -> suggest should still return English-gloss concepts.")
    print("  4. Music/lyrics video -> expect weaker visual matches; review slots manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
