import os
import json

import pytest

pytest.importorskip("sqlmodel")

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")
os.environ["TRANSCRIBE_BACKEND"] = "local"

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.database import engine
from app.main import app
from app.models import Transcript
from app.transcription_service import TranscriptPayload, TranscriptWordPayload


def _create_project(client: TestClient) -> str:
    response = client.post(
        "/api/v1/projects",
        json={"name": "Vibe Route Test", "fps": 30, "width": 1080, "height": 1920},
    )
    assert response.status_code == 200
    return response.json()["id"]


def _upload_video(client: TestClient, project_id: str) -> str:
    response = client.post(
        "/api/v1/media/upload",
        data={"project_id": project_id},
        files={"file": ("vibe.mp4", b"fake-video-data", "video/mp4")},
    )
    assert response.status_code == 200
    return response.json()["id"]


def test_vibe_add_subtitles_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr("app.routers.media.probe_stream_flags", lambda _: {"has_video": True, "has_audio": True})
    monkeypatch.setattr("app.routers.vibe.detect_silence_ranges", lambda *_args, **_kwargs: [])

    def fake_generate(_path: str, _duration_sec: float) -> TranscriptPayload:
        words = [
            TranscriptWordPayload(id="w1", text="hello", start_sec=0.2, end_sec=0.6),
            TranscriptWordPayload(id="w2", text="there", start_sec=0.6, end_sec=1.1),
            TranscriptWordPayload(id="w3", text="friend", start_sec=2.0, end_sec=2.4),
        ]
        return TranscriptPayload(source="test", language="en", text="hello there friend", words=words, is_mock=False)

    monkeypatch.setattr("app.routers.vibe.generate_transcript", fake_generate)

    with TestClient(app) as client:
        project_id = _create_project(client)
        asset_id = _upload_video(client, project_id)
        response = client.post(
            f"/api/v1/vibe/apply?project_id={project_id}",
            json={"action": "add_subtitles", "asset_id": asset_id},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["action"] == "add_subtitles"
        assert payload["transcript_id"] is not None
        assert payload["preview_job"]["kind"] == "preview"
        video_track = next(track for track in payload["timeline"]["tracks"] if track["kind"] == "video")
        assert len(video_track["clips"][0]["text_overlays"]) >= 1


def test_vibe_add_subtitles_regenerates_low_quality_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_REGENERATE_LOW_QUALITY", "true")
    monkeypatch.setenv("TRANSCRIBE_MIN_WORDS_PER_SEC", "0.45")
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 120.0)
    monkeypatch.setattr("app.routers.media.probe_stream_flags", lambda _: {"has_video": True, "has_audio": True})
    monkeypatch.setattr("app.routers.vibe.detect_silence_ranges", lambda *_args, **_kwargs: [])

    def fake_generate(_path: str, _duration_sec: float) -> TranscriptPayload:
        words = [
            TranscriptWordPayload(id=f"w{idx}", text=f"word{idx}", start_sec=idx * 0.4, end_sec=(idx * 0.4) + 0.2)
            for idx in range(80)
        ]
        return TranscriptPayload(
            source="test",
            language="en",
            text=" ".join(word.text for word in words),
            words=words,
            is_mock=False,
        )

    monkeypatch.setattr("app.routers.vibe.generate_transcript", fake_generate)

    with TestClient(app) as client:
        project_id = _create_project(client)
        asset_id = _upload_video(client, project_id)

        low_words = [
            {"id": "lw1", "text": "hello", "start_sec": 0.1, "end_sec": 0.5},
            {"id": "lw2", "text": "there", "start_sec": 0.5, "end_sec": 1.0},
        ]
        with Session(engine) as session:
            low_row = Transcript(
                project_id=project_id,
                asset_id=asset_id,
                source="faster_whisper",
                language="en",
                text="hello there",
                words_json=json.dumps(low_words),
                duration_sec=120.0,
                is_mock=False,
            )
            session.add(low_row)
            session.commit()
            session.refresh(low_row)
            low_id = low_row.id

        response = client.post(
            f"/api/v1/vibe/apply?project_id={project_id}",
            json={"action": "add_subtitles", "asset_id": asset_id},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["transcript_id"] is not None
        assert payload["transcript_id"] != low_id
        video_track = next(track for track in payload["timeline"]["tracks"] if track["kind"] == "video")
        assert len(video_track["clips"][0]["text_overlays"]) >= 1


def test_vibe_auto_cut_pauses_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr("app.routers.media.probe_stream_flags", lambda _: {"has_video": True, "has_audio": True})
    monkeypatch.setattr(
        "app.routers.vibe.detect_silence_ranges",
        lambda *_args, **_kwargs: [(1.0, 2.0), (4.0, 4.8)],
    )

    def fake_generate(_path: str, _duration_sec: float) -> TranscriptPayload:
        # Words span right up to silence boundaries.
        # Silence: (1.0, 2.0), (4.0, 4.8). Word gaps match exactly.
        words = [
            TranscriptWordPayload(id="w1", text="hello", start_sec=0.0, end_sec=0.5),
            TranscriptWordPayload(id="w2", text="there", start_sec=0.5, end_sec=1.0),
            TranscriptWordPayload(id="w3", text="my",    start_sec=2.0, end_sec=2.5),
            TranscriptWordPayload(id="w4", text="dear",  start_sec=2.5, end_sec=3.0),
            TranscriptWordPayload(id="w5", text="friend", start_sec=3.0, end_sec=3.5),
            TranscriptWordPayload(id="w6", text="how",   start_sec=3.5, end_sec=4.0),
            TranscriptWordPayload(id="w7", text="are",   start_sec=4.8, end_sec=5.3),
            TranscriptWordPayload(id="w8", text="you",   start_sec=5.3, end_sec=5.8),
            TranscriptWordPayload(id="w9", text="today", start_sec=5.8, end_sec=6.3),
        ]
        return TranscriptPayload(source="test", language="en", text="hello there my dear friend how are you today", words=words, is_mock=False)

    monkeypatch.setattr("app.routers.vibe.generate_transcript", fake_generate)

    with TestClient(app) as client:
        project_id = _create_project(client)
        asset_id = _upload_video(client, project_id)
        response = client.post(
            f"/api/v1/vibe/apply?project_id={project_id}",
            json={"action": "auto_cut_pauses", "asset_id": asset_id, "options": {"remove_filler_words": False}},
        )
        assert response.status_code == 200
        payload = response.json()
        video_track = next(track for track in payload["timeline"]["tracks"] if track["kind"] == "video")
        assert len(video_track["clips"]) == 3
        assert payload["timeline"]["duration_sec"] == pytest.approx(6.2, abs=0.01)


def test_vibe_trim_start_end_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr("app.routers.media.probe_stream_flags", lambda _: {"has_video": True, "has_audio": True})
    monkeypatch.setattr("app.routers.vibe.detect_silence_ranges", lambda *_args, **_kwargs: [(0.0, 0.5), (7.8, 8.0)])

    def fake_generate(_path: str, _duration_sec: float) -> TranscriptPayload:
        words = [
            TranscriptWordPayload(id="w1", text="start", start_sec=0.7, end_sec=1.0),
            TranscriptWordPayload(id="w2", text="end", start_sec=7.2, end_sec=7.4),
        ]
        return TranscriptPayload(source="test", language="en", text="start end", words=words, is_mock=False)

    monkeypatch.setattr("app.routers.vibe.generate_transcript", fake_generate)

    with TestClient(app) as client:
        project_id = _create_project(client)
        asset_id = _upload_video(client, project_id)
        response = client.post(
            f"/api/v1/vibe/apply?project_id={project_id}",
            json={"action": "trim_start_end", "asset_id": asset_id},
        )
        assert response.status_code == 200
        payload = response.json()
        video_track = next(track for track in payload["timeline"]["tracks"] if track["kind"] == "video")
        clip = video_track["clips"][0]
        assert clip["start_sec"] == pytest.approx(0.64, abs=0.01)
        assert clip["end_sec"] == pytest.approx(7.46, abs=0.01)
