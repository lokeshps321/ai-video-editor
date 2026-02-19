import os

import pytest

pytest.importorskip("sqlmodel")

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")

from fastapi.testclient import TestClient

from app.main import app
from app.transcription_service import TranscriptPayload, TranscriptWordPayload


def _create_project(client: TestClient, name: str = "Transcript Test") -> str:
    response = client.post(
        "/api/v1/projects",
        json={"name": name, "fps": 30, "width": 1080, "height": 1920},
    )
    assert response.status_code == 200
    return response.json()["id"]


def _upload_video(client: TestClient, project_id: str) -> str:
    response = client.post(
        "/api/v1/media/upload",
        data={"project_id": project_id},
        files={"file": ("demo.mp4", b"fake-video-bytes", "video/mp4")},
    )
    assert response.status_code == 200
    return response.json()["id"]


def test_transcript_generate_and_cut_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIPT_CUT_CONTEXT_SEC", "0")
    monkeypatch.setenv("TRANSCRIPT_CUT_MERGE_GAP_SEC", "0.08")
    monkeypatch.setenv("TRANSCRIPT_CUT_MIN_REMOVAL_SEC", "0")
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr("app.routers.media.probe_stream_flags", lambda _: {"has_video": True, "has_audio": True})

    def fake_generate_transcript(_path: str, _duration_sec: float) -> TranscriptPayload:
        words = [
            TranscriptWordPayload(id="w1", text="hello", start_sec=0.5, end_sec=1.0),
            TranscriptWordPayload(id="w2", text="brave", start_sec=1.0, end_sec=1.3),
            TranscriptWordPayload(id="w3", text="new", start_sec=2.0, end_sec=2.4),
            TranscriptWordPayload(id="w4", text="world", start_sec=2.4, end_sec=2.9),
        ]
        return TranscriptPayload(
            source="test_provider",
            language="en",
            text="hello brave new world",
            words=words,
            is_mock=False,
        )

    monkeypatch.setattr("app.routers.transcript.generate_transcript", fake_generate_transcript)

    with TestClient(app) as client:
        project_id = _create_project(client)
        asset_id = _upload_video(client, project_id)

        generate_res = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id},
        )
        assert generate_res.status_code == 200
        generated = generate_res.json()
        transcript = generated["transcript"]
        assert transcript["source"] == "test_provider"
        assert len(transcript["words"]) == 4
        transcript_id = transcript["id"]

        cut_res = client.post(
            f"/api/v1/transcript/cut?project_id={project_id}",
            json={"transcript_id": transcript_id, "kept_word_ids": ["w1", "w4"]},
        )
        assert cut_res.status_code == 200
        cut_payload = cut_res.json()
        assert cut_payload["kept_word_count"] == 2
        assert cut_payload["removed_word_count"] == 2

        video_track = next(track for track in cut_payload["timeline"]["tracks"] if track["kind"] == "video")
        assert len(video_track["clips"]) == 2
        assert video_track["clips"][0]["start_sec"] == 0.0
        assert video_track["clips"][0]["end_sec"] == 1.0
        assert video_track["clips"][1]["start_sec"] == 2.4
        assert video_track["clips"][1]["end_sec"] == 8.0


def test_transcript_cut_applies_context_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIPT_CUT_CONTEXT_SEC", "0.18")
    monkeypatch.setenv("TRANSCRIPT_CUT_MERGE_GAP_SEC", "0.08")
    monkeypatch.setenv("TRANSCRIPT_CUT_MIN_REMOVAL_SEC", "0.35")
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr("app.routers.media.probe_stream_flags", lambda _: {"has_video": True, "has_audio": True})

    def fake_generate_transcript(_path: str, _duration_sec: float) -> TranscriptPayload:
        words = [
            TranscriptWordPayload(id="w1", text="hello", start_sec=0.5, end_sec=1.0),
            TranscriptWordPayload(id="w2", text="brave", start_sec=1.0, end_sec=1.3),
            TranscriptWordPayload(id="w3", text="new", start_sec=2.0, end_sec=2.4),
            TranscriptWordPayload(id="w4", text="world", start_sec=2.4, end_sec=2.9),
        ]
        return TranscriptPayload(
            source="test_provider",
            language="en",
            text="hello brave new world",
            words=words,
            is_mock=False,
        )

    monkeypatch.setattr("app.routers.transcript.generate_transcript", fake_generate_transcript)

    with TestClient(app) as client:
        project_id = _create_project(client, name="Transcript Padding Test")
        asset_id = _upload_video(client, project_id)

        generated = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id},
        )
        assert generated.status_code == 200
        transcript_id = generated.json()["transcript"]["id"]

        cut_res = client.post(
            f"/api/v1/transcript/cut?project_id={project_id}",
            json={"transcript_id": transcript_id, "kept_word_ids": ["w1", "w4"]},
        )
        assert cut_res.status_code == 200
        cut_payload = cut_res.json()
        video_track = next(track for track in cut_payload["timeline"]["tracks"] if track["kind"] == "video")
        assert len(video_track["clips"]) == 2
        assert video_track["clips"][0]["start_sec"] == pytest.approx(0.0, abs=0.001)
        assert video_track["clips"][0]["end_sec"] == pytest.approx(1.18, abs=0.01)
        assert video_track["clips"][1]["start_sec"] == pytest.approx(2.22, abs=0.01)
        assert video_track["clips"][1]["end_sec"] == pytest.approx(8.0, abs=0.001)


def test_transcript_cut_preserves_unedited_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIPT_CUT_CONTEXT_SEC", "0")
    monkeypatch.setenv("TRANSCRIPT_CUT_MERGE_GAP_SEC", "0.08")
    monkeypatch.setenv("TRANSCRIPT_CUT_MIN_REMOVAL_SEC", "0")
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 40.0)
    monkeypatch.setattr("app.routers.media.probe_stream_flags", lambda _: {"has_video": True, "has_audio": True})

    def fake_generate_transcript(_path: str, _duration_sec: float) -> TranscriptPayload:
        words = [
            TranscriptWordPayload(id="w1", text="alpha", start_sec=0.5, end_sec=1.0),
            TranscriptWordPayload(id="w2", text="beta", start_sec=2.0, end_sec=2.4),
            TranscriptWordPayload(id="w3", text="gamma", start_sec=2.5, end_sec=3.0),
            TranscriptWordPayload(id="w4", text="delta", start_sec=30.0, end_sec=30.4),
        ]
        return TranscriptPayload(
            source="test_provider",
            language="en",
            text="alpha beta gamma delta",
            words=words,
            is_mock=False,
        )

    monkeypatch.setattr("app.routers.transcript.generate_transcript", fake_generate_transcript)

    with TestClient(app) as client:
        project_id = _create_project(client, name="Transcript Gap Preservation Test")
        asset_id = _upload_video(client, project_id)

        generated = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id},
        )
        assert generated.status_code == 200
        transcript_id = generated.json()["transcript"]["id"]

        cut_res = client.post(
            f"/api/v1/transcript/cut?project_id={project_id}",
            json={"transcript_id": transcript_id, "kept_word_ids": ["w1", "w3", "w4"]},
        )
        assert cut_res.status_code == 200
        cut_payload = cut_res.json()
        video_track = next(track for track in cut_payload["timeline"]["tracks"] if track["kind"] == "video")
        assert len(video_track["clips"]) == 2
        assert video_track["clips"][0]["start_sec"] == pytest.approx(0.0, abs=0.001)
        assert video_track["clips"][0]["end_sec"] == pytest.approx(1.0, abs=0.001)
        assert video_track["clips"][1]["start_sec"] == pytest.approx(2.5, abs=0.001)
        assert video_track["clips"][1]["end_sec"] == pytest.approx(40.0, abs=0.001)


def test_transcript_allows_videos_over_60_seconds_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 75.0)
    monkeypatch.setattr("app.routers.media.probe_stream_flags", lambda _: {"has_video": True, "has_audio": True})
    monkeypatch.setattr(
        "app.routers.transcript.generate_transcript",
        lambda _path, _duration_sec: TranscriptPayload(
            source="test_provider",
            language="en",
            text="hello world",
            words=[
                TranscriptWordPayload(id="w1", text="hello", start_sec=0.0, end_sec=0.4),
                TranscriptWordPayload(id="w2", text="world", start_sec=0.4, end_sec=0.8),
            ],
            is_mock=False,
        ),
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="Long Video Rejection")
        asset_id = _upload_video(client, project_id)
        response = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id},
        )
        assert response.status_code == 200
