import os
import json

import pytest

pytest.importorskip("sqlmodel")

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")
os.environ["TRANSCRIBE_BACKEND"] = "local"
# Disable pre-compute vocal isolation for test performance
os.environ.setdefault("TRANSCRIBE_VOCAL_ISOLATION_PRECOMPUTE", "false")

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
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setattr(
        "app.routers.vibe.detect_silence_ranges", lambda *_args, **_kwargs: []
    )

    def fake_generate(_path: str, _duration_sec: float) -> TranscriptPayload:
        words = [
            TranscriptWordPayload(id="w1", text="hello", start_sec=0.2, end_sec=0.6),
            TranscriptWordPayload(id="w2", text="there", start_sec=0.6, end_sec=1.1),
            TranscriptWordPayload(id="w3", text="friend", start_sec=2.0, end_sec=2.4),
        ]
        return TranscriptPayload(
            source="test",
            language="en",
            text="hello there friend",
            words=words,
            is_mock=False,
        )

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
        saved_project = client.get(f"/api/v1/projects/{project_id}")
        assert saved_project.status_code == 200
        assert (
            payload["preview_job"]["timeline_version"]
            == saved_project.json()["timeline_version"]
        )
        video_track = next(
            track for track in payload["timeline"]["tracks"] if track["kind"] == "video"
        )
        overlays = video_track["clips"][0]["text_overlays"]
        assert len(overlays) == 3
        assert overlays[0]["text"] == "hello"


def test_vibe_add_subtitles_regenerates_low_quality_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_REGENERATE_LOW_QUALITY", "true")
    monkeypatch.setenv("TRANSCRIBE_MIN_WORDS_PER_SEC", "0.45")
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 120.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setattr(
        "app.routers.vibe.detect_silence_ranges", lambda *_args, **_kwargs: []
    )

    def fake_generate(_path: str, _duration_sec: float) -> TranscriptPayload:
        words = [
            TranscriptWordPayload(
                id=f"w{idx}",
                text=f"word{idx}",
                start_sec=idx * 0.4,
                end_sec=(idx * 0.4) + 0.2,
            )
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
        video_track = next(
            track for track in payload["timeline"]["tracks"] if track["kind"] == "video"
        )
        assert len(video_track["clips"][0]["text_overlays"]) >= 1


def test_vibe_add_subtitles_uses_requested_language_and_does_not_reuse_mismatched_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_REGENERATE_LOW_QUALITY", "false")
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 12.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setattr(
        "app.routers.vibe.detect_silence_ranges", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        "app.routers.vibe.enqueue_render_job", lambda *_args, **_kwargs: None
    )

    captured_kwargs: dict[str, object] = {}

    def fake_generate(_path: str, _duration_sec: float, **kwargs) -> TranscriptPayload:
        captured_kwargs.update(kwargs)
        words = [
            TranscriptWordPayload(id="w1", text="ಬಿಸಿಲು", start_sec=0.2, end_sec=0.7),
            TranscriptWordPayload(id="w2", text="ಎಂದು", start_sec=0.7, end_sec=1.1),
        ]
        return TranscriptPayload(
            source="test", language="kn", text="ಬಿಸಿಲು ಎಂದು", words=words, is_mock=False
        )

    monkeypatch.setattr("app.routers.vibe.generate_transcript", fake_generate)

    with TestClient(app) as client:
        project_id = _create_project(client)
        asset_id = _upload_video(client, project_id)

        with Session(engine) as session:
            row = Transcript(
                project_id=project_id,
                asset_id=asset_id,
                source="groq",
                language="en",
                text="wrong language",
                words_json=json.dumps(
                    [
                        {"id": "w1", "text": "wrong", "start_sec": 0.1, "end_sec": 0.4},
                        {"id": "w2", "text": "words", "start_sec": 0.4, "end_sec": 0.7},
                    ]
                ),
                duration_sec=12.0,
                is_mock=False,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            old_id = row.id

        response = client.post(
            f"/api/v1/vibe/apply?project_id={project_id}",
            json={
                "action": "add_subtitles",
                "asset_id": asset_id,
                "options": {"transcript_language": "kn"},
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["transcript_id"] != old_id
        assert captured_kwargs.get("language_hint") == "kn"
        video_track = next(
            track for track in payload["timeline"]["tracks"] if track["kind"] == "video"
        )
        overlays = video_track["clips"][0]["text_overlays"]
        assert overlays
        assert overlays[0]["text"] == "ಬಿಸಿಲು ಎಂದು"


def test_vibe_add_subtitles_regenerates_when_latest_transcript_is_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_ALLOW_MOCK_FALLBACK", "false")
    monkeypatch.setenv("TRANSCRIBE_REGENERATE_LOW_QUALITY", "false")
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setattr(
        "app.routers.vibe.detect_silence_ranges", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        "app.routers.vibe.enqueue_render_job", lambda *_args, **_kwargs: None
    )

    captured_kwargs: dict[str, object] = {}

    def fake_generate(_path: str, _duration_sec: float, **kwargs) -> TranscriptPayload:
        captured_kwargs.update(kwargs)
        words = [
            TranscriptWordPayload(id="rw1", text="real", start_sec=0.2, end_sec=0.6),
            TranscriptWordPayload(
                id="rw2", text="captions", start_sec=0.6, end_sec=1.0
            ),
        ]
        return TranscriptPayload(
            source="test",
            language="en",
            text="real captions",
            words=words,
            is_mock=False,
        )

    monkeypatch.setattr("app.routers.vibe.generate_transcript", fake_generate)

    with TestClient(app) as client:
        project_id = _create_project(client)
        asset_id = _upload_video(client, project_id)

        mock_words = [
            {
                "id": f"mw{idx}",
                "text": f"mock{idx}",
                "start_sec": idx * 0.4,
                "end_sec": (idx * 0.4) + 0.2,
            }
            for idx in range(12)
        ]
        with Session(engine) as session:
            mock_row = Transcript(
                project_id=project_id,
                asset_id=asset_id,
                source="mock",
                language="en",
                text=" ".join(item["text"] for item in mock_words),
                words_json=json.dumps(mock_words),
                duration_sec=8.0,
                is_mock=True,
            )
            session.add(mock_row)
            session.commit()
            session.refresh(mock_row)
            mock_id = mock_row.id

        response = client.post(
            f"/api/v1/vibe/apply?project_id={project_id}",
            json={"action": "add_subtitles", "asset_id": asset_id},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["transcript_id"] != mock_id
        assert captured_kwargs.get("allow_mock_fallback") is False
        video_track = next(
            track for track in payload["timeline"]["tracks"] if track["kind"] == "video"
        )
        overlays = video_track["clips"][0]["text_overlays"]
        assert overlays
        assert overlays[0]["text"] == "real"


def test_vibe_add_subtitles_fails_when_uploaded_media_file_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setattr(
        "app.routers.vibe.storage.resolve_upload_asset",
        lambda _path: "/tmp/does-not-exist-vibe.mp4",
    )

    with TestClient(app) as client:
        project_id = _create_project(client)
        asset_id = _upload_video(client, project_id)
        response = client.post(
            f"/api/v1/vibe/apply?project_id={project_id}",
            json={"action": "add_subtitles", "asset_id": asset_id},
        )

    assert response.status_code == 404
    assert (
        response.json()["detail"]
        == "Uploaded media file is missing. Re-upload the asset and try again."
    )


def test_vibe_add_subtitles_sanitizes_existing_pathological_word_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_REGENERATE_LOW_QUALITY", "false")
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setattr(
        "app.routers.vibe.detect_silence_ranges", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        "app.routers.vibe.generate_transcript",
        lambda *_args, **_kwargs: pytest.fail("unexpected regenerate"),
    )

    with TestClient(app) as client:
        project_id = _create_project(client)
        asset_id = _upload_video(client, project_id)

        bad_words = [
            {"id": "w1", "text": "As", "start_sec": 0.02, "end_sec": 7.82},
            {"id": "w2", "text": "I", "start_sec": 7.82, "end_sec": 26.44},
            {"id": "w3", "text": "walk", "start_sec": 26.4, "end_sec": 26.72},
        ]
        with Session(engine) as session:
            row = Transcript(
                project_id=project_id,
                asset_id=asset_id,
                source="groq",
                language="en",
                text="As I walk",
                words_json=json.dumps(bad_words),
                duration_sec=30.0,
                is_mock=False,
            )
            session.add(row)
            session.commit()

        response = client.post(
            f"/api/v1/vibe/apply?project_id={project_id}",
            json={"action": "add_subtitles", "asset_id": asset_id},
        )
        assert response.status_code == 200
        payload = response.json()
        video_track = next(
            track for track in payload["timeline"]["tracks"] if track["kind"] == "video"
        )
        overlays = video_track["clips"][0]["text_overlays"]
        assert overlays
        assert overlays[0]["text"] == "As"
        assert overlays[0]["duration_sec"] <= 1.2


def test_vibe_add_subtitles_uses_client_supplied_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 12.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setattr(
        "app.routers.vibe.detect_silence_ranges", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        "app.routers.vibe.generate_transcript",
        lambda *_args, **_kwargs: pytest.fail("unexpected regenerate"),
    )

    with TestClient(app) as client:
        project_id = _create_project(client)
        asset_id = _upload_video(client, project_id)

        with Session(engine) as session:
            row = Transcript(
                project_id=project_id,
                asset_id=asset_id,
                source="groq",
                language="en",
                text="old words",
                words_json=json.dumps(
                    [
                        {"id": "w1", "text": "old", "start_sec": 0.2, "end_sec": 0.4},
                        {"id": "w2", "text": "words", "start_sec": 0.4, "end_sec": 0.7},
                    ]
                ),
                duration_sec=12.0,
                is_mock=False,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            transcript_id = row.id

        response = client.post(
            f"/api/v1/vibe/apply?project_id={project_id}",
            json={
                "action": "add_subtitles",
                "asset_id": asset_id,
                "options": {
                    "words": [
                        {"id": "c1", "text": "new", "start_sec": 0.5, "end_sec": 0.8},
                        {
                            "id": "c2",
                            "text": "caption",
                            "start_sec": 0.8,
                            "end_sec": 1.1,
                        },
                    ]
                },
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["transcript_id"] == transcript_id
        video_track = next(
            track for track in payload["timeline"]["tracks"] if track["kind"] == "video"
        )
        overlays = video_track["clips"][0]["text_overlays"]
        assert overlays
        assert overlays[0]["text"] == "new"
        assert overlays[1]["text"] == "caption"


def test_vibe_auto_cut_pauses_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
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
            TranscriptWordPayload(id="w3", text="my", start_sec=2.0, end_sec=2.5),
            TranscriptWordPayload(id="w4", text="dear", start_sec=2.5, end_sec=3.0),
            TranscriptWordPayload(id="w5", text="friend", start_sec=3.0, end_sec=3.5),
            TranscriptWordPayload(id="w6", text="how", start_sec=3.5, end_sec=4.0),
            TranscriptWordPayload(id="w7", text="are", start_sec=4.8, end_sec=5.3),
            TranscriptWordPayload(id="w8", text="you", start_sec=5.3, end_sec=5.8),
            TranscriptWordPayload(id="w9", text="today", start_sec=5.8, end_sec=6.3),
        ]
        return TranscriptPayload(
            source="test",
            language="en",
            text="hello there my dear friend how are you today",
            words=words,
            is_mock=False,
        )

    monkeypatch.setattr("app.routers.vibe.generate_transcript", fake_generate)

    with TestClient(app) as client:
        project_id = _create_project(client)
        asset_id = _upload_video(client, project_id)
        response = client.post(
            f"/api/v1/vibe/apply?project_id={project_id}",
            json={
                "action": "auto_cut_pauses",
                "asset_id": asset_id,
                "options": {"remove_filler_words": False},
            },
        )
        assert response.status_code == 200
        payload = response.json()
        video_track = next(
            track for track in payload["timeline"]["tracks"] if track["kind"] == "video"
        )
        assert len(video_track["clips"]) == 3
        assert payload["timeline"]["duration_sec"] == pytest.approx(6.2, abs=0.01)


def test_vibe_auto_cut_pauses_uses_conservative_filler_detection_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_FILLER_AGGRESSIVE_SINGLE_WORDS", "false")
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setattr(
        "app.routers.vibe.detect_silence_ranges", lambda *_args, **_kwargs: []
    )

    def fake_generate(_path: str, _duration_sec: float) -> TranscriptPayload:
        words = [
            TranscriptWordPayload(id="w1", text="right", start_sec=0.00, end_sec=0.18),
            TranscriptWordPayload(id="w2", text="now", start_sec=0.18, end_sec=0.33),
            TranscriptWordPayload(id="w3", text="you", start_sec=0.33, end_sec=0.46),
            TranscriptWordPayload(id="w4", text="know", start_sec=0.46, end_sec=0.58),
            TranscriptWordPayload(id="w5", text="um", start_sec=0.62, end_sec=0.74),
            TranscriptWordPayload(id="w6", text="move", start_sec=0.74, end_sec=0.92),
        ]
        return TranscriptPayload(
            source="test",
            language="en",
            text="right now you know um move",
            words=words,
            is_mock=False,
        )

    monkeypatch.setattr("app.routers.vibe.generate_transcript", fake_generate)

    with TestClient(app) as client:
        project_id = _create_project(client)
        asset_id = _upload_video(client, project_id)
        response = client.post(
            f"/api/v1/vibe/apply?project_id={project_id}",
            json={"action": "auto_cut_pauses", "asset_id": asset_id},
        )
        assert response.status_code == 200
        payload = response.json()
        assert "Removed 2 filler word(s)." in (payload.get("details") or "")


def test_vibe_auto_cut_pauses_respects_string_false_for_remove_fillers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setattr(
        "app.routers.vibe.detect_silence_ranges", lambda *_args, **_kwargs: []
    )

    def fake_generate(_path: str, _duration_sec: float) -> TranscriptPayload:
        words = [
            TranscriptWordPayload(id="w1", text="um", start_sec=0.20, end_sec=0.34),
            TranscriptWordPayload(id="w2", text="hello", start_sec=0.34, end_sec=0.62),
        ]
        return TranscriptPayload(
            source="test", language="en", text="um hello", words=words, is_mock=False
        )

    monkeypatch.setattr("app.routers.vibe.generate_transcript", fake_generate)

    with TestClient(app) as client:
        project_id = _create_project(client)
        asset_id = _upload_video(client, project_id)
        response = client.post(
            f"/api/v1/vibe/apply?project_id={project_id}",
            json={
                "action": "auto_cut_pauses",
                "asset_id": asset_id,
                "options": {"remove_filler_words": "false"},
            },
        )
        assert response.status_code == 200
        payload = response.json()
        details = (payload.get("details") or "").lower()
        assert "filler word" not in details


def test_vibe_trim_start_end_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setattr(
        "app.routers.vibe.detect_silence_ranges",
        lambda *_args, **_kwargs: [(0.0, 0.5), (7.8, 8.0)],
    )

    def fake_generate(_path: str, _duration_sec: float) -> TranscriptPayload:
        words = [
            TranscriptWordPayload(id="w1", text="start", start_sec=0.7, end_sec=1.0),
            TranscriptWordPayload(id="w2", text="end", start_sec=7.2, end_sec=7.4),
        ]
        return TranscriptPayload(
            source="test", language="en", text="start end", words=words, is_mock=False
        )

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
        video_track = next(
            track for track in payload["timeline"]["tracks"] if track["kind"] == "video"
        )
        clip = video_track["clips"][0]
        assert clip["start_sec"] == pytest.approx(0.64, abs=0.01)
        assert clip["end_sec"] == pytest.approx(7.46, abs=0.01)
