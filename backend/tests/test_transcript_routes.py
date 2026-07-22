import os
import threading
from pathlib import Path

import pytest

pytest.importorskip("sqlmodel")

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")
# Disable pre-compute vocal isolation for test performance
os.environ.setdefault("TRANSCRIBE_VOCAL_ISOLATION_PRECOMPUTE", "false")

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.main import app
from app.database import engine
from app.models import MediaAsset
from app.routers import transcript as transcript_routes
from app.storage import storage
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


def _set_transcript_timing_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_TIMESTAMP_OFFSET_SEC", "0")
    monkeypatch.setenv("TRANSCRIPT_WEAK_REGION_RETRY_ENABLED", "false")


def test_transcript_generate_and_cut_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_transcript_timing_test_env(monkeypatch)
    monkeypatch.setenv("TRANSCRIPT_CUT_CONTEXT_SEC", "0")
    monkeypatch.setenv("TRANSCRIPT_CUT_MERGE_GAP_SEC", "0.08")
    monkeypatch.setenv("TRANSCRIPT_CUT_MIN_REMOVAL_SEC", "0")
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )

    seen_language: list[str | None] = []

    def fake_generate_transcript(
        _path: str,
        _duration_sec: float,
        *,
        language_hint: str | None = None,
    ) -> TranscriptPayload:
        seen_language.append(language_hint)
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

    monkeypatch.setattr(
        "app.routers.transcript.generate_transcript", fake_generate_transcript
    )

    with TestClient(app) as client:
        project_id = _create_project(client)
        asset_id = _upload_video(client, project_id)

        generate_res = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id, "language": "kn"},
        )
        assert generate_res.status_code == 200
        assert seen_language == ["kn"]
        generated = generate_res.json()
        transcript = generated["transcript"]
        original_words = transcript["words"]
        original_timeline = generated["timeline"]
        assert transcript["source"] == "test_provider"
        assert len(transcript["words"]) == 4
        assert transcript["quality_label"] in {"trusted", "needs_review"}
        assert 0.0 <= transcript["quality_score"] <= 1.0
        transcript_id = transcript["id"]

        cut_res = client.post(
            f"/api/v1/transcript/cut?project_id={project_id}",
            json={"transcript_id": transcript_id, "kept_word_ids": ["w1", "w4"]},
        )
        assert cut_res.status_code == 200
        cut_payload = cut_res.json()
        assert cut_payload["kept_word_count"] == 2
        assert cut_payload["removed_word_count"] == 2
        persisted_transcript = client.get(
            f"/api/v1/transcript?project_id={project_id}&transcript_id={transcript_id}"
        )
        assert persisted_transcript.status_code == 200
        assert [word["text"] for word in persisted_transcript.json()["words"]] == [
            "hello",
            "world",
        ]

        video_track = next(
            track
            for track in cut_payload["timeline"]["tracks"]
            if track["kind"] == "video"
        )
        assert len(video_track["clips"]) == 2
        assert video_track["clips"][0]["start_sec"] == 0.0
        assert video_track["clips"][0]["end_sec"] == pytest.approx(1.0, abs=0.02)
        assert video_track["clips"][1]["start_sec"] == 2.4
        assert video_track["clips"][1]["end_sec"] == 8.0

        restore_res = client.post(
            f"/api/v1/transcript/{transcript_id}/restore?project_id={project_id}",
            json={"words": original_words, "timeline": original_timeline},
        )
        assert restore_res.status_code == 200
        restore_payload = restore_res.json()
        assert [word["text"] for word in restore_payload["transcript"]["words"]] == [
            "hello",
            "brave",
            "new",
            "world",
        ]
        restored_video_track = next(
            track
            for track in restore_payload["timeline"]["tracks"]
            if track["kind"] == "video"
        )
        assert len(restored_video_track["clips"]) == 1
        assert restored_video_track["clips"][0]["start_sec"] == pytest.approx(0.0)
        assert restored_video_track["clips"][0]["end_sec"] == pytest.approx(8.0)


def test_transcript_cut_applies_context_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_transcript_timing_test_env(monkeypatch)
    monkeypatch.setenv("TRANSCRIPT_CUT_CONTEXT_SEC", "0.18")
    monkeypatch.setenv("TRANSCRIPT_CUT_MERGE_GAP_SEC", "0.08")
    monkeypatch.setenv("TRANSCRIPT_CUT_MIN_REMOVAL_SEC", "0.35")
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )

    def fake_generate_transcript(
        _path: str,
        _duration_sec: float,
        *,
        language_hint: str | None = None,
    ) -> TranscriptPayload:
        del language_hint
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

    monkeypatch.setattr(
        "app.routers.transcript.generate_transcript", fake_generate_transcript
    )

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
        video_track = next(
            track
            for track in cut_payload["timeline"]["tracks"]
            if track["kind"] == "video"
        )
        assert len(video_track["clips"]) == 2
        assert video_track["clips"][0]["start_sec"] == pytest.approx(0.0, abs=0.001)
        assert video_track["clips"][0]["end_sec"] == pytest.approx(1.0, abs=0.02)
        assert video_track["clips"][1]["start_sec"] == pytest.approx(2.4, abs=0.01)
        assert video_track["clips"][1]["end_sec"] == pytest.approx(8.0, abs=0.001)


def test_transcript_cut_preserves_unedited_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_transcript_timing_test_env(monkeypatch)
    monkeypatch.setenv("TRANSCRIPT_CUT_CONTEXT_SEC", "0")
    monkeypatch.setenv("TRANSCRIPT_CUT_MERGE_GAP_SEC", "0.08")
    monkeypatch.setenv("TRANSCRIPT_CUT_MIN_REMOVAL_SEC", "0")
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 40.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )

    def fake_generate_transcript(
        _path: str,
        _duration_sec: float,
        *,
        language_hint: str | None = None,
    ) -> TranscriptPayload:
        del language_hint
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

    monkeypatch.setattr(
        "app.routers.transcript.generate_transcript", fake_generate_transcript
    )

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
        video_track = next(
            track
            for track in cut_payload["timeline"]["tracks"]
            if track["kind"] == "video"
        )
        assert len(video_track["clips"]) == 2
        assert video_track["clips"][0]["start_sec"] == pytest.approx(0.0, abs=0.001)
        assert video_track["clips"][0]["end_sec"] == pytest.approx(1.0, abs=0.001)
        assert video_track["clips"][1]["start_sec"] == pytest.approx(2.5, abs=0.001)
        assert video_track["clips"][1]["end_sec"] == pytest.approx(40.0, abs=0.001)


def test_transcript_cut_first_word_preserves_leading_gap_and_resyncs_captions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_transcript_timing_test_env(monkeypatch)
    monkeypatch.setenv("TRANSCRIPT_CUT_CONTEXT_SEC", "0")
    monkeypatch.setenv("TRANSCRIPT_CUT_MERGE_GAP_SEC", "0.08")
    monkeypatch.setenv("TRANSCRIPT_CUT_MIN_REMOVAL_SEC", "0")
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )

    def fake_generate_transcript(
        _path: str,
        _duration_sec: float,
        *,
        language_hint: str | None = None,
    ) -> TranscriptPayload:
        del language_hint
        words = [
            TranscriptWordPayload(id="w1", text="hello", start_sec=0.5, end_sec=1.0),
            TranscriptWordPayload(id="w2", text="world", start_sec=1.0, end_sec=1.3),
            TranscriptWordPayload(id="w3", text="again", start_sec=2.0, end_sec=2.4),
        ]
        return TranscriptPayload(
            source="test_provider",
            language="en",
            text="hello world again",
            words=words,
            is_mock=False,
        )

    monkeypatch.setattr(
        "app.routers.transcript.generate_transcript", fake_generate_transcript
    )
    monkeypatch.setattr(
        "app.routers.vibe.generate_transcript", fake_generate_transcript
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="Transcript First Word Cut Test")
        asset_id = _upload_video(client, project_id)

        generated = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id},
        )
        assert generated.status_code == 200
        transcript_id = generated.json()["transcript"]["id"]

        add_caption_res = client.post(
            f"/api/v1/vibe/apply?project_id={project_id}",
            json={
                "action": "add_subtitles",
                "asset_id": asset_id,
                "options": {"style": "static"},
            },
        )
        assert add_caption_res.status_code == 200

        cut_res = client.post(
            f"/api/v1/transcript/cut?project_id={project_id}",
            json={"transcript_id": transcript_id, "kept_word_ids": ["w2", "w3"]},
        )
        assert cut_res.status_code == 200
        payload = cut_res.json()

        video_track = next(
            track for track in payload["timeline"]["tracks"] if track["kind"] == "video"
        )
        assert len(video_track["clips"]) == 2
        assert video_track["clips"][0]["start_sec"] == pytest.approx(0.0, abs=0.001)
        assert video_track["clips"][0]["end_sec"] == pytest.approx(0.5, abs=0.001)
        assert video_track["clips"][1]["start_sec"] == pytest.approx(1.0, abs=0.001)

        first_clip_overlays = video_track["clips"][0]["text_overlays"]
        second_clip_overlays = video_track["clips"][1]["text_overlays"]
        assert first_clip_overlays == []
        assert second_clip_overlays
        assert "hello" not in second_clip_overlays[0]["text"].lower()
        assert "world" in second_clip_overlays[0]["text"].lower()


def test_transcript_allows_videos_over_60_seconds_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 75.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setattr(
        "app.routers.transcript.generate_transcript",
        lambda _path, _duration_sec, **_kwargs: TranscriptPayload(
            source="test_provider",
            language="en",
            text="hello world",
            words=[
                TranscriptWordPayload(
                    id="w1", text="hello", start_sec=0.0, end_sec=0.4
                ),
                TranscriptWordPayload(
                    id="w2", text="world", start_sec=0.4, end_sec=0.8
                ),
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


def test_update_word_text_resyncs_existing_captions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_transcript_timing_test_env(monkeypatch)
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )

    def fake_generate_transcript(
        _path: str,
        _duration_sec: float,
        *,
        language_hint: str | None = None,
    ) -> TranscriptPayload:
        del language_hint
        words = [
            TranscriptWordPayload(id="w1", text="hello", start_sec=0.2, end_sec=0.45),
            TranscriptWordPayload(id="w2", text="world", start_sec=0.45, end_sec=0.7),
        ]
        return TranscriptPayload(
            source="test_provider",
            language="en",
            text="hello world",
            words=words,
            is_mock=False,
        )

    monkeypatch.setattr(
        "app.routers.transcript.generate_transcript", fake_generate_transcript
    )
    monkeypatch.setattr(
        "app.routers.vibe.generate_transcript", fake_generate_transcript
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="Caption Sync Test")
        asset_id = _upload_video(client, project_id)

        generated = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id},
        )
        assert generated.status_code == 200
        transcript_id = generated.json()["transcript"]["id"]

        add_caption_res = client.post(
            f"/api/v1/vibe/apply?project_id={project_id}",
            json={
                "action": "add_subtitles",
                "asset_id": asset_id,
                "options": {"style": "static"},
            },
        )
        assert add_caption_res.status_code == 200

        update_res = client.patch(
            f"/api/v1/transcript/{transcript_id}/words/w1?project_id={project_id}",
            json={"text": "namaskara"},
        )
        assert update_res.status_code == 200
        payload = update_res.json()
        assert payload["captions_synced"] is True
        assert payload["transcript"]["words"][0]["text"] == "namaskara"
        video_track = next(
            track for track in payload["timeline"]["tracks"] if track["kind"] == "video"
        )
        overlays = video_track["clips"][0]["text_overlays"]
        assert overlays
        assert "namaskara" in overlays[0]["text"]
        assert "hello" not in overlays[0]["text"]


def test_update_transcript_range_blanks_text_without_cutting_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_transcript_timing_test_env(monkeypatch)
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )

    def fake_generate_transcript(
        _path: str,
        _duration_sec: float,
        *,
        language_hint: str | None = None,
    ) -> TranscriptPayload:
        del language_hint
        words = [
            TranscriptWordPayload(id="w1", text="hello", start_sec=0.2, end_sec=0.45),
            TranscriptWordPayload(id="w2", text="wrong", start_sec=0.45, end_sec=0.7),
            TranscriptWordPayload(id="w3", text="line", start_sec=0.7, end_sec=0.95),
        ]
        return TranscriptPayload(
            source="test_provider",
            language="en",
            text="hello wrong line",
            words=words,
            is_mock=False,
        )

    monkeypatch.setattr(
        "app.routers.transcript.generate_transcript", fake_generate_transcript
    )
    monkeypatch.setattr(
        "app.routers.vibe.generate_transcript", fake_generate_transcript
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="Transcript Range Edit Test")
        asset_id = _upload_video(client, project_id)

        generated = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id},
        )
        assert generated.status_code == 200
        transcript_id = generated.json()["transcript"]["id"]

        add_caption_res = client.post(
            f"/api/v1/vibe/apply?project_id={project_id}",
            json={
                "action": "add_subtitles",
                "asset_id": asset_id,
                "options": {"style": "static"},
            },
        )
        assert add_caption_res.status_code == 200

        update_res = client.patch(
            f"/api/v1/transcript/{transcript_id}/range?project_id={project_id}",
            json={"start_word_id": "w2", "end_word_id": "w3", "mode": "blank"},
        )
        assert update_res.status_code == 200
        payload = update_res.json()
        assert payload["captions_synced"] is True
        assert [word["text"] for word in payload["transcript"]["words"]] == ["hello"]
        assert any(
            region["status"] == "blanked" for region in payload["transcript"]["regions"]
        )
        video_track = next(
            track for track in payload["timeline"]["tracks"] if track["kind"] == "video"
        )
        assert len(video_track["clips"]) == 1
        assert video_track["clips"][0]["start_sec"] == pytest.approx(0.0, abs=0.001)
        assert video_track["clips"][0]["end_sec"] == pytest.approx(8.0, abs=0.001)


def test_update_transcript_range_deletes_text_and_resyncs_captions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_transcript_timing_test_env(monkeypatch)
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )

    def fake_generate_transcript(
        _path: str,
        _duration_sec: float,
        *,
        language_hint: str | None = None,
    ) -> TranscriptPayload:
        del language_hint
        words = [
            TranscriptWordPayload(id="w1", text="gangsta", start_sec=0.2, end_sec=0.45),
            TranscriptWordPayload(
                id="w2", text="paradise", start_sec=0.45, end_sec=0.8
            ),
            TranscriptWordPayload(id="w3", text="forever", start_sec=0.8, end_sec=1.2),
        ]
        return TranscriptPayload(
            source="test_provider",
            language="en",
            text="gangsta paradise forever",
            words=words,
            is_mock=False,
        )

    monkeypatch.setattr(
        "app.routers.transcript.generate_transcript", fake_generate_transcript
    )
    monkeypatch.setattr(
        "app.routers.vibe.generate_transcript", fake_generate_transcript
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="Transcript Delete Range Test")
        asset_id = _upload_video(client, project_id)

        generated = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id},
        )
        assert generated.status_code == 200
        transcript_id = generated.json()["transcript"]["id"]

        add_caption_res = client.post(
            f"/api/v1/vibe/apply?project_id={project_id}",
            json={
                "action": "add_subtitles",
                "asset_id": asset_id,
                "options": {"style": "static"},
            },
        )
        assert add_caption_res.status_code == 200

        update_res = client.patch(
            f"/api/v1/transcript/{transcript_id}/range?project_id={project_id}",
            json={"start_word_id": "w2", "end_word_id": "w2", "mode": "delete"},
        )
        assert update_res.status_code == 200
        payload = update_res.json()
        assert payload["captions_synced"] is True
        assert [word["text"] for word in payload["transcript"]["words"]] == [
            "gangsta",
            "forever",
        ]
        assert not any(
            region["status"] == "blanked" for region in payload["transcript"]["regions"]
        )
        video_track = next(
            track for track in payload["timeline"]["tracks"] if track["kind"] == "video"
        )
        overlays = video_track["clips"][0]["text_overlays"]
        assert overlays
        overlay_text = " ".join(item["text"] for item in overlays)
        assert "paradise" not in overlay_text


def test_repeated_transcript_deletes_do_not_reintroduce_prior_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_transcript_timing_test_env(monkeypatch)
    monkeypatch.setenv("TRANSCRIPT_CUT_CONTEXT_SEC", "0")
    monkeypatch.setenv("TRANSCRIPT_CUT_MERGE_GAP_SEC", "0.08")
    monkeypatch.setenv("TRANSCRIPT_CUT_MIN_REMOVAL_SEC", "0")
    monkeypatch.setenv("TRANSCRIPT_CUT_ASR_HEAD_PAD_SEC", "0")
    monkeypatch.setenv("TRANSCRIPT_CUT_ASR_TAIL_PAD_SEC", "0")
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )

    def fake_generate_transcript(
        _path: str,
        _duration_sec: float,
        *,
        language_hint: str | None = None,
    ) -> TranscriptPayload:
        del language_hint
        words = [
            TranscriptWordPayload(id="w1", text="one", start_sec=0.2, end_sec=0.45),
            TranscriptWordPayload(id="w2", text="two", start_sec=0.45, end_sec=0.8),
            TranscriptWordPayload(id="w3", text="three", start_sec=2.0, end_sec=2.4),
            TranscriptWordPayload(id="w4", text="four", start_sec=3.0, end_sec=3.4),
        ]
        return TranscriptPayload(
            source="test_provider",
            language="en",
            text="one two three four",
            words=words,
            is_mock=False,
        )

    monkeypatch.setattr(
        "app.routers.transcript.generate_transcript", fake_generate_transcript
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="Transcript Repeated Delete Test")
        asset_id = _upload_video(client, project_id)

        generated = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id},
        )
        assert generated.status_code == 200
        transcript_id = generated.json()["transcript"]["id"]

        first_delete = client.patch(
            f"/api/v1/transcript/{transcript_id}/range?project_id={project_id}",
            json={"start_word_id": "w2", "end_word_id": "w2", "mode": "delete"},
        )
        assert first_delete.status_code == 200
        second_delete = client.patch(
            f"/api/v1/transcript/{transcript_id}/range?project_id={project_id}",
            json={"start_word_id": "w3", "end_word_id": "w3", "mode": "delete"},
        )
        assert second_delete.status_code == 200
        payload = second_delete.json()
        assert [word["text"] for word in payload["transcript"]["words"]] == [
            "one",
            "four",
        ]

        video_track = next(
            track for track in payload["timeline"]["tracks"] if track["kind"] == "video"
        )
        source_ranges = [
            (clip["start_sec"], clip["end_sec"]) for clip in video_track["clips"]
        ]
        assert len(source_ranges) == 2
        assert source_ranges[0] == pytest.approx((0.0, 0.45), abs=0.02)
        assert source_ranges[1] == pytest.approx((3.0, 8.0), abs=0.02)


def test_repeated_transcript_cuts_do_not_reintroduce_prior_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_transcript_timing_test_env(monkeypatch)
    monkeypatch.setenv("TRANSCRIPT_CUT_CONTEXT_SEC", "0")
    monkeypatch.setenv("TRANSCRIPT_CUT_MERGE_GAP_SEC", "0.08")
    monkeypatch.setenv("TRANSCRIPT_CUT_MIN_REMOVAL_SEC", "0")
    monkeypatch.setenv("TRANSCRIPT_CUT_ASR_HEAD_PAD_SEC", "0")
    monkeypatch.setenv("TRANSCRIPT_CUT_ASR_TAIL_PAD_SEC", "0")
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )

    def fake_generate_transcript(
        _path: str,
        _duration_sec: float,
        *,
        language_hint: str | None = None,
    ) -> TranscriptPayload:
        del language_hint
        words = [
            TranscriptWordPayload(id="w1", text="one", start_sec=0.2, end_sec=0.45),
            TranscriptWordPayload(id="w2", text="two", start_sec=0.45, end_sec=0.8),
            TranscriptWordPayload(id="w3", text="three", start_sec=2.0, end_sec=2.4),
            TranscriptWordPayload(id="w4", text="four", start_sec=3.0, end_sec=3.4),
        ]
        return TranscriptPayload(
            source="test_provider",
            language="en",
            text="one two three four",
            words=words,
            is_mock=False,
        )

    monkeypatch.setattr(
        "app.routers.transcript.generate_transcript", fake_generate_transcript
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="Transcript Repeated Cut Test")
        asset_id = _upload_video(client, project_id)

        generated = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id},
        )
        assert generated.status_code == 200
        transcript_id = generated.json()["transcript"]["id"]

        first_cut = client.post(
            f"/api/v1/transcript/cut?project_id={project_id}",
            json={"transcript_id": transcript_id, "kept_word_ids": ["w1", "w3", "w4"]},
        )
        assert first_cut.status_code == 200
        second_cut = client.post(
            f"/api/v1/transcript/cut?project_id={project_id}",
            json={"transcript_id": transcript_id, "kept_word_ids": ["w1", "w4"]},
        )
        assert second_cut.status_code == 200

        video_track = next(
            track for track in second_cut.json()["timeline"]["tracks"] if track["kind"] == "video"
        )
        source_ranges = [
            (clip["start_sec"], clip["end_sec"]) for clip in video_track["clips"]
        ]
        assert len(source_ranges) == 2
        assert source_ranges[0] == pytest.approx((0.0, 0.45), abs=0.02)
        assert source_ranges[1] == pytest.approx((3.0, 8.0), abs=0.02)


def test_transcript_response_includes_romanized_display_text_for_indic_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_transcript_timing_test_env(monkeypatch)
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )

    def fake_generate_transcript(
        _path: str,
        _duration_sec: float,
        *,
        language_hint: str | None = None,
    ) -> TranscriptPayload:
        del language_hint
        words = [
            TranscriptWordPayload(id="w1", text="ನಮಸ್ಕಾರ", start_sec=0.2, end_sec=0.55),
            TranscriptWordPayload(id="w2", text="ಲೋಕ", start_sec=0.55, end_sec=0.9),
        ]
        return TranscriptPayload(
            source="test_provider",
            language="kn",
            text="ನಮಸ್ಕಾರ ಲೋಕ",
            words=words,
            is_mock=False,
        )

    monkeypatch.setattr(
        "app.routers.transcript.generate_transcript", fake_generate_transcript
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="Transcript Romanized Display Test")
        asset_id = _upload_video(client, project_id)

        generated = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id, "language": "kn"},
        )
        assert generated.status_code == 200
        words = generated.json()["transcript"]["words"]
        assert words[0]["text"] == "ನಮಸ್ಕಾರ"
        assert isinstance(words[0].get("display_text"), str)
        assert words[0]["display_text"]
        assert words[0]["display_text"] != words[0]["text"]


def test_lyrics_reference_filename_hint_recovers_descriptive_alias_for_opaque_upload() -> (
    None
):
    opaque_filename = "3ab0d2cd-eead-4c27-af35-6d98b7dba7c7.mp4"

    with TestClient(app) as client:
        project_id = _create_project(client, name="Opaque Upload Alias Test")
        response = client.post(
            "/api/v1/media/upload",
            data={"project_id": project_id},
            files={"file": (opaque_filename, b"fake-video-bytes", "video/mp4")},
        )
        assert response.status_code == 200
        asset_id = response.json()["id"]

        with Session(engine) as session:
            asset = session.get(MediaAsset, asset_id)
            assert asset is not None
            session.add(
                MediaAsset(
                    project_id=project_id,
                    media_type="video",
                    filename="Coolio - Gangsta's Paradise (Official Music Video).mp4",
                    storage_path=f"library/{opaque_filename}",
                    mime_type="video/mp4",
                    duration_sec=256.0,
                    metadata_json="{}",
                )
            )
            session.commit()
            session.refresh(asset)

            resolved = transcript_routes._lyrics_reference_filename_hint(session, asset)

        assert resolved == "Coolio - Gangsta's Paradise (Official Music Video).mp4"


def test_transcript_uses_ingested_source_title_for_lyrics_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setenv("TRANSCRIBE_REUSE_EXISTING_ON_GENERATE", "false")
    monkeypatch.setenv("TRANSCRIBE_TIMESTAMP_REFINEMENT_ENABLED", "false")
    monkeypatch.setenv("TRANSCRIPT_WEAK_REGION_RETRY_ENABLED", "false")

    def fake_chunked(*_args, **_kwargs) -> TranscriptPayload:
        return TranscriptPayload(
            source="test_provider",
            language="en",
            text="kesariya tera ishq hai piya",
            words=[
                TranscriptWordPayload(
                    id="w1", text="kesariya", start_sec=0.0, end_sec=0.4
                ),
                TranscriptWordPayload(
                    id="w2", text="tera", start_sec=0.4, end_sec=0.8
                ),
            ],
            is_mock=False,
        )

    lyrics_hints: list[str] = []

    def fake_lyrics_lookup(
        transcript: TranscriptPayload, *, filename: str, duration_sec: float
    ) -> TranscriptPayload:
        del duration_sec
        lyrics_hints.append(filename)
        return transcript

    monkeypatch.setattr(
        "app.routers.transcript._generate_transcript_payload_chunked", fake_chunked
    )
    monkeypatch.setattr(
        "app.routers.transcript.maybe_apply_reference_lyrics", fake_lyrics_lookup
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="URL Ingest Lyrics Title")
        asset_id = _upload_video(client, project_id)
        with Session(engine) as session:
            asset = session.get(MediaAsset, asset_id)
            assert asset is not None
            asset.filename = "1a2b3c4d5e6f.mp4"
            asset.metadata_json = '{"source_title": "Kesariya"}'
            session.add(asset)
            session.commit()

        response = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id, "mode": "auto"},
        )

    assert response.status_code == 200
    assert lyrics_hints == ["Kesariya"]


def test_related_library_transcript_prefers_lyrics_ref_match_for_opaque_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> (
    None
):
    monkeypatch.setenv("TRANSCRIBE_REUSE_CROSS_LIBRARY", "true")
    opaque_filename = "3ab0d2cd-eead-4c27-af35-6d98b7dba7c7.mp4"

    with TestClient(app) as client:
        project_id = _create_project(client, name="Opaque Upload Transcript Reuse Test")
        response = client.post(
            "/api/v1/media/upload",
            data={"project_id": project_id},
            files={"file": (opaque_filename, b"fake-video-bytes", "video/mp4")},
        )
        assert response.status_code == 200
        asset_id = response.json()["id"]

        with Session(engine) as session:
            asset = session.get(MediaAsset, asset_id)
            assert asset is not None

            related_asset = MediaAsset(
                project_id=project_id,
                media_type="video",
                filename="Coolio - Gangsta's Paradise (Official Music Video).mp4",
                storage_path=f"library/{opaque_filename}",
                mime_type="video/mp4",
                duration_sec=asset.duration_sec,
                metadata_json="{}",
            )
            session.add(related_asset)
            session.commit()
            session.refresh(related_asset)

            plain = transcript_routes._store_transcript_items(
                session,
                project_id=project_id,
                asset_id=related_asset.id,
                duration_sec=float(asset.duration_sec or 8.0),
                source="chunked:groq_gapfill_gapfill",
                language="English",
                is_mock=False,
                items=[
                    {"id": "plain-1", "text": "bad", "start_sec": 0.0, "end_sec": 0.4},
                ],
            )
            preferred = transcript_routes._store_transcript_items(
                session,
                project_id=project_id,
                asset_id=related_asset.id,
                duration_sec=float(asset.duration_sec or 8.0),
                source="chunked:groq_gapfill_gapfill_lyrics_ref",
                language="English",
                is_mock=False,
                items=[
                    {"id": "lyrics-1", "text": "as", "start_sec": 0.0, "end_sec": 0.4},
                    {
                        "id": "lyrics-2",
                        "text": "walk",
                        "start_sec": 0.4,
                        "end_sec": 0.8,
                    },
                ],
            )

            reusable = transcript_routes._related_library_transcript(
                session,
                asset,
                requested_language=None,
            )
            plain_id = plain.id
            preferred_id = preferred.id

        assert plain_id != preferred_id
        assert reusable is not None
        assert reusable.id == preferred_id


def test_lyrics_reference_filename_hint_matches_identical_opaque_copy_by_content() -> (
    None
):
    opaque_filename = "d43eff19-7d7f-4740-839b-ef2455c224da.mp4"
    upload_bytes = b"gangsta-paradise-demo-video"

    with TestClient(app) as client:
        project_id = _create_project(client, name="Opaque Upload Content Alias Test")
        response = client.post(
            "/api/v1/media/upload",
            data={"project_id": project_id},
            files={"file": (opaque_filename, upload_bytes, "video/mp4")},
        )
        assert response.status_code == 200
        asset_id = response.json()["id"]

        with Session(engine) as session:
            asset = session.get(MediaAsset, asset_id)
            assert asset is not None

            related_storage = "library/coolio-original-copy.mp4"
            related_path = Path(storage.resolve_upload_asset(related_storage))
            related_path.parent.mkdir(parents=True, exist_ok=True)
            related_path.write_bytes(upload_bytes)

            session.add(
                MediaAsset(
                    project_id=project_id,
                    media_type="video",
                    filename="Coolio - Gangsta's Paradise (Official Music Video).mp4",
                    storage_path=related_storage,
                    mime_type="video/mp4",
                    duration_sec=asset.duration_sec,
                    metadata_json="{}",
                )
            )
            session.commit()
            session.refresh(asset)

            resolved = transcript_routes._lyrics_reference_filename_hint(session, asset)

        assert resolved == "Coolio - Gangsta's Paradise (Official Music Video).mp4"


def test_related_library_transcript_matches_identical_opaque_copy_by_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_REUSE_CROSS_LIBRARY", "true")
    opaque_filename = "d43eff19-7d7f-4740-839b-ef2455c224da.mp4"
    upload_bytes = b"gangsta-paradise-demo-video"

    with TestClient(app) as client:
        project_id = _create_project(
            client, name="Opaque Upload Content Transcript Reuse Test"
        )
        response = client.post(
            "/api/v1/media/upload",
            data={"project_id": project_id},
            files={"file": (opaque_filename, upload_bytes, "video/mp4")},
        )
        assert response.status_code == 200
        asset_id = response.json()["id"]

        with Session(engine) as session:
            asset = session.get(MediaAsset, asset_id)
            assert asset is not None

            related_storage = "library/coolio-content-match.mp4"
            related_path = Path(storage.resolve_upload_asset(related_storage))
            related_path.parent.mkdir(parents=True, exist_ok=True)
            related_path.write_bytes(upload_bytes)

            related_asset = MediaAsset(
                project_id=f"{project_id}-library",
                media_type="video",
                filename="Coolio - Gangsta's Paradise (Official Music Video).mp4",
                storage_path=related_storage,
                mime_type="video/mp4",
                duration_sec=asset.duration_sec,
                metadata_json="{}",
            )
            session.add(related_asset)
            session.commit()
            session.refresh(related_asset)

            preferred = transcript_routes._store_transcript_items(
                session,
                project_id=related_asset.project_id,
                asset_id=related_asset.id,
                duration_sec=float(asset.duration_sec or 8.0),
                source="chunked:groq_gapfill_lyrics_ref",
                language="English",
                is_mock=False,
                items=[
                    {"id": "lyrics-1", "text": "as", "start_sec": 0.0, "end_sec": 0.4},
                    {
                        "id": "lyrics-2",
                        "text": "walk",
                        "start_sec": 0.4,
                        "end_sec": 0.8,
                    },
                ],
            )

            reusable = transcript_routes._related_library_transcript(
                session,
                asset,
                requested_language=None,
            )

        assert reusable is not None
        assert reusable.id == preferred.id


def test_related_library_transcript_skips_user_edited_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIBE_REUSE_CROSS_LIBRARY", "true")
    opaque_filename = "c2e29ee0-4f97-4af5-9f83-9f4c2fbfb51d.mp4"

    with TestClient(app) as client:
        project_id = _create_project(client, name="Transcript Reuse Edited Candidate")
        response = client.post(
            "/api/v1/media/upload",
            data={"project_id": project_id},
            files={"file": (opaque_filename, b"fake-video-bytes", "video/mp4")},
        )
        assert response.status_code == 200
        asset_id = response.json()["id"]

        with Session(engine) as session:
            asset = session.get(MediaAsset, asset_id)
            assert asset is not None

            related_asset = MediaAsset(
                project_id=f"{project_id}-library",
                media_type="video",
                filename="Coolio - Gangsta's Paradise (Official Music Video).mp4",
                storage_path=f"library/{opaque_filename}",
                mime_type="video/mp4",
                duration_sec=asset.duration_sec,
                metadata_json="{}",
            )
            session.add(related_asset)
            session.commit()
            session.refresh(related_asset)

            pristine = transcript_routes._store_transcript_items(
                session,
                project_id=related_asset.project_id,
                asset_id=related_asset.id,
                duration_sec=float(asset.duration_sec or 8.0),
                source="chunked:groq_gapfill_lyrics_ref",
                language="English",
                is_mock=False,
                items=[
                    {"id": "lyrics-1", "text": "as", "start_sec": 0.0, "end_sec": 0.4},
                    {"id": "lyrics-2", "text": "walk", "start_sec": 0.4, "end_sec": 0.8},
                ],
            )
            edited = transcript_routes._store_transcript_items(
                session,
                project_id=related_asset.project_id,
                asset_id=related_asset.id,
                duration_sec=float(asset.duration_sec or 8.0),
                source="chunked:groq_gapfill_lyrics_ref",
                language="English",
                is_mock=False,
                items=[
                    {"id": "edit-1", "text": "as", "start_sec": 0.0, "end_sec": 0.4},
                    {"id": "edit-2", "text": "walk", "start_sec": 0.4, "end_sec": 0.8},
                ],
            )
            transcript_routes._persist_transcript_items(
                edited,
                session=session,
                items=[
                    {
                        "id": "edit-1",
                        "text": "as",
                        "start_sec": 0.0,
                        "end_sec": 0.4,
                    },
                    {
                        "id": "edit-2",
                        "text": "walked",
                        "start_sec": 0.4,
                        "end_sec": 0.8,
                        "source_pass": "manual",
                    },
                ],
            )

            reusable = transcript_routes._related_library_transcript(
                session,
                asset,
                requested_language=None,
            )

        assert reusable is not None
        assert reusable.id == pristine.id


def test_transcript_generate_retries_fast_mode_after_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setenv("TRANSCRIBE_REUSE_EXISTING_ON_GENERATE", "false")
    monkeypatch.setenv("TRANSCRIBE_FAST_MODE", "false")

    seen_fast_mode: list[bool] = []

    def fake_chunked(
        _source_path: str,
        _duration_sec: float,
        *,
        language_hint: str | None,
        fast_mode: bool,
        prompt: str | None,
        progress_callback=None,
    ) -> TranscriptPayload:
        del language_hint, prompt, progress_callback
        seen_fast_mode.append(fast_mode)
        if len(seen_fast_mode) == 1:
            raise RuntimeError("ffmpeg died with signal 9")
        return TranscriptPayload(
            source="test_provider",
            language="en",
            text="hello world",
            words=[
                TranscriptWordPayload(
                    id="w1", text="hello", start_sec=0.0, end_sec=0.4
                ),
                TranscriptWordPayload(
                    id="w2", text="world", start_sec=0.4, end_sec=0.8
                ),
            ],
            is_mock=False,
        )

    monkeypatch.setattr(
        "app.routers.transcript._generate_transcript_payload_chunked", fake_chunked
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="Transcript SIGKILL Fast Retry")
        asset_id = _upload_video(client, project_id)
        response = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id},
        )
        assert response.status_code == 200
        assert seen_fast_mode == [False, True]


def test_transcript_generate_bypasses_chunking_for_short_clips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 120.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setenv("TRANSCRIBE_REUSE_EXISTING_ON_GENERATE", "false")
    monkeypatch.setenv("TRANSCRIBE_CHUNK_BYPASS_MAX_DURATION_SEC", "300")

    seen_calls: list[tuple[str | None, bool, str | None]] = []

    def fake_generate(
        _source_path: str,
        _duration_sec: float,
        *,
        language_hint: str | None,
        allow_mock_fallback: bool,
        fast_mode: bool,
        prompt: str | None,
    ) -> TranscriptPayload:
        seen_calls.append((language_hint, fast_mode, prompt))
        return TranscriptPayload(
            source="groq",
            language="en",
            text="gangstas paradise",
            words=[
                TranscriptWordPayload(
                    id="w1", text="gangstas", start_sec=0.0, end_sec=0.4
                ),
                TranscriptWordPayload(
                    id="w2", text="paradise", start_sec=0.4, end_sec=0.8
                ),
            ],
            is_mock=False,
        )

    def fail_if_chunked(*_args, **_kwargs) -> TranscriptPayload:
        raise AssertionError("chunk extraction path should be bypassed for short clips")

    monkeypatch.setattr(
        "app.routers.transcript._call_generate_transcript_compatible", fake_generate
    )
    monkeypatch.setattr("app.routers.transcript._extract_audio_chunk", fail_if_chunked)

    with TestClient(app) as client:
        project_id = _create_project(client, name="Transcript Full File Bypass")
        asset_id = _upload_video(client, project_id)
        response = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id, "language": "en", "prompt": "lyric prompt"},
        )
        assert response.status_code == 200
        assert seen_calls == [("en", False, "lyric prompt")]
        assert response.json()["transcript"]["source"] == "groq"


def test_transcript_generate_auto_mode_uses_speech_path_for_song_filenames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 195.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setenv("TRANSCRIBE_REUSE_EXISTING_ON_GENERATE", "false")
    monkeypatch.setenv("TRANSCRIBE_TIMESTAMP_REFINEMENT_ENABLED", "false")

    seen_modes: list[str | None] = []

    def fake_chunked(
        _source_path: str,
        _duration_sec: float,
        *,
        language_hint: str | None,
        fast_mode: bool,
        prompt: str | None,
        progress_callback=None,
        translate_to_english: bool | None = None,
        mode: str | None = None,
        optimize_for_speed: bool | None = None,
        bypass_max_duration_sec_override: float | None = None,
        chunk_duration_sec_override: float | None = None,
        chunk_overlap_sec_override: float | None = None,
        chunk_parallelism_override: int | None = None,
    ) -> TranscriptPayload:
        del (
            _source_path,
            _duration_sec,
            language_hint,
            fast_mode,
            prompt,
            progress_callback,
            translate_to_english,
            optimize_for_speed,
            bypass_max_duration_sec_override,
            chunk_duration_sec_override,
            chunk_overlap_sec_override,
            chunk_parallelism_override,
        )
        seen_modes.append(mode)
        return TranscriptPayload(
            source="test_provider",
            language="en",
            text="hello world",
            words=[
                TranscriptWordPayload(
                    id="w1", text="hello", start_sec=0.0, end_sec=0.4
                ),
                TranscriptWordPayload(
                    id="w2", text="world", start_sec=0.4, end_sec=0.8
                ),
            ],
            is_mock=False,
        )

    monkeypatch.setattr(
        "app.routers.transcript._generate_transcript_payload_chunked", fake_chunked
    )
    monkeypatch.setattr(
        "app.routers.transcript.maybe_apply_reference_lyrics",
        lambda payload, *, filename, duration_sec: payload,
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="Song Mode Auto Resolve")
        asset_id = _upload_video(client, project_id)
        with Session(engine) as session:
            asset = session.get(MediaAsset, asset_id)
            assert asset is not None
            asset.filename = (
                "Googly_-_Bisilu_Kudreyondu_Full_Song_Video_Yash_Kriti_Kharbanda_720P.mp4"
            )
            session.add(asset)
            session.commit()

        response = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id, "mode": "auto"},
        )

    assert response.status_code == 200
    assert seen_modes == ["speech"]


def test_songlike_speech_strategy_skips_weak_region_retry() -> None:
    strategy = transcript_routes._resolve_transcript_generation_strategy(
        195.0,
        "speech",
        song_like_media=True,
    )

    assert strategy.mode == "speech"
    assert strategy.skip_weak_region_retry is True


def test_transcript_generate_song_auto_skips_stale_existing_transcript_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 195.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setenv("TRANSCRIBE_REUSE_EXISTING_ON_GENERATE", "true")
    monkeypatch.setenv("TRANSCRIBE_REUSE_CROSS_LIBRARY", "false")
    monkeypatch.setenv("TRANSCRIBE_TIMESTAMP_REFINEMENT_ENABLED", "false")

    seen_modes: list[str | None] = []

    def fake_chunked(
        _source_path: str,
        _duration_sec: float,
        *,
        language_hint: str | None,
        fast_mode: bool,
        prompt: str | None,
        progress_callback=None,
        translate_to_english: bool | None = None,
        mode: str | None = None,
        optimize_for_speed: bool | None = None,
        bypass_max_duration_sec_override: float | None = None,
        chunk_duration_sec_override: float | None = None,
        chunk_overlap_sec_override: float | None = None,
        chunk_parallelism_override: int | None = None,
    ) -> TranscriptPayload:
        del (
            _source_path,
            _duration_sec,
            language_hint,
            fast_mode,
            prompt,
            progress_callback,
            translate_to_english,
            optimize_for_speed,
            bypass_max_duration_sec_override,
            chunk_duration_sec_override,
            chunk_overlap_sec_override,
            chunk_parallelism_override,
        )
        seen_modes.append(mode)
        return TranscriptPayload(
            source="fresh_provider",
            language="en",
            text="as i walk",
            words=[
                TranscriptWordPayload(id="w1", text="as", start_sec=0.0, end_sec=0.3),
                TranscriptWordPayload(id="w2", text="i", start_sec=0.3, end_sec=0.5),
                TranscriptWordPayload(
                    id="w3", text="walk", start_sec=0.5, end_sec=0.9
                ),
            ],
            is_mock=False,
        )

    monkeypatch.setattr(
        "app.routers.transcript._generate_transcript_payload_chunked", fake_chunked
    )
    monkeypatch.setattr(
        "app.routers.transcript.maybe_apply_reference_lyrics",
        lambda payload, *, filename, duration_sec: payload,
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="Stale Song Transcript Reuse")
        asset_id = _upload_video(client, project_id)
        with Session(engine) as session:
            asset = session.get(MediaAsset, asset_id)
            assert asset is not None
            asset.filename = (
                "Coolio_-_Gangsta_s_Paradise_feat._L.V._Official_Music_Video_1080P.mp4"
            )
            session.add(asset)
            session.commit()

            transcript_routes._store_transcript_items(
                session,
                project_id=project_id,
                asset_id=asset_id,
                duration_sec=195.0,
                source="chunked:groq_gapfill",
                language="English",
                is_mock=False,
                items=[
                    {"id": "old-1", "text": "bad", "start_sec": 0.0, "end_sec": 0.4},
                    {"id": "old-2", "text": "transcript", "start_sec": 0.4, "end_sec": 0.9},
                ],
            )

        response = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id, "mode": "auto"},
        )

    assert response.status_code == 200
    assert seen_modes == ["speech"]
    assert response.json()["transcript"]["source"] == "fresh_provider"


def test_transcript_generate_defaults_to_fresh_transcript_when_existing_one_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 15.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.delenv("TRANSCRIBE_REUSE_EXISTING_ON_GENERATE", raising=False)
    monkeypatch.setenv("TRANSCRIBE_REUSE_CROSS_LIBRARY", "false")

    seen_calls = 0

    def fake_generate(
        _source_path: str,
        _duration_sec: float,
        *,
        language_hint: str | None,
        allow_mock_fallback: bool,
        fast_mode: bool,
        prompt: str | None,
    ) -> TranscriptPayload:
        nonlocal seen_calls
        del language_hint, allow_mock_fallback, fast_mode, prompt
        seen_calls += 1
        return TranscriptPayload(
            source="fresh_provider",
            language="en",
            text="fresh transcript",
            words=[
                TranscriptWordPayload(id="new-1", text="fresh", start_sec=0.0, end_sec=0.4),
                TranscriptWordPayload(
                    id="new-2", text="transcript", start_sec=0.4, end_sec=0.9
                ),
            ],
            is_mock=False,
        )

    monkeypatch.setattr(
        "app.routers.transcript._call_generate_transcript_compatible", fake_generate
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="Fresh Transcript Default")
        asset_id = _upload_video(client, project_id)

        with Session(engine) as session:
            transcript_routes._store_transcript_items(
                session,
                project_id=project_id,
                asset_id=asset_id,
                duration_sec=15.0,
                source="chunked:groq_gapfill",
                language="en",
                is_mock=False,
                items=[
                    {"id": "old-1", "text": "old", "start_sec": 0.0, "end_sec": 0.4},
                    {"id": "old-2", "text": "words", "start_sec": 0.4, "end_sec": 0.8},
                ],
            )

        response = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id, "language": "en"},
        )

    assert response.status_code == 200
    assert seen_calls == 1
    assert response.json()["transcript"]["source"] == "fresh_provider"


def test_transcript_generate_auto_language_ignores_cached_transcript_when_reuse_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 15.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setenv("TRANSCRIBE_REUSE_EXISTING_ON_GENERATE", "true")
    monkeypatch.delenv("TRANSCRIBE_REUSE_AUTO_LANGUAGE_EXISTING", raising=False)
    monkeypatch.setenv("TRANSCRIBE_REUSE_CROSS_LIBRARY", "false")

    seen_calls = 0

    def fake_generate(
        _source_path: str,
        _duration_sec: float,
        *,
        language_hint: str | None,
        allow_mock_fallback: bool,
        fast_mode: bool,
        prompt: str | None,
    ) -> TranscriptPayload:
        nonlocal seen_calls
        del language_hint, allow_mock_fallback, fast_mode, prompt
        seen_calls += 1
        return TranscriptPayload(
            source="fresh_sarvam",
            language="kn",
            text="fresh kannada",
            words=[
                TranscriptWordPayload(id="new-1", text="fresh", start_sec=0.0, end_sec=0.4),
                TranscriptWordPayload(
                    id="new-2", text="kannada", start_sec=0.4, end_sec=0.9
                ),
            ],
            is_mock=False,
        )

    monkeypatch.setattr(
        "app.routers.transcript._call_generate_transcript_compatible", fake_generate
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="Auto Ignores Cached Transcript")
        asset_id = _upload_video(client, project_id)

        with Session(engine) as session:
            transcript_routes._store_transcript_items(
                session,
                project_id=project_id,
                asset_id=asset_id,
                duration_sec=15.0,
                source="groq",
                language="ta",
                is_mock=False,
                items=[
                    {"id": "old-1", "text": "old", "start_sec": 0.0, "end_sec": 0.4},
                    {"id": "old-2", "text": "tamil", "start_sec": 0.4, "end_sec": 0.8},
                ],
            )

        response = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id, "language": "auto"},
        )

    assert response.status_code == 200
    assert seen_calls == 1
    assert response.json()["transcript"]["source"] == "fresh_sarvam"


def test_transcript_generate_explicit_language_can_reuse_cached_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 15.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setenv("TRANSCRIBE_REUSE_EXISTING_ON_GENERATE", "true")
    monkeypatch.setenv("TRANSCRIBE_REUSE_CROSS_LIBRARY", "false")

    seen_calls = 0

    def fake_generate(*_args, **_kwargs) -> TranscriptPayload:
        nonlocal seen_calls
        seen_calls += 1
        return TranscriptPayload(
            source="fresh_provider",
            language="kn",
            text="fresh",
            words=[TranscriptWordPayload(id="new", text="fresh", start_sec=0.0, end_sec=0.4)],
            is_mock=False,
        )

    monkeypatch.setattr(
        "app.routers.transcript._call_generate_transcript_compatible", fake_generate
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="Explicit Reuses Cached Transcript")
        asset_id = _upload_video(client, project_id)

        with Session(engine) as session:
            transcript_routes._store_transcript_items(
                session,
                project_id=project_id,
                asset_id=asset_id,
                duration_sec=15.0,
                source="sarvam",
                language="kn",
                is_mock=False,
                items=[
                    {"id": "old-1", "text": "ಹಳೆಯ", "start_sec": 0.0, "end_sec": 0.4},
                    {"id": "old-2", "text": "ಪಠ್ಯ", "start_sec": 0.4, "end_sec": 0.8},
                ],
            )

        response = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id, "language": "kn"},
        )

    assert response.status_code == 200
    assert seen_calls == 0
    assert response.json()["transcript"]["source"] == "sarvam"


def test_get_latest_prefers_real_transcript_over_newer_mock() -> None:
    with TestClient(app) as client:
        project_id = _create_project(client, name="Latest Transcript Real Preferred")
        asset_id = _upload_video(client, project_id)

        with Session(engine) as session:
            real = transcript_routes._store_transcript_items(
                session,
                project_id=project_id,
                asset_id=asset_id,
                duration_sec=8.0,
                source="groq",
                language="en",
                is_mock=False,
                items=[
                    {"id": "real-1", "text": "real", "start_sec": 0.0, "end_sec": 0.4},
                ],
            )
            mock = transcript_routes._store_transcript_items(
                session,
                project_id=project_id,
                asset_id=asset_id,
                duration_sec=8.0,
                source="mock",
                language="en",
                is_mock=True,
                items=[
                    {"id": "mock-1", "text": "mock", "start_sec": 0.0, "end_sec": 0.4},
                ],
            )
            real_id = real.id
            mock_id = mock.id

        latest = client.get(f"/api/v1/transcript?project_id={project_id}")
        assert latest.status_code == 200
        assert latest.json()["id"] == real_id
        explicit_mock = client.get(
            f"/api/v1/transcript?project_id={project_id}&transcript_id={mock_id}"
        )
        assert explicit_mock.status_code == 200
        assert explicit_mock.json()["id"] == mock_id


def test_generate_transcript_payload_chunked_uses_overlap_but_keeps_only_core_words(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("TMP_DIR", str(tmp_path))
    monkeypatch.setenv("TRANSCRIBE_CHUNK_DURATION_SEC", "45")
    monkeypatch.setenv("TRANSCRIBE_CHUNK_OVERLAP_SEC", "3")
    monkeypatch.setenv("TRANSCRIBE_CHUNK_BYPASS_MAX_DURATION_SEC", "0")
    seen_ranges: list[tuple[float, float]] = []

    def fake_extract(
        _source_path: str, start_sec: float, duration_sec: float, output_path
    ) -> None:
        seen_ranges.append((start_sec, duration_sec))
        output_path.write_bytes(b"chunk")

    def fake_generate(
        _source_path: str,
        _duration_sec: float,
        *,
        language_hint: str | None,
        allow_mock_fallback: bool,
        fast_mode: bool,
        prompt: str | None,
    ) -> TranscriptPayload:
        del _duration_sec, language_hint, allow_mock_fallback, fast_mode, prompt
        if os.path.basename(_source_path) == "chunk_0000.wav":
            return TranscriptPayload(
                source="groq",
                language="en",
                text="alpha beta bridge",
                words=[
                    TranscriptWordPayload(
                        id="w1", text="alpha", start_sec=40.0, end_sec=40.4
                    ),
                    TranscriptWordPayload(
                        id="w2", text="beta", start_sec=44.0, end_sec=44.4
                    ),
                    TranscriptWordPayload(
                        id="w3", text="bridge", start_sec=46.0, end_sec=46.3
                    ),
                ],
                is_mock=False,
            )
        return TranscriptPayload(
            source="groq",
            language="en",
            text="bridge gamma delta",
            words=[
                TranscriptWordPayload(
                    id="w4", text="bridge", start_sec=3.5, end_sec=3.8
                ),
                TranscriptWordPayload(
                    id="w5", text="gamma", start_sec=5.0, end_sec=5.4
                ),
                TranscriptWordPayload(
                    id="w6", text="delta", start_sec=13.0, end_sec=13.4
                ),
            ],
            is_mock=False,
        )

    monkeypatch.setattr("app.routers.transcript._extract_audio_chunk", fake_extract)
    monkeypatch.setattr(
        "app.routers.transcript._call_generate_transcript_compatible", fake_generate
    )

    result = transcript_routes._generate_transcript_payload_chunked(
        "demo.mp4",
        90.0,
        language_hint="en",
        fast_mode=False,
        prompt=None,
    )
    assert seen_ranges == [(0.0, 48.0), (42.0, 48.0)]
    assert [word.text for word in result.words] == [
        "alpha",
        "beta",
        "bridge",
        "gamma",
        "delta",
    ]
    assert result.source == "chunked:groq"


def test_generate_transcript_payload_chunked_respects_mock_fallback_setting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("TMP_DIR", str(tmp_path))
    monkeypatch.setenv("TRANSCRIBE_ALLOW_MOCK_FALLBACK", "false")
    monkeypatch.setenv("TRANSCRIBE_CHUNK_BYPASS_MAX_DURATION_SEC", "600")
    seen_allow_mock: list[bool] = []

    def fake_generate(
        _source_path: str,
        _duration_sec: float,
        *,
        language_hint: str | None,
        allow_mock_fallback: bool,
        fast_mode: bool,
        prompt: str | None,
    ) -> TranscriptPayload:
        del language_hint, fast_mode, prompt
        seen_allow_mock.append(allow_mock_fallback)
        raise RuntimeError("cloud provider failed")

    monkeypatch.setattr(
        "app.routers.transcript._call_generate_transcript_compatible", fake_generate
    )

    with pytest.raises(RuntimeError, match="cloud provider failed"):
        transcript_routes._generate_transcript_payload_chunked(
            "demo.mp4",
            90.0,
            language_hint="en",
            fast_mode=False,
            prompt=None,
        )

    assert seen_allow_mock == [False]


def test_generate_transcript_payload_chunked_transcribes_chunks_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("TMP_DIR", str(tmp_path))
    monkeypatch.setenv("TRANSCRIBE_CHUNK_DURATION_SEC", "45")
    monkeypatch.setenv("TRANSCRIBE_CHUNK_OVERLAP_SEC", "0")
    monkeypatch.setenv("TRANSCRIBE_CHUNK_BYPASS_MAX_DURATION_SEC", "0")
    monkeypatch.setenv("TRANSCRIBE_CHUNK_PARALLELISM", "2")
    barrier = threading.Barrier(2)

    def fake_extract(
        _source_path: str, start_sec: float, duration_sec: float, output_path
    ) -> None:
        del _source_path, start_sec, duration_sec
        output_path.write_bytes(b"chunk")

    def fake_generate(
        _source_path: str,
        _duration_sec: float,
        *,
        language_hint: str | None,
        allow_mock_fallback: bool,
        fast_mode: bool,
        prompt: str | None,
    ) -> TranscriptPayload:
        del _duration_sec, language_hint, allow_mock_fallback, fast_mode, prompt
        barrier.wait(timeout=1.0)
        name = os.path.basename(_source_path)
        if name == "chunk_0000.wav":
            return TranscriptPayload(
                source="groq",
                language="en",
                text="alpha beta",
                words=[
                    TranscriptWordPayload(
                        id="w1", text="alpha", start_sec=1.0, end_sec=1.4
                    ),
                    TranscriptWordPayload(
                        id="w2", text="beta", start_sec=10.0, end_sec=10.4
                    ),
                ],
                is_mock=False,
            )
        return TranscriptPayload(
            source="groq",
            language="en",
            text="gamma delta",
            words=[
                TranscriptWordPayload(
                    id="w3", text="gamma", start_sec=1.0, end_sec=1.4
                ),
                TranscriptWordPayload(
                    id="w4", text="delta", start_sec=10.0, end_sec=10.4
                ),
            ],
            is_mock=False,
        )

    monkeypatch.setattr("app.routers.transcript._extract_audio_chunk", fake_extract)
    monkeypatch.setattr(
        "app.routers.transcript._call_generate_transcript_compatible", fake_generate
    )

    result = transcript_routes._generate_transcript_payload_chunked(
        "demo.mp4",
        90.0,
        language_hint="en",
        fast_mode=False,
        prompt=None,
    )

    assert [word.text for word in result.words] == ["alpha", "beta", "gamma", "delta"]
    assert result.source == "chunked:groq"


def test_transcript_generate_humanizes_sigkill_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setenv("TRANSCRIBE_REUSE_EXISTING_ON_GENERATE", "false")
    monkeypatch.setenv("TRANSCRIBE_FAST_MODE", "false")

    def always_fail(
        _source_path: str,
        _duration_sec: float,
        *,
        language_hint: str | None,
        fast_mode: bool,
        prompt: str | None,
        progress_callback=None,
    ) -> TranscriptPayload:
        del language_hint, fast_mode, prompt, progress_callback
        raise RuntimeError("subprocess terminated by SIGKILL (signal 9)")

    monkeypatch.setattr(
        "app.routers.transcript._generate_transcript_payload_chunked", always_fail
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="Transcript SIGKILL Message")
        asset_id = _upload_video(client, project_id)
        response = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id},
        )
        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "SIGKILL" in detail
        assert "TRANSCRIBE_BACKEND=groq" in detail


def test_transcript_generate_retries_weak_regions_before_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 6.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setenv("TRANSCRIBE_REUSE_EXISTING_ON_GENERATE", "false")
    monkeypatch.setenv("TRANSCRIPT_WEAK_REGION_RETRY_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIPT_WEAK_RETRY_PAD_SEC", "0.2")
    monkeypatch.setenv("TRANSCRIPT_WEAK_RETRY_MAX_REGIONS", "2")

    def fake_chunked(
        _source_path: str,
        _duration_sec: float,
        *,
        language_hint: str | None,
        fast_mode: bool,
        prompt: str | None,
        progress_callback=None,
    ) -> TranscriptPayload:
        del (
            _source_path,
            _duration_sec,
            language_hint,
            fast_mode,
            prompt,
            progress_callback,
        )
        return TranscriptPayload(
            source="test_provider",
            language="en",
            text="hello mumble world",
            words=[
                TranscriptWordPayload(
                    id="w1", text="hello", start_sec=0.2, end_sec=0.45
                ),
                TranscriptWordPayload(
                    id="w2",
                    text="mumble",
                    start_sec=1.0,
                    end_sec=1.35,
                    source_pass="rescue",
                ),
                TranscriptWordPayload(
                    id="w3", text="world", start_sec=1.8, end_sec=2.1
                ),
            ],
            is_mock=False,
        )

    def fake_extract(
        _source_path: str, _start_sec: float, _duration_sec: float, output_path
    ) -> None:
        output_path.write_bytes(b"retry")

    def fake_retry_generate(
        _source_path: str,
        _duration_sec: float,
        *,
        language_hint: str | None,
        allow_mock_fallback: bool,
        fast_mode: bool,
        prompt: str | None,
    ) -> TranscriptPayload:
        del (
            _source_path,
            _duration_sec,
            language_hint,
            allow_mock_fallback,
            fast_mode,
            prompt,
        )
        return TranscriptPayload(
            source="retry_provider",
            language="en",
            text="brave new",
            words=[
                TranscriptWordPayload(
                    id="rw1",
                    text="brave",
                    start_sec=0.22,
                    end_sec=0.42,
                    confidence=0.95,
                ),
                TranscriptWordPayload(
                    id="rw2", text="new", start_sec=0.42, end_sec=0.62, confidence=0.95
                ),
            ],
            is_mock=False,
        )

    monkeypatch.setattr(
        "app.routers.transcript._generate_transcript_payload_chunked", fake_chunked
    )
    monkeypatch.setattr("app.routers.transcript._extract_audio_chunk", fake_extract)
    monkeypatch.setattr(
        "app.routers.transcript._call_generate_transcript_compatible",
        fake_retry_generate,
    )

    with TestClient(app) as client:
        project_id = _create_project(client, name="Transcript Weak Retry")
        asset_id = _upload_video(client, project_id)
        response = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": asset_id, "language": "en"},
        )
        assert response.status_code == 200
        transcript = response.json()["transcript"]
        assert [word["text"] for word in transcript["words"]] == [
            "hello",
            "brave",
            "new",
            "world",
        ]
        assert not any(region["status"] == "weak" for region in transcript["regions"])
