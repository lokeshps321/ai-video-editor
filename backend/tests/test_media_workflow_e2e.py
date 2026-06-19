import os

import pytest

pytest.importorskip("sqlmodel")

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")
# Disable pre-compute vocal isolation for test performance
os.environ.setdefault("TRANSCRIBE_VOCAL_ISOLATION_PRECOMPUTE", "false")

from fastapi.testclient import TestClient

from app.main import app


def test_media_workflow_create_upload_list_waveform_e2e(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 9.8)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setattr(
        "app.routers.media.extract_waveform_peaks",
        lambda *_args, **_kwargs: [0.2, 0.6, 0.9],
    )

    with TestClient(app) as client:
        create_res = client.post(
            "/api/v1/projects",
            json={
                "name": "Media E2E Workflow",
                "fps": 30,
                "width": 1080,
                "height": 1920,
            },
        )
        assert create_res.status_code == 200
        project_id = create_res.json()["id"]

        upload_res = client.post(
            "/api/v1/media/upload",
            data={"project_id": project_id},
            files={"file": ("journey.mp4", b"fake-media-bytes", "video/mp4")},
        )
        assert upload_res.status_code == 200
        asset = upload_res.json()
        assert asset["project_id"] == project_id
        assert asset["filename"] == "journey.mp4"
        assert asset["duration_sec"] == 9.8

        list_res = client.get(f"/api/v1/media?project_id={project_id}")
        assert list_res.status_code == 200
        items = list_res.json()
        assert any(item["id"] == asset["id"] for item in items)

        waveform_res = client.get(f"/api/v1/media/{asset['id']}/waveform?num_peaks=3")
        assert waveform_res.status_code == 200
        waveform = waveform_res.json()
        assert waveform["asset_id"] == asset["id"]
        assert waveform["duration_sec"] == 9.8
        assert waveform["num_peaks"] == 3
        assert waveform["peaks"] == [0.2, 0.6, 0.9]
