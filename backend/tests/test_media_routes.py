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


def _create_project(client: TestClient, name: str = "Media Route Test") -> str:
    response = client.post(
        "/api/v1/projects",
        json={"name": name, "fps": 30, "width": 1080, "height": 1920},
    )
    assert response.status_code == 200
    return response.json()["id"]


def _upload_media(
    client: TestClient, project_id: str, filename: str = "clip.mp4"
) -> dict:
    response = client.post(
        "/api/v1/media/upload",
        data={"project_id": project_id},
        files={"file": (filename, b"fake-media-bytes", "video/mp4")},
    )
    assert response.status_code == 200
    return response.json()


def test_media_upload_success_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 6.5)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )

    with TestClient(app) as client:
        project_id = _create_project(client)
        uploaded = _upload_media(client, project_id, filename="sample.mp4")

        assert uploaded["project_id"] == project_id
        assert uploaded["media_type"] == "video"
        assert uploaded["filename"] == "sample.mp4"
        assert uploaded["duration_sec"] == 6.5

        list_res = client.get(f"/api/v1/media?project_id={project_id}")
        assert list_res.status_code == 200
        items = list_res.json()
        assert any(item["id"] == uploaded["id"] for item in items)


def test_media_upload_validates_multipart_file_field() -> None:
    with TestClient(app) as client:
        project_id = _create_project(client, name="Media Empty Filename Test")
        response = client.post(
            "/api/v1/media/upload",
            data={"project_id": project_id},
            files={"file": ("", b"invalid", "video/mp4")},
        )
        assert response.status_code == 422


def test_ingest_url_returns_400_for_invalid_url() -> None:
    with TestClient(app) as client:
        project_id = _create_project(client, name="Ingest Invalid Url Test")
        response = client.post(
            f"/api/v1/ingest/url?project_id={project_id}",
            json={"url": "not-a-valid-url"},
        )
        assert response.status_code == 400


def test_media_upload_returns_404_for_missing_project() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/media/upload",
            data={"project_id": "missing-project-id"},
            files={"file": ("sample.mp4", b"fake-media-bytes", "video/mp4")},
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Project not found"


def test_media_upload_returns_500_when_storage_save_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raise_save_error(*_args, **_kwargs):
        raise RuntimeError("simulated storage failure")

    monkeypatch.setattr("app.routers.media.storage.save_upload", _raise_save_error)

    with TestClient(app, raise_server_exceptions=False) as client:
        project_id = _create_project(client, name="Media Storage Error Test")
        response = client.post(
            "/api/v1/media/upload",
            data={"project_id": project_id},
            files={"file": ("sample.mp4", b"fake-media-bytes", "video/mp4")},
        )
        assert response.status_code == 500


def test_media_waveform_success_and_num_peaks_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _capture_extract(
        path: str, *, num_peaks: int = 800, duration_sec: float | None = None
    ) -> list[float]:
        captured["path"] = path
        captured["num_peaks"] = num_peaks
        captured["duration_sec"] = duration_sec
        return [0.1, 0.5, 1.0]

    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 12.0)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setattr("app.routers.media.extract_waveform_peaks", _capture_extract)

    with TestClient(app) as client:
        project_id = _create_project(client, name="Waveform Success Test")
        uploaded = _upload_media(client, project_id, filename="wave.mp4")

        waveform_res = client.get(
            f"/api/v1/media/{uploaded['id']}/waveform?num_peaks=9999"
        )
        assert waveform_res.status_code == 200
        payload = waveform_res.json()
        assert payload["asset_id"] == uploaded["id"]
        assert payload["num_peaks"] == 3
        assert payload["duration_sec"] == 12.0
        assert payload["peaks"] == [0.1, 0.5, 1.0]
        assert captured["num_peaks"] == 2000
        assert captured["duration_sec"] == 12.0


def test_media_waveform_returns_404_for_unknown_asset() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/media/nonexistent-asset/waveform")
        assert response.status_code == 404
        assert response.json()["detail"] == "Media asset not found"


def test_media_waveform_returns_500_on_extraction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_extract_error(*_args, **_kwargs):
        raise RuntimeError("simulated waveform extraction failure")

    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 4.2)
    monkeypatch.setattr(
        "app.routers.media.probe_stream_flags",
        lambda _: {"has_video": True, "has_audio": True},
    )
    monkeypatch.setattr(
        "app.routers.media.extract_waveform_peaks", _raise_extract_error
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        project_id = _create_project(client, name="Waveform Error Test")
        uploaded = _upload_media(client, project_id, filename="wave-error.mp4")
        response = client.get(f"/api/v1/media/{uploaded['id']}/waveform")
        assert response.status_code == 500
