import os
import time
from pathlib import Path

import pytest

pytest.importorskip("sqlmodel")

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")

from fastapi.testclient import TestClient

from app.main import app


def test_project_create_and_prompt_apply() -> None:
    with TestClient(app) as client:
        create_res = client.post(
            "/api/v1/projects",
            json={"name": "API Test Project", "fps": 30, "width": 1080, "height": 1920},
        )
        assert create_res.status_code == 200
        project = create_res.json()
        project_id = project["id"]
        assert any(track["kind"] == "overlay" for track in project["timeline"]["tracks"])

        parse_res = client.post("/api/v1/prompt/parse", json={"prompt": "set aspect 16:9"})
        assert parse_res.status_code == 200
        parsed = parse_res.json()
        assert parsed["operations"][0]["op_type"] == "set_aspect_ratio"

        apply_res = client.post(
            f"/api/v1/prompt/apply?project_id={project_id}",
            json={"prompt": "set aspect 16:9"},
        )
        assert apply_res.status_code == 200
        applied = apply_res.json()
        assert applied["timeline"]["resolution"]["width"] == 1920
        assert applied["timeline"]["resolution"]["height"] == 1080


def test_render_preview_records_job_events() -> None:
    with TestClient(app) as client:
        create_res = client.post(
            "/api/v1/projects",
            json={"name": "Render Event Test", "fps": 30, "width": 1080, "height": 1920},
        )
        assert create_res.status_code == 200
        project_id = create_res.json()["id"]

        preview_res = client.post(f"/api/v1/render/preview?project_id={project_id}")
        assert preview_res.status_code == 200
        job = preview_res.json()
        job_id = job["id"]
        assert job["status"] == "queued"

        for _ in range(20):
            status_res = client.get(f"/api/v1/jobs/{job_id}")
            assert status_res.status_code == 200
            status = status_res.json()["status"]
            if status not in {"queued", "running"}:
                break
            time.sleep(0.1)

        events_res = client.get(f"/api/v1/jobs/{job_id}/events")
        assert events_res.status_code == 200
        events = events_res.json()
        assert len(events) >= 2
        assert events[0]["stage"] == "queued"


def test_render_preview_uses_requested_portrait_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _capture_enqueue(job_id: str, export_settings) -> None:
        captured["job_id"] = job_id
        captured["aspect_ratio"] = export_settings.aspect_ratio
        captured["resolution"] = export_settings.resolution
        captured["fps"] = export_settings.fps
        captured["quality"] = export_settings.quality

    monkeypatch.setattr("app.routers.render.enqueue_render_job", _capture_enqueue)

    with TestClient(app) as client:
        create_res = client.post(
            "/api/v1/projects",
            json={"name": "Preview Portrait Test", "fps": 30, "width": 1080, "height": 1920},
        )
        assert create_res.status_code == 200
        project_id = create_res.json()["id"]

        preview_res = client.post(
            f"/api/v1/render/preview?project_id={project_id}",
            json={"format": "mp4", "aspect_ratio": "9:16", "resolution": "1080p", "fps": 60, "quality": "high"},
        )
        assert preview_res.status_code == 200
        job = preview_res.json()
        assert job["status"] == "queued"
        assert captured["job_id"] == job["id"]
        assert captured["aspect_ratio"] == "9:16"
        assert captured["resolution"] == "720p"
        assert captured["fps"] == 60
        assert captured["quality"] == "low"


def test_render_export_accepts_aspect_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _capture_enqueue(job_id: str, export_settings) -> None:
        captured["job_id"] = job_id
        captured["aspect_ratio"] = export_settings.aspect_ratio
        captured["resolution"] = export_settings.resolution

    monkeypatch.setattr("app.routers.render.enqueue_render_job", _capture_enqueue)

    with TestClient(app) as client:
        create_res = client.post(
            "/api/v1/projects",
            json={"name": "Render Export Aspect Test", "fps": 30, "width": 1080, "height": 1920},
        )
        assert create_res.status_code == 200
        project_id = create_res.json()["id"]

        export_res = client.post(
            f"/api/v1/render/export?project_id={project_id}",
            json={"format": "mp4", "aspect_ratio": "9:16", "resolution": "1080p", "fps": 30, "quality": "high"},
        )
        assert export_res.status_code == 200
        job = export_res.json()
        assert job["status"] == "queued"
        assert captured["job_id"] == job["id"]
        assert captured["aspect_ratio"] == "9:16"
        assert captured["resolution"] == "1080p"


def test_ingest_url_creates_media_asset_and_events(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download_video_with_ytdlp(url: str, project_id: str) -> tuple[str, str]:
        project_dir = Path(os.environ["UPLOAD_DIR"]) / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        file_path = project_dir / "ingested-sample.mp4"
        file_path.write_bytes(b"fake-media")
        return str(file_path), f"{project_id}/{file_path.name}"

    monkeypatch.setattr("app.jobs.download_video_with_ytdlp", fake_download_video_with_ytdlp)
    monkeypatch.setattr("app.jobs.probe_stream_flags", lambda _: {"has_video": True, "has_audio": True})
    monkeypatch.setattr("app.jobs.probe_duration_seconds", lambda _: 12.5)

    with TestClient(app) as client:
        create_res = client.post(
            "/api/v1/projects",
            json={"name": "Ingest Event Test", "fps": 30, "width": 1080, "height": 1920},
        )
        assert create_res.status_code == 200
        project_id = create_res.json()["id"]

        ingest_res = client.post(
            f"/api/v1/ingest/url?project_id={project_id}",
            json={"url": "https://example.com/video"},
        )
        assert ingest_res.status_code == 200
        job_id = ingest_res.json()["id"]

        for _ in range(20):
            status_res = client.get(f"/api/v1/jobs/{job_id}")
            assert status_res.status_code == 200
            status = status_res.json()["status"]
            if status not in {"queued", "running"}:
                break
            time.sleep(0.1)

        final_status = client.get(f"/api/v1/jobs/{job_id}").json()
        assert final_status["status"] == "completed"
        assert final_status["output_path"].endswith("/ingested-sample.mp4")

        media_res = client.get(f"/api/v1/media?project_id={project_id}")
        assert media_res.status_code == 200
        media_items = media_res.json()
        assert any(item["filename"] == "ingested-sample.mp4" for item in media_items)

        events_res = client.get(f"/api/v1/jobs/{job_id}/events")
        assert events_res.status_code == 200
        stages = [event["stage"] for event in events_res.json()]
        assert "queued" in stages
        assert "download" in stages
        assert "register" in stages
        assert "complete" in stages
