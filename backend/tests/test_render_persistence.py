from __future__ import annotations

from fastapi.testclient import TestClient
from sqlmodel import Session

from app import jobs
from app.database import engine
from app.jobs import create_job
from app.main import app
from app.models import MediaAsset
from app.schemas import Clip, ExportSettings
from app.storage import storage
from app.timeline_service import get_timeline_row, load_timeline_state, save_timeline_state


def _create_project(client: TestClient, name: str) -> str:
    response = client.post(
        "/api/v1/projects",
        json={"name": name, "fps": 30, "width": 1920, "height": 1080},
    )
    assert response.status_code == 200
    return response.json()["id"]


def test_export_does_not_reuse_active_job_from_an_older_timeline(
    monkeypatch,
) -> None:
    monkeypatch.setattr("app.routers.render.enqueue_render_job", lambda *_: None)

    with TestClient(app) as client:
        project_id = _create_project(client, "Export revision isolation")
        payload = {
            "format": "mp4",
            "aspect_ratio": "16:9",
            "resolution": "1080p",
            "fps": 30,
            "quality": "high",
        }
        first = client.post(
            f"/api/v1/render/export?project_id={project_id}",
            json=payload,
        )
        assert first.status_code == 200

        edited = client.post(
            f"/api/v1/prompt/apply?project_id={project_id}",
            json={"prompt": "set aspect 9:16"},
        )
        assert edited.status_code == 200

        second = client.post(
            f"/api/v1/render/export?project_id={project_id}",
            json=payload,
        )
        assert second.status_code == 200
        assert second.json()["id"] != first.json()["id"]
        assert second.json()["timeline_version"] == edited.json()["version"]


def test_reopened_project_only_restores_preview_for_its_current_timeline() -> None:
    with TestClient(app) as client:
        project_id = _create_project(client, "Current preview restoration")

        with Session(engine) as session:
            timeline = get_timeline_row(session, project_id)
            job = create_job(
                session,
                project_id,
                "preview",
                timeline_version=timeline.version,
            )
            output_path = storage.output_path(project_id, "mp4")
            with open(output_path, "wb") as output:
                output.write(b"saved-preview")
            job.status = "completed"
            job.progress = 100
            job.output_path = storage.to_public_render_path(output_path)
            session.add(job)
            session.commit()
            job_id = job.id

        restored = client.get(f"/api/v1/projects/{project_id}/preview")
        assert restored.status_code == 200
        assert restored.json()["id"] == job_id

        edited = client.post(
            f"/api/v1/prompt/apply?project_id={project_id}",
            json={"prompt": "set aspect 9:16"},
        )
        assert edited.status_code == 200

        stale = client.get(f"/api/v1/projects/{project_id}/preview")
        assert stale.status_code == 200
        assert stale.json() is None


def test_render_worker_uses_the_timeline_revision_captured_by_the_job(
    monkeypatch,
) -> None:
    with TestClient(app) as client:
        project_id = _create_project(client, "Queued render snapshot")

    captured: dict[str, float] = {}
    with Session(engine) as session:
        asset = MediaAsset(
            project_id=project_id,
            media_type="video",
            filename="source.mp4",
            storage_path="source.mp4",
            mime_type="video/mp4",
            duration_sec=10.0,
        )
        session.add(asset)
        session.commit()
        session.refresh(asset)

        timeline = get_timeline_row(session, project_id)
        queued_state = load_timeline_state(timeline)
        video_track = next(
            track for track in queued_state.tracks if track.kind == "video"
        )
        video_track.clips = [
            Clip(
                id="queued-clip",
                asset_id=asset.id,
                start_sec=1.0,
                end_sec=5.0,
                timeline_start_sec=0.0,
            )
        ]
        queued_state.duration_sec = 4.0
        timeline = save_timeline_state(
            session,
            timeline,
            queued_state,
            source="test",
        )
        job = create_job(
            session,
            project_id,
            "preview",
            timeline_version=timeline.version,
        )

        newer_state = load_timeline_state(timeline)
        newer_state.tracks[0].clips[0].start_sec = 2.0
        save_timeline_state(
            session,
            timeline,
            newer_state,
            source="test",
        )
        job_id = job.id

    def capture_command(*, timeline, **_kwargs):
        clip = next(
            track.clips[0] for track in timeline.tracks if track.kind == "video"
        )
        captured["start_sec"] = clip.start_sec
        return ["ffmpeg"]

    monkeypatch.setattr(jobs, "build_ffmpeg_command", capture_command)
    monkeypatch.setattr(jobs, "run_ffmpeg", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(jobs, "_asset_has_audio", lambda *_: False)

    jobs.process_render_job(job_id, ExportSettings(format="mp4"))

    assert captured["start_sec"] == 1.0
