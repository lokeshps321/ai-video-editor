import os

import pytest

pytest.importorskip("sqlmodel")

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")

from fastapi.testclient import TestClient

from app.main import app


def _create_project(client: TestClient, name: str = "Timeline Route Test") -> str:
    response = client.post(
        "/api/v1/projects",
        json={"name": name, "fps": 30, "width": 1080, "height": 1920},
    )
    assert response.status_code == 200
    return response.json()["id"]


def _upload_video(client: TestClient, project_id: str, filename: str = "clip.mp4") -> str:
    response = client.post(
        "/api/v1/media/upload",
        data={"project_id": project_id},
        files={"file": (filename, b"fake-video-bytes", "video/mp4")},
    )
    assert response.status_code == 200
    return response.json()["id"]


def _video_clip_id(timeline: dict) -> str:
    for track in timeline["tracks"]:
        if track["kind"] == "video" and track["clips"]:
            return track["clips"][0]["id"]
    raise AssertionError("expected a video clip")


def test_undo_restores_prior_clip_position_after_move_clip() -> None:
    client = TestClient(app)
    project_id = _create_project(client)
    asset_id = _upload_video(client, project_id)

    add_res = client.post(
        f"/api/v1/timeline/operations?project_id={project_id}",
        json={
            "operations": [
                {
                    "op_type": "add_clip",
                    "source": "ui",
                    "params": {
                        "asset_id": asset_id,
                        "start_sec": 0.0,
                        "end_sec": 5.0,
                        "timeline_start_sec": 0.0,
                    },
                }
            ]
        },
    )
    assert add_res.status_code == 200
    add_payload = add_res.json()
    clip_id = _video_clip_id(add_payload["timeline"])
    assert add_payload["timeline_can_undo"] is True

    move_res = client.post(
        f"/api/v1/timeline/operations?project_id={project_id}",
        json={
            "operations": [
                {
                    "op_type": "move_clip",
                    "source": "ui",
                    "params": {
                        "clip": clip_id,
                        "timeline_start_sec": 2.5,
                    },
                }
            ]
        },
    )
    assert move_res.status_code == 200
    moved_clip = next(
        clip
        for track in move_res.json()["timeline"]["tracks"]
        if track["kind"] == "video"
        for clip in track["clips"]
        if clip["id"] == clip_id
    )
    assert moved_clip["timeline_start_sec"] == 2.5

    undo_res = client.post(f"/api/v1/projects/{project_id}/undo")
    assert undo_res.status_code == 200
    undo_payload = undo_res.json()
    restored_clip = next(
        clip
        for track in undo_payload["timeline"]["tracks"]
        if track["kind"] == "video"
        for clip in track["clips"]
        if clip["id"] == clip_id
    )
    assert restored_clip["timeline_start_sec"] == 0.0
    assert undo_payload["timeline_can_redo"] is True
