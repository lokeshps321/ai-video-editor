import os

import pytest

pytest.importorskip("sqlmodel")

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")

from fastapi.testclient import TestClient

from app.main import app


def _create_project(
    client: TestClient,
    name: str = "Timeline Route Test",
    fps: int = 30,
) -> str:
    response = client.post(
        "/api/v1/projects",
        json={"name": name, "fps": fps, "width": 1080, "height": 1920},
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


def _add_video_clip(
    client: TestClient,
    project_id: str,
    asset_id: str,
    *,
    end_sec: float = 5,
) -> dict:
    response = client.post(
        f"/api/v1/timeline/operations?project_id={project_id}",
        json={
            "operations": [
                {
                    "op_type": "add_clip",
                    "params": {
                        "asset_id": asset_id,
                        "start_sec": 0,
                        "end_sec": end_sec,
                        "timeline_start_sec": 0,
                    },
                }
            ]
        },
    )
    assert response.status_code == 200
    return response.json()


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


def test_multi_operation_request_creates_one_undo_version() -> None:
    client = TestClient(app)
    project_id = _create_project(client, "Atomic multi operation")
    first_asset_id = _upload_video(client, project_id, "first.mp4")
    second_asset_id = _upload_video(client, project_id, "second.mp4")

    response = client.post(
        f"/api/v1/timeline/operations?project_id={project_id}",
        json={
            "operations": [
                {
                    "op_type": "add_clip",
                    "source": "ui",
                    "params": {
                        "asset_id": first_asset_id,
                        "start_sec": 0,
                        "end_sec": 2,
                        "timeline_start_sec": 0,
                    },
                },
                {
                    "op_type": "add_clip",
                    "source": "ui",
                    "params": {
                        "asset_id": second_asset_id,
                        "start_sec": 0,
                        "end_sec": 2,
                        "timeline_start_sec": 2,
                    },
                },
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["version"] == 1
    assert response.json()["applied_ops"] == ["add_clip", "add_clip"]

    undo_response = client.post(f"/api/v1/projects/{project_id}/undo")
    assert undo_response.status_code == 200
    assert undo_response.json()["timeline_version"] == 0
    assert all(not track["clips"] for track in undo_response.json()["timeline"]["tracks"])


def test_later_invalid_operation_rolls_back_entire_request() -> None:
    client = TestClient(app)
    project_id = _create_project(client, "Atomic rollback")
    asset_id = _upload_video(client, project_id)

    response = client.post(
        f"/api/v1/timeline/operations?project_id={project_id}",
        json={
            "operations": [
                {
                    "op_type": "add_clip",
                    "source": "ui",
                    "params": {
                        "asset_id": asset_id,
                        "start_sec": 0,
                        "end_sec": 2,
                        "timeline_start_sec": 0,
                    },
                },
                {
                    "op_type": "trim_clip",
                    "source": "ui",
                    "params": {
                        "clip": "missing-clip",
                        "start_sec": 0.2,
                        "end_sec": 1.8,
                    },
                },
            ]
        },
    )

    assert response.status_code == 400
    project = client.get(f"/api/v1/projects/{project_id}").json()
    assert project["timeline_version"] == 0
    assert all(not track["clips"] for track in project["timeline"]["tracks"])
    assert client.get(
        f"/api/v1/timeline/history?project_id={project_id}"
    ).json() == []


def test_invalid_trim_does_not_create_version() -> None:
    client = TestClient(app)
    project_id = _create_project(client, "Invalid trim version")
    asset_id = _upload_video(client, project_id)
    add_response = client.post(
        f"/api/v1/timeline/operations?project_id={project_id}",
        json={
            "operations": [
                {
                    "op_type": "add_clip",
                    "params": {
                        "asset_id": asset_id,
                        "start_sec": 0,
                        "end_sec": 2,
                        "timeline_start_sec": 0,
                    },
                }
            ]
        },
    )
    clip_id = _video_clip_id(add_response.json()["timeline"])

    trim_response = client.post(
        f"/api/v1/timeline/operations?project_id={project_id}",
        json={
            "operations": [
                {
                    "op_type": "trim_clip",
                    "params": {
                        "clip": clip_id,
                        "start_sec": 1.01,
                        "end_sec": 1.02,
                    },
                }
            ]
        },
    )

    assert trim_response.status_code == 400
    project = client.get(f"/api/v1/projects/{project_id}").json()
    assert project["timeline_version"] == 1


def test_invalid_edit_preserves_redo_until_successful_edit() -> None:
    client = TestClient(app)
    project_id = _create_project(client, "Redo preservation")
    asset_id = _upload_video(client, project_id)
    add_response = client.post(
        f"/api/v1/timeline/operations?project_id={project_id}",
        json={
            "operations": [
                {
                    "op_type": "add_clip",
                    "params": {
                        "asset_id": asset_id,
                        "start_sec": 0,
                        "end_sec": 4,
                        "timeline_start_sec": 0,
                    },
                }
            ]
        },
    )
    clip_id = _video_clip_id(add_response.json()["timeline"])
    move_url = f"/api/v1/timeline/operations?project_id={project_id}"
    move_payload = {
        "operations": [
            {
                "op_type": "move_clip",
                "params": {"clip": clip_id, "timeline_start_sec": 2},
            }
        ]
    }
    assert client.post(move_url, json=move_payload).json()["version"] == 2
    assert client.post(f"/api/v1/projects/{project_id}/undo").json()["timeline_version"] == 1

    invalid_response = client.post(
        move_url,
        json={
            "operations": [
                {
                    "op_type": "trim_clip",
                    "params": {
                        "clip": clip_id,
                        "start_sec": 1.01,
                        "end_sec": 1.02,
                    },
                }
            ]
        },
    )
    assert invalid_response.status_code == 400
    after_invalid = client.get(f"/api/v1/projects/{project_id}").json()
    assert after_invalid["timeline_version"] == 1
    assert after_invalid["timeline_can_redo"] is True

    successful_response = client.post(move_url, json=move_payload)
    assert successful_response.status_code == 200
    assert successful_response.json()["version"] == 2
    assert successful_response.json()["timeline_can_redo"] is False


@pytest.mark.parametrize("fps", [24, 30, 60])
def test_legacy_and_frame_aware_move_payloads_return_frame_aligned_seconds(
    fps: int,
) -> None:
    client = TestClient(app)
    project_id = _create_project(client, f"Frame payload {fps}", fps=fps)
    asset_id = _upload_video(client, project_id)
    add_response = client.post(
        f"/api/v1/timeline/operations?project_id={project_id}",
        json={
            "operations": [
                {
                    "op_type": "add_clip",
                    "params": {
                        "asset_id": asset_id,
                        "start_sec": 0,
                        "end_sec": 4,
                        "timeline_start_sec": 0,
                    },
                }
            ]
        },
    )
    clip_id = _video_clip_id(add_response.json()["timeline"])

    move_response = client.post(
        f"/api/v1/timeline/operations?project_id={project_id}",
        json={
            "operations": [
                {
                    "op_type": "move_clip",
                    "params": {
                        "clip": clip_id,
                        "timeline_start_sec": 9,
                        "timeline_start_frame": 3,
                    },
                }
            ]
        },
    )

    assert move_response.status_code == 200
    moved_clip_id = _video_clip_id(move_response.json()["timeline"])
    moved_clip = next(
        clip
        for track in move_response.json()["timeline"]["tracks"]
        for clip in track["clips"]
        if clip["id"] == moved_clip_id
    )
    assert moved_clip["timeline_start_sec"] == pytest.approx(3 / fps)


@pytest.mark.parametrize("fps", [24, 30, 60])
def test_frame_aware_trim_split_and_one_step_undo_at_supported_fps(
    fps: int,
) -> None:
    client = TestClient(app)
    project_id = _create_project(client, f"Trim split undo {fps}", fps=fps)
    asset_id = _upload_video(client, project_id)
    added = _add_video_clip(client, project_id, asset_id)
    clip_id = _video_clip_id(added["timeline"])
    url = f"/api/v1/timeline/operations?project_id={project_id}"

    trim_response = client.post(
        url,
        json={
            "expected_version": 1,
            "operations": [
                {
                    "op_type": "trim_clip",
                    "params": {
                        "clip": clip_id,
                        "start_frame": 2,
                        "end_frame": 20,
                    },
                }
            ],
        },
    )
    assert trim_response.status_code == 200
    trimmed = next(
        clip
        for track in trim_response.json()["timeline"]["tracks"]
        for clip in track["clips"]
        if clip["id"] == clip_id
    )
    assert trimmed["start_sec"] == pytest.approx(2 / fps)
    assert trimmed["end_sec"] == pytest.approx(20 / fps)

    split_response = client.post(
        url,
        json={
            "expected_version": 2,
            "operations": [
                {
                    "op_type": "split_clip",
                    "params": {"clip": clip_id, "at_frame": 10},
                }
            ],
        },
    )
    assert split_response.status_code == 200
    clips_after_split = [
        clip
        for track in split_response.json()["timeline"]["tracks"]
        if track["kind"] == "video"
        for clip in track["clips"]
    ]
    assert len(clips_after_split) == 2
    # at_frame is a project-timeline coordinate. The clip starts at source
    # frame 2, so splitting ten timeline frames in lands on source frame 12.
    assert clips_after_split[0]["end_sec"] == pytest.approx(12 / fps)
    assert clips_after_split[1]["start_sec"] == pytest.approx(12 / fps)

    undo_response = client.post(f"/api/v1/projects/{project_id}/undo")
    assert undo_response.status_code == 200
    restored_clips = [
        clip
        for track in undo_response.json()["timeline"]["tracks"]
        if track["kind"] == "video"
        for clip in track["clips"]
    ]
    assert undo_response.json()["timeline_version"] == 2
    assert len(restored_clips) == 1
    assert restored_clips[0]["start_sec"] == pytest.approx(2 / fps)
    assert restored_clips[0]["end_sec"] == pytest.approx(20 / fps)


def test_invalid_frame_coordinate_is_rejected_without_new_version() -> None:
    client = TestClient(app)
    project_id = _create_project(client, "Invalid frame payload")

    response = client.post(
        f"/api/v1/timeline/operations?project_id={project_id}",
        json={
            "operations": [
                {
                    "op_type": "move_clip",
                    "params": {
                        "clip": "missing",
                        "timeline_start_sec": 1,
                        "timeline_start_frame": -1,
                    },
                }
            ]
        },
    )

    assert response.status_code == 422
    assert client.get(
        f"/api/v1/projects/{project_id}"
    ).json()["timeline_version"] == 0


def test_stale_expected_version_returns_conflict_without_overwriting() -> None:
    client = TestClient(app)
    project_id = _create_project(client, "Stale edit conflict")
    asset_id = _upload_video(client, project_id)
    url = f"/api/v1/timeline/operations?project_id={project_id}"

    first_response = client.post(
        url,
        json={
            "expected_version": 0,
            "operations": [
                {
                    "op_type": "add_clip",
                    "params": {
                        "asset_id": asset_id,
                        "start_sec": 0,
                        "end_sec": 2,
                        "timeline_start_sec": 0,
                    },
                }
            ],
        },
    )
    assert first_response.status_code == 200
    assert first_response.json()["version"] == 1

    stale_response = client.post(
        url,
        json={
            "expected_version": 0,
            "operations": [
                {
                    "op_type": "set_aspect_ratio",
                    "params": {"ratio": "16:9"},
                }
            ],
        },
    )

    assert stale_response.status_code == 409
    project = client.get(f"/api/v1/projects/{project_id}").json()
    assert project["timeline_version"] == 1
    assert project["timeline"]["resolution"] == {"width": 1080, "height": 1920}


@pytest.mark.parametrize(
    ("op_type", "params"),
    [
        ("trim_clip", {}),
        ("add_clip", {"asset_id": "asset", "start_sec": None, "end_sec": 1}),
        ("add_clip", {"asset_id": "asset", "start_sec": 0, "end_sec": float("inf")}),
    ],
)
def test_malformed_operation_parameters_return_client_error(
    op_type: str,
    params: dict[str, object],
) -> None:
    client = TestClient(app, raise_server_exceptions=False)
    project_id = _create_project(client, "Malformed operation")
    url = f"/api/v1/timeline/operations?project_id={project_id}"
    if params.get("end_sec") == float("inf"):
        response = client.post(
            url,
            content=(
                '{"operations":[{"op_type":"add_clip","params":'
                '{"asset_id":"asset","start_sec":0,"end_sec":1e999}}]}'
            ),
            headers={"Content-Type": "application/json"},
        )
    else:
        response = client.post(
            url,
            json={
                "operations": [
                    {
                        "op_type": op_type,
                        "params": params,
                    }
                ]
            },
        )

    assert response.status_code in {400, 422}
    assert client.get(
        f"/api/v1/projects/{project_id}"
    ).json()["timeline_version"] == 0


def test_timeline_route_does_not_mask_database_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app, raise_server_exceptions=False)
    project_id = _create_project(client, "Database failure propagation")

    def fail_save(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("database failed")

    monkeypatch.setattr(
        "app.routers.timeline.save_timeline_state",
        fail_save,
    )

    response = client.post(
        f"/api/v1/timeline/operations?project_id={project_id}",
        json={
            "operations": [
                {
                    "op_type": "set_aspect_ratio",
                    "params": {"ratio": "16:9"},
                }
            ]
        },
    )

    assert response.status_code == 500
