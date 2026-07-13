import json

from sqlmodel import SQLModel, Session, create_engine

from app.models import MediaAsset, Project
from app.routers.timeline import smart_reframe_main_video
from app.schemas import OperationPayload, SmartReframeRequest
from app.smart_reframe_service import plan_reel_smart_reframe
from app.timeline_service import (
    apply_operation,
    create_timeline_for_project,
    get_timeline_row,
    load_timeline_state,
    save_timeline_state,
)


def test_smart_reframe_uses_center_crop_when_no_subject_is_available() -> None:
    plan = plan_reel_smart_reframe(
        width=1920,
        height=1080,
        clip_duration_sec=5.0,
    )

    assert plan.crop == {"x": 656, "y": 0, "width": 608, "height": 1080}
    assert plan.crop_keyframes == []
    assert plan.uses_subject_tracking is False


def test_smart_reframe_builds_smoothed_subject_tracking_keyframes() -> None:
    plan = plan_reel_smart_reframe(
        width=1920,
        height=1080,
        clip_duration_sec=4.0,
        focus_x=0.2,
        focus_track=[
            {"time_sec": 0.0, "x_ratio": 0.2},
            {"time_sec": 2.0, "x_ratio": 0.8},
            {"time_sec": 4.0, "x_ratio": 0.7},
        ],
    )

    assert plan.crop == {"x": 80, "y": 0, "width": 608, "height": 1080}
    assert plan.uses_subject_tracking is True
    assert plan.crop_keyframes[0] == {"time_sec": 0.0, "x": 80, "y": 0}
    assert plan.crop_keyframes[-1] == {"time_sec": 4.0, "x": 678, "y": 0}
    assert 80 < int(plan.crop_keyframes[1]["x"]) < 1112


def test_smart_reframe_shifts_focus_track_to_the_clips_trim_window() -> None:
    # Source focus track: subject at 0.2 for the first 10s, then swings to 0.8
    # at 10s and holds. A clip trimmed to source seconds 10-14 should track
    # the 0.8 position throughout, not the 0.2 position from before the trim.
    plan = plan_reel_smart_reframe(
        width=1920,
        height=1080,
        clip_duration_sec=4.0,
        focus_track=[
            {"time_sec": 0.0, "x_ratio": 0.2},
            {"time_sec": 10.0, "x_ratio": 0.8},
            {"time_sec": 14.0, "x_ratio": 0.8},
        ],
        clip_start_sec=10.0,
    )

    assert plan.uses_subject_tracking is True
    assert plan.crop_keyframes[0] == {"time_sec": 0.0, "x": 1232, "y": 0}
    assert all(int(kf["x"]) == 1232 for kf in plan.crop_keyframes)


def test_smart_reframe_leaves_vertical_video_without_an_extra_crop() -> None:
    plan = plan_reel_smart_reframe(
        width=1080,
        height=1920,
        clip_duration_sec=5.0,
    )

    assert plan.crop is None
    assert plan.crop_keyframes == []


def test_smart_reframe_route_persists_a_tracked_crop_to_the_timeline() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        project = Project(name="Smart Reframe Test", width=1080, height=1920)
        session.add(project)
        session.commit()
        session.refresh(project)
        create_timeline_for_project(session, project)

        asset = MediaAsset(
            project_id=project.id,
            media_type="video",
            filename="wide.mp4",
            storage_path="missing-wide.mp4",
            mime_type="video/mp4",
            duration_sec=5.0,
            metadata_json=json.dumps(
                {
                    "width": 1920,
                    "height": 1080,
                    "focus_x": 0.2,
                    "focus_track": [
                        {"time_sec": 0.0, "x_ratio": 0.2},
                        {"time_sec": 5.0, "x_ratio": 0.8},
                    ],
                }
            ),
        )
        session.add(asset)
        session.commit()

        timeline = get_timeline_row(session, project.id)
        state = load_timeline_state(timeline)
        add_clip = OperationPayload(
            op_type="add_clip",
            params={
                "asset_id": asset.id,
                "start_sec": 0.0,
                "end_sec": 5.0,
                "timeline_start_sec": 0.0,
            },
            source="ui",
        )
        apply_operation(state, add_clip)
        save_timeline_state(session, timeline, state, source="ui", operation=add_clip)

        response = smart_reframe_main_video(
            SmartReframeRequest(),
            project.id,
            session,
            {"sub": "test-user"},
        )

        assert response.reframed_clip_count == 1
        assert response.tracked_clip_count == 1
        assert response.center_crop_clip_count == 0
        clip = next(
            clip
            for track in response.timeline.tracks
            if track.kind == "video"
            for clip in track.clips
        )
        assert clip.transform.crop is not None
        assert clip.transform.crop.width == 608
        assert clip.transform.crop_keyframes[0].x == 80
        assert clip.transform.crop_keyframes[-1].x > clip.transform.crop_keyframes[0].x
