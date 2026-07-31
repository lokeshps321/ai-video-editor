import pytest

pytest.importorskip("sqlmodel")

from sqlmodel import Session

from app.database import engine
from app.models import Project
from app.schemas import Clip, OperationPayload, TextOverlay, TimelineState, Track
from app.timeline_service import (
    apply_operation,
    create_timeline_for_project,
    get_timeline_row,
    load_timeline_state,
    make_default_timeline,
    save_timeline_state,
)


def make_timeline() -> TimelineState:
    return TimelineState(
        tracks=[
            Track(
                id="video-1",
                kind="video",
                clips=[
                    Clip(
                        id="clip-a",
                        asset_id="asset-a",
                        start_sec=0,
                        end_sec=10,
                        timeline_start_sec=0,
                    )
                ],
            ),
            Track(id="audio-1", kind="audio", clips=[]),
        ]
    )


def test_split_clip_operation() -> None:
    state = make_timeline()
    op = OperationPayload(op_type="split_clip", params={"clip": "clip-a", "at_sec": 4})
    apply_operation(state, op)
    video_track = state.tracks[0]
    assert len(video_track.clips) == 2
    assert video_track.clips[0].end_sec == 4
    assert video_track.clips[1].start_sec == 4


def test_set_aspect_ratio_operation() -> None:
    state = make_timeline()
    op = OperationPayload(op_type="set_aspect_ratio", params={"ratio": "16:9"})
    apply_operation(state, op)
    assert state.resolution.width == 1920
    assert state.resolution.height == 1080


def test_set_speed_recalculates_duration() -> None:
    state = make_timeline()
    op = OperationPayload(op_type="set_speed", params={"clip": "clip-a", "speed": 2})
    apply_operation(state, op)
    assert state.duration_sec == 5


def test_reorder_clips_operation() -> None:
    state = make_timeline()
    apply_operation(
        state,
        OperationPayload(
            op_type="add_clip",
            params={
                "asset_id": "asset-b",
                "start_sec": 0,
                "end_sec": 3,
                "timeline_start_sec": 10,
            },
        ),
    )
    first = state.tracks[0].clips[0].id
    second = state.tracks[0].clips[1].id
    apply_operation(
        state,
        OperationPayload(
            op_type="reorder_clips",
            params={"track_kind": "video", "clip_order": [second, first], "ripple": True},
        ),
    )
    assert state.tracks[0].clips[0].id == second
    assert state.tracks[0].clips[0].timeline_start_sec == 0


def test_track_volume_and_solo_state() -> None:
    state = make_timeline()
    apply_operation(
        state,
        OperationPayload(op_type="set_volume", params={"track_kind": "audio", "volume": 0.5, "solo": True}),
    )
    audio_track = state.tracks[1]
    assert audio_track.volume == 0.5
    assert audio_track.solo is True


def test_set_volume_targets_specific_track_id() -> None:
    state = make_timeline()
    state.tracks.append(Track(id="audio-2", kind="audio", clips=[], volume=0.25, mute=False, solo=False))

    apply_operation(
        state,
        OperationPayload(
            op_type="set_volume",
            params={
                "track_id": "audio-2",
                "track_kind": "audio",
                "volume": 0.8,
                "mute": True,
                "solo": True,
            },
        ),
    )

    assert state.tracks[1].id == "audio-1"
    assert state.tracks[1].volume == 1.0
    assert state.tracks[1].mute is False
    assert state.tracks[1].solo is False

    assert state.tracks[2].id == "audio-2"
    assert state.tracks[2].volume == 0.8
    assert state.tracks[2].mute is True
    assert state.tracks[2].solo is True


def test_set_volume_clip_index_with_audio_track_hint() -> None:
    state = make_timeline()
    apply_operation(
        state,
        OperationPayload(
            op_type="add_audio_track",
            params={
                "asset_id": "asset-audio",
                "start_sec": 0,
                "end_sec": 5,
                "timeline_start_sec": 0,
            },
        ),
    )
    apply_operation(
        state,
        OperationPayload(
            op_type="set_volume",
            params={
                "clip": 1,
                "track_kind": "audio",
                "fade_out_sec": 1.2,
            },
        ),
    )
    audio_clip = state.tracks[1].clips[0]
    assert audio_clip.audio.fade_out_sec == 1.2


def test_replace_video_track_clips_operation() -> None:
    state = make_timeline()
    apply_operation(
        state,
        OperationPayload(
            op_type="replace_video_track_clips",
            params={
                "asset_id": "asset-a",
                "ranges": [
                    {"start_sec": 0.0, "end_sec": 1.2},
                    {"start_sec": 2.5, "end_sec": 4.0},
                ],
            },
        ),
    )
    video_track = state.tracks[0]
    assert len(video_track.clips) == 2
    assert video_track.clips[0].start_sec == 0.0
    assert video_track.clips[0].end_sec == 1.2
    assert video_track.clips[0].timeline_start_sec == 0.0
    assert video_track.clips[1].start_sec == 2.5
    assert video_track.clips[1].end_sec == 4.0


def test_set_subtitles_operation() -> None:
    state = make_timeline()
    apply_operation(
        state,
        OperationPayload(
            op_type="set_subtitles",
            params={
                "asset_id": "asset-a",
                "style": "karaoke",
                "words": [
                    {"id": "w1", "text": "hello", "start_sec": 0.2, "end_sec": 0.5},
                    {"id": "w2", "text": "world", "start_sec": 0.5, "end_sec": 0.8},
                    {"id": "w3", "text": "again", "start_sec": 1.8, "end_sec": 2.2},
                ],
            },
        ),
    )
    clip = state.tracks[0].clips[0]
    assert len(clip.text_overlays) == 3
    assert clip.text_overlays[0].text == "hello"
    assert clip.text_overlays[0].style == "karaoke"


def test_set_subtitles_respects_overlay_cap_by_merging_chunks() -> None:
    state = make_timeline()
    words = [
        {
            "id": f"w{idx}",
            "text": f"word{idx}",
            "start_sec": round(0.2 + (idx * 0.22), 3),
            "end_sec": round(0.35 + (idx * 0.22), 3),
        }
        for idx in range(20)
    ]
    apply_operation(
        state,
        OperationPayload(
            op_type="set_subtitles",
            params={
                "asset_id": "asset-a",
                "style": "karaoke",
                "max_caption_overlays": 5,
                "words": words,
            },
        ),
    )
    clip = state.tracks[0].clips[0]
    assert len(clip.text_overlays) == 5
    assert any(" " in overlay.text for overlay in clip.text_overlays)


def test_set_subtitles_non_karaoke_can_group_words_until_duration_limit() -> None:
    state = make_timeline()
    apply_operation(
        state,
        OperationPayload(
            op_type="set_subtitles",
            params={
                "asset_id": "asset-a",
                "style": "static",
                "max_words_per_caption": 8,
                "max_gap_sec": 1.0,
                "max_caption_duration_sec": 0.6,
                "words": [
                    {"id": "w1", "text": "this", "start_sec": 0.2, "end_sec": 0.45},
                    {"id": "w2", "text": "is", "start_sec": 0.46, "end_sec": 0.7},
                    {"id": "w3", "text": "too", "start_sec": 0.71, "end_sec": 0.95},
                    {"id": "w4", "text": "long", "start_sec": 0.96, "end_sec": 1.2},
                ],
            },
        ),
    )
    clip = state.tracks[0].clips[0]
    assert len(clip.text_overlays) == 2
    assert clip.text_overlays[0].text == "this is"
    assert clip.text_overlays[1].text == "too long"


def test_set_subtitles_karaoke_word_level_overlays_do_not_overlap() -> None:
    state = make_timeline()
    apply_operation(
        state,
        OperationPayload(
            op_type="set_subtitles",
            params={
                "asset_id": "asset-a",
                "style": "karaoke",
                "words": [
                    {"id": "w1", "text": "one", "start_sec": 0.20, "end_sec": 0.70},
                    {"id": "w2", "text": "two", "start_sec": 0.68, "end_sec": 1.00},
                    {"id": "w3", "text": "three", "start_sec": 0.98, "end_sec": 1.30},
                ],
            },
        ),
    )
    clip = state.tracks[0].clips[0]
    assert len(clip.text_overlays) == 3

    first_start = clip.text_overlays[0].start_sec
    first_end = first_start + clip.text_overlays[0].duration_sec
    second_start = clip.text_overlays[1].start_sec
    second_end = second_start + clip.text_overlays[1].duration_sec
    third_start = clip.text_overlays[2].start_sec

    assert first_end <= (second_start + 1e-6)
    assert second_end <= (third_start + 1e-6)


def test_set_subtitles_caps_overlay_duration_for_pathological_word_timestamps() -> None:
    state = make_timeline()
    apply_operation(
        state,
        OperationPayload(
            op_type="set_subtitles",
            params={
                "asset_id": "asset-a",
                "style": "karaoke",
                "words": [
                    {"id": "w1", "text": "as", "start_sec": 0.02, "end_sec": 7.82},
                    {"id": "w2", "text": "i", "start_sec": 7.82, "end_sec": 9.0},
                ],
            },
        ),
    )
    clip = state.tracks[0].clips[0]
    assert len(clip.text_overlays) == 2
    assert clip.text_overlays[0].duration_sec <= 0.96


def test_set_subtitles_karaoke_preserves_natural_duration_when_not_pathological() -> None:
    state = make_timeline()
    apply_operation(
        state,
        OperationPayload(
            op_type="set_subtitles",
            params={
                "asset_id": "asset-a",
                "style": "karaoke",
                "words": [
                    {"id": "w1", "text": "forever", "start_sec": 0.20, "end_sec": 1.32},
                ],
            },
        ),
    )
    clip = state.tracks[0].clips[0]
    assert len(clip.text_overlays) == 1
    assert clip.text_overlays[0].duration_sec >= 1.1


def test_set_subtitles_hormozi_preset_applies_caption_style_fields() -> None:
    state = make_timeline()
    apply_operation(
        state,
        OperationPayload(
            op_type="set_subtitles",
            params={
                "asset_id": "asset-a",
                "style": "hormozi_bold",
                "words": [
                    {"id": "w1", "text": "hello", "start_sec": 0.2, "end_sec": 0.45},
                ],
            },
        ),
    )
    clip = state.tracks[0].clips[0]
    assert len(clip.text_overlays) == 1
    overlay = clip.text_overlays[0]
    assert overlay.style == "hormozi_bold"
    assert overlay.font_name == "Montserrat-Bold"
    assert overlay.font_size == 24
    assert overlay.color == "&H0000FFFF"
    assert overlay.highlight_color == "&H0000FFFF"
    assert overlay.outline_color == "&H00000000"
    assert overlay.outline_width == 2
    assert overlay.margin_v == 50


def test_set_subtitles_only_clears_matching_asset_overlays() -> None:
    state = TimelineState(
        tracks=[
            Track(
                id="video-1",
                kind="video",
                clips=[
                    Clip(
                        id="clip-a",
                        asset_id="asset-a",
                        start_sec=0,
                        end_sec=10,
                        timeline_start_sec=0,
                    ),
                    Clip(
                        id="clip-b",
                        asset_id="asset-b",
                        start_sec=0,
                        end_sec=10,
                        timeline_start_sec=10,
                    ),
                ],
            ),
            Track(id="audio-1", kind="audio", clips=[]),
        ]
    )
    state.tracks[0].clips[1].text_overlays = [
        TextOverlay(
            id="other-caption",
            text="keep me",
            start_sec=0.2,
            duration_sec=0.8,
        )
    ]

    apply_operation(
        state,
        OperationPayload(
            op_type="set_subtitles",
            params={
                "asset_id": "asset-a",
                "style": "static",
                "words": [
                    {"id": "w1", "text": "hello", "start_sec": 0.2, "end_sec": 0.45},
                ],
            },
        ),
    )

    assert len(state.tracks[0].clips[0].text_overlays) == 1
    assert len(state.tracks[0].clips[1].text_overlays) == 1
    assert state.tracks[0].clips[1].text_overlays[0].text == "keep me"


def test_text_overlay_lifecycle_operations() -> None:
    state = make_timeline()
    apply_operation(
        state,
        OperationPayload(
            op_type="add_text_overlay",
            params={
                "clip": "clip-a",
                "text": "Hook line",
                "start_sec": 1.0,
                "duration_sec": 1.6,
                "style": "creator",
            },
        ),
    )
    clip = state.tracks[0].clips[0]
    assert len(clip.text_overlays) == 1
    overlay_id = clip.text_overlays[0].id

    apply_operation(
        state,
        OperationPayload(
            op_type="move_text_overlay",
            params={"clip": "clip-a", "overlay": overlay_id, "start_sec": 2.75},
        ),
    )
    assert clip.text_overlays[0].start_sec == 2.75

    apply_operation(
        state,
        OperationPayload(
            op_type="trim_text_overlay",
            params={"clip": "clip-a", "overlay": overlay_id, "start_sec": 2.2, "duration_sec": 2.4},
        ),
    )
    assert clip.text_overlays[0].start_sec == 2.2
    assert clip.text_overlays[0].duration_sec == 2.4

    apply_operation(
        state,
        OperationPayload(
            op_type="delete_text_overlay",
            params={"clip": "clip-a", "overlay": overlay_id},
        ),
    )
    assert clip.text_overlays == []


def test_update_text_overlay_operation() -> None:
    state = make_timeline()
    apply_operation(
        state,
        OperationPayload(
            op_type="add_text_overlay",
            params={
                "clip": "clip-a",
                "text": "Before edit",
                "start_sec": 1.0,
                "duration_sec": 1.6,
            },
        ),
    )
    clip = state.tracks[0].clips[0]
    overlay_id = clip.text_overlays[0].id

    apply_operation(
        state,
        OperationPayload(
            op_type="update_text_overlay",
            params={"clip": "clip-a", "overlay": overlay_id, "text": "After edit"},
        ),
    )
    assert clip.text_overlays[0].text == "After edit"

    apply_operation(
        state,
        OperationPayload(
            op_type="update_text_overlay",
            params={"clip": "clip-a", "overlay": overlay_id, "text": "   "},
        ),
    )
    assert clip.text_overlays[0].text == "After edit"


def test_text_overlay_trim_clamps_to_clip_window() -> None:
    state = make_timeline()
    apply_operation(
        state,
        OperationPayload(
            op_type="add_text_overlay",
            params={
                "clip": "clip-a",
                "text": "CTA",
                "start_sec": 8.5,
                "duration_sec": 1.4,
            },
        ),
    )
    clip = state.tracks[0].clips[0]
    overlay_id = clip.text_overlays[0].id

    apply_operation(
        state,
        OperationPayload(
            op_type="move_text_overlay",
            params={"clip": "clip-a", "overlay": overlay_id, "start_sec": 9.7},
        ),
    )
    assert clip.text_overlays[0].start_sec == 8.6

    apply_operation(
        state,
        OperationPayload(
            op_type="trim_text_overlay",
            params={"clip": "clip-a", "overlay": overlay_id, "start_sec": 9.9, "duration_sec": 5.0},
        ),
    )
    assert clip.text_overlays[0].start_sec == 9.9
    assert clip.text_overlays[0].duration_sec == 0.1


def test_default_timeline_includes_overlay_track() -> None:
    project = Project(name="Overlay Track Default", fps=30, width=1080, height=1920)
    state = make_default_timeline(project)
    assert any(track.kind == "overlay" for track in state.tracks)


def test_broll_clip_lifecycle_operations() -> None:
    state = make_timeline()

    apply_operation(
        state,
        OperationPayload(
            op_type="add_broll_clip",
            params={
                "asset_id": "asset-broll",
                "start_sec": 0.4,
                "end_sec": 1.9,
                "timeline_start_sec": 2.0,
                "opacity": 0.45,
                "crop": {"x": 10, "y": 0, "width": 720, "height": 1280},
            },
        ),
    )
    overlay_track = next(track for track in state.tracks if track.kind == "overlay")
    assert len(overlay_track.clips) == 1
    clip_id = overlay_track.clips[0].id
    assert overlay_track.clips[0].broll_opacity == 0.45
    assert overlay_track.clips[0].audio.mute is True
    assert overlay_track.clips[0].transform.crop is not None
    assert overlay_track.clips[0].transform.crop.width == 720

    apply_operation(
        state,
        OperationPayload(
            op_type="move_broll_clip",
            params={"clip": clip_id, "timeline_start_sec": 3.25},
        ),
    )
    assert overlay_track.clips[0].timeline_start_sec == pytest.approx(98 / 30)

    apply_operation(
        state,
        OperationPayload(
            op_type="trim_broll_clip",
            params={"clip": clip_id, "start_sec": 0.5, "end_sec": 1.4},
        ),
    )
    assert overlay_track.clips[0].start_sec == 0.5
    assert overlay_track.clips[0].end_sec == 1.4

    apply_operation(
        state,
        OperationPayload(
            op_type="set_broll_opacity",
            params={"clip": clip_id, "opacity": 0.9},
        ),
    )
    assert overlay_track.clips[0].broll_opacity == 0.9

    apply_operation(
        state,
        OperationPayload(
            op_type="delete_broll_clip",
            params={"clip": clip_id},
        ),
    )
    assert len(overlay_track.clips) == 0


def test_replace_video_track_keeps_overlay_clips() -> None:
    state = make_timeline()
    apply_operation(
        state,
        OperationPayload(
            op_type="add_broll_clip",
            params={
                "asset_id": "asset-broll",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "timeline_start_sec": 0.2,
                "opacity": 0.7,
            },
        ),
    )
    overlay_track = next(track for track in state.tracks if track.kind == "overlay")
    overlay_clip_id = overlay_track.clips[0].id

    apply_operation(
        state,
        OperationPayload(
            op_type="replace_video_track_clips",
            params={
                "asset_id": "asset-a",
                "ranges": [
                    {"start_sec": 0.0, "end_sec": 2.0},
                    {"start_sec": 3.0, "end_sec": 4.0},
                ],
            },
        ),
    )
    assert len(overlay_track.clips) == 1
    assert overlay_track.clips[0].id == overlay_clip_id


def test_paste_clip_preserves_properties_and_regenerates_ids() -> None:
    state = make_timeline()
    source = state.tracks[0].clips[0]
    source.speed = 2.0
    source.text_overlays = [
        TextOverlay(id="overlay-1", text="hi", start_sec=1, duration_sec=2)
    ]
    payload = source.model_dump()
    payload.pop("id")
    apply_operation(
        state,
        OperationPayload(
            op_type="paste_clip",
            params={
                "clip": payload,
                "track_kind": "video",
                "timeline_start_sec": 5.0,
                "ripple": True,
            },
        ),
    )
    video_track = state.tracks[0]
    assert len(video_track.clips) == 2
    pasted = video_track.clips[1]
    assert pasted.id != source.id
    assert pasted.speed == 2.0
    assert pasted.text_overlays[0].id != "overlay-1"
    # ripple packs the track back-to-back
    assert pasted.timeline_start_sec == 5.0


def test_move_clip_to_track_id_destination() -> None:
    state = make_timeline()
    apply_operation(
        state,
        OperationPayload(
            op_type="move_clip",
            params={
                "clip": "clip-a",
                "track_kind": "audio",
                "track_id": "audio-1",
                "timeline_start_sec": 0.0,
                "ripple": True,
                "source_ripple": True,
            },
        ),
    )
    assert state.tracks[0].clips == []
    assert [clip.id for clip in state.tracks[1].clips] == ["clip-a"]


@pytest.mark.parametrize(
    ("fps", "requested_sec", "expected_sec"),
    [
        (24, 0.1, 2 / 24),
        (30, 0.1, 3 / 30),
        (60, 0.1, 6 / 60),
    ],
)
def test_move_clip_normalizes_legacy_seconds_to_project_frames(
    fps: int,
    requested_sec: float,
    expected_sec: float,
) -> None:
    state = make_timeline()
    state.fps = fps

    apply_operation(
        state,
        OperationPayload(
            op_type="move_clip",
            params={"clip": "clip-a", "timeline_start_sec": requested_sec},
        ),
    )

    assert state.tracks[0].clips[0].timeline_start_sec == pytest.approx(expected_sec)


@pytest.mark.parametrize("fps", [24, 30, 60])
def test_trim_clip_rounds_inward_without_expanding_source_range(fps: int) -> None:
    state = make_timeline()
    state.fps = fps
    frame_sec = 1 / fps

    apply_operation(
        state,
        OperationPayload(
            op_type="trim_clip",
            params={
                "clip": "clip-a",
                "start_sec": frame_sec * 1.1,
                "end_sec": frame_sec * 4.9,
            },
        ),
    )

    clip = state.tracks[0].clips[0]
    assert clip.start_sec == pytest.approx(frame_sec * 2)
    assert clip.end_sec == pytest.approx(frame_sec * 4)


def test_trim_clip_uses_speed_adjusted_source_frame_boundaries() -> None:
    state = make_timeline()
    state.fps = 30
    state.tracks[0].clips[0].speed = 2.0

    apply_operation(
        state,
        OperationPayload(
            op_type="trim_clip",
            params={"clip": "clip-a", "start_sec": 0.04, "end_sec": 0.19},
        ),
    )

    clip = state.tracks[0].clips[0]
    assert clip.start_sec == pytest.approx(2 / 30)
    assert clip.end_sec == pytest.approx(4 / 30)
    assert ((clip.end_sec - clip.start_sec) / clip.speed) * state.fps == pytest.approx(1)


def test_typed_trim_frames_are_absolute_source_time_frames_at_non_unit_speed() -> None:
    state = make_timeline()
    state.fps = 30
    state.tracks[0].clips[0].speed = 2.0

    apply_operation(
        state,
        OperationPayload(
            op_type="trim_clip",
            params={
                "clip": "clip-a",
                "start_sec": 9,
                "end_sec": 10,
                "start_frame": 2,
                "end_frame": 4,
            },
        ),
    )

    clip = state.tracks[0].clips[0]
    assert clip.start_sec == pytest.approx(2 / 30)
    assert clip.end_sec == pytest.approx(4 / 30)


@pytest.mark.parametrize("fps", [24, 30, 60])
def test_split_clip_normalizes_to_project_frame_boundary(fps: int) -> None:
    state = make_timeline()
    state.fps = fps

    apply_operation(
        state,
        OperationPayload(
            op_type="split_clip",
            params={"clip": "clip-a", "at_sec": (4.4 / fps)},
        ),
    )

    left, right = state.tracks[0].clips
    assert left.end_sec == pytest.approx(4 / fps)
    assert right.start_sec == pytest.approx(4 / fps)
    assert right.timeline_start_sec == pytest.approx(4 / fps)


def test_frame_coordinate_takes_precedence_over_legacy_seconds() -> None:
    state = make_timeline()
    state.fps = 24

    apply_operation(
        state,
        OperationPayload(
            op_type="move_clip",
            params={
                "clip": "clip-a",
                "timeline_start_sec": 9.0,
                "timeline_start_frame": 3,
            },
        ),
    )

    assert state.tracks[0].clips[0].timeline_start_sec == pytest.approx(3 / 24)


@pytest.mark.parametrize("invalid_frame", [-1, 1.5, True, "3"])
def test_frame_coordinates_require_non_negative_integers(invalid_frame: object) -> None:
    with pytest.raises(ValueError):
        OperationPayload(
            op_type="move_clip",
            params={"clip": "clip-a", "timeline_start_frame": invalid_frame},
        )


def test_legacy_save_detects_version_changed_after_timeline_was_loaded() -> None:
    with Session(engine) as setup_session:
        project = Project(name="Legacy optimistic save", fps=30, width=1080, height=1920)
        setup_session.add(project)
        setup_session.commit()
        setup_session.refresh(project)
        project_id = project.id
        create_timeline_for_project(setup_session, project)

    with Session(engine) as first_session, Session(engine) as stale_session:
        first_timeline = get_timeline_row(first_session, project_id)
        stale_timeline = get_timeline_row(stale_session, project_id)
        first_state = load_timeline_state(first_timeline)
        stale_state = load_timeline_state(stale_timeline)
        apply_operation(
            first_state,
            OperationPayload(op_type="set_aspect_ratio", params={"ratio": "16:9"}),
        )
        apply_operation(
            stale_state,
            OperationPayload(op_type="set_aspect_ratio", params={"ratio": "1:1"}),
        )

        saved = save_timeline_state(
            first_session,
            first_timeline,
            first_state,
            source="ui",
        )
        assert saved.version == 1

        with pytest.raises(Exception, match="stale"):
            save_timeline_state(
                stale_session,
                stale_timeline,
                stale_state,
                source="ui",
            )

    with Session(engine) as verify_session:
        timeline = get_timeline_row(verify_session, project_id)
        assert timeline.version == 1
        assert load_timeline_state(timeline).resolution.width == 1920
