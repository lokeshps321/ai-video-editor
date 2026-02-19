import pytest

pytest.importorskip("sqlmodel")

from app.schemas import Clip, OperationPayload, TimelineState, Track
from app.timeline_service import apply_operation


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
                "max_words_per_caption": 2,
                "words": [
                    {"id": "w1", "text": "hello", "start_sec": 0.2, "end_sec": 0.5},
                    {"id": "w2", "text": "world", "start_sec": 0.5, "end_sec": 0.8},
                    {"id": "w3", "text": "again", "start_sec": 1.8, "end_sec": 2.2},
                ],
            },
        ),
    )
    clip = state.tracks[0].clips[0]
    assert len(clip.text_overlays) == 2
    assert clip.text_overlays[0].text == "hello world"
    assert clip.text_overlays[0].style == "karaoke"
