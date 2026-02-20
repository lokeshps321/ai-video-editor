from app.render_service import build_ffmpeg_command
from app.schemas import Clip, ExportSettings, TextOverlay, TimelineState, Track, Transition


def _timeline() -> TimelineState:
    video_clip = Clip(
        id="clip-v1",
        asset_id="asset-v1",
        start_sec=0,
        end_sec=5,
        timeline_start_sec=0,
        speed=1.0,
    )
    audio_clip = Clip(
        id="clip-a1",
        asset_id="asset-a1",
        start_sec=0,
        end_sec=5,
        timeline_start_sec=1.25,
        speed=1.0,
    )
    return TimelineState(
        resolution={"width": 1080, "height": 1920},
        tracks=[
            Track(id="track-v", kind="video", clips=[video_clip]),
            Track(id="track-a", kind="audio", clips=[audio_clip]),
        ],
        duration_sec=5,
    )


def test_build_ffmpeg_command_includes_audio_pipeline() -> None:
    state = _timeline()
    command = build_ffmpeg_command(
        timeline=state,
        clip_inputs=[(state.tracks[0].clips[0], "/tmp/video.mp4")],
        clip_has_audio_flags=[True],
        bg_audio_inputs=[(state.tracks[1].clips[0], "/tmp/music.mp3")],
        bg_has_audio_flags=[True],
        output_path="/tmp/out.mp4",
        export_settings=ExportSettings(format="mp4", resolution="1080p", fps=30, quality="high"),
    )
    joined = " ".join(command)
    assert "[v0]null[vmain]" in joined
    assert "[va0]anull[amain]" in joined
    assert "amix=inputs=2" in joined
    assert "[aout]" in joined
    assert "/tmp/out.mp4" in joined


def test_build_ffmpeg_command_uses_portrait_resolution() -> None:
    state = _timeline()
    command = build_ffmpeg_command(
        timeline=state,
        clip_inputs=[(state.tracks[0].clips[0], "/tmp/video.mp4")],
        clip_has_audio_flags=[False],
        bg_audio_inputs=[],
        bg_has_audio_flags=[],
        output_path="/tmp/out.mp4",
        export_settings=ExportSettings(format="mp4", resolution="720p", fps=30, quality="medium"),
    )
    joined = " ".join(command)
    assert "scale=720:1280" in joined


def test_build_ffmpeg_command_uses_fast_preset_for_low_quality() -> None:
    state = _timeline()
    command = build_ffmpeg_command(
        timeline=state,
        clip_inputs=[(state.tracks[0].clips[0], "/tmp/video.mp4")],
        clip_has_audio_flags=[False],
        bg_audio_inputs=[],
        bg_has_audio_flags=[],
        output_path="/tmp/out.mp4",
        export_settings=ExportSettings(format="mp4", resolution="720p", fps=24, quality="low"),
    )
    joined = " ".join(command)
    assert "-preset ultrafast" in joined


def test_build_ffmpeg_command_uses_transition_xfade() -> None:
    state = _timeline()
    second = state.tracks[0].clips[0].model_copy(deep=True)
    second.id = "clip-v2"
    second.timeline_start_sec = 5
    second.transition = Transition(type="dissolve", duration_sec=0.5)
    state.tracks[0].clips = [state.tracks[0].clips[0], second]

    command = build_ffmpeg_command(
        timeline=state,
        clip_inputs=[
            (state.tracks[0].clips[0], "/tmp/video1.mp4"),
            (state.tracks[0].clips[1], "/tmp/video2.mp4"),
        ],
        clip_has_audio_flags=[True, True],
        bg_audio_inputs=[],
        bg_has_audio_flags=[],
        output_path="/tmp/out.mp4",
        export_settings=ExportSettings(format="mp4", resolution="1080p", fps=30, quality="high"),
    )
    joined = " ".join(command)
    assert "xfade=transition=dissolve" in joined
    assert "acrossfade=d=0.500" in joined


def test_build_ffmpeg_command_applies_bg_fade_before_delay() -> None:
    state = _timeline()
    audio_clip = state.tracks[1].clips[0].model_copy(deep=True)
    audio_clip.timeline_start_sec = 2.0
    audio_clip.audio.fade_out_sec = 1.0

    command = build_ffmpeg_command(
        timeline=state,
        clip_inputs=[(state.tracks[0].clips[0], "/tmp/video.mp4")],
        clip_has_audio_flags=[True],
        bg_audio_inputs=[(audio_clip, "/tmp/music.mp3")],
        bg_has_audio_flags=[True],
        output_path="/tmp/out.mp4",
        export_settings=ExportSettings(format="mp4", resolution="1080p", fps=30, quality="high"),
    )
    joined = " ".join(command)
    fade_idx = joined.find("afade=t=out")
    delay_idx = joined.find("adelay=2000|2000")
    assert fade_idx != -1
    assert delay_idx != -1
    assert fade_idx < delay_idx


def test_build_ffmpeg_command_karaoke_style_uses_compatible_drawtext_options() -> None:
    state = _timeline()
    video_clip = state.tracks[0].clips[0]
    video_clip.text_overlays = [
        TextOverlay(
            id="ov-1",
            text="hello world",
            start_sec=0.0,
            duration_sec=1.2,
            style="karaoke",
        )
    ]
    command = build_ffmpeg_command(
        timeline=state,
        clip_inputs=[(video_clip, "/tmp/video.mp4")],
        clip_has_audio_flags=[False],
        bg_audio_inputs=[],
        bg_has_audio_flags=[],
        output_path="/tmp/out.mp4",
        export_settings=ExportSettings(format="mp4", resolution="1080p", fps=30, quality="high"),
    )
    joined = " ".join(command)
    assert "drawtext=" in joined
    assert "style=karaoke" not in joined
    assert "fontcolor_expr=" not in joined
    assert "alpha='if(lt(mod(t-" in joined


def test_build_ffmpeg_command_composites_broll_overlay_with_opacity() -> None:
    state = _timeline()
    overlay_clip = Clip(
        id="clip-b1",
        asset_id="asset-b1",
        start_sec=0,
        end_sec=2,
        timeline_start_sec=0.5,
        speed=1.0,
        broll_opacity=0.5,
    )
    command = build_ffmpeg_command(
        timeline=state,
        clip_inputs=[(state.tracks[0].clips[0], "/tmp/video.mp4")],
        clip_has_audio_flags=[False],
        bg_audio_inputs=[],
        bg_has_audio_flags=[],
        output_path="/tmp/out.mp4",
        export_settings=ExportSettings(format="mp4", resolution="1080p", fps=30, quality="high"),
        overlay_inputs=[(overlay_clip, "/tmp/broll.mp4")],
        overlay_has_video_flags=[True],
    )
    joined = " ".join(command)
    assert "/tmp/broll.mp4" in joined
    assert "colorchannelmixer=aa=0.500" in joined
    assert "overlay=(W-w)/2:(H-h)/2" in joined


def test_build_ffmpeg_command_applies_text_after_broll_overlay() -> None:
    state = _timeline()
    video_clip = state.tracks[0].clips[0]
    video_clip.text_overlays = [
        TextOverlay(
            id="ov-2",
            text="hello world",
            start_sec=0.0,
            duration_sec=1.0,
            style="static",
        )
    ]
    overlay_clip = Clip(
        id="clip-b2",
        asset_id="asset-b2",
        start_sec=0,
        end_sec=2,
        timeline_start_sec=0.2,
        speed=1.0,
        broll_opacity=1.0,
    )
    command = build_ffmpeg_command(
        timeline=state,
        clip_inputs=[(video_clip, "/tmp/video.mp4")],
        clip_has_audio_flags=[False],
        bg_audio_inputs=[],
        bg_has_audio_flags=[],
        output_path="/tmp/out.mp4",
        export_settings=ExportSettings(format="mp4", resolution="1080p", fps=30, quality="high"),
        overlay_inputs=[(overlay_clip, "/tmp/broll.mp4")],
        overlay_has_video_flags=[True],
    )
    joined = " ".join(command)
    overlay_idx = joined.find("overlay=(W-w)/2:(H-h)/2")
    drawtext_idx = joined.find("drawtext=text='hello world'")
    assert overlay_idx != -1
    assert drawtext_idx != -1
    assert overlay_idx < drawtext_idx
