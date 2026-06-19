import re
from io import StringIO
from pathlib import Path

from app.render_service import _build_ass_subtitle_file, _resolve_h264_video_encoder, build_ffmpeg_command, run_ffmpeg
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


def _extract_ass_path(command: list[str]) -> Path:
    joined = " ".join(command)
    match = re.search(r"(?:subtitles|ass)='([^']+\.ass)'", joined)
    assert match is not None
    return Path(match.group(1))


def test_build_ass_subtitle_file_uses_ass_centisecond_timestamps() -> None:
    ass_path = Path(
        _build_ass_subtitle_file(
            [{"text": "hello", "start": 0.5, "end": 1.5}],
            out_w=1080,
            out_h=1920,
        )
    )
    content = ass_path.read_text(encoding="utf-8")
    ass_path.unlink(missing_ok=True)
    assert "0:00:00.50" in content
    assert "0:00:01.50" in content
    assert "0:00:00.500" not in content


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


def test_build_ffmpeg_command_uses_landscape_resolution_by_default() -> None:
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
    assert "scale=1280:720" in joined
    assert "force_original_aspect_ratio=decrease" in joined
    assert "pad=1280:720:(ow-iw)/2:(oh-ih)/2" in joined


def test_build_ffmpeg_command_uses_portrait_resolution_when_requested() -> None:
    state = _timeline()
    command = build_ffmpeg_command(
        timeline=state,
        clip_inputs=[(state.tracks[0].clips[0], "/tmp/video.mp4")],
        clip_has_audio_flags=[False],
        bg_audio_inputs=[],
        bg_has_audio_flags=[],
        output_path="/tmp/out.mp4",
        export_settings=ExportSettings(format="mp4", aspect_ratio="9:16", resolution="720p", fps=30, quality="medium"),
    )
    joined = " ".join(command)
    assert "scale=720:1280" in joined
    assert "force_original_aspect_ratio=decrease" in joined
    assert "pad=720:1280:(ow-iw)/2:(oh-ih)/2" in joined


def test_build_ass_subtitle_file_keeps_portrait_captions_in_lower_third() -> None:
    ass_path = Path(
        _build_ass_subtitle_file(
            [
                {
                    "text": "portrait caption",
                    "start": 0.0,
                    "end": 1.0,
                    "font_size": 30,
                    "alignment": 2,
                    "margin_v": 140,
                    "outline_width": 3,
                    "shadow": 2,
                }
            ],
            out_w=1080,
            out_h=1920,
        )
    )
    content = ass_path.read_text(encoding="utf-8")
    ass_path.unlink(missing_ok=True)
    style_line = next(line for line in content.splitlines() if line.startswith("Style: Default,"))
    fields = style_line.split(",")
    font_size = int(fields[2])
    margin_v = int(fields[21])
    assert 80 <= font_size <= 110
    assert 160 <= margin_v <= 250


def test_build_ass_subtitle_file_disables_synthetic_bold_for_indic_fallback_fonts() -> None:
    ass_path = Path(
        _build_ass_subtitle_file(
            [
                {
                    "text": "ಬಿಸಿಲುದರೆ ಎಂದು",
                    "start": 0.0,
                    "end": 1.2,
                    "font_name": "Arial-Bold",
                    "font_size": 30,
                    "highlight_color": "&H0000FF00",
                    "word_timings": [
                        {"text": "ಬಿಸಿಲುದರೆ", "start_tl": 0.0, "end_tl": 0.7},
                        {"text": "ಎಂದು", "start_tl": 0.7, "end_tl": 1.2},
                    ],
                }
            ],
            out_w=1080,
            out_h=1920,
        )
    )
    content = ass_path.read_text(encoding="utf-8")
    ass_path.unlink(missing_ok=True)
    style_line = next(line for line in content.splitlines() if line.startswith("Style: Default,"))
    fields = style_line.split(",")
    assert fields[1] == "Lohit Kannada"
    assert fields[8] == "0"
    assert "{\\c" not in content


def test_build_ass_subtitle_file_keeps_basic_white_primary_for_indic_styles() -> None:
    ass_path = Path(
        _build_ass_subtitle_file(
            [
                {
                    "text": "ಬಿಸಿಲುದರೆ ಎಂದು",
                    "start": 0.0,
                    "end": 1.2,
                    "style": "basic_white",
                    "font_name": "Arial-Bold",
                    "font_size": 30,
                    "color": "&H00FFFFFF",
                    "highlight_color": "&H0000FF00",
                    "word_timings": [
                        {"text": "ಬಿಸಿಲುದರೆ", "start_tl": 0.0, "end_tl": 0.7},
                        {"text": "ಎಂದು", "start_tl": 0.7, "end_tl": 1.2},
                    ],
                }
            ],
            out_w=1080,
            out_h=1920,
        )
    )
    content = ass_path.read_text(encoding="utf-8")
    ass_path.unlink(missing_ok=True)
    style_line = next(line for line in content.splitlines() if line.startswith("Style: Default,"))
    fields = style_line.split(",")
    assert fields[3] == "&H00FFFFFF"
    assert fields[6] == "&H9600FF00"
    assert fields[17] == "4"


def test_build_ass_subtitle_file_uses_highlight_color_as_primary_for_indic_color_led_styles() -> None:
    ass_path = Path(
        _build_ass_subtitle_file(
            [
                {
                    "text": "ಬಿಸಿಲುದರೆ ಎಂದು",
                    "start": 0.0,
                    "end": 1.2,
                    "style": "hormozi_green",
                    "font_name": "Arial-Bold",
                    "font_size": 30,
                    "color": "&H00FFFFFF",
                    "highlight_color": "&H0000FF00",
                    "word_timings": [
                        {"text": "ಬಿಸಿಲುದರೆ", "start_tl": 0.0, "end_tl": 0.7},
                        {"text": "ಎಂದು", "start_tl": 0.7, "end_tl": 1.2},
                    ],
                }
            ],
            out_w=1080,
            out_h=1920,
        )
    )
    content = ass_path.read_text(encoding="utf-8")
    ass_path.unlink(missing_ok=True)
    style_line = next(line for line in content.splitlines() if line.startswith("Style: Default,"))
    fields = style_line.split(",")
    assert fields[3] == "&H0000FF00"
    assert fields[6] == "&H9600FF00"


def test_build_ffmpeg_command_uses_fast_preset_for_low_quality(monkeypatch) -> None:
    state = _timeline()
    monkeypatch.setattr("app.render_service._resolve_h264_video_encoder", lambda: "libx264")
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


def test_build_ffmpeg_command_uses_nvenc_when_enabled(monkeypatch) -> None:
    state = _timeline()
    monkeypatch.setattr("app.render_service._resolve_h264_video_encoder", lambda: "h264_nvenc")
    command = build_ffmpeg_command(
        timeline=state,
        clip_inputs=[(state.tracks[0].clips[0], "/tmp/video.mp4")],
        clip_has_audio_flags=[False],
        bg_audio_inputs=[],
        bg_has_audio_flags=[],
        output_path="/tmp/out.mp4",
        export_settings=ExportSettings(format="mp4", resolution="720p", fps=24, quality="high"),
    )
    joined = " ".join(command)
    assert "-c:v h264_nvenc" in joined
    assert "-preset p5" in joined
    assert "-cq 21" in joined


def test_resolve_h264_video_encoder_falls_back_when_nvenc_probe_fails(monkeypatch) -> None:
    monkeypatch.setattr("app.render_service._ffmpeg_encoder_usable", lambda encoder_name: False)
    assert _resolve_h264_video_encoder() == "libx264"


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
    assert "ass='" in joined
    assert "shaping=complex" in joined
    assert "drawtext=" not in joined
    ass_path = _extract_ass_path(command)
    ass_content = ass_path.read_text(encoding="utf-8")
    ass_path.unlink(missing_ok=True)
    assert "hello world" in ass_content
    assert "Dialogue: 1,0:00:00.00,0:00:01.20,Default" in ass_content


def test_build_ffmpeg_command_uses_configured_subtitle_font(monkeypatch) -> None:
    state = _timeline()
    video_clip = state.tracks[0].clips[0]
    video_clip.text_overlays = [
        TextOverlay(
            id="ov-font",
            text="ನಮಸ್ಕಾರ",
            start_sec=0.0,
            duration_sec=1.0,
            style="static",
        )
    ]
    monkeypatch.setenv("RENDER_SUBTITLE_FONTFILE", "/tmp/subtitle-font.ttf")
    monkeypatch.setattr("app.render_service._font_exists", lambda path: path == "/tmp/subtitle-font.ttf")
    command = build_ffmpeg_command(
        timeline=state,
        clip_inputs=[(video_clip, "/tmp/video.mp4")],
        clip_has_audio_flags=[False],
        bg_audio_inputs=[],
        bg_has_audio_flags=[],
        output_path="/tmp/out.mp4",
        export_settings=ExportSettings(format="mp4", resolution="1080p", fps=30, quality="high"),
    )
    ass_path = _extract_ass_path(command)
    ass_content = ass_path.read_text(encoding="utf-8")
    ass_path.unlink(missing_ok=True)
    assert "Style: Default," in ass_content
    assert "ನಮಸ್ಕಾರ" in ass_content


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
    assert "force_original_aspect_ratio=decrease" in joined
    assert "pad=1920:1080:(ow-iw)/2:(oh-ih)/2" in joined
    assert "force_original_aspect_ratio=increase" in joined
    assert "crop=1920:1080:(iw-ow)/2:(ih-oh)/2" in joined
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
    subtitles_idx = joined.find("ass='")
    assert overlay_idx != -1
    assert subtitles_idx != -1
    assert overlay_idx < subtitles_idx


def test_build_ffmpeg_command_uses_filter_complex_script_for_large_graph(monkeypatch) -> None:
    state = _timeline()
    video_clip = state.tracks[0].clips[0]
    video_clip.text_overlays = [
        TextOverlay(
            id="ov-long",
            text="long subtitle text " * 200,
            start_sec=0.0,
            duration_sec=4.0,
            style="static",
        )
    ]
    monkeypatch.setattr("app.render_service._MAX_INLINE_FILTER_COMPLEX_CHARS", 64)
    command = build_ffmpeg_command(
        timeline=state,
        clip_inputs=[(video_clip, "/tmp/video.mp4")],
        clip_has_audio_flags=[False],
        bg_audio_inputs=[],
        bg_has_audio_flags=[],
        output_path="/tmp/out.mp4",
        export_settings=ExportSettings(format="mp4", resolution="1080p", fps=30, quality="high"),
    )
    assert "-filter_complex_script" in command
    script_path = Path(command[command.index("-filter_complex_script") + 1])
    script_content = script_path.read_text(encoding="utf-8")
    assert "ass='" in script_content
    assert "shaping=complex" in script_content
    ass_path = _extract_ass_path(["ffmpeg", "-filter_complex", script_content])
    ass_content = ass_path.read_text(encoding="utf-8")
    script_path.unlink(missing_ok=True)
    ass_path.unlink(missing_ok=True)
    assert "long subtitle text" in ass_content


def test_run_ffmpeg_cleans_temp_filter_script(monkeypatch, tmp_path) -> None:
    script_path = tmp_path / "graph.txt"
    script_path.write_text("[0:v]null[v0]", encoding="utf-8")

    class _Process:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = StringIO("progress=end\n")
            self.stderr = StringIO("")

        def wait(self) -> int:
            return self.returncode

    def _fake_popen(command: list[str], stdout, stderr, text: bool, bufsize: int) -> _Process:
        assert "-filter_complex_script" in command
        assert "-progress" in command
        return _Process()

    monkeypatch.setattr("app.render_service.subprocess.Popen", _fake_popen)
    run_ffmpeg(["ffmpeg", "-filter_complex_script", str(script_path), "-f", "null", "-"])
    assert not script_path.exists()
