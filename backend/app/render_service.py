from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Iterable

from .config import get_settings
from .schemas import Clip, ExportSettings, Resolution, TimelineState

settings = get_settings()


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _resolution_dims(resolution: str, timeline_resolution: Resolution) -> tuple[int, int]:
    short_side_map = {
        "720p": 720,
        "1080p": 1080,
        "4k": 2160,
    }
    short_side = short_side_map[resolution]
    src_w = max(timeline_resolution.width, 2)
    src_h = max(timeline_resolution.height, 2)
    scale = short_side / min(src_w, src_h)
    out_w = _even(int(round(src_w * scale)))
    out_h = _even(int(round(src_h * scale)))
    return out_w, out_h


def _quality_to_crf(quality: str) -> int:
    mapping = {
        "low": 30,
        "medium": 25,
        "high": 20,
        "max": 16,
    }
    return mapping[quality]


def _quality_to_x264_preset(quality: str) -> str:
    mapping = {
        "low": "ultrafast",
        "medium": "veryfast",
        "high": "medium",
        "max": "slow",
    }
    return mapping[quality]


def _clip_duration(clip: Clip) -> float:
    return (clip.end_sec - clip.start_sec) / max(clip.speed, 0.01)


def _escape_drawtext(text: str) -> str:
    # Older ffmpeg builds are fragile around single quotes inside drawtext text values.
    # Normalize apostrophes out to avoid breaking filter parsing.
    normalized = text.replace("'", "")
    return (
        normalized.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace(",", "\\,")
        .replace("%", "\\%")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(";", "\\;")
    )


def _escape_drawtext_expr(expr: str) -> str:
    return (
        expr.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace(",", "\\,")
        .replace("'", "\\'")
    )


def _float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _atempo_chain(speed: float) -> str:
    # FFmpeg atempo accepts [0.5, 2.0], so decompose out-of-range speeds.
    if speed <= 0:
        return "atempo=1.0"
    factors: list[float] = []
    remaining = speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.5f}".rstrip("0").rstrip(".") for factor in factors)


def _apply_preset_filters(chain: list[str], preset: str | None) -> None:
    if not preset:
        return
    key = preset.strip().lower()
    if key == "warm":
        chain.append("colorbalance=rs=0.08:bs=-0.05")
    elif key == "cool":
        chain.append("colorbalance=rs=-0.06:bs=0.08")
    elif key == "cinematic":
        chain.append("eq=contrast=1.15:saturation=0.9")
        chain.append("vignette=angle=PI/4")
    elif key == "vintage":
        chain.append("curves=preset=vintage")
    elif key in {"mono", "blackwhite", "b&w"}:
        chain.append("hue=s=0")


def _video_filters_for_clip(clip: Clip, out_w: int, out_h: int, fps: int) -> str:
    chain: list[str] = []
    if clip.transform.crop:
        crop = clip.transform.crop
        chain.append(f"crop={crop.width}:{crop.height}:{crop.x}:{crop.y}")
    if clip.transform.rotate == 90:
        chain.append("transpose=1")
    elif clip.transform.rotate == 180:
        chain.append("transpose=1,transpose=1")
    elif clip.transform.rotate == 270:
        chain.append("transpose=2")
    if clip.transform.flip == "horizontal":
        chain.append("hflip")
    elif clip.transform.flip == "vertical":
        chain.append("vflip")
    if clip.speed != 1:
        chain.append(f"setpts=PTS/{clip.speed}")

    adj = clip.adjustments
    brightness = _float(adj.brightness + (adj.exposure * 0.35), -1.0, 1.0)
    contrast = _float(adj.contrast, 0.0, 3.0)
    saturation = _float(adj.saturation, 0.0, 4.0)
    chain.append(
        "eq="
        f"brightness={brightness:.4f}:"
        f"contrast={contrast:.4f}:"
        f"saturation={saturation:.4f}"
    )

    _apply_preset_filters(chain, adj.preset)
    temperature = _float(adj.temperature, -1.0, 1.0)
    if abs(temperature) > 0.001:
        rs = _float(temperature * 0.35, -1.0, 1.0)
        bs = _float(-temperature * 0.35, -1.0, 1.0)
        chain.append(f"colorbalance=rs={rs:.4f}:bs={bs:.4f}")

    chain.append(f"fps={fps}")
    chain.append(f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease")
    chain.append(f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2")
    chain.append("format=yuv420p")
    return ",".join(chain)


def _transition_duration(previous: Clip, current: Clip) -> float:
    if not current.transition:
        return 0.0
    raw = max(0.0, float(current.transition.duration_sec))
    max_allowed = min(_clip_duration(previous), _clip_duration(current)) * 0.45
    return _float(raw, 0.0, max_allowed)


def _timeline_layout(clips: list[Clip]) -> tuple[list[float], list[float]]:
    starts: list[float] = []
    transitions: list[float] = []
    if not clips:
        return starts, transitions
    starts.append(0.0)
    transitions.append(0.0)
    for idx in range(1, len(clips)):
        prev = clips[idx - 1]
        current = clips[idx]
        d = _transition_duration(prev, current)
        start = starts[idx - 1] + _clip_duration(prev) - d
        starts.append(max(0.0, round(start, 3)))
        transitions.append(round(d, 3))
    return starts, transitions


def _collect_text_overlays(clips: Iterable[Clip], render_starts: dict[str, float]) -> list[tuple[str, float, float, str, str, int, str, str]]:
    overlays: list[tuple[str, float, float, str, str, int, str, str]] = []
    for clip in clips:
        base = render_starts.get(clip.id, clip.timeline_start_sec)
        for item in clip.text_overlays:
            start = base + item.start_sec
            end = start + item.duration_sec
            overlays.append((item.text, start, end, item.x, item.y, item.font_size, item.style, item.color))
    return overlays


def _style_drawtext_options(style: str, start: float, end: float, font_size: int, x: str, y: str, color: str) -> str:
    normalized = style.lower()
    if normalized == "fade":
        fade_in = min(0.25, max(end - start, 0.1) * 0.3)
        fade_out = min(0.25, max(end - start, 0.1) * 0.3)
        alpha = (
            f"if(lt(t,{start:.3f}),0,"
            f"if(lt(t,{start + fade_in:.3f}),(t-{start:.3f})/{fade_in:.3f},"
            f"if(lt(t,{max(end - fade_out, start):.3f}),1,({end:.3f}-t)/{fade_out:.3f})))"
        )
        return f"x={x}:y={y}:fontsize={font_size}:fontcolor={color}:alpha='{_escape_drawtext_expr(alpha)}'"
    if normalized == "pop":
        pop_end = start + 0.3
        size_expr = (
            f"if(lt(t,{pop_end:.3f}),"
            f"{font_size}*(1.35-0.35*((t-{start:.3f})/0.30)),{font_size})"
        )
        return (
            f"x={x}:y={y}:fontsize='{_escape_drawtext_expr(size_expr)}':"
            f"fontcolor={color}:borderw=2:bordercolor=black@0.5"
        )
    if normalized == "bounce":
        y_expr = f"{y}+18*sin((t-{start:.3f})*12)"
        return f"x={x}:y='{_escape_drawtext_expr(y_expr)}':fontsize={font_size}:fontcolor={color}:borderw=2:bordercolor=black@0.6"
    if normalized == "typewriter":
        alpha = f"if(lt(t,{start + 0.08:.3f}),0,1)"
        return f"x={x}:y={y}:fontsize={font_size}:fontcolor={color}:alpha='{_escape_drawtext_expr(alpha)}'"
    if normalized == "karaoke":
        # Avoid fontcolor_expr here: some ffmpeg builds accept it but render nothing.
        # Pulse alpha instead, while keeping a standard fontcolor path.
        pulse_alpha = f"if(lt(mod(t-{start:.3f},0.45),0.22),1,0.78)"
        return (
            f"x={x}:y={y}:fontsize={font_size}:fontcolor={color}:"
            f"alpha='{_escape_drawtext_expr(pulse_alpha)}':"
            "borderw=2:bordercolor=black@0.6"
        )
    return f"x={x}:y={y}:fontsize={font_size}:fontcolor={color}:borderw=2:bordercolor=black@0.5"


def _volume_expression(clip: Clip) -> str:
    if clip.audio.mute:
        return "0"
    if not clip.audio.keyframes:
        return f"{max(clip.audio.volume, 0.0):.4f}"
    points = [(0.0, max(clip.audio.volume, 0.0))]
    for keyframe in sorted(clip.audio.keyframes, key=lambda item: item.time_sec):
        points.append((max(0.0, keyframe.time_sec), max(0.0, keyframe.volume)))
    dedup: list[tuple[float, float]] = []
    for time_sec, volume in points:
        if dedup and abs(dedup[-1][0] - time_sec) < 1e-6:
            dedup[-1] = (time_sec, volume)
        else:
            dedup.append((time_sec, volume))
    points = dedup
    if len(points) == 1:
        return f"{points[0][1]:.4f}"
    expr = f"{points[-1][1]:.4f}"
    for idx in range(len(points) - 2, -1, -1):
        t0, v0 = points[idx]
        t1, v1 = points[idx + 1]
        if t1 <= t0:
            continue
        span = max(t1 - t0, 0.001)
        lerp = f"({v0:.4f}+({(v1 - v0):.4f})*(t-{t0:.3f})/{span:.3f})"
        expr = f"if(lt(t,{t1:.3f}),{lerp},{expr})"
    return expr


def _xfade_transition(name: str | None) -> str:
    mapping = {
        "fade": "fade",
        "dissolve": "dissolve",
        "slide_left": "slideleft",
        "slide_right": "slideright",
        "slide_up": "slideup",
        "slide_down": "slidedown",
        "wipe": "wipeleft",
        "zoom": "zoomin",
    }
    if not name:
        return "fade"
    return mapping.get(name, "fade")


def build_ffmpeg_command(
    timeline: TimelineState,
    clip_inputs: list[tuple[Clip, str]],
    clip_has_audio_flags: list[bool],
    bg_audio_inputs: list[tuple[Clip, str]],
    bg_has_audio_flags: list[bool],
    output_path: str,
    export_settings: ExportSettings,
    overlay_inputs: list[tuple[Clip, str]] | None = None,
    overlay_has_video_flags: list[bool] | None = None,
) -> list[str]:
    out_w, out_h = _resolution_dims(export_settings.resolution, timeline.resolution)
    fps = export_settings.fps
    overlay_inputs = list(overlay_inputs or [])
    if overlay_has_video_flags is None:
        overlay_has_video_flags = [True for _ in overlay_inputs]
    else:
        overlay_has_video_flags = list(overlay_has_video_flags)

    if not clip_inputs:
        raise ValueError("No video clips in timeline")
    if len(clip_inputs) != len(clip_has_audio_flags):
        raise ValueError("clip_has_audio_flags length mismatch")
    if len(overlay_inputs) != len(overlay_has_video_flags):
        raise ValueError("overlay_has_video_flags length mismatch")
    if len(bg_audio_inputs) != len(bg_has_audio_flags):
        raise ValueError("bg_has_audio_flags length mismatch")

    clip_pairs = sorted(zip(clip_inputs, clip_has_audio_flags, strict=True), key=lambda item: item[0][0].timeline_start_sec)
    clip_inputs = [pair[0] for pair in clip_pairs]
    clip_has_audio_flags = [pair[1] for pair in clip_pairs]
    overlay_pairs = sorted(
        zip(overlay_inputs, overlay_has_video_flags, strict=True),
        key=lambda item: item[0][0].timeline_start_sec,
    )
    overlay_inputs = [pair[0] for pair in overlay_pairs]
    overlay_has_video_flags = [pair[1] for pair in overlay_pairs]
    bg_pairs = sorted(zip(bg_audio_inputs, bg_has_audio_flags, strict=True), key=lambda item: item[0][0].timeline_start_sec)
    bg_audio_inputs = [pair[0] for pair in bg_pairs]
    bg_has_audio_flags = [pair[1] for pair in bg_pairs]
    ordered_clips = [clip for clip, _ in clip_inputs]
    clip_starts, transition_durations = _timeline_layout(ordered_clips)
    render_starts = {clip.id: clip_starts[idx] for idx, clip in enumerate(ordered_clips)}

    cmd = [settings.ffmpeg_bin, "-y"]
    for clip, src in clip_inputs:
        cmd.extend(
            [
                "-ss",
                f"{clip.start_sec}",
                "-to",
                f"{clip.end_sec}",
                "-i",
                src,
            ]
        )
    for clip, src in overlay_inputs:
        cmd.extend(
            [
                "-ss",
                f"{clip.start_sec}",
                "-to",
                f"{clip.end_sec}",
                "-i",
                src,
            ]
        )
    for clip, src in bg_audio_inputs:
        cmd.extend(
            [
                "-ss",
                f"{clip.start_sec}",
                "-to",
                f"{clip.end_sec}",
                "-i",
                src,
            ]
        )

    filter_parts: list[str] = []
    for idx, (clip, _src) in enumerate(clip_inputs):
        vf = _video_filters_for_clip(clip, out_w, out_h, fps)
        filter_parts.append(f"[{idx}:v]{vf}[v{idx}]")
        duration = max(_clip_duration(clip), 0.1)
        if clip_has_audio_flags[idx]:
            af = f"[{idx}:a]atrim=duration={duration:.3f},asetpts=PTS-STARTPTS"
            if clip.speed != 1:
                af += f",{_atempo_chain(clip.speed)}"
            af += ",aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"
            af += f",volume='{_volume_expression(clip)}'"
            if clip.audio.fade_in_sec > 0:
                af += f",afade=t=in:st=0:d={clip.audio.fade_in_sec:.3f}"
            if clip.audio.fade_out_sec > 0:
                fade_start = max(duration - clip.audio.fade_out_sec, 0.0)
                af += f",afade=t=out:st={fade_start:.3f}:d={clip.audio.fade_out_sec:.3f}"
            af += f"[va{idx}]"
        else:
            af = f"anullsrc=r=48000:cl=stereo,atrim=duration={duration:.3f},aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[va{idx}]"
        filter_parts.append(af)

    if len(clip_inputs) == 1:
        filter_parts.append("[v0]null[vmain]")
        filter_parts.append("[va0]anull[amain]")
    else:
        previous_video = "v0"
        previous_audio = "va0"
        current_end = _clip_duration(ordered_clips[0])
        for idx in range(1, len(clip_inputs)):
            duration = transition_durations[idx]
            current_video = f"v{idx}"
            current_audio = f"va{idx}"
            next_video = f"vx{idx}"
            next_audio = f"ax{idx}"
            if duration > 0.03:
                transition = _xfade_transition(ordered_clips[idx].transition.type if ordered_clips[idx].transition else None)
                offset = max(current_end - duration, 0.0)
                filter_parts.append(
                    f"[{previous_video}][{current_video}]xfade=transition={transition}:duration={duration:.3f}:offset={offset:.3f}[{next_video}]"
                )
                filter_parts.append(
                    f"[{previous_audio}][{current_audio}]acrossfade=d={duration:.3f}:c1=tri:c2=tri[{next_audio}]"
                )
                current_end = current_end + _clip_duration(ordered_clips[idx]) - duration
            else:
                filter_parts.append(
                    f"[{previous_video}][{previous_audio}][{current_video}][{current_audio}]concat=n=2:v=1:a=1[{next_video}][{next_audio}]"
                )
                current_end = current_end + _clip_duration(ordered_clips[idx])
            previous_video = next_video
            previous_audio = next_audio
        filter_parts.append(f"[{previous_video}]null[vmain]")
        filter_parts.append(f"[{previous_audio}]anull[amain]")

    last_video_stream = "vmain"
    if overlay_inputs:
        overlay_base_index = len(clip_inputs)
        for idx, (clip, _src) in enumerate(overlay_inputs):
            if not overlay_has_video_flags[idx]:
                continue
            source_stream_index = overlay_base_index + idx
            overlay_stream = f"ov{idx}"
            vf = _video_filters_for_clip(clip, out_w, out_h, fps)
            filter_parts.append(f"[{source_stream_index}:v]{vf}[{overlay_stream}]")

            opacity = _float(clip.broll_opacity, 0.0, 1.0)
            if opacity < 0.999:
                mixed_stream = f"ovm{idx}"
                filter_parts.append(
                    f"[{overlay_stream}]format=rgba,colorchannelmixer=aa={opacity:.3f}[{mixed_stream}]"
                )
                overlay_stream = mixed_stream

            start = max(0.0, float(clip.timeline_start_sec))
            end = start + max(_clip_duration(clip), 0.1)
            next_stream = f"vov{idx}"
            overlay_enable = _escape_drawtext_expr(f"between(t,{start:.3f},{end:.3f})")
            filter_parts.append(
                f"[{last_video_stream}][{overlay_stream}]"
                f"overlay=(W-w)/2:(H-h)/2:enable='{overlay_enable}'"
                f"[{next_stream}]"
            )
            last_video_stream = next_stream

    text_overlays = _collect_text_overlays([clip for clip, _ in clip_inputs], render_starts)
    text_overlays = sorted(text_overlays, key=lambda item: item[1])
    for idx, (text, start, end, x, y, font_size, style, color) in enumerate(text_overlays):
        src = last_video_stream
        dst = f"vtxt{idx}"
        safe = _escape_drawtext(text)
        style_options = _style_drawtext_options(style, start, end, font_size, x, y, color)
        enable_expr = _escape_drawtext_expr(f"between(t,{start:.3f},{end:.3f})")
        filter_parts.append(
            f"[{src}]drawtext=text='{safe}':"
            f"{style_options}:enable='{enable_expr}'[{dst}]"
        )
        last_video_stream = dst

    has_audio = True
    if bg_audio_inputs:
        base_index = len(clip_inputs) + len(overlay_inputs)
        mix_parts = ["[amain]"]
        for offset, (clip, _src) in enumerate(bg_audio_inputs):
            stream_idx = base_index + offset
            duration = max(_clip_duration(clip), 0.1)
            label = f"bg{offset}"
            if bg_has_audio_flags[offset]:
                chain = f"[{stream_idx}:a]atrim=duration={duration:.3f},asetpts=PTS-STARTPTS"
            else:
                chain = f"anullsrc=r=48000:cl=stereo,atrim=duration={duration:.3f}"
            if clip.speed != 1:
                chain += f",{_atempo_chain(clip.speed)}"
            chain += ",aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"
            chain += f",volume='{_volume_expression(clip)}'"
            if clip.audio.fade_in_sec > 0:
                chain += f",afade=t=in:st=0:d={clip.audio.fade_in_sec:.3f}"
            if clip.audio.fade_out_sec > 0:
                fade_start = max(duration - clip.audio.fade_out_sec, 0.0)
                chain += f",afade=t=out:st={fade_start:.3f}:d={clip.audio.fade_out_sec:.3f}"
            if clip.timeline_start_sec > 0:
                delay_ms = int(round(clip.timeline_start_sec * 1000))
                chain += f",adelay={delay_ms}|{delay_ms}"
            chain += f"[{label}]"
            filter_parts.append(chain)
            mix_parts.append(f"[{label}]")
        filter_parts.append(f"{''.join(mix_parts)}amix=inputs={len(mix_parts)}:duration=longest:normalize=0[aout]")
    else:
        filter_parts.append("[amain]anull[aout]")

    filter_complex = ";".join(filter_parts)
    cmd.extend(["-filter_complex", filter_complex])
    cmd.extend(["-map", f"[{last_video_stream}]"])
    if has_audio:
        cmd.extend(["-map", "[aout]"])
    cmd.extend(["-r", str(fps)])
    cmd.extend(["-crf", str(_quality_to_crf(export_settings.quality))])
    if export_settings.bitrate:
        cmd.extend(["-b:v", export_settings.bitrate])
    if export_settings.format == "webm":
        cmd.extend(["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p"])
        if has_audio:
            cmd.extend(["-c:a", "libopus", "-b:a", "160k"])
        else:
            cmd.extend(["-an"])
    else:
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                _quality_to_x264_preset(export_settings.quality),
                "-pix_fmt",
                "yuv420p",
            ]
        )
        if has_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            cmd.extend(["-an"])
    cmd.append("-shortest")
    cmd.append(output_path)
    return cmd


def run_ffmpeg(command: list[str]) -> None:
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        formatted = " ".join(shlex.quote(part) for part in command)
        raise RuntimeError(
            f"ffmpeg failed ({process.returncode})\n"
            f"command: {formatted}\n"
            f"stderr: {process.stderr.strip()}"
        )


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
