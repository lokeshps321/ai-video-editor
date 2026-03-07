from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlmodel import Session, delete, select

from .models import OperationRecord, Project, Timeline, TimelineVersion
from .schemas import (
    AudioKeyframe,
    Clip,
    Crop,
    CropKeyframe,
    ExportSettings,
    OperationPayload,
    Resolution,
    TextOverlay,
    TimelineState,
    Track,
    Transition,
)

_LEGACY_SUBTITLE_STYLES = {"static", "pop", "bounce", "typewriter", "karaoke", "fade", "creator"}
_CAPTION_STYLE_ALIASES = {
    "kinetic": "hormozi_bold",
    "minimal": "minimalist",
    "bold": "hormozi_bold",
    "gradient": "neon_gamer",
    "outline": "neon_gamer",
    "cinema": "cinematic_serif",
    "cinematic": "cinematic_serif",
    "serif": "cinematic_serif",
    "retro": "retro_vhs",
    "vhs": "retro_vhs",
    "pop": "pop_color",
    "colorful": "pop_color",
    "gold": "elegant_gold",
    "elegant": "elegant_gold",
    "luxury": "elegant_gold",
    "street": "street_impact",
    "impact": "street_impact",
    "urban": "street_impact",
}
_CAPTION_STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "hormozi_bold": {
        "render_style": "hormozi_bold",
        "font_name": "Montserrat-Bold",
        "font_size": 24,
        "primary_color": "&H0000FFFF",
        "highlight_color": "&H0000FFFF",
        "outline_color": "&H00000000",
        "outline_width": 2,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 50,
        "max_words_per_caption": 1,
        "max_gap_sec": 0.18,
        "max_caption_duration_sec": 0.95,
        "max_caption_display_sec": 1.05,
        "caption_guard_sec": 0.002,
        "min_duration_sec": 0.06,
    },
    "neon_gamer": {
        "render_style": "neon_gamer",
        "font_name": "Orbitron-Regular",
        "font_size": 22,
        "primary_color": "&H00FFFFFF",
        "highlight_color": "&H00FF0000",
        "outline_color": "&H00FF0000",
        "outline_width": 1,
        "shadow": 2,
        "alignment": 2,
        "margin_v": 50,
        "max_words_per_caption": 1,
        "max_gap_sec": 0.18,
        "max_caption_duration_sec": 0.95,
        "max_caption_display_sec": 1.05,
        "caption_guard_sec": 0.002,
        "min_duration_sec": 0.06,
    },
    "minimalist": {
        "render_style": "minimalist",
        "font_name": "Helvetica-Light",
        "font_size": 18,
        "primary_color": "&H00FFFFFF",
        "highlight_color": "&H00EEEEEE",
        "outline_color": "&H00000000",
        "outline_width": 0,
        "shadow": 1,
        "alignment": 2,
        "margin_v": 30,
        "max_words_per_caption": 4,
        "max_gap_sec": 0.45,
        "max_caption_duration_sec": 2.2,
        "max_caption_display_sec": 2.6,
        "caption_guard_sec": 0.01,
        "min_duration_sec": 0.12,
    },
    "cinematic_serif": {
        "render_style": "cinematic_serif",
        "font_name": "Georgia",
        "font_size": 20,
        "primary_color": "&H00E0E8FF",
        "highlight_color": "&H0000D4FF",
        "outline_color": "&H00000000",
        "outline_width": 1,
        "shadow": 2,
        "alignment": 2,
        "margin_v": 40,
        "max_words_per_caption": 3,
        "max_gap_sec": 0.40,
        "max_caption_duration_sec": 1.8,
        "max_caption_display_sec": 2.2,
        "caption_guard_sec": 0.008,
        "min_duration_sec": 0.10,
    },
    "retro_vhs": {
        "render_style": "retro_vhs",
        "font_name": "Courier-Bold",
        "font_size": 20,
        "primary_color": "&H00FFFF00",
        "highlight_color": "&H00FF00FF",
        "outline_color": "&H00800040",
        "outline_width": 2,
        "shadow": 1,
        "alignment": 2,
        "margin_v": 45,
        "max_words_per_caption": 1,
        "max_gap_sec": 0.15,
        "max_caption_duration_sec": 0.85,
        "max_caption_display_sec": 0.95,
        "caption_guard_sec": 0.002,
        "min_duration_sec": 0.05,
    },
    "pop_color": {
        "render_style": "pop_color",
        "font_name": "Poppins-Bold",
        "font_size": 24,
        "primary_color": "&H008080FF",
        "highlight_color": "&H0000FF80",
        "outline_color": "&H00000000",
        "outline_width": 2,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 50,
        "max_words_per_caption": 2,
        "max_gap_sec": 0.22,
        "max_caption_duration_sec": 1.1,
        "max_caption_display_sec": 1.4,
        "caption_guard_sec": 0.005,
        "min_duration_sec": 0.08,
    },
    "elegant_gold": {
        "render_style": "elegant_gold",
        "font_name": "Playfair-Display",
        "font_size": 20,
        "primary_color": "&H0000D4FF",
        "highlight_color": "&H0000E8FF",
        "outline_color": "&H00101010",
        "outline_width": 1,
        "shadow": 3,
        "alignment": 2,
        "margin_v": 35,
        "max_words_per_caption": 4,
        "max_gap_sec": 0.50,
        "max_caption_duration_sec": 2.4,
        "max_caption_display_sec": 2.8,
        "caption_guard_sec": 0.01,
        "min_duration_sec": 0.12,
    },
    "street_impact": {
        "render_style": "street_impact",
        "font_name": "Impact",
        "font_size": 26,
        "primary_color": "&H00FFFFFF",
        "highlight_color": "&H000000FF",
        "outline_color": "&H00000000",
        "outline_width": 4,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 50,
        "max_words_per_caption": 1,
        "max_gap_sec": 0.16,
        "max_caption_duration_sec": 0.90,
        "max_caption_display_sec": 1.0,
        "caption_guard_sec": 0.002,
        "min_duration_sec": 0.06,
    },
}


def _normalize_subtitle_style(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _CAPTION_STYLE_ALIASES.get(normalized, normalized)


def _as_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _as_float(value: object, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _merged_caption_style_presets(raw: object) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {
        style_id: dict(config) for style_id, config in _CAPTION_STYLE_PRESETS.items()
    }
    if not isinstance(raw, dict):
        return merged

    for style_key, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        style_id = _normalize_subtitle_style(style_key)
        if not style_id:
            continue
        base = dict(merged.get(style_id, {}))
        base.setdefault("render_style", style_id)
        base.setdefault("font_size", 48)
        base.setdefault("primary_color", "white")
        base.setdefault("highlight_color", base["primary_color"])
        base.setdefault("outline_color", "black@0.5")
        base.setdefault("outline_width", 2)
        base.setdefault("shadow", 0)
        base.setdefault("alignment", 2)
        base.setdefault("margin_v", 80)
        base.setdefault("max_words_per_caption", 3)
        base.setdefault("max_gap_sec", 0.35)
        base.setdefault("max_caption_duration_sec", 1.15)
        base.setdefault("max_caption_display_sec", 1.8)
        base.setdefault("caption_guard_sec", 0.01)
        base.setdefault("min_duration_sec", 0.12)

        font_name = payload.get("font_name", base.get("font_name"))
        if font_name is not None:
            normalized_font = str(font_name).strip()
            base["font_name"] = normalized_font or None

        if "render_style" in payload:
            render_style = _normalize_subtitle_style(payload.get("render_style"))
            if render_style:
                base["render_style"] = render_style
        if "font_size" in payload:
            base["font_size"] = _as_int(payload.get("font_size"), int(base["font_size"]), 10, 160)
        if "primary_color" in payload:
            base["primary_color"] = str(payload.get("primary_color") or base["primary_color"])
        if "highlight_color" in payload:
            base["highlight_color"] = str(payload.get("highlight_color") or base["highlight_color"])
        if "outline_color" in payload:
            base["outline_color"] = str(payload.get("outline_color") or base["outline_color"])
        if "outline_width" in payload:
            base["outline_width"] = _as_int(payload.get("outline_width"), int(base["outline_width"]), 0, 12)
        if "shadow" in payload:
            base["shadow"] = _as_int(payload.get("shadow"), int(base["shadow"]), 0, 12)
        if "alignment" in payload:
            base["alignment"] = _as_int(payload.get("alignment"), int(base["alignment"]), 1, 9)
        if "margin_v" in payload:
            base["margin_v"] = _as_int(payload.get("margin_v"), int(base["margin_v"]), 0, 500)

        if "max_words_per_caption" in payload:
            base["max_words_per_caption"] = _as_int(
                payload.get("max_words_per_caption"),
                int(base["max_words_per_caption"]),
                1,
                8,
            )
        if "max_gap_sec" in payload:
            base["max_gap_sec"] = _as_float(
                payload.get("max_gap_sec"),
                float(base["max_gap_sec"]),
                0.03,
                2.0,
            )
        if "max_caption_duration_sec" in payload:
            base["max_caption_duration_sec"] = _as_float(
                payload.get("max_caption_duration_sec"),
                float(base["max_caption_duration_sec"]),
                0.25,
                8.0,
            )
        if "max_caption_display_sec" in payload:
            base["max_caption_display_sec"] = _as_float(
                payload.get("max_caption_display_sec"),
                float(base["max_caption_display_sec"]),
                0.25,
                10.0,
            )
        if "caption_guard_sec" in payload:
            base["caption_guard_sec"] = _as_float(
                payload.get("caption_guard_sec"),
                float(base["caption_guard_sec"]),
                0.0,
                0.1,
            )
        if "min_duration_sec" in payload:
            base["min_duration_sec"] = _as_float(
                payload.get("min_duration_sec"),
                float(base["min_duration_sec"]),
                0.03,
                1.0,
            )
        merged[style_id] = base
    return merged


def _caption_x_expression(alignment: int) -> str:
    if alignment in {1, 4, 7}:
        return "40"
    if alignment in {3, 6, 9}:
        return "(w-text_w)-40"
    return "(w-text_w)/2"


def _caption_y_expression(alignment: int, margin_v: int) -> str:
    if alignment in {1, 2, 3}:
        # Keep captions visible above player controls and inside active picture
        # area when videos are letterboxed/pillarboxed. Use a comma-free min()
        # equivalent because drawtext option parsing is fragile with unescaped commas.
        a = f"((h-text_h)-{margin_v})"
        b = "((h*0.70)-(text_h/2))"
        return f"(({a}+{b}-abs({a}-{b}))/2)"
    if alignment in {4, 5, 6}:
        return "(h-text_h)/2"
    return str(margin_v)


def _coalesce_caption_chunks(
    chunks: list[list[dict[str, Any]]],
    max_caption_overlays: int,
) -> list[list[dict[str, Any]]]:
    if max_caption_overlays <= 0 or len(chunks) <= max_caption_overlays:
        return chunks
    group_size = max(1, (len(chunks) + max_caption_overlays - 1) // max_caption_overlays)
    merged: list[list[dict[str, Any]]] = []
    for index in range(0, len(chunks), group_size):
        combined: list[dict[str, Any]] = []
        for part in chunks[index : index + group_size]:
            combined.extend(part)
        if combined:
            merged.append(combined)
    return merged


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def make_default_timeline(project: Project) -> TimelineState:
    return TimelineState(
        fps=project.fps,
        resolution=Resolution(width=project.width, height=project.height),
        tracks=[
            Track(id=str(uuid4()), kind="video", clips=[]),
            Track(id=str(uuid4()), kind="audio", clips=[]),
            Track(id=str(uuid4()), kind="overlay", clips=[]),
        ],
    )


def create_timeline_for_project(session: Session, project: Project) -> Timeline:
    state = make_default_timeline(project)
    timeline = Timeline(
        project_id=project.id,
        version=0,
        state_json=state.model_dump_json(),
    )
    session.add(timeline)
    session.add(
        TimelineVersion(
            project_id=project.id,
            version=0,
            state_json=state.model_dump_json(),
        )
    )
    session.commit()
    session.refresh(timeline)
    return timeline


def get_timeline_row(session: Session, project_id: str) -> Timeline:
    timeline = session.exec(select(Timeline).where(Timeline.project_id == project_id)).first()
    if not timeline:
        raise ValueError("timeline not found")
    return timeline


def load_timeline_state(timeline: Timeline) -> TimelineState:
    return TimelineState.model_validate_json(timeline.state_json)


def save_timeline_state(
    session: Session,
    timeline: Timeline,
    state: TimelineState,
    *,
    source: str,
    operation: OperationPayload | None = None,
) -> Timeline:
    # If user edited after undo, drop forward history and start a new linear version chain.
    session.exec(
        delete(TimelineVersion).where(
            TimelineVersion.project_id == timeline.project_id,
            TimelineVersion.version > timeline.version,
        )
    )
    timeline.version += 1
    timeline.state_json = state.model_dump_json()
    timeline.updated_at = _utcnow()
    session.add(timeline)
    session.add(
        TimelineVersion(
            project_id=timeline.project_id,
            version=timeline.version,
            state_json=timeline.state_json,
        )
    )
    if operation is not None:
        session.add(
            OperationRecord(
                project_id=timeline.project_id,
                op_type=operation.op_type,
                source=source,
                payload_json=_json_dumps(operation.model_dump()),
            )
        )
    session.commit()
    session.refresh(timeline)
    return timeline


def _primary_track(state: TimelineState, kind: str = "video") -> Track:
    for track in state.tracks:
        if track.kind == kind:
            return track
    track = Track(id=str(uuid4()), kind=kind, clips=[])
    state.tracks.append(track)
    return track


def _track_by_ref(state: TimelineState, track_ref: str | None = None, kind: str | None = None) -> Track:
    if track_ref is not None:
        normalized = str(track_ref).strip()
        if normalized:
            for track in state.tracks:
                if track.id == normalized:
                    return track
            raise ValueError(f"track not found: {normalized}")
    if kind is None:
        raise ValueError("track kind is required when track_id is not provided")
    return _primary_track(state, kind)


def _clip_duration_on_timeline(clip: Clip) -> float:
    return max((clip.end_sec - clip.start_sec) / max(clip.speed, 0.01), 0.0)


def _recalculate_duration(state: TimelineState) -> None:
    max_t = 0.0
    for track in state.tracks:
        for clip in track.clips:
            max_t = max(max_t, clip.timeline_start_sec + _clip_duration_on_timeline(clip))
    state.duration_sec = round(max_t, 3)


def _ripple_track(track: Track, *, sort_by_timeline: bool = True) -> None:
    cursor = 0.0
    if sort_by_timeline:
        track.clips = sorted(track.clips, key=lambda c: c.timeline_start_sec)
    for clip in track.clips:
        clip.timeline_start_sec = round(cursor, 3)
        cursor += _clip_duration_on_timeline(clip)


def _clip_index_by_ref(state: TimelineState, clip_ref: str | int, track_kind: str | None = None) -> tuple[Track, int]:
    if isinstance(clip_ref, int):
        ordered: list[tuple[Track, int, float]] = []
        for track in state.tracks:
            if track_kind is not None:
                if track.kind != track_kind:
                    continue
            elif track.kind != "video":
                continue
            for idx, clip in enumerate(track.clips):
                ordered.append((track, idx, clip.timeline_start_sec))
        ordered.sort(key=lambda item: item[2])
        if clip_ref <= 0 or clip_ref > len(ordered):
            raise ValueError(f"clip index out of range: {clip_ref}")
        track, idx, _ = ordered[clip_ref - 1]
        return track, idx

    for track in state.tracks:
        if track_kind is not None and track.kind != track_kind:
            continue
        for idx, clip in enumerate(track.clips):
            if clip.id == str(clip_ref):
                return track, idx
    raise ValueError(f"clip not found: {clip_ref}")


def _text_overlay_index_by_ref(
    state: TimelineState,
    overlay_ref: str,
    *,
    clip_ref: str | int | None = None,
) -> tuple[Track, int, int]:
    normalized_clip_ref = _normalize_clip_ref(clip_ref) if clip_ref is not None else None
    clip_filter_id: str | None = None
    if normalized_clip_ref is not None:
        track, clip_idx = _clip_index_by_ref(state, normalized_clip_ref)
        clip_filter_id = track.clips[clip_idx].id

    overlay_id = str(overlay_ref)
    for track in state.tracks:
        if track.kind != "video":
            continue
        for clip_idx, clip in enumerate(track.clips):
            if clip_filter_id is not None and clip.id != clip_filter_id:
                continue
            for overlay_idx, overlay in enumerate(clip.text_overlays):
                if overlay.id == overlay_id:
                    return track, clip_idx, overlay_idx
    raise ValueError(f"text overlay not found: {overlay_id}")


def _normalize_clip_ref(value: Any) -> str | int:
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit():
        return int(text)
    return text


def _ensure_time_window(start_sec: float, end_sec: float) -> tuple[float, float]:
    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec")
    return (round(start_sec, 3), round(end_sec, 3))


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))


def _apply_trim(state: TimelineState, params: dict[str, Any]) -> None:
    track, idx = _clip_index_by_ref(state, _normalize_clip_ref(params["clip"]))
    clip = track.clips[idx]
    start_sec, end_sec = _ensure_time_window(float(params["start_sec"]), float(params["end_sec"]))
    clip.start_sec = start_sec
    clip.end_sec = end_sec


def _apply_split(state: TimelineState, params: dict[str, Any]) -> None:
    track, idx = _clip_index_by_ref(state, _normalize_clip_ref(params["clip"]))
    clip = track.clips[idx]
    at_sec = float(params["at_sec"])
    # Accept split in source-time or timeline-time.
    if clip.start_sec < at_sec < clip.end_sec:
        split_source_sec = at_sec
    else:
        rel = at_sec - clip.timeline_start_sec
        split_source_sec = clip.start_sec + (rel * clip.speed)
    if split_source_sec <= clip.start_sec or split_source_sec >= clip.end_sec:
        raise ValueError("split point outside clip window")

    left = clip.model_copy()
    left.end_sec = round(split_source_sec, 3)

    right = clip.model_copy()
    right.id = str(uuid4())
    right.start_sec = round(split_source_sec, 3)
    right.timeline_start_sec = round(
        clip.timeline_start_sec + _clip_duration_on_timeline(left),
        3,
    )
    track.clips[idx] = left
    track.clips.insert(idx + 1, right)


def _apply_merge(state: TimelineState, params: dict[str, Any]) -> None:
    refs = params.get("clips", [])
    if len(refs) < 2:
        raise ValueError("merge_clips requires at least two clips")
    normalized = [_normalize_clip_ref(item) for item in refs]
    first_track, _ = _clip_index_by_ref(state, normalized[0])

    selected_ids: list[str] = []
    selected: list[Clip] = []
    for ref in normalized:
        track, idx = _clip_index_by_ref(state, ref)
        if track.id != first_track.id:
            raise ValueError("all merged clips must belong to the same track")
        clip = track.clips[idx]
        if clip.id in selected_ids:
            continue
        selected_ids.append(clip.id)
        selected.append(clip)
    if len(selected) < 2:
        raise ValueError("merge_clips requires two unique clips")

    first_start = min(clip.timeline_start_sec for clip in selected)
    cursor = first_start
    for clip in selected:
        clip.timeline_start_sec = round(cursor, 3)
        cursor += _clip_duration_on_timeline(clip)

    first_track.clips = [clip for clip in first_track.clips if clip.id not in selected_ids] + selected
    first_track.clips = sorted(first_track.clips, key=lambda c: c.timeline_start_sec)
    if bool(params.get("gapless", True)):
        _ripple_track(first_track)


def _apply_transition(state: TimelineState, params: dict[str, Any]) -> None:
    track, idx = _clip_index_by_ref(state, _normalize_clip_ref(params["clip"]))
    transition_type = str(params.get("type", "fade")).lower()
    if transition_type in {"none", "off", "clear"}:
        track.clips[idx].transition = None
        return
    transition = Transition(
        type=transition_type,
        duration_sec=float(params.get("duration_sec", 0.5)),
    )
    track.clips[idx].transition = transition


def _apply_add_text_overlay(state: TimelineState, params: dict[str, Any]) -> None:
    clip_ref = params.get("clip")
    if clip_ref is None:
        track = _primary_track(state, "video")
        if not track.clips:
            raise ValueError("no clip available for text overlay")
        clip = sorted(track.clips, key=lambda c: c.timeline_start_sec)[0]
    else:
        track, idx = _clip_index_by_ref(state, _normalize_clip_ref(clip_ref))
        clip = track.clips[idx]

    overlay = TextOverlay(
        id=str(uuid4()),
        text=str(params["text"]),
        start_sec=float(params["start_sec"]),
        duration_sec=float(params.get("duration_sec", 2.0)),
        x=str(params.get("x", "(w-text_w)/2")),
        y=str(params.get("y", "(h-text_h)-80")),
        font_size=int(params.get("font_size", 48)),
        font_name=(str(params.get("font_name", "")).strip() or None),
        color=str(params.get("color", "white")),
        highlight_color=(str(params.get("highlight_color", "")).strip() or None),
        outline_color=str(params.get("outline_color", "black@0.5")),
        outline_width=int(params.get("outline_width", 2)),
        shadow=int(params.get("shadow", 0)),
        alignment=int(params.get("alignment", 2)),
        margin_v=int(params.get("margin_v", 80)),
        style=str(params.get("style", "static")),
    )
    clip.text_overlays.append(overlay)


def _apply_move_text_overlay(state: TimelineState, params: dict[str, Any]) -> None:
    track, clip_idx, overlay_idx = _text_overlay_index_by_ref(
        state,
        str(params["overlay"]),
        clip_ref=params.get("clip"),
    )
    clip = track.clips[clip_idx]
    overlay = clip.text_overlays[overlay_idx]
    clip_source_duration = max(clip.end_sec - clip.start_sec, 0.1)
    max_start = max(0.0, clip_source_duration - overlay.duration_sec)
    overlay.start_sec = round(max(0.0, min(float(params.get("start_sec", overlay.start_sec)), max_start)), 3)


def _apply_trim_text_overlay(state: TimelineState, params: dict[str, Any]) -> None:
    track, clip_idx, overlay_idx = _text_overlay_index_by_ref(
        state,
        str(params["overlay"]),
        clip_ref=params.get("clip"),
    )
    clip = track.clips[clip_idx]
    overlay = clip.text_overlays[overlay_idx]
    clip_source_duration = max(clip.end_sec - clip.start_sec, 0.1)
    start_sec = float(params.get("start_sec", overlay.start_sec))
    duration_sec = float(params.get("duration_sec", overlay.duration_sec))
    start_sec = max(0.0, min(start_sec, clip_source_duration - 0.05))
    max_duration = max(0.05, clip_source_duration - start_sec)
    overlay.start_sec = round(start_sec, 3)
    overlay.duration_sec = round(max(0.05, min(duration_sec, max_duration)), 3)


def _apply_delete_text_overlay(state: TimelineState, params: dict[str, Any]) -> None:
    track, clip_idx, overlay_idx = _text_overlay_index_by_ref(
        state,
        str(params["overlay"]),
        clip_ref=params.get("clip"),
    )
    del track.clips[clip_idx].text_overlays[overlay_idx]


def _apply_add_clip(state: TimelineState, params: dict[str, Any]) -> None:
    track_kind = str(params.get("track_kind", "video"))
    track = _primary_track(state, track_kind)
    start_sec = float(params.get("start_sec", 0.0))
    end_sec = params.get("end_sec")
    if end_sec is None:
        duration = float(params.get("duration_sec", 0))
        if duration <= 0:
            raise ValueError("add_clip requires end_sec or positive duration_sec")
        end_sec = start_sec + duration
    end_sec = float(end_sec)
    _ensure_time_window(start_sec, end_sec)
    timeline_start = float(params.get("timeline_start_sec", state.duration_sec))
    clip = Clip(
        id=str(uuid4()),
        asset_id=str(params["asset_id"]),
        start_sec=round(start_sec, 3),
        end_sec=round(end_sec, 3),
        timeline_start_sec=round(timeline_start, 3),
    )
    track.clips.append(clip)


def _apply_add_audio_track(state: TimelineState, params: dict[str, Any]) -> None:
    params = {**params, "track_kind": "audio"}
    _apply_add_clip(state, params)


def _apply_add_broll_clip(state: TimelineState, params: dict[str, Any]) -> None:
    track = _primary_track(state, "overlay")
    start_sec = float(params.get("start_sec", 0.0))
    end_sec = params.get("end_sec")
    if end_sec is None:
        duration = float(params.get("duration_sec", 0.0))
        if duration <= 0:
            raise ValueError("add_broll_clip requires end_sec or positive duration_sec")
        end_sec = start_sec + duration
    end_sec = float(end_sec)
    _ensure_time_window(start_sec, end_sec)

    timeline_start = float(params.get("timeline_start_sec", max(0.0, start_sec)))
    opacity = _clamp_unit_interval(float(params.get("opacity", params.get("broll_opacity", 1.0))))
    clip = Clip(
        id=str(uuid4()),
        asset_id=str(params["asset_id"]),
        start_sec=round(start_sec, 3),
        end_sec=round(end_sec, 3),
        timeline_start_sec=round(timeline_start, 3),
        broll_opacity=round(opacity, 3),
    )
    crop_payload = params.get("crop")
    if isinstance(crop_payload, dict):
        try:
            clip.transform.crop = Crop(
                x=int(crop_payload.get("x", 0)),
                y=int(crop_payload.get("y", 0)),
                width=int(crop_payload.get("width", 0)),
                height=int(crop_payload.get("height", 0)),
            )
        except (TypeError, ValueError):
            pass
    keyframe_payload = params.get("crop_keyframes")
    if isinstance(keyframe_payload, list):
        keyframes: list[CropKeyframe] = []
        for raw in keyframe_payload:
            if not isinstance(raw, dict):
                continue
            try:
                keyframes.append(
                    CropKeyframe(
                        time_sec=max(0.0, float(raw.get("time_sec", 0.0))),
                        x=int(raw.get("x", 0)),
                        y=int(raw.get("y", 0)),
                    )
                )
            except (TypeError, ValueError):
                continue
        keyframes.sort(key=lambda item: item.time_sec)
        clip.transform.crop_keyframes = keyframes
    preset = params.get("preset")
    if isinstance(preset, str) and preset.strip():
        clip.adjustments.preset = preset.strip()
    # B-roll defaults to visual overlay semantics; keep overlay audio muted.
    clip.audio.mute = True
    track.clips.append(clip)
    track.clips = sorted(track.clips, key=lambda item: item.timeline_start_sec)


def _apply_move_broll_clip(state: TimelineState, params: dict[str, Any]) -> None:
    track, idx = _clip_index_by_ref(state, _normalize_clip_ref(params["clip"]), "overlay")
    clip = track.clips[idx]
    clip.timeline_start_sec = round(float(params.get("timeline_start_sec", clip.timeline_start_sec)), 3)
    track.clips = sorted(track.clips, key=lambda item: item.timeline_start_sec)


def _apply_trim_broll_clip(state: TimelineState, params: dict[str, Any]) -> None:
    track, idx = _clip_index_by_ref(state, _normalize_clip_ref(params["clip"]), "overlay")
    clip = track.clips[idx]
    start_sec = float(params.get("start_sec", clip.start_sec))
    end_sec = float(params.get("end_sec", clip.end_sec))
    start_sec, end_sec = _ensure_time_window(start_sec, end_sec)
    clip.start_sec = start_sec
    clip.end_sec = end_sec


def _apply_delete_broll_clip(state: TimelineState, params: dict[str, Any]) -> None:
    track, idx = _clip_index_by_ref(state, _normalize_clip_ref(params["clip"]), "overlay")
    del track.clips[idx]
    if bool(params.get("ripple", False)):
        _ripple_track(track)


def _apply_set_broll_opacity(state: TimelineState, params: dict[str, Any]) -> None:
    track, idx = _clip_index_by_ref(state, _normalize_clip_ref(params["clip"]), "overlay")
    if "opacity" not in params and "broll_opacity" not in params:
        raise ValueError("set_broll_opacity requires opacity")
    value = params.get("opacity", params.get("broll_opacity"))
    track.clips[idx].broll_opacity = round(_clamp_unit_interval(float(value)), 3)


def _apply_volume(state: TimelineState, params: dict[str, Any]) -> None:
    clip_ref = params.get("clip")
    if clip_ref is not None:
        clip_norm = _normalize_clip_ref(clip_ref)
        track_hint = str(params.get("track_kind", "")).strip().lower() or None
        try:
            track, idx = _clip_index_by_ref(state, clip_norm, track_hint)
        except ValueError:
            if track_hint is None:
                raise
            track, idx = _clip_index_by_ref(state, clip_norm)
        clip = track.clips[idx]
        if "volume" in params:
            clip.audio.volume = float(params["volume"])
        if "fade_in_sec" in params:
            clip.audio.fade_in_sec = max(0.0, float(params["fade_in_sec"]))
        if "fade_out_sec" in params:
            clip.audio.fade_out_sec = max(0.0, float(params["fade_out_sec"]))
        if "mute" in params:
            clip.audio.mute = bool(params["mute"])
        if "keyframes" in params:
            keyframes = params["keyframes"]
            if not isinstance(keyframes, list):
                raise ValueError("keyframes must be a list")
            parsed: list[AudioKeyframe] = []
            for item in keyframes:
                if not isinstance(item, dict):
                    raise ValueError("invalid keyframe payload")
                parsed.append(
                    AudioKeyframe(
                        time_sec=max(0.0, float(item.get("time_sec", 0.0))),
                        volume=max(0.0, float(item.get("volume", 1.0))),
                    )
                )
            parsed.sort(key=lambda entry: entry.time_sec)
            clip.audio.keyframes = parsed
        return

    track_kind = str(params.get("track_kind", "audio"))
    track_id = str(params.get("track_id", "")).strip() or None
    track = _track_by_ref(state, track_id, track_kind)
    if "volume" in params:
        track.volume = max(0.0, float(params.get("volume", 1.0)))
    if "mute" in params:
        track.mute = bool(params["mute"])
    if "solo" in params:
        track.solo = bool(params["solo"])


def _apply_speed(state: TimelineState, params: dict[str, Any]) -> None:
    track, idx = _clip_index_by_ref(state, _normalize_clip_ref(params["clip"]))
    track.clips[idx].speed = max(0.25, min(4.0, float(params["speed"])))


def _apply_aspect_ratio(state: TimelineState, params: dict[str, Any]) -> None:
    ratio = str(params["ratio"]).replace(" ", "")
    mapping = {
        "9:16": (1080, 1920),
        "16:9": (1920, 1080),
        "1:1": (1080, 1080),
        "4:5": (1080, 1350),
    }
    if ratio not in mapping:
        raise ValueError("unsupported aspect ratio")
    width, height = mapping[ratio]
    state.resolution = Resolution(width=width, height=height)


def _apply_crop_resize(state: TimelineState, params: dict[str, Any]) -> None:
    track, idx = _clip_index_by_ref(state, _normalize_clip_ref(params["clip"]))
    clip = track.clips[idx]
    if {"x", "y", "width", "height"} <= params.keys():
        clip.transform.crop = Crop(
            x=int(params["x"]),
            y=int(params["y"]),
            width=int(params["width"]),
            height=int(params["height"]),
        )
    if "width" in params and "height" in params and params.get("apply_to_resolution"):
        state.resolution = Resolution(width=int(params["width"]), height=int(params["height"]))


def _apply_rotate(state: TimelineState, params: dict[str, Any]) -> None:
    track, idx = _clip_index_by_ref(state, _normalize_clip_ref(params["clip"]))
    degrees = int(params["degrees"])
    if degrees not in (0, 90, 180, 270):
        raise ValueError("rotation must be one of: 0, 90, 180, 270")
    track.clips[idx].transform.rotate = degrees


def _apply_flip(state: TimelineState, params: dict[str, Any]) -> None:
    track, idx = _clip_index_by_ref(state, _normalize_clip_ref(params["clip"]))
    direction = str(params["direction"]).lower()
    if direction in ("none", "off", "clear"):
        track.clips[idx].transform.flip = None
        return
    if direction not in ("horizontal", "vertical"):
        raise ValueError("flip direction must be horizontal or vertical")
    track.clips[idx].transform.flip = direction


def _apply_adjustments(state: TimelineState, params: dict[str, Any]) -> None:
    track, idx = _clip_index_by_ref(state, _normalize_clip_ref(params["clip"]))
    clip = track.clips[idx]
    for key in ("brightness", "contrast", "saturation", "exposure", "temperature", "preset"):
        if key in params:
            if key == "preset" and str(params[key]).lower() in {"none", "off", "clear", ""}:
                clip.adjustments.preset = None
            else:
                setattr(clip.adjustments, key, params[key])


def _apply_delete_clip(state: TimelineState, params: dict[str, Any]) -> None:
    track, idx = _clip_index_by_ref(state, _normalize_clip_ref(params["clip"]))
    del track.clips[idx]
    _ripple_track(track)


def _apply_ripple_edit(state: TimelineState, params: dict[str, Any]) -> None:
    track_kind = str(params.get("track_kind", "video"))
    track = _primary_track(state, track_kind)
    _ripple_track(track)


def _apply_export_settings(state: TimelineState, params: dict[str, Any]) -> None:
    state.export_settings = ExportSettings.model_validate(
        {
            **state.export_settings.model_dump(),
            **params,
        }
    )


def _apply_move_clip(state: TimelineState, params: dict[str, Any]) -> None:
    source_track, idx = _clip_index_by_ref(state, _normalize_clip_ref(params["clip"]))
    clip = source_track.clips.pop(idx)
    destination_kind = str(params.get("track_kind", source_track.kind))
    destination_track = _primary_track(state, destination_kind)
    clip.timeline_start_sec = round(float(params.get("timeline_start_sec", clip.timeline_start_sec)), 3)
    destination_track.clips.append(clip)
    destination_track.clips = sorted(destination_track.clips, key=lambda item: item.timeline_start_sec)
    if bool(params.get("ripple", False)):
        _ripple_track(destination_track)
    if source_track.id != destination_track.id and bool(params.get("source_ripple", False)):
        _ripple_track(source_track)


def _apply_reorder_clips(state: TimelineState, params: dict[str, Any]) -> None:
    track_kind = str(params.get("track_kind", "video"))
    track = _primary_track(state, track_kind)
    order = params.get("clip_order", [])
    if not isinstance(order, list) or not order:
        raise ValueError("clip_order must be a non-empty list")

    by_id = {clip.id: clip for clip in track.clips}
    reordered: list[Clip] = []
    seen: set[str] = set()
    for item in order:
        clip_id = str(item)
        clip = by_id.get(clip_id)
        if not clip or clip_id in seen:
            continue
        seen.add(clip_id)
        reordered.append(clip)
    for clip in track.clips:
        if clip.id not in seen:
            reordered.append(clip)
    track.clips = reordered
    if bool(params.get("ripple", True)):
        _ripple_track(track, sort_by_timeline=False)


def _apply_replace_video_track_clips(state: TimelineState, params: dict[str, Any]) -> None:
    asset_id = str(params.get("asset_id", "")).strip()
    if not asset_id:
        raise ValueError("asset_id is required")
    raw_ranges = params.get("ranges")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise ValueError("ranges must be a non-empty list")

    video_track = _primary_track(state, "video")
    rebuilt: list[Clip] = []
    cursor = 0.0
    for item in raw_ranges:
        if not isinstance(item, dict):
            raise ValueError("range items must be objects")
        start_sec, end_sec = _ensure_time_window(float(item["start_sec"]), float(item["end_sec"]))
        clip = Clip(
            id=str(uuid4()),
            asset_id=asset_id,
            start_sec=start_sec,
            end_sec=end_sec,
            timeline_start_sec=round(cursor, 3),
        )
        rebuilt.append(clip)
        cursor += _clip_duration_on_timeline(clip)
    video_track.clips = rebuilt

    if bool(params.get("clear_audio_tracks", False)):
        for track in state.tracks:
            if track.kind == "audio":
                track.clips = []


def _apply_clear_subtitles(state: TimelineState, params: dict[str, Any]) -> None:
    """Remove all text_overlays from video track clips."""
    asset_id = str(params.get("asset_id", "")).strip() or None
    video_track = _primary_track(state, "video")
    for clip in video_track.clips:
        if asset_id and clip.asset_id != asset_id:
            continue
        clip.text_overlays = []


def _apply_set_subtitles(state: TimelineState, params: dict[str, Any]) -> None:
    raw_words = params.get("words")
    if not isinstance(raw_words, list) or not raw_words:
        raise ValueError("words must be a non-empty list")

    asset_id = str(params.get("asset_id", "")).strip() or None
    style = _normalize_subtitle_style(params.get("style", "hormozi_bold")) or "hormozi_bold"
    style_presets = _merged_caption_style_presets(params.get("caption_styles"))
    preset = style_presets.get(style)
    allowed_styles = set(_LEGACY_SUBTITLE_STYLES) | set(style_presets)
    if style not in allowed_styles:
        style = "creator"
        preset = None

    if preset is not None:
        default_max_words = int(preset.get("max_words_per_caption", 3))
        default_max_gap_sec = float(preset.get("max_gap_sec", 0.35))
        default_max_caption_duration_sec = float(preset.get("max_caption_duration_sec", 1.15))
        default_max_caption_display_sec = float(preset.get("max_caption_display_sec", 1.8))
        default_caption_guard_sec = float(preset.get("caption_guard_sec", 0.01))
        min_duration_sec = float(preset.get("min_duration_sec", 0.12))
    else:
        if style == "karaoke":
            default_max_words = 1
            default_max_gap_sec = 0.18
            default_max_caption_duration_sec = 0.9
            default_max_caption_display_sec = 0.95
            default_caption_guard_sec = 0.002
            min_duration_sec = 0.06
        elif style == "creator":
            default_max_words = 4
            default_max_gap_sec = 0.45
            default_max_caption_duration_sec = 2.0
            default_max_caption_display_sec = 2.4
            default_caption_guard_sec = 0.01
            min_duration_sec = 0.12
        else:
            default_max_words = 3
            default_max_gap_sec = 0.35
            default_max_caption_duration_sec = 1.15
            default_max_caption_display_sec = 1.8
            default_caption_guard_sec = 0.01
            min_duration_sec = 0.12

    max_words = int(params.get("max_words_per_caption", default_max_words))
    max_words = max(1, min(max_words, 8))
    max_chars = int(params.get("max_chars_per_caption", 42))
    max_chars = max(10, min(max_chars, 80))
    max_gap_sec = max(0.03, float(params.get("max_gap_sec", default_max_gap_sec)))
    max_caption_duration_sec = max(0.25, float(params.get("max_caption_duration_sec", default_max_caption_duration_sec)))
    max_caption_display_sec = max(0.25, float(params.get("max_caption_display_sec", default_max_caption_display_sec)))
    caption_guard_sec = max(0.0, float(params.get("caption_guard_sec", default_caption_guard_sec)))
    max_caption_display_hard_cap_factor = max(
        1.1,
        float(params.get("max_caption_display_hard_cap_factor", 1.8)),
    )
    max_caption_overlays = _as_int(
        params.get("max_caption_overlays", 0),
        0,
        0,
        5000,
    )
    clear_existing = bool(params.get("clear_existing", True))

    # Punctuation characters that signal sentence/clause boundaries
    sentence_end_chars = {'.', '!', '?'}
    clause_break_chars = {',', ';', ':', '—', '–'}

    min_duration_sec = max(0.03, min(float(min_duration_sec), 1.0))
    render_style = str(preset.get("render_style", style)) if preset is not None else style
    overlay_alignment = _as_int(preset.get("alignment"), 2, 1, 9) if preset is not None else 2
    overlay_margin_v = _as_int(preset.get("margin_v"), 80, 0, 500) if preset is not None else 80
    overlay_x = _caption_x_expression(overlay_alignment) if preset is not None else "(w-text_w)/2"
    overlay_y = (
        _caption_y_expression(overlay_alignment, overlay_margin_v)
        if preset is not None
        else ("(h-text_h)-112" if style == "creator" else "(h-text_h)-80")
    )
    overlay_font_size = (
        _as_int(preset.get("font_size"), 48, 10, 160)
        if preset is not None
        else (52 if style == "creator" else 48)
    )
    overlay_font_name = (
        (str(preset.get("font_name")).strip() or None)
        if preset is not None and preset.get("font_name") is not None
        else None
    )
    overlay_color = str(preset.get("primary_color", "white")) if preset is not None else "white"
    overlay_highlight_color = (
        str(preset.get("highlight_color", overlay_color))
        if preset is not None
        else None
    )
    overlay_outline_color = str(preset.get("outline_color", "black@0.5")) if preset is not None else "black@0.5"
    overlay_outline_width = _as_int(preset.get("outline_width"), 2, 0, 12) if preset is not None else 2
    overlay_shadow = _as_int(preset.get("shadow"), 0, 0, 12) if preset is not None else 0

    video_track = _primary_track(state, "video")
    clip_chunks: list[tuple[Clip, list[list[dict[str, Any]]]]] = []
    for clip in video_track.clips:
        if clear_existing:
            clip.text_overlays = []
        if asset_id and clip.asset_id != asset_id:
            continue

        clip_words: list[dict[str, Any]] = []
        for item in raw_words:
            if not isinstance(item, dict):
                continue
            start_sec = float(item.get("start_sec", 0.0))
            end_sec = float(item.get("end_sec", 0.0))
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            if end_sec <= clip.start_sec or start_sec >= clip.end_sec:
                continue
            clip_words.append(
                {
                    "text": text,
                    "start_sec": max(start_sec, clip.start_sec),
                    "end_sec": min(end_sec, clip.end_sec),
                }
            )
        if not clip_words:
            continue
        clip_words.sort(key=lambda item: float(item["start_sec"]))

        # --- Punctuation-aware chunking ---
        chunks: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        prev_end: float | None = None
        for item in clip_words:
            start_sec = float(item["start_sec"])
            word_text = str(item["text"])
            should_split = False

            # Split on time gap
            if current and prev_end is not None and (start_sec - prev_end) > max_gap_sec:
                should_split = True
            # Split on max word count
            elif current and len(current) >= max_words:
                should_split = True
            # Split on max character count
            elif current:
                running_text = " ".join(str(w["text"]) for w in current) + " " + word_text
                if len(running_text) > max_chars:
                    should_split = True
            # Split when caption would stay on-screen too long (prevents merged phrases).
            if not should_split and current:
                candidate_span = max(float(item["end_sec"]) - float(current[0]["start_sec"]), 0.0)
                if candidate_span > max_caption_duration_sec:
                    should_split = True
            # Split after sentence-ending punctuation on previous word
            if not should_split and current and len(current) >= 2:
                prev_text = str(current[-1]["text"])
                if prev_text and prev_text[-1] in sentence_end_chars:
                    should_split = True
            # Split after clause punctuation if chunk is getting long
            if not should_split and current and len(current) >= max(2, max_words - 1):
                prev_text = str(current[-1]["text"])
                if prev_text and prev_text[-1] in clause_break_chars:
                    should_split = True

            if should_split:
                chunks.append(current)
                current = []
            current.append(item)
            prev_end = float(item["end_sec"])
        if current:
            chunks.append(current)
        if chunks:
            clip_chunks.append((clip, chunks))

    overlays_remaining = max_caption_overlays if max_caption_overlays > 0 else None
    clips_remaining = len(clip_chunks)
    for clip, raw_chunks in clip_chunks:
        chunks = raw_chunks
        if overlays_remaining is not None:
            if overlays_remaining <= 0:
                clips_remaining = max(clips_remaining - 1, 0)
                continue
            fair_share = max(1, overlays_remaining // max(clips_remaining, 1))
            clip_budget = min(overlays_remaining, fair_share)
            chunks = _coalesce_caption_chunks(chunks, clip_budget)
            overlays_remaining -= len(chunks)
            clips_remaining = max(clips_remaining - 1, 0)

        for idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            source_start = float(chunk[0]["start_sec"])
            source_end = max(float(chunk[-1]["end_sec"]), source_start + min_duration_sec)
            if idx + 1 < len(chunks):
                next_start = float(chunks[idx + 1][0]["start_sec"])
                max_end_before_next = max(source_start + min_duration_sec, next_start - caption_guard_sec)
                source_end = min(source_end, max_end_before_next)
            # Hard cap subtitle visibility only for clearly pathological ASR spans.
            natural_span = max(float(chunk[-1]["end_sec"]) - source_start, 0.0)
            if natural_span > (max_caption_display_sec * max_caption_display_hard_cap_factor):
                source_end = min(source_end, source_start + max_caption_display_sec)
            duration = max(round(source_end - source_start, 3), min_duration_sec)
            caption_text = " ".join(str(item["text"]) for item in chunk)
            if style == "creator":
                caption_text = caption_text.upper()

            overlay = TextOverlay(
                id=str(uuid4()),
                text=caption_text,
                start_sec=round(max(source_start - clip.start_sec, 0.0), 3),
                duration_sec=duration,
                x=overlay_x,
                y=overlay_y,
                font_size=overlay_font_size,
                font_name=overlay_font_name,
                color=overlay_color,
                highlight_color=overlay_highlight_color,
                outline_color=overlay_outline_color,
                outline_width=overlay_outline_width,
                shadow=overlay_shadow,
                alignment=overlay_alignment,
                margin_v=overlay_margin_v,
                style=render_style,
                word_timings=[{
                    "text": str(w["text"]),
                    "start_sec": float(w["start_sec"]),
                    "end_sec": float(w["end_sec"])
                } for w in chunk]
            )
            clip.text_overlays.append(overlay)


def apply_operation(state: TimelineState, operation: OperationPayload) -> TimelineState:
    params = operation.params
    op_type = operation.op_type

    handlers = {
        "add_clip": _apply_add_clip,
        "trim_clip": _apply_trim,
        "split_clip": _apply_split,
        "merge_clips": _apply_merge,
        "set_transition": _apply_transition,
        "add_text_overlay": _apply_add_text_overlay,
        "move_text_overlay": _apply_move_text_overlay,
        "trim_text_overlay": _apply_trim_text_overlay,
        "delete_text_overlay": _apply_delete_text_overlay,
        "add_audio_track": _apply_add_audio_track,
        "add_broll_clip": _apply_add_broll_clip,
        "move_broll_clip": _apply_move_broll_clip,
        "trim_broll_clip": _apply_trim_broll_clip,
        "delete_broll_clip": _apply_delete_broll_clip,
        "set_broll_opacity": _apply_set_broll_opacity,
        "set_volume": _apply_volume,
        "set_speed": _apply_speed,
        "set_aspect_ratio": _apply_aspect_ratio,
        "crop_resize": _apply_crop_resize,
        "rotate_clip": _apply_rotate,
        "flip_clip": _apply_flip,
        "set_adjustments": _apply_adjustments,
        "delete_clip": _apply_delete_clip,
        "ripple_edit": _apply_ripple_edit,
        "move_clip": _apply_move_clip,
        "reorder_clips": _apply_reorder_clips,
        "replace_video_track_clips": _apply_replace_video_track_clips,
        "clear_subtitles": _apply_clear_subtitles,
        "set_subtitles": _apply_set_subtitles,
        "set_export_settings": _apply_export_settings,
        "import_media": lambda *_: None,
    }
    if op_type not in handlers:
        raise ValueError(f"unsupported operation: {op_type}")

    handlers[op_type](state, params)
    _recalculate_duration(state)
    return state


def undo_timeline(session: Session, project_id: str) -> Timeline:
    timeline = get_timeline_row(session, project_id)
    if timeline.version == 0:
        return timeline
    prev = session.exec(
        select(TimelineVersion).where(
            TimelineVersion.project_id == project_id,
            TimelineVersion.version == timeline.version - 1,
        )
    ).first()
    if not prev:
        return timeline
    timeline.version = prev.version
    timeline.state_json = prev.state_json
    timeline.updated_at = _utcnow()
    session.add(timeline)
    session.commit()
    session.refresh(timeline)
    return timeline


def redo_timeline(session: Session, project_id: str) -> Timeline:
    timeline = get_timeline_row(session, project_id)
    nxt = session.exec(
        select(TimelineVersion).where(
            TimelineVersion.project_id == project_id,
            TimelineVersion.version == timeline.version + 1,
        )
    ).first()
    if not nxt:
        return timeline
    timeline.version = nxt.version
    timeline.state_json = nxt.state_json
    timeline.updated_at = _utcnow()
    session.add(timeline)
    session.commit()
    session.refresh(timeline)
    return timeline
