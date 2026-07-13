from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_REEL_ASPECT_RATIO = 9.0 / 16.0


@dataclass(frozen=True)
class SmartReframePlan:
    """A non-destructive 9:16 crop plan for one source-video clip."""

    crop: dict[str, int] | None
    crop_keyframes: list[dict[str, float | int]]
    uses_subject_tracking: bool


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _crop_x(width: int, crop_width: int, focus_x: float) -> int:
    center_x = int(round(_clamp(focus_x, 0.0, 1.0) * width))
    return max(0, min(width - crop_width, center_x - (crop_width // 2)))


def plan_reel_smart_reframe(
    *,
    width: int,
    height: int,
    clip_duration_sec: float,
    focus_x: float | None = None,
    focus_track: object = None,
    clip_start_sec: float = 0.0,
) -> SmartReframePlan:
    """Build a full-screen 9:16 crop plan for a wide source video.

    The plan never stretches pixels and never fabricates a subject location.  It
    follows a supplied focus track when available; otherwise it produces a
    deterministic centre crop.  Portrait and square inputs already fill (or
    nearly fill) the Reel canvas through the renderer's cover filter, so they
    are intentionally left alone here.

    ``focus_track`` timestamps are relative to the start of the source media
    file, while the FFmpeg crop expression runs in clip-relative time (the
    renderer trims with ``-ss``/``-to``, which rebases ``t`` to 0 at the trim
    point).  ``clip_start_sec`` converts between the two so a trimmed clip
    follows the subject's position during the trimmed window rather than
    whatever the subject was doing at the start of the source file.
    """

    if width <= 0 or height <= 0 or clip_duration_sec <= 0 or width <= height:
        return SmartReframePlan(None, [], False)

    crop_width = int(round(height * _REEL_ASPECT_RATIO))
    crop_width = max(2, min(crop_width, width))
    if crop_width >= width:
        return SmartReframePlan(None, [], False)

    fallback_focus = 0.5 if focus_x is None else _clamp(float(focus_x), 0.0, 1.0)
    crop = {
        "x": _crop_x(width, crop_width, fallback_focus),
        "y": 0,
        "width": crop_width,
        "height": height,
    }

    if not isinstance(focus_track, list):
        return SmartReframePlan(crop, [], False)

    keyframes: list[dict[str, float | int]] = []
    previous_x: int | None = None
    for raw in focus_track:
        if not isinstance(raw, dict):
            continue
        try:
            time_sec = float(raw.get("time_sec", 0.0)) - clip_start_sec
            x_ratio = float(raw.get("x_ratio", fallback_focus))
        except (TypeError, ValueError):
            continue
        if time_sec < 0:
            continue
        if time_sec > clip_duration_sec:
            break

        target_x = _crop_x(width, crop_width, x_ratio)
        # Smooth positions once more at application time.  The detector already
        # smooths samples, but this avoids visible jumps on a noisy video.
        x = (
            target_x
            if previous_x is None
            else int(round((0.65 * previous_x) + (0.35 * target_x)))
        )
        previous_x = x
        keyframes.append({"time_sec": round(time_sec, 3), "x": x, "y": 0})

    if not keyframes:
        return SmartReframePlan(crop, [], False)

    if float(keyframes[0]["time_sec"]) > 0:
        keyframes.insert(0, {"time_sec": 0.0, "x": int(keyframes[0]["x"]), "y": 0})
    if float(keyframes[-1]["time_sec"]) < clip_duration_sec:
        keyframes.append(
            {
                "time_sec": round(clip_duration_sec, 3),
                "x": int(keyframes[-1]["x"]),
                "y": 0,
            }
        )

    deduped: list[dict[str, float | int]] = []
    for item in keyframes:
        if deduped and abs(float(item["time_sec"]) - float(deduped[-1]["time_sec"])) < 0.001:
            deduped[-1] = item
            continue
        if (
            deduped
            and abs(int(item["x"]) - int(deduped[-1]["x"])) < 1
            and float(item["time_sec"]) - float(deduped[-1]["time_sec"]) < 0.12
        ):
            continue
        deduped.append(item)

    # FFmpeg expressions stay predictable and cheap even for long inputs.
    if len(deduped) > 24:
        stride = max(1, len(deduped) // 24)
        reduced = [deduped[0]]
        reduced.extend(deduped[idx] for idx in range(stride, len(deduped), stride))
        if reduced[-1]["time_sec"] != deduped[-1]["time_sec"]:
            reduced.append(deduped[-1])
        deduped = reduced[:24]

    return SmartReframePlan(crop, deduped, len(deduped) >= 2)
