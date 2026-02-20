from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import OperationPayload, PromptParseResponse

TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d+))?$")
SECONDS_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?$", re.IGNORECASE)
POLITE_PREFIX_RE = re.compile(r"^(?:please|pls|can you|could you|would you|kindly)\s+", re.IGNORECASE)
NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
}


def parse_timecode(value: str) -> float:
    value = value.strip()
    match = TIME_RE.match(value)
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        millis = match.group(4) or "0"
        return hours * 3600 + minutes * 60 + seconds + float(f"0.{millis}")
    sec = SECONDS_RE.match(value)
    if sec:
        return float(sec.group(1))
    raise ValueError(f"invalid timecode: {value}")


def _parse_clip_ref(token: str) -> int | str:
    token = token.strip()
    return int(token) if token.isdigit() else token


@dataclass(frozen=True)
class PromptPattern:
    regex: re.Pattern[str]
    handler_name: str
    confidence: float = 0.97


PATTERNS: list[PromptPattern] = [
    PromptPattern(
        regex=re.compile(
            r"^(?:trim|cut)\s+(?:clip\s+)?(\S+)\s+(?:from\s+)?(\S+)\s+(?:to|-)\s+(\S+)$",
            re.IGNORECASE,
        ),
        handler_name="trim",
    ),
    PromptPattern(
        regex=re.compile(r"^split\s+(?:clip\s+)?(\S+)\s+(?:at\s+)?(\S+)$", re.IGNORECASE),
        handler_name="split",
    ),
    PromptPattern(
        regex=re.compile(r"^(?:merge|join|combine)\s+clips?\s+(.+)$", re.IGNORECASE),
        handler_name="merge",
    ),
    PromptPattern(
        regex=re.compile(
            r'^add\s+(?:text|title|caption)\s+"(.+)"\s+at\s+(\S+)(?:\s+for\s+(\S+))?$',
            re.IGNORECASE,
        ),
        handler_name="add_text",
    ),
    PromptPattern(
        regex=re.compile(
            r"^add\s+(?:text|title|caption)\s+'(.+)'\s+at\s+(\S+)(?:\s+for\s+(\S+))?$",
            re.IGNORECASE,
        ),
        handler_name="add_text",
    ),
    PromptPattern(
        regex=re.compile(
            r"^add\s+(?:text|title|caption)\s+(.+?)\s+at\s+(\S+)(?:\s+for\s+(\S+))?$",
            re.IGNORECASE,
        ),
        handler_name="add_text",
        confidence=0.9,
    ),
    PromptPattern(
        regex=re.compile(r"^(?:set\s+)?aspect(?:\s+ratio)?\s+(\d+:\d+)$", re.IGNORECASE),
        handler_name="aspect",
    ),
    PromptPattern(
        regex=re.compile(
            r"^(?:add\s+)?transition(?:\s+to)?\s+clip\s+(\S+)\s+(fade|dissolve|slide_left|slide_right|slide_up|slide_down|zoom|wipe)\s+(\S+)$",
            re.IGNORECASE,
        ),
        handler_name="transition",
    ),
    PromptPattern(
        regex=re.compile(r"^(?:set\s+)?speed(?:\s+of)?\s+clip\s+(\S+)\s+(?:to\s+)?(\d+(?:\.\d+)?)(?:x)?$", re.IGNORECASE),
        handler_name="speed",
    ),
    PromptPattern(
        regex=re.compile(r"^fade\s+(in|out)\s+clip\s+(\S+)\s+(\S+)$", re.IGNORECASE),
        handler_name="fade",
    ),
    PromptPattern(
        regex=re.compile(
            r"^rotate\s+clip\s+(\S+)\s+(90|180|270|0)(?:deg|degree|degrees)?$",
            re.IGNORECASE,
        ),
        handler_name="rotate",
    ),
    PromptPattern(
        regex=re.compile(r"^flip\s+clip\s+(\S+)\s+(horizontal(?:ly)?|vertical(?:ly)?)$", re.IGNORECASE),
        handler_name="flip",
    ),
    PromptPattern(
        regex=re.compile(
            r"^crop\s+clip\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)$",
            re.IGNORECASE,
        ),
        handler_name="crop",
    ),
    PromptPattern(
        regex=re.compile(
            r"^set\s+brightness\s+clip\s+(\S+)\s+(-?\d+(?:\.\d+)?)$",
            re.IGNORECASE,
        ),
        handler_name="brightness",
        confidence=0.95,
    ),
    PromptPattern(
        regex=re.compile(
            r"^set\s+contrast\s+clip\s+(\S+)\s+(\d+(?:\.\d+)?)$",
            re.IGNORECASE,
        ),
        handler_name="contrast",
        confidence=0.95,
    ),
    PromptPattern(
        regex=re.compile(
            r"^set\s+saturation\s+clip\s+(\S+)\s+(\d+(?:\.\d+)?)$",
            re.IGNORECASE,
        ),
        handler_name="saturation",
        confidence=0.95,
    ),
    PromptPattern(
        regex=re.compile(
            r"^set\s+exposure\s+clip\s+(\S+)\s+(-?\d+(?:\.\d+)?)$",
            re.IGNORECASE,
        ),
        handler_name="exposure",
        confidence=0.95,
    ),
    PromptPattern(
        regex=re.compile(
            r"^set\s+temperature\s+clip\s+(\S+)\s+(-?\d+(?:\.\d+)?)$",
            re.IGNORECASE,
        ),
        handler_name="temperature",
        confidence=0.95,
    ),
    PromptPattern(
        regex=re.compile(
            r"^set\s+preset\s+clip\s+(\S+)\s+(warm|cool|cinematic|vintage|mono)$",
            re.IGNORECASE,
        ),
        handler_name="preset",
        confidence=0.95,
    ),
    PromptPattern(
        regex=re.compile(r"^set\s+volume\s+clip\s+(\S+)\s+(\d+(?:\.\d+)?)$", re.IGNORECASE),
        handler_name="volume",
        confidence=0.95,
    ),
    PromptPattern(
        regex=re.compile(r"^(mute|unmute)\s+clip\s+(\S+)$", re.IGNORECASE),
        handler_name="mute",
        confidence=0.95,
    ),
    PromptPattern(
        regex=re.compile(r"^audio\s+fade\s+(in|out)\s+clip\s+(\S+)\s+(\S+)$", re.IGNORECASE),
        handler_name="audio_fade",
        confidence=0.95,
    ),
    PromptPattern(
        regex=re.compile(r"^(?:track|set\s+track)\s+(video|audio)\s+volume\s+(?:to\s+)?(\d+(?:\.\d+)?)$", re.IGNORECASE),
        handler_name="track_volume",
    ),
    PromptPattern(
        regex=re.compile(r"^(mute|unmute|solo|unsolo)\s+track\s+(video|audio)$", re.IGNORECASE),
        handler_name="track_state",
    ),
    PromptPattern(
        regex=re.compile(r"^(mute|unmute|solo|unsolo)\s+(video|audio)\s+track$", re.IGNORECASE),
        handler_name="track_state",
    ),
    PromptPattern(
        regex=re.compile(r"^(?:delete|remove)\s+clip\s+(\S+)$", re.IGNORECASE),
        handler_name="delete",
    ),
    PromptPattern(
        regex=re.compile(r"^(?:move|shift)\s+clip\s+(\S+)\s+(?:to|at)\s+(\S+)$", re.IGNORECASE),
        handler_name="move",
    ),
    PromptPattern(
        regex=re.compile(r"^(?:ripple|ripple\s+edit)\s+(video|audio)$", re.IGNORECASE),
        handler_name="ripple",
    ),
    PromptPattern(
        regex=re.compile(
            r"^(?:export|render)\s+(720p|1080p|4k)\s+(24|30|60)\s*fps\s+(low|medium|high|max)(?:\s+quality)?\s+(mp4|mov|webm)$",
            re.IGNORECASE,
        ),
        handler_name="export",
        confidence=0.98,
    ),
    PromptPattern(
        regex=re.compile(
            r"^add\s+(?:clip|video)\s+asset\s+(\S+)\s+from\s+(\S+)\s+to\s+(\S+)(?:\s+at\s+(\S+))?$",
            re.IGNORECASE,
        ),
        handler_name="add_clip",
    ),
    PromptPattern(
        regex=re.compile(
            r"^add\s+audio\s+asset\s+(\S+)\s+from\s+(\S+)\s+to\s+(\S+)(?:\s+at\s+(\S+))?$",
            re.IGNORECASE,
        ),
        handler_name="add_audio",
    ),
]


def _split_csv_refs(value: str) -> list[int | str]:
    refs: list[int | str] = []
    for token in re.split(r"[,\s]+", value):
        item = token.strip()
        if not item:
            continue
        refs.append(_parse_clip_ref(item))
    return refs


def _normalize_prompt(text: str) -> str:
    text = " ".join(text.strip().split())
    text = re.sub(r"[.!?]+$", "", text)
    text = POLITE_PREFIX_RE.sub("", text)
    text = re.sub(r"\s+(?:please|pls)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bclip\s*#\s*(\d+)\b", r"clip \1", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*(?:sec|secs|second|seconds)\b",
        r"\1s",
        text,
        flags=re.IGNORECASE,
    )
    for word, number in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\s+clip\b", f"clip {number}", text, flags=re.IGNORECASE)
        text = re.sub(rf"\bclip\s+{word}\b", f"clip {number}", text, flags=re.IGNORECASE)
    return text


def _build_operation(pattern: PromptPattern, match: re.Match[str]) -> OperationPayload:
    handler = pattern.handler_name
    if handler == "trim":
        return OperationPayload(
            op_type="trim_clip",
            source="prompt",
            params={
                "clip": _parse_clip_ref(match.group(1)),
                "start_sec": parse_timecode(match.group(2)),
                "end_sec": parse_timecode(match.group(3)),
            },
        )
    if handler == "split":
        return OperationPayload(
            op_type="split_clip",
            source="prompt",
            params={
                "clip": _parse_clip_ref(match.group(1)),
                "at_sec": parse_timecode(match.group(2)),
            },
        )
    if handler == "merge":
        return OperationPayload(
            op_type="merge_clips",
            source="prompt",
            params={"clips": _split_csv_refs(match.group(1))},
        )
    if handler == "add_text":
        start_sec = parse_timecode(match.group(2))
        duration_sec = parse_timecode(match.group(3)) if match.group(3) else 2.0
        return OperationPayload(
            op_type="add_text_overlay",
            source="prompt",
            params={
                "text": match.group(1),
                "start_sec": start_sec,
                "duration_sec": duration_sec,
            },
        )
    if handler == "aspect":
        return OperationPayload(
            op_type="set_aspect_ratio",
            source="prompt",
            params={"ratio": match.group(1)},
        )
    if handler == "transition":
        return OperationPayload(
            op_type="set_transition",
            source="prompt",
            params={
                "clip": _parse_clip_ref(match.group(1)),
                "type": match.group(2).lower(),
                "duration_sec": parse_timecode(match.group(3)),
            },
        )
    if handler == "speed":
        return OperationPayload(
            op_type="set_speed",
            source="prompt",
            params={"clip": _parse_clip_ref(match.group(1)), "speed": float(match.group(2))},
        )
    if handler == "fade":
        fade_type = match.group(1).lower()
        return OperationPayload(
            op_type="set_transition",
            source="prompt",
            params={
                "clip": _parse_clip_ref(match.group(2)),
                "type": "fade",
                "duration_sec": parse_timecode(match.group(3)),
                "mode": fade_type,
            },
        )
    if handler == "rotate":
        return OperationPayload(
            op_type="rotate_clip",
            source="prompt",
            params={
                "clip": _parse_clip_ref(match.group(1)),
                "degrees": int(match.group(2)),
            },
        )
    if handler == "flip":
        direction = match.group(2).lower()
        if direction.startswith("horizontal"):
            direction = "horizontal"
        elif direction.startswith("vertical"):
            direction = "vertical"
        return OperationPayload(
            op_type="flip_clip",
            source="prompt",
            params={
                "clip": _parse_clip_ref(match.group(1)),
                "direction": direction,
            },
        )
    if handler == "crop":
        return OperationPayload(
            op_type="crop_resize",
            source="prompt",
            params={
                "clip": _parse_clip_ref(match.group(1)),
                "x": int(match.group(2)),
                "y": int(match.group(3)),
                "width": int(match.group(4)),
                "height": int(match.group(5)),
            },
        )
    if handler == "brightness":
        return OperationPayload(
            op_type="set_adjustments",
            source="prompt",
            params={
                "clip": _parse_clip_ref(match.group(1)),
                "brightness": float(match.group(2)),
            },
        )
    if handler == "contrast":
        return OperationPayload(
            op_type="set_adjustments",
            source="prompt",
            params={
                "clip": _parse_clip_ref(match.group(1)),
                "contrast": float(match.group(2)),
            },
        )
    if handler == "saturation":
        return OperationPayload(
            op_type="set_adjustments",
            source="prompt",
            params={
                "clip": _parse_clip_ref(match.group(1)),
                "saturation": float(match.group(2)),
            },
        )
    if handler == "exposure":
        return OperationPayload(
            op_type="set_adjustments",
            source="prompt",
            params={
                "clip": _parse_clip_ref(match.group(1)),
                "exposure": float(match.group(2)),
            },
        )
    if handler == "temperature":
        return OperationPayload(
            op_type="set_adjustments",
            source="prompt",
            params={
                "clip": _parse_clip_ref(match.group(1)),
                "temperature": float(match.group(2)),
            },
        )
    if handler == "preset":
        return OperationPayload(
            op_type="set_adjustments",
            source="prompt",
            params={
                "clip": _parse_clip_ref(match.group(1)),
                "preset": match.group(2).lower(),
            },
        )
    if handler == "volume":
        return OperationPayload(
            op_type="set_volume",
            source="prompt",
            params={"clip": _parse_clip_ref(match.group(1)), "volume": float(match.group(2))},
        )
    if handler == "mute":
        return OperationPayload(
            op_type="set_volume",
            source="prompt",
            params={
                "clip": _parse_clip_ref(match.group(2)),
                "mute": match.group(1).lower() == "mute",
            },
        )
    if handler == "audio_fade":
        mode = match.group(1).lower()
        sec = parse_timecode(match.group(3))
        payload: dict[str, int | str | float] = {
            "clip": _parse_clip_ref(match.group(2)),
            "track_kind": "audio",
        }
        if mode == "in":
            payload["fade_in_sec"] = sec
        else:
            payload["fade_out_sec"] = sec
        return OperationPayload(
            op_type="set_volume",
            source="prompt",
            params=payload,
        )
    if handler == "track_volume":
        return OperationPayload(
            op_type="set_volume",
            source="prompt",
            params={
                "track_kind": match.group(1).lower(),
                "volume": float(match.group(2)),
            },
        )
    if handler == "track_state":
        action = match.group(1).lower()
        track_kind = match.group(2).lower()
        payload: dict[str, object] = {"track_kind": track_kind}
        if action in {"mute", "unmute"}:
            payload["mute"] = action == "mute"
        if action in {"solo", "unsolo"}:
            payload["solo"] = action == "solo"
        return OperationPayload(
            op_type="set_volume",
            source="prompt",
            params=payload,
        )
    if handler == "delete":
        return OperationPayload(
            op_type="delete_clip",
            source="prompt",
            params={"clip": _parse_clip_ref(match.group(1))},
        )
    if handler == "move":
        return OperationPayload(
            op_type="move_clip",
            source="prompt",
            params={
                "clip": _parse_clip_ref(match.group(1)),
                "timeline_start_sec": parse_timecode(match.group(2)),
                "ripple": False,
            },
        )
    if handler == "ripple":
        return OperationPayload(
            op_type="ripple_edit",
            source="prompt",
            params={"track_kind": match.group(1).lower()},
        )
    if handler == "export":
        return OperationPayload(
            op_type="set_export_settings",
            source="prompt",
            params={
                "resolution": match.group(1).lower(),
                "fps": int(match.group(2)),
                "quality": match.group(3).lower(),
                "format": match.group(4).lower(),
            },
        )
    if handler == "add_clip":
        timeline_start = parse_timecode(match.group(4)) if match.group(4) else 0.0
        return OperationPayload(
            op_type="add_clip",
            source="prompt",
            params={
                "asset_id": match.group(1),
                "start_sec": parse_timecode(match.group(2)),
                "end_sec": parse_timecode(match.group(3)),
                "timeline_start_sec": timeline_start,
            },
        )
    if handler == "add_audio":
        timeline_start = parse_timecode(match.group(4)) if match.group(4) else 0.0
        return OperationPayload(
            op_type="add_audio_track",
            source="prompt",
            params={
                "asset_id": match.group(1),
                "start_sec": parse_timecode(match.group(2)),
                "end_sec": parse_timecode(match.group(3)),
                "timeline_start_sec": timeline_start,
            },
        )
    raise ValueError("unsupported prompt handler")


def parse_prompt(prompt: str) -> PromptParseResponse:
    text = _normalize_prompt(prompt)
    if not text:
        return PromptParseResponse(
            prompt=prompt,
            confidence=0.0,
            operations=[],
            errors=["Prompt is empty."],
            suggestions=[
                'Try: trim clip 1 from 00:05 to 00:12',
                'Try: add text "Subscribe" at 00:02 for 3s',
                "Try: transition clip 1 dissolve 0.6s",
            ],
        )

    for pattern in PATTERNS:
        match = pattern.regex.match(text)
        if not match:
            continue
        try:
            op = _build_operation(pattern, match)
        except ValueError as exc:
            return PromptParseResponse(
                prompt=prompt,
                confidence=0.0,
                operations=[],
                errors=[str(exc)],
            )
        return PromptParseResponse(
            prompt=prompt,
            confidence=pattern.confidence,
            operations=[op],
        )

    return PromptParseResponse(
        prompt=prompt,
        confidence=0.0,
        operations=[],
        errors=["Could not parse prompt with the supported command set."],
        suggestions=[
            "trim clip 1 from 00:05 to 00:12",
            "split clip 2 at 00:07",
            "merge clips 1,2,3",
            'add text "New drop" at 00:02 for 3s',
            "set aspect 9:16",
            "crop clip 1 0 0 1080 1920",
            "set saturation clip 1 1.2",
            "track audio volume 0.7",
            "audio fade out clip 1 1.2s",
            "move clip 2 to 00:05",
            "delete clip 2",
            "export 1080p 30fps high mp4",
        ],
    )
