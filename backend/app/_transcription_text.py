from __future__ import annotations

import os
import re
import threading
from math import isfinite


_TRANSCRIPTION_RUNTIME = threading.local()


def _runtime_profile() -> str | None:
    value = getattr(_TRANSCRIPTION_RUNTIME, "profile", None)
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _runtime_mode() -> str | None:
    value = getattr(_TRANSCRIPTION_RUNTIME, "mode", None)
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _normalize_transcription_mode(mode: str | None) -> str:
    normalized = (mode or "auto").strip().lower()
    if normalized in {"speech", "song"}:
        return normalized
    return "auto"


def _clean_word(value: str) -> str:
    return " ".join(value.strip().split())


def _normalize_token(value: str) -> str:
    return re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", value.lower()).strip()


def _ascii_latin_ratio(value: str) -> float:
    alpha = [char for char in value if char.isalpha()]
    if not alpha:
        return 1.0
    latin = [char for char in alpha if ("A" <= char <= "Z") or ("a" <= char <= "z")]
    return len(latin) / len(alpha)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)
    return max(minimum, value)


def _is_placeholder_config_value(value: str | None) -> bool:
    if value is None:
        return True
    raw = value.strip()
    if not raw:
        return True
    lowered = raw.lower()
    if lowered in {
        "your_endpoint",
        "your_key",
        "changeme",
        "change_me",
        "replace_me",
        "none",
        "null",
    }:
        return True
    if lowered.startswith("your_"):
        return True
    if lowered.startswith("<") and lowered.endswith(">") and len(lowered) > 2:
        return True
    return False


def _parse_timestamp_value(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed):
        return None
    return parsed


def _clamp_time(value: float, duration_sec: float) -> float:
    return max(0.0, min(value, duration_sec))


def _normalize_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed):
        return None
    return max(0.0, min(parsed, 1.0))
