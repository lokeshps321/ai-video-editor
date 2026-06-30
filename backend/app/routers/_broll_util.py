from __future__ import annotations

import json
from datetime import datetime, timezone

from ..models import BrollCandidate, BrollSlot, MediaAsset, Project
from ._broll_constants import _WORD_RE


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _filename_tokens(filename: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(filename)}


def _is_vertical_project(project: Project) -> bool:
    return int(project.height) >= int(project.width)


def _parse_asset_metadata(asset: MediaAsset | None) -> dict[str, object]:
    if asset is None:
        return {}
    try:
        payload = json.loads(asset.metadata_json or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_anchor_word_ids(row: BrollSlot) -> list[str]:
    try:
        parsed = json.loads(row.anchor_word_ids_json or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def _parse_reason_json(row: BrollCandidate) -> dict[str, object]:
    try:
        parsed = json.loads(row.reason_json or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
