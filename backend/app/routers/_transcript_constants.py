from __future__ import annotations

import re


_OPAQUE_MEDIA_FILENAME_RE = re.compile(
    r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_RELATED_MEDIA_DURATION_TOLERANCE_SEC = 1.5
_RELATED_MEDIA_MAX_CANDIDATES = 120
_RELATED_MEDIA_HASH_SAMPLE_BYTES = 256 * 1024
_ARABIC_SCRIPT_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
)
