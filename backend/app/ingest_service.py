from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from .config import get_settings
from .storage import storage

settings = get_settings()

_SOURCE_TITLE_PREFIX = "__CLIPMIND_SOURCE_TITLE__"
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]+")


def _source_title_from_ytdlp_output(output: str) -> str | None:
    """Read the source title printed by yt-dlp after a successful move.

    Media is intentionally stored under a UUID to avoid filename collisions and
    filesystem-safety issues.  The original title is still needed as the
    human-facing media name and as the hint for song-lyrics matching.
    """
    for line in reversed(str(output or "").splitlines()):
        if not line.startswith(_SOURCE_TITLE_PREFIX):
            continue
        title = _CONTROL_CHARS_RE.sub(" ", line[len(_SOURCE_TITLE_PREFIX) :])
        title = re.sub(r"\s+", " ", title).strip()
        if title:
            return title[:240]
    return None


def source_title_to_filename(title: str | None, downloaded_path: str) -> str:
    """Create a display filename without allowing a title to become a path."""
    suffix = Path(downloaded_path).suffix.lower() or ".mp4"
    normalized = _CONTROL_CHARS_RE.sub(" ", str(title or ""))
    normalized = normalized.replace("/", " - ").replace("\\", " - ")
    normalized = re.sub(r"\s+", " ", normalized).strip(" .")
    if normalized.lower().endswith(suffix):
        normalized = normalized[: -len(suffix)].rstrip(" .")
    return f"{(normalized or Path(downloaded_path).stem)[:240]}{suffix}"


def validate_ingest_url(url: str) -> str:
    normalized = url.strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Invalid URL. Provide a full http(s) link.")
    return normalized


def download_video_with_ytdlp(url: str, project_id: str) -> tuple[str, str, str | None]:
    normalized_url = validate_ingest_url(url)
    if shutil.which(settings.yt_dlp_bin) is None:
        raise RuntimeError(f"{settings.yt_dlp_bin} not found in PATH")

    project_dir = storage.upload_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = uuid4().hex
    output_template = project_dir / f"{file_prefix}.%(ext)s"
    cmd = [
        settings.yt_dlp_bin,
        "--no-playlist",
        "--restrict-filenames",
        "--merge-output-format",
        "mp4",
        # Keep the physical filename opaque, but emit the platform title so
        # the asset can be labelled correctly and used for lyric lookup.
        "--print",
        f"after_move:{_SOURCE_TITLE_PREFIX}%(title)s",
    ]
    # YouTube extraction needs a JS runtime; yt-dlp only auto-detects deno,
    # so point it at node when that's what the host has.
    if shutil.which("deno") is None and shutil.which("node") is not None:
        cmd += ["--js-runtimes", "node"]
    cmd += [
        "-o",
        str(output_template),
        normalized_url,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(stderr or "URL ingestion failed with yt-dlp") from exc

    candidates = sorted(
        project_dir.glob(f"{file_prefix}.*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    file_path = next((path for path in candidates if path.suffix != ".part"), None)
    if file_path is None:
        raise RuntimeError("yt-dlp did not produce an output file")

    relative = str(file_path.resolve().relative_to(storage.upload_root))
    source_title = _source_title_from_ytdlp_output(result.stdout or "")
    return str(file_path.resolve()), relative, source_title
