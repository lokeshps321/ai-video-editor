from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from .config import get_settings
from .storage import storage

settings = get_settings()


def validate_ingest_url(url: str) -> str:
    normalized = url.strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Invalid URL. Provide a full http(s) link.")
    return normalized


def download_video_with_ytdlp(url: str, project_id: str) -> tuple[str, str]:
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
        "-o",
        str(output_template),
        normalized_url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
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
    return str(file_path.resolve()), relative
