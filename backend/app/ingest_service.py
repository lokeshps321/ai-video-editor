from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import requests

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


def download_video_with_apify(url: str, project_id: str) -> tuple[str, str, str | None]:
    """Download video from YouTube using Apify (100 free runs/month)."""
    import os

    normalized_url = validate_ingest_url(url)
    apify_token = os.getenv("APIFY_API_TOKEN")

    if not apify_token:
        raise RuntimeError(
            "APIFY_API_TOKEN not configured. "
            "Sign up at apify.com, get your free tier API token, and set APIFY_API_TOKEN in .env"
        )

    project_dir = storage.upload_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = uuid4().hex
    output_file = project_dir / f"{file_prefix}.mp4"

    # Call Apify YouTube Video Downloader actor
    # Actor: streamers/youtube-video-downloader (free tier: 100 runs/month)
    actor_id = "streamers/youtube-video-downloader"
    api_url = f"https://api.apify.com/v2/acts/{actor_id}/runs"

    headers = {
        "Authorization": f"Bearer {apify_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "youtubeUrl": normalized_url,
        "maxHeight": 1080,
    }

    try:
        # Start the Apify actor run
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        run_data = response.json()

        if not run_data.get("data"):
            raise RuntimeError(f"Apify API error: {run_data}")

        run_id = run_data["data"]["id"]

        # Poll for completion (with timeout)
        max_wait = 300  # 5 minutes max
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status_url = f"https://api.apify.com/v2/acts/{actor_id}/runs/{run_id}"
            status_resp = requests.get(status_url, headers=headers, timeout=30)
            status_resp.raise_for_status()
            run_status = status_resp.json()

            status = run_status.get("data", {}).get("status")
            if status == "SUCCEEDED":
                break
            elif status in ("FAILED", "ABORTED", "TIMED_OUT"):
                raise RuntimeError(f"Apify run failed with status: {status}")

            time.sleep(2)
        else:
            raise RuntimeError("Apify download timeout (>5 minutes)")

        # Get the output dataset
        dataset_url = f"https://api.apify.com/v2/acts/{actor_id}/runs/{run_id}/dataset/items"
        dataset_resp = requests.get(dataset_url, headers=headers, timeout=30)
        dataset_resp.raise_for_status()
        dataset_items = dataset_resp.json()

        if not dataset_items.get("data"):
            raise RuntimeError("No download URL returned from Apify")

        video_item = dataset_items["data"][0]
        download_url = video_item.get("url")
        video_title = video_item.get("title", "Downloaded Video")

        if not download_url:
            raise RuntimeError("Apify did not return a download URL")

        # Download the video file from the URL
        video_resp = requests.get(download_url, timeout=600, stream=True)
        video_resp.raise_for_status()

        with open(output_file, "wb") as f:
            for chunk in video_resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if not output_file.exists() or output_file.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty or missing")

        relative = str(output_file.resolve().relative_to(storage.upload_root))
        return str(output_file.resolve()), relative, video_title

    except requests.RequestException as exc:
        raise RuntimeError(f"Apify download failed: {exc}") from exc


def download_video_with_ytdlp(url: str, project_id: str) -> tuple[str, str, str | None]:
    normalized_url = validate_ingest_url(url)

    # Security: Disable URL-based ingestion due to SSRF/parser-differential complexity
    # Users should download locally and upload files instead (more secure and reliable)
    raise RuntimeError(
        "URL-based video ingestion is disabled for security. "
        "Please download the video locally and upload the MP4/MOV file instead."
    )

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
    # YouTube extraction needs a JS runtime; yt-dlp will auto-detect deno or node
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


