from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import requests

from .config import get_settings
from .storage import storage

settings = get_settings()
logger = logging.getLogger("clipmind.ingest")

ProgressCallback = Callable[[int, str], None]


def _emit_progress(
    progress_callback: ProgressCallback | None,
    progress: int,
    message: str,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(max(0, min(100, int(progress))), message)
    except Exception:
        logger.debug("ingest progress callback failed", exc_info=True)

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
        title = _normalize_source_title(line[len(_SOURCE_TITLE_PREFIX) :])
        if title:
            return title
    return None


def _normalize_source_title(raw: str) -> str | None:
    title = _CONTROL_CHARS_RE.sub(" ", str(raw or ""))
    title = re.sub(r"\s+", " ", title).strip()
    return title[:240] or None


def _read_source_title_file(path: Path) -> str | None:
    """Title written by yt-dlp's --print-to-file, if the download got that far."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(raw.splitlines()):
        title = _normalize_source_title(line)
        if title:
            return title
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


def _apify_api_token() -> str:
    return (os.getenv("APIFY_API_TOKEN", "") or "").strip()


def _ingest_provider() -> str:
    configured = (os.getenv("INGEST_PROVIDER", "auto") or "auto").strip().lower()
    if configured in {"apify", "ytdlp", "auto"}:
        return configured
    return "auto"


def _apify_quality() -> str:
    quality = (os.getenv("APIFY_YOUTUBE_QUALITY", "720") or "720").strip()
    if quality in {"360", "480", "720", "1080", "1440", "2160"}:
        return quality
    return "720"


def _extract_apify_download_url(item: dict) -> str | None:
    output = item.get("output")
    if isinstance(output, dict):
        url = output.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    for key in (
        "downloadedFileUrl",
        "downloadUrl",
        "download_url",
        "url",
        "fileUrl",
    ):
        value = item.get(key)
        if isinstance(value, str) and value.strip().startswith("http"):
            return value.strip()
    return None


def _extract_apify_title(item: dict, fallback_url: str) -> str:
    for key in ("title", "name", "videoTitle", "filename"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            cleaned = value.strip()
            if cleaned.lower().endswith(".mp4"):
                cleaned = cleaned[: -len(".mp4")]
            return cleaned[:240]
    video_id = item.get("videoId")
    if isinstance(video_id, str) and video_id.strip():
        return video_id.strip()[:240]
    return Path(urlparse(fallback_url).path).name or "Downloaded Video"


def _apify_actor_id() -> str:
    # streamers actor returns a real MP4 on free Apify accounts via KV store.
    # epctex often returns only {"demo": true} on free plans.
    configured = (
        os.getenv("APIFY_YOUTUBE_ACTOR", "streamers~youtube-video-downloader") or ""
    ).strip()
    configured = configured.replace("/", "~")
    return configured or "streamers~youtube-video-downloader"


def _apify_actor_payload(normalized_url: str, actor_id: str) -> dict:
    quality = _apify_quality()
    quality_label = quality if quality.endswith("p") else f"{quality}p"
    if "streamers" in actor_id:
        return {
            "videos": [{"url": normalized_url}],
            "storeInKVStore": True,
            "preferredQuality": quality_label,
            "preferredFormat": "mp4",
            "filenameTemplateParts": ["title"],
        }
    if "eunit" in actor_id:
        return {
            "startUrls": [{"url": normalized_url}],
            "downloadMode": "save-best-progressive",
            "preferredContainer": "mp4",
            "maxHeight": int(quality) if quality.isdigit() else 720,
        }
    # epctex / generic PPE downloaders
    return {
        "startUrls": [normalized_url],
        "videoIds": [],
        "quality": quality if quality.isdigit() else "720",
        "storageType": "apify",
    }


def download_video_with_apify(
    url: str,
    project_id: str,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, str, str | None]:
    """Download video from YouTube using Apify (better for cloud than yt-dlp)."""
    normalized_url = validate_ingest_url(url)
    apify_token = _apify_api_token()

    if not apify_token:
        raise RuntimeError(
            "APIFY_API_TOKEN not configured. "
            "Sign up at apify.com, get your API token, and set APIFY_API_TOKEN in .env"
        )

    project_dir = storage.upload_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = uuid4().hex
    output_file = project_dir / f"{file_prefix}.mp4"

    actor_id = _apify_actor_id()
    api_url = f"https://api.apify.com/v2/acts/{actor_id}/runs"
    headers = {
        "Authorization": f"Bearer {apify_token}",
        "Content-Type": "application/json",
    }
    payload = _apify_actor_payload(normalized_url, actor_id)

    try:
        _emit_progress(progress_callback, 22, "Starting cloud video fetch...")
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=180,
        )
        if response.status_code >= 400:
            detail = response.text[:300]
            raise RuntimeError(
                f"Apify start failed HTTP {response.status_code}: {detail}"
            )
        run_data = response.json()
        data = run_data.get("data") if isinstance(run_data, dict) else None
        if not isinstance(data, dict):
            raise RuntimeError(f"Apify API error: {run_data}")

        run_id = data.get("id")
        if not run_id:
            raise RuntimeError(f"No run ID in response: {run_data}")

        max_wait = int(os.getenv("APIFY_YOUTUBE_TIMEOUT_SEC", "300") or "300")
        start_time = time.time()
        run_status_data = data
        _emit_progress(progress_callback, 28, "Cloud downloader started...")

        while time.time() - start_time < max_wait:
            status = str(run_status_data.get("status") or "")
            if status == "SUCCEEDED":
                break
            if status in {"FAILED", "ABORTED", "TIMED-OUT", "TIMED_OUT"}:
                message = run_status_data.get("statusMessage") or status
                raise RuntimeError(f"Apify run failed: {message}")

            elapsed = time.time() - start_time
            # Keep UI moving while Apify works (often 20-90s).
            paced = 28 + int(min(34, (elapsed / max(max_wait, 1)) * 34))
            status_label = status.lower() if status else "running"
            _emit_progress(
                progress_callback,
                paced,
                f"Fetching video in cloud ({status_label})...",
            )

            status_url = f"https://api.apify.com/v2/actor-runs/{run_id}"
            status_resp = requests.get(status_url, headers=headers, timeout=30)
            status_resp.raise_for_status()
            status_body = status_resp.json()
            run_status_data = (
                status_body.get("data") if isinstance(status_body, dict) else None
            )
            if not isinstance(run_status_data, dict):
                raise RuntimeError(f"Apify status error: {status_body}")
            time.sleep(2)
        else:
            raise RuntimeError("Apify download timeout")

        _emit_progress(progress_callback, 64, "Cloud fetch finished, reading result...")
        dataset_id = run_status_data.get("defaultDatasetId") or data.get(
            "defaultDatasetId"
        )
        if not dataset_id:
            raise RuntimeError(f"No dataset ID in run response: {run_status_data}")

        dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
        dataset_resp = requests.get(dataset_url, headers=headers, timeout=60)
        dataset_resp.raise_for_status()
        dataset_items = dataset_resp.json()

        if not isinstance(dataset_items, list) or not dataset_items:
            raise RuntimeError(f"No results in dataset: {dataset_items}")

        video_item = dataset_items[0]
        if not isinstance(video_item, dict):
            raise RuntimeError(f"Unexpected dataset item: {video_item}")
        if video_item.get("demo") is True:
            raise RuntimeError(
                "Apify free-plan demo response (no real video). "
                "Use streamers~youtube-video-downloader or a paid Apify plan."
            )
        item_status = str(video_item.get("status") or "").lower()
        if item_status in {"error", "failed"}:
            error = video_item.get("error") or video_item.get("status")
            raise RuntimeError(f"Apify download item failed: {error}")

        download_url = _extract_apify_download_url(video_item)
        video_title = _extract_apify_title(video_item, normalized_url)
        if not download_url:
            raise RuntimeError(f"No download URL in result: {video_item}")

        _emit_progress(progress_callback, 70, "Saving video to your project...")
        video_resp = requests.get(
            download_url,
            headers={"Authorization": f"Bearer {apify_token}"},
            timeout=600,
            stream=True,
        )
        video_resp.raise_for_status()

        total = int(video_resp.headers.get("Content-Length") or 0)
        written = 0
        last_emit = 70
        with open(output_file, "wb") as f:
            for chunk in video_resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if total > 0:
                    pct = 70 + int((written / total) * 12)
                    if pct >= last_emit + 2:
                        last_emit = pct
                        _emit_progress(
                            progress_callback,
                            pct,
                            "Saving video to your project...",
                        )

        if not output_file.exists() or output_file.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty or missing")

        relative = str(output_file.resolve().relative_to(storage.upload_root))
        _emit_progress(progress_callback, 84, "Video saved, finishing up...")
        logger.info(
            "Apify ingest succeeded actor=%s project=%s title=%s",
            actor_id,
            project_id,
            video_title,
        )
        return str(output_file.resolve()), relative, video_title

    except requests.RequestException as exc:
        raise RuntimeError(f"Apify download failed: {exc}") from exc


_YTDLP_PROGRESS_RE = re.compile(r"\[download\]\s+(\d+(?:\.\d+)?)%")


def _ytdlp_auth_args() -> list[str]:
    """Optional cookie auth for yt-dlp.

    Public videos need no auth from a residential IP. Datacenter IPs (prod) hit
    YouTube's bot check, which cookies are the supported way around -- yt-dlp
    removed OAuth login entirely.
    """
    cookies_file = settings.yt_dlp_cookies_file
    if cookies_file:
        if Path(cookies_file).is_file():
            logger.debug("yt-dlp: using cookies file")
            return ["--cookies", cookies_file]
        logger.warning(
            "YTDLP_COOKIES_FILE is set but does not exist; continuing without cookies"
        )

    from_browser = settings.yt_dlp_cookies_from_browser
    if from_browser:
        logger.debug("yt-dlp: using cookies from browser")
        return ["--cookies-from-browser", from_browser]

    return []


def _ytdlp_format_selector() -> str:
    """Cap the download at H.264/<=max_height so the merge into mp4 is a remux.

    Without this yt-dlp happily picks 2160p AV1, which is slow to fetch and
    pointless for the editor timeline.
    """
    height = settings.yt_dlp_max_height
    return (
        f"bestvideo[height<={height}][vcodec^=avc1]+bestaudio[acodec^=mp4a]/"
        f"bestvideo[height<={height}]+bestaudio/"
        f"best[height<={height}]/best"
    )


_YTDLP_COOKIE_HINTS = (
    "sign in to confirm",
    "confirm you're not a bot",
    "use --cookies",
    "login required",
    "this video is private",
    "age",
)


def _ytdlp_failure_message(detail: str) -> str:
    """Front the raw yt-dlp dump with something actionable for the UI banner."""
    if not detail:
        return "URL ingestion failed with yt-dlp"
    lowered = detail.lower()
    if any(hint in lowered for hint in _YTDLP_COOKIE_HINTS):
        return (
            "YouTube is asking this server to sign in. Set YTDLP_COOKIES_FILE "
            "(exported cookies.txt) or YTDLP_COOKIES_FROM_BROWSER and retry.\n\n"
            f"{detail}"
        )
    return detail


def download_video_with_ytdlp(
    url: str,
    project_id: str,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, str, str | None]:
    """Download video with yt-dlp, optionally authenticated with cookies."""
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
        "-f",
        _ytdlp_format_selector(),
        "--merge-output-format",
        "mp4",
        "--newline",
    ]
    # YouTube now gates its media URLs behind a JS signature/"n" challenge.
    # Solving it needs both a JS runtime (only deno is enabled by default, so
    # point yt-dlp at node) and the EJS solver script. Without the pair, the
    # extractor falls back to the android-vr client and every media URL 403s.
    if shutil.which("node") is not None:
        cmd += ["--js-runtimes", "node"]
    if settings.yt_dlp_remote_components:
        cmd += ["--remote-components", settings.yt_dlp_remote_components]
    cmd += _ytdlp_auth_args()
    # Keep the physical filename opaque, but emit the platform title so the
    # asset can be labelled correctly and used for lyric lookup.  This goes to
    # a side file rather than stdout because `--print` implies `--quiet`, which
    # suppresses the `[download] NN%` lines the progress bar is parsed from.
    # The file lives outside project_dir so it can't be mistaken for the media.
    title_fd, title_file_name = tempfile.mkstemp(prefix="ytdlp-title-", suffix=".txt")
    os.close(title_fd)
    title_file = Path(title_file_name)
    cmd += [
        "--print-to-file",
        "after_move:%(title)s",
        str(title_file),
        "-o",
        str(output_template),
        normalized_url,
    ]
    _emit_progress(progress_callback, 5, "Downloading video...")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    output_chunks: list[str] = []
    last_reported = -1
    try:
        try:
            for line in process.stdout:
                output_chunks.append(line)
                match = _YTDLP_PROGRESS_RE.search(line)
                if not match:
                    continue
                # yt-dlp restarts at 0% for each stream (video, then audio), so
                # keep the reported value monotonic within the 5-70 band this
                # stage owns.
                percent = 5 + int(float(match.group(1)) * 0.65)
                if percent <= last_reported:
                    continue
                last_reported = percent
                _emit_progress(progress_callback, percent, "Downloading video...")
            returncode = process.wait()
        except Exception:
            process.kill()
            process.wait(timeout=10)
            raise

        combined_output = "".join(output_chunks)
        if returncode != 0:
            detail = combined_output.strip()
            raise RuntimeError(_ytdlp_failure_message(detail))

        _emit_progress(
            progress_callback, 75, "Local download finished, preparing media..."
        )
        candidates = sorted(
            project_dir.glob(f"{file_prefix}.*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        file_path = next((path for path in candidates if path.suffix != ".part"), None)
        if file_path is None:
            raise RuntimeError("yt-dlp did not produce an output file")

        relative = str(file_path.resolve().relative_to(storage.upload_root))
        source_title = _read_source_title_file(title_file) or (
            _source_title_from_ytdlp_output(combined_output)
        )
        return str(file_path.resolve()), relative, source_title
    finally:
        title_file.unlink(missing_ok=True)


def download_video_from_url(
    url: str,
    project_id: str,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, str, str | None]:
    """Prefer yt-dlp (OAuth2); fall back to Apify only when yt-dlp fails or forced."""
    provider = _ingest_provider()
    has_apify = bool(_apify_api_token())

    if provider == "apify":
        return download_video_with_apify(
            url, project_id, progress_callback=progress_callback
        )

    try:
        return download_video_with_ytdlp(
            url, project_id, progress_callback=progress_callback
        )
    except Exception as exc:
        if provider == "ytdlp" or not has_apify:
            raise
        logger.warning(
            "yt-dlp ingest failed; falling back to Apify: %s",
            type(exc).__name__,
        )
        _emit_progress(
            progress_callback,
            25,
            "Local download failed, trying cloud fetch...",
        )
        return download_video_with_apify(
            url, project_id, progress_callback=progress_callback
        )
