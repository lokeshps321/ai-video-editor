from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from shutil import which

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings

logger = logging.getLogger(__name__)
from .database import init_db
from .jobs import (
    fail_orphaned_active_jobs,
    start_ingest_workers,
    start_render_workers,
    stop_ingest_workers,
    stop_render_workers,
)
from .routers.ingest import router as ingest_router
from .routers.media import router as media_router
from .routers.projects import router as projects_router
from .routers.prompt import router as prompt_router
from .routers.broll import router as broll_router
from .routers.render import router as render_router
from .routers.timeline import router as timeline_router
from .routers.transcript import router as transcript_router
from .routers.vibe import router as vibe_router

settings = get_settings()
_startup_log = logging.getLogger(__name__)


def _should_start_background_workers() -> bool:
    raw = (os.getenv("DISABLE_BACKGROUND_WORKERS", "") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return False
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return True


@asynccontextmanager
async def _lifespan(application: FastAPI):  # noqa: ARG001
    # ── startup ──────────────────────────────────────────────────────
    init_db()
    recovered = fail_orphaned_active_jobs()
    if recovered:
        _startup_log.info(
            "[jobs] Marked %d orphaned queued/running jobs as failed on startup",
            recovered,
        )
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.render_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.tmp_dir).mkdir(parents=True, exist_ok=True)
    if _should_start_background_workers():
        start_render_workers()
        start_ingest_workers()

    yield  # app is now running

    # ── shutdown ─────────────────────────────────────────────────────
    if _should_start_background_workers():
        stop_render_workers()
        stop_ingest_workers()


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects_router)
app.include_router(media_router)
app.include_router(ingest_router)
app.include_router(prompt_router)
app.include_router(broll_router)
app.include_router(timeline_router)
app.include_router(render_router)
app.include_router(transcript_router)
app.include_router(vibe_router)


# Custom video serving endpoints with range request support for seeking
# These MUST come before StaticFiles mounts to intercept video requests
@app.get("/static/renders/{file_path:path}")
async def serve_render_video(file_path: str, request: Request):
    video_path = Path(settings.render_dir) / file_path
    if not video_path.exists() or not video_path.is_file():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="File not found")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        # Parse range header (e.g., "bytes=0-1023")
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0]) if range_match[0] else 0
        end = (
            int(range_match[1])
            if len(range_match) > 1 and range_match[1]
            else file_size - 1
        )
        end = min(end, file_size - 1)
        content_length = end - start + 1

        def iter_file():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(8192, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(iter_file(), status_code=206, headers=headers)

    # No range request, return full file
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/static/uploads/{file_path:path}")
async def serve_upload_video(file_path: str, request: Request):
    video_path = Path(settings.upload_dir) / file_path
    if not video_path.exists() or not video_path.is_file():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="File not found")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        # Parse range header (e.g., "bytes=0-1023")
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0]) if range_match[0] else 0
        end = (
            int(range_match[1])
            if len(range_match) > 1 and range_match[1]
            else file_size - 1
        )
        end = min(end, file_size - 1)
        content_length = end - start + 1

        def iter_file():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(8192, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(iter_file(), status_code=206, headers=headers)

    # No range request, return full file
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/health")
def health() -> dict[str, str]:
    vocal_backend = (
        os.getenv("TRANSCRIBE_VOCAL_ISOLATION_BACKEND", "auto") or "auto"
    ).strip() or "auto"
    vocal_enabled = (
        (os.getenv("TRANSCRIBE_VOCAL_ISOLATION_ENABLED", "true") or "true")
        .strip()
        .lower()
    )

    # Check Redis health if using RQ workers
    redis_status = "not_configured"
    if os.getenv("USE_RQ_WORKERS", "false").lower() in {"1", "true", "yes", "on"}:
        try:
            from .queue import check_redis_health

            redis_status = "connected" if check_redis_health() else "disconnected"
        except Exception as exc:
            logger.warning("Redis health probe failed: %s", exc)
            redis_status = "error"

    return {
        "status": "ok",
        "ffmpeg": "available" if which(settings.ffmpeg_bin) else "missing",
        "ffprobe": "available" if which(settings.ffprobe_bin) else "missing",
        "yt_dlp": "available" if which(settings.yt_dlp_bin) else "missing",
        "redis": redis_status,
        "vocal_isolation_enabled": "true"
        if vocal_enabled in {"1", "true", "yes", "on"}
        else "false",
        "vocal_isolation_backend": vocal_backend,
    }
