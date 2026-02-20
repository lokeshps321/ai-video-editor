from __future__ import annotations

from pathlib import Path
from shutil import which

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import get_settings
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
app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    recovered = fail_orphaned_active_jobs()
    if recovered:
        print(f"[jobs] Marked {recovered} orphaned queued/running jobs as failed on startup")
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.render_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.tmp_dir).mkdir(parents=True, exist_ok=True)
    start_render_workers()
    start_ingest_workers()


@app.on_event("shutdown")
def on_shutdown() -> None:
    stop_render_workers()
    stop_ingest_workers()


app.include_router(projects_router)
app.include_router(media_router)
app.include_router(ingest_router)
app.include_router(prompt_router)
app.include_router(broll_router)
app.include_router(timeline_router)
app.include_router(render_router)
app.include_router(transcript_router)
app.include_router(vibe_router)

app.mount("/static/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")
app.mount("/static/renders", StaticFiles(directory=settings.render_dir), name="renders")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "ffmpeg": "available" if which(settings.ffmpeg_bin) else "missing",
        "ffprobe": "available" if which(settings.ffprobe_bin) else "missing",
        "yt_dlp": "available" if which(settings.yt_dlp_bin) else "missing",
    }
