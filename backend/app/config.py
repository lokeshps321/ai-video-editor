from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    backend_dir = Path(__file__).resolve().parents[1]
    env_file = backend_dir / ".env"
    load_dotenv(env_file, override=False)


_load_env()


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    app_name: str
    database_url: str
    upload_dir: str
    render_dir: str
    tmp_dir: str
    allowed_origins: list[str]
    max_upload_mb: int
    storage_backend: str
    ffmpeg_bin: str
    ffprobe_bin: str
    yt_dlp_bin: str
    max_transcribe_duration_sec: float
    max_concurrent_render_jobs: int
    max_concurrent_ingest_jobs: int


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "Prompt Video Editor API"),
        database_url=os.getenv("DATABASE_URL", "sqlite:///./app.db"),
        upload_dir=os.getenv("UPLOAD_DIR", "./uploads"),
        render_dir=os.getenv("RENDER_DIR", "./renders"),
        tmp_dir=os.getenv("TMP_DIR", "./tmp"),
        allowed_origins=_split_csv(os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "1024")),
        storage_backend=os.getenv("STORAGE_BACKEND", "local").lower(),
        ffmpeg_bin=os.getenv("FFMPEG_BIN", "ffmpeg"),
        ffprobe_bin=os.getenv("FFPROBE_BIN", "ffprobe"),
        yt_dlp_bin=os.getenv("YT_DLP_BIN", "yt-dlp"),
        max_transcribe_duration_sec=max(0.0, float(os.getenv("MAX_TRANSCRIBE_DURATION_SEC", "0"))),
        max_concurrent_render_jobs=max(1, int(os.getenv("MAX_CONCURRENT_RENDER_JOBS", "2"))),
        max_concurrent_ingest_jobs=max(1, int(os.getenv("MAX_CONCURRENT_INGEST_JOBS", "1"))),
    )
