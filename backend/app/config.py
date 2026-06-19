from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values


def _load_env() -> None:
    backend_dir = Path(__file__).resolve().parents[1]
    # Load .env first, then .env.local (local overrides base)
    # We always write .env values to os.environ so that restarting
    # the server picks up any changes made to .env files.
    for env_file in (backend_dir / ".env", backend_dir / ".env.local"):
        if not env_file.exists():
            continue
        values = dotenv_values(env_file)
        for key, value in values.items():
            if value is None:
                continue
            os.environ[key] = value


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
    render_video_encoder: str
    yt_dlp_bin: str
    max_transcribe_duration_sec: float
    max_concurrent_render_jobs: int
    max_concurrent_ingest_jobs: int
    broll_external_enabled: bool
    broll_external_timeout_sec: float
    broll_external_per_query: int
    broll_external_max_queries: int
    broll_external_download_max_mb: int
    pexels_api_key: str
    pixabay_api_key: str
    broll_ai_enabled: bool
    broll_embed_model: str
    broll_embed_device: str
    broll_entity_enabled: bool
    broll_semantic_weight: float
    broll_entity_weight: float
    broll_metadata_weight: float
    broll_duration_weight: float
    broll_confidence_autopick_threshold: float
    broll_blocklist_terms: list[str]
    broll_shortform_min_sec: float
    broll_shortform_max_sec: float
    broll_audio_reactive_enabled: bool
    broll_audio_reactive_window_sec: float
    broll_audio_reactive_sample_rate: int
    broll_auto_reframe_enabled: bool
    broll_generative_enabled: bool
    broll_generative_api_url: str
    broll_generative_api_key: str
    broll_generative_timeout_sec: float
    broll_generative_model: str
    broll_generative_min_external_score: float
    clerk_secret_key: str


def _as_bool(value: str, default: bool) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "ClipMind API"),
        database_url=os.getenv("DATABASE_URL", "sqlite:///./app.db"),
        upload_dir=os.getenv("UPLOAD_DIR", "./uploads"),
        render_dir=os.getenv("RENDER_DIR", "./renders"),
        tmp_dir=os.getenv("TMP_DIR", "./tmp"),
        allowed_origins=_split_csv(
            os.getenv(
                "ALLOWED_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,"
                "http://localhost:5175,http://127.0.0.1:5175,http://localhost:5176,http://127.0.0.1:5176,"
                "http://localhost:5177,http://127.0.0.1:5177,http://localhost:5178,http://127.0.0.1:5178,"
                "http://localhost:5179,http://127.0.0.1:5179,http://localhost:4173,http://127.0.0.1:4173",
            )
        ),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "1024")),
        storage_backend=os.getenv("STORAGE_BACKEND", "local").lower(),
        ffmpeg_bin=os.getenv("FFMPEG_BIN", "ffmpeg"),
        ffprobe_bin=os.getenv("FFPROBE_BIN", "ffprobe"),
        render_video_encoder=os.getenv("RENDER_VIDEO_ENCODER", "auto").strip().lower()
        or "auto",
        yt_dlp_bin=os.getenv("YT_DLP_BIN", "yt-dlp"),
        max_transcribe_duration_sec=max(
            0.0, float(os.getenv("MAX_TRANSCRIBE_DURATION_SEC", "0"))
        ),
        max_concurrent_render_jobs=max(
            1, int(os.getenv("MAX_CONCURRENT_RENDER_JOBS", "2"))
        ),
        max_concurrent_ingest_jobs=max(
            1, int(os.getenv("MAX_CONCURRENT_INGEST_JOBS", "1"))
        ),
        broll_external_enabled=_as_bool(
            os.getenv("BROLL_EXTERNAL_ENABLED", "true"), True
        ),
        broll_external_timeout_sec=max(
            2.0, float(os.getenv("BROLL_EXTERNAL_TIMEOUT_SEC", "12"))
        ),
        broll_external_per_query=max(
            1, min(40, int(os.getenv("BROLL_EXTERNAL_PER_QUERY", "8")))
        ),
        broll_external_max_queries=max(
            1, min(8, int(os.getenv("BROLL_EXTERNAL_MAX_QUERIES", "4")))
        ),
        broll_external_download_max_mb=max(
            5, int(os.getenv("BROLL_EXTERNAL_DOWNLOAD_MAX_MB", "512"))
        ),
        # Accept common typo "PIXELS_API_KEY" as a fallback to reduce setup friction.
        pexels_api_key=(
            os.getenv("PEXELS_API_KEY") or os.getenv("PIXELS_API_KEY", "")
        ).strip(),
        pixabay_api_key=os.getenv("PIXABAY_API_KEY", "").strip(),
        broll_ai_enabled=_as_bool(os.getenv("BROLL_AI_ENABLED", "true"), True),
        broll_embed_model=os.getenv(
            "BROLL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip(),
        broll_embed_device=os.getenv("BROLL_EMBED_DEVICE", "cpu").strip().lower()
        or "cpu",
        broll_entity_enabled=_as_bool(os.getenv("BROLL_ENTITY_ENABLED", "true"), True),
        broll_semantic_weight=max(
            0.0, float(os.getenv("BROLL_SEMANTIC_WEIGHT", "0.55"))
        ),
        broll_entity_weight=max(0.0, float(os.getenv("BROLL_ENTITY_WEIGHT", "0.20"))),
        broll_metadata_weight=max(
            0.0, float(os.getenv("BROLL_METADATA_WEIGHT", "0.15"))
        ),
        broll_duration_weight=max(
            0.0, float(os.getenv("BROLL_DURATION_WEIGHT", "0.10"))
        ),
        broll_confidence_autopick_threshold=max(
            0.0,
            min(1.0, float(os.getenv("BROLL_CONFIDENCE_AUTOPICK_THRESHOLD", "0.78"))),
        ),
        broll_blocklist_terms=[
            item.lower() for item in _split_csv(os.getenv("BROLL_BLOCKLIST_TERMS", ""))
        ],
        broll_shortform_min_sec=max(
            0.2, float(os.getenv("BROLL_SHORTFORM_MIN_SEC", "0.8"))
        ),
        broll_shortform_max_sec=max(
            0.4, float(os.getenv("BROLL_SHORTFORM_MAX_SEC", "2.0"))
        ),
        broll_audio_reactive_enabled=_as_bool(
            os.getenv("BROLL_AUDIO_REACTIVE_ENABLED", "true"), True
        ),
        broll_audio_reactive_window_sec=max(
            0.05, float(os.getenv("BROLL_AUDIO_REACTIVE_WINDOW_SEC", "0.24"))
        ),
        broll_audio_reactive_sample_rate=max(
            4000, int(os.getenv("BROLL_AUDIO_REACTIVE_SAMPLE_RATE", "11025"))
        ),
        broll_auto_reframe_enabled=_as_bool(
            os.getenv("BROLL_AUTO_REFRAME_ENABLED", "true"), True
        ),
        broll_generative_enabled=_as_bool(
            os.getenv("BROLL_GENERATIVE_ENABLED", "false"), False
        ),
        broll_generative_api_url=os.getenv("BROLL_GENERATIVE_API_URL", "").strip(),
        broll_generative_api_key=os.getenv("BROLL_GENERATIVE_API_KEY", "").strip(),
        broll_generative_timeout_sec=max(
            2.0, float(os.getenv("BROLL_GENERATIVE_TIMEOUT_SEC", "30"))
        ),
        broll_generative_model=os.getenv("BROLL_GENERATIVE_MODEL", "auto").strip(),
        broll_generative_min_external_score=max(
            0.0,
            min(1.0, float(os.getenv("BROLL_GENERATIVE_MIN_EXTERNAL_SCORE", "0.64"))),
        ),
        clerk_secret_key=os.getenv("CLERK_SECRET_KEY", "").strip(),
    )
