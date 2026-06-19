"""Redis Queue configuration for background job processing."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from redis import Redis
from rq import Queue

logger = logging.getLogger(__name__)


def _get_redis_url() -> str:
    """Get Redis URL from environment or use default local Redis."""
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


@lru_cache
def get_redis_connection() -> Redis:
    """Get or create Redis connection."""
    redis_url = _get_redis_url()
    return Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


@lru_cache
def get_render_queue() -> Queue:
    """Get the render job queue."""
    return Queue("renders", connection=get_redis_connection())


@lru_cache
def get_ingest_queue() -> Queue:
    """Get the ingest job queue."""
    return Queue("ingests", connection=get_redis_connection())


@lru_cache
def get_vocal_isolation_queue() -> Queue:
    """Get the vocal isolation pre-compute queue."""
    return Queue("vocal_isolation", connection=get_redis_connection())


def check_redis_health() -> bool:
    """Check if Redis is reachable."""
    try:
        redis = get_redis_connection()
        return redis.ping()
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        return False
