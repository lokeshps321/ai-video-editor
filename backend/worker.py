#!/usr/bin/env python3
"""RQ Worker for processing background jobs.

Usage:
    python worker.py              # Start worker with default queues
    python worker.py --queues renders,ingests  # Specific queues
    python worker.py --verbosity DEBUG  # More logging

Run this in a separate terminal from your main FastAPI app.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add backend to path so imports work
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from redis import Redis
from rq import Queue, Worker
from rq.logutils import setup_loghandlers

from app.queue import get_redis_connection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ Worker for ClipMind")
    parser.add_argument(
        "--queues",
        type=str,
        default="renders,ingests",
        help="Comma-separated list of queues to listen to (default: renders,ingests)",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--burst",
        action="store_true",
        help="Run in burst mode (exit after processing all available jobs)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Setup logging
    setup_loghandlers(args.verbosity)
    logger = logging.getLogger(__name__)
    
    # Get Redis connection
    redis_conn = get_redis_connection()
    
    # Test connection
    try:
        redis_conn.ping()
        logger.info(f"Connected to Redis")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        logger.error("Make sure Redis is running: redis-server")
        return 1
    
    # Parse queue names
    queue_names = [q.strip() for q in args.queues.split(",")]
    queues = [Queue(name, connection=redis_conn) for name in queue_names]
    
    logger.info(f"Listening on queues: {', '.join(queue_names)}")
    logger.info(f"Verbosity: {args.verbosity}")
    if args.burst:
        logger.info("Running in burst mode")
    
    # Create and start worker
    worker = Worker(queues, connection=redis_conn)
    
    try:
        # Register signal handlers for graceful shutdown
        worker.work(
            burst=args.burst,
            logging_level=args.verbosity,
        )
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
