from __future__ import annotations

import logging
from collections.abc import Generator

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from .config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

connect_args: dict[str, object] = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {
        "check_same_thread": False,
        "timeout": 10,  # Retry for 10s before raising "database is locked"
    }

engine = create_engine(settings.database_url, echo=False, connect_args=connect_args)


@event.listens_for(Engine, "connect")
def _set_sqlite_pragmas(dbapi_connection: object, _connection_record: object) -> None:
    """Enable WAL mode and busy timeout for all SQLite connections.
    WAL allows concurrent reads alongside writes (no more 'database is locked').
    """
    if not settings.database_url.startswith("sqlite"):
        return
    cursor = dbapi_connection.cursor()  # type: ignore[union-attr]
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=10000")  # 10s retry window
    cursor.execute("PRAGMA synchronous=NORMAL")  # Safe + faster than FULL
    cursor.close()


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    _migrate_add_owner_id()
    _migrate_add_job_timeline_version()
    _migrate_add_broll_slot_meaning_json()


def _migrate_add_owner_id() -> None:
    """Add owner_id column to project table if it doesn't exist (safe idempotent migration)."""
    from sqlalchemy import inspect, text

    with engine.connect() as conn:
        inspector = inspect(engine)
        columns = [col["name"] for col in inspector.get_columns("project")]
        if "owner_id" not in columns:
            conn.execute(text("ALTER TABLE project ADD COLUMN owner_id TEXT"))
            conn.commit()
            logger.info("Migration: added owner_id column to project table")


def _migrate_add_job_timeline_version() -> None:
    """Link persisted render jobs to the timeline revision they rendered."""
    from sqlalchemy import inspect, text

    from .models import Job

    table_name = Job.__table__.name
    with engine.connect() as conn:
        inspector = inspect(engine)
        columns = [col["name"] for col in inspector.get_columns(table_name)]
        if "timeline_version" not in columns:
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN timeline_version INTEGER"))
            conn.commit()
            logger.info("Migration: added timeline_version column to %s", table_name)


def _migrate_add_broll_slot_meaning_json() -> None:
    """Persist multilingual B-roll reasoning for databases created before this field."""
    from sqlalchemy import inspect, text

    from .models import BrollSlot

    table_name = BrollSlot.__table__.name
    with engine.connect() as conn:
        inspector = inspect(engine)
        columns = [col["name"] for col in inspector.get_columns(table_name)]
        if "meaning_json" not in columns:
            conn.execute(
                text(
                    f"ALTER TABLE {table_name} "
                    "ADD COLUMN meaning_json TEXT DEFAULT '{}' NOT NULL"
                )
            )
            conn.commit()
            logger.info("Migration: added meaning_json column to %s", table_name)


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
