from __future__ import annotations

import logging
from collections.abc import Generator

from sqlalchemy import event, text
from sqlalchemy.engine import Connection, Engine
from sqlmodel import Session, SQLModel, create_engine

from .config import get_settings

logger = logging.getLogger(__name__)
_TIMELINE_MIGRATION_LOCK_ID = 7_404_121_905_031

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
    dialect_name = engine.dialect.name
    if dialect_name == "sqlite":
        with engine.connect() as conn:
            conn.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                SQLModel.metadata.create_all(conn)
                _migrate_timeline_version_uniqueness(connection=conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    elif dialect_name == "postgresql":
        with engine.begin() as conn:
            conn.execute(
                text("SELECT pg_advisory_xact_lock(:lock_id)"),
                {"lock_id": _TIMELINE_MIGRATION_LOCK_ID},
            )
            SQLModel.metadata.create_all(conn)
            _migrate_timeline_version_uniqueness(connection=conn)
    else:
        SQLModel.metadata.create_all(engine)
        _migrate_timeline_version_uniqueness()
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


def _migrate_timeline_version_uniqueness(
    *,
    connection: Connection | None = None,
) -> None:
    """Deduplicate legacy rows and enforce one snapshot per project version."""
    from sqlalchemy import inspect

    from .models import TimelineVersion

    table_name = TimelineVersion.__table__.name
    index_name = "uq_timelineversion_project_version"

    def migrate_locked(conn: Connection) -> None:
        inspector = inspect(conn)
        if table_name not in inspector.get_table_names():
            return
        if conn.dialect.name == "postgresql":
            conn.execute(
                text(f"LOCK TABLE {table_name} IN ACCESS EXCLUSIVE MODE")
            )
        inspector = inspect(conn)
        unique_shapes = {
            tuple(constraint.get("column_names") or [])
            for constraint in inspector.get_unique_constraints(table_name)
        }
        unique_shapes.update(
            tuple(index.get("column_names") or [])
            for index in inspector.get_indexes(table_name)
            if index.get("unique")
        )
        if ("project_id", "version") in unique_shapes:
            return

        conn.execute(
            text(
                f"DELETE FROM {table_name} "
                f"WHERE id NOT IN ("
                f"SELECT MAX(id) FROM {table_name} GROUP BY project_id, version"
                f")"
            )
        )
        if conn.dialect.name in {"sqlite", "postgresql"}:
            create_index = (
                f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} "
                f"ON {table_name} (project_id, version)"
            )
        else:
            create_index = (
                f"CREATE UNIQUE INDEX {index_name} "
                f"ON {table_name} (project_id, version)"
            )
        conn.execute(text(create_index))
        logger.info(
            "Migration: enforced unique timeline versions on %s",
            table_name,
        )

    if connection is not None:
        migrate_locked(connection)
        return

    dialect_name = engine.dialect.name
    if dialect_name == "sqlite":
        with engine.connect() as conn:
            conn.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                migrate_locked(conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return
    if dialect_name == "postgresql":
        with engine.begin() as conn:
            conn.execute(
                text("SELECT pg_advisory_xact_lock(:lock_id)"),
                {"lock_id": _TIMELINE_MIGRATION_LOCK_ID},
            )
            migrate_locked(conn)
        return
    with engine.begin() as conn:
        migrate_locked(conn)


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
