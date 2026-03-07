from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from .config import get_settings

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
    cursor.execute("PRAGMA synchronous=NORMAL")   # Safe + faster than FULL
    cursor.close()


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session

