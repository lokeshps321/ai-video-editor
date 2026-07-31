from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

pytest.importorskip("sqlmodel")

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from app import database


def test_timeline_version_migration_deduplicates_and_adds_unique_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = getattr(database, "_migrate_timeline_version_uniqueness", None)
    assert migration is not None

    migration_engine = create_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    with migration_engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE timelineversion ("
                "id INTEGER PRIMARY KEY, "
                "project_id TEXT NOT NULL, "
                "version INTEGER NOT NULL, "
                "state_json TEXT NOT NULL"
                ")"
            )
        )
        conn.execute(
            text(
                "INSERT INTO timelineversion "
                "(id, project_id, version, state_json) VALUES "
                "(1, 'project-a', 1, 'old'), "
                "(2, 'project-a', 1, 'new'), "
                "(3, 'project-a', 2, 'next')"
            )
        )

    monkeypatch.setattr(database, "engine", migration_engine)
    migration()

    with migration_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, project_id, version, state_json "
                "FROM timelineversion ORDER BY id"
            )
        ).all()
    assert rows == [
        (2, "project-a", 1, "new"),
        (3, "project-a", 2, "next"),
    ]
    assert any(
        index["unique"]
        and index["column_names"] == ["project_id", "version"]
        for index in inspect(migration_engine).get_indexes("timelineversion")
    )

    with pytest.raises(IntegrityError):
        with migration_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO timelineversion "
                    "(project_id, version, state_json) "
                    "VALUES ('project-a', 2, 'duplicate')"
                )
            )


def test_timeline_version_migration_is_safe_during_concurrent_sqlite_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration_engine = create_engine(
        f"sqlite:///{tmp_path / 'concurrent-migration.db'}",
        connect_args={"check_same_thread": False, "timeout": 5},
    )
    with migration_engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE timelineversion ("
                "id INTEGER PRIMARY KEY, "
                "project_id TEXT NOT NULL, "
                "version INTEGER NOT NULL, "
                "state_json TEXT NOT NULL"
                ")"
            )
        )
        conn.execute(
            text(
                "INSERT INTO timelineversion "
                "(id, project_id, version, state_json) VALUES "
                "(1, 'project-a', 1, 'old'), "
                "(2, 'project-a', 1, 'new')"
            )
        )

    monkeypatch.setattr(database, "engine", migration_engine)
    workers = 6
    barrier = Barrier(workers)

    def migrate_from_separate_startup() -> None:
        barrier.wait()
        database._migrate_timeline_version_uniqueness()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(migrate_from_separate_startup)
            for _ in range(workers)
        ]
        for future in futures:
            future.result()

    with migration_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, state_json FROM timelineversion "
                "WHERE project_id = 'project-a' AND version = 1"
            )
        ).all()
    assert rows == [(2, "new")]
    assert any(
        index["unique"]
        and index["column_names"] == ["project_id", "version"]
        for index in inspect(migration_engine).get_indexes("timelineversion")
    )


def test_concurrent_sqlite_init_serializes_create_all_and_version_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration_engine = create_engine(
        f"sqlite:///{tmp_path / 'concurrent-init.db'}",
        connect_args={"check_same_thread": False, "timeout": 5},
    )
    monkeypatch.setattr(database, "engine", migration_engine)
    workers = 4
    barrier = Barrier(workers)

    def initialize_from_separate_startup() -> None:
        barrier.wait()
        database.init_db()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(initialize_from_separate_startup)
            for _ in range(workers)
        ]
        for future in futures:
            future.result()

    inspector = inspect(migration_engine)
    assert "timelineversion" in inspector.get_table_names()
    unique_shapes = {
        tuple(constraint.get("column_names") or [])
        for constraint in inspector.get_unique_constraints("timelineversion")
    }
    unique_shapes.update(
        tuple(index.get("column_names") or [])
        for index in inspector.get_indexes("timelineversion")
        if index.get("unique")
    )
    assert ("project_id", "version") in unique_shapes
