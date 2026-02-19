from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Project(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    name: str
    fps: int = 30
    width: int = 1080
    height: int = 1920
    created_at: datetime = Field(default_factory=_utcnow, nullable=False)


class Timeline(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    project_id: str = Field(index=True, unique=True)
    version: int = 0
    state_json: str
    created_at: datetime = Field(default_factory=_utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=_utcnow, nullable=False)


class TimelineVersion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: str = Field(index=True)
    version: int = Field(index=True)
    state_json: str
    created_at: datetime = Field(default_factory=_utcnow, nullable=False)


class MediaAsset(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    project_id: str = Field(index=True)
    media_type: str
    filename: str
    storage_path: str
    mime_type: str
    duration_sec: Optional[float] = None
    metadata_json: str = "{}"
    created_at: datetime = Field(default_factory=_utcnow, nullable=False)


class Transcript(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    project_id: str = Field(index=True)
    asset_id: str = Field(index=True)
    source: str = "mock"
    language: Optional[str] = None
    text: str
    words_json: str = "[]"
    duration_sec: float = 0.0
    is_mock: bool = False
    created_at: datetime = Field(default_factory=_utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=_utcnow, nullable=False)


class OperationRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: str = Field(index=True)
    op_type: str = Field(index=True)
    source: str = "ui"
    payload_json: str
    created_at: datetime = Field(default_factory=_utcnow, nullable=False)


class Job(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    project_id: str = Field(index=True)
    kind: str
    status: str = "queued"
    progress: int = 0
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=_utcnow, nullable=False)


class JobEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(index=True)
    project_id: str = Field(index=True)
    stage: str
    status: str
    progress: int
    message: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow, nullable=False)
