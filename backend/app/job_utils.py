"""Compatibility wrappers for legacy worker imports.

Historically, some worker paths imported execute helpers from this module.
Keep these wrappers delegating to the current job processors so stale imports
do not crash with missing symbols during enqueue/worker execution.
"""

from __future__ import annotations

from .jobs import process_ingest_url_job, process_render_job
from .schemas import ExportSettings


def execute_render_job(job_id: str, project_id: str | None = None, export_settings: dict | None = None) -> str:
    """Run a render job using the current processor implementation."""
    _ = project_id
    settings = ExportSettings.model_validate(export_settings or {})
    process_render_job(job_id, settings)
    return job_id


def execute_ingest_job(job_id: str, project_id: str | None = None, url: str = "") -> str:
    """Run an ingest job using the current processor implementation."""
    _ = project_id
    process_ingest_url_job(job_id, url)
    return job_id
