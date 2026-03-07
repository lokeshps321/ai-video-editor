from __future__ import annotations

import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from ..database import get_session
from ..jobs import create_job, enqueue_render_job, find_recent_active_job, get_latest_job_event, list_job_events
from ..models import Job, Project
from ..schemas import ExportSettings, JobEventResponse, JobResponse, RenderRequest
from ..timeline_service import get_timeline_row, load_timeline_state
from ..storage import storage

router = APIRouter(prefix="/api/v1", tags=["render"])


def _to_job_response(session: Session, job: Job) -> JobResponse:
    latest_event = get_latest_job_event(session, job.id)
    return JobResponse(
        id=job.id,
        project_id=job.project_id,
        kind=job.kind,
        status=job.status,
        progress=job.progress,
        stage=latest_event.stage if latest_event else None,
        message=latest_event.message if latest_event else None,
        output_path=job.output_path,
        error=job.error,
    )


@router.post("/render/preview", response_model=JobResponse)
def render_preview(
    project_id: str,
    force: bool = False,
    payload: RenderRequest | None = None,
    session: Session = Depends(get_session),
) -> JobResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not force:
        active = find_recent_active_job(session, project_id, kind="preview", within_seconds=0)
        if active:
            return _to_job_response(session, active)

    timeline = get_timeline_row(session, project_id)
    state = load_timeline_state(timeline)
    inferred_aspect_ratio = "9:16" if state.resolution.height >= state.resolution.width else "16:9"
    job = create_job(session, project_id, kind="preview")
    request = ExportSettings(
        format="mp4",
        aspect_ratio=payload.aspect_ratio if payload is not None else inferred_aspect_ratio,
        resolution="720p",
        fps=payload.fps if payload is not None else state.fps,
        quality="low",
    )
    enqueue_render_job(job.id, request)
    return _to_job_response(session, job)


@router.post("/render/export", response_model=JobResponse)
def render_export(
    payload: RenderRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> JobResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    active = find_recent_active_job(session, project_id, kind="export", within_seconds=0)
    if active:
        return _to_job_response(session, active)

    job = create_job(session, project_id, kind="export")
    export_settings = ExportSettings.model_validate(payload.model_dump())
    enqueue_render_job(job.id, export_settings)
    return _to_job_response(session, job)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, session: Session = Depends(get_session)) -> JobResponse:
    job = session.exec(select(Job).where(Job.id == job_id)).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _to_job_response(session, job)


@router.get("/jobs/{job_id}/events", response_model=list[JobEventResponse])
def get_job_events(job_id: str, session: Session = Depends(get_session)) -> list[JobEventResponse]:
    job = session.exec(select(Job).where(Job.id == job_id)).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    rows = list_job_events(session, job_id)
    return [
        JobEventResponse(
            id=row.id or 0,
            job_id=row.job_id,
            project_id=row.project_id,
            stage=row.stage,
            status=row.status,
            progress=row.progress,
            message=row.message,
            created_at=row.created_at.isoformat(),
        )
        for row in rows
    ]


@router.get("/jobs/{job_id}/download")
def download_job_output(job_id: str, session: Session = Depends(get_session)) -> FileResponse:
    job = session.exec(select(Job).where(Job.id == job_id)).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed" or not job.output_path:
        raise HTTPException(status_code=400, detail="Job not completed or has no output")

    abs_path = storage.resolve_render_asset(job.output_path)
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="Output file not found on disk")

    ext = os.path.splitext(abs_path)[1]
    filename = f"export{ext}" if ext else "export.mp4"
    return FileResponse(
        path=abs_path,
        filename=filename,
        content_disposition_type="attachment"
    )
