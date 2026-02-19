from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..database import get_session
from ..jobs import create_job, enqueue_render_job, find_recent_active_job, list_job_events
from ..models import Job, Project
from ..schemas import ExportSettings, JobEventResponse, JobResponse, RenderRequest

router = APIRouter(prefix="/api/v1", tags=["render"])


def _to_job_response(job: Job) -> JobResponse:
    return JobResponse(
        id=job.id,
        project_id=job.project_id,
        kind=job.kind,
        status=job.status,
        progress=job.progress,
        output_path=job.output_path,
        error=job.error,
    )


@router.post("/render/preview", response_model=JobResponse)
def render_preview(
    project_id: str,
    force: bool = False,
    session: Session = Depends(get_session),
) -> JobResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not force:
        active = find_recent_active_job(session, project_id, kind="preview", within_seconds=0)
        if active:
            return _to_job_response(active)

    job = create_job(session, project_id, kind="preview")
    request = ExportSettings(format="mp4", resolution="720p", fps=24, quality="low")
    enqueue_render_job(job.id, request)
    return _to_job_response(job)


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
        return _to_job_response(active)

    job = create_job(session, project_id, kind="export")
    export_settings = ExportSettings.model_validate(payload.model_dump())
    enqueue_render_job(job.id, export_settings)
    return _to_job_response(job)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, session: Session = Depends(get_session)) -> JobResponse:
    job = session.exec(select(Job).where(Job.id == job_id)).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _to_job_response(job)


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
