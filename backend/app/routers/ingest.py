from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..database import get_session
from ..ingest_service import validate_ingest_url
from ..jobs import create_job, enqueue_ingest_url_job, find_recent_active_job
from ..models import Job, Project
from ..schemas import IngestUrlRequest, JobResponse

router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])


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


@router.post("/url", response_model=JobResponse)
def ingest_url(
    payload: IngestUrlRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> JobResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        normalized_url = validate_ingest_url(payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    active = find_recent_active_job(session, project_id, kind="ingest_url", within_seconds=0)
    if active:
        return _to_job_response(active)

    job = create_job(session, project_id, kind="ingest_url")
    enqueue_ingest_url_job(job.id, normalized_url)
    return _to_job_response(job)
