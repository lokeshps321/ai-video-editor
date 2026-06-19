from __future__ import annotations

import logging
import shutil
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from ..database import get_session
from ..deps import get_current_user
from ..models import (
    BrollCandidate,
    BrollChoice,
    BrollPlan,
    BrollPlanBeat,
    BrollSlot,
    Job,
    JobEvent,
    MediaAsset,
    OperationRecord,
    Project,
    Timeline,
    TimelineVersion,
    Transcript,
)
from ..project_response import build_project_response
from ..schemas import ProjectCreateRequest, ProjectResponse
from ..storage import storage
from ..timeline_service import (
    create_timeline_for_project,
    get_timeline_row,
    load_timeline_state,
    redo_timeline,
    undo_timeline,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


class ProjectRenameRequest(BaseModel):
    name: str


@router.post("", response_model=ProjectResponse)
def create_project(
    payload: ProjectCreateRequest,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> ProjectResponse:
    project = Project(
        name=payload.name,
        fps=payload.fps,
        width=payload.width,
        height=payload.height,
        owner_id=current_user["sub"],
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    create_timeline_for_project(session, project)
    return build_project_response(session, project)


@router.get("", response_model=list[ProjectResponse])
def list_projects(
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> list[ProjectResponse]:
    projects = session.exec(
        select(Project)
        .where(Project.owner_id == current_user["sub"])
        .order_by(Project.created_at.desc())
    ).all()
    return [build_project_response(session, project) for project in projects]


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> ProjectResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id and project.owner_id != current_user["sub"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return build_project_response(session, project)


@router.patch("/{project_id}", response_model=ProjectResponse)
def rename_project(
    project_id: str,
    payload: ProjectRenameRequest,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> ProjectResponse:
    """Rename a project."""
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id and project.owner_id != current_user["sub"]:
        raise HTTPException(status_code=403, detail="Access denied")
    new_name = payload.name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Project name cannot be empty")
    project.name = new_name
    session.add(project)
    session.commit()
    session.refresh(project)
    return build_project_response(session, project)


@router.delete("/{project_id}")
def delete_project(
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, str]:
    """Delete a project and all associated data."""
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id and project.owner_id != current_user["sub"]:
        raise HTTPException(status_code=403, detail="Access denied")

    # Cascade-delete all related rows in dependency order.
    _cascade_tables = [
        (JobEvent, "project_id"),
        (Job, "project_id"),
        (BrollChoice, "project_id"),
        (BrollCandidate, "project_id"),
        (BrollSlot, "project_id"),
        (BrollPlanBeat, "project_id"),
        (BrollPlan, "project_id"),
        (OperationRecord, "project_id"),
        (Transcript, "project_id"),
        (TimelineVersion, "project_id"),
        (Timeline, "project_id"),
        (MediaAsset, "project_id"),
    ]
    for model_cls, col_name in _cascade_tables:
        col = getattr(model_cls, col_name)
        rows = session.exec(select(model_cls).where(col == project_id)).all()
        for row in rows:
            session.delete(row)

    session.delete(project)
    session.commit()

    # Clean up upload/render directories on disk (best-effort).
    for root_dir in [storage.upload_root, storage.render_root]:
        project_dir = root_dir / project_id
        if project_dir.exists():
            try:
                shutil.rmtree(project_dir)
            except Exception as exc:
                logger.warning("Failed to clean up %s: %s", project_dir, exc)

    return {"detail": "Project deleted"}


@router.post("/{project_id}/undo", response_model=ProjectResponse)
def undo(
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> ProjectResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id and project.owner_id != current_user["sub"]:
        raise HTTPException(status_code=403, detail="Access denied")
    undo_timeline(session, project_id)
    return build_project_response(session, project)


@router.post("/{project_id}/redo", response_model=ProjectResponse)
def redo(
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> ProjectResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id and project.owner_id != current_user["sub"]:
        raise HTTPException(status_code=403, detail="Access denied")
    redo_timeline(session, project_id)
    return build_project_response(session, project)
