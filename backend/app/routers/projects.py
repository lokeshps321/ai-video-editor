from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..database import get_session
from ..models import Project
from ..schemas import ProjectCreateRequest, ProjectResponse
from ..timeline_service import (
    create_timeline_for_project,
    get_timeline_row,
    load_timeline_state,
    redo_timeline,
    undo_timeline,
)

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


@router.post("", response_model=ProjectResponse)
def create_project(payload: ProjectCreateRequest, session: Session = Depends(get_session)) -> ProjectResponse:
    project = Project(
        name=payload.name,
        fps=payload.fps,
        width=payload.width,
        height=payload.height,
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    timeline = create_timeline_for_project(session, project)
    return ProjectResponse(
        id=project.id,
        name=project.name,
        fps=project.fps,
        width=project.width,
        height=project.height,
        timeline=load_timeline_state(timeline),
    )


@router.get("", response_model=list[ProjectResponse])
def list_projects(session: Session = Depends(get_session)) -> list[ProjectResponse]:
    projects = session.exec(select(Project).order_by(Project.created_at.desc())).all()
    results: list[ProjectResponse] = []
    for project in projects:
        timeline = get_timeline_row(session, project.id)
        results.append(
            ProjectResponse(
                id=project.id,
                name=project.name,
                fps=project.fps,
                width=project.width,
                height=project.height,
                timeline=load_timeline_state(timeline),
            )
        )
    return results


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str, session: Session = Depends(get_session)) -> ProjectResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    timeline = get_timeline_row(session, project.id)
    return ProjectResponse(
        id=project.id,
        name=project.name,
        fps=project.fps,
        width=project.width,
        height=project.height,
        timeline=load_timeline_state(timeline),
    )


@router.post("/{project_id}/undo", response_model=ProjectResponse)
def undo(project_id: str, session: Session = Depends(get_session)) -> ProjectResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    timeline = undo_timeline(session, project_id)
    return ProjectResponse(
        id=project.id,
        name=project.name,
        fps=project.fps,
        width=project.width,
        height=project.height,
        timeline=load_timeline_state(timeline),
    )


@router.post("/{project_id}/redo", response_model=ProjectResponse)
def redo(project_id: str, session: Session = Depends(get_session)) -> ProjectResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    timeline = redo_timeline(session, project_id)
    return ProjectResponse(
        id=project.id,
        name=project.name,
        fps=project.fps,
        width=project.width,
        height=project.height,
        timeline=load_timeline_state(timeline),
    )

