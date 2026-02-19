from __future__ import annotations

from fastapi import Depends, HTTPException
from sqlmodel import Session, select

from .database import get_session
from .models import Project


def get_project_or_404(project_id: str, session: Session = Depends(get_session)) -> Project:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

