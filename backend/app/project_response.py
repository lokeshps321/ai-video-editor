from __future__ import annotations

from sqlmodel import Session

from .models import Project
from .schemas import ProjectResponse
from .timeline_service import get_timeline_row, load_timeline_state, timeline_version_caps


def build_project_response(session: Session, project: Project) -> ProjectResponse:
    timeline = get_timeline_row(session, project.id)
    version, can_undo, can_redo = timeline_version_caps(session, project.id)
    return ProjectResponse(
        id=project.id,
        name=project.name,
        fps=project.fps,
        width=project.width,
        height=project.height,
        timeline=load_timeline_state(timeline),
        timeline_version=version,
        timeline_can_undo=can_undo,
        timeline_can_redo=can_redo,
    )
