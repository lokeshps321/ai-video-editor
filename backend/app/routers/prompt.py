from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..database import get_session
from ..models import Project
from ..prompt_parser import parse_prompt
from ..schemas import OperationApplyResponse, PromptApplyRequest, PromptParseRequest, PromptParseResponse
from ..timeline_service import (
    apply_operation,
    get_timeline_row,
    load_timeline_state,
    save_timeline_state,
)

router = APIRouter(prefix="/api/v1/prompt", tags=["prompt"])


@router.post("/parse", response_model=PromptParseResponse)
def parse_prompt_route(payload: PromptParseRequest) -> PromptParseResponse:
    return parse_prompt(payload.prompt)


@router.post("/apply", response_model=OperationApplyResponse)
def apply_prompt(
    payload: PromptApplyRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> OperationApplyResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    parsed = parse_prompt(payload.prompt)
    if parsed.errors:
        raise HTTPException(status_code=400, detail={"errors": parsed.errors, "suggestions": parsed.suggestions})

    timeline = get_timeline_row(session, project_id)
    state = load_timeline_state(timeline)
    applied_ops: list[str] = []
    for operation in parsed.operations:
        try:
            apply_operation(state, operation)
            timeline = save_timeline_state(
                session,
                timeline,
                state,
                source="prompt",
                operation=operation,
            )
            applied_ops.append(operation.op_type)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return OperationApplyResponse(
        project_id=project_id,
        version=timeline.version,
        timeline=load_timeline_state(timeline),
        applied_ops=applied_ops,
    )

