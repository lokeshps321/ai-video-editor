from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..database import get_session
from ..deps import get_current_user, require_project_owner
from ..models import Project
from ..prompt_parser import parse_prompt
from ..schemas import (
    OperationApplyResponse,
    PromptApplyRequest,
    PromptParseRequest,
    PromptParseResponse,
)
from ..timeline_service import (
    apply_operation,
    get_timeline_row,
    load_timeline_state,
    save_timeline_state,
    StaleTimelineError,
    timeline_version_caps,
)

router = APIRouter(prefix="/api/v1/prompt", tags=["prompt"])


@router.post("/parse", response_model=PromptParseResponse)
def parse_prompt_route(
    payload: PromptParseRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> PromptParseResponse:
    return parse_prompt(payload.prompt)


@router.post("/apply", response_model=OperationApplyResponse)
def apply_prompt(
    payload: PromptApplyRequest,
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> OperationApplyResponse:
    require_project_owner(session, project_id, current_user)

    parsed = parse_prompt(payload.prompt)
    if parsed.errors:
        raise HTTPException(
            status_code=400,
            detail={"errors": parsed.errors, "suggestions": parsed.suggestions},
        )

    timeline = get_timeline_row(session, project_id)
    state = load_timeline_state(timeline).model_copy(deep=True)
    applied_ops: list[str] = []
    try:
        for operation in parsed.operations:
            apply_operation(state, operation)
            applied_ops.append(operation.op_type)
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if parsed.operations:
        try:
            timeline = save_timeline_state(
                session,
                timeline,
                state,
                source="prompt",
                operations=parsed.operations,
                expected_version=payload.expected_version,
            )
        except StaleTimelineError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    version, can_undo, can_redo = timeline_version_caps(session, project_id)
    return OperationApplyResponse(
        project_id=project_id,
        version=version,
        timeline=load_timeline_state(timeline),
        applied_ops=applied_ops,
        timeline_can_undo=can_undo,
        timeline_can_redo=can_redo,
    )
