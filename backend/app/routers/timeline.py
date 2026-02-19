from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..database import get_session
from ..models import OperationRecord, Project
from ..schemas import OperationApplyRequest, OperationApplyResponse, OperationHistoryItem
from ..timeline_service import apply_operation, get_timeline_row, load_timeline_state, save_timeline_state

router = APIRouter(prefix="/api/v1/timeline", tags=["timeline"])


@router.post("/operations", response_model=OperationApplyResponse)
def apply_operations(
    payload: OperationApplyRequest,
    project_id: str,
    session: Session = Depends(get_session),
) -> OperationApplyResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    timeline = get_timeline_row(session, project_id)
    state = load_timeline_state(timeline)
    applied_ops: list[str] = []
    for operation in payload.operations:
        try:
            apply_operation(state, operation)
            timeline = save_timeline_state(
                session,
                timeline,
                state,
                source=operation.source,
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


@router.get("/history", response_model=list[OperationHistoryItem])
def get_operation_history(project_id: str, session: Session = Depends(get_session)) -> list[OperationHistoryItem]:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    rows = session.exec(
        select(OperationRecord)
        .where(OperationRecord.project_id == project_id)
        .order_by(OperationRecord.id.desc())
    ).all()
    return [
        OperationHistoryItem(
            id=row.id or 0,
            project_id=row.project_id,
            op_type=row.op_type,
            source=row.source,
            payload_json=row.payload_json,
            created_at=row.created_at.isoformat(),
        )
        for row in rows
    ]
