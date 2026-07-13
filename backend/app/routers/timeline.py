from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..database import get_session
from ..deps import get_current_user
from ..models import MediaAsset, OperationRecord, Project
from ..project_response import build_project_response
from ..schemas import (
    Crop,
    CropKeyframe,
    OperationApplyRequest,
    OperationApplyResponse,
    OperationHistoryItem,
    OperationPayload,
    SmartReframeRequest,
    SmartReframeResponse,
)
from ..smart_reframe_service import plan_reel_smart_reframe
from ..timeline_service import (
    apply_operation,
    get_timeline_row,
    load_timeline_state,
    save_timeline_state,
    timeline_version_caps,
)
from ._broll_media import _ensure_asset_focus_metadata

router = APIRouter(prefix="/api/v1/timeline", tags=["timeline"])


@router.post("/operations", response_model=OperationApplyResponse)
def apply_operations(
    payload: OperationApplyRequest,
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
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

    version, can_undo, can_redo = timeline_version_caps(session, project_id)
    return OperationApplyResponse(
        project_id=project_id,
        version=version,
        timeline=load_timeline_state(timeline),
        applied_ops=applied_ops,
        timeline_can_undo=can_undo,
        timeline_can_redo=can_redo,
    )


@router.post("/smart-reframe", response_model=SmartReframeResponse)
def smart_reframe_main_video(
    payload: SmartReframeRequest,
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> SmartReframeResponse:
    """Apply a non-destructive subject-aware 9:16 crop to main video clips."""

    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    timeline = get_timeline_row(session, project_id)
    state = load_timeline_state(timeline)
    requested_ids = {clip_id for clip_id in payload.clip_ids if clip_id.strip()}
    video_clips = [
        clip
        for track in state.tracks
        if track.kind == "video"
        for clip in track.clips
        if not requested_ids or clip.id in requested_ids
    ]
    asset_ids = {clip.asset_id for clip in video_clips}
    assets = list(
        session.exec(
            select(MediaAsset).where(
                MediaAsset.project_id == project_id,
                MediaAsset.id.in_(asset_ids),
            )
        ).all()
    ) if asset_ids else []
    assets_by_id = {asset.id: asset for asset in assets}

    reframed = 0
    tracked = 0
    center_crop = 0
    skipped = 0
    for clip in video_clips:
        asset = assets_by_id.get(clip.asset_id)
        if asset is None or asset.media_type != "video":
            skipped += 1
            continue

        metadata = _ensure_asset_focus_metadata(session, asset)
        try:
            width = int(metadata.get("width") or 0)
            height = int(metadata.get("height") or 0)
        except (TypeError, ValueError):
            skipped += 1
            continue
        try:
            focus_x = (
                float(metadata["focus_x"])
                if metadata.get("focus_x") is not None
                else None
            )
        except (TypeError, ValueError):
            focus_x = None

        source_duration = max(float(clip.end_sec) - float(clip.start_sec), 0.0)
        plan = plan_reel_smart_reframe(
            width=width,
            height=height,
            clip_duration_sec=source_duration,
            focus_x=focus_x,
            focus_track=metadata.get("focus_track"),
            clip_start_sec=float(clip.start_sec),
        )
        if plan.crop is None:
            skipped += 1
            continue

        clip.transform.crop = Crop(**plan.crop)
        clip.transform.crop_keyframes = [
            CropKeyframe(**item) for item in plan.crop_keyframes
        ]
        reframed += 1
        if plan.uses_subject_tracking:
            tracked += 1
        else:
            center_crop += 1

    if reframed:
        operation = OperationPayload(
            op_type="smart_reframe",
            params={
                "target_aspect_ratio": "9:16",
                "clip_ids": [clip.id for clip in video_clips],
                "reframed_clip_count": reframed,
                "tracked_clip_count": tracked,
                "center_crop_clip_count": center_crop,
            },
            source="ui",
        )
        timeline = save_timeline_state(
            session,
            timeline,
            state,
            source="ui",
            operation=operation,
        )

    version, can_undo, can_redo = timeline_version_caps(session, project_id)
    return SmartReframeResponse(
        project_id=project_id,
        reframed_clip_count=reframed,
        tracked_clip_count=tracked,
        center_crop_clip_count=center_crop,
        skipped_clip_count=skipped,
        version=version,
        timeline=load_timeline_state(timeline),
        timeline_can_undo=can_undo,
        timeline_can_redo=can_redo,
    )


@router.get("/history", response_model=list[OperationHistoryItem])
def get_operation_history(
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> list[OperationHistoryItem]:
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
