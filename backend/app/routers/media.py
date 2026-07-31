from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from ..database import get_session
from ..deps import get_current_user, require_project_owner
from ..jobs import (
    create_job,
    enqueue_vocal_isolation_job,
    should_precompute_vocal_isolation,
)
from ..media_utils import (
    extract_frame_thumbnail,
    extract_waveform_peaks,
    infer_media_type,
    probe_duration_seconds,
    probe_stream_flags,
)
from ..models import MediaAsset, Project
from ..schemas import MediaUploadResponse
from ..storage import storage

router = APIRouter(prefix="/api/v1/media", tags=["media"])


@router.post("/upload", response_model=MediaUploadResponse)
async def upload_media(
    project_id: str = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> MediaUploadResponse:
    require_project_owner(session, project_id, current_user)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid file")

    absolute_path, relative_path = await storage.save_upload(file, project_id)
    media_type = infer_media_type(file.content_type or "", file.filename)
    duration_sec = (
        probe_duration_seconds(absolute_path)
        if media_type in {"video", "audio"}
        else None
    )
    stream_flags = (
        probe_stream_flags(absolute_path)
        if media_type in {"video", "audio"}
        else {"has_video": False, "has_audio": False}
    )
    metadata = {"content_type": file.content_type, **stream_flags}

    asset = MediaAsset(
        project_id=project_id,
        media_type=media_type,
        filename=file.filename,
        storage_path=relative_path,
        mime_type=file.content_type or "application/octet-stream",
        duration_sec=duration_sec,
        metadata_json=json.dumps(metadata),
    )
    session.add(asset)
    session.commit()
    session.refresh(asset)

    # Trigger background vocal isolation if applicable
    if should_precompute_vocal_isolation(asset):
        job = create_job(session, project_id, "vocal_isolation")
        enqueue_vocal_isolation_job(job.id, asset.id)

    return MediaUploadResponse(
        id=asset.id,
        project_id=asset.project_id,
        media_type=asset.media_type,
        filename=asset.filename,
        storage_path=storage.to_public_upload_path(relative_path),
        duration_sec=asset.duration_sec,
    )


@router.get("", response_model=list[MediaUploadResponse])
def list_media(
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> list[MediaUploadResponse]:
    require_project_owner(session, project_id, current_user)
    items = session.exec(
        select(MediaAsset)
        .where(MediaAsset.project_id == project_id)
        .order_by(MediaAsset.created_at.desc())
    ).all()
    return [
        MediaUploadResponse(
            id=item.id,
            project_id=item.project_id,
            media_type=item.media_type,
            filename=item.filename,
            storage_path=storage.to_public_upload_path(item.storage_path),
            duration_sec=item.duration_sec,
        )
        for item in items
    ]


@router.get("/{asset_id}/thumbnail")
def get_thumbnail(
    asset_id: str,
    t: float = 0.0,
    w: int = 160,
    session: Session = Depends(get_session),
) -> FileResponse:
    """Return a single cached JPEG frame for timeline filmstrips.

    Served without auth (uploaded media is already public via ``/static``) so
    the frontend can load it directly through an <img> tag.
    """
    asset = session.exec(select(MediaAsset).where(MediaAsset.id == asset_id)).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Media asset not found")
    if asset.media_type != "video":
        raise HTTPException(status_code=400, detail="Asset is not a video")

    width = max(48, min(480, int(w)))
    time_key = max(0.0, round(float(t), 2))
    # Clamp to just inside the duration so a stale/over-range request still
    # yields a real frame instead of failing on a seek past the end.
    if asset.duration_sec and asset.duration_sec > 0:
        time_key = min(time_key, max(0.0, round(asset.duration_sec - 0.1, 2)))
    cache_dir = Path(storage.tmp_root) / "thumbs" / asset_id
    out_path = cache_dir / f"{time_key:.2f}_{width}.jpg"

    if not out_path.exists():
        source_path = storage.resolve_upload_asset(asset.storage_path)
        ok = extract_frame_thumbnail(
            source_path,
            str(out_path),
            time_sec=time_key,
            width=width,
        )
        if not ok:
            raise HTTPException(status_code=422, detail="Could not extract frame")

    return FileResponse(
        out_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/{asset_id}/waveform")
def get_waveform(
    asset_id: str,
    num_peaks: int = 800,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict:
    """Return audio amplitude peaks for waveform visualisation."""
    asset = session.exec(select(MediaAsset).where(MediaAsset.id == asset_id)).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Media asset not found")
    require_project_owner(session, asset.project_id, current_user)

    absolute_path = storage.resolve_upload_asset(asset.storage_path)
    peaks = extract_waveform_peaks(
        str(absolute_path),
        num_peaks=min(num_peaks, 2000),
        duration_sec=asset.duration_sec,
    )
    return {
        "asset_id": asset_id,
        "num_peaks": len(peaks),
        "duration_sec": asset.duration_sec,
        "peaks": peaks,
    }
