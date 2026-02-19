from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session, select

from ..database import get_session
from ..media_utils import infer_media_type, probe_duration_seconds, probe_stream_flags, extract_waveform_peaks
from ..models import MediaAsset, Project
from ..schemas import MediaUploadResponse
from ..storage import storage

router = APIRouter(prefix="/api/v1/media", tags=["media"])


@router.post("/upload", response_model=MediaUploadResponse)
async def upload_media(
    project_id: str = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> MediaUploadResponse:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid file")

    absolute_path, relative_path = await storage.save_upload(file, project_id)
    media_type = infer_media_type(file.content_type or "", file.filename)
    duration_sec = probe_duration_seconds(absolute_path) if media_type in {"video", "audio"} else None
    stream_flags = probe_stream_flags(absolute_path) if media_type in {"video", "audio"} else {"has_video": False, "has_audio": False}
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

    return MediaUploadResponse(
        id=asset.id,
        project_id=asset.project_id,
        media_type=asset.media_type,
        filename=asset.filename,
        storage_path=storage.to_public_upload_path(relative_path),
        duration_sec=asset.duration_sec,
    )


@router.get("", response_model=list[MediaUploadResponse])
def list_media(project_id: str, session: Session = Depends(get_session)) -> list[MediaUploadResponse]:
    items = session.exec(
        select(MediaAsset).where(MediaAsset.project_id == project_id).order_by(MediaAsset.created_at.desc())
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


@router.get("/{asset_id}/waveform")
def get_waveform(
    asset_id: str,
    num_peaks: int = 800,
    session: Session = Depends(get_session),
) -> dict:
    """Return audio amplitude peaks for waveform visualisation."""
    asset = session.exec(select(MediaAsset).where(MediaAsset.id == asset_id)).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Media asset not found")

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

