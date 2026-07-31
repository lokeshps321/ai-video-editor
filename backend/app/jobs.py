from __future__ import annotations

import json
import logging
import mimetypes
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from sqlmodel import Session, select

from .config import get_settings
from .database import engine
from .ingest_service import download_video_with_ytdlp, source_title_to_filename
from .media_utils import probe_duration_seconds, probe_stream_flags
from .models import Job, JobEvent, MediaAsset, TimelineVersion
from .render_service import build_ffmpeg_command, ensure_parent_dir, run_ffmpeg
from .schemas import Clip, ExportSettings, TimelineState
from .storage import storage
from .timeline_service import get_timeline_row, load_timeline_state
from .transcription_service import precompute_vocal_isolation

settings = get_settings()
_render_log = logging.getLogger(__name__)
_LOCAL_RENDER_JOB_SLOTS = threading.BoundedSemaphore(
    value=max(1, settings.max_concurrent_render_jobs)
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_job(
    session: Session,
    project_id: str,
    kind: str,
    *,
    timeline_version: int | None = None,
) -> Job:
    job = Job(
        project_id=project_id,
        kind=kind,
        timeline_version=timeline_version,
        status="queued",
        progress=0,
    )
    session.add(job)
    session.add(
        JobEvent(
            job_id=job.id,
            project_id=project_id,
            stage="queued",
            status="queued",
            progress=0,
            message="Job queued",
        )
    )
    session.commit()
    session.refresh(job)
    return job


def find_recent_active_job(
    session: Session,
    project_id: str,
    kind: str,
    *,
    within_seconds: int = 120,
) -> Job | None:
    cutoff = (
        _utcnow() - timedelta(seconds=within_seconds) if within_seconds > 0 else None
    )
    jobs = session.exec(
        select(Job)
        .where(
            Job.project_id == project_id,
            Job.kind == kind,
            Job.status.in_(["queued", "running"]),
        )
        .order_by(Job.updated_at.desc())
    ).all()
    for job in jobs:
        updated = job.updated_at
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        if cutoff is None or updated >= cutoff:
            return job
    return None


def fail_orphaned_active_jobs() -> int:
    with Session(engine) as session:
        active = session.exec(
            select(Job).where(Job.status.in_(["queued", "running"]))
        ).all()
        count = 0
        for job in active:
            _set_job_status(
                session,
                job,
                status="failed",
                progress=100,
                stage="failed",
                message="Job interrupted by server restart. Re-run the action.",
                error="job_interrupted_by_restart",
            )
            count += 1
        return count


def _set_job_status(
    session: Session,
    job: Job,
    *,
    status: str,
    progress: int,
    stage: str,
    message: str | None = None,
    error: str | None = None,
    output_path: str | None = None,
) -> None:
    job.status = status
    job.progress = progress
    job.updated_at = _utcnow()
    if error is not None:
        job.error = error
    if output_path is not None:
        job.output_path = output_path
    session.add(job)
    session.add(
        JobEvent(
            job_id=job.id,
            project_id=job.project_id,
            stage=stage,
            status=status,
            progress=progress,
            message=message,
        )
    )
    session.commit()


def get_latest_job_event(session: Session, job_id: str) -> JobEvent | None:
    return session.exec(
        select(JobEvent).where(JobEvent.job_id == job_id).order_by(JobEvent.id.desc())
    ).first()


def set_job_status(
    session: Session,
    job: Job,
    *,
    status: str,
    progress: int,
    stage: str,
    message: str | None = None,
    error: str | None = None,
    output_path: str | None = None,
) -> None:
    _set_job_status(
        session,
        job,
        status=status,
        progress=progress,
        stage=stage,
        message=message,
        error=error,
        output_path=output_path,
    )


def process_render_job(job_id: str, export_settings: ExportSettings) -> None:
    with Session(engine) as session:
        job = session.exec(select(Job).where(Job.id == job_id)).first()
        if not job:
            return

        try:
            _set_job_status(
                session,
                job,
                status="running",
                progress=5,
                stage="running",
                message="Preparing timeline and media inputs",
            )
            timeline_row = get_timeline_row(session, job.project_id)
            if job.timeline_version is None:
                state = load_timeline_state(timeline_row)
            else:
                timeline_snapshot = session.exec(
                    select(TimelineVersion).where(
                        TimelineVersion.project_id == job.project_id,
                        TimelineVersion.version == job.timeline_version,
                    )
                ).first()
                if timeline_snapshot is None:
                    raise RuntimeError(
                        "The timeline revision queued for this render no longer exists. "
                        "Queue a new render."
                    )
                state = TimelineState.model_validate_json(timeline_snapshot.state_json)

            video_track = next(
                (track for track in state.tracks if track.kind == "video"), None
            )
            audio_tracks = [track for track in state.tracks if track.kind == "audio"]
            overlay_tracks = [
                track for track in state.tracks if track.kind == "overlay"
            ]
            if not video_track or not video_track.clips:
                raise RuntimeError(
                    "No video clips found. Add at least one video clip before rendering."
                )

            assets = session.exec(
                select(MediaAsset).where(MediaAsset.project_id == job.project_id)
            ).all()
            by_id = {asset.id: asset for asset in assets}

            video_clips_sorted = sorted(
                video_track.clips, key=lambda c: c.timeline_start_sec
            )
            video_inputs: list[tuple[Clip, str]] = []
            video_audio_flags: list[bool] = []
            for clip in video_clips_sorted:
                asset = by_id.get(clip.asset_id)
                if not asset:
                    raise RuntimeError(f"Missing media asset: {clip.asset_id}")
                source_path = storage.resolve_upload_asset(asset.storage_path)
                video_inputs.append((clip, source_path))
                video_audio_flags.append(_asset_has_audio(asset, source_path))

            overlay_inputs: list[tuple[Clip, str]] = []
            overlay_video_flags: list[bool] = []
            for track in overlay_tracks:
                for clip in sorted(track.clips, key=lambda c: c.timeline_start_sec):
                    asset = by_id.get(clip.asset_id)
                    if not asset:
                        continue
                    source_path = storage.resolve_upload_asset(asset.storage_path)
                    has_video = _asset_has_video(asset, source_path)
                    if not has_video:
                        continue
                    overlay_inputs.append((clip, source_path))
                    overlay_video_flags.append(has_video)

            active_audio_tracks = [track for track in audio_tracks if track.solo]
            if not active_audio_tracks:
                active_audio_tracks = [
                    track for track in audio_tracks if not track.mute
                ]

            audio_inputs: list[tuple[Clip, str]] = []
            audio_flags: list[bool] = []
            for track in active_audio_tracks:
                for clip in sorted(track.clips, key=lambda c: c.timeline_start_sec):
                    asset = by_id.get(clip.asset_id)
                    if not asset:
                        continue
                    source_path = storage.resolve_upload_asset(asset.storage_path)
                    normalized_clip = clip.model_copy(deep=True)
                    normalized_clip.audio.volume = max(
                        0.0, normalized_clip.audio.volume * max(0.0, track.volume)
                    )
                    if track.mute:
                        normalized_clip.audio.mute = True
                    audio_inputs.append((normalized_clip, source_path))
                    audio_flags.append(_asset_has_audio(asset, source_path))

            ext = export_settings.format
            output_path = storage.output_path(job.project_id, ext)
            ensure_parent_dir(output_path)
            _set_job_status(
                session,
                job,
                status="running",
                progress=20,
                stage="build",
                message="Building FFmpeg command",
            )
            _render_log.info(
                "RENDER_CLIPS project=%s clips=[%s]",
                job.project_id,
                ", ".join(
                    f"{c.start_sec:.3f}-{c.end_sec:.3f}(tl={c.timeline_start_sec:.3f})"
                    for c, _ in video_inputs
                ),
            )
            command = build_ffmpeg_command(
                timeline=state,
                clip_inputs=video_inputs,
                clip_has_audio_flags=video_audio_flags,
                bg_audio_inputs=audio_inputs,
                bg_has_audio_flags=audio_flags,
                output_path=output_path,
                export_settings=export_settings,
                overlay_inputs=overlay_inputs,
                overlay_has_video_flags=overlay_video_flags,
            )

            _set_job_status(
                session,
                job,
                status="running",
                progress=35,
                stage="render",
                message="Rendering video",
            )
            progress_state = {
                "last_progress": 35,
                "last_update_monotonic": time.monotonic(),
            }

            def handle_render_progress(fraction: float) -> None:
                bounded_fraction = max(0.0, min(1.0, fraction))
                next_progress = max(35, min(99, 35 + int(round(bounded_fraction * 64))))
                now = time.monotonic()
                if next_progress <= progress_state["last_progress"]:
                    return
                if (
                    next_progress - progress_state["last_progress"] < 2
                    and now - progress_state["last_update_monotonic"] < 0.75
                ):
                    return
                progress_state["last_progress"] = next_progress
                progress_state["last_update_monotonic"] = now
                _set_job_status(
                    session,
                    job,
                    status="running",
                    progress=next_progress,
                    stage="render",
                    message=f"Rendering video ({next_progress}%)",
                )

            run_ffmpeg(
                command,
                duration_sec=max(float(state.duration_sec or 0.0), 0.0),
                progress_callback=handle_render_progress,
            )
            _set_job_status(
                session,
                job,
                status="completed",
                progress=100,
                stage="complete",
                message="Render completed",
                output_path=storage.to_public_render_path(output_path),
            )
        except Exception as exc:  # noqa: BLE001
            _set_job_status(
                session,
                job,
                status="failed",
                progress=100,
                stage="failed",
                message=str(exc),
                error=str(exc),
            )


def process_ingest_url_job(job_id: str, url: str) -> None:
    with Session(engine) as session:
        job = session.exec(select(Job).where(Job.id == job_id)).first()
        if not job:
            return

        try:
            _set_job_status(
                session,
                job,
                status="running",
                progress=5,
                stage="running",
                message="Preparing URL ingestion",
            )
            _set_job_status(
                session,
                job,
                status="running",
                progress=20,
                stage="download",
                message="Downloading source video",
            )
            download_result = download_video_with_ytdlp(url, job.project_id)
            # Accept the older two-item return shape too. It keeps existing
            # deployment extensions/mocks working while URL ingestion rolls
            # out the title-aware result.
            absolute_path, relative_path = download_result[:2]
            source_title = (
                str(download_result[2]).strip()
                if len(download_result) > 2 and download_result[2]
                else None
            )

            _set_job_status(
                session,
                job,
                status="running",
                progress=70,
                stage="probe",
                message="Probing downloaded media",
            )
            stream_flags = probe_stream_flags(absolute_path)
            if not stream_flags.get("has_video", False):
                raise RuntimeError("Downloaded media has no video stream")
            duration_sec = probe_duration_seconds(absolute_path)

            _set_job_status(
                session,
                job,
                status="running",
                progress=85,
                stage="register",
                message="Registering media in project",
            )
            file_name = source_title_to_filename(source_title, absolute_path)
            mime_type = mimetypes.guess_type(absolute_path)[0] or "video/mp4"
            metadata = {
                "source_url": url,
                "source_title": source_title,
                **stream_flags,
            }
            asset = MediaAsset(
                project_id=job.project_id,
                media_type="video",
                filename=file_name,
                storage_path=relative_path,
                mime_type=mime_type,
                duration_sec=duration_sec,
                metadata_json=json.dumps(metadata),
            )
            session.add(asset)
            session.commit()
            session.refresh(asset)

            # Trigger background vocal isolation if applicable
            if should_precompute_vocal_isolation(asset):
                vocal_job = create_job(session, job.project_id, "vocal_isolation")
                enqueue_vocal_isolation_job(vocal_job.id, asset.id)

            _set_job_status(
                session,
                job,
                status="completed",
                progress=100,
                stage="complete",
                message=f"Ingested {file_name}",
                output_path=storage.to_public_upload_path(relative_path),
            )
        except Exception as exc:  # noqa: BLE001
            _set_job_status(
                session,
                job,
                status="failed",
                progress=100,
                stage="failed",
                message=str(exc),
                error=str(exc),
            )


def _asset_has_audio(asset: MediaAsset, source_path: str) -> bool:
    try:
        payload = json.loads(asset.metadata_json or "{}")
        value = payload.get("has_audio")
        if isinstance(value, bool):
            return value
    except json.JSONDecodeError:
        pass
    return probe_stream_flags(source_path).get("has_audio", False)


def _asset_has_video(asset: MediaAsset, source_path: str) -> bool:
    try:
        payload = json.loads(asset.metadata_json or "{}")
        value = payload.get("has_video")
        if isinstance(value, bool):
            return value
    except json.JSONDecodeError:
        pass
    return probe_stream_flags(source_path).get("has_video", False)


def _use_rq_workers() -> bool:
    """Check if we should use RQ workers instead of local threads."""
    return os.getenv("USE_RQ_WORKERS", "false").lower() in {"1", "true", "yes", "on"}


def start_render_workers() -> None:
    """Start render workers (no-op when using RQ)."""
    pass


def stop_render_workers() -> None:
    """Stop render workers (no-op when using RQ)."""
    pass


def _run_render_job_with_local_limit(
    job_id: str, export_settings: ExportSettings
) -> None:
    _LOCAL_RENDER_JOB_SLOTS.acquire()
    try:
        process_render_job(job_id, export_settings)
    finally:
        _LOCAL_RENDER_JOB_SLOTS.release()


def enqueue_render_job(job_id: str, export_settings: ExportSettings) -> None:
    """Enqueue a render job (uses RQ if enabled, otherwise processes immediately)."""
    if _use_rq_workers():
        from .queue import get_render_queue

        queue = get_render_queue()
        queue.enqueue(
            execute_render_job,
            job_id=job_id,
            export_settings=export_settings.model_dump(),
            job_timeout=3600,  # 1 hour timeout
            failure_ttl=86400,  # Keep failed jobs for 24 hours
        )
    else:
        # Fallback to in-process background execution (development without Redis).
        # Bound concurrent ffmpeg work to reduce local OOM/SIGKILL risk.
        threading.Thread(
            target=_run_render_job_with_local_limit,
            args=(job_id, export_settings),
            name=f"render-job-{job_id[:8]}",
            daemon=True,
        ).start()


def start_ingest_workers() -> None:
    """Start ingest workers (no-op when using RQ)."""
    pass


def stop_ingest_workers() -> None:
    """Stop ingest workers (no-op when using RQ)."""
    pass


def enqueue_ingest_url_job(job_id: str, url: str) -> None:
    """Enqueue an ingest job (uses RQ if enabled, otherwise processes immediately)."""
    if _use_rq_workers():
        from .queue import get_ingest_queue

        queue = get_ingest_queue()
        queue.enqueue(
            execute_ingest_url_job,
            job_id=job_id,
            url=url,
            job_timeout=1800,  # 30 minute timeout
            failure_ttl=86400,  # Keep failed jobs for 24 hours
        )
    else:
        # Fallback to in-process background execution (development without Redis).
        threading.Thread(
            target=process_ingest_url_job,
            args=(job_id, url),
            name=f"ingest-job-{job_id[:8]}",
            daemon=True,
        ).start()


def execute_render_job(job_id: str, export_settings: dict) -> str:
    """Execute render job (called by RQ worker)."""
    settings_model = ExportSettings.model_validate(export_settings)
    process_render_job(job_id, settings_model)
    return job_id


def execute_ingest_url_job(job_id: str, url: str) -> str:
    """Execute ingest job (called by RQ worker)."""
    process_ingest_url_job(job_id, url)
    return job_id


def list_job_events(session: Session, job_id: str) -> list[JobEvent]:
    return session.exec(
        select(JobEvent).where(JobEvent.job_id == job_id).order_by(JobEvent.id.asc())
    ).all()


# ---------------------------------------------------------------------------
# Vocal Isolation Pre-compute Job
# ---------------------------------------------------------------------------


def process_vocal_isolation_job(job_id: str, asset_id: str) -> None:
    """Process a vocal isolation job for a media asset.

    This extracts the vocal stem from the uploaded video/audio and stores it
    alongside the original file for fast transcription later.
    """
    with Session(engine) as session:
        job = session.exec(select(Job).where(Job.id == job_id)).first()
        if not job:
            return

        asset = session.exec(
            select(MediaAsset).where(MediaAsset.id == asset_id)
        ).first()
        if not asset:
            _set_job_status(
                session,
                job,
                status="failed",
                progress=100,
                stage="failed",
                message="Media asset not found",
                error="asset_not_found",
            )
            return

        try:
            _set_job_status(
                session,
                job,
                status="running",
                progress=5,
                stage="running",
                message="Starting vocal isolation",
            )

            # Get absolute path to the source file
            source_path = storage.resolve_upload_asset(asset.storage_path)

            # Determine output directory (same as source file's directory)
            source_dir = str(Path(source_path).parent)

            _set_job_status(
                session,
                job,
                status="running",
                progress=10,
                stage="isolating",
                message="Extracting vocals from audio",
            )

            # Run vocal isolation
            vocal_stem_filename = precompute_vocal_isolation(
                source_path,
                source_dir,
            )

            if vocal_stem_filename:
                # Update asset metadata with vocal stem path
                metadata = json.loads(asset.metadata_json or "{}")
                metadata["vocal_stem_filename"] = vocal_stem_filename
                asset.metadata_json = json.dumps(metadata)
                session.add(asset)
                session.commit()

                _set_job_status(
                    session,
                    job,
                    status="completed",
                    progress=100,
                    stage="complete",
                    message=f"Vocal isolation complete: {vocal_stem_filename}",
                    output_path=vocal_stem_filename,
                )
            else:
                # Isolation failed or was skipped - not a hard error
                _set_job_status(
                    session,
                    job,
                    status="completed",
                    progress=100,
                    stage="complete",
                    message="Vocal isolation skipped (disabled or not applicable)",
                )

        except Exception as exc:  # noqa: BLE001
            _set_job_status(
                session,
                job,
                status="failed",
                progress=100,
                stage="failed",
                message=str(exc),
                error=str(exc),
            )


def enqueue_vocal_isolation_job(job_id: str, asset_id: str) -> None:
    """Enqueue a vocal isolation job (uses RQ if enabled, otherwise processes in thread)."""
    if _use_rq_workers():
        from .queue import get_vocal_isolation_queue

        queue = get_vocal_isolation_queue()
        queue.enqueue(
            execute_vocal_isolation_job,
            job_id=job_id,
            asset_id=asset_id,
            job_timeout=1800,  # 30 minute timeout for isolation
            failure_ttl=86400,  # Keep failed jobs for 24 hours
        )
    else:
        # Fallback to in-process background execution (development without Redis)
        threading.Thread(
            target=process_vocal_isolation_job,
            args=(job_id, asset_id),
            name=f"vocal-isolation-{job_id[:8]}",
            daemon=True,
        ).start()


def execute_vocal_isolation_job(job_id: str, asset_id: str) -> str:
    """Execute vocal isolation job (called by RQ worker)."""
    process_vocal_isolation_job(job_id, asset_id)
    return job_id


def should_precompute_vocal_isolation(asset: MediaAsset) -> bool:
    """Determine if vocal isolation should be pre-computed for an asset.

    Returns True if:
    - The asset is video or audio type
    - Vocal isolation is enabled
    - Pre-compute is enabled
    """
    # Only process video/audio
    if asset.media_type not in {"video", "audio"}:
        return False

    # Check if vocal isolation is globally enabled
    vocal_enabled = os.getenv("TRANSCRIBE_VOCAL_ISOLATION_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not vocal_enabled:
        return False

    # Check if pre-compute is enabled (default: true when vocal isolation is enabled)
    precompute_enabled = os.getenv(
        "TRANSCRIBE_VOCAL_ISOLATION_PRECOMPUTE", "true"
    ).lower() in {"1", "true", "yes", "on"}
    if not precompute_enabled:
        return False

    return True
