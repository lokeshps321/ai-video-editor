from __future__ import annotations

import json
import mimetypes
import queue
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from sqlmodel import Session, select

from .config import get_settings
from .database import engine
from .ingest_service import download_video_with_ytdlp
from .media_utils import probe_duration_seconds, probe_stream_flags
from .models import Job, JobEvent, MediaAsset
from .render_service import build_ffmpeg_command, ensure_parent_dir, run_ffmpeg
from .schemas import Clip, ExportSettings
from .storage import storage
from .timeline_service import get_timeline_row, load_timeline_state

settings = get_settings()
_QUEUE_POLL_TIMEOUT_SEC = 0.5
_QUEUE_SENTINEL = object()


class RenderJobQueue:
    def __init__(self, max_workers: int) -> None:
        self.max_workers = max(1, max_workers)
        self._queue: queue.Queue[object] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._started = False
        self._stop_requested = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._stop_requested = False
            self._threads = []
            for idx in range(self.max_workers):
                thread = threading.Thread(
                    target=self._worker_loop,
                    name=f"render-worker-{idx + 1}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
            self._started = True

    def stop(self, *, timeout_sec: float = 2.0) -> None:
        with self._lock:
            if not self._started:
                return
            self._stop_requested = True
            threads = list(self._threads)
            for _ in threads:
                self._queue.put(_QUEUE_SENTINEL)
        for thread in threads:
            thread.join(timeout=timeout_sec)
        with self._lock:
            self._threads = []
            self._started = False
            self._stop_requested = False

    def enqueue(self, job_id: str, export_settings: ExportSettings) -> None:
        self.start()
        self._queue.put((job_id, export_settings.model_dump()))

    def size(self) -> int:
        return self._queue.qsize()

    def _worker_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=_QUEUE_POLL_TIMEOUT_SEC)
            except queue.Empty:
                with self._lock:
                    if self._stop_requested:
                        return
                continue
            try:
                if item is _QUEUE_SENTINEL:
                    return
                job_id, payload = item  # type: ignore[misc]
                export_settings = ExportSettings.model_validate(payload)
                process_render_job(job_id, export_settings)
            finally:
                self._queue.task_done()


class IngestJobQueue:
    def __init__(self, max_workers: int) -> None:
        self.max_workers = max(1, max_workers)
        self._queue: queue.Queue[object] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._started = False
        self._stop_requested = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._stop_requested = False
            self._threads = []
            for idx in range(self.max_workers):
                thread = threading.Thread(
                    target=self._worker_loop,
                    name=f"ingest-worker-{idx + 1}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
            self._started = True

    def stop(self, *, timeout_sec: float = 2.0) -> None:
        with self._lock:
            if not self._started:
                return
            self._stop_requested = True
            threads = list(self._threads)
            for _ in threads:
                self._queue.put(_QUEUE_SENTINEL)
        for thread in threads:
            thread.join(timeout=timeout_sec)
        with self._lock:
            self._threads = []
            self._started = False
            self._stop_requested = False

    def enqueue(self, job_id: str, url: str) -> None:
        self.start()
        self._queue.put((job_id, url))

    def size(self) -> int:
        return self._queue.qsize()

    def _worker_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=_QUEUE_POLL_TIMEOUT_SEC)
            except queue.Empty:
                with self._lock:
                    if self._stop_requested:
                        return
                continue
            try:
                if item is _QUEUE_SENTINEL:
                    return
                job_id, url = item  # type: ignore[misc]
                process_ingest_url_job(job_id, str(url))
            finally:
                self._queue.task_done()


_render_queue = RenderJobQueue(settings.max_concurrent_render_jobs)
_ingest_queue = IngestJobQueue(settings.max_concurrent_ingest_jobs)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_job(session: Session, project_id: str, kind: str) -> Job:
    job = Job(project_id=project_id, kind=kind, status="queued", progress=0)
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
    cutoff = _utcnow() - timedelta(seconds=within_seconds) if within_seconds > 0 else None
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
            state = load_timeline_state(timeline_row)

            video_track = next((track for track in state.tracks if track.kind == "video"), None)
            audio_tracks = [track for track in state.tracks if track.kind == "audio"]
            overlay_tracks = [track for track in state.tracks if track.kind == "overlay"]
            if not video_track or not video_track.clips:
                raise RuntimeError("No video clips found. Add at least one video clip before rendering.")

            assets = session.exec(select(MediaAsset).where(MediaAsset.project_id == job.project_id)).all()
            by_id = {asset.id: asset for asset in assets}

            video_clips_sorted = sorted(video_track.clips, key=lambda c: c.timeline_start_sec)
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
                active_audio_tracks = [track for track in audio_tracks if not track.mute]

            audio_inputs: list[tuple[Clip, str]] = []
            audio_flags: list[bool] = []
            for track in active_audio_tracks:
                for clip in sorted(track.clips, key=lambda c: c.timeline_start_sec):
                    asset = by_id.get(clip.asset_id)
                    if not asset:
                        continue
                    source_path = storage.resolve_upload_asset(asset.storage_path)
                    normalized_clip = clip.model_copy(deep=True)
                    normalized_clip.audio.volume = max(0.0, normalized_clip.audio.volume * max(0.0, track.volume))
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
            run_ffmpeg(command)
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
            absolute_path, relative_path = download_video_with_ytdlp(url, job.project_id)

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
            file_name = Path(absolute_path).name
            mime_type = mimetypes.guess_type(file_name)[0] or "video/mp4"
            metadata = {
                "source_url": url,
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


def start_render_workers() -> None:
    _render_queue.start()


def stop_render_workers() -> None:
    _render_queue.stop()


def enqueue_render_job(job_id: str, export_settings: ExportSettings) -> None:
    _render_queue.enqueue(job_id, export_settings)


def start_ingest_workers() -> None:
    _ingest_queue.start()


def stop_ingest_workers() -> None:
    _ingest_queue.stop()


def enqueue_ingest_url_job(job_id: str, url: str) -> None:
    _ingest_queue.enqueue(job_id, url)


def list_job_events(session: Session, job_id: str) -> list[JobEvent]:
    return session.exec(
        select(JobEvent)
        .where(JobEvent.job_id == job_id)
        .order_by(JobEvent.id.asc())
    ).all()
