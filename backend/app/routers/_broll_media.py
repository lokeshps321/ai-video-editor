from __future__ import annotations

import mimetypes
import re
import subprocess
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi import HTTPException
from sqlmodel import Session, select

from ..config import get_settings
from ..media_utils import probe_duration_seconds, probe_stream_flags
from ..models import BrollCandidate, MediaAsset, Project
from ..storage import storage
from ._broll_util import (
    _is_vertical_project,
    _json_dumps,
    _parse_asset_metadata,
    _parse_reason_json,
)

settings = get_settings()


def _resolve_asset_video_path(asset: MediaAsset) -> str:
    return storage.resolve_upload_asset(asset.storage_path)


@lru_cache(maxsize=64)
def _probe_video_dimensions(path: str, mtime_ns: int) -> tuple[int, int]:
    cmd = [
        settings.ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return (0, 0)
    raw = (proc.stdout or "").strip().splitlines()
    if not raw:
        return (0, 0)
    first = raw[0].strip()
    if "x" not in first:
        return (0, 0)
    width_raw, height_raw = first.split("x", 1)
    try:
        width = max(0, int(float(width_raw)))
        height = max(0, int(float(height_raw)))
    except ValueError:
        return (0, 0)
    return (width, height)


@lru_cache(maxsize=32)
def _extract_audio_transients(
    path: str,
    mtime_ns: int,
    sample_rate: int,
) -> tuple[float, ...]:
    try:
        import numpy as np  # type: ignore
    except Exception:
        return ()

    cmd = [
        settings.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True, timeout=120)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ()

    if not proc.stdout:
        return ()
    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if samples.size < sample_rate // 4:
        return ()

    hop = max(96, int(sample_rate * 0.02))
    frame_count = samples.size // hop
    if frame_count < 8:
        return ()

    trimmed = samples[: frame_count * hop]
    matrix = np.abs(trimmed.reshape(frame_count, hop))
    energy = np.sqrt(np.mean(matrix * matrix, axis=1))
    if energy.size < 4:
        return ()

    delta = np.maximum(energy[1:] - energy[:-1], 0.0)
    if delta.size == 0:
        return ()
    baseline = float(np.median(delta))
    spread = float(np.percentile(delta, 88) - baseline)
    threshold = baseline + max(spread * 0.30, 0.003)
    candidate_idx = np.where(delta >= threshold)[0] + 1
    if candidate_idx.size == 0:
        return ()

    min_step = max(1, int(round(0.16 / 0.02)))
    picked: list[int] = []
    for idx in candidate_idx.tolist():
        if not picked:
            picked.append(idx)
            continue
        if idx - picked[-1] >= min_step:
            picked.append(idx)
            continue
        prev_idx = picked[-1]
        if float(delta[idx - 1]) > float(delta[prev_idx - 1]):
            picked[-1] = idx
    if not picked:
        return ()
    times = tuple(round((idx * hop) / float(sample_rate), 3) for idx in picked[:6000])
    return times


def _snap_time_to_transient(
    value: float, transients: tuple[float, ...], window_sec: float
) -> float:
    if not transients:
        return value
    idx = bisect_left(transients, value)
    candidates: list[float] = []
    if idx < len(transients):
        candidates.append(float(transients[idx]))
    if idx > 0:
        candidates.append(float(transients[idx - 1]))
    if not candidates:
        return value
    best = min(candidates, key=lambda item: abs(item - value))
    if abs(best - value) <= window_sec:
        return best
    return value


def _snap_chunks_to_audio_grid(
    chunks: list[dict[str, object]],
    audio_path: str,
    *,
    min_chunk_sec: float,
    max_chunk_sec: float,
) -> list[dict[str, object]]:
    if not settings.broll_audio_reactive_enabled:
        return chunks
    path = Path(audio_path)
    if not path.exists():
        return chunks
    transients = _extract_audio_transients(
        str(path.resolve()),
        path.stat().st_mtime_ns,
        max(4000, settings.broll_audio_reactive_sample_rate),
    )
    if not transients:
        return chunks

    snapped: list[dict[str, object]] = []
    prev_end = 0.0
    window_sec = max(0.05, settings.broll_audio_reactive_window_sec)
    for chunk in chunks:
        start_sec = float(chunk["start_sec"])
        end_sec = float(chunk["end_sec"])
        start_sec = _snap_time_to_transient(start_sec, transients, window_sec)
        end_sec = _snap_time_to_transient(end_sec, transients, window_sec)

        if max_chunk_sec > 0 and end_sec - start_sec > max_chunk_sec:
            end_sec = start_sec + max_chunk_sec
        if min_chunk_sec > 0 and end_sec - start_sec < min_chunk_sec:
            end_sec = start_sec + min_chunk_sec

        start_sec = max(prev_end, start_sec)
        end_sec = max(start_sec + 0.06, end_sec)
        updated = dict(chunk)
        updated["start_sec"] = round(start_sec, 3)
        updated["end_sec"] = round(end_sec, 3)
        snapped.append(updated)
        prev_end = end_sec
    return snapped


def _detect_focus_track(
    path: str, *, max_samples: int = 60
) -> list[dict[str, float]] | None:
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            return None

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = 30.0

        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        samples: list[tuple[float, float]] = []
        previous_gray = None
        frame_idx = 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(max(frame_count, max_samples * 4) // max_samples))
        while len(samples) < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            frame_idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=5, minSize=(36, 36)
            )
            if len(faces) > 0:
                largest = max(faces, key=lambda item: int(item[2]) * int(item[3]))
                focus = float(largest[0] + (largest[2] / 2)) / float(width)
                previous_gray = gray
            else:
                focus = 0.5
                if previous_gray is not None:
                    diff = cv2.absdiff(gray, previous_gray)
                    _, mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
                    moments = cv2.moments(mask)
                    if moments["m00"] > 0:
                        motion_x = float(moments["m10"] / moments["m00"])
                        focus = motion_x / float(width)
            focus = max(0.0, min(1.0, focus))
            time_sec = max(0.0, frame_idx / fps)
            samples.append((round(time_sec, 3), focus))
            previous_gray = gray

        if not samples:
            return None

        smoothed: list[tuple[float, float]] = []
        for idx, (time_sec, focus) in enumerate(samples):
            if idx == 0:
                smoothed.append((time_sec, focus))
                continue
            prev_focus = smoothed[-1][1]
            smooth_focus = (0.68 * prev_focus) + (0.32 * focus)
            smoothed.append((time_sec, smooth_focus))

        keyframes: list[dict[str, float]] = []
        for time_sec, focus in smoothed:
            if not keyframes:
                keyframes.append({"time_sec": 0.0, "x_ratio": round(focus, 4)})
                continue
            prev = keyframes[-1]
            if (
                abs(float(prev["x_ratio"]) - float(focus)) < 0.008
                and (time_sec - float(prev["time_sec"])) < 0.20
            ):
                continue
            keyframes.append(
                {"time_sec": round(time_sec, 3), "x_ratio": round(focus, 4)}
            )

        if not keyframes:
            return None
        duration = max(float(samples[-1][0]), 0.0)
        if duration > 0.05 and keyframes[-1]["time_sec"] < duration:
            keyframes.append(
                {
                    "time_sec": round(duration, 3),
                    "x_ratio": float(keyframes[-1]["x_ratio"]),
                }
            )

        if len(keyframes) > 24:
            stride = max(1, len(keyframes) // 24)
            reduced = [keyframes[0]]
            reduced.extend(
                keyframes[idx] for idx in range(stride, len(keyframes), stride)
            )
            if reduced[-1]["time_sec"] != keyframes[-1]["time_sec"]:
                reduced.append(keyframes[-1])
            keyframes = reduced[:24]
        return keyframes
    finally:
        cap.release()


def _detect_focus_x_ratio(path: str) -> float | None:
    track = _detect_focus_track(path)
    if not track:
        return None
    values = sorted(float(item.get("x_ratio", 0.5)) for item in track)
    if not values:
        return None
    return values[len(values) // 2]


def _analyze_center_visual_risk(
    path: str, *, max_samples: int = 40
) -> tuple[float, float, str]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return (0.5, 0.15, "medium")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return (0.5, 0.15, "medium")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            return (0.5, 0.15, "medium")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(max(frame_count, max_samples * 5) // max_samples))

        brightness_values: list[float] = []
        texture_values: list[float] = []
        frame_idx = 0
        while len(brightness_values) < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            frame_idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crop_w = max(10, int(width * 0.42))
            crop_h = max(10, int(height * 0.34))
            x0 = max(0, min(width - crop_w, (width - crop_w) // 2))
            y0 = max(0, min(height - crop_h, (height - crop_h) // 2))
            roi = gray[y0 : y0 + crop_h, x0 : x0 + crop_w]
            if roi.size == 0:
                continue
            brightness_values.append(float(np.mean(roi)) / 255.0)
            lap = cv2.Laplacian(roi, cv2.CV_64F)
            texture_values.append(float(np.var(lap)) / 3000.0)

        if not brightness_values:
            return (0.5, 0.15, "medium")
        brightness = max(0.0, min(1.0, float(np.mean(brightness_values))))
        texture = max(0.0, min(1.0, float(np.mean(texture_values))))
        if brightness >= 0.64 or texture >= 0.26:
            risk = "high"
        elif brightness >= 0.54 or texture >= 0.17:
            risk = "medium"
        else:
            risk = "low"
        return (round(brightness, 3), round(texture, 3), risk)
    finally:
        cap.release()


def _ensure_asset_focus_metadata(
    session: Session, asset: MediaAsset
) -> dict[str, object]:
    metadata = _parse_asset_metadata(asset)
    path = Path(_resolve_asset_video_path(asset))
    if not path.exists():
        return metadata

    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)
    if width <= 0 or height <= 0:
        width, height = _probe_video_dimensions(
            str(path.resolve()), path.stat().st_mtime_ns
        )
        if width > 0 and height > 0:
            metadata["width"] = width
            metadata["height"] = height

    if (
        settings.broll_auto_reframe_enabled
        and width > 0
        and height > 0
        and width > height
    ):
        if "focus_track" not in metadata:
            focus_track = _detect_focus_track(str(path.resolve()))
            if focus_track:
                metadata["focus_track"] = focus_track
        if "focus_x" not in metadata:
            focus_x = _detect_focus_x_ratio(str(path.resolve()))
            if focus_x is not None:
                metadata["focus_x"] = round(float(focus_x), 4)

    if any(
        key not in metadata
        for key in ("center_brightness", "center_texture", "text_safety_risk")
    ):
        brightness, texture, risk = _analyze_center_visual_risk(str(path.resolve()))
        metadata["center_brightness"] = brightness
        metadata["center_texture"] = texture
        metadata["text_safety_risk"] = risk

    asset.metadata_json = _json_dumps(metadata)
    session.add(asset)
    return metadata


def _build_vertical_crop(
    project: Project, width: int, height: int, focus_x: float | None
) -> dict[str, int] | None:
    if width <= 0 or height <= 0:
        return None
    if not _is_vertical_project(project):
        return None
    if width <= height:
        return None

    target_ratio = float(project.width) / float(project.height)
    crop_width = int(round(height * target_ratio))
    crop_width = max(2, min(crop_width, width))
    crop_height = height
    focus = 0.5 if focus_x is None else max(0.0, min(1.0, focus_x))
    center_x = int(round(focus * width))
    left = max(0, min(width - crop_width, center_x - (crop_width // 2)))
    return {
        "x": int(left),
        "y": 0,
        "width": int(crop_width),
        "height": int(crop_height),
    }


def _build_vertical_crop_keyframes(
    project: Project,
    width: int,
    height: int,
    focus_track: object,
    *,
    clip_duration_sec: float,
) -> list[dict[str, float | int]]:
    if not isinstance(focus_track, list):
        return []
    if not _is_vertical_project(project):
        return []
    if width <= 0 or height <= 0 or width <= height:
        return []
    if clip_duration_sec <= 0:
        return []

    target_ratio = float(project.width) / float(project.height)
    crop_width = int(round(height * target_ratio))
    crop_width = max(2, min(crop_width, width))

    keyframes: list[dict[str, float | int]] = []
    previous_x: int | None = None
    for raw in focus_track:
        if not isinstance(raw, dict):
            continue
        try:
            time_sec = float(raw.get("time_sec", 0.0))
            x_ratio = float(raw.get("x_ratio", 0.5))
        except (TypeError, ValueError):
            continue
        if time_sec < 0:
            continue
        if time_sec > clip_duration_sec:
            break
        center_x = int(round(max(0.0, min(1.0, x_ratio)) * width))
        x = max(0, min(width - crop_width, center_x - (crop_width // 2)))
        if previous_x is not None:
            x = int(round((0.65 * previous_x) + (0.35 * x)))
        previous_x = x
        keyframes.append(
            {
                "time_sec": round(time_sec, 3),
                "x": int(x),
                "y": 0,
            }
        )
    if not keyframes:
        return []

    first = keyframes[0]
    if float(first["time_sec"]) > 0:
        keyframes.insert(0, {"time_sec": 0.0, "x": int(first["x"]), "y": 0})
    last = keyframes[-1]
    if float(last["time_sec"]) < clip_duration_sec:
        keyframes.append(
            {
                "time_sec": round(clip_duration_sec, 3),
                "x": int(last["x"]),
                "y": 0,
            }
        )

    deduped: list[dict[str, float | int]] = []
    for item in keyframes:
        if (
            deduped
            and abs(float(item["time_sec"]) - float(deduped[-1]["time_sec"])) < 0.001
        ):
            deduped[-1] = item
            continue
        if (
            deduped
            and abs(float(item["x"]) - float(deduped[-1]["x"])) < 1
            and (float(item["time_sec"]) - float(deduped[-1]["time_sec"])) < 0.12
        ):
            continue
        deduped.append(item)
    if len(deduped) > 24:
        stride = max(1, len(deduped) // 24)
        reduced = [deduped[0]]
        reduced.extend(deduped[idx] for idx in range(stride, len(deduped), stride))
        if reduced[-1]["time_sec"] != deduped[-1]["time_sec"]:
            reduced.append(deduped[-1])
        deduped = reduced[:24]
    return deduped


def _text_safety_preset_from_metadata(
    metadata: dict[str, object],
) -> tuple[str | None, float]:
    risk = str(metadata.get("text_safety_risk") or "").strip().lower()
    if risk == "high":
        return ("text_safe_soft", 0.76)
    if risk == "medium":
        return ("text_safe_mild", 0.82)
    return (None, 1.0)


def _safe_filename_from_url(url: str, fallback_stem: str = "broll") -> str:
    parsed = urlparse(url)
    stem = Path(parsed.path).stem or fallback_stem
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".webm", ".mkv"}:
        suffix = ".mp4"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    if not cleaned:
        cleaned = fallback_stem
    return f"{cleaned}-{uuid4().hex[:8]}{suffix}"


def _download_external_video(project_id: str, source_url: str) -> tuple[str, str, str]:
    parsed = urlparse(source_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=422, detail="B-roll source URL must be http(s)")

    project_dir = storage.upload_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename_from_url(source_url)
    destination = project_dir / filename

    max_bytes = max(5, settings.broll_external_download_max_mb) * 1024 * 1024
    timeout = httpx.Timeout(max(2.0, settings.broll_external_timeout_sec))

    total = 0
    try:
        with httpx.stream(
            "GET", source_url, timeout=timeout, follow_redirects=True
        ) as response:
            response.raise_for_status()
            content_type = (
                (response.headers.get("content-type") or "video/mp4")
                .split(";")[0]
                .strip()
            )
            with destination.open("wb") as stream:
                for chunk in response.iter_bytes(1024 * 256):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"External B-roll file too large (> {settings.broll_external_download_max_mb} MB)",
                        )
                    stream.write(chunk)
    except HTTPException:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=502, detail=f"Failed to download external B-roll: {exc}"
        ) from exc

    relative = str(destination.resolve().relative_to(storage.upload_root))
    mime_type = mimetypes.guess_type(destination.name)[0] or "video/mp4"
    return (str(destination.resolve()), relative, mime_type)


def _find_existing_asset_for_source_url(
    session: Session,
    *,
    project_id: str,
    source_url: str,
) -> MediaAsset | None:
    normalized_source = source_url.strip()
    if not normalized_source:
        return None

    assets = list(
        session.exec(
            select(MediaAsset)
            .where(
                MediaAsset.project_id == project_id, MediaAsset.media_type == "video"
            )
            .order_by(MediaAsset.created_at.desc())
        ).all()
    )
    for asset in assets:
        metadata = _parse_asset_metadata(asset)
        if str(metadata.get("source_url") or "").strip() != normalized_source:
            continue
        if Path(_resolve_asset_video_path(asset)).exists():
            return asset
    return None


def _materialize_candidate_asset(
    session: Session, project_id: str, candidate: BrollCandidate
) -> MediaAsset:
    if candidate.asset_id:
        existing = session.exec(
            select(MediaAsset).where(
                MediaAsset.id == candidate.asset_id, MediaAsset.project_id == project_id
            )
        ).first()
        if existing:
            return existing
    if not candidate.source_url:
        raise HTTPException(
            status_code=422, detail="Selected candidate has no importable source URL"
        )

    existing_for_source = _find_existing_asset_for_source_url(
        session,
        project_id=project_id,
        source_url=candidate.source_url,
    )
    if existing_for_source is not None:
        candidate.asset_id = existing_for_source.id
        session.add(candidate)
        return existing_for_source

    # AI-generated clips are already on disk (local file path, not a URL)
    is_local_generated = (
        candidate.source_type == "generated_image_video"
        and not candidate.source_url.startswith("http")
        and Path(candidate.source_url).is_file()
    )

    if is_local_generated:
        # Copy the generated video from tmp to project uploads
        src_path = Path(candidate.source_url)
        upload_dir = Path(settings.upload_dir) / project_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest_name = f"ai_broll_{src_path.stem}.mp4"
        dest_path = upload_dir / dest_name
        import shutil

        shutil.copy2(str(src_path), str(dest_path))
        absolute_path = str(dest_path.resolve())
        relative_path = f"{project_id}/{dest_name}"
        guessed_mime = "video/mp4"
        # Clean up temp file
        try:
            src_path.unlink(missing_ok=True)
        except OSError:
            pass
    else:
        absolute_path, relative_path, guessed_mime = _download_external_video(
            project_id, candidate.source_url
        )

    stream_flags = probe_stream_flags(absolute_path)
    if not stream_flags.get("has_video", False):
        Path(absolute_path).unlink(missing_ok=True)
        raise HTTPException(
            status_code=422, detail="Selected B-roll source has no video stream"
        )

    reason_payload = _parse_reason_json(candidate)
    path_obj = Path(absolute_path)
    probe_width, probe_height = _probe_video_dimensions(
        str(path_obj.resolve()), path_obj.stat().st_mtime_ns
    )
    width = int(reason_payload.get("width") or probe_width or 0)
    height = int(reason_payload.get("height") or probe_height or 0)

    # Skip heavy OpenCV analysis for AI-generated clips (no real faces to track)
    focus_track = None
    focus_x = None
    if not is_local_generated:
        focus_track = (
            _detect_focus_track(str(path_obj.resolve()))
            if settings.broll_auto_reframe_enabled and width > height
            else None
        )
        if focus_track:
            ratios = sorted(
                float(item.get("x_ratio", 0.5))
                for item in focus_track
                if isinstance(item, dict)
            )
            if ratios:
                focus_x = ratios[len(ratios) // 2]
        if focus_x is None and settings.broll_auto_reframe_enabled and width > height:
            focus_x = _detect_focus_x_ratio(str(path_obj.resolve()))

    brightness, texture, risk = _analyze_center_visual_risk(str(path_obj.resolve()))

    duration_sec = probe_duration_seconds(absolute_path)
    source_filename = candidate.source_label or Path(relative_path).name
    metadata = {
        "source_type": candidate.source_type,
        "source_url": candidate.source_url,
        "width": width,
        "height": height,
        "center_brightness": brightness,
        "center_texture": texture,
        "text_safety_risk": risk,
        **stream_flags,
    }
    if focus_x is not None:
        metadata["focus_x"] = round(float(focus_x), 4)
    if focus_track:
        metadata["focus_track"] = focus_track
    if is_local_generated:
        metadata["ai_generated"] = True
    asset = MediaAsset(
        project_id=project_id,
        media_type="video",
        filename=source_filename[:180],
        storage_path=relative_path,
        mime_type=guessed_mime,
        duration_sec=duration_sec,
        metadata_json=_json_dumps(metadata),
    )
    session.add(asset)
    session.flush()
    candidate.asset_id = asset.id
    session.add(candidate)
    return asset
