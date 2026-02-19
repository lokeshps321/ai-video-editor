from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .config import get_settings

settings = get_settings()


def probe_duration_seconds(path: str) -> float | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    cmd = [
        settings.ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(file_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(proc.stdout)
        value = payload.get("format", {}).get("duration")
        return float(value) if value is not None else None
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError):
        return None


def probe_stream_flags(path: str) -> dict[str, bool]:
    file_path = Path(path)
    if not file_path.exists():
        return {"has_video": False, "has_audio": False}
    cmd = [
        settings.ffprobe_bin,
        "-v",
        "error",
        "-show_streams",
        "-of",
        "json",
        str(file_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(proc.stdout)
        streams = payload.get("streams", [])
        has_video = any(str(stream.get("codec_type")) == "video" for stream in streams)
        has_audio = any(str(stream.get("codec_type")) == "audio" for stream in streams)
        return {"has_video": has_video, "has_audio": has_audio}
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return {"has_video": False, "has_audio": False}


def infer_media_type(mime_type: str, filename: str) -> str:
    lower = (mime_type or "").lower()
    if lower.startswith("video/"):
        return "video"
    if lower.startswith("audio/"):
        return "audio"
    if lower.startswith("image/"):
        return "image"

    suffix = Path(filename).suffix.lower()
    if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return "video"
    if suffix in {".mp3", ".wav", ".aac", ".m4a"}:
        return "audio"
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return "image"
    return "unknown"


def detect_silence_ranges(
    path: str,
    *,
    noise_db: float = -35.0,
    min_silence_sec: float = 0.35,
    max_duration_sec: float | None = None,
) -> list[tuple[float, float]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    cmd = [
        settings.ffmpeg_bin,
        "-hide_banner",
        "-i",
        str(file_path),
    ]
    if max_duration_sec is not None and max_duration_sec > 0:
        cmd.extend(["-t", f"{float(max_duration_sec):.3f}"])
    cmd.extend([
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence_sec}",
        "-f",
        "null",
        "-",
    ])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return []

    stderr = proc.stderr or ""
    start_pattern = re.compile(r"silence_start:\s*([0-9]*\.?[0-9]+)")
    end_pattern = re.compile(r"silence_end:\s*([0-9]*\.?[0-9]+)")

    starts = [float(match.group(1)) for match in start_pattern.finditer(stderr)]
    ends = [float(match.group(1)) for match in end_pattern.finditer(stderr)]

    ranges: list[tuple[float, float]] = []
    end_idx = 0
    for start_sec in starts:
        while end_idx < len(ends) and ends[end_idx] <= start_sec:
            end_idx += 1
        if end_idx >= len(ends):
            break
        end_sec = ends[end_idx]
        end_idx += 1
        if end_sec > start_sec:
            ranges.append((round(start_sec, 3), round(end_sec, 3)))
    return ranges


def extract_waveform_peaks(
    path: str,
    *,
    num_peaks: int = 800,
    duration_sec: float | None = None,
) -> list[float]:
    """Extract audio amplitude peaks for waveform visualisation.

    Uses FFmpeg to decode the audio track into raw f32le samples, then computes
    the peak (absolute max) for each chunk.  Returns *num_peaks* float values
    in [0.0, 1.0].
    """
    file_path = Path(path)
    if not file_path.exists():
        return []

    if duration_sec is None:
        duration_sec = probe_duration_seconds(path) or 1.0
    duration_sec = max(duration_sec, 0.1)

    # Calculate samples per peak to get the desired number of peaks
    sample_rate = 8000  # low rate is fine for visualisation
    total_samples = int(sample_rate * duration_sec)
    chunk_size = max(total_samples // num_peaks, 1)

    cmd = [
        settings.ffmpeg_bin,
        "-hide_banner",
        "-i", str(file_path),
        "-ac", "1",              # mono
        "-ar", str(sample_rate), # low sample rate
        "-f", "f32le",           # raw 32-bit floats
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, check=True, timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return []

    import struct

    raw = proc.stdout
    num_samples = len(raw) // 4
    if num_samples == 0:
        return []

    # Recompute chunk_size from actual sample count
    chunk_size = max(num_samples // num_peaks, 1)
    peaks: list[float] = []
    offset = 0
    while offset < num_samples:
        end = min(offset + chunk_size, num_samples)
        chunk_bytes = raw[offset * 4 : end * 4]
        floats = struct.unpack(f"<{end - offset}f", chunk_bytes)
        peak = max(abs(v) for v in floats) if floats else 0.0
        peaks.append(min(peak, 1.0))
        offset = end

    # Normalise to [0, 1]
    max_peak = max(peaks) if peaks else 1.0
    if max_peak > 0:
        peaks = [p / max_peak for p in peaks]
    return peaks
