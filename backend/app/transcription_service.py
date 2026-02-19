from __future__ import annotations

import os
import re
import sys
import ctypes
import subprocess
from collections import Counter
from math import isfinite
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from .config import get_settings
from .media_utils import detect_silence_ranges


settings = get_settings()

DEFAULT_MUSIC_RETRY_PROMPT = (
    "Transcribe speech and sung lyrics verbatim in the original language. Preserve repeated chorus lines and ad-libs. Do not paraphrase. Do not translate."
)

# ---------------------------------------------------------------------------
# Filler words (used by vibe auto-cut and hallucination heuristic)
# ---------------------------------------------------------------------------
FILLER_WORDS: set[str] = {
    "um", "uh", "uhm", "umm", "hmm", "hm", "ah", "er", "eh",
    "like", "basically", "literally", "actually", "right",
    "you know", "i mean", "sort of", "kind of", "so yeah",
}


@dataclass(frozen=True)
class TranscriptWordPayload:
    id: str
    text: str
    start_sec: float
    end_sec: float
    confidence: float | None = None


@dataclass(frozen=True)
class TranscriptPayload:
    source: str
    language: str | None
    text: str
    words: list[TranscriptWordPayload]
    is_mock: bool


def _clean_word(value: str) -> str:
    return " ".join(value.strip().split())


def _ascii_latin_ratio(value: str) -> float:
    alpha = [char for char in value if char.isalpha()]
    if not alpha:
        return 1.0
    latin = [char for char in alpha if ("A" <= char <= "Z") or ("a" <= char <= "z")]
    return len(latin) / len(alpha)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)
    return max(minimum, value)


def _clamp_time(value: float, duration_sec: float) -> float:
    return max(0.0, min(value, duration_sec))


def _normalize_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed):
        return None
    return max(0.0, min(parsed, 1.0))


def _normalize_words(words: list[TranscriptWordPayload], duration_sec: float) -> list[TranscriptWordPayload]:
    min_confidence = _env_float("TRANSCRIBE_WORD_MIN_CONFIDENCE", 0.15, 0.0)
    normalized: list[TranscriptWordPayload] = []
    for item in sorted(words, key=lambda entry: entry.start_sec):
        start_sec = round(_clamp_time(float(item.start_sec), duration_sec), 3)
        end_sec = round(_clamp_time(float(item.end_sec), duration_sec), 3)
        if end_sec <= start_sec:
            end_sec = round(min(duration_sec, start_sec + 0.05), 3)
        text = _clean_word(item.text)
        if not text:
            continue
        confidence = _normalize_confidence(item.confidence)
        # Filter out words with very low confidence (likely hallucinations)
        if confidence is not None and confidence < min_confidence:
            continue
        normalized.append(
            TranscriptWordPayload(
                id=item.id,
                text=text,
                start_sec=start_sec,
                end_sec=end_sec,
                confidence=confidence,
            )
        )
    return normalized


def _detect_hallucinations(words: list[TranscriptWordPayload]) -> list[TranscriptWordPayload]:
    """Remove repeated phrase loops — a common Whisper failure mode.

    Detects when the same short phrase is repeated 3+ times consecutively
    and collapses it to a single occurrence.
    """
    if len(words) < 6:
        return words

    cleaned: list[TranscriptWordPayload] = []
    i = 0
    while i < len(words):
        # Try phrase lengths 1-4 words
        found_repeat = False
        for phrase_len in range(1, min(5, (len(words) - i) // 2 + 1)):
            phrase_texts = tuple(w.text.lower().strip(".,!?") for w in words[i : i + phrase_len])
            repeat_count = 1
            j = i + phrase_len
            while j + phrase_len <= len(words):
                next_texts = tuple(w.text.lower().strip(".,!?") for w in words[j : j + phrase_len])
                if next_texts == phrase_texts:
                    repeat_count += 1
                    j += phrase_len
                else:
                    break
            if repeat_count >= 3:
                # Keep only the first occurrence
                cleaned.extend(words[i : i + phrase_len])
                i = j
                found_repeat = True
                break
        if not found_repeat:
            cleaned.append(words[i])
            i += 1

    return cleaned


def _build_preprocess_filter_chain() -> str:
    default_chain = "pan=mono|c0=0.5*c0+0.5*c1"
    return (os.getenv("TRANSCRIBE_PREPROCESS_FILTER_CHAIN", default_chain) or default_chain).strip()


def _prepare_transcription_source(path: str, format: str = "wav") -> tuple[str, Path | None]:
    if not _env_bool("TRANSCRIBE_PREPROCESS_AUDIO", True):
        return path, None

    source_path = Path(path)
    if not source_path.exists():
        return path, None

    tmp_dir = Path(os.getenv("TMP_DIR", settings.tmp_dir))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path = tmp_dir / f"transcribe-pre-{uuid4()}.{format}"
    sample_rate = _env_int("TRANSCRIBE_PREPROCESS_SAMPLE_RATE", 16000, 8000)
    filter_chain = _build_preprocess_filter_chain()
    cmd = [
        settings.ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-af",
        filter_chain,
    ]
    # Add codec for compressed formats
    if format == "mp3":
        cmd.extend(["-codec:a", "libmp3lame", "-b:a", "64k"])
    cmd.append(str(output_path))

    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return path, None

    if process.returncode != 0 or not output_path.exists():
        print(f"[transcribe] ffmpeg preprocess failed (rc={process.returncode}): {process.stderr}", file=sys.stderr)
        output_path.unlink(missing_ok=True)
        return path, None
    if output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        return path, None
    return str(output_path), output_path


@lru_cache(maxsize=1)
def _prime_cuda_runtime_libraries() -> None:
    # In venv installs, CUDA user-space libs can live under site-packages/nvidia/*/lib.
    # Preload them so CTranslate2 can resolve CUDA 12 symbols without system-wide installs.
    search_roots: list[Path] = []
    site_dir = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    search_roots.append(site_dir)

    loaded_any = False
    lib_names = [
        "libcudart.so.12",
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcudnn.so.9",
        "libcudnn_ops_infer.so.9",
        "libcudnn_cnn_infer.so.9",
    ]
    subdirs = [
        ("nvidia", "cuda_runtime", "lib"),
        ("nvidia", "cublas", "lib"),
        ("nvidia", "cudnn", "lib"),
    ]

    for root in search_roots:
        for subdir in subdirs:
            lib_dir = root.joinpath(*subdir)
            if not lib_dir.exists():
                continue
            for name in lib_names:
                lib_path = lib_dir / name
                if not lib_path.exists():
                    continue
                try:
                    ctypes.CDLL(str(lib_path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
                    loaded_any = True
                except OSError:
                    continue

    if loaded_any:
        ld_library_path = os.getenv("LD_LIBRARY_PATH", "")
        paths = [
            str(search_roots[0] / "nvidia" / "cuda_runtime" / "lib"),
            str(search_roots[0] / "nvidia" / "cublas" / "lib"),
            str(search_roots[0] / "nvidia" / "cudnn" / "lib"),
        ]
        merged = [path for path in paths if Path(path).exists()]
        if ld_library_path:
            merged.append(ld_library_path)
        os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


@lru_cache(maxsize=4)
def _load_faster_whisper_model(model_name: str, device: str, compute_type: str) -> object:
    _prime_cuda_runtime_libraries()
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]

    return WhisperModel(model_name, device=device, compute_type=compute_type)


@lru_cache(maxsize=1)
def _gpu_available() -> bool:
    _prime_cuda_runtime_libraries()
    try:
        import ctranslate2  # type: ignore[import-not-found]

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def _resolve_device_and_compute_type() -> tuple[str, str]:
    raw_device = (os.getenv("TRANSCRIBE_DEVICE", "auto") or "auto").strip().lower()
    if raw_device in {"", "auto"}:
        device = "cuda" if _gpu_available() else "cpu"
    else:
        device = raw_device

    raw_compute = (os.getenv("TRANSCRIBE_COMPUTE_TYPE", "auto") or "auto").strip().lower()
    if raw_compute in {"", "auto"}:
        if device == "cuda":
            compute_type = (os.getenv("TRANSCRIBE_COMPUTE_TYPE_CUDA", "float16") or "float16").strip() or "float16"
        else:
            compute_type = (os.getenv("TRANSCRIBE_COMPUTE_TYPE_CPU", "int8") or "int8").strip() or "int8"
    else:
        compute_type = raw_compute
    return device, compute_type


def _build_from_faster_whisper(
    path: str,
    duration_sec: float,
    *,
    model_name: str | None = None,
    beam_size: int | None = None,
    force_vad_filter: bool | None = None,
) -> TranscriptPayload | None:
    transcribe_path, cleanup_path = _prepare_transcription_source(path)
    resolved_model_name = (model_name or os.getenv("TRANSCRIBE_MODEL", "base.en")).strip() or "base.en"
    device, compute_type = _resolve_device_and_compute_type()

    try:
        try:
            model = _load_faster_whisper_model(resolved_model_name, device, compute_type)
        except Exception:  # noqa: BLE001
            if device == "cuda":
                # CUDA may be unavailable despite configuration; retry on CPU.
                try:
                    model = _load_faster_whisper_model(
                        resolved_model_name,
                        "cpu",
                        (os.getenv("TRANSCRIBE_COMPUTE_TYPE_CPU", "int8") or "int8").strip() or "int8",
                    )
                except Exception:  # noqa: BLE001
                    return None
            else:
                return None

        transcribe_kwargs: dict[str, object] = {
            "beam_size": int(beam_size) if beam_size is not None else _env_int("TRANSCRIBE_BEAM_SIZE", 5, 1),
            "word_timestamps": True,
            "condition_on_previous_text": _env_bool("TRANSCRIBE_CONDITION_ON_PREVIOUS_TEXT", False),
            "no_speech_threshold": _env_float("TRANSCRIBE_NO_SPEECH_THRESHOLD", 0.6, 0.0),
            "log_prob_threshold": _env_float("TRANSCRIBE_LOG_PROB_THRESHOLD", -1.0, -10.0),
        }
        # Temperature fallback: start deterministic, retry with higher temperature on failure
        raw_temps = (os.getenv("TRANSCRIBE_TEMPERATURE", "") or "").strip()
        if raw_temps:
            try:
                temps = [float(t.strip()) for t in raw_temps.split(",") if t.strip()]
                if len(temps) == 1:
                    transcribe_kwargs["temperature"] = temps[0]
                elif temps:
                    transcribe_kwargs["temperature"] = temps
            except ValueError:
                pass  # Use faster-whisper default
        language = (os.getenv("TRANSCRIBE_LANGUAGE", "") or "").strip()
        if language:
            transcribe_kwargs["language"] = language
        initial_prompt = (os.getenv("TRANSCRIBE_INITIAL_PROMPT", "") or "").strip()
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt
        vad_filter_enabled = _env_bool("TRANSCRIBE_VAD_FILTER", False) if force_vad_filter is None else force_vad_filter
        if vad_filter_enabled:
            transcribe_kwargs["vad_filter"] = True
            transcribe_kwargs["vad_parameters"] = {"min_silence_duration_ms": 250}

        try:
            segments, info = model.transcribe(transcribe_path, **transcribe_kwargs)
        except Exception:  # noqa: BLE001
            if device == "cuda":
                # CUDA may partially initialize but fail during decode (e.g., missing user-space CUDA libs).
                try:
                    cpu_compute_type = (os.getenv("TRANSCRIBE_COMPUTE_TYPE_CPU", "int8") or "int8").strip() or "int8"
                    cpu_model = _load_faster_whisper_model(resolved_model_name, "cpu", cpu_compute_type)
                    segments, info = cpu_model.transcribe(transcribe_path, **transcribe_kwargs)
                except Exception:  # noqa: BLE001
                    return None
            else:
                return None

        words: list[TranscriptWordPayload] = []
        for segment in segments:
            segment_text = _clean_word(str(getattr(segment, "text", "") or ""))
            segment_start = float(getattr(segment, "start", 0.0) or 0.0)
            segment_end = float(getattr(segment, "end", segment_start + 0.2) or (segment_start + 0.2))

            segment_words = list(getattr(segment, "words", []) or [])
            if segment_words:
                for word in segment_words:
                    token = _clean_word(str(getattr(word, "word", "") or ""))
                    if not token:
                        continue
                    start_sec = float(getattr(word, "start", segment_start) or segment_start)
                    end_sec = float(
                        getattr(word, "end", max(start_sec + 0.05, segment_end)) or max(start_sec + 0.05, segment_end)
                    )
                    confidence = _normalize_confidence(getattr(word, "probability", None))
                    words.append(
                        TranscriptWordPayload(
                            id=str(uuid4()),
                            text=token,
                            start_sec=start_sec,
                            end_sec=end_sec,
                            confidence=confidence,
                        )
                    )
                continue

            if not segment_text:
                continue
            parts = segment_text.split()
            if not parts:
                continue
            span = max(segment_end - segment_start, 0.1)
            step = span / len(parts)
            for idx, token in enumerate(parts):
                start_sec = segment_start + (idx * step)
                end_sec = segment_start + ((idx + 1) * step)
                words.append(
                    TranscriptWordPayload(
                        id=str(uuid4()),
                        text=token,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        confidence=None,
                    )
                )

        normalized = _normalize_words(words, duration_sec)
        # Remove hallucinated repeated phrases
        if _env_bool("TRANSCRIBE_HALLUCINATION_FILTER", True):
            normalized = _detect_hallucinations(normalized)
        if len(normalized) < 2:
            return None
        text = " ".join(item.text for item in normalized)
        language = getattr(info, "language", None)
        return TranscriptPayload(
            source="faster_whisper",
            language=str(language) if language else None,
            text=text,
            words=normalized,
            is_mock=False,
        )
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)


def _min_expected_word_count(duration_sec: float) -> int:
    words_per_sec = _env_float("TRANSCRIBE_MIN_WORDS_PER_SEC", 0.45, 0.05)
    return max(8, int(round(max(duration_sec, 1.0) * words_per_sec)))


def _is_low_coverage(payload: TranscriptPayload, duration_sec: float) -> bool:
    return len(payload.words) < _min_expected_word_count(duration_sec)


def _end_coverage_ratio(payload: TranscriptPayload, duration_sec: float) -> float:
    if duration_sec <= 0 or not payload.words:
        return 0.0
    last_end = max(float(word.end_sec) for word in payload.words)
    return max(0.0, min(last_end / max(duration_sec, 0.1), 1.0))


def _max_word_gap_sec(payload: TranscriptPayload, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    if not payload.words:
        return duration_sec
    ordered = sorted(payload.words, key=lambda item: (float(item.start_sec), float(item.end_sec)))
    max_gap = max(float(ordered[0].start_sec), 0.0)
    for idx in range(1, len(ordered)):
        prev_end = float(ordered[idx - 1].end_sec)
        start = float(ordered[idx].start_sec)
        max_gap = max(max_gap, max(0.0, start - prev_end))
    max_gap = max(max_gap, max(0.0, duration_sec - float(ordered[-1].end_sec)))
    return max_gap


def _has_suspicious_long_gap(payload: TranscriptPayload, duration_sec: float) -> bool:
    threshold_sec = _env_float("TRANSCRIBE_MAX_WORD_GAP_SEC", 24.0, 2.0)
    min_words = _env_int("TRANSCRIBE_GAP_CHECK_MIN_WORDS", 20, 0)
    if len(payload.words) < min_words:
        return False
    if duration_sec < threshold_sec * 1.5:
        return False
    return _max_word_gap_sec(payload, duration_sec) >= threshold_sec


def _has_sparse_window(payload: TranscriptPayload, duration_sec: float) -> bool:
    min_words_for_check = _env_int("TRANSCRIBE_GAP_CHECK_MIN_WORDS", 20, 0)
    if len(payload.words) < min_words_for_check:
        return False

    window_sec = _env_float("TRANSCRIBE_SPARSE_WINDOW_SEC", 20.0, 5.0)
    min_words = _env_int("TRANSCRIBE_SPARSE_WINDOW_MIN_WORDS", 4, 0)
    if min_words <= 0:
        return False
    if duration_sec < window_sec * 1.5:
        return False

    step_sec = _env_float("TRANSCRIBE_SPARSE_WINDOW_STEP_SEC", max(5.0, window_sec / 2.0), 1.0)
    start_at_sec = _env_float("TRANSCRIBE_SPARSE_WINDOW_START_SEC", 20.0, 0.0)
    starts = sorted(float(word.start_sec) for word in payload.words)
    if not starts:
        return True

    left = 0
    right = 0
    total = len(starts)
    cursor = max(0.0, min(start_at_sec, max(0.0, duration_sec - window_sec)))
    while cursor + window_sec <= duration_sec + 1e-6:
        while left < total and starts[left] < cursor:
            left += 1
        if right < left:
            right = left
        window_end = cursor + window_sec
        while right < total and starts[right] < window_end:
            right += 1
        if (right - left) < min_words:
            return True
        cursor += step_sec
    return False


def _find_long_gaps(
    payload: TranscriptPayload,
    duration_sec: float,
    *,
    min_gap_sec: float,
) -> list[tuple[float, float]]:
    if duration_sec <= 0 or min_gap_sec <= 0:
        return []
    if not payload.words:
        return [(0.0, duration_sec)] if duration_sec >= min_gap_sec else []

    ordered = sorted(payload.words, key=lambda item: (float(item.start_sec), float(item.end_sec)))
    gaps: list[tuple[float, float]] = []
    cursor = max(0.0, float(ordered[0].start_sec))
    if cursor >= min_gap_sec:
        gaps.append((0.0, cursor))

    for idx in range(1, len(ordered)):
        prev_end = float(ordered[idx - 1].end_sec)
        start = float(ordered[idx].start_sec)
        if start - prev_end >= min_gap_sec:
            gaps.append((max(0.0, prev_end), min(duration_sec, start)))
    tail_gap = max(0.0, duration_sec - float(ordered[-1].end_sec))
    if tail_gap >= min_gap_sec:
        gaps.append((max(0.0, float(ordered[-1].end_sec)), duration_sec))
    return gaps


def _merge_gap_fill_transcript(
    primary: TranscriptPayload,
    secondary: TranscriptPayload | None,
    duration_sec: float,
) -> TranscriptPayload | None:
    if secondary is None or not secondary.words:
        return None

    default_gap_fill_sec = max(4.0, _env_float("TRANSCRIBE_MAX_WORD_GAP_SEC", 24.0, 2.0) * 0.75)
    min_gap_sec = _env_float("TRANSCRIBE_GAP_FILL_MIN_SEC", default_gap_fill_sec, 1.0)
    pad_sec = _env_float("TRANSCRIBE_GAP_FILL_PAD_SEC", 0.18, 0.0)
    gaps = _find_long_gaps(primary, duration_sec, min_gap_sec=min_gap_sec)
    if not gaps:
        return None

    additions: list[TranscriptWordPayload] = []
    for word in sorted(secondary.words, key=lambda item: (float(item.start_sec), float(item.end_sec))):
        center = (float(word.start_sec) + float(word.end_sec)) / 2.0
        for start_sec, end_sec in gaps:
            if (start_sec - pad_sec) <= center <= (end_sec + pad_sec):
                additions.append(word)
                break
    if not additions:
        return None

    merged = sorted(
        primary.words + additions,
        key=lambda item: (float(item.start_sec), float(item.end_sec), item.text.lower()),
    )
    deduped: list[TranscriptWordPayload] = []
    for word in merged:
        if deduped:
            prev = deduped[-1]
            if (
                word.text.strip().lower() == prev.text.strip().lower()
                and abs(float(word.start_sec) - float(prev.start_sec)) <= 0.05
                and abs(float(word.end_sec) - float(prev.end_sec)) <= 0.05
            ):
                continue
        deduped.append(word)

    normalized = _normalize_words(deduped, duration_sec)
    if _env_bool("TRANSCRIBE_HALLUCINATION_FILTER", True):
        normalized = _detect_hallucinations(normalized)
    if len(normalized) <= len(primary.words):
        return None
    return TranscriptPayload(
        source=f"{primary.source}_gapfill",
        language=primary.language or secondary.language,
        text=" ".join(word.text for word in normalized),
        words=normalized,
        is_mock=False,
    )


def _gap_fill_word_count(
    primary: TranscriptPayload,
    secondary: TranscriptPayload,
    duration_sec: float,
) -> int:
    default_gap_fill_sec = max(4.0, _env_float("TRANSCRIBE_MAX_WORD_GAP_SEC", 24.0, 2.0) * 0.75)
    min_gap_sec = _env_float("TRANSCRIBE_GAP_FILL_MIN_SEC", default_gap_fill_sec, 1.0)
    pad_sec = _env_float("TRANSCRIBE_GAP_FILL_PAD_SEC", 0.18, 0.0)
    gaps = _find_long_gaps(primary, duration_sec, min_gap_sec=min_gap_sec)
    if not gaps:
        return 0
    filled = 0
    for word in secondary.words:
        center = (float(word.start_sec) + float(word.end_sec)) / 2.0
        for start_sec, end_sec in gaps:
            if (start_sec - pad_sec) <= center <= (end_sec + pad_sec):
                filled += 1
                break
    return filled


def _pick_best_gap_fill_candidate(
    primary: TranscriptPayload,
    candidates: list[TranscriptPayload],
    duration_sec: float,
) -> TranscriptPayload | None:
    best: TranscriptPayload | None = None
    best_score: tuple[int, float, int] | None = None
    for candidate in candidates:
        score = (
            _gap_fill_word_count(primary, candidate, duration_sec),
            -_max_word_gap_sec(candidate, duration_sec),
            len(candidate.words),
        )
        if best is None or best_score is None or score > best_score:
            best = candidate
            best_score = score
    return best


def _confidence_stats(payload: TranscriptPayload) -> tuple[int, int, float]:
    threshold = _env_float("TRANSCRIBE_LOW_CONFIDENCE_THRESHOLD", 0.6, 0.0)
    values: list[float] = []
    for word in payload.words:
        confidence = _normalize_confidence(word.confidence)
        if confidence is None:
            continue
        values.append(confidence)
    if not values:
        return 0, 0, 1.0
    low_count = sum(1 for value in values if value < threshold)
    avg_confidence = sum(values) / len(values)
    return len(values), low_count, avg_confidence


def _is_low_confidence_quality(payload: TranscriptPayload) -> bool:
    min_words = _env_int("TRANSCRIBE_LOW_CONFIDENCE_MIN_WORDS", 30, 0)
    trigger_ratio = _env_float("TRANSCRIBE_LOW_CONFIDENCE_RATIO_TRIGGER", 0.18, 0.0)
    total, low_count, _ = _confidence_stats(payload)
    if total < min_words:
        return False
    return (low_count / total) >= trigger_ratio


def _pick_better_transcript(
    primary: TranscriptPayload | None,
    secondary: TranscriptPayload | None,
    duration_sec: float,
) -> TranscriptPayload | None:
    if primary is None:
        return secondary
    if secondary is None:
        return primary

    primary_count = len(primary.words)
    secondary_count = len(secondary.words)
    primary_low = _is_low_coverage(primary, duration_sec)
    secondary_low = _is_low_coverage(secondary, duration_sec)

    if primary_low and not secondary_low:
        return secondary
    if secondary_count >= primary_count + max(12, int(round(primary_count * 0.15))):
        return secondary

    primary_coverage = _end_coverage_ratio(primary, duration_sec)
    secondary_coverage = _end_coverage_ratio(secondary, duration_sec)
    if secondary_coverage >= primary_coverage + 0.08:
        return secondary

    primary_gap = _max_word_gap_sec(primary, duration_sec)
    secondary_gap = _max_word_gap_sec(secondary, duration_sec)
    if secondary_gap + 5.0 <= primary_gap:
        return secondary

    min_words = _env_int("TRANSCRIBE_LOW_CONFIDENCE_MIN_WORDS", 30, 0)
    primary_total, primary_low_count, primary_avg = _confidence_stats(primary)
    secondary_total, secondary_low_count, secondary_avg = _confidence_stats(secondary)
    if primary_total >= min_words and secondary_total >= min_words:
        primary_low_ratio = primary_low_count / primary_total
        secondary_low_ratio = secondary_low_count / secondary_total
        # Prefer a meaningful drop in risky words.
        if secondary_low_ratio + 0.05 <= primary_low_ratio:
            return secondary
        # If risk ratios are similar, pick the one with better average confidence.
        if secondary_low_ratio <= primary_low_ratio + 0.01 and secondary_avg >= primary_avg + 0.03:
            return secondary

    return primary


def _silence_ratio_for_profile(path: str, duration_sec: float) -> float | None:
    if duration_sec <= 0:
        return None
    if not Path(path).exists():
        return None

    min_analyze_sec = _env_float("TRANSCRIBE_PROFILE_MIN_ANALYZE_SEC", 25.0, 5.0)
    analyze_sec = min(duration_sec, _env_float("TRANSCRIBE_PROFILE_ANALYZE_SEC", 120.0, min_analyze_sec))
    if analyze_sec < min_analyze_sec:
        return None

    noise_db = _env_float("TRANSCRIBE_PROFILE_SILENCE_NOISE_DB", -35.0, -80.0)
    min_silence_sec = _env_float("TRANSCRIBE_PROFILE_MIN_SILENCE_SEC", 0.35, 0.05)
    silences = detect_silence_ranges(
        path,
        noise_db=noise_db,
        min_silence_sec=min_silence_sec,
        max_duration_sec=analyze_sec,
    )
    if not silences:
        return 0.0

    silence_total = 0.0
    for start_sec, end_sec in silences:
        start = max(0.0, min(float(start_sec), analyze_sec))
        end = max(0.0, min(float(end_sec), analyze_sec))
        if end > start:
            silence_total += end - start
    return max(0.0, min(silence_total / analyze_sec, 1.0))


def _resolve_transcription_profile(path: str, duration_sec: float) -> str:
    requested = (os.getenv("TRANSCRIBE_PROFILE", "auto") or "auto").strip().lower()
    if requested in {"speech", "music", "mixed"}:
        return requested
    if requested not in {"", "auto"}:
        return "mixed"

    silence_ratio = _silence_ratio_for_profile(path, duration_sec)
    if silence_ratio is None:
        return "mixed"

    speech_min_ratio = _env_float("TRANSCRIBE_PROFILE_SPEECH_MIN_SILENCE_RATIO", 0.10, 0.0)
    music_max_ratio = _env_float("TRANSCRIBE_PROFILE_MUSIC_MAX_SILENCE_RATIO", 0.04, 0.0)
    if silence_ratio >= speech_min_ratio:
        return "speech"
    if silence_ratio <= music_max_ratio:
        return "music"
    return "mixed"


def _resolve_groq_prompt_strategy(
    profile: str,
    primary_prompt: str | None,
    retry_prompt: str | None,
    retry_try_no_prompt: bool,
) -> tuple[str | None, str | None, bool]:
    if profile == "speech":
        speech_primary_prompt = (os.getenv("TRANSCRIBE_GROQ_PROMPT_SPEECH", "") or "").strip() or primary_prompt
        speech_retry_prompt = (os.getenv("TRANSCRIBE_GROQ_RETRY_PROMPT_SPEECH", "") or "").strip() or None
        speech_retry_try_no_prompt = _env_bool("TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT_SPEECH", False)
        return speech_primary_prompt, speech_retry_prompt, speech_retry_try_no_prompt
    if profile == "music":
        music_primary_prompt = (os.getenv("TRANSCRIBE_GROQ_PROMPT_MUSIC", "") or "").strip() or primary_prompt
        music_retry_prompt = (
            (os.getenv("TRANSCRIBE_GROQ_RETRY_PROMPT_MUSIC", "") or "").strip()
            or retry_prompt
            or DEFAULT_MUSIC_RETRY_PROMPT
        )
        music_retry_try_no_prompt = _env_bool("TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT_MUSIC", retry_try_no_prompt)
        return music_primary_prompt, music_retry_prompt, music_retry_try_no_prompt
    return primary_prompt, retry_prompt, retry_try_no_prompt


def _should_retry_groq_for_profile(profile: str, payload: TranscriptPayload, duration_sec: float) -> bool:
    common_retry = _is_low_coverage(payload, duration_sec) or _has_suspicious_long_gap(payload, duration_sec)
    if common_retry:
        return True
    if profile == "speech":
        return False
    return _has_sparse_window(payload, duration_sec)


def _build_mock_transcript(duration_sec: float) -> TranscriptPayload:
    safe_duration = max(duration_sec, 3.0)
    base_words = [
        "this",
        "is",
        "a",
        "generated",
        "transcript",
        "preview",
        "edit",
        "the",
        "text",
        "to",
        "cut",
        "the",
        "video",
        "automatically",
    ]
    target_count = max(8, int(round(safe_duration * 2.2)))
    step = safe_duration / target_count
    words: list[TranscriptWordPayload] = []
    for idx in range(target_count):
        token = base_words[idx % len(base_words)]
        start_sec = idx * step
        end_sec = min(safe_duration, start_sec + max(0.08, step * 0.82))
        words.append(
            TranscriptWordPayload(
                id=str(uuid4()),
                text=token,
                start_sec=start_sec,
                end_sec=end_sec,
                confidence=None,
            )
        )
    normalized = _normalize_words(words, safe_duration)
    text = " ".join(item.text for item in normalized)
    return TranscriptPayload(
        source="mock",
        language="en",
        text=text,
        words=normalized,
        is_mock=True,
    )


# ---------------------------------------------------------------------------
# Groq Cloud API backend (much faster than local, uses Whisper Large V3)
# ---------------------------------------------------------------------------
def _extract_audio_for_cloud(path: str) -> tuple[str, Path | None]:
    """Fast, lightweight audio extraction for cloud APIs (no heavy filters).

    Produces a small MP3 mono file suitable for upload to Groq (< 25MB limit).
    """
    source_path = Path(path)
    if not source_path.exists():
        return path, None

    tmp_dir = Path(os.getenv("TMP_DIR", settings.tmp_dir))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path = tmp_dir / f"groq-audio-{uuid4()}.mp3"

    cmd = [
        settings.ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(source_path),
        "-vn",                       # no video
        "-ac", "1",                  # mono
        "-ar", "16000",              # 16kHz is Whisper's native rate
        "-codec:a", "libmp3lame",
        "-b:a", "64k",              # 64kbps mono = ~0.5MB/min
        str(output_path),
    ]

    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        output_path.unlink(missing_ok=True)
        return path, None

    if process.returncode != 0 or not output_path.exists():
        output_path.unlink(missing_ok=True)
        return path, None
    if output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        return path, None
    return str(output_path), output_path


def _build_from_groq(
    path: str,
    duration_sec: float,
    *,
    model_name: str = "whisper-large-v3-turbo",
    prompt: str | None = None,
) -> TranscriptPayload | None:
    """Transcribe via Groq's cloud API. Returns None on failure."""
    api_key = (os.getenv("GROQ_API_KEY", "") or "").strip()
    if not api_key:
        return None

    try:
        from groq import Groq
    except ImportError:
        return None

    # Fast lightweight extraction (no heavy filters — Groq handles normalization)
    source_path, cleanup_path = _extract_audio_for_cloud(path)
    try:
        client = Groq(api_key=api_key)

        language = (os.getenv("TRANSCRIBE_LANGUAGE", "") or "").strip() or None
        resolved_prompt = (
            prompt.strip()
            if prompt is not None
            else (
                (os.getenv("TRANSCRIBE_GROQ_PROMPT", "") or "").strip()
                or (os.getenv("TRANSCRIBE_INITIAL_PROMPT", "") or "").strip()
            )
        ) or None
        request_kwargs: dict[str, object] = {
            "model": model_name,
            "response_format": "verbose_json",
            "timestamp_granularities": ["word", "segment"],
            "language": language,
        }
        if resolved_prompt:
            request_kwargs["prompt"] = resolved_prompt
        with open(source_path, "rb") as audio_file:
            request_payload = {
                **request_kwargs,
                "file": (Path(source_path).name, audio_file),
            }
            try:
                response = client.audio.transcriptions.create(**request_payload)
            except TypeError:
                # Some SDK versions may not expose `prompt` yet.
                request_payload.pop("prompt", None)
                response = client.audio.transcriptions.create(**request_payload)

        # Parse words from response.
        # Groq SDK may return `words` as dict entries instead of typed objects.
        words: list[TranscriptWordPayload] = []
        raw_words = getattr(response, "words", None) or []
        for item in raw_words:
            if isinstance(item, dict):
                word_text = str(item.get("word") or item.get("text") or "")
                start = float(item.get("start", 0.0) or 0.0)
                end = float(item.get("end", 0.0) or 0.0)
            else:
                word_text = getattr(item, "word", "") or getattr(item, "text", "") or ""
                start = float(getattr(item, "start", 0.0) or 0.0)
                end = float(getattr(item, "end", 0.0) or 0.0)
            if not word_text.strip():
                continue
            words.append(
                TranscriptWordPayload(
                    id=str(uuid4()),
                    text=word_text.strip(),
                    start_sec=start,
                    end_sec=end,
                    confidence=None,  # Groq doesn't return per-word confidence
                )
            )

        normalized = _normalize_words(words, duration_sec)
        if not normalized:
            # Fallback: some responses omit granular words but include timed segments.
            raw_segments = getattr(response, "segments", None) or []
            recovered_words: list[TranscriptWordPayload] = []
            for segment in raw_segments:
                if isinstance(segment, dict):
                    segment_text = str(segment.get("text") or "")
                    segment_start = float(segment.get("start", 0.0) or 0.0)
                    segment_end = float(segment.get("end", segment_start + 0.2) or (segment_start + 0.2))
                else:
                    segment_text = str(getattr(segment, "text", "") or "")
                    segment_start = float(getattr(segment, "start", 0.0) or 0.0)
                    segment_end = float(getattr(segment, "end", segment_start + 0.2) or (segment_start + 0.2))
                parts = segment_text.strip().split()
                if not parts:
                    continue
                span = max(segment_end - segment_start, 0.1)
                step = span / len(parts)
                for idx, token in enumerate(parts):
                    recovered_words.append(
                        TranscriptWordPayload(
                            id=str(uuid4()),
                            text=token,
                            start_sec=segment_start + (idx * step),
                            end_sec=segment_start + ((idx + 1) * step),
                            confidence=None,
                        )
                    )
            normalized = _normalize_words(recovered_words, duration_sec)
        if _env_bool("TRANSCRIBE_HALLUCINATION_FILTER", True):
            normalized = _detect_hallucinations(normalized)
        if not normalized:
            return None

        text = getattr(response, "text", "") or " ".join(w.text for w in normalized)
        return TranscriptPayload(
            source="groq",
            language=getattr(response, "language", None) or language,
            text=text.strip(),
            words=normalized,
            is_mock=False,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return None
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)


def _extract_audio_window_for_cloud(path: str, start_sec: float, end_sec: float) -> tuple[str | None, Path | None]:
    source_path = Path(path)
    if not source_path.exists():
        return None, None
    clip_start = max(0.0, float(start_sec))
    clip_end = max(clip_start, float(end_sec))
    clip_duration = clip_end - clip_start
    if clip_duration < 0.4:
        return None, None

    tmp_dir = Path(os.getenv("TMP_DIR", settings.tmp_dir))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path = tmp_dir / f"groq-window-{uuid4()}.mp3"
    cmd = [
        settings.ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{clip_start:.3f}",
        "-t",
        f"{clip_duration:.3f}",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "64k",
        str(output_path),
    ]
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        output_path.unlink(missing_ok=True)
        return None, None
    if process.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        return None, None
    return str(output_path), output_path


def _rescue_groq_gaps(
    path: str,
    duration_sec: float,
    primary: TranscriptPayload,
    *,
    profile: str,
    model_name: str,
    prompt: str | None,
) -> TranscriptPayload | None:
    if profile == "music":
        default_min_gap_sec = 6.0
        default_max_chunks = 8
        default_max_window_sec = 12.0
        default_pad_sec = 0.35
    elif profile == "mixed":
        default_min_gap_sec = 8.0
        default_max_chunks = 5
        default_max_window_sec = 20.0
        default_pad_sec = 0.35
    else:
        default_min_gap_sec = 10.0
        default_max_chunks = 3
        default_max_window_sec = 45.0
        default_pad_sec = 0.35

    min_gap_sec = _env_float(
        f"TRANSCRIBE_RESCUE_MIN_GAP_SEC_{profile.upper()}",
        _env_float("TRANSCRIBE_RESCUE_MIN_GAP_SEC", default_min_gap_sec, 2.0),
        2.0,
    )
    max_chunks = _env_int(
        f"TRANSCRIBE_RESCUE_MAX_CHUNKS_{profile.upper()}",
        _env_int("TRANSCRIBE_RESCUE_MAX_CHUNKS", default_max_chunks, 0),
        0,
    )
    max_window_sec = _env_float(
        f"TRANSCRIBE_RESCUE_MAX_WINDOW_SEC_{profile.upper()}",
        _env_float("TRANSCRIBE_RESCUE_MAX_WINDOW_SEC", default_max_window_sec, 5.0),
        5.0,
    )
    pad_sec = _env_float(
        f"TRANSCRIBE_RESCUE_PAD_SEC_{profile.upper()}",
        _env_float("TRANSCRIBE_RESCUE_PAD_SEC", default_pad_sec, 0.0),
        0.0,
    )
    gaps = _find_long_gaps(primary, duration_sec, min_gap_sec=min_gap_sec)
    if not gaps or max_chunks <= 0:
        return None

    ordered_gaps = sorted(gaps, key=lambda gap: (float(gap[1]) - float(gap[0])), reverse=True)
    collected_words: list[TranscriptWordPayload] = []
    used_chunks = 0

    for gap_start, gap_end in ordered_gaps:
        if used_chunks >= max_chunks:
            break
        window_start = max(0.0, float(gap_start) - pad_sec)
        window_end = min(duration_sec, float(gap_end) + pad_sec)
        cursor = window_start
        while cursor < window_end and used_chunks < max_chunks:
            chunk_end = min(window_end, cursor + max_window_sec)
            window_path, cleanup_path = _extract_audio_window_for_cloud(path, cursor, chunk_end)
            used_chunks += 1
            if not window_path:
                cursor = chunk_end
                continue
            try:
                window_payload = _build_from_groq(
                    window_path,
                    chunk_end - cursor,
                    model_name=model_name,
                    prompt=prompt,
                )
            finally:
                if cleanup_path is not None:
                    cleanup_path.unlink(missing_ok=True)
            if window_payload is not None:
                for word in window_payload.words:
                    collected_words.append(
                        TranscriptWordPayload(
                            id=str(uuid4()),
                            text=word.text,
                            start_sec=float(word.start_sec) + cursor,
                            end_sec=float(word.end_sec) + cursor,
                            confidence=word.confidence,
                        )
                    )
            cursor = chunk_end

    if _env_bool("TRANSCRIBE_RESCUE_SCRIPT_FILTER", True) and collected_words:
        primary_text = " ".join(word.text for word in primary.words[:600])
        primary_alpha_count = sum(1 for char in primary_text if char.isalpha())
        primary_latin_ratio = _ascii_latin_ratio(primary_text)
        primary_min_alpha = _env_int("TRANSCRIBE_RESCUE_PRIMARY_MIN_ALPHA", 40, 0)
        primary_latin_min = _env_float("TRANSCRIBE_RESCUE_PRIMARY_LATIN_RATIO", 0.65, 0.0)
        rescue_token_latin_min = _env_float("TRANSCRIBE_RESCUE_TOKEN_LATIN_MIN_RATIO", 0.35, 0.0)
        if primary_alpha_count >= primary_min_alpha and primary_latin_ratio >= primary_latin_min:
            filtered_words: list[TranscriptWordPayload] = []
            drop_non_ascii_tokens = _env_bool("TRANSCRIBE_RESCUE_DROP_NON_ASCII_TOKENS", True)
            for word in collected_words:
                alpha_count = sum(1 for char in word.text if char.isalpha())
                if alpha_count < 2:
                    filtered_words.append(word)
                    continue
                if drop_non_ascii_tokens and any(char.isalpha() and ord(char) > 127 for char in word.text):
                    continue
                if _ascii_latin_ratio(word.text) >= rescue_token_latin_min:
                    filtered_words.append(word)
            collected_words = filtered_words

    if not collected_words:
        return None
    secondary = TranscriptPayload(
        source="groq_rescue",
        language=primary.language,
        text=" ".join(word.text for word in collected_words),
        words=_normalize_words(collected_words, duration_sec),
        is_mock=False,
    )
    return _merge_gap_fill_transcript(primary, secondary, duration_sec)


def generate_transcript(path: str, duration_sec: float) -> TranscriptPayload:
    safe_duration = max(float(duration_sec), 0.1)
    profile = _resolve_transcription_profile(path, safe_duration)
    backend = (os.getenv("TRANSCRIBE_BACKEND", "auto") or "auto").strip().lower()
    groq_primary_model = (os.getenv("TRANSCRIBE_GROQ_MODEL", "whisper-large-v3-turbo") or "whisper-large-v3-turbo").strip()
    groq_retry_model = (os.getenv("TRANSCRIBE_GROQ_RETRY_MODEL", "whisper-large-v3") or "whisper-large-v3").strip()
    groq_primary_prompt = (os.getenv("TRANSCRIBE_GROQ_PROMPT", "") or "").strip() or None
    groq_retry_prompt = (
        (os.getenv("TRANSCRIBE_GROQ_RETRY_PROMPT", "") or "").strip()
        or groq_primary_prompt
    )
    groq_retry_try_no_prompt = _env_bool("TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT", True)
    groq_primary_prompt, groq_retry_prompt, groq_retry_try_no_prompt = _resolve_groq_prompt_strategy(
        profile,
        groq_primary_prompt,
        groq_retry_prompt,
        groq_retry_try_no_prompt,
    )
    groq_retry_enabled = _env_bool("TRANSCRIBE_GROQ_ENABLE_RETRY", True)
    groq_retry_min_duration_sec = _env_float("TRANSCRIBE_GROQ_RETRY_MIN_DURATION_SEC", 60.0, 0.0)

    # ------------------------------------------------------------------
    # Groq cloud backend: fast, no local GPU needed
    # ------------------------------------------------------------------
    if backend in {"groq", "auto"}:
        groq_result = _build_from_groq(
            path,
            safe_duration,
            model_name=groq_primary_model,
            prompt=groq_primary_prompt,
        )
        if groq_result is not None:
            should_retry_groq = (
                groq_retry_enabled
                and safe_duration >= groq_retry_min_duration_sec
                and _should_retry_groq_for_profile(profile, groq_result, safe_duration)
            )
            if should_retry_groq and groq_retry_model and groq_retry_model != groq_primary_model:
                groq_retry_candidates: list[TranscriptPayload] = []
                groq_retry = _build_from_groq(
                    path,
                    safe_duration,
                    model_name=groq_retry_model,
                    prompt=groq_retry_prompt,
                )
                if groq_retry is not None:
                    groq_retry_candidates.append(groq_retry)
                if groq_retry_try_no_prompt and groq_retry_prompt:
                    groq_retry_no_prompt = _build_from_groq(
                        path,
                        safe_duration,
                        model_name=groq_retry_model,
                        prompt=None,
                    )
                    if groq_retry_no_prompt is not None:
                        groq_retry_candidates.append(groq_retry_no_prompt)
                groq_retry = _pick_best_gap_fill_candidate(groq_result, groq_retry_candidates, safe_duration)
                merged = _merge_gap_fill_transcript(groq_result, groq_retry, safe_duration)
                if merged is not None:
                    min_gap_fill_words = _env_int("TRANSCRIBE_MIN_GAP_FILL_WORDS", 3, 1)
                    added_words = max(len(merged.words) - len(groq_result.words), 0)
                    if added_words >= min_gap_fill_words:
                        groq_result = merged
                    else:
                        preferred_gap_fill = _pick_better_transcript(groq_result, merged, safe_duration)
                        if preferred_gap_fill is not None:
                            groq_result = preferred_gap_fill
                # Only replace the full transcript with retry when primary remains low coverage.
                if _is_low_coverage(groq_result, safe_duration):
                    preferred_groq = _pick_better_transcript(groq_result, groq_retry, safe_duration)
                    if preferred_groq is not None:
                        return preferred_groq
            unresolved_gaps = _has_suspicious_long_gap(groq_result, safe_duration) or (
                profile != "speech" and _has_sparse_window(groq_result, safe_duration)
            )
            rescue_enabled = _env_bool("TRANSCRIBE_ENABLE_GAP_RESCUE", True)
            if unresolved_gaps and rescue_enabled and profile != "speech":
                rescue_model = (os.getenv("TRANSCRIBE_GROQ_RESCUE_MODEL", "") or "").strip() or groq_retry_model
                rescue_prompt = (
                    (os.getenv("TRANSCRIBE_GROQ_RESCUE_PROMPT", "") or "").strip()
                    or groq_retry_prompt
                    or DEFAULT_MUSIC_RETRY_PROMPT
                )
                rescue = _rescue_groq_gaps(
                    path,
                    safe_duration,
                    groq_result,
                    profile=profile,
                    model_name=rescue_model,
                    prompt=rescue_prompt,
                )
                if rescue is not None:
                    min_added_rescue_words = _env_int("TRANSCRIBE_MIN_RESCUE_ADDED_WORDS", 2, 1)
                    rescue_added = max(len(rescue.words) - len(groq_result.words), 0)
                    if rescue_added >= min_added_rescue_words:
                        groq_result = rescue
            return groq_result
        if backend == "groq":
            # User explicitly chose groq but it failed
            if _env_bool("TRANSCRIBE_ALLOW_MOCK_FALLBACK", True):
                return _build_mock_transcript(safe_duration)
            raise RuntimeError("Groq transcription failed. Check GROQ_API_KEY and network.")

    # ------------------------------------------------------------------
    # Local faster-whisper backend (original path)
    # ------------------------------------------------------------------
    primary_model = (os.getenv("TRANSCRIBE_MODEL", "base.en") or "base.en").strip() or "base.en"
    retry_model = (os.getenv("TRANSCRIBE_RETRY_MODEL", "medium") or "medium").strip() or "medium"
    retry_beam_size = _env_int("TRANSCRIBE_RETRY_BEAM_SIZE", 8, 1)
    allow_quality_retry = _env_bool("TRANSCRIBE_ENABLE_QUALITY_RETRY", True)
    retry_min_duration_sec = _env_float("TRANSCRIBE_RETRY_MIN_DURATION_SEC", 90.0, 0.0)
    can_retry = allow_quality_retry and safe_duration >= retry_min_duration_sec

    from_faster_whisper = _build_from_faster_whisper(path, safe_duration, model_name=primary_model)
    if from_faster_whisper is not None:
        should_retry = (
            _is_low_coverage(from_faster_whisper, safe_duration)
            or _is_low_confidence_quality(from_faster_whisper)
            or _has_suspicious_long_gap(from_faster_whisper, safe_duration)
        )
        if can_retry and should_retry:
            retry_result = _build_from_faster_whisper(
                path,
                safe_duration,
                model_name=retry_model,
                beam_size=retry_beam_size,
                force_vad_filter=False,
            )
            preferred = _pick_better_transcript(from_faster_whisper, retry_result, safe_duration)
            if preferred is not None:
                return preferred
        return from_faster_whisper

    if can_retry:
        retry_result = _build_from_faster_whisper(
            path,
            safe_duration,
            model_name=retry_model,
            beam_size=retry_beam_size,
            force_vad_filter=False,
        )
        if retry_result is not None:
            return retry_result
    if _env_bool("TRANSCRIBE_ALLOW_MOCK_FALLBACK", True):
        return _build_mock_transcript(safe_duration)
    raise RuntimeError("Transcription failed for the selected model. Verify model availability and compute settings.")
