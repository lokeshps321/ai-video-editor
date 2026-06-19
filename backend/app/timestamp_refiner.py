"""
Per-word timestamp refinement using audio energy analysis.

This module refines Whisper/Groq word timestamps by detecting actual speech onsets
in the audio waveform. Whisper timestamps often have systematic drift where early
words appear before actual speech.

V2: Efficient batch processing - extracts audio energy ONCE for the full file,
then uses numpy to refine all word onsets in a single pass.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.transcription_service import TranscriptWordPayload

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    return default


def _env_float(key: str, default: float, minimum: float = 0.0) -> float:
    try:
        val = float(os.environ.get(key, default))
        return max(minimum, val)
    except (ValueError, TypeError):
        return default


def _env_int(key: str, default: int, minimum: int = 1) -> int:
    try:
        val = int(os.environ.get(key, default))
        return max(minimum, val)
    except (ValueError, TypeError):
        return default


# -----------------------------------------------------------------------------
# Audio Energy Extraction (Full File - Single Pass)
# -----------------------------------------------------------------------------

def extract_full_audio_energy(
    audio_path: str,
    duration_sec: float,
    sample_rate: int = 16000,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract audio energy envelope for the ENTIRE file in a single FFmpeg call.
    
    Returns:
        Tuple of (timestamps, energy_values) arrays, or None on error.
    """
    if duration_sec <= 0:
        return None
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        # Extract full audio in a single FFmpeg call
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", audio_path,
            "-ac", "1",  # mono
            "-ar", str(sample_rate),
            "-f", "wav",
            tmp_path,
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            logger.warning("FFmpeg extraction failed: %s", result.stderr.decode()[:200])
            return None
        
        # Read raw audio samples
        import wave
        with wave.open(tmp_path, "rb") as wf:
            n_frames = wf.getnframes()
            if n_frames == 0:
                return None
            raw = wf.readframes(n_frames)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            samples /= 32768.0  # Normalize to [-1, 1]
        
        os.unlink(tmp_path)
        
        # Compute RMS energy with sliding window
        window_ms = _env_int("TRANSCRIBE_ENERGY_WINDOW_SIZE_MS", 10, 5)
        window_samples = int(sample_rate * window_ms / 1000)
        hop_samples = max(1, window_samples // 4)  # 75% overlap for finer resolution
        
        n_windows = max(1, (len(samples) - window_samples) // hop_samples + 1)
        energy = np.zeros(n_windows, dtype=np.float32)
        timestamps = np.zeros(n_windows, dtype=np.float32)
        
        for i in range(n_windows):
            start_idx = i * hop_samples
            end_idx = min(start_idx + window_samples, len(samples))
            window = samples[start_idx:end_idx]
            energy[i] = np.sqrt(np.mean(window ** 2))  # RMS
            timestamps[i] = start_idx / sample_rate
        
        # Smooth the energy envelope
        smoothing = _env_int("TRANSCRIBE_ENERGY_SMOOTHING_WINDOW", 5, 1)
        if smoothing > 1 and len(energy) > smoothing:
            kernel = np.ones(smoothing, dtype=np.float32) / smoothing
            energy = np.convolve(energy, kernel, mode='same')
        
        return timestamps, energy
        
    except Exception as e:
        logger.warning("Full energy extraction failed: %s", e)
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None


def find_onset_near(
    timestamps: np.ndarray,
    energy: np.ndarray,
    expected_time: float,
    search_before_sec: float = 0.20,
    search_after_sec: float = 0.12,
) -> float | None:
    """
    Find the speech onset closest to expected_time using energy derivative.

    Searches before and after the expected time to find the actual energy
    rise that corresponds to the start of a spoken word/lyric.  The default
    window is intentionally wider (200 ms / 120 ms) so that song lyrics,
    which can drift up to ~300 ms from ASR timestamps, are still caught.
    """
    if len(timestamps) < 3:
        return None

    window_start = expected_time - search_before_sec
    window_end   = expected_time + search_after_sec

    mask = (timestamps >= window_start) & (timestamps <= window_end)
    if not np.any(mask):
        return None

    indices = np.where(mask)[0]
    if len(indices) < 3:
        return None

    energy_window = energy[indices]
    time_window   = timestamps[indices]

    # Compute derivative (energy changes)
    derivative = np.diff(energy_window)
    if len(derivative) == 0:
        return None

    # Find the steepest energy rise (onset)
    onset_threshold = _env_float("TRANSCRIBE_ENERGY_ONSET_THRESHOLD", 0.012, 0.001)
    max_deriv = np.max(derivative)

    if max_deriv > onset_threshold:
        onset_idx = np.argmax(derivative)
        # Use the frame *before* the peak derivative (where the rise starts)
        best_idx = max(0, onset_idx - 1)
        return float(time_window[best_idx])

    # Fallback: find first frame above dynamic threshold
    mean_energy = np.mean(energy_window)
    std_energy  = np.std(energy_window)
    if std_energy > 0.001:
        threshold = mean_energy + 0.25 * std_energy
        above = np.where(energy_window > threshold)[0]
        if len(above) > 0:
            return float(time_window[above[0]])

    return None


def find_offset_near(
    timestamps: np.ndarray,
    energy: np.ndarray,
    expected_end: float,
    search_before_sec: float = 0.08,
    search_after_sec: float = 0.15,
) -> float | None:
    """
    Find the speech offset (end) closest to expected_end using energy drop.

    Searches for the steepest energy *decrease* after the expected end time
    to locate when the voice actually stops.
    """
    if len(timestamps) < 3:
        return None

    window_start = expected_end - search_before_sec
    window_end   = expected_end + search_after_sec

    mask = (timestamps >= window_start) & (timestamps <= window_end)
    if not np.any(mask):
        return None

    indices = np.where(mask)[0]
    if len(indices) < 3:
        return None

    energy_window = energy[indices]
    time_window   = timestamps[indices]

    derivative = np.diff(energy_window)
    if len(derivative) == 0:
        return None

    # Find the steepest energy *drop* (offset)
    drop_threshold = _env_float("TRANSCRIBE_ENERGY_ONSET_THRESHOLD", 0.012, 0.001)
    min_deriv = np.min(derivative)

    if min_deriv < -drop_threshold:
        drop_idx = np.argmin(derivative)
        # Return the frame just after the steepest drop
        best_idx = min(drop_idx + 1, len(time_window) - 1)
        return float(time_window[best_idx])

    return None


# -----------------------------------------------------------------------------
# Per-Word Timestamp Refinement
# -----------------------------------------------------------------------------

@dataclass
class RefinementResult:
    """Result of timestamp refinement for a single word."""
    original_start: float
    refined_start: float
    offset_applied: float
    method: str  # "energy", "confidence", "none"


def refine_word_timestamp(
    word_start: float,
    word_end: float,
    audio_path: str,
    confidence: float | None = None,
    search_window_sec: float = 0.15,
) -> RefinementResult:
    """
    Refine a single word's timestamp using energy-based onset detection.
    (Legacy single-word API - prefer refine_word_timestamps_batch)
    """
    method = "none"
    refined_start = word_start
    
    # Strategy 1: Energy-based onset detection
    energy_enabled = _env_bool("TRANSCRIBE_ENERGY_REFINEMENT_ENABLED", True)
    
    if energy_enabled and audio_path:
        extract_start = max(0, word_start - search_window_sec - 0.1)
        extract_end = word_end + search_window_sec + 0.1
        
        result = extract_full_audio_energy(audio_path, extract_end - extract_start)
        
        if result is not None:
            timestamps, energy = result
            # Adjust timestamps to absolute time
            timestamps = timestamps + extract_start
            onset_time = find_onset_near(
                timestamps, energy, word_start, search_window_sec, search_window_sec
            )
            
            if onset_time is not None:
                shift = onset_time - word_start
                if abs(shift) <= search_window_sec:
                    refined_start = onset_time
                    method = "energy"
    
    # Strategy 2: Confidence-based offset (fallback)
    if method == "none" and confidence is not None:
        conf_enabled = _env_bool("TRANSCRIBE_CONFIDENCE_OFFSET_ENABLED", True)
        conf_threshold = _env_float("TRANSCRIBE_CONFIDENCE_OFFSET_THRESHOLD", 0.7, 0.0)
        conf_offset = _env_float("TRANSCRIBE_CONFIDENCE_OFFSET_SEC", 0.05, 0.0)
        
        if conf_enabled and confidence < conf_threshold:
            refined_start = word_start + conf_offset
            method = "confidence"
    
    return RefinementResult(
        original_start=word_start,
        refined_start=refined_start,
        offset_applied=refined_start - word_start,
        method=method,
    )


def refine_word_timestamps_batch(
    words: list,  # List of TranscriptWordPayload
    audio_path: str,
    max_words: int = 2000,
) -> list:
    """
    Refine timestamps for a batch of words efficiently.
    
    Extracts audio energy ONCE for the entire file, then finds speech onsets
    for all words in a single numpy pass. Much faster than per-word FFmpeg calls.
    
    Args:
        words: List of TranscriptWordPayload objects
        audio_path: Path to audio/video file
        max_words: Maximum words to process
    
    Returns:
        List of words with refined timestamps
    """
    from app.transcription_service import TranscriptWordPayload, _copy_word_payload
    
    if not words or not audio_path:
        return words
    
    refinement_enabled = _env_bool("TRANSCRIBE_TIMESTAMP_REFINEMENT_ENABLED", True)
    if not refinement_enabled:
        return words
    
    energy_enabled = _env_bool("TRANSCRIBE_ENERGY_REFINEMENT_ENABLED", True)
    
    # Determine duration from words
    if not words:
        return words
    
    duration_sec = max(float(w.end_sec) for w in words) + 1.0
    
    # Extract audio energy ONCE for the entire file
    full_energy = None
    if energy_enabled:
        full_energy = extract_full_audio_energy(audio_path, duration_sec)
        if full_energy is not None:
            logger.info(
                f"Extracted energy envelope: {len(full_energy[0])} frames "
                f"for {duration_sec:.1f}s audio"
            )
    
    # Limit words to process
    if len(words) > max_words:
        words_to_refine = words[:max_words]
        remaining = words[max_words:]
    else:
        words_to_refine = words
        remaining = []
    
    # Configuration — wider defaults so song lyrics (which drift more) are caught
    search_before = _env_float("TRANSCRIBE_ENERGY_SEARCH_BEFORE_SEC", 0.20, 0.01)
    search_after  = _env_float("TRANSCRIBE_ENERGY_SEARCH_AFTER_SEC",  0.12, 0.01)
    max_shift     = _env_float("TRANSCRIBE_ENERGY_MAX_SHIFT_SEC",     0.25, 0.01)
    refine_ends   = _env_bool("TRANSCRIBE_ENERGY_REFINE_ENDS", True)
    conf_enabled  = _env_bool("TRANSCRIBE_CONFIDENCE_OFFSET_ENABLED", True)
    conf_threshold = _env_float("TRANSCRIBE_CONFIDENCE_OFFSET_THRESHOLD", 0.7, 0.0)
    conf_offset   = _env_float("TRANSCRIBE_CONFIDENCE_OFFSET_SEC", 0.04, 0.0)
    
    refined_words = []
    total_offset = 0.0
    refined_count = 0
    
    for word in words_to_refine:
        new_start = word.start_sec
        new_end   = word.end_sec
        method    = "none"

        # Strategy 1: Energy-based onset detection (uses cached full energy)
        if full_energy is not None:
            ts, energy = full_energy
            onset = find_onset_near(
                ts, energy, word.start_sec,
                search_before, search_after,
            )
            if onset is not None:
                shift = onset - word.start_sec
                if abs(shift) <= max_shift:
                    new_start = onset
                    method = "energy"

            # Also refine the end time using energy drop detection
            if refine_ends and full_energy is not None:
                offset_time = find_offset_near(
                    ts, energy, word.end_sec,
                    search_before_sec=0.08,
                    search_after_sec=0.15,
                )
                if offset_time is not None:
                    end_shift = offset_time - word.end_sec
                    if abs(end_shift) <= max_shift and offset_time > new_start + 0.02:
                        new_end = offset_time

        # Strategy 2: Confidence-based offset (fallback for low-confidence words)
        if method == "none" and conf_enabled and word.confidence is not None:
            if word.confidence < conf_threshold:
                new_start = word.start_sec + conf_offset
                method = "confidence"

        if method != "none" or new_end != word.end_sec:
            # Guarantee start < end with a minimum 20ms duration
            new_end = max(new_start + 0.02, new_end)
            refined_word = _copy_word_payload(
                word,
                start_sec=max(0.0, new_start),
                end_sec=max(new_start + 0.02, new_end),
            )
            total_offset += (new_start - word.start_sec)
            if method != "none":
                refined_count += 1
        else:
            refined_word = word

        refined_words.append(refined_word)
    
    refined_words.extend(remaining)
    
    if refined_count > 0:
        avg_offset = total_offset / refined_count
        logger.info(
            f"Refined {refined_count}/{len(words_to_refine)} word timestamps, "
            f"avg offset: {avg_offset*1000:.1f}ms"
        )
    
    return refined_words


# -----------------------------------------------------------------------------
# Quick Refinement (No Audio Analysis)
# -----------------------------------------------------------------------------

def apply_confidence_based_offset(
    words: list,  # List of TranscriptWordPayload
) -> list:
    """
    Apply confidence-based offset without audio analysis.
    
    Low-confidence words from Whisper often have earlier timestamps than actual speech.
    """
    from app.transcription_service import TranscriptWordPayload, _copy_word_payload
    
    if not words:
        return words
    
    enabled = _env_bool("TRANSCRIBE_CONFIDENCE_OFFSET_ENABLED", True)
    if not enabled:
        return words
    
    threshold = _env_float("TRANSCRIBE_CONFIDENCE_OFFSET_THRESHOLD", 0.7, 0.0)
    offset_sec = _env_float("TRANSCRIBE_CONFIDENCE_OFFSET_SEC", 0.04, 0.0)
    
    refined = []
    for word in words:
        if word.confidence is not None and word.confidence < threshold:
            duration = word.end_sec - word.start_sec
            new_start = word.start_sec + offset_sec
            new_end = new_start + duration
            refined.append(_copy_word_payload(word, start_sec=new_start, end_sec=new_end))
        else:
            refined.append(word)
    
    return refined
