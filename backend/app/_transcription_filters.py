from __future__ import annotations

import logging

from .media_utils import detect_silence_ranges
from ._transcription_constants import (
    _ANYWHERE_HALLUCINATION_PHRASES,
    _HALLUCINATION_SINGLETONS_IN_GAPS,
    _HALLUCINATION_STOPWORDS,
    _HEAD_HALLUCINATION_PHRASES,
    _PROMPT_LEAKAGE_PHRASES,
    _PROMPT_LEAKAGE_SINGLETONS,
    _TAIL_HALLUCINATION_PHRASES,
)
from ._transcription_payloads import (
    TranscriptPayload,
    TranscriptWordPayload,
    _copy_word_payload,
    _normalize_source_pass,
    _word_midpoint_sec,
)
from ._transcription_text import (
    _clamp_time,
    _clean_word,
    _env_bool,
    _env_float,
    _env_int,
    _normalize_confidence,
    _normalize_token,
    _runtime_mode,
    _runtime_profile,
)

_perf_logger = logging.getLogger(__name__)


def _normalize_words(
    words: list[TranscriptWordPayload],
    duration_sec: float,
    *,
    apply_offset: bool = True,
    offset_sec: float | None = None,
) -> list[TranscriptWordPayload]:
    min_confidence = _env_float("TRANSCRIBE_WORD_MIN_CONFIDENCE", 0.15, 0.0)
    min_word_duration_sec = _env_float("TRANSCRIBE_MIN_WORD_DURATION_SEC", 0.05, 0.01)
    max_word_duration_sec = _env_float(
        "TRANSCRIBE_MAX_WORD_DURATION_SEC", 1.2, min_word_duration_sec
    )
    next_word_guard_sec = _env_float("TRANSCRIBE_WORD_NEXT_GUARD_SEC", 0.01, 0.0)
    # Global timestamp offset: only applied during initial generation (apply_offset=True)
    # NOT during storage/reading to avoid triple-application
    if offset_sec is not None:
        # Providers such as Sarvam supply a calibrated adjustment. It must
        # take precedence over the generic Whisper/Groq environment offset.
        try:
            timestamp_offset_sec = max(-5.0, min(5.0, float(offset_sec)))
        except (TypeError, ValueError):
            timestamp_offset_sec = 0.0
    elif apply_offset:
        timestamp_offset_sec = _env_float(
            "TRANSCRIBE_TIMESTAMP_OFFSET_SEC", 0.0, -5.0
        )
    else:
        timestamp_offset_sec = 0.0

    prelim: list[TranscriptWordPayload] = []
    for item in sorted(words, key=lambda entry: entry.start_sec):
        # Apply global timestamp offset (only on initial generation)
        raw_start = float(item.start_sec) + timestamp_offset_sec
        raw_end = float(item.end_sec) + timestamp_offset_sec
        start_sec = round(_clamp_time(raw_start, duration_sec), 3)
        end_sec = round(_clamp_time(raw_end, duration_sec), 3)
        if end_sec <= start_sec:
            end_sec = round(min(duration_sec, start_sec + min_word_duration_sec), 3)
        text = _clean_word(item.text)
        if not text:
            continue
        confidence = _normalize_confidence(item.confidence)
        # Filter out words with very low confidence (likely hallucinations)
        if confidence is not None and confidence < min_confidence:
            continue
        prelim.append(
            TranscriptWordPayload(
                id=item.id,
                text=text,
                start_sec=start_sec,
                end_sec=end_sec,
                # Cached romanization survives only if the text didn't change.
                display_text=item.display_text if text == item.text else None,
                confidence=confidence,
                quality_score=item.quality_score,
                quality_label=item.quality_label,
                source_pass=_normalize_source_pass(item.source_pass),
                speaker_id=item.speaker_id,
                speaker_label=item.speaker_label,
            )
        )

    normalized: list[TranscriptWordPayload] = []
    for index, item in enumerate(prelim):
        start_sec = float(item.start_sec)
        end_sec = min(float(item.end_sec), start_sec + max_word_duration_sec)

        if index + 1 < len(prelim):
            next_start = float(prelim[index + 1].start_sec)
            if next_start > start_sec:
                end_sec = min(end_sec, next_start - next_word_guard_sec)

        if end_sec <= start_sec:
            fallback_end = min(duration_sec, start_sec + min_word_duration_sec)
            if index + 1 < len(prelim):
                next_start = float(prelim[index + 1].start_sec)
                if next_start > start_sec:
                    fallback_end = min(
                        fallback_end,
                        max(start_sec + 0.001, next_start - next_word_guard_sec),
                    )
            end_sec = fallback_end

        if end_sec <= start_sec:
            continue

        normalized.append(
            TranscriptWordPayload(
                id=item.id,
                text=item.text,
                start_sec=round(start_sec, 3),
                end_sec=round(min(end_sec, duration_sec), 3),
                display_text=item.display_text,
                confidence=item.confidence,
                quality_score=item.quality_score,
                quality_label=item.quality_label,
                source_pass=_normalize_source_pass(item.source_pass),
                speaker_id=item.speaker_id,
                speaker_label=item.speaker_label,
            )
        )
    return normalized


def sanitize_transcript_words(
    words: list[TranscriptWordPayload],
    duration_sec: float,
    *,
    apply_filters: bool = False,
    apply_offset: bool = True,
) -> list[TranscriptWordPayload]:
    normalized = _normalize_words(
        words, max(float(duration_sec), 0.0), apply_offset=apply_offset
    )
    if apply_filters:
        normalized = _trim_known_tail_hallucination(
            normalized, max(float(duration_sec), 0.0)
        )
    return normalized


def _detect_hallucinations(
    words: list[TranscriptWordPayload],
) -> list[TranscriptWordPayload]:
    """Remove repeated phrase loops — a common Whisper failure mode.

    Detects when the same short phrase is repeated 3+ times consecutively
    and collapses it to a single occurrence.
    """
    if len(words) < 6:
        return words

    # Collapse quick duplicate bursts first (e.g., "I'm I'm worried worried").
    # These are common in music-heavy or noisy sources and should not survive
    # to editing UX as separate words.
    dup_gap_sec = _env_float("TRANSCRIBE_HALLUCINATION_DUP_GAP_SEC", 0.35, 0.0)
    deduped: list[TranscriptWordPayload] = []
    for item in words:
        token = _normalize_token(item.text)
        if not token:
            deduped.append(item)
            continue
        if deduped:
            prev = deduped[-1]
            prev_token = _normalize_token(prev.text)
            if prev_token and prev_token == token:
                gap_sec = float(item.start_sec) - float(prev.end_sec)
                if gap_sec <= dup_gap_sec:
                    deduped[-1] = _copy_word_payload(
                        prev,
                        start_sec=float(prev.start_sec),
                        end_sec=max(float(prev.end_sec), float(item.end_sec)),
                        confidence=prev.confidence
                        if prev.confidence is not None
                        else item.confidence,
                        quality_score=prev.quality_score
                        if prev.quality_score is not None
                        else item.quality_score,
                        quality_label=prev.quality_label or item.quality_label,
                        source_pass=_normalize_source_pass(
                            prev.source_pass or item.source_pass
                        ),
                    )
                    continue
        deduped.append(item)

    words = deduped
    if len(words) < 6:
        return words

    profile = _runtime_profile()
    mode = _runtime_mode()
    if mode == "song" and not _env_bool("TRANSCRIBE_COLLAPSE_REPEATS_IN_SONG", False):
        return words
    if profile in {"music", "mixed"}:
        if _env_bool("TRANSCRIBE_PRESERVE_MUSIC_REPEATS", True):
            return words
        if not _env_bool("TRANSCRIBE_COLLAPSE_REPEATS_IN_MUSIC", False):
            return words

    cleaned: list[TranscriptWordPayload] = []
    i = 0
    while i < len(words):
        # Try phrase lengths 1-4 words
        found_repeat = False
        for phrase_len in range(1, min(5, (len(words) - i) // 2 + 1)):
            phrase_texts = tuple(
                _normalize_token(w.text) for w in words[i : i + phrase_len]
            )
            if any(not token for token in phrase_texts):
                continue
            repeat_count = 1
            j = i + phrase_len
            while j + phrase_len <= len(words):
                next_texts = tuple(
                    _normalize_token(w.text) for w in words[j : j + phrase_len]
                )
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

    # Keep default behavior simple to avoid over-pruning valid words.
    # Strict mode enables stronger phrase/backtrack suppression for noisy music-heavy audio.
    if not _env_bool("TRANSCRIBE_HALLUCINATION_STRICT", False):
        return cleaned

    # Collapse repeated short phrases that recur many times in a tight window
    # (e.g. lyric loops) while preserving the first occurrence.
    min_occurrences = _env_int("TRANSCRIBE_HALLUCINATION_REPEAT_MIN_OCCURRENCES", 3, 2)
    repeat_window_sec = _env_float(
        "TRANSCRIBE_HALLUCINATION_REPEAT_WINDOW_SEC", 20.0, 2.0
    )
    if len(cleaned) < 8 or min_occurrences <= 1:
        return cleaned

    tokens = [_normalize_token(item.text) for item in cleaned]
    drop_indices: set[int] = set()

    def _mark_cluster(indices: list[int], phrase_len: int) -> None:
        if len(indices) < min_occurrences:
            return
        # Keep first occurrence, drop subsequent repeats.
        for idx in indices[1:]:
            for offset in range(phrase_len):
                drop_indices.add(idx + offset)

    for phrase_len in (4, 3):
        if len(tokens) < phrase_len:
            continue
        phrase_map: dict[tuple[str, ...], list[int]] = {}
        for idx in range(0, len(tokens) - phrase_len + 1):
            phrase = tuple(tokens[idx : idx + phrase_len])
            if any(not token for token in phrase):
                continue
            if all(token in _HALLUCINATION_STOPWORDS for token in phrase):
                continue
            phrase_map.setdefault(phrase, []).append(idx)

        for indices in phrase_map.values():
            if len(indices) < min_occurrences:
                continue
            cluster: list[int] = []
            for idx in indices:
                if not cluster:
                    cluster.append(idx)
                    continue
                first_start = float(cleaned[cluster[0]].start_sec)
                current_start = float(cleaned[idx].start_sec)
                if current_start - first_start <= repeat_window_sec:
                    cluster.append(idx)
                    continue
                _mark_cluster(cluster, phrase_len)
                cluster = [idx]
            _mark_cluster(cluster, phrase_len)

    if not drop_indices:
        return cleaned

    filtered = [item for idx, item in enumerate(cleaned) if idx not in drop_indices]
    if not filtered:
        filtered = cleaned

    # Collapse short "backtrack" glitches (A B A or A B A B) that appear in
    # noisy mixes after ASR decoding.
    backtrack_gap_sec = _env_float(
        "TRANSCRIBE_HALLUCINATION_BACKTRACK_GAP_SEC", 1.2, 0.2
    )
    if len(filtered) < 3:
        return filtered
    normalized = [_normalize_token(item.text) for item in filtered]
    remove_backtrack: set[int] = set()

    for idx in range(0, len(filtered) - 2):
        a = normalized[idx]
        c = normalized[idx + 2]
        if not a or not c or a != c:
            continue
        span = float(filtered[idx + 2].start_sec) - float(filtered[idx].start_sec)
        if span <= backtrack_gap_sec:
            remove_backtrack.add(idx + 2)

    for idx in range(0, len(filtered) - 3):
        a = normalized[idx]
        b = normalized[idx + 1]
        c = normalized[idx + 2]
        d = normalized[idx + 3]
        if not a or not b:
            continue
        if a == c and b == d:
            span = float(filtered[idx + 3].start_sec) - float(filtered[idx].start_sec)
            if span <= backtrack_gap_sec * 1.8:
                remove_backtrack.add(idx + 2)
                remove_backtrack.add(idx + 3)

    if not remove_backtrack:
        return filtered
    collapsed = [
        item for idx, item in enumerate(filtered) if idx not in remove_backtrack
    ]
    return collapsed if collapsed else filtered


def _detect_sparse_hallucinations(
    words: list[TranscriptWordPayload],
    duration_sec: float,
) -> list[TranscriptWordPayload]:
    """Remove isolated low-confidence words in music/silence gaps.

    Whisper often hallucinates a handful of random words during long stretches
    of instrumental audio or silence.  This filter scans for sparse windows
    (few words spread across a long time span) with low average confidence and
    removes them.

    For song/music profiles, the filter is more aggressive when confidence data
    is unavailable (e.g. Groq backend): any sparse cluster surrounded by large
    gaps is likely a hallucination in an instrumental section.
    """
    if len(words) < 3 or duration_sec <= 0:
        return words

    sparse_window_sec = _env_float(
        "TRANSCRIBE_SPARSE_HALLUCINATION_WINDOW_SEC", 8.0, 3.0
    )
    sparse_max_words = _env_int("TRANSCRIBE_SPARSE_HALLUCINATION_MAX_WORDS", 5, 1)
    sparse_max_confidence = _env_float(
        "TRANSCRIBE_SPARSE_HALLUCINATION_MAX_CONFIDENCE", 0.55, 0.0
    )
    sparse_min_gap_before = _env_float(
        "TRANSCRIBE_SPARSE_HALLUCINATION_MIN_GAP_SEC", 3.0, 1.0
    )

    if not _env_bool("TRANSCRIBE_SPARSE_HALLUCINATION_FILTER", True):
        return words

    # In song/music profiles, be more aggressive with gap detection and
    # treat missing-confidence clusters as hallucinations.
    profile = _runtime_profile()
    mode = _runtime_mode()
    is_music_context = profile in {"music", "mixed"} or mode == "song"
    music_min_gap = _env_float(
        "TRANSCRIBE_SPARSE_HALLUCINATION_MUSIC_MIN_GAP_SEC", 2.0, 0.5
    )
    effective_min_gap = music_min_gap if is_music_context else sparse_min_gap_before

    drop_indices: set[int] = set()

    # Build a sorted list with indices for efficient scanning
    indexed = list(enumerate(words))

    for i, word_i in indexed:
        # Find runs of words that are sparse (few words in a wide time window)
        # Look for a gap before or after this word that indicates a non-speech region
        gap_before = (
            float(word_i.start_sec)
            if i == 0
            else (float(word_i.start_sec) - float(words[i - 1].end_sec))
        )
        if gap_before < effective_min_gap and i > 0:
            continue

        # Collect consecutive words that fall within the sparse window
        cluster_indices: list[int] = [i]
        cluster_end = float(word_i.end_sec)
        for j in range(i + 1, len(words)):
            if float(words[j].start_sec) - float(word_i.start_sec) > sparse_window_sec:
                break
            cluster_indices.append(j)
            cluster_end = float(words[j].end_sec)

        # Check if there's also a gap after the cluster
        last_idx = cluster_indices[-1]
        gap_after = (
            (duration_sec - cluster_end)
            if last_idx == len(words) - 1
            else (float(words[last_idx + 1].start_sec) - cluster_end)
        )
        if gap_after < effective_min_gap and last_idx < len(words) - 1:
            continue

        # This cluster is sparse (surrounded by gaps) - check word count and confidence
        if len(cluster_indices) > sparse_max_words:
            continue

        # Check average confidence of the cluster
        confidences = [
            float(words[idx].confidence)
            for idx in cluster_indices
            if words[idx].confidence is not None
        ]
        if not confidences:
            # No confidence data available (common with Groq backend).
            # In music/song context, sparse clusters surrounded by large gaps
            # during instrumental sections are almost certainly hallucinations.
            if is_music_context:
                drop_indices.update(cluster_indices)
            else:
                # Fallback: only drop if all words are stopwords
                all_stopwords = all(
                    _normalize_token(words[idx].text) in _HALLUCINATION_STOPWORDS
                    for idx in cluster_indices
                )
                if all_stopwords:
                    drop_indices.update(cluster_indices)
            continue

        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence <= sparse_max_confidence:
            drop_indices.update(cluster_indices)

    if not drop_indices:
        return words

    _perf_logger.debug(
        "Sparse hallucination filter removed %d word(s) in %s context: %s",
        len(drop_indices),
        "music/song" if is_music_context else "speech",
        [words[idx].text for idx in sorted(drop_indices)],
    )
    filtered = [word for idx, word in enumerate(words) if idx not in drop_indices]
    return filtered if filtered else words


def _drop_impossible_rate_words(
    words: list[TranscriptWordPayload],
    duration_sec: float,
) -> list[TranscriptWordPayload]:
    """Drop runs of words that arrive faster than a human can utter them.

    Whisper-family models hallucinate whole sentences into a fraction of a
    second when fed music or noise — the observed case was 14 words inside
    0.28 s (~50 words/sec). Unlike the phrase blocklists this signal is
    language-agnostic, so it protects Tamil, Hindi and English alike.

    Deliberately conservative: a run must be at least
    ``TRANSCRIBE_HALLUCINATION_MIN_RUN`` words long, and if the rule wants to
    delete more than ``TRANSCRIBE_RATE_HALLUCINATION_MAX_DROP_RATIO`` of the
    transcript we assume the *timestamps* are broken globally (e.g. flat
    provider timings) rather than the text, and bail out entirely.
    """
    if not _env_bool("TRANSCRIBE_RATE_HALLUCINATION_FILTER", True):
        return words

    min_run = _env_int("TRANSCRIBE_HALLUCINATION_MIN_RUN", 5, 2)
    max_words_per_sec = _env_float("TRANSCRIBE_MAX_WORDS_PER_SEC", 9.0, 1.0)
    max_drop_ratio = _env_float("TRANSCRIBE_RATE_HALLUCINATION_MAX_DROP_RATIO", 0.5, 0.0)
    if len(words) < min_run:
        return words

    ordered = sorted(
        words, key=lambda item: (float(item.start_sec), float(item.end_sec))
    )
    drop_indices: set[int] = set()
    for start in range(len(ordered) - min_run + 1):
        end = start + min_run - 1
        span = float(ordered[end].end_sec) - float(ordered[start].start_sec)
        if span > 0 and (min_run / span) <= max_words_per_sec:
            continue
        drop_indices.update(range(start, end + 1))

    if not drop_indices:
        return words
    if len(drop_indices) > len(ordered) * max_drop_ratio:
        _perf_logger.warning(
            "[transcribe] impossible-rate rule matched %d/%d words; timestamps look "
            "globally broken, skipping the filter",
            len(drop_indices),
            len(ordered),
        )
        return words

    _perf_logger.info(
        "[transcribe] dropped %d word(s) exceeding %.1f words/sec: %s",
        len(drop_indices),
        max_words_per_sec,
        [ordered[idx].text for idx in sorted(drop_indices)],
    )
    filtered = [word for idx, word in enumerate(ordered) if idx not in drop_indices]
    return filtered if filtered else words


def _collapse_repetition_loops(
    words: list[TranscriptWordPayload],
) -> list[TranscriptWordPayload]:
    """Collapse a phrase that repeats back-to-back beyond a plausible limit.

    Songs legitimately repeat lines, so repetition alone is not evidence of a
    hallucination. What separates a decoder loop from a chorus is *timing*: a
    real chorus is separated by instrumental bars, while a loop re-emits the
    phrase immediately. Repeats therefore only count when consecutive blocks
    are within ``TRANSCRIBE_REPEAT_LOOP_MAX_GAP_SEC`` of each other.
    """
    if not _env_bool("TRANSCRIBE_REPEAT_LOOP_FILTER", True):
        return words

    max_repeats = _env_int("TRANSCRIBE_MAX_NGRAM_REPEATS", 2, 1)
    max_gap_sec = _env_float("TRANSCRIBE_REPEAT_LOOP_MAX_GAP_SEC", 0.6, 0.0)
    max_ngram = _env_int("TRANSCRIBE_REPEAT_LOOP_MAX_NGRAM", 6, 1)
    if len(words) < 2:
        return words

    ordered = sorted(
        words, key=lambda item: (float(item.start_sec), float(item.end_sec))
    )
    tokens = [_normalize_token(word.text) for word in ordered]
    total = len(ordered)
    drop_indices: set[int] = set()

    # Gap preceding each word. A repeated n-gram can start at any offset, so the
    # instrumental pause that distinguishes a chorus from a loop may land in the
    # middle of a block rather than on its seam. Every gap inside the candidate
    # run must therefore be tight, not just the seams between blocks.
    gaps = [0.0] + [
        max(0.0, float(ordered[i].start_sec) - float(ordered[i - 1].end_sec))
        for i in range(1, total)
    ]

    def _is_tight(lo: int, hi: int) -> bool:
        return all(gaps[k] <= max_gap_sec for k in range(lo + 1, hi))

    index = 0
    while index < total:
        matched = False
        widest = min(max_ngram, (total - index) // 2)
        for size in range(widest, 0, -1):
            base = tokens[index : index + size]
            if not all(base):
                continue
            if not _is_tight(index, index + size):
                continue
            repeats = 1
            cursor = index + size
            while cursor + size <= total and tokens[cursor : cursor + size] == base:
                if not _is_tight(cursor - 1, cursor + size):
                    break
                repeats += 1
                cursor += size
            if repeats > max_repeats:
                drop_indices.update(
                    range(index + size * max_repeats, index + size * repeats)
                )
                _perf_logger.info(
                    "[transcribe] collapsed a %d-word phrase repeated %dx: %r",
                    size,
                    repeats,
                    " ".join(base),
                )
                index += size * repeats
                matched = True
                break
        if not matched:
            index += 1

    if not drop_indices:
        return words
    filtered = [word for idx, word in enumerate(ordered) if idx not in drop_indices]
    return filtered if filtered else words


def _drop_words_in_nonvocal_regions(
    words: list[TranscriptWordPayload],
    duration_sec: float,
    *,
    audio_path: str | None,
) -> list[TranscriptWordPayload]:
    """Remove words that land inside long no-vocal spans.

    For music-heavy content, ASR can hallucinate isolated words during
    instrumental sections. Detect these spans on the prepared audio input and
    drop words whose midpoint clearly falls inside them.
    """
    if len(words) < 2 or duration_sec <= 0 or not audio_path:
        return words
    if _runtime_profile() not in {"music", "mixed"}:
        return words
    if not _env_bool("TRANSCRIBE_NONVOCAL_REGION_FILTER", True):
        return words

    min_nonvocal_sec = _env_float("TRANSCRIBE_NONVOCAL_REGION_MIN_SEC", 1.6, 0.4)
    noise_db = _env_float("TRANSCRIBE_NONVOCAL_REGION_NOISE_DB", -34.0, -80.0)
    guard_sec = _env_float("TRANSCRIBE_NONVOCAL_REGION_GUARD_SEC", 0.12, 0.0)
    max_regions = _env_int("TRANSCRIBE_NONVOCAL_REGION_MAX_SPANS", 48, 1)

    try:
        silence_ranges = detect_silence_ranges(
            audio_path,
            noise_db=noise_db,
            min_silence_sec=min_nonvocal_sec,
            max_duration_sec=duration_sec,
        )
    except Exception:  # noqa: BLE001
        return words

    if not silence_ranges:
        return words

    protected_ranges: list[tuple[float, float]] = []
    for start_sec, end_sec in silence_ranges[:max_regions]:
        start = max(0.0, min(float(start_sec), duration_sec))
        end = max(0.0, min(float(end_sec), duration_sec))
        if end - start < min_nonvocal_sec:
            continue
        inner_start = min(end, start + guard_sec)
        inner_end = max(inner_start, end - guard_sec)
        if inner_end - inner_start < max(0.2, min_nonvocal_sec * 0.35):
            continue
        protected_ranges.append((inner_start, inner_end))

    if not protected_ranges:
        return words

    filtered: list[TranscriptWordPayload] = []
    for word in words:
        midpoint = _word_midpoint_sec(word)
        if any(start <= midpoint <= end for start, end in protected_ranges):
            continue
        filtered.append(word)
    return filtered if filtered else words


def _trim_known_tail_hallucination(
    words: list[TranscriptWordPayload],
    duration_sec: float,
) -> list[TranscriptWordPayload]:
    if len(words) < 3 or duration_sec <= 0:
        return words
    if not _env_bool("TRANSCRIBE_TAIL_PHRASE_FILTER", True):
        return words

    min_gap_before_sec = _env_float("TRANSCRIBE_TAIL_PHRASE_MIN_GAP_SEC", 0.3, 0.0)
    max_phrase_span_sec = _env_float("TRANSCRIBE_TAIL_PHRASE_MAX_SPAN_SEC", 3.0, 0.1)
    min_remaining_words = _env_int("TRANSCRIBE_TAIL_PHRASE_MIN_REMAINING_WORDS", 3, 0)
    max_phrase_words = max(
        2, min(_env_int("TRANSCRIBE_TAIL_PHRASE_MAX_WORDS", 6, 2), 8)
    )

    normalized_tokens = [_normalize_token(item.text) for item in words]
    total = len(words)
    max_len = min(total, max_phrase_words)
    for phrase_len in range(max_len, 1, -1):
        start_idx = total - phrase_len
        phrase_tokens = tuple(normalized_tokens[start_idx:])
        if any(not token for token in phrase_tokens):
            continue
        if phrase_tokens not in _TAIL_HALLUCINATION_PHRASES:
            continue
        if start_idx < min_remaining_words:
            continue

        phrase_start = float(words[start_idx].start_sec)
        phrase_end = float(words[-1].end_sec)
        phrase_span = max(0.0, phrase_end - phrase_start)
        if phrase_span > max_phrase_span_sec:
            continue

        previous_end = float(words[start_idx - 1].end_sec) if start_idx > 0 else 0.0
        gap_before = max(0.0, phrase_start - previous_end)
        if gap_before < min_gap_before_sec:
            continue

        trimmed = words[:start_idx]
        return trimmed if trimmed else words

    return words


def _trim_sparse_music_tail(
    words: list[TranscriptWordPayload],
    duration_sec: float,
    *,
    force: bool = False,
) -> list[TranscriptWordPayload]:
    if len(words) < 3 or duration_sec <= 0:
        return words
    if not force and _runtime_profile() not in {"music", "mixed"}:
        return words
    if not _env_bool("TRANSCRIBE_SPARSE_MUSIC_TAIL_FILTER", True):
        return words

    max_tail_words = _env_int("TRANSCRIBE_SPARSE_MUSIC_TAIL_MAX_WORDS", 6, 1)
    max_tail_span_sec = _env_float(
        "TRANSCRIBE_SPARSE_MUSIC_TAIL_MAX_SPAN_SEC", 4.0, 0.2
    )
    default_tail_gap_sec = 4.0 if duration_sec <= 30.0 else 2.5
    min_tail_gap_sec = _env_float(
        "TRANSCRIBE_SPARSE_MUSIC_TAIL_MIN_GAP_SEC", default_tail_gap_sec, 0.5
    )
    if duration_sec <= 30.0:
        min_tail_gap_sec = max(min_tail_gap_sec, default_tail_gap_sec)
    min_prev_gap_sec = _env_float("TRANSCRIBE_SPARSE_MUSIC_TAIL_PREV_GAP_SEC", 1.2, 0.0)
    cluster_gap_sec = _env_float(
        "TRANSCRIBE_SPARSE_MUSIC_TAIL_CLUSTER_GAP_SEC", 0.85, 0.05
    )
    max_avg_confidence = _env_float(
        "TRANSCRIBE_SPARSE_MUSIC_TAIL_MAX_CONFIDENCE", 0.72, 0.0
    )

    tail_gap_sec = max(0.0, duration_sec - float(words[-1].end_sec))
    if tail_gap_sec < min_tail_gap_sec:
        return words

    start_idx = len(words) - 1
    while start_idx > 0:
        gap_sec = max(
            0.0, float(words[start_idx].start_sec) - float(words[start_idx - 1].end_sec)
        )
        if gap_sec > cluster_gap_sec:
            break
        start_idx -= 1

    tail_cluster = words[start_idx:]
    if len(tail_cluster) > max_tail_words:
        return words

    cluster_start = float(tail_cluster[0].start_sec)
    cluster_end = float(tail_cluster[-1].end_sec)
    if (cluster_end - cluster_start) > max_tail_span_sec:
        return words

    prev_gap_sec = (
        cluster_start
        if start_idx == 0
        else max(0.0, cluster_start - float(words[start_idx - 1].end_sec))
    )
    if prev_gap_sec < min_prev_gap_sec:
        return words

    confidences = [
        float(item.confidence) for item in tail_cluster if item.confidence is not None
    ]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence > max_avg_confidence:
            return words

    trimmed = words[:start_idx]
    return trimmed if trimmed else words


def trim_songlike_tail_hallucination(
    payload: TranscriptPayload,
    *,
    duration_sec: float,
) -> TranscriptPayload:
    if payload.is_mock or not payload.words:
        return payload

    trimmed_words = _trim_sparse_music_tail(
        payload.words,
        duration_sec,
        force=True,
    )
    if len(trimmed_words) == len(payload.words):
        return payload

    return TranscriptPayload(
        source=payload.source,
        language=payload.language,
        text=" ".join(word.text for word in trimmed_words).strip(),
        words=trimmed_words,
        is_mock=payload.is_mock,
    )


def trim_song_mode_to_manual_lyrics_span(
    payload: TranscriptPayload,
) -> TranscriptPayload:
    if payload.is_mock or not payload.words:
        return payload
    if not payload.source.endswith("_lyrics_ref"):
        return payload

    manual_indices = [
        index
        for index, word in enumerate(payload.words)
        if _normalize_source_pass(word.source_pass) == "manual"
    ]
    if not manual_indices:
        return payload

    start_index = manual_indices[0]
    end_index = manual_indices[-1] + 1
    trimmed_words = payload.words[start_index:end_index]
    if len(trimmed_words) == len(payload.words):
        return payload

    return TranscriptPayload(
        source=payload.source,
        language=payload.language,
        text=" ".join(word.text for word in trimmed_words).strip(),
        words=trimmed_words,
        is_mock=payload.is_mock,
    )


def _trim_known_head_hallucination(
    words: list[TranscriptWordPayload],
) -> list[TranscriptWordPayload]:
    """Remove known hallucination phrases from the START of transcript.

    Common YouTube intro/outro phrases like "Thank you for watching" often appear
    as hallucinations at the beginning of music transcripts. This function removes
    them if they appear within the first few seconds and are followed by a gap.
    """
    if len(words) < 3:
        return words
    if not _env_bool("TRANSCRIBE_HEAD_PHRASE_FILTER", True):
        return words

    min_gap_after_sec = _env_float("TRANSCRIBE_HEAD_PHRASE_MIN_GAP_SEC", 0.2, 0.0)
    max_phrase_span_sec = _env_float("TRANSCRIBE_HEAD_PHRASE_MAX_SPAN_SEC", 15.0, 0.1)
    max_phrase_start_sec = _env_float("TRANSCRIBE_HEAD_PHRASE_MAX_START_SEC", 20.0, 0.0)
    min_remaining_words = _env_int("TRANSCRIBE_HEAD_PHRASE_MIN_REMAINING_WORDS", 5, 0)
    max_phrase_words = max(
        2, min(_env_int("TRANSCRIBE_HEAD_PHRASE_MAX_WORDS", 7, 2), 10)
    )

    # Only process if first word starts within the max start window
    first_start = float(words[0].start_sec)
    if first_start > max_phrase_start_sec:
        return words

    normalized_tokens = [_normalize_token(item.text) for item in words]
    total = len(words)

    # Try matching phrases of different lengths starting from longest
    for phrase_len in range(min(max_phrase_words, total), 1, -1):
        phrase_tokens = tuple(normalized_tokens[:phrase_len])
        if any(not token for token in phrase_tokens):
            continue
        if phrase_tokens not in _HEAD_HALLUCINATION_PHRASES:
            continue

        remaining = total - phrase_len
        if remaining < min_remaining_words:
            continue

        phrase_start = float(words[0].start_sec)
        phrase_end = float(words[phrase_len - 1].end_sec)
        phrase_span = max(0.0, phrase_end - phrase_start)
        if phrase_span > max_phrase_span_sec:
            continue

        # Check for gap after the phrase
        next_start = float(words[phrase_len].start_sec)
        gap_after = max(0.0, next_start - phrase_end)
        if gap_after < min_gap_after_sec:
            continue

        # Remove the head hallucination phrase
        trimmed = words[phrase_len:]
        return trimmed if trimmed else words

    return words


def _trim_prompt_leakage_phrases(
    words: list[TranscriptWordPayload],
) -> list[TranscriptWordPayload]:
    if len(words) < 2:
        return words
    if not _env_bool("TRANSCRIBE_PROMPT_LEAKAGE_FILTER", True):
        return words
    if _runtime_profile() not in {"music", "mixed"}:
        return words

    normalized_tokens = [_normalize_token(item.text) for item in words]
    kept: list[TranscriptWordPayload] = []
    index = 0
    while index < len(words):
        matched = False
        for phrase in _PROMPT_LEAKAGE_PHRASES:
            phrase_len = len(phrase)
            if index + phrase_len > len(words):
                continue
            if tuple(normalized_tokens[index : index + phrase_len]) == phrase:
                index += phrase_len
                matched = True
                break
        if matched:
            continue
        token = normalized_tokens[index]
        if token in _PROMPT_LEAKAGE_SINGLETONS:
            index += 1
            continue
        kept.append(words[index])
        index += 1

    return kept if kept else words


def _remove_hallucination_phrases_in_silence_gaps(
    words: list[TranscriptWordPayload],
    duration_sec: float,
) -> list[TranscriptWordPayload]:
    """Remove known hallucination phrases that appear *anywhere* in the transcript.

    Whisper / Groq commonly hallucinates YouTube-style sign-off phrases like
    "thank you for watching" or "please subscribe" during silence or music
    transitions in the *middle* of a recording — not just at the head / tail.
    This function scans every position and removes a match when the phrase is
    clearly isolated by silence gaps on both sides, making it very unlikely to
    be real speech.

    The filter is conservative by default:
      • requires a gap of at least ``TRANSCRIBE_ANYWHERE_PHRASE_MIN_GAP_SEC``
        (default 0.8 s) before the phrase AND after it (or near the
        start/end of the audio).
      • the entire phrase must fit within ``TRANSCRIBE_ANYWHERE_PHRASE_MAX_SPAN_SEC``
        (default 3.5 s).
      • at least ``TRANSCRIBE_ANYWHERE_PHRASE_MIN_REMAINING_WORDS`` (default 3)
        real words must remain outside the detected phrase cluster.
    """
    if len(words) < 3 or duration_sec <= 0:
        return words
    if not _env_bool("TRANSCRIBE_ANYWHERE_PHRASE_FILTER", True):
        return words

    min_gap_sec = _env_float("TRANSCRIBE_ANYWHERE_PHRASE_MIN_GAP_SEC", 0.8, 0.2)
    max_phrase_span_sec = _env_float(
        "TRANSCRIBE_ANYWHERE_PHRASE_MAX_SPAN_SEC", 3.5, 0.5
    )
    max_phrase_words = max(
        2, min(_env_int("TRANSCRIBE_ANYWHERE_PHRASE_MAX_WORDS", 8, 2), 12)
    )
    min_remaining_words = _env_int(
        "TRANSCRIBE_ANYWHERE_PHRASE_MIN_REMAINING_WORDS", 3, 0
    )

    normalized_tokens = [_normalize_token(item.text) for item in words]
    drop_indices: set[int] = set()
    total = len(words)

    for i in range(total):
        if i in drop_indices:
            continue

        # Gap before this word
        if i == 0:
            gap_before = float(words[0].start_sec)
        else:
            gap_before = max(
                0.0, float(words[i].start_sec) - float(words[i - 1].end_sec)
            )

        # Require a meaningful silence before the phrase so we don’t accidentally
        # clip real speech that happens to match (e.g., genuine gratitude mid-talk).
        if gap_before < min_gap_sec:
            continue

        # Check single-word singletons first (e.g., isolated "subscribe")
        singleton_token = normalized_tokens[i]
        if singleton_token in _HALLUCINATION_SINGLETONS_IN_GAPS:
            # Also need a gap after
            if i + 1 < total:
                gap_after = max(
                    0.0, float(words[i + 1].start_sec) - float(words[i].end_sec)
                )
            else:
                gap_after = max(0.0, duration_sec - float(words[i].end_sec))
            if gap_after >= min_gap_sec:
                remaining = total - 1  # would remove 1 word
                if remaining >= min_remaining_words:
                    drop_indices.add(i)
                    continue

        # Try matching multi-word phrases from longest to shortest
        matched_phrase_len = 0
        for phrase_len in range(min(max_phrase_words, total - i), 1, -1):
            phrase_tokens = tuple(normalized_tokens[i : i + phrase_len])
            if any(not token for token in phrase_tokens):
                continue
            if phrase_tokens not in _ANYWHERE_HALLUCINATION_PHRASES:
                continue

            phrase_start = float(words[i].start_sec)
            phrase_end = float(words[i + phrase_len - 1].end_sec)
            if (phrase_end - phrase_start) > max_phrase_span_sec:
                continue

            last_idx = i + phrase_len - 1
            if last_idx + 1 < total:
                gap_after = max(
                    0.0,
                    float(words[last_idx + 1].start_sec) - phrase_end,
                )
                near_end = (duration_sec - phrase_end) <= min_gap_sec * 1.5
            else:
                gap_after = max(0.0, duration_sec - phrase_end)
                near_end = True

            # Both sides must be silent (or phrase is near start/end of file)
            if gap_after < min_gap_sec and not near_end:
                continue

            remaining = total - phrase_len
            if remaining < min_remaining_words:
                continue

            matched_phrase_len = phrase_len
            break

        if matched_phrase_len:
            for offset in range(matched_phrase_len):
                drop_indices.add(i + offset)

    if not drop_indices:
        return words

    dropped_texts = [words[idx].text for idx in sorted(drop_indices)]
    _perf_logger.debug(
        "Hallucination anywhere-filter removed %d word(s): %s",
        len(drop_indices),
        dropped_texts,
    )
    filtered = [word for idx, word in enumerate(words) if idx not in drop_indices]
    return filtered if filtered else words


def _drop_sparse_words_in_music_gaps(
    words: list[TranscriptWordPayload],
    duration_sec: float,
) -> list[TranscriptWordPayload]:
    """Remove sparse isolated words that appear in low-density regions of songs.

    Songs typically have dense clusters of lyrics (verses, chorus) separated by
    instrumental breaks. Whisper/Groq often hallucinates random words during
    these breaks. This filter computes the local word density and drops words
    in regions that are dramatically sparser than the densest sections — those
    words are almost certainly hallucinations in instrumental gaps.
    """
    if len(words) < 6 or duration_sec <= 0:
        return words
    profile = _runtime_profile()
    mode = _runtime_mode()
    if profile not in {"music", "mixed"} and mode != "song":
        return words
    if not _env_bool("TRANSCRIBE_MUSIC_GAP_DENSITY_FILTER", True):
        return words

    density_window_sec = _env_float(
        "TRANSCRIBE_MUSIC_GAP_DENSITY_WINDOW_SEC", 6.0, 2.0
    )
    density_ratio_threshold = _env_float(
        "TRANSCRIBE_MUSIC_GAP_DENSITY_RATIO", 0.15, 0.01
    )
    min_dense_cluster_words = _env_int(
        "TRANSCRIBE_MUSIC_GAP_MIN_DENSE_WORDS", 6, 2
    )
    max_sparse_cluster_words = _env_int(
        "TRANSCRIBE_MUSIC_GAP_MAX_SPARSE_WORDS", 4, 1
    )
    min_gap_around_sec = _env_float(
        "TRANSCRIBE_MUSIC_GAP_MIN_SURROUNDING_GAP_SEC", 1.5, 0.3
    )

    # Step 1: compute per-word local density (words per second in surrounding window)
    word_densities: list[float] = []
    for i, word in enumerate(words):
        center = (float(word.start_sec) + float(word.end_sec)) * 0.5
        window_start = max(0.0, center - density_window_sec * 0.5)
        window_end = min(duration_sec, center + density_window_sec * 0.5)
        window_span = max(window_end - window_start, 0.5)
        count_in_window = sum(
            1 for w in words
            if float(w.end_sec) > window_start and float(w.start_sec) < window_end
        )
        word_densities.append(count_in_window / window_span)

    # Step 2: find the peak density (representative of actual lyrics sections)
    if not word_densities:
        return words
    peak_density = max(word_densities)
    if peak_density < 0.5:  # If peak is very low, entire track may be sparse — skip
        return words

    # Step 3: identify words in sparse regions below the threshold
    drop_indices: set[int] = set()
    i = 0
    while i < len(words):
        if word_densities[i] >= peak_density * density_ratio_threshold:
            i += 1
            continue

        # Start a sparse cluster
        cluster = [i]
        j = i + 1
        while j < len(words) and word_densities[j] < peak_density * density_ratio_threshold:
            cluster.append(j)
            j += 1

        # Only drop small clusters (large ones might be actual quiet vocal sections)
        if len(cluster) <= max_sparse_cluster_words:
            cluster_start = float(words[cluster[0]].start_sec)
            cluster_end = float(words[cluster[-1]].end_sec)

            gap_before = (
                cluster_start
                if cluster[0] == 0
                else max(0.0, cluster_start - float(words[cluster[0] - 1].end_sec))
            )
            gap_after = (
                max(0.0, duration_sec - cluster_end)
                if cluster[-1] == len(words) - 1
                else max(0.0, float(words[cluster[-1] + 1].start_sec) - cluster_end)
            )

            if gap_before >= min_gap_around_sec or gap_after >= min_gap_around_sec:
                drop_indices.update(cluster)

        i = j

    if not drop_indices:
        return words

    _perf_logger.debug(
        "Music gap density filter removed %d word(s): %s",
        len(drop_indices),
        [words[idx].text for idx in sorted(drop_indices)],
    )
    filtered = [word for idx, word in enumerate(words) if idx not in drop_indices]
    return filtered if filtered else words


def _apply_word_filters(
    words: list[TranscriptWordPayload],
    duration_sec: float,
    *,
    audio_path: str | None = None,
) -> list[TranscriptWordPayload]:
    filtered = list(words)
    # Language-agnostic rules run first: the phrase blocklists below only know
    # English sign-offs, so novel hallucinations have to be caught by shape.
    filtered = _drop_impossible_rate_words(filtered, duration_sec)
    filtered = _collapse_repetition_loops(filtered)
    if _env_bool("TRANSCRIBE_HALLUCINATION_FILTER", True):
        filtered = _detect_hallucinations(filtered)
        filtered = _detect_sparse_hallucinations(filtered, duration_sec)
        filtered = _drop_words_in_nonvocal_regions(
            filtered, duration_sec, audio_path=audio_path
        )
        # Song/music-specific: remove isolated words in low-density instrumental gaps
        filtered = _drop_sparse_words_in_music_gaps(filtered, duration_sec)
    filtered = _trim_prompt_leakage_phrases(filtered)
    # Remove hallucination phrases from both head and tail of transcript
    filtered = _trim_known_head_hallucination(filtered)
    filtered = _trim_known_tail_hallucination(filtered, duration_sec)
    filtered = _trim_sparse_music_tail(filtered, duration_sec)
    # Remove known hallucination phrases that appear anywhere in the transcript
    # (e.g., "thank you for watching" mid-video during a silence gap).
    filtered = _remove_hallucination_phrases_in_silence_gaps(filtered, duration_sec)
    return filtered


def _song_validator_support_flags(
    words: list[TranscriptWordPayload],
    validator_words: list[TranscriptWordPayload],
    *,
    time_pad_sec: float,
) -> list[bool]:
    validator_index: dict[str, list[TranscriptWordPayload]] = {}
    for validator_word in validator_words:
        token = _normalize_token(validator_word.text)
        if not token:
            continue
        validator_index.setdefault(token, []).append(validator_word)

    supported: list[bool] = []
    for word in words:
        token = _normalize_token(word.text)
        if not token:
            supported.append(True)
            continue
        midpoint = _word_midpoint_sec(word)
        matches = validator_index.get(token, [])
        is_supported = any(
            abs(_word_midpoint_sec(candidate) - midpoint) <= time_pad_sec
            or (
                float(candidate.start_sec) <= (float(word.end_sec) + time_pad_sec)
                and float(candidate.end_sec) >= (float(word.start_sec) - time_pad_sec)
            )
            for candidate in matches
        )
        supported.append(is_supported)
    return supported


def _stabilize_song_words_with_validator(
    words: list[TranscriptWordPayload],
    validator_words: list[TranscriptWordPayload],
    duration_sec: float,
) -> list[TranscriptWordPayload]:
    if len(words) < 4 or len(validator_words) < 2:
        return words

    time_pad_sec = _env_float("TRANSCRIBE_SONG_VALIDATION_TIME_PAD_SEC", 0.55, 0.1)
    cluster_gap_sec = _env_float(
        "TRANSCRIBE_SONG_VALIDATION_CLUSTER_GAP_SEC", 0.55, 0.05
    )
    boundary_gap_sec = _env_float(
        "TRANSCRIBE_SONG_VALIDATION_BOUNDARY_GAP_SEC", 0.9, 0.1
    )
    max_cluster_words = _env_int("TRANSCRIBE_SONG_VALIDATION_MAX_CLUSTER_WORDS", 8, 1)
    max_cluster_span_sec = _env_float(
        "TRANSCRIBE_SONG_VALIDATION_MAX_CLUSTER_SPAN_SEC", 2.4, 0.2
    )
    edge_window_sec = _env_float(
        "TRANSCRIBE_SONG_VALIDATION_EDGE_WINDOW_SEC", 18.0, 1.0
    )

    exact_supported = _song_validator_support_flags(
        words,
        validator_words,
        time_pad_sec=time_pad_sec,
    )

    soft_supported = list(exact_supported)
    for index, word in enumerate(words):
        if soft_supported[index]:
            continue
        token = _normalize_token(word.text)
        if not token or token not in _HALLUCINATION_STOPWORDS:
            continue
        if (index > 0 and exact_supported[index - 1]) or (
            index + 1 < len(words) and exact_supported[index + 1]
        ):
            soft_supported[index] = True

    validator_midpoints = [_word_midpoint_sec(word) for word in validator_words]
    drop_indices: set[int] = set()
    index = 0
    while index < len(words):
        if soft_supported[index]:
            index += 1
            continue
        cluster = [index]
        next_index = index + 1
        while next_index < len(words):
            gap_sec = max(
                0.0,
                float(words[next_index].start_sec)
                - float(words[next_index - 1].end_sec),
            )
            if soft_supported[next_index] or gap_sec > cluster_gap_sec:
                break
            cluster.append(next_index)
            next_index += 1

        cluster_start = float(words[cluster[0]].start_sec)
        cluster_end = float(words[cluster[-1]].end_sec)
        cluster_span = max(0.0, cluster_end - cluster_start)
        prev_gap = (
            cluster_start
            if cluster[0] == 0
            else max(0.0, cluster_start - float(words[cluster[0] - 1].end_sec))
        )
        next_gap = (
            max(0.0, duration_sec - cluster_end)
            if cluster[-1] == len(words) - 1
            else max(0.0, float(words[cluster[-1] + 1].start_sec) - cluster_end)
        )
        at_edge = (
            cluster_start <= edge_window_sec
            or (duration_sec - cluster_end) <= edge_window_sec
        )
        surrounded_by_gap = prev_gap >= boundary_gap_sec or next_gap >= boundary_gap_sec
        validator_nearby = any(
            (cluster_start - time_pad_sec) <= midpoint <= (cluster_end + time_pad_sec)
            for midpoint in validator_midpoints
        )

        if (
            len(cluster) <= max_cluster_words
            and cluster_span <= max_cluster_span_sec
            and (at_edge or surrounded_by_gap or not validator_nearby)
        ):
            drop_indices.update(cluster)

        index = next_index

    if not drop_indices:
        return words

    filtered = [word for idx, word in enumerate(words) if idx not in drop_indices]
    return filtered if filtered else words
