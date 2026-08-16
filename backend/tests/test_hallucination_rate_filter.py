"""Language-agnostic hallucination detection.

The existing hallucination filters are exact-phrase blocklists of English
YouTube sign-offs, so a novel hallucination sails straight through. These tests
pin the two signals that generalize across languages:

  * physically impossible word rate (a decoder dumping a whole sentence into a
    fraction of a second), and
  * runaway repetition loops.

The fixture is the real payload captured from the Tamil song "Thaensudare"
(job bced16fe-…), which contained both.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")

from app import _transcription_filters as tf
from app._transcription_payloads import TranscriptWordPayload

_FIXTURE = Path(__file__).parent / "fixtures" / "tamil_song_hallucination.json"


def _load_fixture() -> tuple[list[TranscriptWordPayload], float]:
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    words = [
        TranscriptWordPayload(
            id=item["id"],
            text=item["text"],
            start_sec=float(item["start_sec"]),
            end_sec=float(item["end_sec"]),
            confidence=item.get("confidence"),
            source_pass=item.get("source_pass"),
        )
        for item in raw["words"]
    ]
    return words, float(raw["duration_sec"])


def _build(spec: list[tuple[str, float, float]]) -> list[TranscriptWordPayload]:
    return [
        TranscriptWordPayload(
            id=f"w{index}", text=text, start_sec=start, end_sec=end, source_pass="primary"
        )
        for index, (text, start, end) in enumerate(spec)
    ]


def _texts(words: list[TranscriptWordPayload]) -> str:
    return " ".join(word.text for word in words).lower()


# ---------------------------------------------------------------------------
# Impossible word rate
# ---------------------------------------------------------------------------


def test_impossible_rate_burst_is_removed_from_real_payload():
    """14 words in 0.28s (~50 words/sec) is not physically utterable."""
    words, duration = _load_fixture()
    assert "address on the screen" in _texts(words), "fixture lost its hallucination"

    filtered = tf._drop_impossible_rate_words(words, duration)

    assert "address on the screen" not in _texts(filtered)
    assert "if you have any questions" not in _texts(filtered)


def test_normal_speech_rate_is_preserved():
    """~3 words/sec is ordinary speech and must survive untouched."""
    words = _build(
        [(f"word{i}", i * 0.33, i * 0.33 + 0.3) for i in range(12)]
    )

    filtered = tf._drop_impossible_rate_words(words, 6.0)

    assert len(filtered) == len(words)


def test_fast_but_plausible_speech_is_preserved():
    """8 words/sec is fast auctioneer-grade speech, still real."""
    words = _build([(f"w{i}", i * 0.125, i * 0.125 + 0.12) for i in range(16)])

    filtered = tf._drop_impossible_rate_words(words, 2.0)

    assert len(filtered) == len(words)


def test_short_burst_below_min_run_is_kept():
    """Two crammed words are a timestamp glitch, not a hallucinated sentence."""
    words = _build(
        [("hello", 0.0, 0.5), ("there", 0.5, 1.0), ("ok", 1.0, 1.01), ("yes", 1.01, 1.02)]
    )

    filtered = tf._drop_impossible_rate_words(words, 5.0)

    assert len(filtered) == len(words)


def test_rate_filter_never_empties_the_transcript():
    """A pathological all-fast transcript degrades to a no-op, not to nothing."""
    words = _build([(f"w{i}", i * 0.01, i * 0.01 + 0.005) for i in range(40)])

    filtered = tf._drop_impossible_rate_words(words, 1.0)

    assert filtered, "filter must not delete the entire transcript"


# ---------------------------------------------------------------------------
# Repetition loops
# ---------------------------------------------------------------------------


def test_repetition_loop_is_collapsed_in_real_payload():
    """'I am a dream.' x3 back-to-back is a decoder loop."""
    words, _ = _load_fixture()
    assert _texts(words).count("i am a dream") == 3

    filtered = tf._collapse_repetition_loops(words)

    assert _texts(filtered).count("i am a dream") <= 2


def test_spaced_chorus_repetition_is_preserved():
    """A real chorus repeats across instrumental bars and must be kept."""
    spec: list[tuple[str, float, float]] = []
    cursor = 0.0
    for _ in range(4):
        for token in ("thean", "sudare", "en", "uyire"):
            spec.append((token, cursor, cursor + 0.4))
            cursor += 0.5
        cursor += 3.0  # instrumental gap between chorus repeats
    words = _build(spec)

    filtered = tf._collapse_repetition_loops(words)

    assert len(filtered) == len(words), "spaced chorus must survive"


def test_two_adjacent_repeats_are_allowed():
    """Doubling a line is a normal lyrical device."""
    spec: list[tuple[str, float, float]] = []
    cursor = 0.0
    for _ in range(2):
        for token in ("vaa", "vaa", "nilave"):
            spec.append((token, cursor, cursor + 0.3))
            cursor += 0.4
    words = _build(spec)

    filtered = tf._collapse_repetition_loops(words)

    assert len(filtered) == len(words)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_apply_word_filters_removes_the_impossible_burst(monkeypatch):
    """The new rules must be reachable from the real filter entry point."""
    monkeypatch.setenv("TRANSCRIBE_HALLUCINATION_FILTER", "true")
    words, duration = _load_fixture()

    filtered = tf._apply_word_filters(words, duration)

    assert "address on the screen" not in _texts(filtered)


@pytest.mark.parametrize("flag", ["false", "0"])
def test_rate_filter_can_be_disabled(monkeypatch, flag):
    monkeypatch.setenv("TRANSCRIBE_RATE_HALLUCINATION_FILTER", flag)
    words, duration = _load_fixture()

    filtered = tf._drop_impossible_rate_words(words, duration)

    assert len(filtered) == len(words)


# ---------------------------------------------------------------------------
# Quality scoring must measure something, not default to "trusted"
# ---------------------------------------------------------------------------


def test_quality_scoring_marks_impossible_rate_words_weak():
    """Groq returns confidence=None, so rate is the only signal available."""
    from app.routers._transcript_format import _annotate_word_quality

    words, duration = _load_fixture()
    burst = [w for w in words if 108.0 <= w.start_sec <= 110.0]
    assert len(burst) >= 5, "fixture lost the impossible-rate burst"

    annotated = _annotate_word_quality(words, duration)
    burst_ids = {w.id for w in burst}
    burst_scored = [w for w in annotated if w.id in burst_ids]

    assert burst_scored
    assert all(w.quality_label == "weak" for w in burst_scored), [
        (w.text, w.quality_score, w.quality_label) for w in burst_scored
    ]


def test_quality_scoring_keeps_normal_speech_trusted():
    from app.routers._transcript_format import _annotate_word_quality

    words = _build([(f"w{i}", i * 0.4, i * 0.4 + 0.35) for i in range(10)])

    annotated = _annotate_word_quality(words, 4.0)

    assert all(w.quality_label == "trusted" for w in annotated)


def test_quality_scoring_is_not_a_flat_constant():
    """Regression: every word used to score exactly 0.88 regardless of content."""
    from app.routers._transcript_format import _annotate_word_quality

    words, duration = _load_fixture()

    scores = {w.quality_score for w in _annotate_word_quality(words, duration)}

    assert len(scores) > 1, f"quality score is still a constant: {scores}"


# ---------------------------------------------------------------------------
# Vocal isolation degradation must be visible
# ---------------------------------------------------------------------------


def test_vocal_isolation_model_dir_is_not_under_tmp():
    """/tmp is cleared on reboot, which silently re-breaks isolation."""
    from app import transcription_service as ts

    assert not ts._vocal_isolation_model_dir().startswith("/tmp/")


def test_vocal_isolation_model_dir_is_configurable(monkeypatch):
    from app import transcription_service as ts

    monkeypatch.setenv("TRANSCRIBE_VOCAL_ISOLATION_MODEL_DIR", "/opt/models")

    assert ts._vocal_isolation_model_dir() == "/opt/models"


def test_unavailable_isolation_produces_a_warning():
    from app import transcription_service as ts

    ts._set_vocal_isolation_status("unavailable")
    warning = ts.consume_vocal_isolation_warning()

    assert warning is not None
    assert "background music" in warning


def test_isolation_warning_is_one_shot():
    from app import transcription_service as ts

    ts._set_vocal_isolation_status("unavailable")
    assert ts.consume_vocal_isolation_warning() is not None
    assert ts.consume_vocal_isolation_warning() is None


def test_successful_isolation_produces_no_warning():
    from app import transcription_service as ts

    ts._set_vocal_isolation_status("ok")

    assert ts.consume_vocal_isolation_warning() is None


def test_disabled_isolation_produces_no_warning():
    from app import transcription_service as ts

    ts._set_vocal_isolation_status("disabled")

    assert ts.consume_vocal_isolation_warning() is None
