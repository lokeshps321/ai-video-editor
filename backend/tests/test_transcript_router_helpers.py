from __future__ import annotations

import os

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")

from app.routers import transcript as transcript_router
from app.schemas import Clip, TimelineState, Track
from app.transcription_service import TranscriptPayload, TranscriptWordPayload


def test_full_range_transcript_generation_keeps_the_existing_v1_clip() -> None:
    clip = Clip(
        id="v1-original",
        asset_id="asset-v1",
        start_sec=0.0,
        end_sec=12.5,
        timeline_start_sec=0.0,
    )
    state = TimelineState(
        tracks=[Track(id="video-track", kind="video", clips=[clip])],
        duration_sec=12.5,
    )

    assert transcript_router._video_track_already_matches_ranges(
        state,
        asset_id="asset-v1",
        ranges=[{"start_sec": 0.0, "end_sec": 12.5}],
    )


def test_video_range_match_requires_a_real_no_op() -> None:
    state = TimelineState(
        tracks=[
            Track(
                id="video-track",
                kind="video",
                clips=[
                    Clip(
                        id="v1-trimmed",
                        asset_id="asset-v1",
                        start_sec=0.0,
                        end_sec=6.0,
                        timeline_start_sec=0.0,
                    )
                ],
            ),
            Track(
                id="audio-track",
                kind="audio",
                clips=[
                    Clip(
                        id="music",
                        asset_id="asset-a1",
                        start_sec=0.0,
                        end_sec=6.0,
                        timeline_start_sec=0.0,
                    )
                ],
            ),
        ],
        duration_sec=6.0,
    )

    assert not transcript_router._video_track_already_matches_ranges(
        state,
        asset_id="asset-v1",
        ranges=[{"start_sec": 0.0, "end_sec": 6.0}],
    )


def test_materialize_transcript_items_exposes_blanked_regions() -> None:
    items = [
        {"id": "w1", "text": "hello", "start_sec": 0.5, "end_sec": 0.9, "source_pass": "primary"},
        {"id": "blank-1", "text": "", "start_sec": 4.0, "end_sec": 5.0, "blanked": True, "kind": "blank_region"},
        {"id": "w2", "text": "world", "start_sec": 8.0, "end_sec": 8.4, "source_pass": "primary"},
    ]

    _stored, words, text, regions = transcript_router._materialize_transcript_items(items, 12.0)

    assert text == "hello world"
    assert [word.text for word in words] == ["hello", "world"]
    assert any(region.status == "blanked" for region in regions)


def test_summarize_transcript_quality_flags_weak_regions_for_review() -> None:
    items = [
        {"id": "w1", "text": "clean", "start_sec": 0.1, "end_sec": 0.4, "confidence": 0.95, "source_pass": "primary"},
        {"id": "w2", "text": "unclear", "start_sec": 0.5, "end_sec": 0.9, "source_pass": "rescue"},
        {"id": "w3", "text": "ending", "start_sec": 1.0, "end_sec": 1.3, "confidence": 0.94, "source_pass": "primary"},
    ]

    _stored, words, _text, regions = transcript_router._materialize_transcript_items(items, 4.0)
    quality_score, quality_label, weak_word_count, weak_word_ratio, issue_region_count = (
        transcript_router._summarize_transcript_quality(words, regions)
    )

    assert quality_label == "needs_review"
    assert weak_word_count >= 1
    assert weak_word_ratio > 0
    assert issue_region_count >= 1
    assert 0.0 <= quality_score <= 1.0


def test_apply_range_update_items_replace_blank_preserve_and_delete() -> None:
    items = [
        {"id": "w1", "text": "lose", "start_sec": 0.0, "end_sec": 0.3, "source_pass": "primary"},
        {"id": "w2", "text": "my", "start_sec": 0.3, "end_sec": 0.6, "source_pass": "primary"},
        {"id": "w3", "text": "mind", "start_sec": 0.6, "end_sec": 0.9, "source_pass": "primary"},
    ]

    replaced = transcript_router._apply_range_update_items(
        items,
        duration_sec=3.0,
        start_word_id="w1",
        end_word_id="w3",
        mode="replace",
        text="keep my soul",
    )
    _stored_replace, words_replace, text_replace, _regions_replace = transcript_router._materialize_transcript_items(replaced, 3.0)
    assert text_replace == "keep my soul"
    assert [word.text for word in words_replace] == ["keep", "my", "soul"]

    preserved = transcript_router._apply_range_update_items(
        replaced,
        duration_sec=3.0,
        start_word_id="w1",
        end_word_id="w3",
        mode="preserve",
        text=None,
    )
    _stored_preserve, words_preserve, _text_preserve, _regions_preserve = transcript_router._materialize_transcript_items(preserved, 3.0)
    assert all(word.quality_label == "trusted" for word in words_preserve)
    assert all(word.source_pass == "manual" for word in words_preserve)

    blanked = transcript_router._apply_range_update_items(
        preserved,
        duration_sec=3.0,
        start_word_id="w1",
        end_word_id="w3",
        mode="blank",
        text=None,
    )
    _stored_blank, words_blank, text_blank, regions_blank = transcript_router._materialize_transcript_items(blanked, 3.0)
    assert words_blank == []
    assert text_blank == ""
    assert any(region.status == "blanked" for region in regions_blank)

    deleted = transcript_router._apply_range_update_items(
        items,
        duration_sec=3.0,
        start_word_id="w2",
        end_word_id="w2",
        mode="delete",
        text=None,
    )
    _stored_delete, words_delete, text_delete, regions_delete = transcript_router._materialize_transcript_items(deleted, 3.0)
    assert [word.text for word in words_delete] == ["lose", "mind"]
    assert text_delete == "lose mind"
    assert not any(region.status == "blanked" for region in regions_delete)


def test_retry_weak_regions_in_items_replaces_region_with_better_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("TRANSCRIPT_WEAK_REGION_RETRY_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIPT_WEAK_RETRY_PAD_SEC", "0.2")
    monkeypatch.setenv("TRANSCRIPT_WEAK_RETRY_MAX_REGIONS", "2")

    items = [
        {"id": "w1", "text": "hello", "start_sec": 0.2, "end_sec": 0.45, "source_pass": "primary"},
        {"id": "w2", "text": "mumble", "start_sec": 1.0, "end_sec": 1.35, "source_pass": "rescue"},
        {"id": "w3", "text": "world", "start_sec": 1.8, "end_sec": 2.1, "source_pass": "primary"},
    ]

    def fake_extract(
        _source_path: str, _start_sec: float, _duration_sec: float, output_path
    ) -> None:
        output_path.write_bytes(b"retry")

    def fake_generate(
        _source_path: str,
        _duration_sec: float,
        *,
        language_hint: str | None,
        allow_mock_fallback: bool,
        fast_mode: bool,
        prompt: str | None,
    ) -> TranscriptPayload:
        del _source_path, _duration_sec, language_hint, allow_mock_fallback, fast_mode, prompt
        return TranscriptPayload(
            source="retry_provider",
            language="en",
            text="clear line",
            words=[
                TranscriptWordPayload(id="rw1", text="clear", start_sec=0.22, end_sec=0.42, confidence=0.94),
                TranscriptWordPayload(id="rw2", text="line", start_sec=0.42, end_sec=0.62, confidence=0.94),
            ],
            is_mock=False,
        )

    monkeypatch.setattr(transcript_router, "_extract_audio_chunk", fake_extract)
    monkeypatch.setattr(
        transcript_router, "_call_generate_transcript_compatible", fake_generate
    )

    updated = transcript_router._retry_weak_regions_in_items(
        str(tmp_path / "demo.mp4"),
        3.0,
        items=items,
        language_hint="en",
        prompt=None,
    )
    _stored, words, text, regions = transcript_router._materialize_transcript_items(
        updated, 3.0
    )

    assert text == "hello clear line world"
    assert [word.text for word in words] == ["hello", "clear", "line", "world"]
    assert all(word.quality_label == "trusted" for word in words[1:3])
    assert not any(region.status == "weak" for region in regions)


def test_resolve_transcript_generation_strategy_prefers_chunking_for_shortform_song() -> None:
    short_auto = transcript_router._resolve_transcript_generation_strategy(24.0, "auto")
    assert short_auto.mode == "auto"
    assert short_auto.optimize_for_speed is True
    assert short_auto.bypass_max_duration_sec is not None
    assert short_auto.bypass_max_duration_sec >= 24.0

    short_song = transcript_router._resolve_transcript_generation_strategy(75.0, "song")
    assert short_song.mode == "song"
    assert short_song.optimize_for_speed is True
    assert short_song.bypass_max_duration_sec == 0.0
    assert short_song.chunk_duration_sec == 45.0
    assert short_song.chunk_overlap_sec == 2.5
    assert short_song.chunk_parallelism == 2
    assert short_song.skip_timestamp_refinement is True
    assert short_song.skip_weak_region_retry is True

    long_auto = transcript_router._resolve_transcript_generation_strategy(150.0, "auto")
    assert long_auto.optimize_for_speed is False
    assert long_auto.chunk_duration_sec is None


def test_keep_ranges_from_deleted_words_preserves_extra_context_around_weak_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSCRIPT_CUT_CONTEXT_SEC", "0")
    monkeypatch.setenv("TRANSCRIPT_CUT_MERGE_GAP_SEC", "0")
    monkeypatch.setenv("TRANSCRIPT_CUT_MIN_REMOVAL_SEC", "0")
    monkeypatch.setenv("TRANSCRIPT_CUT_WEAK_REGION_SAFETY_SEC", "0.12")

    items = [
        {"id": "w1", "text": "hello", "start_sec": 0.0, "end_sec": 0.4, "source_pass": "primary"},
        {"id": "w2", "text": "maybe", "start_sec": 0.4, "end_sec": 0.8, "source_pass": "rescue"},
        {"id": "w3", "text": "world", "start_sec": 1.0, "end_sec": 1.4, "source_pass": "primary"},
    ]
    _stored, words, _text, _regions = transcript_router._materialize_transcript_items(items, 2.0)

    ranges = transcript_router._keep_ranges_from_deleted_words(
        words,
        2.0,
        {"w1", "w3"},
    )

    assert ranges == [
        {"start_sec": 0.0, "end_sec": 0.515},
        {"start_sec": 0.88, "end_sec": 2.0},
    ]
