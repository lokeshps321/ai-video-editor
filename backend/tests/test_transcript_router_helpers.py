from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")

from app.routers import transcript as transcript_router


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


def test_apply_range_update_items_replace_blank_and_preserve() -> None:
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
