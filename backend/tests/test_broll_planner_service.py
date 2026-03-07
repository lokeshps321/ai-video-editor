import os

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")
os.environ.setdefault("BROLL_LLM_ENABLED", "false")

from app.broll_planner_service import plan_broll


def _planner_words() -> list[dict[str, object]]:
    segments = [
        "viral opening hook camera",
        "product demo screen workflow",
        "team meeting office planning",
        "factory process machine detail",
        "customer problem broken checkout",
        "simple fix software dashboard",
        "warehouse packing order speed",
        "marketing campaign analytics growth",
        "laptop design prototype review",
        "phone app feature walkthrough",
        "money metrics revenue result",
        "studio creator editing workflow",
        "shop owner sales improvement",
        "support queue issue response",
        "device testing quality control",
        "street delivery rider motion",
        "software launch crowd reaction",
        "critical lesson process change",
        "success result dashboard spike",
        "payoff customer team celebration",
        "final takeaway creator studio",
        "outro call action result",
    ]
    words: list[dict[str, object]] = []
    word_index = 1
    for segment_index, segment in enumerate(segments):
        base_start = float(segment_index * 5.2)
        for token_index, token in enumerate(segment.split()):
            start_sec = round(base_start + (token_index * 0.48), 3)
            end_sec = round(start_sec + 0.28, 3)
            text = token
            if token_index == len(segment.split()) - 1:
                text = f"{text}."
            words.append(
                {
                    "id": f"pw{word_index}",
                    "text": text,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                }
            )
            word_index += 1
    return words


def test_plan_broll_balances_beats_across_transcript(monkeypatch) -> None:
    monkeypatch.setattr("app.broll_planner_service._cloud_plan", lambda **_: (None, None))
    result = plan_broll(
        words=_planner_words(),
        transcript_text=" ".join(str(word["text"]) for word in _planner_words()),
        transcript_duration_sec=120.0,
        max_slots=8,
        min_chunk_words=3,
        assets=[
            {"id": "a1", "filename": "factory-floor.mp4", "metadata_text": "factory machine process"},
            {"id": "a2", "filename": "creator-studio.mp4", "metadata_text": "creator studio workflow"},
        ],
        include_external_sources=False,
    )

    beats = result["beats"]
    assert isinstance(beats, list)
    assert len(beats) >= 6
    assert max(float(beat["start_sec"]) for beat in beats) > 100.0

    coverage = result["coverage"]
    sections = {
        item["section_label"]: item["beat_count"]
        for item in coverage["coverage_sections"]
    }
    assert sections["hook"] >= 1
    assert sections["body"] >= 1
    assert sections["payoff"] >= 1
    assert sections["payoff"] + sections["outro"] >= 1


def test_plan_broll_prioritizes_metric_and_process_visuals(monkeypatch) -> None:
    monkeypatch.setattr("app.broll_planner_service._cloud_plan", lambda **_: (None, None))
    segments = [
        "so yeah this part is fine",
        "dashboard conversion jumped 42 percent",
        "hands packing orders on dashboard screen workflow",
        "and then we just keep talking",
        "plain discussion keeps moving",
        "final result revenue spike",
    ]
    words: list[dict[str, object]] = []
    word_index = 1
    for segment_index, segment in enumerate(segments):
        base_start = float(segment_index * 4.5)
        tokens = segment.split()
        for token_index, token in enumerate(tokens):
            start_sec = round(base_start + (token_index * 0.42), 3)
            end_sec = round(start_sec + 0.26, 3)
            text = f"{token}." if token_index == len(tokens) - 1 else token
            words.append(
                {
                    "id": f"mw{word_index}",
                    "text": text,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                }
            )
            word_index += 1

    result = plan_broll(
        words=words,
        transcript_text=" ".join(str(word["text"]) for word in words),
        transcript_duration_sec=30.0,
        max_slots=4,
        min_chunk_words=3,
        assets=[],
        include_external_sources=True,
    )

    beats = result["beats"]
    assert any("42" in str(beat["segment_text"]) for beat in beats)
    assert any(
        beat["shot_style"] == "detail" and "workflow" in str(beat["segment_text"]).lower()
        for beat in beats
    )
    assert any("metric" in str(beat["rationale"]).lower() or "payoff" in str(beat["rationale"]).lower() for beat in beats)
