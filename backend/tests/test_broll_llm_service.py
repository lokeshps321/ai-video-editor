import os

os.environ.setdefault("BROLL_LLM_ENABLED", "false")

from app.broll_llm_service import build_broll_search_strategy, infer_broll_domain_context


def test_infer_broll_domain_context_prefers_motorsport() -> None:
    context = infer_broll_domain_context(
        transcript_text=(
            "And that's Hayes in the upgraded Apex. The team were penalized after Monza, "
            "but the driver is coming through the field on the final lap."
        ),
        asset_filenames=["F1 The Movie 4K.mp4"],
    )

    assert context["domain"] == "motorsport"
    assert "motorsport" in str(context["summary"]).lower()


def test_build_broll_search_strategy_rewrites_noisy_race_conflict() -> None:
    strategy = build_broll_search_strategy(
        chunk_text="Plan C's for combat. And that's Hayes in the upgraded Apex.",
        concept_text="plan c's combat",
        visual_intent="abstract_support",
        query_hints=["plan c's combat", "Hayes upgraded Apex", "opening hook"],
        max_queries=6,
        domain_context={
            "domain": "motorsport",
            "summary": "motorsport race coverage with pit crew, telemetry, garage tension, and track action",
            "anchors": ["race", "pit", "garage"],
        },
    )

    queries = [str(item["query"]).lower() for item in strategy["queries"]]
    joined = " ".join(queries)

    assert strategy["visual_intent"] == "process_step"
    assert any(term in joined for term in ("race", "pit", "garage", "telemetry"))
    assert "military" not in joined
    assert "combat" not in joined


def test_infer_broll_domain_context_detects_music_video_from_filename() -> None:
    context = infer_broll_domain_context(
        transcript_text="I've been spending most our lives living in a gangster's paradise",
        asset_filenames=["Coolio_-_Gangsta_s_Paradise_feat._L.V._Official_Music_Video_1080P.mp4"],
    )

    assert context["domain"] == "music"


def test_build_broll_search_strategy_rejects_general_scene_for_music_lyrics() -> None:
    strategy = build_broll_search_strategy(
        chunk_text="I've been spending most our lives living in a gangster's paradise",
        concept_text="general scene",
        visual_intent="abstract_support",
        query_hints=["I've been spending most our lives living in a gangster's paradise"],
        max_queries=6,
        domain_context={
            "domain": "music",
            "summary": "reflective urban rap music video with night streets, prayer, struggle, and city lights",
            "anchors": ["urban streets", "night lights", "rap performance"],
        },
    )

    queries = [str(item["query"]).lower() for item in strategy["queries"]]
    joined = " ".join(queries)

    assert strategy["search_concept"] != "general scene"
    assert all(query != "general scene" for query in queries)
    assert any(term in joined for term in ("urban", "night", "rapper", "graffiti", "street"))
