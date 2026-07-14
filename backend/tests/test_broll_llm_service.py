import os

os.environ["BROLL_LLM_ENABLED"] = "false"

import pytest
from app.broll_llm_service import (
    build_broll_search_strategy,
    infer_broll_domain_context,
)

# app.config deliberately loads backend/.env during import. Restore the unit-test
# defaults after importing the service so ordinary strategy tests never call a
# real provider; individual tests explicitly monkeypatch the integration seam.
os.environ["BROLL_LLM_ENABLED"] = "false"
os.environ["BROLL_SARVAM_TRANSLATE_ENABLED"] = "false"


@pytest.mark.parametrize(
    ("language_hint", "sarvam_code"),
    [
        ("as", "as-IN"),
        ("bn", "bn-IN"),
        ("brx", "brx-IN"),
        ("doi", "doi-IN"),
        ("gu", "gu-IN"),
        ("hi", "hi-IN"),
        ("kn", "kn-IN"),
        ("ks", "ks-IN"),
        ("kok", "kok-IN"),
        ("mai", "mai-IN"),
        ("ml", "ml-IN"),
        ("mni", "mni-IN"),
        ("mr", "mr-IN"),
        ("ne", "ne-IN"),
        ("od", "od-IN"),
        ("pa", "pa-IN"),
        ("sa", "sa-IN"),
        ("sat", "sat-IN"),
        ("sd", "sd-IN"),
        ("ta", "ta-IN"),
        ("te", "te-IN"),
        ("ur", "ur-IN"),
    ],
)
def test_sarvam_broll_translation_supports_all_indic_languages(
    language_hint: str, sarvam_code: str
) -> None:
    from app.broll_llm_service import _sarvam_source_language_code

    assert _sarvam_source_language_code(language_hint) == sarvam_code


def test_broll_chat_prefers_gemini_with_existing_provider_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.broll_llm_service._llm_enabled", lambda: True)
    monkeypatch.setattr(
        "app.broll_llm_service._broll_llm_provider", lambda: "gemini"
    )
    monkeypatch.setattr(
        "app.broll_llm_service._gemini_chat_json", lambda _prompt: {"ok": True}
    )
    monkeypatch.setattr(
        "app.broll_llm_service._openai_chat_json",
        lambda _prompt: (_ for _ in ()).throw(AssertionError("should not fall back")),
    )

    from app.broll_llm_service import _chat_json

    assert _chat_json({"goal": "test"}) == {
        "ok": True,
        "__broll_planner_provider": "gemini-3.5-flash",
    }


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
        asset_filenames=[
            "Coolio_-_Gangsta_s_Paradise_feat._L.V._Official_Music_Video_1080P.mp4"
        ],
    )

    assert context["domain"] == "music"


def test_build_broll_search_strategy_rejects_general_scene_for_music_lyrics() -> None:
    strategy = build_broll_search_strategy(
        chunk_text="I've been spending most our lives living in a gangster's paradise",
        concept_text="general scene",
        visual_intent="abstract_support",
        query_hints=[
            "I've been spending most our lives living in a gangster's paradise"
        ],
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
    assert any(
        term in joined for term in ("urban", "night", "rapper", "graffiti", "street")
    )


def test_build_broll_search_strategy_translates_non_english_music_to_english_queries() -> (
    None
):
    strategy = build_broll_search_strategy(
        chunk_text="ಬೀಚೆಕಾಯೊ ಎಂದು ನಿನ್ನ ಕಾಯುತ್ತೇನೆ ಮನದಾಳದಲ್ಲಿ",
        concept_text="ಬೀಚೆಕಾಯೊ ಎಂದು",
        visual_intent="abstract_support",
        query_hints=["ಬೀಚೆಕಾಯೊ ಎಂದು", "ಮನದಾಳದಲ್ಲಿ"],
        max_queries=6,
        domain_context={
            "domain": "music",
            "summary": "music and performance video with studio sessions, stage moments, and audience reaction",
            "anchors": ["music", "stage", "studio"],
        },
        language_hint="kn",
    )

    queries = [str(item["query"]) for item in strategy["queries"]]
    joined = " ".join(queries).lower()

    assert strategy["search_concept"] == "cinematic music emotion"
    assert all(query.isascii() for query in queries)
    assert any(term in joined for term in ("window", "sunset", "city", "ocean", "rain"))


def test_build_broll_search_strategy_rejects_non_english_llm_output_for_cross_lingual_music(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.broll_llm_service._llm_enabled", lambda: True)
    monkeypatch.setattr(
        "app.broll_llm_service._chat_json",
        lambda _prompt: {
            "search_concept": "ಕನ್ನಡ ಪ್ರೀತಿ ಹಾಡು",
            "visual_intent": "abstract_support",
            "stockability": "medium",
            "blocked_terms": [],
            "queries": [
                {"query": "ಕನ್ನಡ ಪ್ರೀತಿ ಹಾಡು", "mode": "abstract"},
                {"query": "ಮಳೆ ಕಿಟಕಿ", "mode": "abstract"},
            ],
            "rationale": "returned untranslated lyric queries",
        },
    )

    strategy = build_broll_search_strategy(
        chunk_text="ಬೀಚೆಕಾಯೊ ಎಂದು ನಿನ್ನ ಕಾಯುತ್ತೇನೆ ಮನದಾಳದಲ್ಲಿ",
        concept_text="ಬೀಚೆಕಾಯೊ ಎಂದು",
        visual_intent="abstract_support",
        query_hints=["ಬೀಚೆಕಾಯೊ ಎಂದು", "ಮನದಾಳದಲ್ಲಿ"],
        max_queries=6,
        domain_context={
            "domain": "music",
            "summary": "music and performance video with studio sessions, stage moments, and audience reaction",
            "anchors": ["music", "stage", "studio"],
        },
        language_hint="kn",
    )

    queries = [str(item["query"]) for item in strategy["queries"]]

    assert strategy["search_concept"] == "cinematic music emotion"
    assert all(query.isascii() for query in queries)


def test_build_broll_search_strategy_uses_translated_visual_gloss_before_query_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.broll_llm_service._llm_enabled", lambda: True)

    def _fake_chat(prompt: dict[str, object]) -> dict[str, object]:
        goal = str(prompt.get("goal") or "")
        if (
            goal
            == "Translate a non-English transcript beat into English meaning for B-roll search."
        ):
            return {
                "english_gloss": "woman waiting by the window in the rain thinking about her lover",
                "english_search_concept": "rainy longing by window",
                "english_query_hints": [
                    "woman waiting by window rain",
                    "rain on glass cinematic",
                    "lonely romantic night interior",
                ],
                "rationale": "captured the romantic longing visually",
            }
        if (
            goal
            == "Convert one noisy transcript beat into strong stock-video retrieval queries."
        ):
            assert (
                prompt.get("beat_text")
                == "woman waiting by the window in the rain thinking about her lover"
            )
            assert prompt.get("concept_text") == "rainy longing by window"
            return {
                "search_concept": "rainy longing by window",
                "visual_intent": "abstract_support",
                "stockability": "high",
                "blocked_terms": [],
                "queries": [
                    {"query": "woman waiting by window rain", "mode": "literal"},
                    {"query": "rain on glass cinematic", "mode": "abstract"},
                    {"query": "lonely romantic night interior", "mode": "environment"},
                ],
                "rationale": "used the translated visual gloss",
            }
        raise AssertionError(f"Unexpected goal: {goal}")

    monkeypatch.setattr("app.broll_llm_service._chat_json", _fake_chat)

    strategy = build_broll_search_strategy(
        chunk_text="ನಿನ್ನ ಕಾದು ಕಿಟಕಿಯ ಬಳಿ ಮಳೆಯಲ್ಲಿ ನಿಂತಿರುವೆ",
        concept_text="ನಿನ್ನ ಕಾದು",
        visual_intent="abstract_support",
        query_hints=["ನಿನ್ನ ಕಾದು", "ಕಿಟಕಿಯ ಬಳಿ", "ಮಳೆ"],
        max_queries=6,
        domain_context={
            "domain": "music",
            "summary": "music and performance video with studio sessions, stage moments, and audience reaction",
            "anchors": ["music", "stage", "studio"],
        },
        language_hint="kn",
    )

    assert strategy["search_concept"] == "rainy longing by window"
    assert (
        strategy["english_gloss"]
        == "woman waiting by the window in the rain thinking about her lover"
    )
    assert all(str(item["query"]).isascii() for item in strategy["queries"])


def test_indic_broll_uses_sarvam_translation_before_gemini_visual_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.broll_llm_service._llm_enabled", lambda: True)
    monkeypatch.setattr(
        "app.broll_llm_service._translate_indic_text_to_english",
        lambda **_kwargs: {
            "english_text": "I wait for you by the window in the rain.",
            "provider": "sarvam-translate:v1",
            "source_language_code": "kn-IN",
        },
    )
    monkeypatch.setattr(
        "app.broll_llm_service._broll_llm_provider", lambda: "gemini"
    )
    monkeypatch.setattr("app.broll_llm_service._gemini_api_key", lambda: "test-key")

    def _fake_chat(prompt: dict[str, object]) -> dict[str, object]:
        goal = str(prompt.get("goal") or "")
        if goal == (
            "Turn a verified English Indic-language translation into B-roll meaning "
            "and stock-video search ideas."
        ):
            assert prompt["source_language"] == "kn-IN"
            assert (
                prompt["verified_english_translation"]
                == "I wait for you by the window in the rain."
            )
            return {
                "english_gloss": "someone waits by a rainy window for their loved one",
                "english_search_concept": "rainy longing by window",
                "english_query_hints": [
                    "person waiting by rainy window",
                    "rain on glass at night",
                ],
                "rationale": "Used the verified English translation for the scene.",
                "__broll_planner_provider": "gemini-3.5-flash",
            }
        if goal == "Convert one noisy transcript beat into strong stock-video retrieval queries.":
            raise AssertionError("translated Indic B-roll should not call Gemini twice")
        raise AssertionError(f"Unexpected goal: {goal}")

    monkeypatch.setattr("app.broll_llm_service._chat_json", _fake_chat)

    strategy = build_broll_search_strategy(
        chunk_text="ನಿನ್ನ ಕಾದು ಕಿಟಕಿಯ ಬಳಿ ಮಳೆಯಲ್ಲಿ ನಿಂತಿರುವೆ",
        concept_text="ನಿನ್ನ ಕಾದು",
        visual_intent="abstract_support",
        query_hints=["ನಿನ್ನ ಕಾದು", "ಕಿಟಕಿಯ ಬಳಿ", "ಮಳೆ"],
        max_queries=6,
        domain_context={
            "domain": "music",
            "summary": "music and performance video with studio sessions and emotion",
            "anchors": ["music", "stage"],
        },
        language_hint="kn",
    )

    assert strategy["english_gloss"] == (
        "someone waits by a rainy window for their loved one"
    )
    assert strategy["translation_provider"] == "sarvam-translate:v1"
    assert strategy["planner_provider"] == "gemini-3.5-flash"
    assert [item["query"] for item in strategy["queries"]] == [
        "person waiting by rainy window",
        "rain on glass at night",
    ]


def test_build_broll_search_strategy_prefers_manual_english_gloss_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.broll_llm_service._llm_enabled", lambda: True)

    def _fake_chat(prompt: dict[str, object]) -> dict[str, object]:
        goal = str(prompt.get("goal") or "")
        if (
            goal
            == "Convert one noisy transcript beat into strong stock-video retrieval queries."
        ):
            assert (
                prompt.get("english_gloss_override")
                == "woman waiting by the window in the rain"
            )
        if (
            goal
            == "Convert one noisy transcript beat into strong stock-video retrieval queries."
        ):
            assert (
                prompt.get("english_gloss_override")
                == "woman waiting by the window in the rain"
            )
            assert prompt.get("beat_text") == "woman waiting by the window in the rain"
            return {
                "search_concept": "woman waiting by the window in the rain",
                "visual_intent": "abstract_support",
                "stockability": "high",
                "blocked_terms": [],
                "queries": [
                    {"query": "woman waiting by window rain", "mode": "literal"},
                    {"query": "rain on glass cinematic", "mode": "abstract"},
                ],
                "rationale": "used manual gloss override",
            }
        raise AssertionError(f"Unexpected goal: {goal}")

    monkeypatch.setattr("app.broll_llm_service._chat_json", _fake_chat)

    strategy = build_broll_search_strategy(
        chunk_text="ನಿನ್ನ ಕಾದು ಕಿಟಕಿಯ ಬಳಿ ಮಳೆಯಲ್ಲಿ ನಿಂತಿರುವೆ",
        concept_text="ನಿನ್ನ ಕಾದು",
        visual_intent="abstract_support",
        query_hints=["ನಿನ್ನ ಕಾದು", "ಕಿಟಕಿಯ ಬಳಿ", "ಮಳೆ"],
        max_queries=6,
        domain_context={
            "domain": "music",
            "summary": "music and performance video with studio sessions, stage moments, and audience reaction",
            "anchors": ["music", "stage", "studio"],
        },
        language_hint="kn",
        english_gloss_override="woman waiting by the window in the rain",
    )

    assert strategy["search_concept"] == "woman waiting by the window in the rain"
    assert strategy["english_gloss"] == "woman waiting by the window in the rain"
    assert strategy["gloss_override_used"] == "woman waiting by the window in the rain"
