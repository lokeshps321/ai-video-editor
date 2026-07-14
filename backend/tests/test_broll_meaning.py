from __future__ import annotations

import json

from app.models import BrollSlot
from app.routers import broll as broll_router


def test_broll_meaning_persists_code_switched_source_and_english_retrieval() -> None:
    strategy: dict[str, object] = {
        "search_concept": "startup founder planning in an office",
        "query_packets": [
            {"query": "startup founder planning office", "mode": "process"},
            {"query": "business team whiteboard", "mode": "literal"},
        ],
        "rationale": "Used an English visual gloss for stock retrieval.",
        "raw_strategy": {
            "english_gloss": "I started a startup and planned the first launch.",
            "gloss_override_used": "startup founder planning a launch",
            "normalized_source_text": "ನಾನು ಸ್ಟಾರ್ಟಪ್ ಆರಂಭಿಸಿದೆ",
            "meaning_review_required": True,
            "meaning_warning": "Romanized lyric spelling is ambiguous.",
        },
    }

    meaning = broll_router._build_slot_meaning(
        source_text="ನಾನು startup ಆರಂಭಿಸಿದೆ",
        language_hint="kn",
        search_strategy=strategy,
    )
    slot = BrollSlot(
        project_id="project-1",
        transcript_id="transcript-1",
        concept_text="ನಾನು startup ಆರಂಭಿಸಿದೆ",
        meaning_json=json.dumps(meaning),
    )

    response = broll_router._parse_slot_meaning(slot, [])

    assert response.source_languages == ["kn", "en"]
    assert response.code_switched is True
    assert response.english_gloss == "I started a startup and planned the first launch."
    assert response.search_concept == "startup founder planning in an office"
    assert response.search_queries == [
        "startup founder planning office",
        "business team whiteboard",
    ]
    assert response.gloss_override_used == "startup founder planning a launch"
    assert response.normalized_source_text == "ನಾನು ಸ್ಟಾರ್ಟಪ್ ಆರಂಭಿಸಿದೆ"
    assert response.meaning_review_required is True
    assert response.meaning_warning == "Romanized lyric spelling is ambiguous."
