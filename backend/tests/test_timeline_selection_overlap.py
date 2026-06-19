"""Mirrors frontend/src/utils/timelineSelection.ts overlap selection logic."""


def select_transcript_word_ids_in_range(
    words: list[dict[str, float | str]],
    start_sec: float,
    end_sec: float,
) -> list[str]:
    lo = min(start_sec, end_sec)
    hi = max(start_sec, end_sec)
    return [
        str(word["id"])
        for word in words
        if float(word["start_sec"]) < hi and float(word["end_sec"]) > lo
    ]


def test_range_select_uses_overlap_not_strict_containment() -> None:
    words = [
        {"id": "w1", "start_sec": 0.0, "end_sec": 1.0},
        {"id": "w2", "start_sec": 0.8, "end_sec": 1.6},
        {"id": "w3", "start_sec": 2.0, "end_sec": 2.5},
    ]

    selected = select_transcript_word_ids_in_range(words, 0.5, 1.2)
    assert selected == ["w1", "w2"]

    edge_only = select_transcript_word_ids_in_range(words, 1.0, 2.0)
    assert edge_only == ["w2"]

    empty = select_transcript_word_ids_in_range(words, 3.0, 4.0)
    assert empty == []
