from __future__ import annotations

from uuid import uuid4

from app.diarization_service import (
    DiarizedEntry,
    apply_speaker_labels,
    assign_speaker_ids_to_words,
    diarized_entries_to_words,
)
from app.lyrics_reference_service import looks_like_duet_media, parse_duet_artists
from app.transcription_service import TranscriptWordPayload


def test_looks_like_duet_media_detects_feat_filename() -> None:
    filename = (
        "Lose My Mind (Movie Version) 4K  Don Toliver (feat. Doja Cat) "
        "[From F1® The Movie]  #F1Movie_1080p.mp4"
    )
    assert looks_like_duet_media(filename)


def test_parse_duet_artists_from_f1_filename() -> None:
    filename = (
        "Lose My Mind (Movie Version) 4K  Don Toliver (feat. Doja Cat) "
        "[From F1® The Movie]  #F1Movie_1080p.mp4"
    )
    primary, featured = parse_duet_artists(filename)
    assert primary == "Don Toliver"
    assert featured == "Doja Cat"


def test_assign_speaker_ids_to_words_by_overlap() -> None:
    words = [
        TranscriptWordPayload(
            id=str(uuid4()),
            text="hello",
            start_sec=0.0,
            end_sec=0.4,
        ),
        TranscriptWordPayload(
            id=str(uuid4()),
            text="world",
            start_sec=3.0,
            end_sec=3.4,
        ),
    ]
    entries = [
        DiarizedEntry("hello there", 0.0, 1.0, "speaker_0"),
        DiarizedEntry("world again", 2.8, 4.0, "speaker_1"),
    ]
    tagged = assign_speaker_ids_to_words(words, entries)
    assert tagged[0].speaker_id == "speaker_0"
    assert tagged[1].speaker_id == "speaker_1"


def test_apply_speaker_labels_maps_primary_and_featured() -> None:
    words = [
        TranscriptWordPayload(
            id="a",
            text="one",
            start_sec=0.0,
            end_sec=0.2,
            speaker_id="speaker_0",
        ),
        TranscriptWordPayload(
            id="b",
            text="two",
            start_sec=1.0,
            end_sec=1.2,
            speaker_id="speaker_1",
        ),
    ]
    labeled = apply_speaker_labels(
        words,
        primary_artist="Don Toliver",
        featured_artist="Doja Cat",
    )
    assert labeled[0].speaker_label == "Don Toliver"
    assert labeled[1].speaker_label == "Doja Cat"


def test_diarized_entries_to_words_assigns_speaker_ids() -> None:
    entries = [
        DiarizedEntry("lose my mind", 1.0, 2.0, "speaker_1"),
        DiarizedEntry("race commentary", 3.0, 4.0, "speaker_0"),
    ]
    words = diarized_entries_to_words(entries, 5.0)
    assert len(words) >= 4
    speakers = {word.speaker_id for word in words}
    assert speakers == {"speaker_0", "speaker_1"}
