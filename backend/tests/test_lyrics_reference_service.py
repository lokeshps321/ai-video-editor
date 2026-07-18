from app.lyrics_reference_service import (
    LyricsReference,
    align_reference_lyrics,
    looks_like_song_media,
    parse_track_hints,
)
from app.transcription_service import TranscriptPayload, TranscriptWordPayload


def _payload(entries: list[tuple[float, float, str]]) -> TranscriptPayload:
    words = [
        TranscriptWordPayload(id=f"w{idx}", text=text, start_sec=start_sec, end_sec=end_sec)
        for idx, (start_sec, end_sec, text) in enumerate(entries)
    ]
    return TranscriptPayload(
        source="chunked:groq_gapfill",
        language="English",
        text=" ".join(word.text for word in words),
        words=words,
        is_mock=False,
    )


def _entries_from_tokens(tokens: list[str], *, step_sec: float = 0.2) -> list[tuple[float, float, str]]:
    entries: list[tuple[float, float, str]] = []
    cursor = 0.0
    for token in tokens:
        start_sec = round(cursor, 3)
        end_sec = round(cursor + step_sec, 3)
        entries.append((start_sec, end_sec, token))
        cursor += step_sec
    return entries


def test_parse_track_hints_strips_music_video_noise() -> None:
    artist, track, query = parse_track_hints(
        "Coolio_-_Gangsta_s_Paradise_feat._L.V._Official_Music_Video_1080P.mp4"
    )
    assert artist == "Coolio"
    assert track == "Gangsta s Paradise feat. L.V"
    assert query == "Coolio Gangsta s Paradise feat. L.V"


def test_parse_track_hints_extracts_track_and_artist_from_movie_song_filename() -> None:
    artist, track, query = parse_track_hints(
        "Lose My Mind (Movie Version) 4K  Don Toliver (feat. Doja Cat) [From F1® The Movie]  #F1Movie_1080p.mp4"
    )
    assert artist == "Don Toliver (feat. Doja Cat)"
    assert track == "Lose My Mind"
    assert query == "Don Toliver (feat. Doja Cat) Lose My Mind"


def test_looks_like_song_media_matches_music_video_filename() -> None:
    assert looks_like_song_media(
        "Googly_-_Bisilu_Kudreyondu_Full_Song_Video_Yash_Kriti_Kharbanda_720P.mp4"
    )
    assert not looks_like_song_media("weekly_product_walkthrough_meeting.mp4")


def test_align_reference_lyrics_replaces_bad_asr_words_with_reference() -> None:
    payload = _payload(
        [
            (0.0, 0.2, "They've"),
            (0.2, 0.4, "been"),
            (0.4, 0.6, "spending"),
            (0.6, 0.8, "most"),
            (0.8, 1.0, "our"),
            (1.0, 1.2, "lives"),
            (1.2, 1.4, "living"),
            (1.4, 1.6, "in"),
            (1.6, 1.8, "the"),
            (1.8, 2.0, "justice"),
            (2.0, 2.2, "paradise"),
            (2.2, 2.4, "Tell"),
            (2.4, 2.6, "me"),
            (2.6, 2.8, "why"),
            (2.8, 3.0, "are"),
            (3.0, 3.2, "we"),
            (3.2, 3.4, "so"),
            (3.4, 3.6, "blind"),
            (3.6, 3.8, "to"),
            (3.8, 4.0, "see"),
        ]
    )
    reference = LyricsReference(
        track_name="Gangsta's Paradise",
        artist_name="Coolio",
        plain_lyrics=(
            "They've been spending most their lives living in the gangsta's paradise "
            "Tell me why are we so blind to see"
        ),
        duration_sec=4.0,
        score=0.99,
    )

    result = align_reference_lyrics(payload, reference, duration_sec=4.0)

    assert "justice" not in result.text.lower()
    assert "gangsta's paradise" in result.text.lower()
    assert result.source.endswith("_lyrics_ref")
    assert all(word.source_pass == "manual" for word in result.words)
    assert any((word.quality_score or 0) < 1.0 for word in result.words)


def test_align_reference_lyrics_preserves_repeated_chorus_order() -> None:
    payload = _payload(
        _entries_from_tokens(
            [
                "Keep",
                "spendin'",
                "most",
                "our",
                "lives",
                "livin'",
                "in",
                "the",
                "gangsta's",
                "paradise",
                "Tell",
                "me",
                "why",
                "are",
                "we",
                "so",
                "blind",
                "to",
                "see",
                "That",
                "the",
                "ones",
                "we",
                "hurt",
                "are",
                "you",
                "and",
                "me",
                "Keep",
                "spendin'",
                "most",
                "our",
                "me",
                "lives",
                "livin'",
                "in",
                "the",
                "justice",
                "paradise",
                "Share",
                "me",
                "why",
                "are",
                "we",
                "so",
                "blind",
                "to",
                "see",
                "That",
                "the",
                "ones",
                "we",
                "hurt",
                "are",
                "you",
                "and",
                "me",
            ]
        )
    )
    reference = LyricsReference(
        track_name="Gangsta's Paradise",
        artist_name="Coolio",
        plain_lyrics="\n".join(
            [
                "Keep spendin' most our lives livin' in the gangsta's paradise",
                "Tell me why are we so blind to see",
                "That the ones we hurt are you and me",
                "Keep spendin' most our lives livin' in the gangsta's paradise",
                "Tell me why are we so blind to see",
                "That the ones we hurt are you and me",
            ]
        ),
        duration_sec=11.4,
        score=0.99,
    )

    result = align_reference_lyrics(payload, reference, duration_sec=11.4)
    lower_text = result.text.lower()

    assert "justice" not in lower_text
    assert "share me why" not in lower_text
    assert (
        lower_text.count("keep spendin' most our lives livin' in the gangsta's paradise")
        == 2
    )
    assert lower_text.count("tell me why are we so blind to see") == 2
    assert len({word.id for word in result.words}) == len(result.words)



def test_align_reference_lyrics_prefers_synced_lyrics_when_available() -> None:
    payload = _payload(
        _entries_from_tokens(
            [
                "as",
                "i",
                "walk",
                "through",
                "the",
                "valley",
                "of",
                "the",
                "shadow",
                "of",
                "death",
                "i",
                "take",
                "a",
                "look",
                "at",
                "my",
                "life",
                "and",
                "realise",
                "there's",
                "nothing",
                "left",
                "cause",
                "i've",
                "been",
                "blasting",
                "and",
                "laughing",
                "so",
                "long",
                "that",
                "my",
                "mind",
                "is",
                "gone",
            ],
            step_sec=0.25,
        )
    )
    synced_lines = [
        "[00:00.00] As I walk through the valley",
        "[00:02.00] of the shadow of death",
        "[00:04.00] I take a look at my life",
        "[00:06.00] and realize there's nothing left",
        "[00:08.00] Cause I've been blasting and laughing",
        "[00:10.00] so long that my mind is gone",
    ]
    reference = LyricsReference(
        track_name="Gangsta's Paradise",
        artist_name="Coolio",
        plain_lyrics=" ".join(line.split("] ", 1)[1] for line in synced_lines),
        duration_sec=12.0,
        score=0.99,
        synced_lyrics="\n".join(synced_lines),
    )

    result = align_reference_lyrics(payload, reference, duration_sec=12.0)
    lower_text = result.text.lower()

    assert lower_text.startswith("as i walk through the valley of the shadow of death")
    assert "realize there's nothing left" in lower_text or "realise there's nothing left" in lower_text
    assert result.words[0].start_sec == 0.0
    assert any(word.text.lower() in {"realize", "realise"} for word in result.words)


def test_align_reference_lyrics_trims_synced_prefix_noise() -> None:
    payload = _payload(
        _entries_from_tokens(
            [
                "comments",
                "questions",
                "please",
                "post",
                "them",
                "below",
                "as",
                "i",
                "walk",
                "through",
                "the",
                "valley",
                "of",
                "the",
                "shadow",
                "of",
                "death",
                "i",
                "take",
                "a",
                "look",
                "at",
                "my",
                "life",
                "and",
                "realize",
                "there's",
                "nothing",
                "left",
            ],
            step_sec=0.25,
        )
    )
    synced_lines = [
        "[00:01.50] As I walk through the valley",
        "[00:03.50] of the shadow of death",
        "[00:05.50] I take a look at my life",
        "[00:07.50] and realize there's nothing left",
        "[00:09.50] Cause I've been blasting and laughing",
        "[00:11.50] so long that my mind is gone",
    ]
    reference = LyricsReference(
        track_name="Gangsta's Paradise",
        artist_name="Coolio",
        plain_lyrics=" ".join(line.split("] ", 1)[1] for line in synced_lines),
        duration_sec=14.0,
        score=0.99,
        synced_lyrics="\n".join(synced_lines),
    )

    result = align_reference_lyrics(payload, reference, duration_sec=14.0)

    assert result.text.lower().startswith("as i walk through the valley")
    assert "comments questions please" not in result.text.lower()


def test_align_reference_lyrics_keeps_asr_line_when_synced_lyrics_disagree_too_much() -> None:
    payload = _payload(
        _entries_from_tokens(
            [
                "to",
                "be",
                "treated",
                "like",
                "a",
                "punk",
                "you",
                "know",
                "that's",
                "unheard",
                "of",
                "still",
                "more",
                "words",
                "to",
                "keep",
                "alignment",
                "safe",
                "for",
                "testing",
                "more",
                "tokens",
                "again",
                "here",
            ],
            step_sec=0.5,
        )
    )
    synced_lines = [
        "[00:00.00] Me be treated like a punk you know that's unheard of",
        "[00:06.00] different words for another line right here now",
        "[00:12.00] more different words for another line right here",
        "[00:18.00] more different words for another line right here",
        "[00:24.00] more different words for another line right here",
        "[00:30.00] more different words for another line right here",
    ]
    reference = LyricsReference(
        track_name="Gangsta's Paradise",
        artist_name="Coolio",
        plain_lyrics=" ".join(line.split("] ", 1)[1] for line in synced_lines),
        duration_sec=36.0,
        score=0.99,
        synced_lyrics="\n".join(synced_lines),
    )

    result = align_reference_lyrics(payload, reference, duration_sec=36.0)
    lower_text = result.text.lower()

    assert "to be treated like a punk" in lower_text
    assert "me be treated like a punk" not in lower_text


def test_align_reference_lyrics_rescues_synced_line_when_timestamps_drift() -> None:
    payload = _payload(
        _entries_from_tokens(
            [
                "As",
                "I",
                "walk",
                "through",
                "the",
                "valley",
                "of",
                "the",
                "shadow",
                "of",
                "death",
                "I",
                "take",
                "a",
                "look",
                "at",
                "my",
                "life",
                "and",
                "realize",
                "there's",
                "nothing",
                "left",
                "To",
                "be",
                "treated",
                "like",
                "a",
                "punk",
                "you",
                "know",
                "that's",
                "unheard",
                "of",
            ],
            step_sec=0.35,
        )
    )
    synced_lines = [
        "[00:00.00] As I walk through the valley of the shadow of death",
        "[00:05.20] I take a look at my life and realize there's nothing left",
        "[00:12.80] To be treated like a punk you know that's unheard of",
        "[00:18.00] more different words for another line right here",
        "[00:24.00] more different words for another line right here",
        "[00:30.00] more different words for another line right here",
    ]
    reference = LyricsReference(
        track_name="Gangsta's Paradise",
        artist_name="Coolio",
        plain_lyrics=" ".join(line.split("] ", 1)[1] for line in synced_lines),
        duration_sec=36.0,
        score=0.99,
        synced_lyrics="\n".join(synced_lines),
    )

    result = align_reference_lyrics(payload, reference, duration_sec=36.0)
    lower_text = result.text.lower()

    assert "to be treated like a punk" in lower_text
    assert result.source.endswith("_lyrics_ref")


def test_align_reference_lyrics_estimates_global_synced_offset() -> None:
    payload = _payload(
        _entries_from_tokens(
            [
                "As",
                "I",
                "walk",
                "through",
                "the",
                "valley",
                "of",
                "the",
                "shadow",
                "of",
                "death",
                "I",
                "take",
                "a",
                "look",
                "at",
                "my",
                "life",
                "and",
                "realize",
                "there's",
                "nothing",
                "left",
                "Cause",
                "I've",
                "been",
                "blastin'",
                "and",
                "laughin'",
                "so",
                "long",
                "that",
                "my",
                "mind",
                "is",
                "gone",
            ],
            step_sec=0.33,
        )
    )
    synced_lines = [
        "[00:12.00] As I walk through the valley of the shadow of death",
        "[00:15.20] I take a look at my life and realize there's nothing left",
        "[00:18.40] Cause I've been blastin' and laughin' so long that my mind is gone",
        "[00:24.00] more different words for another line right here",
        "[00:30.00] more different words for another line right here",
        "[00:36.00] more different words for another line right here",
    ]
    reference = LyricsReference(
        track_name="Gangsta's Paradise",
        artist_name="Coolio",
        plain_lyrics=" ".join(line.split("] ", 1)[1] for line in synced_lines),
        duration_sec=42.0,
        score=0.99,
        synced_lyrics="\n".join(synced_lines),
    )

    result = align_reference_lyrics(payload, reference, duration_sec=42.0)
    lower_text = result.text.lower()

    assert lower_text.startswith("as i walk through the valley of the shadow of death")
    assert "realize there's nothing left" in lower_text
    assert result.source.endswith("_lyrics_ref")


def test_align_reference_lyrics_sanitizes_obvious_reference_corruption() -> None:
    payload = _payload(
        _entries_from_tokens(
            [
                "But",
                "I",
                "ain't",
                "never",
                "crossed",
                "a",
                "man",
                "that",
                "didn't",
                "deserve",
                "it",
                "Me",
                "be",
                "treated",
                "like",
                "a",
                "punk",
                "you",
                "know",
                "that's",
                "unheard",
                "of",
            ],
            step_sec=0.35,
        )
    )
    synced_lines = [
        "[00:00.00] But I ain't never crossed a man that didn't deserve it",
        "[00:04.20] Me be treated like a punk you know that's unheard of",
        "[00:10.00] more different words for another line right here",
        "[00:16.00] more different words for another line right here",
        "[00:22.00] more different words for another line right here",
        "[00:28.00] more different words for another line right here",
    ]
    reference = LyricsReference(
        track_name="Gangsta's Paradise",
        artist_name="Coolio",
        plain_lyrics=" ".join(line.split("] ", 1)[1] for line in synced_lines),
        duration_sec=34.0,
        score=0.99,
        synced_lyrics="\n".join(synced_lines),
    )

    result = align_reference_lyrics(payload, reference, duration_sec=34.0)
    lower_text = result.text.lower()

    assert "to be treated like a punk" in lower_text
    assert "me be treated like a punk" not in lower_text


def test_align_reference_lyrics_skips_when_alignment_ratio_is_too_low(monkeypatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_LYRICS_REFERENCE_MIN_ALIGNMENT_RATIO", "0.95")
    payload = _payload([(0.0, 0.2, "hello"), (0.2, 0.4, "world")] * 12)
    reference = LyricsReference(
        track_name="Song",
        artist_name="Artist",
        plain_lyrics="different words entirely " * 10,
        duration_sec=4.8,
        score=0.99,
    )

    result = align_reference_lyrics(payload, reference, duration_sec=4.8)
    assert result is payload


def _span_words(entries: list[tuple[float, float, str]]) -> list[TranscriptWordPayload]:
    return [
        TranscriptWordPayload(id=f"s{idx}", text=text, start_sec=start_sec, end_sec=end_sec)
        for idx, (start_sec, end_sec, text) in enumerate(entries)
    ]


def test_synced_line_span_uses_short_gap_directly() -> None:
    from app.lyrics_reference_service import _synced_line_span

    span = _synced_line_span(
        ["one", "two", "three"],
        adjusted_start_sec=10.0,
        next_start_sec=13.0,
    )
    assert abs(span - 2.98) < 1e-6


def test_synced_line_span_heuristic_for_long_gap_without_asr() -> None:
    from app.lyrics_reference_service import _synced_line_span

    tokens = ["w"] * 10
    span = _synced_line_span(
        tokens,
        adjusted_start_sec=10.0,
        next_start_sec=16.0,
    )
    # 0.24 * 10 * 1.6 = 3.84 — no longer squeezed to 2.4s, still under the gap
    assert abs(span - 3.84) < 1e-6


def test_synced_line_span_caps_at_ceiling_for_huge_gap() -> None:
    from app.lyrics_reference_service import _synced_line_span

    tokens = ["w"] * 30
    span = _synced_line_span(
        tokens,
        adjusted_start_sec=10.0,
        next_start_sec=25.0,
    )
    assert span <= 8.0


def test_synced_line_span_measures_from_asr_words() -> None:
    from app.lyrics_reference_service import _synced_line_span

    tokens = ["w"] * 10
    asr_words = _span_words(
        [(10.0 + 0.55 * i, 10.0 + 0.55 * i + 0.4, f"t{i}") for i in range(10)]
    )
    span = _synced_line_span(
        tokens,
        adjusted_start_sec=10.0,
        next_start_sec=16.0,
        asr_words=asr_words,
    )
    # Last ASR word ends at 10 + 0.55*9 + 0.4 = 15.35 → measured span 5.35s
    assert abs(span - 5.35) < 1e-6


def test_synced_line_span_falls_back_when_asr_sparse() -> None:
    from app.lyrics_reference_service import _synced_line_span

    tokens = ["w"] * 10
    asr_words = _span_words([(10.0, 10.4, "t0"), (11.0, 11.4, "t1")])
    span = _synced_line_span(
        tokens,
        adjusted_start_sec=10.0,
        next_start_sec=16.0,
        asr_words=asr_words,
    )
    assert abs(span - 3.84) < 1e-6
