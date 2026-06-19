#!/usr/bin/env python3
"""Live transcript language audit across supported UI languages."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
FIXTURES = BACKEND / "tests" / "fixtures" / "language_clips"
DOWNLOADS = Path.home() / "Downloads"

# Load env
for env_file in (BACKEND / ".env", BACKEND / ".env.local"):
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

sys.path.insert(0, str(BACKEND))
from app.diarization_service import maybe_enhance_duet_transcript  # noqa: E402
from app.transcription_service import (  # noqa: E402
    _normalize_detected_language,
    generate_transcript,
)

UI_LANGUAGES = (
    "en",
    "kn",
    "hi",
    "ta",
    "te",
    "ml",
    "mr",
    "bn",
    "gu",
    "pa",
    "or",
    "ur",
)

LIVE_SOURCES: dict[str, list[Path]] = {
    "kn": [
        DOWNLOADS
        / "Googly_-_Bisilu_Kudreyondu_Full_Song_Video_Yash_Kriti_Kharbanda_720P.mp4",
        FIXTURES / "kn.wav",
    ],
    "ta": [
        DOWNLOADS
        / "vidssave.com 3 - Po Nee Po Video _ Dhanush, Shruti _ Anirudh 1080P.mp4",
        FIXTURES / "ta.wav",
    ],
    "en": [
        DOWNLOADS
        / "Coolio_-_Gangsta_s_Paradise_feat._L.V._Official_Music_Video_1080P.mp4",
        FIXTURES / "en.wav",
    ],
    "en_duet": [
        DOWNLOADS
        / "Lose My Mind (Movie Version) 4K  Don Toliver (feat. Doja Cat) [From F1® The Movie]  #F1Movie_1080p.mp4",
    ],
    "hi": [FIXTURES / "hi.wav"],
    "te": [FIXTURES / "te.wav"],
    "ml": [FIXTURES / "ml.wav"],
    "mr": [FIXTURES / "mr.wav"],
    "bn": [FIXTURES / "bn.wav"],
    "gu": [FIXTURES / "gu.wav"],
    "pa": [FIXTURES / "pa.wav"],
    "ur": [FIXTURES / "ur.wav"],
}


def resolve_media(code: str) -> Path | None:
    for candidate in LIVE_SOURCES.get(code, []):
        if candidate.exists():
            return candidate
    return None


def to_wav(src: Path, dst: Path, seconds: float = 30.0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-t",
            str(seconds),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ],
        capture_output=True,
        check=True,
    )


def run_case(label: str, wav: Path, duration: float, hint: str | None, fast: bool) -> tuple[bool, str]:
    try:
        result = generate_transcript(
            str(wav),
            duration,
            language_hint=hint,
            fast_mode=fast,
            allow_mock_fallback=False,
        )
    except Exception as exc:
        return False, f"ERROR {exc}"
    norm = _normalize_detected_language(result.language)
    expected = label if label != "auto" else None
    if expected is None:
        return True, f"auto->{norm!r} via {result.source}"
    ok = norm == expected or (expected == "en" and norm == "en")
    status = "OK" if ok else f"FAIL got {norm!r}"
    return ok, f"{status} via {result.source}"


def main() -> int:
    tmp = BACKEND / "tmp" / "language_audit"
    tmp.mkdir(parents=True, exist_ok=True)
    failures = 0
    skipped = 0

    print(f"{'Lang':5} {'Mode':16} {'Result':40}")
    print("-" * 65)

    for code in UI_LANGUAGES:
        media = resolve_media(code)
        if media is None:
            print(f"{code:5} {'all':16} SKIP (no sample media)")
            skipped += 1
            continue
        wav = tmp / f"{code}.wav"
        to_wav(media, wav, 30.0)
        duration = 30.0
        for mode_name, hint, fast in (
            ("auto", None, False),
            ("auto+fast", None, True),
            (f"pick-{code}", code, False),
        ):
            ok, detail = run_case(
                code if hint else "auto",
                wav,
                duration,
                hint,
                fast,
            )
            if not ok and hint == code:
                failures += 1
            mark = "PASS" if ok else "FAIL"
            print(f"{code:5} {mode_name:16} {mark:5} {detail}")

    duet_media = resolve_media("en_duet")
    if duet_media is None:
        print("en_duet SKIP (no F1 duet sample media)")
    else:
        duet_wav = tmp / "en_duet.wav"
        to_wav(duet_media, duet_wav, 60.0)
        duet_duration = 60.0
        base = generate_transcript(
            str(duet_wav),
            duet_duration,
            language_hint="en",
            fast_mode=False,
            allow_mock_fallback=False,
            filename=duet_media.name,
        )
        enhanced = maybe_enhance_duet_transcript(
            base,
            audio_path=str(duet_wav),
            duration_sec=duet_duration,
            filename=duet_media.name,
            language_hint="en",
        )
        speakers = {word.speaker_id for word in enhanced.words if word.speaker_id}
        lang = _normalize_detected_language(enhanced.language)
        duet_ok = lang == "en" and len(speakers) >= 2 and len(enhanced.words) >= 20
        if not duet_ok:
            failures += 1
        print(
            f"en_duet {'PASS' if duet_ok else 'FAIL'} "
            f"lang={lang!r} speakers={len(speakers)} words={len(enhanced.words)} "
            f"source={enhanced.source}"
        )

    print("-" * 65)
    print(f"failures={failures} skipped_languages={skipped}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
