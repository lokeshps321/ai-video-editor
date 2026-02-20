# Transcript + Subtitle Context Handoff (2026-02-18)

## Why this file exists
This is a working handoff so context is not lost between sessions.  
Scope is transcript generation and subtitle reliability for mixed content (songs + commentary, podcasts, normal speech videos).

## Current status (as of this handoff)
- Backend is running at `http://localhost:8000`.
- Latest runtime uses dynamic transcription profile selection (`speech` / `mixed` / `music`).
- Gap rescue feature is implemented in code but currently **disabled in runtime**:
  - `backend/.env`: `TRANSCRIBE_ENABLE_GAP_RESCUE=false`
- Current behavior is the "previous was better" mode requested by user.

## User-reported issues covered
- Small timeline edits previously caused output to look like half video.
- Preview render was slow.
- `add_subtitles` sometimes produced no visible subtitles.
- Song + commentary videos had partial transcript/subtitles.
- Loud-music sections missed words.
- Early segment around `0:23-0:30` repeatedly missed or produced noisy text depending on strategy.

## What was implemented

### 1) Stability + render/subtitle fixes
- Preview jobs use lightweight defaults for speed:
  - `resolution=720p`, `fps=24`, `quality=low`
  - Files: `backend/app/routers/render.py`, `backend/app/routers/vibe.py`
- x264 quality mapping tuned for faster preview:
  - `low -> ultrafast`
  - File: `backend/app/render_service.py`
- Karaoke subtitle rendering fixed for ffmpeg compatibility:
  - Avoided `fontcolor_expr` path that rendered invisible text on some builds.
  - File: `backend/app/render_service.py`
- Vibe action preview race fixed:
  - Always queue fresh preview after vibe apply actions.
  - File: `backend/app/routers/vibe.py`

### 2) Transcript quality controls
- Added quality heuristics:
  - low coverage detection
  - suspicious long-gap detection
  - sparse-window detection
- Added model retry and best-candidate selection logic.
- Added Groq prompt controls and fallback behavior.
- Added gap-fill merge strategy that keeps strong primary speech and fills missing windows from retry.
- Main file: `backend/app/transcription_service.py`

### 3) Dynamic behavior (content-adaptive)
- Auto profile detection from silence characteristics:
  - `speech`, `mixed`, `music`
- Profile-specific prompt/retry strategy:
  - speech: conservative retry prompting
  - music: lyric-aware retry prompting
- File: `backend/app/transcription_service.py`
- `detect_silence_ranges` extended to allow limited analysis windows for faster profiling:
  - File: `backend/app/media_utils.py`

### 4) Gap rescue (implemented, currently disabled at runtime)
- Added optional second-pass rescue that re-transcribes unresolved missing windows.
- Added profile-specific rescue chunk sizing (music profile uses smaller windows).
- Added script guard to reduce non-matching/translated rescue tokens.
- File: `backend/app/transcription_service.py`
- Runtime currently disables this (`TRANSCRIBE_ENABLE_GAP_RESCUE=false`) due user preference for cleaner prior behavior.

## Tests status
- Backend tests currently passing: `54 passed`.
- Transcript tests include coverage for:
  - long-gap retry
  - sparse-window retry
  - prompt routing (primary/retry/music/speech)
  - gap-fill acceptance rules
  - gap-rescue call path and profile-aware behavior
- File: `backend/tests/test_transcription_service.py`

## Current runtime transcription config snapshot
Primary relevant runtime flags in `backend/.env`:
- `TRANSCRIBE_BACKEND=groq`
- `TRANSCRIBE_GROQ_MODEL=whisper-large-v3`
- `TRANSCRIBE_GROQ_RETRY_MODEL=whisper-large-v3-turbo`
- `TRANSCRIBE_PROFILE=auto`
- `TRANSCRIBE_GROQ_RETRY_PROMPT_MUSIC=Transcribe speech and sung lyrics verbatim in the original language. Preserve repeated chorus lines and ad-libs. Do not paraphrase. Do not translate.`
- `TRANSCRIBE_GROQ_RETRY_TRY_NO_PROMPT_MUSIC=true`
- `TRANSCRIBE_ENABLE_GAP_RESCUE=false`  (important current choice)

## Last observed behavior notes
- With rescue ON, some runs recovered more missing windows but could add noisy text.
- With rescue OFF (current runtime), transcript was cleaner but may leave some hard windows empty (example around `0:23-0:30` in one F1 run).
- User preferred the cleaner previous behavior over noisier aggressive recovery.

## Files changed in this workstream
- `backend/app/transcription_service.py`
- `backend/app/media_utils.py`
- `backend/app/routers/render.py`
- `backend/app/routers/vibe.py`
- `backend/app/render_service.py`
- `backend/tests/test_transcription_service.py`
- `backend/.env.example`
- Runtime local tuning in `backend/.env`

## Recommended next-session flow
1. Keep baseline as-is (`TRANSCRIBE_ENABLE_GAP_RESCUE=false`) unless user asks for aggressive recovery.
2. For a specific missed region, test with temporary rescue ON on that project, then compare:
   - transcript quality in target window
   - amount of noisy/hallucinated additions
3. If aggressive mode is needed, tune only profile-specific rescue knobs:
   - `TRANSCRIBE_RESCUE_MAX_WINDOW_SEC_MUSIC`
   - `TRANSCRIBE_RESCUE_MAX_CHUNKS_MUSIC`
   - script filter thresholds
4. Re-run subtitle apply and preview render, then confirm exact timestamps with user.

## Operational notes
- Running backend process can be checked with:
  - `ps -ef | rg "uvicorn app.main:app" | rg -v rg`
- API docs:
  - `http://localhost:8000/docs`
- Local env file contains secrets; do not commit `backend/.env`.
