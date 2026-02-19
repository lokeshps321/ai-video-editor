# Prompt Video Editor (Backend + Frontend)

Working implementation of the milestone:

- FastAPI backend with prompt parser, timeline operations, undo/redo, media upload, and render job system.
- React frontend with full editor layout: media bin, prompt console, preview controls, draggable timeline, clip inspector, and render/history panels.
- FFmpeg-based render execution (local) with export presets.

## Repository Layout

- `backend/` API, timeline engine, prompt parsing, render jobs, tests.
- `frontend/` React UI (Vite + TypeScript).

## Implemented Basic Features (15)

1. Cut, Trim, Split (`trim_clip`, `split_clip`, `delete_clip`)
2. Merge / Join (`merge_clips` with seamless ripple)
3. Transitions (`set_transition`, rendered with dissolve/slide/zoom/wipe via FFmpeg xfade)
4. Text / Title overlays (`add_text_overlay`, static + animated caption styles)
5. Add music / audio tracks (`add_audio_track`)
6. Volume / audio controls (`set_volume`, fade in/out, mute, keyframes, track mute/solo/volume)
7. Speed control (`set_speed`)
8. Crop / resize (`crop_resize`, aspect)
9. Rotate / flip (`rotate_clip`, `flip_clip`)
10. Basic filters / adjustments (`set_adjustments`)
11. Undo / redo history (`/projects/{id}/undo`, `/redo`)
12. Timeline model and multi-track editing (`video` + `audio` tracks, drag reorder, move, snap, ripple)
13. Preview player workflow (`render preview` job + frontend player, frame-step, loop region)
14. Import and media management (`/media/upload`, `/media`)
15. Export and render presets (`/render/export`)

## Prompt Command Examples

- `trim clip 1 from 00:05 to 00:12`
- `split clip 1 at 00:08`
- `merge clips 1,2,3`
- `add text "New drop" at 00:02 for 3s`
- `set aspect 9:16`
- `transition clip 1 dissolve 0.6s`
- `speed clip 1 to 1.5x`
- `fade in clip 1 0.5s`
- `rotate clip 1 90`
- `flip clip 1 horizontal`
- `crop clip 1 0 0 720 1280`
- `set brightness clip 1 0.2`
- `set saturation clip 1 1.2`
- `set volume clip 1 0.8`
- `track audio volume 0.7`
- `audio fade out clip 1 1.0s`
- `mute clip 1`
- `solo track audio`
- `move clip 2 to 00:05`
- `delete clip 2`
- `export 1080p 30fps high mp4`

History endpoint:

- `GET /api/v1/timeline/history?project_id=<id>`

## Backend Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Key environment variables:

- `DATABASE_URL` (default `sqlite:///./app.db`; set Supabase/Postgres URL here if needed)
- `UPLOAD_DIR`, `RENDER_DIR`, `TMP_DIR`
- `FFMPEG_BIN`, `FFPROBE_BIN`, `YT_DLP_BIN`
- `TRANSCRIBE_MODEL`, `TRANSCRIBE_RETRY_MODEL` (recommended: `small` + `medium` for better quality)
- `TRANSCRIBE_DEVICE=auto` and `TRANSCRIBE_COMPUTE_TYPE=auto` to use CUDA automatically when available
- `TRANSCRIBE_COMPUTE_TYPE_CUDA=float16`, `TRANSCRIBE_COMPUTE_TYPE_CPU=int8` for per-device overrides
- `TRANSCRIBE_ENABLE_QUALITY_RETRY=true` to auto-retry weak transcripts with higher quality settings
- `TRANSCRIBE_ALLOW_MOCK_FALLBACK=true` to permit synthetic fallback transcripts when ASR model loading fails
- `TRANSCRIBE_MIN_WORDS_PER_SEC` (quality floor; low-word transcripts trigger retry)
- `TRANSCRIBE_RETRY_MIN_DURATION_SEC=90` (retry only for longer videos to avoid slow short-clip transcribes)
- `TRANSCRIBE_BEAM_SIZE`, `TRANSCRIBE_RETRY_BEAM_SIZE`
- `TRANSCRIBE_LOW_CONFIDENCE_THRESHOLD`, `TRANSCRIBE_LOW_CONFIDENCE_RATIO_TRIGGER`, `TRANSCRIBE_LOW_CONFIDENCE_MIN_WORDS`
- `TRANSCRIBE_PREPROCESS_AUDIO=true` enables speech-focused ffmpeg preprocessing before ASR
- `TRANSCRIBE_PREPROCESS_SAMPLE_RATE=16000`, `TRANSCRIBE_PREPROCESS_FILTER_CHAIN=pan=mono|c0=0.5*c0+0.5*c1` tune vocal enhancement filters
- `TRANSCRIBE_VAD_FILTER=false` (recommended for music-heavy content)
- `TRANSCRIBE_REGENERATE_LOW_QUALITY=true` to auto-refresh weak cached transcripts during vibe actions
- `TRANSCRIPT_CUT_CONTEXT_SEC` optional keep-context around retained words (for smoother video cuts)
- `TRANSCRIPT_CUT_MIN_REMOVAL_SEC` ignore micro cuts smaller than this many seconds
- `TRANSCRIPT_CUT_MERGE_GAP_SEC` merge adjacent retained ranges when they are very close
- `MAX_CONCURRENT_RENDER_JOBS`, `MAX_CONCURRENT_INGEST_JOBS`

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Optional:

```bash
VITE_API_BASE=http://localhost:8000 npm run dev
```

Convenience targets:

```bash
make backend-dev
make backend-test
make frontend-dev
```

## Test Backend

```bash
cd backend
pytest
```

## Build Frontend

```bash
cd frontend
npm run build
```

## Notes

- Rendering requires local `ffmpeg` and `ffprobe` binaries in PATH.
- Export pipeline preserves source clip audio and mixes timeline audio tracks with offsets.
- Audio mixing supports track volume/mute/solo and per-clip keyframe volume envelopes.
- This milestone is single-user and no-auth by design.
- Database supports SQLite by default and works with PostgreSQL/Supabase through `DATABASE_URL`.
- `POST /api/v1/ingest/url` queues URL ingestion (yt-dlp) into project media.
- `GET /health` reports API status and binary availability (`ffmpeg`, `ffprobe`, `yt_dlp`).
