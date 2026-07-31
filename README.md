# ClipMind (Backend + Frontend)

Working implementation of the milestone:

- FastAPI backend with prompt parser, timeline operations, undo/redo, media upload, and render job system.
- React frontend with full editor layout: media bin, prompt console, preview controls, draggable timeline, clip inspector, and render/history panels.
- FFmpeg-based render execution (local) with export presets.

## Repository Layout

- `backend/` API, timeline engine, prompt parsing, render jobs, tests.
- `frontend/` React UI (Vite + TypeScript).
- `docs/` deployment and project notes.

## Branding And Theme

- Product copy and editor naming live in `frontend/src/config/brand.ts`.
- Editor feature presets and static editor config live in `frontend/src/config/editor.ts`.
- Editor palette tokens live at the top of `frontend/src/styles.css`.

## Production Notes

- Secrets are no longer meant to live in tracked config. Use `backend/.env.local` for local secrets and platform env vars or `backend/.env.production` for deploys.
- Docker deployment files live in `backend/Dockerfile`, `frontend/Dockerfile`, `frontend/nginx.conf`, and `docker-compose.yml`.
- CI lives in `.github/workflows/ci.yml`.
- Full deployment instructions are in `docs/deployment.md`.
- Render deployments must use the persistent `/var/data` disk declared in `render.yaml`; Render's default filesystem is ephemeral, so keeping SQLite, uploads, previews, and job payloads outside that disk loses reopened-project media after a restart or deploy.

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
cp .env.local.example .env.local
uvicorn app.main:app --reload --port 8000
```

Key environment variables:

- `DATABASE_URL` (default `sqlite:///./app.db`; use PostgreSQL in production)
- `UPLOAD_DIR`, `RENDER_DIR`, `TMP_DIR`
- `GROQ_API_KEY`, `PEXELS_API_KEY`, `PIXABAY_API_KEY`, `OPENAI_API_KEY`, `BROLL_GENERATIVE_API_KEY`
- `FFMPEG_BIN`, `FFPROBE_BIN`, `YT_DLP_BIN`
- `TRANSCRIBE_MODEL`, `TRANSCRIBE_RETRY_MODEL` (recommended: `small` + `medium` for better quality)
- `TRANSCRIBE_DEVICE=auto` and `TRANSCRIBE_COMPUTE_TYPE=auto` to use CUDA automatically when available
- `TRANSCRIBE_COMPUTE_TYPE_CUDA=float16`, `TRANSCRIBE_COMPUTE_TYPE_CPU=int8` for per-device overrides
- `TRANSCRIBE_ENABLE_QUALITY_RETRY=true` to auto-retry weak transcripts with higher quality settings
- `TRANSCRIBE_ALLOW_MOCK_FALLBACK=false` in production so synthetic fallback transcripts never replace real ASR
- `TRANSCRIBE_MIN_WORDS_PER_SEC` (quality floor; low-word transcripts trigger retry)
- `TRANSCRIBE_RETRY_MIN_DURATION_SEC=90` (retry only for longer videos to avoid slow short-clip transcribes)
- `TRANSCRIBE_BEAM_SIZE`, `TRANSCRIBE_RETRY_BEAM_SIZE`
- `TRANSCRIBE_LOW_CONFIDENCE_THRESHOLD`, `TRANSCRIBE_LOW_CONFIDENCE_RATIO_TRIGGER`, `TRANSCRIBE_LOW_CONFIDENCE_MIN_WORDS`
- `TRANSCRIBE_PREPROCESS_AUDIO=true` enables speech-focused ffmpeg preprocessing before ASR
- `TRANSCRIBE_PREPROCESS_SAMPLE_RATE=16000`, `TRANSCRIBE_PREPROCESS_FILTER_CHAIN=pan=mono|c0=0.5*c0+0.5*c1` tune vocal enhancement filters
- `TRANSCRIBE_VOCAL_ISOLATION_ENABLED=true` enables source separation before cloud transcription
- `TRANSCRIBE_VOCAL_ISOLATION_BACKEND=auto|bs_roformer|mdx23c|api|bs_roformer_api|mdx23c_api` selects separation backend
- `TRANSCRIBE_VOCAL_ISOLATION_PROFILES=speech,mixed,music` controls which detected profiles run isolation (`music,mixed` for faster speech workflows)
- `TRANSCRIBE_VOCAL_ISOLATION_FALLBACKS=command,api` optional failover chain when primary backend fails
- `TRANSCRIBE_VOCAL_ISOLATION_MODEL=mdx23c` (plus optional `..._BS_ROFORMER`, `..._MDX23C`) controls model hints
- `TRANSCRIBE_VOCAL_ISOLATION_COMMAND*` configures local command-based separators (BS-RoFormer/MDX23C wrappers)
- `TRANSCRIBE_VOCAL_ISOLATION_API_URL*`, `TRANSCRIBE_VOCAL_ISOLATION_API_KEY`, `TRANSCRIBE_VOCAL_ISOLATION_API_*_FIELD` configure API-based separation
- `TRANSCRIBE_VOCAL_ISOLATION_DEVICE=auto`, `TRANSCRIBE_VOCAL_ISOLATION_TIMEOUT_SEC=1200` tune separation runtime
- `TRANSCRIBE_VOCAL_ISOLATION_NICE=10` reduces desktop lag during local separation runs
- `TRANSCRIBE_VAD_FILTER=false` (recommended for music-heavy content)
- `TRANSCRIBE_REGENERATE_LOW_QUALITY=true` to auto-refresh weak cached transcripts during automated edit actions
- `TRANSCRIBE_FILLER_AGGRESSIVE_SINGLE_WORDS=false` keeps filler removal conservative (avoids cutting normal words like `right`)
- `TRANSCRIBE_MAX_WORD_DURATION_SEC=1.2` and `TRANSCRIBE_WORD_NEXT_GUARD_SEC=0.01` clamp pathological ASR word timings before subtitle generation
- `TRANSCRIPT_CUT_CONTEXT_SEC` optional keep-context around retained words (for smoother video cuts)
- `TRANSCRIPT_CUT_MIN_REMOVAL_SEC` ignore micro cuts smaller than this many seconds
- `TRANSCRIPT_CUT_MERGE_GAP_SEC` merge adjacent retained ranges when they are very close
- `MAX_CONCURRENT_RENDER_JOBS`, `MAX_CONCURRENT_INGEST_JOBS`
- Subtitle behavior: `karaoke` defaults to word-level captions (1 word per overlay) with anti-overlap timing guards; pass `max_words_per_caption`, `max_gap_sec`, `max_caption_duration_sec`, or `max_caption_display_sec` in action options to override.

Example command backend templates:

```bash
TRANSCRIBE_VOCAL_ISOLATION_BACKEND=bs_roformer
TRANSCRIBE_VOCAL_ISOLATION_COMMAND_BS_ROFORMER='python -m bs_roformer_cli --input {input} --output {output_dir} --model {model} --stem {stem} --device {device}'
TRANSCRIBE_VOCAL_ISOLATION_COMMAND_OUTPUT_BS_ROFORMER=vocals.wav
```

```bash
TRANSCRIBE_VOCAL_ISOLATION_BACKEND=mdx23c
TRANSCRIBE_VOCAL_ISOLATION_COMMAND_MDX23C='python -m mdx23c_cli --input {input} --output {output_dir} --model {model} --stem {stem} --device {device}'
TRANSCRIBE_VOCAL_ISOLATION_COMMAND_OUTPUT_MDX23C=vocals.wav
```

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Optional:

```bash
cp .env.example .env.local
VITE_API_BASE=http://localhost:8000 npm run dev
```

```bash
VITE_REQUEST_TIMEOUT_MS=120000 npm run dev
```

Convenience targets:

```bash
make backend-dev
make backend-test
make frontend-dev
make ci
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

## Docker Deploy

```bash
cp backend/.env.production.example backend/.env.production
docker compose build
docker compose up -d
```

## Notes

- Rendering requires local `ffmpeg` and `ffprobe` binaries in PATH.
- Export pipeline preserves source clip audio and mixes timeline audio tracks with offsets.
- Audio mixing supports track volume/mute/solo and per-clip keyframe volume envelopes.
- This milestone is single-user and no-auth by design. If you expose it publicly, put it behind an auth proxy or private network.
- Database supports SQLite by default and works with PostgreSQL/Supabase through `DATABASE_URL`.
- `POST /api/v1/ingest/url` queues URL ingestion (yt-dlp) into project media.
- `GET /health` reports API status and binary availability (`ffmpeg`, `ffprobe`, `yt_dlp`).
