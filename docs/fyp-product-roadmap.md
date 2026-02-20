# FYP Product Direction and Build Roadmap

**Project**: Prompt-first AI video editor for short-form content creators  
**Date**: 2026-02-15  
**Owner**: Lokesh (Final Year Project)  

## 1. Context

This project exists to solve one core problem:

- Short-form creators spend hours converting long or raw footage into publish-ready clips.
- We need to reduce that editing time to minutes while keeping quality high enough for real posting.

Your target is to build something in the same space as:

- OpusClip / Magic Clip style auto clipping.
- Open-source tools like OpenShorts, ClippedAI, Vinci Clips, AI-short-creator.
- Research direction from `arXiv:2509.16811` (prompt-driven agentic long-form video editing with semantic indexing).

## 2. Product Goal

Build a system where a creator can do this:

1. Upload a long video (or paste a URL).
2. Type one prompt (for example: "Create 5 viral shorts for startup founders, energetic tone").
3. Receive multiple captioned, reframed, social-ready clips with minimal manual editing.

## 3. What "Better" Means

To be better than current alternatives, we should optimize for:

1. **Speed**: first usable clip in less than 5 minutes for normal inputs.
2. **Control**: users can refine outputs with natural language and selective manual edits.
3. **Quality**: clip choices feel coherent, not random cuts.
4. **Transparency**: users can inspect intermediate reasoning artifacts (storyboard, selected moments, narration plan).
5. **Cost and openness**: modular architecture, optional local stack, swappable models.

## 4. Primary Users

1. Solo short-form creators (YouTube Shorts, Reels, TikTok).
2. Student creators and beginner editors with low editing skills.
3. Small social media teams producing high clip volume.

## 5. Current Baseline in This Repo

Already built:

1. Prompt-driven timeline editor (trim, split, transitions, overlays, audio controls, filters).
2. Media upload and project management.
3. Render preview/export with FFmpeg.
4. Job system and history.
5. Frontend timeline + inspector + prompt console.

Gap to target product:

1. No full long-form ingestion pipeline (URL download + preprocessing).
2. No automatic clip discovery/ranking for viral moments.
3. No semantic narrative index (characters/events/emotions with robust timestamps).
4. No multi-agent planning/retrieval/render orchestration.
5. No social publishing flow.

## 6. Master Build List (Priority Ordered)

## Phase P0 - Foundation hardening (must stabilize first)

- [ ] Add robust background worker queue (Celery/RQ/Temporal-like pattern) with retries, cancellation, and concurrency limits.
- [ ] Add job states and progress granularity per stage (ingest, ASR, detect, rank, render).
- [ ] Add storage abstraction (local now, object storage later).
- [ ] Add test fixtures for media pipeline and deterministic smoke tests.
- [ ] Add performance logging per job stage.

## Phase P1 - Auto clipping MVP (Opus-style core)

- [ ] Add input ingestion from:
  - [ ] local upload
  - [ ] YouTube URL via `yt-dlp`
- [ ] Add transcription pipeline (faster-whisper).
- [ ] Add speaker diarization (pyannote or equivalent optional mode).
- [ ] Add shot/scene segmentation.
- [ ] Build clip candidate generator:
  - [ ] candidate windows (e.g., 20-90 sec)
  - [ ] hooks and engagement feature extraction
  - [ ] ranking score
- [ ] Build automatic 9:16 reframing:
  - [ ] face/person tracking
  - [ ] fallback blur background mode
- [ ] Add animated caption generation (word or phrase level timing).
- [ ] Generate N clips in one run with downloadable outputs.

## Phase P2 - Prompt-first agentic editing (research-aligned differentiator)

- [ ] Create semantic index schema:
  - [ ] segment summaries
  - [ ] entities/characters
  - [ ] key events and emotions
  - [ ] timestamped scene traces
- [ ] Add planner agent:
  - [ ] converts user prompt into structured storyboard + constraints
- [ ] Add retrieval agent:
  - [ ] aligns storyboard beats to timestamped clips
- [ ] Add narration/script agent:
  - [ ] optional voiceover script generation
- [ ] Add rendering agent:
  - [ ] compile edit plan to FFmpeg graph + overlays
- [ ] Expose intermediate artifacts in UI:
  - [ ] storyboard
  - [ ] selected scenes and reasons
  - [ ] final edit plan JSON

## Phase P3 - Creator workflow and product polish

- [ ] One-click output presets by platform (Shorts/Reels/TikTok).
- [ ] A/B variant generation (tone, pace, hook style).
- [ ] Prompt refinement loop ("make it faster", "more emotional", "remove clip 3").
- [ ] Music recommendation and beat alignment.
- [ ] Template packs (subtitles, intro/outro, CTA styles).
- [ ] Project versioning and rollback.

## Phase P4 - Distribution and feedback loop

- [ ] Social publishing integrations.
- [ ] Content calendar export.
- [ ] Analytics ingestion (watch time proxy metrics where available).
- [ ] Learning loop to improve ranking model from performance outcomes.

## 7. Required System Modules

1. **Ingestion Service**: uploads, URL download, metadata extraction.
2. **Understanding Service**: ASR, diarization, scene detection, semantic extraction.
3. **Ranking Service**: candidate generation and scoring.
4. **Planning Service**: prompt to storyboard/edit intent.
5. **Retrieval Service**: map intent to timeline segments.
6. **Render Service**: FFmpeg assembly, captions, overlays, exports.
7. **Orchestration Layer**: workflow engine, retries, checkpoints.
8. **Creator UI**: prompt-first flow + inspect/edit + export.
9. **Evaluation Layer**: quality scoring, latency tracking, failure analysis.

## 8. Suggested Data Model Additions

Add tables (or collections) for:

1. `source_videos` (input metadata, duration, platform source).
2. `transcripts` (utterances, word timestamps, speaker IDs).
3. `segments` (scene boundaries and descriptors).
4. `semantic_index` (events, entities, emotions, summaries, references).
5. `clip_candidates` (start/end, score breakdown, rationale).
6. `edit_plans` (storyboard + retrieval map + render settings).
7. `deliverables` (rendered outputs and platform targets).
8. `quality_evaluations` (auto checks and human ratings).

## 9. API Roadmap (high-level)

1. `POST /api/v1/ingest/url`
2. `POST /api/v1/ingest/upload`
3. `POST /api/v1/analyze`
4. `GET /api/v1/index/{project_id}`
5. `POST /api/v1/clips/suggest`
6. `POST /api/v1/agent/plan`
7. `POST /api/v1/agent/compose`
8. `POST /api/v1/render/batch`
9. `POST /api/v1/publish/{platform}`

## 10. FYP Evaluation Plan (for final report/demo)

Measure:

1. **Time saved**: manual baseline vs tool-assisted workflow.
2. **Clip quality**: human ratings (hook strength, coherence, watchability).
3. **Temporal correctness**: alignment between transcript events and selected clips.
4. **Usability**: task completion and SUS-like questionnaire.
5. **System performance**: median stage latency and failure rate.

## 11. Scope Control (important for final year timeline)

Must-have for a strong FYP demo:

1. End-to-end from long video to multiple shorts.
2. Prompt-driven regeneration/refinement.
3. Captions + reframing + export presets.
4. At least one interpretable intermediate artifact in UI.
5. Evaluation section with real user/testing evidence.

Can defer:

1. Full social posting integrations.
2. Complex collaboration/auth.
3. Heavy model finetuning.

## 12. Build Sequence We Should Start With

Immediate order:

1. Stabilize worker/job orchestration.
2. Implement ingestion + ASR + segmentation.
3. Implement clip candidate scoring + batch render.
4. Add prompt planner to choose/arrange candidates.
5. Add UI for suggestions and one-click batch export.

---

## Decision Log (initial)

1. Product type: prompt-first short-form repurposing editor.
2. Target: reduce hours of work into minutes.
3. Differentiator: combine Opus-style speed with agentic narrative understanding and transparent intermediate artifacts.
4. Delivery strategy: MVP first (auto clipping), then agentic layer, then polish.

---

## References Used

1. Paper: `https://arxiv.org/abs/2509.16811` and `https://arxiv.org/html/2509.16811v1`
2. OpenShorts: `https://github.com/mutonby/openshorts`
3. ClippedAI: `https://github.com/Shaarav4795/ClippedAI`
4. AI-short-creator: `https://github.com/shreesha345/AI-short-creator`
5. Vinci Clips: `https://github.com/tryvinci/vinci-clips`
6. OpusClip (product references): `https://www.opus.pro/`
7. B-roll feature plan (transcript-safe rollout): `docs/broll-feature-research-and-uiux-plan-2026-02-19.md`
