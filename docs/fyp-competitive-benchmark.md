# FYP Competitive Benchmark and Build Plan

**Project**: Prompt-first AI short-form video editor  
**Date**: 2026-02-15  
**Goal**: Reduce short-form editing from hours to minutes for creators.

## 1. Problem We Are Solving

Short-form creators spend most of their time on:

1. Finding strong moments in long videos.
2. Reframing for vertical formats.
3. Captioning, subtitle cleanup, and pacing.
4. Repeating the same export/publish steps per platform.

This project should automate these steps while keeping creator control.

## 2. Market + Repo Benchmark (What Existing Tools Already Do)

## OpusClip (commercial benchmark)

Observed capabilities from official pages:

1. Long-video to multi-clip generation and social-ready output.
2. Auto-captioning and auto-reframing workflows.
3. Multi-source ingestion support (YouTube and cloud links listed on site).
4. Publish workflow positioning as near one-click.

Why this matters for us:

1. Sets user expectation for speed and output polish.
2. Baseline UX is "upload/paste link -> get multiple clips quickly".

## Magic Clips (tool category benchmark)

Riverside Magic Clips positioning:

1. AI highlight detection for short clips.
2. Platform-ready formats and transcript-assisted editing.
3. Caption + branding customization focus.

Why this matters for us:

1. Users expect automatic first draft plus editable transcript flow.

## OpenShorts (open-source benchmark)

Repository signals:

1. Viral moment detection with Faster-Whisper + Gemini.
2. Dual-mode vertical framing/tracking and fallback layout strategy.
3. Optional direct social posting and S3 backup.
4. Optional dubbing + multilingual flow.

Why this matters for us:

1. Strong reference for open-source end-to-end automation.
2. Good example of practical pipeline integration over perfect theory.

## ClippedAI (open-source benchmark)

Repository signals:

1. Local-first positioning and "free alternative" positioning.
2. Smart clip detection + 9:16 auto-resize + subtitle styling.
3. Engagement scoring + cached transcriptions.
4. Hardware/model tuning guidance in README.

Why this matters for us:

1. Demonstrates users care about cost/privacy and repeat-run efficiency.

## AI-short-creator (open-source benchmark)

Repository signals:

1. Focus on turning long multi-speaker videos into shorts.
2. Captions/transitions with simple setup (Python + Remotion).
3. Lightweight, practical starter architecture.

Why this matters for us:

1. Useful baseline for rapid prototyping of clip + caption workflows.

## Vinci Clips (open-source benchmark)

Repository signals:

1. AI analysis + transcription + diarization + smart clip generation.
2. Queue/status tracking and batch upload handling.
3. Full-stack product architecture (Next.js + Express + MongoDB + FFmpeg + cloud storage).

Why this matters for us:

1. Good reference for production-style architecture and workflow UX.

## Paper: arXiv 2509.16811 (research benchmark)

Key ideas:

1. Prompt-driven editing system for long-form narrative media.
2. Semantic indexing pipeline with temporal segmentation, memory compression, and cross-granularity fusion.
3. Explicit focus on narrative coherence and interpretable intermediate artifacts.
4. Evaluation includes quality/usability and multiple study designs on 400+ videos.

Why this matters for us:

1. This is the strongest direction for your **academic differentiation** (FYP value beyond "just another clipper").

## 3. Gaps in Current Repo vs Target Product

Current repo is already strong on timeline editing and FFmpeg rendering, but key gaps remain:

1. No long-video ingestion workflow (URL import + preprocessing pipeline).
2. No automatic clip candidate discovery/ranking pipeline.
3. No narrative/semantic index for prompt-grounded retrieval.
4. No caption quality pipeline for "viral-ready" defaults.
5. No "generate N variants and choose best" ranking loop.
6. No evaluation harness proving time saved and quality.

## 4. What We Need to Build (Prioritized List)

## Priority A: Must-have (MVP core)

1. Ingestion
   - Local upload + YouTube URL ingestion.
   - Metadata capture and source tracking.
2. Understanding
   - Faster-Whisper transcription with timestamps.
   - Shot/scene segmentation and speaker diarization (optional mode if compute is limited).
3. Candidate generation
   - Sliding-window candidate clips.
   - Hook/engagement scoring and ranking.
4. Auto formatting
   - 9:16 reframing with face/person tracking + fallback blur layout.
   - Caption styles with sane defaults.
5. Batch export
   - Generate top N clips in one job and return downloadable outputs.
6. Reliable orchestration
   - Queue, retries, stage progress events, per-job logs.

## Priority B: FYP differentiation (research + UX value)

1. Prompt-to-plan agent
   - Convert prompt into structured edit intent (tone, audience, length, topic).
2. Semantic index
   - Segment-level summaries, entities, events, emotion tags, timestamp links.
3. Retrieval + composition
   - Map prompt intent to ranked candidate segments.
   - Build explainable edit plan and render trace.
4. Transparent UX
   - Show why clips were selected ("hook", "emotion peak", "topic match").

## Priority C: Product polish

1. One-click presets (Reels, Shorts, TikTok).
2. Variant generation (fast-paced, emotional, educational, etc.).
3. Prompt refinement loop ("shorter", "more energetic", "remove clip 2").
4. Music and basic beat alignment.

## 5. Execution Plan (Build Together)

## Sprint 1 (stabilize platform)

1. Finalize queue + stage events + cancellation/retry foundations.
2. Add API endpoints for job events/logging and wire UI progress timeline.
3. Add regression tests for render queue behavior.

## Sprint 2 (auto-clip MVP)

1. Add ingestion endpoint for YouTube URL.
2. Implement transcription + candidate generation pipeline.
3. Add "Generate clips" UI with top-N outputs.

## Sprint 3 (quality + control)

1. Add auto-reframe + caption styling pipeline.
2. Add clip score breakdown and user-override controls.
3. Add preset-based export bundles.

## Sprint 4 (FYP differentiation)

1. Add semantic index + prompt planning layer.
2. Add retrieval-to-edit-plan pipeline.
3. Add intermediate artifact visualization in UI.

## 6. FYP Evaluation Checklist (what to prove in report/demo)

1. Time reduction: manual workflow vs your system.
2. Output quality: human ratings (hook quality, coherence, watchability).
3. Prompt reliability: % prompts that produce acceptable edits.
4. System reliability: failure rate and median processing time by stage.
5. User control/trust: how often users accept first output vs refine.

## 7. Immediate Next Build Tasks (starting now)

1. Finish backend queue orchestration wiring and tests.
2. Add `/jobs/{id}/events` consumption in frontend render panel.
3. Implement ingestion stub (`/ingest/url`) + task pipeline skeleton.
4. Add transcription service abstraction (`faster-whisper` first).
5. Add first candidate scoring implementation (heuristic baseline).

## 8. Sources

1. Paper: https://arxiv.org/abs/2509.16811  
2. Paper HTML: https://arxiv.org/html/2509.16811v1  
3. OpusClip official site: https://www.opus.pro/  
4. OpusClip API page: https://www.opus.pro/api  
5. Riverside Magic Clips: https://riverside.com/magic-clips  
6. OpenShorts repo: https://github.com/mutonby/openshorts  
7. ClippedAI repo: https://github.com/Shaarav4795/ClippedAI  
8. AI-short-creator repo: https://github.com/shreesha345/AI-short-creator  
9. Vinci Clips repo: https://github.com/tryvinci/vinci-clips  

