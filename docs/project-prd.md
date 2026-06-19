# ClipMind Project PRD

## Document Control

- Product: ClipMind AI Video Editor
- Document type: Product Requirements Document
- Version: 1.0
- Date: March 8, 2026
- Product stage: Final-year project / advanced academic prototype
- Primary deployment mode: local development with FastAPI backend and React frontend

## Executive Summary

ClipMind is an AI-assisted video editor focused on transcript-driven editing, multilingual caption generation, B-roll suggestion, and rapid preview rendering for short-form and creator-style content. The project combines a real timeline editor with AI features that are usually split across separate tools: speech transcription, transcript-based cut decisions, styled subtitles, narrative B-roll planning, and export rendering.

The product goal is not to compete with the full breadth of Adobe Premiere Pro, Final Cut Pro, or CapCut. The goal is to demonstrate a coherent, working editor where AI is part of the editing workflow instead of a disconnected assistant. The system is especially aimed at creator workflows where the spoken or sung words drive the edit.

## Problem Statement

Modern video creation workflows still suffer from four practical problems.

1. Editing spoken content is slow when users must manually scrub waveform and video instead of operating on text.
2. Many tools handle English much better than Indian languages, especially for subtitle rendering and mixed-language content.
3. B-roll recommendation systems often produce generic results for non-English or lyric-heavy content because they search directly from raw transcript text instead of first converting meaning into visual concepts.
4. Users frequently need multiple tools to finish one workflow: one for transcription, one for subtitle styling, one for AI B-roll, and another for final editing.

ClipMind addresses those gaps by building a single editing pipeline around the transcript, the timeline, and AI-assisted visual planning.

## Product Vision

Create a video editor where words are the primary control surface, timeline editing remains visible and editable, and AI accelerates repetitive work without hiding the underlying edit decisions.

## Target Users

### Primary Users

- Student creators making short-form videos, explainers, lyric edits, and social clips
- Solo content creators editing talking-head videos, podcasts, interviews, and music edits
- Final-year project evaluators who need to see a technically meaningful AI editing workflow

### Secondary Users

- Researchers interested in transcript-based multimedia editing
- Developers who want a modular open prototype instead of a black-box SaaS product

## Core User Personas

### Persona 1: The Short-Form Creator

- Creates Reels, Shorts, and lyric videos
- Needs fast subtitle generation and visual rhythm
- Cares about B-roll variety, caption style, and short export turnaround

### Persona 2: The Podcast Editor

- Works with speech-heavy content
- Wants text-first editing of pauses, filler words, and rough transcript issues
- Needs preview confidence before export

### Persona 3: The Academic Demonstrator

- Needs a system that can be explained end to end
- Values architecture clarity, reproducibility, and measurable features
- Needs a defensible novelty claim beyond "we used AI"

## Product Goals

### Primary Goals

- Provide an end-to-end local editing workflow from video upload to preview/export
- Enable transcript generation and transcript-driven editing for speech-centric media
- Support styled captions with correct rendering for major Indian languages
- Generate B-roll suggestions from narrative meaning, not only literal keywords
- Keep all AI decisions editable inside the timeline workflow

### Secondary Goals

- Maintain clean project structure and persistent timeline state
- Support repeatable preview rendering through background jobs
- Provide enough transparency for viva, demonstration, and documentation

## Non-Goals

- Full parity with professional NLEs such as Premiere Pro or DaVinci Resolve
- Enterprise collaboration, cloud team workflows, or multi-user synchronization
- Perfect automatic transcription for every world language and every music genre
- Frame-accurate professional color grading and audio mastering
- Fully automated long-form documentary editing without human review

## Scope

### In Scope

- Project creation
- Video upload and URL ingest
- Transcript generation
- Transcript viewing and word-level editing
- Transcript-based cut workflows
- AI vibe actions such as auto cut pauses, trim start/end, and add subtitles
- Caption preset styling and rendering
- B-roll planning, rerolling, choosing, syncing, and timeline editing
- Preview rendering and export jobs
- Timeline lane editing for clips, overlays, captions, and B-roll

### Out of Scope

- Multi-cam editing
- Team commenting or approvals
- Cloud-hosted inference orchestration
- Fine-grained keyframe animation UI for every transform parameter
- Professional audio mixing buses and plugin chains

## Product Principles

1. The transcript is a first-class editing artifact, not only metadata.
2. AI outputs must stay editable by the user.
3. The timeline must remain visible and understandable.
4. Multilingual handling matters, especially for Indian creator workflows.
5. Preview speed matters because users iterate through edits rapidly.

## End-to-End User Flow

### Flow 1: Basic Transcript-Driven Editing

1. User creates a project.
2. User uploads a video or ingests a URL.
3. User chooses source language or leaves auto mode enabled.
4. System generates transcript and attaches the media to timeline.
5. User edits transcript text, deletes filler sections, and trims content.
6. Timeline updates reflect transcript-driven edits.
7. User renders preview and exports final output.

### Flow 2: Caption-Centric Creator Workflow

1. User uploads or ingests source video.
2. User generates transcript.
3. User selects a caption preset and applies captions.
4. User previews subtitle style and language rendering.
5. User refines edits, then renders and exports.

### Flow 3: AI B-roll Workflow

1. User generates transcript.
2. User opens B-roll Studio.
3. System creates narrative beats and candidate visuals.
4. User inspects original lyric/transcript, English gloss, and search queries.
5. User rerolls or overrides meaning if suggestions are weak.
6. User syncs chosen B-roll to timeline.
7. User drags and trims B-roll on the timeline.

## Functional Requirements

### FR-1 Project Management

- The system shall allow the user to create a project with default frame rate and resolution.
- The system shall persist project timeline state in the backend database.
- The system shall support undo and redo through timeline versions.

### FR-2 Media Ingest

- The system shall upload local video files.
- The system shall support URL ingest as a background job.
- The system shall store media metadata including duration, file path, and media type.

### FR-3 Transcript Generation

- The system shall generate transcripts for uploaded video assets.
- The system shall support source language selection and auto mode.
- The system shall persist transcript text, word timings, language, and source metadata.
- The system shall reuse recent valid transcripts when language matches and reuse is enabled.

### FR-4 Transcript Editing

- The system shall display word-level transcript timing.
- The system shall support word selection, range selection, patch editing, and transcript-based cut actions.
- The system shall identify low-confidence or suspicious regions for review.
- The system shall expose transcript regions and quality signals to the UI.

### FR-5 Vibe Actions

- The system shall provide one-click editing actions including auto cut pauses, trim start/end, and add captions.
- Each vibe action shall update timeline state and trigger a preview job.
- Caption generation shall use the language selected by the user rather than silently reusing an incompatible transcript.

### FR-6 Caption System

- The system shall support multiple caption presets.
- Caption rendering shall work in both live preview context and final preview/export renders.
- Indic-language captions shall use safe font fallback and shaping-aware rendering.
- The system shall preserve style identity where possible without breaking complex-script shaping.

### FR-7 B-roll Planning and Retrieval

- The system shall segment transcript content into candidate B-roll beats.
- The system shall generate English visual gloss for non-English text when retrieving or generating B-roll.
- The system shall expose original transcript line, English gloss, and search query reasoning to the user.
- The system shall support reroll, reject, choose, and manual gloss override actions.

### FR-8 Timeline Editing

- The system shall provide a timeline with video, audio, captions, waveform, and B-roll lanes.
- The system shall support drag-to-move and drag-to-trim for timeline clips.
- The system shall support B-roll drag, trim, opacity change, delete, and reroll.
- The system shall show thumbnail-based visual blocks rather than only abstract bars where practical.

### FR-9 Rendering and Export

- The system shall queue preview and export jobs.
- The system shall expose job status, progress, and downloadable outputs.
- The system shall support at least 9:16 and 16:9 export modes.
- The render path shall use FFmpeg and ASS subtitle generation for performant caption burning.

## Non-Functional Requirements

### NFR-1 Reliability

- Transcript, timeline, and B-roll state must be persisted to the database.
- Orphaned jobs must be marked failed on restart.
- Preview rendering should not corrupt project state if the render fails.

### NFR-2 Performance

- Preview rendering must prioritize usability over maximum visual quality.
- Timeline interactions should feel responsive under normal short-form project sizes.
- AI actions should be asynchronous where they may take significant time.

### NFR-3 Explainability

- AI-assisted B-roll generation should expose intermediate reasoning to the user.
- Transcript quality issues should be visible in the UI.
- The system architecture should remain understandable for academic evaluation.

### NFR-4 Extensibility

- Backend routes should remain modular by domain.
- Timeline operations should be represented as structured operations rather than ad hoc mutations.
- New caption presets and new language handling should be introducible without redesigning the whole stack.

## Supported Language Strategy

The current project is optimized for English and major Indian languages. The strongest experience target is:

- English
- Hindi
- Kannada
- Tamil
- Telugu
- Malayalam

Additional UI language options exist for Assamese, Nepali, Marathi, Bengali, Gujarati, Punjabi, Odia, and Urdu. These are supported as part of the current scope, but not all are equally validated for song-heavy content.

Auto mode is a convenience path, not a guarantee of perfect results for every language and every song. Explicit language selection remains the safer path for non-English music.

## Data Model Summary

### Core Entities

- `Project`: project identity and default canvas metadata
- `Timeline`: current editable timeline state
- `TimelineVersion`: undo/redo history
- `MediaAsset`: uploaded or ingested media
- `Transcript`: transcript payload and word timing data
- `BrollSlot`: editable B-roll placement opportunities
- `BrollCandidate`: candidate visuals for each slot
- `BrollChoice`: user decisions on B-roll candidates
- `Job`: background task tracking
- `JobEvent`: detailed progress logs

## System Architecture Summary

### Frontend

- React + TypeScript single-page editor
- Central `App.tsx` orchestration layer
- Dedicated `Timeline` component for lane rendering and interactions
- API client abstraction in `frontend/src/lib/api.ts`

### Backend

- FastAPI application
- SQLModel for persistence
- FFmpeg-based media processing and rendering
- Router-based domain separation for transcript, B-roll, vibe actions, timeline, render, media, and prompt workflows

## Key UX Requirements

- Language controls must be visible and readable.
- Timeline must look like an editor, not a debug canvas.
- Original transcript text must remain visible during B-roll reasoning.
- Caption styles must remain understandable across preview and export paths.
- Important AI state should not be hidden behind backend-only assumptions.

## Risks

### Risk 1: Automatic transcription failures on song-heavy or mixed-language audio

Mitigation:

- Explicit language selector
- transcription retries
- language-aware probing
- transcript review workflow

### Risk 2: Subtitle rendering failures in complex scripts

Mitigation:

- ASS subtitle rendering
- Indic font fallback
- shaping-safe styling rules

### Risk 3: Generic or semantically weak B-roll

Mitigation:

- beat-level planning
- English gloss conversion
- user override loop
- timeline reroll support

### Risk 4: Timeline feels like a prototype instead of an editor

Mitigation:

- visual filmstrip blocks
- draggable B-roll clips
- track controls
- caption and overlay lanes

## Success Metrics

### Product Success Metrics

- User can go from upload to preview without leaving the application.
- Transcript can drive at least one meaningful editing workflow.
- Captions render correctly for the main demo languages.
- B-roll generation is understandable and user-correctable.
- Preview job system demonstrates repeatable asynchronous processing.

### Demo Success Metrics

- One complete talking-head workflow works end to end.
- One non-English caption workflow works end to end.
- One B-roll workflow demonstrates human-in-the-loop correction.

## Acceptance Criteria

1. A user can create a project, upload video, generate transcript, and see the media on the timeline.
2. A user can edit transcript content and cause the timeline to update accordingly.
3. A user can add styled captions and see them in preview renders.
4. A user can generate B-roll suggestions, inspect reasoning, and sync selected clips to timeline.
5. A user can drag and trim B-roll on the timeline.
6. A user can render preview and export output through background jobs.
7. The system preserves timeline and transcript data across refreshes and restarts.

## Known Limitations

- Auto transcription is not perfect for all music and all world languages.
- The editor is optimized for short-form and creator-style workflows, not full feature-film editing.
- B-roll quality depends on transcript quality and available retrieval/generation sources.
- Some advanced workflows still require manual review, especially for mixed-language or lyric-heavy media.

## Future Work

- Stronger multi-pass auto transcription scoring for songs
- explicit support for more world languages such as Russian, Spanish, Arabic, and Portuguese
- richer timeline layering and keyframe editing
- stronger B-roll reranking by mood, genre, and narrative role
- collaboration, annotations, and cloud-hosted inference

## Final Positioning Statement

ClipMind should be positioned as an integrated AI-assisted editing prototype that demonstrates how transcript-driven editing, multilingual captions, semantic B-roll planning, and timeline operations can work together inside one system. Its academic strength is the integration and language-aware workflow design, not the invention of a brand-new base model.
