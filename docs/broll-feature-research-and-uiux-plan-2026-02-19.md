# B-roll Feature Research + UI/UX Product Plan

**Project**: Prompt Video Editor  
**Date**: 2026-02-19  
**Scope**: Add B-roll as the next feature without breaking transcript-first editing.

## 1. Non-Negotiables

1. Transcript editing remains the primary workflow.
2. B-roll is additive, never destructive to transcript cuts.
3. Every B-roll insertion must be reversible.
4. Default behavior must still render correctly when no B-roll exists.

## 2. What Current Tools Do (and Gaps We Can Beat)

## Tool patterns observed

1. **Script-first editing + stock replacement**
   - Descript supports document-style editing and media replacement.
   - Strength: very low learning curve.
   - Gap: weaker transparent scoring for *why* each B-roll shot was suggested.

2. **Template-driven automatic B-roll insertion**
   - VEED provides transcript editing and automatic B-roll insertion workflows.
   - Strength: fast first draft.
   - Gap: low explainability and weak fine-grained control at word/phrase alignment level.

3. **AI B-roll suggestions from transcript context**
   - OpusClip exposes AI B-roll insertion flow tied to clip context and platform use.
   - Strength: creator-speed UX.
   - Gap: hard to inspect decision rationale and coverage quality.

4. **Integrated AI suite with B-roll tooling**
   - Riverside surfaces AI B-roll inside a broader AI workflow.
   - Strength: one place for many post-production actions.
   - Gap: less editorial precision than dedicated timeline-first tools.

5. **Professional baseline for text-based editing**
   - Adobe Premiere Pro has mature text-based editing.
   - Strength: pro controls and reliability.
   - Gap opportunity for us: simpler AI-native B-roll suggestions with better transcript coupling.

## Product opportunity

To beat current tools, we should combine:

1. Transcript-first speed (what users already like in your app).
2. Explainable B-roll suggestions (score + reason tags).
3. Fine-grained control (accept/reject/replace/lock per transcript chunk).
4. Multi-option alternatives per slot (not a single AI guess).

## 3. Research Findings That Should Drive Our Design

## Papers and practical implications

1. **B-Script (CHI 2019)**
   - Shows transcript-driven B-roll recommendation is practical and reduces search/edit effort.
   - Implication: use transcript concepts + keyword relevance as first-class ranking signals.

2. **QuickCut (Stanford Graphics)**
   - Uses transcript and semantic cues to accelerate edit creation.
   - Implication: keep language/semantic metadata attached to timeline segments.

3. **Computational Video Editing for Dialogue-Driven Scenes (SIGGRAPH 2017)**
   - Demonstrates programmatic editing constraints can preserve narrative coherence.
   - Implication: enforce coverage/continuity rules when inserting B-roll.

4. **ChunkyEdit (CHI 2024)**
   - Advocates chunk-level editing to reduce cognitive load vs token-level interactions.
   - Implication: B-roll insertion points should map to sentence/phrase chunks, not single words by default.

5. **VideoDiff (CHI 2025)**
   - Emphasizes alternative candidates and comparison for user trust.
   - Implication: each B-roll slot should expose multiple ranked options (A/B/C), not only one suggestion.

## 4. UX Direction for Your Product

## Keep current mental model

1. Transcript panel stays where it is and remains the source of truth for story flow.
2. Timeline keeps existing word/audio behavior.
3. B-roll appears as an **additional visual layer** controlled by transcript-linked slots.

## New UI module: B-roll Studio (right-side panel)

1. **Tab A: Suggestions**
   - List of B-roll slots generated from transcript chunks.
   - Each slot shows:
     - time range
     - extracted concept(s)
     - confidence score
     - reason tags (`keyword_match`, `entity_match`, `visual_variety`, `avoid_repetition`)

2. **Tab B: Alternatives**
   - 3-5 candidates per slot with thumbnails.
   - Actions: `Accept`, `Replace`, `Pin`, `Reject`.

3. **Tab C: Library**
   - Uploaded assets + stock + generated clips in one unified picker.

4. **Timeline behavior**
   - Add visible B-roll lane above main words/video lane.
   - Drag handles to trim B-roll in/out while keeping linked transcript anchor.
   - Conflict markers when B-roll overlaps locked regions.

## UX guardrails (important)

1. No auto-delete of transcript content when applying B-roll.
2. B-roll auto-insert only changes overlay lanes, never the main transcript cut state.
3. Undo/redo includes B-roll operations in the same history model.
4. One-click `Disable all B-roll` toggle for fast fallback.

## 5. Technical Plan (Incremental, Safe)

## Phase 1: Data and API foundation (no risky render changes yet)

1. Add B-roll suggestion entities:
   - `broll_slots` (project_id, start_sec, end_sec, anchor_word_ids, concept_text, locked)
   - `broll_candidates` (slot_id, asset_id/source_url, score, reason_json)
   - `broll_choices` (slot_id, chosen_candidate_id, status)

2. Add APIs:
   - `POST /api/v1/broll/suggest`
   - `GET /api/v1/broll/slots?project_id=...`
   - `POST /api/v1/broll/slots/{slot_id}/choose`
   - `POST /api/v1/broll/slots/{slot_id}/reject`

3. Keep existing transcript endpoints untouched.

## Phase 2: Timeline integration

1. Extend timeline state with non-destructive B-roll metadata.
2. Add operation types:
   - `add_broll_clip`
   - `move_broll_clip`
   - `trim_broll_clip`
   - `delete_broll_clip`
   - `set_broll_opacity`
3. Ensure operations never mutate transcript word deletion state.

## Phase 3: Render integration

1. Extend renderer from single-video concat to compositing pipeline:
   - base cut track = existing transcript-driven main track
   - b-roll lane composited with timed overlays (`overlay`, optional alpha/scale)
2. Keep audio from base video by default.
3. Optional ducking when B-roll source audio is enabled.

## Phase 4: Quality loop

1. Add slot-level feedback logging: accept/reject/replace.
2. Re-rank candidates with lightweight heuristics from user actions.
3. Add regression tests to guarantee transcript edits are unchanged when B-roll is toggled on/off.

## 6. Suggestion/Ranking Logic (V1 Heuristic)

For each transcript chunk (sentence or 2-3 sentence block):

1. Extract concepts (noun phrases + named entities).
2. Build retrieval query from top concepts + style intent.
3. Score candidate assets:
   - `0.40 * semantic_match`
   - `0.20 * temporal_fit`
   - `0.15 * visual_quality`
   - `0.15 * diversity_gain`
   - `0.10 * novelty_penalty_inverse`
4. Apply constraints:
   - no repeated identical shot within N seconds
   - avoid replacing high-emotion face-on segments unless user requested
   - cap B-roll coverage ratio (e.g., 15-35% initially)

## 7. Definition of “Perfect UI/UX” for This Feature

Feature is ready only if:

1. User can generate B-roll suggestions in < 10 seconds for short videos.
2. User can approve/reject all slots without leaving transcript workflow.
3. Transcript editing accuracy is unchanged from current baseline.
4. Render output is deterministic and reproducible for same choices.
5. Undo/redo and preview remain stable under rapid edits.

## 8. Acceptance Tests (Must Pass)

1. Transcript-only project renders exactly as before.
2. Adding B-roll with zero accepted slots does not alter output.
3. Accepting one B-roll slot only affects its time range.
4. Deleting transcript words does not silently delete B-roll choices; prompts explicit relink.
5. Export path supports both preview and full export with B-roll enabled.

## 9. Recommended Build Order for Your Repo (Immediate)

1. Implement Phase 1 API/data models first.
2. Build frontend `B-roll Studio` panel with mock data.
3. Wire real suggestion endpoint.
4. Add timeline B-roll lane rendering.
5. Add backend render compositing.
6. Add regression test suite focused on transcript safety.

## 10. References

1. B-Script (arXiv): https://arxiv.org/abs/1909.02818  
2. QuickCut (Stanford): https://graphics.stanford.edu/projects/quickcut/  
3. Computational Video Editing for Dialogue-Driven Scenes (Stanford): https://graphics.stanford.edu/projects/videoediting/  
4. ChunkyEdit (Adobe Research): https://research.adobe.com/publication/chunkyedit/  
5. VideoDiff (ACM DOI): https://dl.acm.org/doi/10.1145/3706598.3713289  
6. OpusClip AI B-roll help: https://www.opus.pro/help-center/what-is-ai-b-roll-and-how-to-use-it  
7. VEED transcript editing guide: https://www.veed.io/learn/how-to-edit-video-using-a-transcript  
8. VEED B-roll guide: https://www.veed.io/learn/how-to-add-b-roll-to-a-video  
9. Riverside AI feature docs: https://support.riverside.fm/hc/en-us/articles/30302678779677-AI-at-Riverside  
10. Adobe Premiere Pro text-based editing: https://helpx.adobe.com/premiere-pro/using/text-based-editing.html  
11. Descript edit-like-doc workflow: https://www.descript.com/how-to-edit-video  
12. Descript media replacement workflow: https://www.descript.com/help/docs/replacing-video-scenes-with-media
