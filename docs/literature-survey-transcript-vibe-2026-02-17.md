# Literature Survey: Transcript-Based + AI-Assisted Video Editing

Date: 2026-02-17
Project: Prompt Video Editor (`backend/` + `frontend/`)

## 1. Scope

This survey focuses on:

- Transcript-driven editing (word/sentence deletion to timeline cuts)
- ASR quality foundations for editing reliability
- AI assistance for subtitles, pause trimming, and narrative restructuring
- Perceived visual quality after jump cuts

## 2. Key Research and What It Means for This Product

## 2.1 Text-Based Editing Interfaces

1. **Text-based Editing of Talking-head Video (CHI 2019)**  
   URL: https://dl.acm.org/doi/10.1145/3290605.3300869  
   Key idea: users edit transcript text directly; system maps text edits to video cuts.  
   Relevance: this is the core interaction pattern of our left-panel editor.

2. **VoCo: Text-Based Insertion and Replacement in Audio Narration (UIST 2016)**  
   URL: https://dl.acm.org/doi/10.1145/2984511.2985063  
   Key idea: edit spoken content through text operations and speech synthesis/replacement.  
   Relevance: shows transcript editing can evolve from cut-only to rewrite/replace workflows.

## 2.2 ASR Foundation and Tradeoffs

3. **Whisper (OpenAI, 2022)**  
   URL: https://arxiv.org/abs/2212.04356  
   Key idea: large-scale weakly supervised ASR with multilingual robustness.  
   Relevance: explains why Whisper-family models remain strong default choices.

4. **Whisper Model Card (OpenAI)**  
   URL: https://raw.githubusercontent.com/openai/whisper/main/model-card.md  
   Key limitations called out by authors: variable performance across language/accent groups; possible hallucinations; reliability caveats in high-stakes contexts.  
   Relevance: transcript edits must handle confidence and uncertainty, not just raw words.

5. **faster-whisper (SYSTRAN)**  
   URL: https://github.com/SYSTRAN/faster-whisper  
   Key idea: CTranslate2-based Whisper inference with improved throughput/memory tradeoffs and word timestamps.  
   Relevance: directly matches our current backend implementation path.

## 2.3 Newer AI Editing Directions

6. **TalkLess: Better Video Editing Through LLM-Driven Transcript Parsing (2025)**  
   URL: https://arxiv.org/abs/2507.15202  
   Key idea: LLM-guided transcript interpretation for editing actions.  
   Relevance: supports extending our simple rule-based text operations toward intent-aware editing.

7. **FluentEditor2: Multi-agent LLM Workflow for Script and Subtitle Refinement (2025)**  
   URL: https://arxiv.org/abs/2602.00560  
   Key idea: iterative multi-agent editing/review loops for cleaner output.  
   Relevance: useful for future "draft + critique + fix" pipeline around captions and script cuts.

8. **HIVE: Human-Centric Intelligent Video Editing (2025)**  
   [Metadata](file:///home/lokesh/ai%20video%20editor/docs/research_corpus/papers/hive_2025.md) | [ArXiv 2507.02790](https://arxiv.org/abs/2507.02790)  
   Key idea: structured representation and multimodal reasoning for narrative editing.  

## 2.5 New Additions (2023-2025 Corpus)

I have "installed" 15 detailed paper summaries into the project. Below is the list with links to local metadata:

### Transcript & Sequential Editing
1. **[TalkLess (2025)](file:///home/lokesh/ai%20video%20editor/docs/research_corpus/papers/talkless_2025.md)**: LLM-driven transcript parsing for intent-aware cuts.
2. **[FluentEditor2 (2025)](file:///home/lokesh/ai%20video%20editor/docs/research_corpus/papers/fluenteditor2_2025.md)**: Multi-agent refinement for subtitles/scripts.
3. **[VideoDirector (2025)](file:///home/lokesh/ai%20video%20editor/docs/research_corpus/papers/videodirector_2025.md)**: Precision text-to-video editing.
4. **[Text-to-Edit (2025)](file:///home/lokesh/ai%20video%20editor/docs/research_corpus/papers/text_to_edit_2025.md)**: Ad creation via Multimodal LLMs.
5. **[ExpressEdit (2024)](file:///home/lokesh/ai%20video%20editor/docs/research_corpus/papers/expressedit_2024.md)**: Natural language + sketching for visual control.
6. **[EditBoard (2024)](file:///home/lokesh/ai%20video%20editor/docs/research_corpus/papers/editboard_2024.md)**: Evaluation benchmarks for text-based models.
7. **[VidEdit (2023)](file:///home/lokesh/ai%20video%20editor/docs/research_corpus/papers/videdit_2023.md)**: Zero-shot spatially aware editing.
8. **[FateZero (2023)](file:///home/lokesh/ai%20video%20editor/docs/research_corpus/papers/fatezero_2023.md)**: Cross-frame structural consistency.

### B-Roll, Generation, and Automation
9. **[RACCooN (2025)](file:///home/lokesh/ai%20video%20editor/docs/research_corpus/papers/raccoon_2025.md)**: Video-to-Paragraph-to-Video editing loop.
10. **[STREAM (2024)](file:///home/lokesh/ai%20video%20editor/docs/research_corpus/papers/stream_2024.md)**: Smart transcript-to-B-Roll rendering.
11. **[VideoDoodles (2023)](file:///home/lokesh/ai%20video%20editor/docs/research_corpus/papers/videodoodles_2023.md)**: Hand-drawn animations with scene awareness.
12. **[Direct-a-Video (2024)](file:///home/lokesh/ai%20video%20editor/docs/research_corpus/papers/direct_a_video_2024.md)**: Multi-object and camera motion control.
13. **[Jump Cut Smoothing (2024)](file:///home/lokesh/ai%20video%20editor/docs/research_corpus/papers/jump_cut_smoothing_2024.md)**: Perceptual quality improvements for talking head cuts.
14. **[Transcript Summarization (2025)](file:///home/lokesh/ai%20video%20editor/docs/research_corpus/papers/transcript_summarization_2025.md)**: Highlight detection via transcripts.
15. **[HIVE (2025)](file:///home/lokesh/ai%20video%20editor/docs/research_corpus/papers/hive_2025.md)**: High-level narrative planning.
   Relevance: points to richer edit planning beyond independent button actions.

## 2.4 Visual Quality After Cuts

9. **Towards Jump Cut Smoothing for Talking Head Videos (2024)**  
   URL: https://arxiv.org/abs/2410.03719  
   Key idea: jump cuts can harm perceived quality; smoothing strategies matter.  
   Relevance: directly tied to user feedback like "audio changed but visuals feel wrong."

## 2.5 Reliability and Confidence

10. **Improving Automatic Speech Recognition and Named Entity Recognition Through Confidence Estimation (2021)**  
    URL: https://arxiv.org/abs/2110.15222  
    Key idea: confidence-aware pipelines improve downstream text processing reliability.  
    Relevance: transcript cuts should avoid blind trust in low-confidence words.

## 3. Where This Project Is Already Strong

Compared with many research prototypes, this project already integrates:

- End-to-end path from upload -> transcript edit -> timeline rewrite -> rendered preview/export
- Word-level timestamp transcript storage and reusable transcript IDs
- Practical "one-click" vibe actions (`add_subtitles`, `auto_cut_pauses`, `trim_start_end`)
- Multi-track timeline/render system (video + audio) instead of transcript-only mock pipeline
- Undo/redo + operation history for production-style iteration

## 4. Gaps Against Literature

The main gaps visible from current code (`backend/app/transcription_service.py`, `backend/app/routers/transcript.py`, `backend/app/routers/vibe.py`):

1. Per-word confidence is now exposed in transcript payload/UI, but cut logic is not yet confidence-weighted.
2. No speaker diarization, so multi-speaker edits are harder.
3. Transcript edits are lexical and timestamp-based, not discourse/intent-aware.
4. Pause trimming uses thresholded silence detection only; not semantic pacing.
5. No learned jump-cut smoothing stage after transcript cuts.
6. No standardized quality benchmark harness (WER + edit quality metrics).

## 5. Improvement Plan (Research-Aligned)

## Phase A: Reliability First (near-term)

1. Extend confidence-aware editing (partially implemented):
   - Store confidence/probability when available. (done)
   - Mark low-confidence words in UI. (done)
   - Block or warn on aggressive cuts inside low-confidence spans. (next)

2. Add cut quality guards:
   - Keep context padding (already introduced in backend env knobs).
   - Enforce minimum removed duration and minimum resulting clip duration.
   - Optional auto-transition insertion for hard jump boundaries.

3. Add async transcription jobs:
   - Queue long transcriptions like render jobs.
   - Stream progress to UI to prevent request-timeout UX failure.

## Phase B: Better Editing Intelligence (mid-term)

1. Sentence/semantic cut mode:
   - Segment transcript into clause/sentence units.
   - Apply edits at discourse boundaries (not raw token boundaries).

2. LLM-assisted editing planner:
   - User intent -> proposed edit plan -> user confirmation -> apply operations.
   - Keep deterministic op logs for reproducibility.

3. Subtitle quality pass:
   - Reading-speed constraints (chars/sec, max line length).
   - Automatic chunk rebalance and punctuation-aware timing.

## Phase C: Evaluation and Product Proof

1. Build an internal benchmark:
   - WER/TER on known clips
   - Edit accuracy: requested kept/deleted intent vs output timeline
   - Visual quality score around cut points

2. Run A/B tests:
   - Baseline transcript cut vs confidence-aware + smoothing pipeline
   - Measure completion time, correction count, user-rated quality

## 6. What We Can Claim as "Better"

With the current architecture plus the above roadmap, the strongest claim is:

- Not just transcript text editing as a demo, but a complete edit engine with rendering and operational controls.

With planned upgrades, a stronger research-backed claim becomes:

- Confidence-aware, quality-controlled transcript editing that preserves intent while improving visual continuity and user trust.
