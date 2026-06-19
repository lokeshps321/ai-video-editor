# ClipMind Literature Survey

## Document Control

- Title: Strict Literature Survey for ClipMind AI Video Editor
- Version: 2.0
- Date: March 8, 2026
- Scope rule: only official venue or publisher pages were used as primary sources
- Allowed source classes in this document:
  - IEEE/CVF Open Access proceedings pages
  - ACL Anthology proceedings pages
  - ACM IUI official proceedings pages
- Excluded from this revision:
  - arXiv-only entries
  - project pages without official proceedings backing
  - blog posts, product pages, and secondary summaries

## Purpose of This Survey

This literature survey is written for the ClipMind final-year project. Its purpose is not only to list recent papers, but to position ClipMind against the strongest relevant academic work from 2023, 2024, and 2025.

The survey focuses on three problem areas that are closest to the actual system we built:

1. video editing and video-to-video transformation,
2. language-driven or instruction-driven editing interfaces,
3. long-video understanding, summarization, and automatic editing support.

ClipMind is not a new diffusion model or a new speech model. Its contribution is a working end-to-end editor where transcript generation, language-aware captions, B-roll planning, timeline editing, and render preview are tied together in one system. Because of that, the comparison standard in this survey is not "does the paper look similar in one module," but "how much of a real editable workflow does the paper cover."

## Selection Logic

The fifteen papers below were selected using the following rules:

1. The publication year had to be 2023, 2024, or 2025.
2. The paper had to be available through an official venue page, not only a preprint page.
3. The paper had to be relevant to at least one of ClipMind's major modules:
   - AI-assisted editing
   - text-driven video manipulation
   - transcript/narrative-based video understanding
   - long-video summarization
   - interactive video editing interfaces
4. The final set had to include both model-level research and editor/workflow research, because ClipMind sits between those two worlds.

## High-Level Research Themes

The selected literature clusters into four themes.

### Theme 1: Zero-Shot and Text-Guided Video Editing

These papers study how an existing video can be edited using text prompts while preserving temporal consistency.

### Theme 2: Controllable Video Transformation with Better Temporal Fidelity

These papers are concerned with motion consistency, inversion, token merging, interpolation, and structure preservation. They matter because any modern AI editing story must respect time, not just single frames.

### Theme 3: Human-AI Interfaces for Video Editing

These papers are important because ClipMind is not just a backend model pipeline. It is an editor with user-facing controls, timeline operations, and human correction loops.

### Theme 4: Long-Video Understanding, Summarization, and Automatic Edit Planning

These papers matter because ClipMind uses transcripts and semantic cues to drive editing, captioning, and B-roll planning rather than treating the transcript as a passive output.

## Survey of Selected Papers

## 1. FateZero: Fusing Attentions for Zero-shot Text-based Video Editing

- Year: 2023
- Venue: ICCV 2023
- Publisher: IEEE/CVF
- Official source: [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/QI_FateZero_Fusing_Attentions_for_Zero-shot_Text-based_Video_Editing_ICCV_2023_paper.html)

### What the paper does

FateZero is one of the best-known zero-shot text-based video editing papers from 2023. It studies how to edit a real input video using text prompts without per-prompt training. The core idea is to preserve structure and motion by capturing and reusing intermediate attention information during inversion and denoising.

### Why it matters to ClipMind

This paper is relevant because it represents the model-centric end of AI video editing: give a video plus a prompt, and generate a modified video. It provides a useful baseline for understanding what "AI video editing" often means in the research literature.

### Limitation relative to ClipMind

FateZero is not an editor product architecture. It does not solve transcript generation, caption generation, timeline manipulation, B-roll reasoning, multilingual handling, or preview/export workflow. It is powerful at visual transformation, but narrow in scope compared to a creator workflow system.

### Takeaway for our project

ClipMind does not compete with FateZero on diffusion novelty. Instead, it solves a broader workflow problem: it makes text, timing, captions, B-roll, and user correction part of one editing loop.

## 2. StableVideo: Text-driven Consistency-aware Diffusion Video Editing

- Year: 2023
- Venue: ICCV 2023
- Publisher: IEEE/CVF
- Official source: [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Chai_StableVideo_Text-driven_Consistency-aware_Diffusion_Video_Editing_ICCV_2023_paper.html)

### What the paper does

StableVideo focuses on consistency-aware video editing. The paper addresses a common weakness in diffusion-based editing: even if the edit looks correct in one frame, the object geometry and appearance may drift across time. StableVideo introduces inter-frame propagation and layered representations to keep edits coherent.

### Why it matters to ClipMind

Temporal consistency is a core challenge in any system that claims to edit video intelligently. Even though ClipMind is not itself a generative video editor, the same design principle appears in our render path and timeline logic: edits must remain stable across time and across preview/export.

### Limitation relative to ClipMind

StableVideo is focused on a visual generation problem. It does not provide a transcript-first interface, editable B-roll planning, caption styling, or end-user timeline operations.

### Takeaway for our project

The paper shows that time-aware consistency is essential. ClipMind applies that lesson at the workflow level rather than the latent-model level.

## 3. Structure and Content-Guided Video Synthesis with Diffusion Models

- Year: 2023
- Venue: ICCV 2023
- Publisher: IEEE/CVF
- Official source: [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Esser_Structure_and_Content-Guided_Video_Synthesis_with_Diffusion_Models_ICCV_2023_paper.html)

### What the paper does

This paper proposes a structure-aware and content-aware latent video diffusion model. It studies how to preserve structural aspects of a source video while editing content based on text or image guidance. It also emphasizes explicit control over structure, content fidelity, and temporal consistency.

### Why it matters to ClipMind

The paper matters because ClipMind also separates different kinds of editing intent. In our system, the timeline holds structure, captions hold language-driven overlays, and B-roll planning changes visual coverage without destroying the base narrative timeline.

### Limitation relative to ClipMind

The paper is strong on generation control, but it is not a practical editor for users who need transcription, subtitle styling, timeline operations, and manual correction.

### Takeaway for our project

This work supports the idea that controllability matters more than raw generation alone. ClipMind extends that philosophy into user-editable, explainable workflow controls.

## 4. ExpressEdit: Video Editing with Natural Language and Sketching

- Year: 2024
- Venue: IUI 2024
- Publisher: ACM
- Official source: [IUI 2024 Proceedings](https://iui.acm.org/2024/proceedings.html)

### What the paper does

ExpressEdit is an HCI-oriented system paper rather than a generative-model paper. It studies how editors express video edits through natural language and sketching. The system interprets temporal, spatial, and operational references in multimodal user commands.

### Why it matters to ClipMind

This paper is very relevant because ClipMind is also a human-in-the-loop editing system. It supports the argument that natural language is useful not as a replacement for the editor, but as an additional control layer for editing intent.

### Limitation relative to ClipMind

ExpressEdit focuses on multimodal command input, not end-to-end transcript-driven editing. It does not address multilingual transcription, caption rendering, timeline rendering pipelines, or B-roll generation.

### Takeaway for our project

The paper validates one of our core design principles: language is a viable editing interface. ClipMind extends that concept by making transcript text itself the control surface for editing operations.

## 5. LAVE: LLM-Powered Agent Assistance and Language Augmentation for Video Editing

- Year: 2024
- Venue: IUI 2024
- Publisher: ACM
- Official source: [IUI 2024 Proceedings](https://iui.acm.org/2024/proceedings.html)

### What the paper does

LAVE explores agent-assisted editing using large language models. The system automatically generates language descriptions for footage and allows the user to edit through both agent assistance and direct manipulation. Its emphasis is on reducing barriers for novice editors while keeping manual control available.

### Why it matters to ClipMind

This is one of the closest conceptual neighbors to ClipMind. Both systems treat language as an operational layer over video editing, and both preserve a role for direct user manipulation rather than forcing a fully automatic pipeline.

### Limitation relative to ClipMind

LAVE is strong on agent assistance, but the published framing does not center multilingual caption rendering, transcript-level correction, Indic-language issues, or B-roll reasoning from lyric meaning.

### Takeaway for our project

LAVE strengthens the academic case for "language-augmented editing." ClipMind contributes by grounding that idea in a transcript-centric architecture with timeline persistence and multilingual caption handling.

## 6. RAVE: Randomized Noise Shuffling for Fast and Consistent Video Editing with Diffusion Models

- Year: 2024
- Venue: CVPR 2024
- Publisher: IEEE/CVF
- Official source: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Kara_RAVE_Randomized_Noise_Shuffling_for_Fast_and_Consistent_Video_Editing_CVPR_2024_paper.html)

### What the paper does

RAVE proposes a lightweight zero-shot video editing method using text-to-image diffusion backbones plus a noise-shuffling strategy to improve temporal consistency. The paper also emphasizes efficiency and faster inference compared to several earlier editing baselines.

### Why it matters to ClipMind

RAVE matters because it represents the state of the art in efficient model-centric video editing around 2024. It shows the field moving from "can we edit videos at all" to "can we edit them consistently and fast enough to be practical."

### Limitation relative to ClipMind

It is still a model paper. It does not cover asset management, transcription, B-roll retrieval, caption presets, or UI explainability.

### Takeaway for our project

RAVE is useful as a benchmark of model sophistication, but ClipMind's contribution lies in editor orchestration, not diffusion mechanics.

## 7. FRESCO: Spatial-Temporal Correspondence for Zero-Shot Video Translation

- Year: 2024
- Venue: CVPR 2024
- Publisher: IEEE/CVF
- Official source: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_FRESCO_Spatial-Temporal_Correspondence_for_Zero-Shot_Video_Translation_CVPR_2024_paper.html)

### What the paper does

FRESCO improves zero-shot video translation by combining intra-frame and inter-frame correspondence. The key idea is that consistency must be enforced both within a frame and across neighboring frames, producing more coherent results.

### Why it matters to ClipMind

The paper is relevant because it reinforces a broad lesson: video systems must respect both local and temporal structure. ClipMind follows that principle in a different layer of the stack by preserving timeline structure while adding overlays such as captions and B-roll.

### Limitation relative to ClipMind

FRESCO does not target editing UX or semantic creator workflows. It focuses on transformation quality, not transcript-guided operations or user explainability.

### Takeaway for our project

This work helps justify why ClipMind separates timeline semantics from visual overlays and maintains explicit, structured timeline state instead of one opaque AI output.

## 8. MaskINT: Video Editing via Interpolative Non-autoregressive Masked Transformers

- Year: 2024
- Venue: CVPR 2024
- Publisher: IEEE/CVF
- Official source: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_MaskINT_Video_Editing_via_Interpolative_Non-autoregressive_Masked_Transformers_CVPR_2024_paper.html)

### What the paper does

MaskINT breaks video editing into two stages: edit sparse keyframes, then interpolate the intermediate frames efficiently with a non-autoregressive masked transformer. The motivation is to make text-based video editing more efficient and practical.

### Why it matters to ClipMind

The paper matters because practical editing systems often need staged processing rather than one giant monolithic model call. ClipMind uses the same engineering mindset: transcription, captioning, B-roll generation, and rendering are separate stages that can be retried or corrected independently.

### Limitation relative to ClipMind

MaskINT does not provide editor-level user controls or transcript-aware operations. It solves an efficiency problem in video generation rather than an end-to-end editing workflow.

### Takeaway for our project

The paper supports a modular pipeline philosophy. ClipMind applies that philosophy to creator tooling instead of latent video synthesis.

## 9. A Video is Worth 256 Bases: Spatial-Temporal Expectation-Maximization Inversion for Zero-Shot Video Editing

- Year: 2024
- Venue: CVPR 2024
- Publisher: IEEE/CVF
- Official source: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Li_A_Video_is_Worth_256_Bases_Spatial-Temporal_Expectation-Maximization_Inversion_for_CVPR_2024_paper.html)

### What the paper does

This paper improves video inversion for zero-shot editing through a compact basis representation. Instead of using fully dense time-varying inversion features, it models the video using a smaller set of bases and expectation-maximization updates, improving temporal consistency and reducing cost.

### Why it matters to ClipMind

It is relevant because it addresses a common research bottleneck in video editing: how to preserve the source video while making controllable changes efficiently.

### Limitation relative to ClipMind

The paper stays entirely in the model domain. It does not discuss interaction design, multilingual media, caption rendering, or timeline UX.

### Takeaway for our project

The research is useful for understanding advanced video-editing methods, but ClipMind's contribution is broader system integration and a user-operable workflow.

## 10. VidToMe: Video Token Merging for Zero-Shot Video Editing

- Year: 2024
- Venue: CVPR 2024
- Publisher: IEEE/CVF
- Official source: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Li_VidToMe_Video_Token_Merging_for_Zero-Shot_Video_Editing_CVPR_2024_paper.html)

### What the paper does

VidToMe improves temporal consistency and memory efficiency by merging similar self-attention tokens across frames. The method addresses the fact that video editing with image diffusion models can become memory-heavy and temporally unstable.

### Why it matters to ClipMind

This paper is part of the same important pattern as RAVE and FRESCO: practical video editing requires explicit temporal handling. Even if ClipMind's main AI layer is transcript-driven rather than diffusion-driven, the same principle informs our preview and render architecture.

### Limitation relative to ClipMind

VidToMe is not concerned with editing workflows, only with generation quality and efficiency. It provides no transcript, caption, or creator-oriented interface layer.

### Takeaway for our project

The paper reinforces that time structure is not optional. ClipMind addresses time explicitly through timeline state, word timings, clip timings, and rendering jobs.

## 11. GenVideo: One-shot Target-image and Shape Aware Video Editing using T2I Diffusion Models

- Year: 2024
- Venue: CVPR Workshops 2024
- Publisher: IEEE/CVF
- Official source: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024W/GCV/html/Harsha_GenVideo_One-shot_Target-image_and_Shape_Aware_Video_Editing_using_T2I_CVPRW_2024_paper.html)

### What the paper does

GenVideo studies target-image-aware video editing. Rather than relying only on a text prompt, it uses a reference target image and shape-aware masking logic to guide edits, especially when object geometry differs from the source.

### Why it matters to ClipMind

This paper is relevant because creator workflows often need stronger guidance than plain prompt text. In ClipMind, a similar principle appears in B-roll generation: raw transcript text is often not enough, so the system converts transcript meaning into more controlled English visual gloss before search.

### Limitation relative to ClipMind

GenVideo still focuses on video transformation, not editor workflow. It does not manage textual transcript operations, captions, or semantic shot planning on a timeline.

### Takeaway for our project

The paper supports the idea that intermediate semantic representations are useful. ClipMind applies that idea to transcript-to-visual-gloss conversion for B-roll.

## 12. VideoDirector: Precise Video Editing via Text-to-Video Models

- Year: 2025
- Venue: CVPR 2025
- Publisher: IEEE/CVF
- Official source: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VideoDirector_Precise_Video_Editing_via_Text-to-Video_Models_CVPR_2025_paper.html)

### What the paper does

VideoDirector is a 2025 paper that pushes beyond image-model-based editing and uses text-to-video models directly for more precise editing. It introduces spatial-temporal decoupled guidance and attention control to improve fidelity, motion smoothness, and realism.

### Why it matters to ClipMind

This is useful as a "state of the field" reference. It shows where frontier research is moving: from adapting image diffusion models toward directly leveraging video-native generative models.

### Limitation relative to ClipMind

Even here, the scope is still video generation quality. There is no end-to-end local editor workflow, no transcript operations, no captions, and no human correction path like the one exposed in ClipMind's UI.

### Takeaway for our project

VideoDirector is strong evidence that cutting-edge video editing research is still mostly model-centric. ClipMind differentiates itself by solving workflow integration and user control rather than pure generative precision.

## 13. From Long Videos to Engaging Clips: A Human-Inspired Video Editing Framework with Multimodal Narrative Understanding

- Short name used in discussion: HIVE
- Year: 2025
- Venue: EMNLP 2025 Industry Track
- Publisher: Association for Computational Linguistics
- Official source: [ACL Anthology](https://aclanthology.org/2025.emnlp-industry.185/)

### What the paper does

HIVE studies automatic editing of long-form videos into engaging short clips. The system uses multimodal narrative understanding, character extraction, dialogue analysis, narrative summarization, scene-level segmentation, highlight detection, and clip pruning.

### Why it matters to ClipMind

This paper is extremely relevant because it shifts the focus from pure frame generation to narrative understanding and editing structure. That is much closer to ClipMind's philosophy, where transcript and semantics drive editing decisions.

### Limitation relative to ClipMind

HIVE is stronger than ClipMind in automatic highlight curation for long-form narrative video. However, it is not primarily a user-facing editor with manual timeline operations, multilingual subtitle rendering, caption presets, or B-roll reasoning controls exposed in the UI.

### Takeaway for our project

HIVE validates the importance of semantic narrative understanding. ClipMind extends that direction into an editable product workflow rather than only an automatic editing framework.

## 14. RACCooN: Versatile Instructional Video Editing with Auto-Generated Narratives

- Year: 2025
- Venue: EMNLP 2025
- Publisher: Association for Computational Linguistics
- Official source: [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1420/)

### What the paper does

RACCooN proposes a video-to-paragraph-to-video editing pipeline. It first generates structured textual descriptions from video content, then uses those auto-generated narratives to drive editing operations such as removal, addition, and modification.

### Why it matters to ClipMind

This paper is highly relevant to ClipMind because it supports a core architectural idea we also use: raw video or raw transcript is often not the best direct editing representation. An intermediate semantic description can make editing more controllable and more user-friendly.

### Limitation relative to ClipMind

RACCooN is oriented toward generative video editing driven by auto-generated narratives. It does not focus on subtitle rendering, transcript correction interfaces, timeline UX, or multilingual song workflows.

### Takeaway for our project

The strongest connection is conceptual: both systems rely on semantic intermediate representations. In ClipMind, the most visible example is the English meaning or visual gloss used to improve B-roll generation from non-English content.

## 15. Detecting and Mitigating Challenges in Zero-Shot Video Summarization with Video LLMs

- Year: 2025
- Venue: Findings of ACL 2025
- Publisher: Association for Computational Linguistics
- Official source: [ACL Anthology](https://aclanthology.org/2025.findings-acl.16/)

### What the paper does

This paper studies the reliability of zero-shot video summarization with video large language models. It evaluates multiple VLLMs, identifies common failure modes, and proposes mitigation strategies such as prompt refinement and lightweight external knowledge injection.

### Why it matters to ClipMind

This paper is important because it is not just optimistic about video LLMs. It explicitly studies failure modes and mitigation. That mindset matches the engineering direction of ClipMind, where we added fallback logic, language-aware routing, debug views, and user correction loops rather than assuming AI outputs are always correct.

### Limitation relative to ClipMind

The paper is about summarization quality, not editing workflow. It does not deliver a timeline editor, caption system, or B-roll workflow.

### Takeaway for our project

This paper supports one of our strongest academic arguments: practical AI editing systems must expose correction paths and reliability checks instead of trusting single-shot model outputs.

## Cross-Paper Synthesis

Across these fifteen papers, several patterns appear clearly.

### Pattern 1: Most advanced research is still model-centric

The CVF papers mostly optimize temporal consistency, inversion quality, attention control, or efficiency in video transformation. They are strong technical works, but they are not full editing systems for creators.

### Pattern 2: HCI papers validate language as an editing interface

ExpressEdit and LAVE are especially useful because they justify the decision to treat language as a real editing control surface. This is one of the strongest connections between the literature and ClipMind.

### Pattern 3: Long-video work is moving toward narrative understanding

HIVE, RACCooN, and the summarization paper show that the field is moving beyond frame-level manipulation toward structure, story, and semantics. This strongly supports ClipMind's transcript-first architecture.

### Pattern 4: Human correction remains necessary

Even the best research systems do not make editing fully automatic in a trustworthy way. This is why ClipMind's editable transcript, B-roll gloss override, language picker, caption presets, reroll actions, and timeline interactions are academically defensible design choices.

## How ClipMind Differs from the Surveyed Literature

ClipMind differs from the above papers in six important ways.

1. It is an integrated editor, not just an isolated model.
2. It treats the transcript as an operational editing layer rather than a side output.
3. It combines captions, B-roll, transcript editing, and rendering in one workflow.
4. It pays special attention to multilingual creator problems, especially major Indian languages.
5. It exposes user correction loops instead of hiding AI internals.
6. It persists state through a real project, timeline, and background render architecture.

This does not mean ClipMind is "better" than all fifteen papers in every technical dimension. That would be false. Many of these papers are stronger in diffusion-based video transformation or benchmark-level automation. The correct claim is narrower and more defensible:

ClipMind addresses an integration gap. It turns several AI editing ideas into a coherent, editable system that a user can actually operate, inspect, correct, and demonstrate end to end.

## Research Gap Identified

The literature suggests a gap between cutting-edge model papers and real creator tooling.

### What existing papers do well

- temporally consistent video transformation,
- prompt-driven visual editing,
- multimodal command interpretation,
- long-video semantic understanding,
- video summarization and automatic clip selection.

### What is still under-served

- transcript-native editing products,
- multilingual subtitle-safe rendering in practical editors,
- semantic B-roll planning for non-English content,
- transparent debug and correction loops for creator workflows,
- end-to-end local systems that connect upload, transcript, edit, preview, and export.

ClipMind is positioned directly in that gap.

## Why These Sources Are Academically Safer

This revised survey is stricter than the earlier mixed version.

- Every paper is anchored to an official proceedings page.
- The proceedings come from recognized academic venues:
  - IEEE/CVF ICCV and CVPR
  - ACM IUI
  - ACL Anthology venues such as EMNLP and Findings of ACL
- No claim in this survey depends on a product blog, random repository, or unsupported project page.

This makes the document much safer for academic submission and viva discussion.

## Conclusion

The literature from 2023 to 2025 shows clear progress in AI-assisted video editing, but most work remains split across separate research tracks: diffusion-based video transformation, multimodal editing interfaces, and long-video semantic understanding. ClipMind's main value is that it connects these ideas into one usable editor architecture.

The strongest academic positioning for ClipMind is therefore not:

> "We invented a new video generation model."

It is:

> "We built an integrated transcript-driven AI video editor that combines multilingual transcription, language-aware subtitle rendering, semantic B-roll planning, timeline-based editing, and render preview into one human-correctable workflow."

That claim is supported by the gap visible across the surveyed literature.

## Official Reference List

1. QI, Chenyang, et al. "FateZero: Fusing Attentions for Zero-shot Text-based Video Editing." ICCV 2023. IEEE/CVF. Official page: [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/QI_FateZero_Fusing_Attentions_for_Zero-shot_Text-based_Video_Editing_ICCV_2023_paper.html)
2. Chai, Wenhao, et al. "StableVideo: Text-driven Consistency-aware Diffusion Video Editing." ICCV 2023. IEEE/CVF. Official page: [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Chai_StableVideo_Text-driven_Consistency-aware_Diffusion_Video_Editing_ICCV_2023_paper.html)
3. Esser, Patrick, et al. "Structure and Content-Guided Video Synthesis with Diffusion Models." ICCV 2023. IEEE/CVF. Official page: [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Esser_Structure_and_Content-Guided_Video_Synthesis_with_Diffusion_Models_ICCV_2023_paper.html)
4. Tilekbay, Bekzat, et al. "ExpressEdit: Video Editing with Natural Language and Sketching." IUI 2024. ACM. Official page: [IUI 2024 Proceedings](https://iui.acm.org/2024/proceedings.html)
5. Wang, Bryan, et al. "LAVE: LLM-Powered Agent Assistance and Language Augmentation for Video Editing." IUI 2024. ACM. Official page: [IUI 2024 Proceedings](https://iui.acm.org/2024/proceedings.html)
6. Kara, Ozgur, et al. "RAVE: Randomized Noise Shuffling for Fast and Consistent Video Editing with Diffusion Models." CVPR 2024. IEEE/CVF. Official page: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Kara_RAVE_Randomized_Noise_Shuffling_for_Fast_and_Consistent_Video_Editing_CVPR_2024_paper.html)
7. Yang, Shuai, et al. "FRESCO: Spatial-Temporal Correspondence for Zero-Shot Video Translation." CVPR 2024. IEEE/CVF. Official page: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_FRESCO_Spatial-Temporal_Correspondence_for_Zero-Shot_Video_Translation_CVPR_2024_paper.html)
8. Ma, Haoyu, et al. "MaskINT: Video Editing via Interpolative Non-autoregressive Masked Transformers." CVPR 2024. IEEE/CVF. Official page: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_MaskINT_Video_Editing_via_Interpolative_Non-autoregressive_Masked_Transformers_CVPR_2024_paper.html)
9. Li, Maomao, et al. "A Video is Worth 256 Bases: Spatial-Temporal Expectation-Maximization Inversion for Zero-Shot Video Editing." CVPR 2024. IEEE/CVF. Official page: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Li_A_Video_is_Worth_256_Bases_Spatial-Temporal_Expectation-Maximization_Inversion_for_CVPR_2024_paper.html)
10. Li, Xirui, et al. "VidToMe: Video Token Merging for Zero-Shot Video Editing." CVPR 2024. IEEE/CVF. Official page: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Li_VidToMe_Video_Token_Merging_for_Zero-Shot_Video_Editing_CVPR_2024_paper.html)
11. Harsha, Sai Sree, et al. "GenVideo: One-shot Target-image and Shape Aware Video Editing using T2I Diffusion Models." CVPR Workshops 2024. IEEE/CVF. Official page: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024W/GCV/html/Harsha_GenVideo_One-shot_Target-image_and_Shape_Aware_Video_Editing_using_T2I_CVPRW_2024_paper.html)
12. Wang, Yukun, et al. "VideoDirector: Precise Video Editing via Text-to-Video Models." CVPR 2025. IEEE/CVF. Official page: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VideoDirector_Precise_Video_Editing_via_Text-to-Video_Models_CVPR_2025_paper.html)
13. Wang, Xiangfeng, et al. "From Long Videos to Engaging Clips: A Human-Inspired Video Editing Framework with Multimodal Narrative Understanding." EMNLP 2025 Industry Track. ACL. Official page: [ACL Anthology](https://aclanthology.org/2025.emnlp-industry.185/)
14. Yoon, Jaehong, Shoubin Yu, and Mohit Bansal. "RACCooN: Versatile Instructional Video Editing with Auto-Generated Narratives." EMNLP 2025. ACL. Official page: [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1420/)
15. Cagliero, Luca, et al. "Detecting and Mitigating Challenges in Zero-Shot Video Summarization with Video LLMs." Findings of ACL 2025. ACL. Official page: [ACL Anthology](https://aclanthology.org/2025.findings-acl.16/)
