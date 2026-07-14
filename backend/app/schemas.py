from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Resolution(BaseModel):
    width: int = 1080
    height: int = 1920


class Crop(BaseModel):
    x: int = 0
    y: int = 0
    width: int = 1080
    height: int = 1920


class CropKeyframe(BaseModel):
    time_sec: float = 0.0
    x: int = 0
    y: int = 0


class ClipTransform(BaseModel):
    crop: Optional[Crop] = None
    crop_keyframes: list[CropKeyframe] = Field(default_factory=list)
    scale: Optional[Resolution] = None
    rotate: int = 0
    flip: Optional[Literal["horizontal", "vertical"]] = None


class ClipAdjustments(BaseModel):
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    exposure: float = 0.0
    temperature: float = 0.0
    preset: Optional[str] = None


class AudioKeyframe(BaseModel):
    time_sec: float
    volume: float


class ClipAudio(BaseModel):
    volume: float = 1.0
    fade_in_sec: float = 0.0
    fade_out_sec: float = 0.0
    mute: bool = False
    keyframes: list[AudioKeyframe] = Field(default_factory=list)


class Transition(BaseModel):
    type: Literal[
        "fade",
        "dissolve",
        "slide_left",
        "slide_right",
        "slide_up",
        "slide_down",
        "zoom",
        "wipe",
    ] = "fade"
    duration_sec: float = 0.5


class TextOverlay(BaseModel):
    id: str
    text: str
    start_sec: float
    duration_sec: float
    x: str = "(w-text_w)/2"
    y: str = "(h-text_h)-80"
    font_size: int = 48
    font_name: Optional[str] = None
    color: str = "white"
    highlight_color: Optional[str] = None
    outline_color: str = "black@0.5"
    outline_width: int = 2
    shadow: int = 0
    alignment: int = 2
    margin_v: int = 80
    style: str = "static"
    word_timings: list[dict[str, Any]] = Field(default_factory=list)


class Clip(BaseModel):
    id: str
    asset_id: str
    start_sec: float
    end_sec: float
    timeline_start_sec: float
    speed: float = 1.0
    broll_opacity: float = 1.0
    transform: ClipTransform = Field(default_factory=ClipTransform)
    adjustments: ClipAdjustments = Field(default_factory=ClipAdjustments)
    audio: ClipAudio = Field(default_factory=ClipAudio)
    transition: Optional[Transition] = None
    text_overlays: list[TextOverlay] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_window(self) -> "Clip":
        if self.end_sec <= self.start_sec:
            raise ValueError("clip end_sec must be greater than start_sec")
        if self.speed <= 0:
            raise ValueError("clip speed must be greater than 0")
        if self.broll_opacity < 0 or self.broll_opacity > 1:
            raise ValueError("clip broll_opacity must be between 0 and 1")
        return self


class Track(BaseModel):
    id: str
    kind: Literal["video", "audio", "text", "overlay"]
    clips: list[Clip] = Field(default_factory=list)
    volume: float = 1.0
    mute: bool = False
    solo: bool = False


class ExportSettings(BaseModel):
    format: Literal["mp4", "mov", "webm"] = "mp4"
    aspect_ratio: Literal["16:9", "9:16"] = "16:9"
    resolution: Literal["720p", "1080p", "4k"] = "1080p"
    fps: Literal[24, 30, 60] = 30
    quality: Literal["low", "medium", "high", "max"] = "high"
    bitrate: Optional[str] = None


class TimelineState(BaseModel):
    fps: int = 30
    resolution: Resolution = Field(default_factory=Resolution)
    tracks: list[Track] = Field(default_factory=list)
    duration_sec: float = 0.0
    export_settings: ExportSettings = Field(default_factory=ExportSettings)


class ProjectCreateRequest(BaseModel):
    name: str
    fps: int = 30
    width: int = 1080
    height: int = 1920


class ProjectResponse(BaseModel):
    id: str
    name: str
    fps: int
    width: int
    height: int
    timeline: TimelineState
    timeline_version: int = 0
    timeline_can_undo: bool = False
    timeline_can_redo: bool = False


class OperationPayload(BaseModel):
    op_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    source: Literal["ui", "prompt"] = "ui"


class OperationApplyRequest(BaseModel):
    operations: list[OperationPayload]


class OperationApplyResponse(BaseModel):
    project_id: str
    version: int
    timeline: TimelineState
    applied_ops: list[str]
    timeline_can_undo: bool = False
    timeline_can_redo: bool = False


class SmartReframeRequest(BaseModel):
    clip_ids: list[str] = Field(default_factory=list)


class SmartReframeResponse(BaseModel):
    project_id: str
    reframed_clip_count: int
    tracked_clip_count: int
    center_crop_clip_count: int
    skipped_clip_count: int
    version: int
    timeline: TimelineState
    timeline_can_undo: bool = False
    timeline_can_redo: bool = False


class PromptParseRequest(BaseModel):
    prompt: str


class PromptParseResponse(BaseModel):
    prompt: str
    confidence: float
    operations: list[OperationPayload]
    errors: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class PromptApplyRequest(BaseModel):
    prompt: str


class IngestUrlRequest(BaseModel):
    url: str


class MediaUploadResponse(BaseModel):
    id: str
    project_id: str
    media_type: str
    filename: str
    storage_path: str
    duration_sec: Optional[float]


class TranscriptWord(BaseModel):
    id: str
    text: str
    display_text: Optional[str] = None
    script_tag: Optional[Literal["latin", "indic", "arabic", "mixed", "other"]] = None
    language_hint: Optional[str] = None
    start_sec: float
    end_sec: float
    confidence: Optional[float] = None
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    quality_label: Optional[Literal["trusted", "weak"]] = None
    source_pass: Optional[Literal["primary", "retry", "rescue", "manual"]] = None
    speaker_id: Optional[str] = None
    speaker_label: Optional[str] = None


class TranscriptRegion(BaseModel):
    start_sec: float
    end_sec: float
    status: Literal["trusted", "weak", "blanked"]
    reason: Optional[str] = None
    word_ids: list[str] = Field(default_factory=list)


class TranscriptGenerateRequest(BaseModel):
    asset_id: str
    mode: Literal["auto", "speech", "song"] = "auto"
    language: Optional[str] = None
    prompt: Optional[str] = None
    translate_to_english: Optional[bool] = None
    force_regenerate: bool = False


class TranscriptCutRequest(BaseModel):
    transcript_id: str
    kept_word_ids: list[str] = Field(default_factory=list)
    context_sec: Optional[float] = Field(default=None, ge=0.0)
    merge_gap_sec: Optional[float] = Field(default=None, ge=0.0)
    min_removed_sec: Optional[float] = Field(default=None, ge=0.0)


class TranscriptRangeUpdateRequest(BaseModel):
    start_word_id: str
    end_word_id: str
    text: Optional[str] = None
    mode: Literal["replace", "blank", "preserve", "delete"] = "replace"

    @model_validator(mode="after")
    def validate_payload(self) -> "TranscriptRangeUpdateRequest":
        if self.mode == "replace" and not (self.text or "").strip():
            raise ValueError("text is required when mode=replace")
        return self


class TranscriptResponse(BaseModel):
    id: str
    project_id: str
    asset_id: str
    source: str
    language: Optional[str]
    text: str
    words: list[TranscriptWord]
    word_count: int = 0
    words_truncated: bool = False
    regions: list[TranscriptRegion] = Field(default_factory=list)
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    quality_label: Literal["trusted", "needs_review"] = "trusted"
    weak_word_count: int = 0
    weak_word_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    issue_region_count: int = 0
    script_tags: list[str] = Field(default_factory=list)
    mixed_script: bool = False
    duration_sec: float
    is_mock: bool
    created_at: str


class TranscriptWordPageResponse(BaseModel):
    transcript_id: str
    project_id: str
    offset: int
    limit: int
    total_words: int
    words: list[TranscriptWord]


class TranscriptGenerateResponse(BaseModel):
    transcript: TranscriptResponse
    timeline: TimelineState
    reused_transcript: bool = False


class TranscriptEditResponse(BaseModel):
    transcript: TranscriptResponse
    timeline: TimelineState
    captions_synced: bool = False


class TranscriptCutResponse(BaseModel):
    project_id: str
    transcript_id: str
    kept_word_count: int
    removed_word_count: int
    timeline: TimelineState
    preview_job: Optional[JobResponse] = None


class BrollSuggestRequest(BaseModel):
    transcript_id: Optional[str] = None
    max_slots: int = Field(default=20, ge=1, le=40)
    candidates_per_slot: int = Field(default=3, ge=1, le=10)
    min_chunk_words: int = Field(default=4, ge=1, le=30)
    replace_existing: bool = True
    include_project_assets: bool = True
    include_external_sources: bool = True
    ai_rerank: bool = True
    # Single-slot selection mode: create a slot for specific transcript words
    anchor_word_ids: Optional[list[str]] = None
    concept_override: Optional[str] = None


class BrollPlanRequest(BaseModel):
    transcript_id: Optional[str] = None
    max_slots: int = Field(default=20, ge=1, le=40)
    min_chunk_words: int = Field(default=4, ge=1, le=30)
    include_project_assets: bool = True
    include_external_sources: bool = True


class BrollRerollRequest(BaseModel):
    candidates_per_slot: int = Field(default=3, ge=1, le=10)
    include_project_assets: bool = True
    include_external_sources: bool = True
    ai_rerank: bool = True
    english_gloss_override: Optional[str] = None


class BrollChooseRequest(BaseModel):
    candidate_id: str


class BrollRejectRequest(BaseModel):
    reason: Optional[str] = None


class BrollCandidateResponse(BaseModel):
    id: str
    project_id: str
    slot_id: str
    asset_id: Optional[str]
    source_type: str
    source_url: Optional[str]
    source_label: Optional[str]
    score: float
    confidence: Optional[float] = None
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    entities: list[str] = Field(default_factory=list)
    visual_intent: Optional[str] = None
    weak_reason_codes: list[str] = Field(default_factory=list)
    reason: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class BrollMeaningResponse(BaseModel):
    source_text: str = ""
    source_languages: list[str] = Field(default_factory=list)
    code_switched: bool = False
    english_gloss: Optional[str] = None
    search_concept: Optional[str] = None
    search_queries: list[str] = Field(default_factory=list)
    rationale: Optional[str] = None
    gloss_override_used: Optional[str] = None
    translation_provider: Optional[str] = None
    planner_provider: Optional[str] = None
    normalized_source_text: Optional[str] = None
    meaning_review_required: bool = False
    meaning_warning: Optional[str] = None


class BrollSlotResponse(BaseModel):
    id: str
    project_id: str
    transcript_id: Optional[str]
    start_sec: float
    end_sec: float
    anchor_word_ids: list[str] = Field(default_factory=list)
    concept_text: str
    locked: bool
    status: str
    review_status: str = "needs_review"
    visual_intent: Optional[str] = None
    review_summary: Optional[str] = None
    weak_reason_codes: list[str] = Field(default_factory=list)
    meaning: BrollMeaningResponse = Field(default_factory=BrollMeaningResponse)
    chosen_candidate_id: Optional[str]
    created_at: str
    updated_at: str
    candidates: list[BrollCandidateResponse] = Field(default_factory=list)


class BrollSuggestResponse(BaseModel):
    project_id: str
    transcript_id: Optional[str]
    created_slots: int
    slots: list[BrollSlotResponse]


class BrollPlanBeatResponse(BaseModel):
    id: str
    beat_index: int
    start_sec: float
    end_sec: float
    timeline_start_sec: Optional[float] = None
    timeline_end_sec: Optional[float] = None
    section_label: str
    intent_label: str
    source_strategy: str
    shot_style: str
    should_place: bool
    confidence: float
    rationale: str
    concept_text: str
    segment_text: str
    anchor_word_ids: list[str] = Field(default_factory=list)
    query_hints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BrollCoverageSectionResponse(BaseModel):
    section_label: str
    start_sec: float
    end_sec: float
    beat_count: int
    target_beats: int


class BrollPlanResponse(BaseModel):
    id: str
    project_id: str
    transcript_id: Optional[str]
    plan_version: str
    fallback_used: bool
    planner_model: Optional[str] = None
    created_at: str
    beats: list[BrollPlanBeatResponse] = Field(default_factory=list)
    uncovered_ranges: list[dict[str, float]] = Field(default_factory=list)
    coverage_sections: list[BrollCoverageSectionResponse] = Field(default_factory=list)


class BrollAutoApplyRequest(BaseModel):
    transcript_id: Optional[str] = None
    max_slots: int = Field(default=20, ge=1, le=40)
    candidates_per_slot: int = Field(default=3, ge=1, le=10)
    min_chunk_words: int = Field(default=4, ge=1, le=30)
    replace_existing: bool = True
    include_project_assets: bool = True
    include_external_sources: bool = True
    ai_rerank: bool = True
    clear_existing_overlay: bool = True
    fallback_to_top_candidate: bool = True
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    overlay_opacity: float = Field(default=1.0, ge=0.0, le=1.0)


class BrollAutoApplyResponse(BaseModel):
    project_id: str
    transcript_id: Optional[str]
    created_slots: int
    auto_chosen_slots: int
    synced_clip_count: int
    skipped_slots: int
    confidence_threshold: float
    skipped_slot_summaries: list["BrollAutoApplySkipSummary"] = Field(default_factory=list)
    timeline: TimelineState
    slots: list[BrollSlotResponse]


class BrollAutoApplySkipSummary(BaseModel):
    slot_id: str
    concept_text: str
    reason: str
    detail: Optional[str] = None


class BrollConfigResponse(BaseModel):
    external_enabled: bool
    pexels_configured: bool
    pixabay_configured: bool
    stock_search_available: bool
    generative_enabled: bool
    llm_rerank_available: bool


class BrollSyncRequest(BaseModel):
    transcript_id: Optional[str] = None
    clear_existing_overlay: bool = True
    overlay_opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    slot_ids: list[str] = Field(default_factory=list)


class BrollSyncResponse(BaseModel):
    project_id: str
    transcript_id: Optional[str]
    synced_clip_count: int
    timeline: TimelineState
    slots: list[BrollSlotResponse]


class BrollUndoResponse(BaseModel):
    project_id: str
    restored_clip_count: int
    timeline: TimelineState
    transaction_action: Optional[str] = None


class JobResponse(BaseModel):
    id: str
    project_id: str
    kind: str
    status: str
    progress: int
    stage: Optional[str] = None
    message: Optional[str] = None
    output_path: Optional[str]
    error: Optional[str]


class JobEventResponse(BaseModel):
    id: int
    job_id: str
    project_id: str
    stage: str
    status: str
    progress: int
    message: Optional[str]
    created_at: str


class RenderRequest(BaseModel):
    format: Literal["mp4", "mov", "webm"] = "mp4"
    aspect_ratio: Literal["16:9", "9:16"] = "16:9"
    resolution: Literal["720p", "1080p", "4k"] = "1080p"
    fps: Literal[24, 30, 60] = 30
    quality: Literal["low", "medium", "high", "max"] = "high"
    bitrate: Optional[str] = None


class VibeActionRequest(BaseModel):
    action: Literal["add_subtitles", "auto_cut_pauses", "trim_start_end"]
    asset_id: Optional[str] = None
    options: dict[str, Any] = Field(default_factory=dict)


class VibeActionResponse(BaseModel):
    project_id: str
    action: str
    transcript_id: Optional[str] = None
    details: Optional[str] = None
    timeline: TimelineState
    preview_job: JobResponse


class OperationHistoryItem(BaseModel):
    id: int
    project_id: str
    op_type: str
    source: str
    payload_json: str
    created_at: str
