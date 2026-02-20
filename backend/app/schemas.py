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


class ClipTransform(BaseModel):
    crop: Optional[Crop] = None
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
    type: Literal["fade", "dissolve", "slide_left", "slide_right", "slide_up", "slide_down", "zoom", "wipe"] = "fade"
    duration_sec: float = 0.5


class TextOverlay(BaseModel):
    id: str
    text: str
    start_sec: float
    duration_sec: float
    x: str = "(w-text_w)/2"
    y: str = "(h-text_h)-80"
    font_size: int = 48
    color: str = "white"
    style: Literal["static", "pop", "bounce", "typewriter", "karaoke", "fade"] = "static"


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
    start_sec: float
    end_sec: float
    confidence: Optional[float] = None


class TranscriptGenerateRequest(BaseModel):
    asset_id: str


class TranscriptCutRequest(BaseModel):
    transcript_id: str
    kept_word_ids: list[str] = Field(default_factory=list)
    context_sec: Optional[float] = Field(default=None, ge=0.0)
    merge_gap_sec: Optional[float] = Field(default=None, ge=0.0)
    min_removed_sec: Optional[float] = Field(default=None, ge=0.0)


class TranscriptResponse(BaseModel):
    id: str
    project_id: str
    asset_id: str
    source: str
    language: Optional[str]
    text: str
    words: list[TranscriptWord]
    duration_sec: float
    is_mock: bool
    created_at: str


class TranscriptGenerateResponse(BaseModel):
    transcript: TranscriptResponse
    timeline: TimelineState


class TranscriptCutResponse(BaseModel):
    project_id: str
    transcript_id: str
    kept_word_count: int
    removed_word_count: int
    timeline: TimelineState
    preview_job: Optional[JobResponse] = None


class BrollSuggestRequest(BaseModel):
    transcript_id: Optional[str] = None
    max_slots: int = Field(default=8, ge=1, le=40)
    candidates_per_slot: int = Field(default=3, ge=1, le=10)
    min_chunk_words: int = Field(default=4, ge=1, le=30)
    replace_existing: bool = True
    include_project_assets: bool = True
    include_external_sources: bool = True
    ai_rerank: bool = True


class BrollRerollRequest(BaseModel):
    candidates_per_slot: int = Field(default=3, ge=1, le=10)
    include_project_assets: bool = True
    include_external_sources: bool = True
    ai_rerank: bool = True


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
    reason: dict[str, Any] = Field(default_factory=dict)
    created_at: str


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
    chosen_candidate_id: Optional[str]
    created_at: str
    updated_at: str
    candidates: list[BrollCandidateResponse] = Field(default_factory=list)


class BrollSuggestResponse(BaseModel):
    project_id: str
    transcript_id: Optional[str]
    created_slots: int
    slots: list[BrollSlotResponse]


class BrollAutoApplyRequest(BaseModel):
    transcript_id: Optional[str] = None
    max_slots: int = Field(default=8, ge=1, le=40)
    candidates_per_slot: int = Field(default=3, ge=1, le=10)
    min_chunk_words: int = Field(default=4, ge=1, le=30)
    replace_existing: bool = True
    include_project_assets: bool = True
    include_external_sources: bool = True
    ai_rerank: bool = True
    clear_existing_overlay: bool = True
    fallback_to_top_candidate: bool = True
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    overlay_opacity: float = Field(default=0.85, ge=0.0, le=1.0)


class BrollAutoApplyResponse(BaseModel):
    project_id: str
    transcript_id: Optional[str]
    created_slots: int
    auto_chosen_slots: int
    synced_clip_count: int
    skipped_slots: int
    confidence_threshold: float
    timeline: TimelineState
    slots: list[BrollSlotResponse]


class JobResponse(BaseModel):
    id: str
    project_id: str
    kind: str
    status: str
    progress: int
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
