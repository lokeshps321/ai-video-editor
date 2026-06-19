export type Resolution = {
  width: number;
  height: number;
};

export type Crop = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type CropKeyframe = {
  time_sec: number;
  x: number;
  y: number;
};

export type ClipTransform = {
  crop: Crop | null;
  crop_keyframes: CropKeyframe[];
  scale: Resolution | null;
  rotate: number;
  flip: "horizontal" | "vertical" | null;
};

export type ClipAdjustments = {
  brightness: number;
  contrast: number;
  saturation: number;
  exposure: number;
  temperature: number;
  preset: string | null;
};

export type AudioKeyframe = {
  time_sec: number;
  volume: number;
};

export type ClipAudio = {
  volume: number;
  fade_in_sec: number;
  fade_out_sec: number;
  mute: boolean;
  keyframes: AudioKeyframe[];
};

export type Transition = {
  type: "fade" | "dissolve" | "slide_left" | "slide_right" | "slide_up" | "slide_down" | "zoom" | "wipe";
  duration_sec: number;
};

export type TextOverlay = {
  id: string;
  text: string;
  start_sec: number;
  duration_sec: number;
  x: string;
  y: string;
  font_size: number;
  font_name?: string | null;
  color: string;
  highlight_color?: string | null;
  outline_color?: string;
  outline_width?: number;
  shadow?: number;
  alignment?: number;
  margin_v?: number;
  style: string;
  word_timings?: Array<{
    text: string;
    start_sec: number;
    end_sec: number;
  }>;
};

export type Clip = {
  id: string;
  asset_id: string;
  start_sec: number;
  end_sec: number;
  timeline_start_sec: number;
  speed: number;
  broll_opacity: number;
  transform: ClipTransform;
  adjustments: ClipAdjustments;
  audio: ClipAudio;
  transition: Transition | null;
  text_overlays: TextOverlay[];
};

export type Track = {
  id: string;
  kind: "video" | "audio" | "text" | "overlay";
  clips: Clip[];
  volume: number;
  mute: boolean;
  solo: boolean;
};

export type Timeline = {
  fps: number;
  resolution: Resolution;
  tracks: Track[];
  duration_sec: number;
};

export type ExportAspectRatio = "16:9" | "9:16";

export type Project = {
  id: string;
  name: string;
  fps: number;
  width: number;
  height: number;
  timeline: Timeline;
  timeline_version?: number;
  timeline_can_undo?: boolean;
  timeline_can_redo?: boolean;
};

export type MediaAsset = {
  id: string;
  project_id: string;
  media_type: string;
  filename: string;
  storage_path: string;
  duration_sec: number | null;
};

export type PromptParse = {
  prompt: string;
  confidence: number;
  operations: Array<{ op_type: string; params: Record<string, unknown> }>;
  errors: string[];
  suggestions: string[];
};

export type Job = {
  id: string;
  project_id: string;
  kind: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  stage?: string | null;
  message?: string | null;
  output_path: string | null;
  error: string | null;
};

export type JobEvent = {
  id: number;
  job_id: string;
  project_id: string;
  stage: string;
  status: string;
  progress: number;
  message: string | null;
  created_at: string;
};

export type TranscriptMode = "auto" | "speech" | "song";

export type OperationHistoryItem = {
  id: number;
  project_id: string;
  op_type: string;
  source: string;
  payload_json: string;
  created_at: string;
};

export type TranscriptWord = {
  id: string;
  text: string;
  display_text?: string | null;
  script_tag?: "latin" | "indic" | "arabic" | "mixed" | "other" | null;
  language_hint?: string | null;
  start_sec: number;
  end_sec: number;
  confidence?: number | null;
  quality_score?: number | null;
  quality_label?: "trusted" | "weak" | null;
  source_pass?: "primary" | "retry" | "rescue" | "manual" | null;
  speaker_id?: string | null;
  speaker_label?: string | null;
};

export type TranscriptRegion = {
  start_sec: number;
  end_sec: number;
  status: "trusted" | "weak" | "blanked";
  reason?: string | null;
  word_ids: string[];
};

export type Transcript = {
  id: string;
  project_id: string;
  asset_id: string;
  source: string;
  language: string | null;
  text: string;
  words: TranscriptWord[];
  regions: TranscriptRegion[];
  script_tags?: string[];
  mixed_script?: boolean;
  duration_sec: number;
  is_mock: boolean;
  created_at: string;
};

export type TranscriptGenerateResponse = {
  transcript: Transcript;
  timeline: Timeline;
  reused_transcript?: boolean;
};

export type TranscriptEditResponse = {
  transcript: Transcript;
  timeline: Timeline;
  captions_synced: boolean;
};

export type TranscriptCutResponse = {
  project_id: string;
  transcript_id: string;
  kept_word_count: number;
  removed_word_count: number;
  timeline: Timeline;
};

export type VibeAction = "add_subtitles" | "auto_cut_pauses" | "trim_start_end";

export type VibeActionResponse = {
  project_id: string;
  action: VibeAction;
  transcript_id: string | null;
  details: string | null;
  timeline: Timeline;
  preview_job: Job;
};

export type BrollCandidate = {
  id: string;
  project_id: string;
  slot_id: string;
  asset_id: string | null;
  source_type: string;
  source_url: string | null;
  source_label: string | null;
  score: number;
  confidence: number | null;
  score_breakdown: Record<string, number>;
  entities: string[];
  visual_intent: string | null;
  weak_reason_codes: string[];
  reason: Record<string, unknown>;
  created_at: string;
};

export type BrollSlot = {
  id: string;
  project_id: string;
  transcript_id: string | null;
  start_sec: number;
  end_sec: number;
  anchor_word_ids: string[];
  concept_text: string;
  locked: boolean;
  status: string;
  review_status: string;
  visual_intent: string | null;
  review_summary: string | null;
  weak_reason_codes: string[];
  chosen_candidate_id: string | null;
  created_at: string;
  updated_at: string;
  candidates: BrollCandidate[];
};

export type BrollSuggestResponse = {
  project_id: string;
  transcript_id: string | null;
  created_slots: number;
  slots: BrollSlot[];
};

export type BrollAutoApplySkipSummary = {
  slot_id: string;
  concept_text: string;
  reason: string;
  detail?: string | null;
};

export type BrollConfig = {
  external_enabled: boolean;
  pexels_configured: boolean;
  pixabay_configured: boolean;
  stock_search_available: boolean;
  generative_enabled: boolean;
  llm_rerank_available: boolean;
};

export type BrollAutoApplyResponse = {
  project_id: string;
  transcript_id: string | null;
  created_slots: number;
  auto_chosen_slots: number;
  synced_clip_count: number;
  skipped_slots: number;
  confidence_threshold: number;
  skipped_slot_summaries?: BrollAutoApplySkipSummary[];
  timeline: Timeline;
  slots: BrollSlot[];
};

export type BrollSyncResponse = {
  project_id: string;
  transcript_id: string | null;
  synced_clip_count: number;
  timeline: Timeline;
  slots: BrollSlot[];
};

export type BrollUndoResponse = {
  project_id: string;
  restored_clip_count: number;
  timeline: Timeline;
  transaction_action: string | null;
};
