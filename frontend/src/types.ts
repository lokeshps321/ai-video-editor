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

export type ClipTransform = {
  crop: Crop | null;
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
  color: string;
  style: "static" | "pop" | "bounce" | "typewriter" | "karaoke" | "fade";
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

export type Project = {
  id: string;
  name: string;
  fps: number;
  width: number;
  height: number;
  timeline: Timeline;
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
  start_sec: number;
  end_sec: number;
  confidence?: number | null;
};

export type Transcript = {
  id: string;
  project_id: string;
  asset_id: string;
  source: string;
  language: string | null;
  text: string;
  words: TranscriptWord[];
  duration_sec: number;
  is_mock: boolean;
  created_at: string;
};

export type TranscriptGenerateResponse = {
  transcript: Transcript;
  timeline: Timeline;
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

export type BrollAutoApplyResponse = {
  project_id: string;
  transcript_id: string | null;
  created_slots: number;
  auto_chosen_slots: number;
  synced_clip_count: number;
  skipped_slots: number;
  confidence_threshold: number;
  timeline: Timeline;
  slots: BrollSlot[];
};
