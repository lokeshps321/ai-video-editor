import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Captions,
  Check,
  Clapperboard,
  Download,
  PlaySquare,
  Redo2,
  RefreshCw,
  RotateCcw,
  Scissors,
  ScissorsLineDashed,
  Sparkles,
  Trash2,
  Undo2,
  UploadCloud,
  Wand2,
} from "lucide-react";
import { api } from "./lib/api";
import type {
  BrollCandidate,
  BrollSlot,
  Clip,
  ExportAspectRatio,
  Job,
  MediaAsset,
  Project,
  Timeline as ProjectTimeline,
  Transcript,
  TranscriptRegion,
  TranscriptWord,
  VibeAction
} from "./types";
import Timeline, {
  type TimelineCaptionBlock,
  type TimelineCaptionSelection,
  type TimelineLane,
  type TimelineLaneClipSelection,
} from "./components/Timeline";
import { BrollCandidateCard } from "./components/BrollCandidateCard";
import { BRAND } from "./config/brand";
import {
  AI_ACTION_ITEMS,
  CAPTION_STYLE_CONFIG_BY_ID,
  CAPTION_STYLE_PRESETS,
  ENABLE_AGGRESSIVE_FILLER_SINGLE_WORDS,
  FEATURE_TAB_ITEMS,
  FILLER_MULTI_WORD_PHRASES,
  FILLER_SINGLE_WORDS_AGGRESSIVE,
  FILLER_SINGLE_WORDS_CONSERVATIVE,
  LOW_CONFIDENCE_THRESHOLD,
  LOW_CONFIDENCE_WARN_MIN_COUNT,
  LOW_CONFIDENCE_WARN_RATIO,
  TRANSCRIPT_LANGUAGE_OPTIONS,
  type FeatureTabId,
} from "./config/editor";

type TextBlock = {
  id: string;
  wordIds: string[];
  text: string;
  startSec: number;
  endSec: number;
};

type BrollIntensity = "low" | "medium" | "high";
type BrollAutoMode = "fast" | "balanced" | "creative";

type BrollGenerationPlan = {
  mode: BrollAutoMode;
  modeLabel: string;
  runtimeHint: string;
  maxSlots: number;
  candidatesPerSlot: number;
  includeProjectAssets: boolean;
  includeExternalSources: boolean;
  aiRerank: boolean;
  minConfidence: number;
  usedExternalFallback: boolean;
};

type InspectorTimelineSelection = TimelineLaneClipSelection;
type InspectorCaptionSelection = TimelineCaptionSelection;

function formatSeconds(value: number): string {
  if (!Number.isFinite(value)) return "0:00";
  const mins = Math.floor(value / 60);
  const secs = Math.floor(value % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function formatFixedSec(value: number): string {
  if (!Number.isFinite(value)) return "0.00";
  return value.toFixed(2);
}

function trimInlineText(value: string, maxLength = 32): string {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength - 1).trimEnd()}…`;
}

function clipTimelineDurationSec(clip: Clip): number {
  return Math.max((clip.end_sec - clip.start_sec) / Math.max(clip.speed, 0.01), 0.1);
}

function assColorToCss(color: string | undefined, fallback: string): string {
  if (!color) return fallback;
  if (!color.startsWith("&H")) return color;
  const raw = color.replace("&H", "").padStart(8, "0");
  const bb = raw.slice(2, 4);
  const gg = raw.slice(4, 6);
  const rr = raw.slice(6, 8);
  return `#${rr}${gg}${bb}`;
}

function findFirstCaptionTimeSec(timeline: ProjectTimeline): number | null {
  let first: number | null = null;
  for (const track of timeline.tracks) {
    if (track.kind !== "video") continue;
    for (const clip of track.clips) {
      for (const overlay of clip.text_overlays ?? []) {
        const absoluteStart = Math.max((clip.timeline_start_sec ?? 0) + (overlay.start_sec ?? 0), 0);
        if (first === null || absoluteStart < first) {
          first = absoluteStart;
        }
      }
    }
  }
  return first;
}

function computeBrollSlotBudget(project: Project, transcript: Transcript): number {
  const fallbackDuration = Math.max(
    1,
    transcript.words.length ? transcript.words[transcript.words.length - 1].end_sec : 0
  );
  const durationSec = Number.isFinite(transcript.duration_sec) && transcript.duration_sec > 0
    ? transcript.duration_sec
    : fallbackDuration;
  const vertical = project.height >= project.width;
  const targetCutSec = vertical ? 1.9 : 2.8;
  const raw = Math.round(durationSec / targetCutSec);
  const minSlots = vertical ? 6 : 4;
  const maxSlots = vertical ? 16 : 10;
  return Math.max(minSlots, Math.min(maxSlots, raw));
}

function clampInt(value: number, minValue: number, maxValue: number): number {
  return Math.max(minValue, Math.min(maxValue, value));
}

function resolveBrollGenerationPlan(
  project: Project,
  transcript: Transcript,
  intensity: BrollIntensity,
  mode: BrollAutoMode,
  projectVideoCount: number
): BrollGenerationPlan {
  const baseMaxSlots = computeBrollSlotBudget(project, transcript);
  const intensityMultiplier = intensity === "low" ? 0.82 : intensity === "high" ? 1.2 : 1.0;
  const isVertical = project.height >= project.width;
  const hasEnoughLocalBroll = projectVideoCount >= 2;

  if (mode === "fast") {
    const maxSlots = clampInt(
      Math.round(baseMaxSlots * 0.4 * intensityMultiplier),
      3,
      isVertical ? 6 : 5
    );
    const usedExternalFallback = !hasEnoughLocalBroll;
    return {
      mode,
      modeLabel: usedExternalFallback ? "Fast (external fallback)" : "Fast",
      runtimeHint: usedExternalFallback ? "1-3 min" : "<1 min target",
      maxSlots,
      candidatesPerSlot: usedExternalFallback ? 3 : 2,
      includeProjectAssets: true,
      includeExternalSources: usedExternalFallback,
      aiRerank: usedExternalFallback,
      minConfidence: intensity === "low" ? 0.82 : intensity === "high" ? 0.72 : 0.76,
      usedExternalFallback,
    };
  }

  if (mode === "balanced") {
    const maxSlots = clampInt(
      Math.round(baseMaxSlots * 0.72 * intensityMultiplier),
      isVertical ? 4 : 3,
      isVertical ? 10 : 8
    );
    return {
      mode,
      modeLabel: "Balanced",
      runtimeHint: "1-3 min",
      maxSlots,
      candidatesPerSlot: 4,
      includeProjectAssets: true,
      includeExternalSources: true,
      aiRerank: true,
      minConfidence: intensity === "low" ? 0.86 : intensity === "high" ? 0.74 : 0.8,
      usedExternalFallback: false,
    };
  }

  const maxSlots = clampInt(
    Math.round(baseMaxSlots * 1.08 * intensityMultiplier),
    isVertical ? 6 : 4,
    isVertical ? 16 : 10
  );
  return {
    mode,
    modeLabel: "Creative",
    runtimeHint: "3-8+ min",
    maxSlots,
    candidatesPerSlot: 5,
    includeProjectAssets: true,
    includeExternalSources: true,
    aiRerank: true,
    minConfidence: intensity === "low" ? 0.88 : intensity === "high" ? 0.76 : 0.82,
    usedExternalFallback: false,
  };
}

function mapEditedTimeToSourceTime(editedSec: number, timelineSortedClips: Clip[]): number {
  if (!Number.isFinite(editedSec)) return 0;
  if (!timelineSortedClips.length) return Math.max(editedSec, 0);

  const timelineSec = Math.max(editedSec, 0);
  for (const clip of timelineSortedClips) {
    const clipDuration = clipTimelineDurationSec(clip);
    const clipStart = clip.timeline_start_sec;
    const clipEnd = clipStart + clipDuration;

    if (timelineSec < clipStart) {
      return clip.start_sec;
    }
    if (timelineSec <= clipEnd) {
      return clip.start_sec + (timelineSec - clipStart) * Math.max(clip.speed, 0.01);
    }
  }

  return timelineSortedClips[timelineSortedClips.length - 1].end_sec;
}

function mapSourceTimeToEditedTime(sourceSec: number, timelineSortedClips: Clip[]): number {
  if (!Number.isFinite(sourceSec)) return 0;
  if (!timelineSortedClips.length) return Math.max(sourceSec, 0);

  const sourceTime = Math.max(sourceSec, 0);
  for (const clip of timelineSortedClips) {
    if (sourceTime >= clip.start_sec && sourceTime <= clip.end_sec) {
      return clip.timeline_start_sec + (sourceTime - clip.start_sec) / Math.max(clip.speed, 0.01);
    }
  }

  // If source time falls in a removed gap, snap to the nearest kept edge.
  let nearestEdited = timelineSortedClips[0].timeline_start_sec;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const clip of timelineSortedClips) {
    const clipDuration = clipTimelineDurationSec(clip);
    const candidates = [
      { source: clip.start_sec, edited: clip.timeline_start_sec },
      { source: clip.end_sec, edited: clip.timeline_start_sec + clipDuration }
    ];
    for (const candidate of candidates) {
      const distance = Math.abs(candidate.source - sourceTime);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestEdited = candidate.edited;
      }
    }
  }
  return nearestEdited;
}

function candidateSourceTag(sourceType: string): string {
  if (sourceType === "pexels_video") return "Pexels";
  if (sourceType === "pixabay_video") return "Pixabay";
  if (sourceType === "generated_video" || sourceType === "generated_image_video") return "GenAI";
  if (sourceType === "project_asset") return "Library";
  return sourceType;
}

function confidenceLabel(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "unknown";
  if (value >= 0.78) return "high";
  if (value >= 0.55) return "medium";
  return "low";
}

function transcriptRegionLabel(region: TranscriptRegion): string {
  if (region.status === "blanked") return "Blanked";
  if (region.status === "trusted") return "Trusted";
  return "Weak";
}

function candidateBreakdownChips(breakdown: Record<string, number>): string[] {
  const keys = ["semantic", "alignment", "specificity", "diversity", "content", "entity", "metadata", "crop", "duration"];
  const chips: string[] = [];
  keys.forEach((key) => {
    const value = breakdown[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      chips.push(`${key} ${(value * 100).toFixed(0)}%`);
    }
  });
  return chips.slice(0, 3);
}

function reasonCodeLabel(code: string): string {
  const customLabels: Record<string, string> = {
    confidence_low: "confidence low",
    crop_weak: "crop weak",
    generated_fallback: "gen fallback",
    intent_weak: "intent mismatch",
    no_candidates: "no candidates",
    semantic_weak: "semantic weak",
    specificity_low: "generic match",
    talking_head_risk: "talking-head risk",
  };
  if (customLabels[code]) return customLabels[code];
  return code.replace(/_/g, " ");
}

function reviewStatusLabel(status: string): string {
  if (status === "approved") return "approved";
  if (status === "ready") return "ready";
  if (status === "rejected") return "rejected";
  if (status === "unfilled") return "unfilled";
  return "needs review";
}

function resolveMediaPath(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${api.apiBase}${path}`;
}

export function resolveBrollCandidatePreviewParams(
  candidate: BrollCandidate,
  mediaById: Map<string, MediaAsset>
): { url: string; type: "image" | "video" } | null {
  if (candidate.reason?.thumbnail_url) {
    return { url: candidate.reason.thumbnail_url as string, type: "image" };
  }
  if (candidate.asset_id) {
    const asset = mediaById.get(candidate.asset_id);
    if (asset?.storage_path) {
      return { url: resolveMediaPath(asset.storage_path), type: "video" };
    }
  }
  const sourceUrl = candidate.source_url?.trim() ?? "";
  if (sourceUrl.startsWith("http://") || sourceUrl.startsWith("https://")) {
    return { url: sourceUrl, type: "video" };
  }
  return null;
}

function readReasonText(reason: Record<string, unknown> | undefined, key: string): string | null {
  const value = reason?.[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function readReasonNumber(reason: Record<string, unknown> | undefined, key: string): number | null {
  const value = reason?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function humanizeBrollMeta(value: string): string {
  return value.replace(/_/g, " ");
}

function buildSentenceBlocks(words: TranscriptWord[]): TextBlock[] {
  if (!words.length) return [];
  const blocks: TextBlock[] = [];
  let current: TranscriptWord[] = [];

  const flush = () => {
    if (!current.length) return;
    const id = `sent-${blocks.length + 1}`;
    blocks.push({
      id,
      wordIds: current.map((word) => word.id),
      text: current.map((word) => word.text).join(" "),
      startSec: current[0].start_sec,
      endSec: current[current.length - 1].end_sec
    });
    current = [];
  };

  for (let idx = 0; idx < words.length; idx += 1) {
    const word = words[idx];
    const prev = idx > 0 ? words[idx - 1] : null;
    if (current.length > 0 && prev && (word.start_sec - prev.end_sec) > 1.1) {
      flush();
    }

    current.push(word);
    const token = word.text.trim();
    const endsSentence = /[.!?]$/.test(token);
    const maxWordsReached = current.length >= 12;
    if (endsSentence || maxWordsReached) {
      flush();
    }
  }
  flush();
  return blocks;
}

function buildParagraphBlocks(sentences: TextBlock[]): TextBlock[] {
  if (!sentences.length) return [];
  const blocks: TextBlock[] = [];
  let current: TextBlock[] = [];

  const flush = () => {
    if (!current.length) return;
    const id = `para-${blocks.length + 1}`;
    blocks.push({
      id,
      wordIds: current.flatMap((item) => item.wordIds),
      text: current.map((item) => item.text).join(" "),
      startSec: current[0].startSec,
      endSec: current[current.length - 1].endSec
    });
    current = [];
  };

  for (let idx = 0; idx < sentences.length; idx += 1) {
    const sentence = sentences[idx];
    const prev = idx > 0 ? sentences[idx - 1] : null;
    if (current.length > 0 && prev && (sentence.startSec - prev.endSec) > 1.5) {
      flush();
    }
    current.push(sentence);
    if (current.length >= 3) {
      flush();
    }
  }
  flush();
  return blocks;
}

function normalizeFillerToken(text: string): string {
  return text.toLowerCase().replace(/^[^a-z0-9']+|[^a-z0-9']+$/g, "").trim();
}

function detectFillerWordIds(words: TranscriptWord[]): Set<string> {
  const result = new Set<string>();
  if (!words.length) return result;

  const tokens = words.map((word) => normalizeFillerToken(word.text));
  const singleWordSet = ENABLE_AGGRESSIVE_FILLER_SINGLE_WORDS
    ? new Set([...FILLER_SINGLE_WORDS_CONSERVATIVE, ...FILLER_SINGLE_WORDS_AGGRESSIVE])
    : FILLER_SINGLE_WORDS_CONSERVATIVE;

  for (let idx = 0; idx < words.length; idx += 1) {
    for (const phrase of FILLER_MULTI_WORD_PHRASES) {
      if (idx + phrase.length > words.length) continue;
      const matches = phrase.every((token, offset) => tokens[idx + offset] === token);
      if (!matches) continue;
      for (let offset = 0; offset < phrase.length; offset += 1) {
        result.add(words[idx + offset].id);
      }
      idx += phrase.length - 1;
      break;
    }
  }

  words.forEach((word, idx) => {
    if (result.has(word.id)) return;
    if (singleWordSet.has(tokens[idx])) {
      result.add(word.id);
    }
  });
  return result;
}

// ── Undo / Redo history ──────────────────────────────────────────────
type UndoEntry = { deletedIds: Set<string> };
const MAX_UNDO = 80;

interface WordProps {
  word: TranscriptWord;
  isDeleted: boolean;
  isSelected: boolean;
  isActive: boolean;
  isFiller: boolean;
  isSearchMatch: boolean;
  isCurrentMatch: boolean;
  hasLowConfidence: boolean;
  isWeakRegionWord: boolean;
  activeWordRef: React.RefObject<HTMLButtonElement | null>;
  isDraggingRef: React.MutableRefObject<boolean>;
  dragStartWordIdRef: React.MutableRefObject<string | null>;
  selectWord: (id: string, shiftHeld: boolean) => void;
  seekToWord: (word: TranscriptWord) => void;
  selectWordRange: (fromId: string, toId: string) => void;
  startEditing: (word: TranscriptWord) => void;
}

const Word = React.memo(({
  word,
  isDeleted,
  isSelected,
  isActive,
  isFiller,
  isSearchMatch,
  isCurrentMatch,
  hasLowConfidence,
  isWeakRegionWord,
  activeWordRef,
  isDraggingRef,
  dragStartWordIdRef,
  selectWord,
  seekToWord,
  selectWordRange,
  startEditing
}: WordProps) => {
  const className = [
    "word",
    isDeleted ? "deleted" : "",
    isSelected ? "selected" : "",
    isActive ? "active" : "",
    isFiller ? "filler" : "",
    isSearchMatch ? "searchMatch" : "",
    isCurrentMatch ? "currentMatch" : "",
    hasLowConfidence ? "lowConfidence" : "",
    isWeakRegionWord ? "weakRegion" : ""
  ]
    .filter(Boolean)
    .join(" ");

  const confidenceHint =
    typeof word.confidence === "number" ? ` · ${(word.confidence * 100).toFixed(0)}%` : "";
  const qualityHint =
    typeof word.quality_score === "number" ? ` · quality ${(word.quality_score * 100).toFixed(0)}%` : "";
  const passHint = word.source_pass ? ` · ${word.source_pass}` : "";
  const labelHint = word.quality_label ? ` · ${word.quality_label}` : "";

  return (
    <button
      id={`word-${word.id}`}
      type="button"
      className={className}
      ref={isActive ? activeWordRef : undefined}
      onMouseDown={(event) => {
        if (event.detail >= 2) return; // let double-click handle
        isDraggingRef.current = true;
        dragStartWordIdRef.current = word.id;
        selectWord(word.id, event.shiftKey);
        seekToWord(word);
      }}
      onMouseEnter={() => {
        if (isDraggingRef.current && dragStartWordIdRef.current) {
          selectWordRange(dragStartWordIdRef.current, word.id);
        }
      }}
      onDoubleClick={() => startEditing(word)}
      title={`${formatSeconds(word.start_sec)} – ${formatSeconds(word.end_sec)}${confidenceHint}${qualityHint}${labelHint}${passHint}`}
    >
      {word.text}
    </button>
  );
});

function App() {
  const [project, setProject] = useState<Project | null>(null);
  const [media, setMedia] = useState<MediaAsset[]>([]);
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>(null);
  const [transcriptLanguage, setTranscriptLanguage] = useState("auto");
  const [transcript, setTranscript] = useState<Transcript | null>(null);

  const [deletedWordIds, setDeletedWordIds] = useState<Set<string>>(new Set());
  const [selectedWordIds, setSelectedWordIds] = useState<Set<string>>(new Set());
  const [anchorWordId, setAnchorWordId] = useState<string | null>(null);

  // Undo / Redo
  const undoStack = useRef<UndoEntry[]>([]);
  const redoStack = useRef<UndoEntry[]>([]);

  // Inline editing
  const [editingWordId, setEditingWordId] = useState<string | null>(null);
  const [editingWordText, setEditingWordText] = useState("");
  const [rangeEditText, setRangeEditText] = useState("");
  const [updatingTranscriptRange, setUpdatingTranscriptRange] = useState(false);
  const editInputRef = useRef<HTMLInputElement | null>(null);

  // Search
  const [searchQuery, setSearchQuery] = useState("");
  const [searchMatchIndex, setSearchMatchIndex] = useState(0);

  // Drag selection
  const isDragging = useRef(false);
  const dragStartWordId = useRef<string | null>(null);

  // Loading states
  const [creatingProject, setCreatingProject] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [generatingTranscript, setGeneratingTranscript] = useState(false);
  const [transcriptStartedAtMs, setTranscriptStartedAtMs] = useState<number | null>(null);
  const [transcriptElapsedSec, setTranscriptElapsedSec] = useState(0);
  const [applyingCut, setApplyingCut] = useState(false);
  const [queueingPreview, setQueueingPreview] = useState(false);
  const [runningAction, setRunningAction] = useState<VibeAction | null>(null);
  const [brollSlots, setBrollSlots] = useState<BrollSlot[]>([]);
  const [loadingBrollSlots, setLoadingBrollSlots] = useState(false);
  const [suggestingBroll, setSuggestingBroll] = useState(false);
  const [autoApplyingBroll, setAutoApplyingBroll] = useState(false);
  const [syncingBroll, setSyncingBroll] = useState(false);
  const [undoingBroll, setUndoingBroll] = useState(false);
  const [brollActionKey, setBrollActionKey] = useState<string | null>(null);
  const [brollTimelineActionKey, setBrollTimelineActionKey] = useState<string | null>(null);
  const [brollDraftStartById, setBrollDraftStartById] = useState<Record<string, string>>({});
  const [brollDraftDurationById, setBrollDraftDurationById] = useState<Record<string, string>>({});
  const [brollDraftOpacityById, setBrollDraftOpacityById] = useState<Record<string, number>>({});
  const [brollSyncMode, setBrollSyncMode] = useState<"replace" | "append">("replace");
  const [brollAutoMode, setBrollAutoMode] = useState<BrollAutoMode>("fast");
  const [brollIntensity, setBrollIntensity] = useState<BrollIntensity>("medium");
  const [brollSuggestJob, setBrollSuggestJob] = useState<Job | null>(null);
  const [expandedBrollSlots, setExpandedBrollSlots] = useState<Record<string, boolean>>({});
  const [activeFeatureTab, setActiveFeatureTab] = useState<FeatureTabId>("broll_studio");
  const [featureDrawerOpen, setFeatureDrawerOpen] = useState(false);
  const [selectedTimelineClip, setSelectedTimelineClip] = useState<InspectorTimelineSelection | null>(null);
  const [selectedBrollClipId, setSelectedBrollClipId] = useState<string | null>(null);
  const [selectedCaptionBlock, setSelectedCaptionBlock] = useState<InspectorCaptionSelection | null>(null);
  const [lockedLaneIds, setLockedLaneIds] = useState<Set<string>>(() => new Set());

  // Captions
  const [captionStyle, setCaptionStyle] = useState<string>("basic_white");
  const [captionResultInfo, setCaptionResultInfo] = useState<string | null>(null);
  const [removingCaptions, setRemovingCaptions] = useState(false);
  const selectedCaptionStyleName = useMemo(
    () => CAPTION_STYLE_PRESETS.find((item) => item.id === captionStyle)?.name ?? captionStyle,
    [captionStyle]
  );

  // Export
  const [exportFormat, setExportFormat] = useState<"mp4" | "mov" | "webm">("mp4");
  const [exportAspectRatio, setExportAspectRatio] = useState<ExportAspectRatio>("9:16");
  const [exportResolution, setExportResolution] = useState<"720p" | "1080p" | "4k">("1080p");
  const [exportFps, setExportFps] = useState<24 | 30 | 60>(30);
  const [exportQuality, setExportQuality] = useState<"low" | "medium" | "high" | "max">("high");
  const [previewFrameAspectRatio, setPreviewFrameAspectRatio] = useState<ExportAspectRatio>("16:9");
  const [showExportFrameGuide, setShowExportFrameGuide] = useState(false);
  const [exportingVideo, setExportingVideo] = useState(false);
  const [exportJob, setExportJob] = useState<Job | null>(null);

  const [previewJob, setPreviewJob] = useState<Job | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewUpdateQueued, setPreviewUpdateQueued] = useState(false);
  const [currentTimeSec, setCurrentTimeSec] = useState(0);

  // Waveform data for timeline
  const [waveformPeaks, setWaveformPeaks] = useState<number[]>([]);

  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<"checking" | "ok" | "down">("checking");

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const lastAppliedSignatureRef = useRef<string>("");
  const lastAutoCutFailedSignatureRef = useRef<string | null>(null);
  const pendingPreviewRefreshRef = useRef(false);
  const transcriptBoxRef = useRef<HTMLDivElement | null>(null);
  const activeWordRef = useRef<HTMLButtonElement | null>(null);
  const autoCreateAttemptedRef = useRef(false);
  const pendingCaptionSeekRef = useRef<{ jobId: string; targetSec: number } | null>(null);
  const rafPendingRef = useRef(false);

  const videoAssets = useMemo(() => media.filter((asset) => asset.media_type === "video"), [media]);

  const selectedVideoAsset = useMemo(() => {
    if (!selectedAssetId) return videoAssets[0] ?? null;
    return videoAssets.find((asset) => asset.id === selectedAssetId) ?? null;
  }, [selectedAssetId, videoAssets]);

  const videoClips = useMemo<Clip[]>(() => {
    if (!project) return [];
    const videoTrack = project.timeline.tracks.find((track) => track.kind === "video");
    return (videoTrack?.clips ?? []).slice().sort((a, b) => a.timeline_start_sec - b.timeline_start_sec);
  }, [project]);

  const timelineLanes = useMemo<TimelineLane[]>(() => {
    if (!project) return [];
    let videoIndex = 0;
    let audioIndex = 0;
    const lanes: TimelineLane[] = [];
    for (const track of project.timeline.tracks) {
      if (track.kind !== "video" && track.kind !== "audio") continue;
      const label = track.kind === "video" ? `V${++videoIndex}` : `A${++audioIndex}`;
      lanes.push({
        id: track.id,
        label,
        kind: track.kind,
        clips: (track.clips ?? []).slice().sort((a, b) => a.timeline_start_sec - b.timeline_start_sec),
        mute: track.mute,
        solo: track.solo,
        volume: track.volume,
      });
    }
    return lanes;
  }, [project]);

  const overlayClips = useMemo<Clip[]>(() => {
    if (!project) return [];
    const overlayTrack = project.timeline.tracks.find((track) => track.kind === "overlay");
    return overlayTrack?.clips ?? [];
  }, [project]);

  const sortedOverlayClips = useMemo<Clip[]>(
    () => overlayClips.slice().sort((a, b) => a.timeline_start_sec - b.timeline_start_sec),
    [overlayClips]
  );

  const mediaById = useMemo(() => {
    const index = new Map<string, MediaAsset>();
    media.forEach((item) => index.set(item.id, item));
    return index;
  }, [media]);

  const assetUrlById = useMemo(() => {
    const index = new Map<string, string>();
    media.forEach((item) => {
      index.set(item.id, resolveMediaPath(item.storage_path));
    });
    return index;
  }, [media]);

  const assetDurationById = useMemo(() => {
    const index = new Map<string, number | null>();
    media.forEach((item) => {
      index.set(item.id, item.duration_sec ?? null);
    });
    return index;
  }, [media]);

  const captionBlocks = useMemo<TimelineCaptionBlock[]>(() => {
    const blocks: TimelineCaptionBlock[] = [];
    timelineLanes.forEach((lane) => {
      if (lane.kind !== "video") return;
      lane.clips.forEach((clip) => {
        const speed = Math.max(clip.speed, 0.01);
        (clip.text_overlays ?? []).forEach((overlay) => {
          blocks.push({
            id: overlay.id,
            clipId: clip.id,
            laneId: lane.id,
            laneLabel: lane.label,
            text: overlay.text,
            style: overlay.style,
            startSec: clip.timeline_start_sec + (overlay.start_sec / speed),
            durationSec: overlay.duration_sec / speed,
            clipTimelineStartSec: clip.timeline_start_sec,
            clipSourceDurationSec: Math.max(clip.end_sec - clip.start_sec, 0.05),
            clipSpeed: speed,
          });
        });
      });
    });
    return blocks.sort((a, b) => a.startSec - b.startSec);
  }, [timelineLanes]);

  const transcriptWordIndex = useMemo(() => {
    const index = new Map<string, number>();
    transcript?.words.forEach((word, idx) => {
      index.set(word.id, idx);
    });
    return index;
  }, [transcript]);

  const transcriptWordsById = useMemo(() => {
    const index = new Map<string, TranscriptWord>();
    transcript?.words.forEach((word) => {
      index.set(word.id, word);
    });
    return index;
  }, [transcript]);

  const timelineAssistWords = useMemo<TranscriptWord[]>(() => {
    if (!transcript) return [];
    return transcript.words
      .map((word) => {
        const editedStart = mapSourceTimeToEditedTime(word.start_sec, videoClips);
        const editedEnd = mapSourceTimeToEditedTime(word.end_sec, videoClips);
        return {
          ...word,
          start_sec: editedStart,
          end_sec: Math.max(editedEnd, editedStart),
        };
      })
      .sort((a, b) => a.start_sec - b.start_sec);
  }, [transcript, videoClips]);

  const sentenceBlocks = useMemo(() => buildSentenceBlocks(transcript?.words ?? []), [transcript?.words]);
  const paragraphBlocks = useMemo(() => buildParagraphBlocks(sentenceBlocks), [sentenceBlocks]);

  const deletedSignature = useMemo(() => Array.from(deletedWordIds).sort().join(","), [deletedWordIds]);

  const keptWordIds = useMemo(() => {
    if (!transcript) return [] as string[];
    return transcript.words.filter((word) => !deletedWordIds.has(word.id)).map((word) => word.id);
  }, [transcript, deletedWordIds]);

  const selectedTranscriptWords = useMemo(() => {
    if (!transcript) return [] as TranscriptWord[];
    return transcript.words.filter((word) => selectedWordIds.has(word.id));
  }, [transcript, selectedWordIds]);

  const selectedTranscriptRange = useMemo(() => {
    if (!selectedTranscriptWords.length) return null;
    const startWord = selectedTranscriptWords[0];
    const endWord = selectedTranscriptWords[selectedTranscriptWords.length - 1];
    return {
      startWordId: startWord.id,
      endWordId: endWord.id,
      startSec: startWord.start_sec,
      endSec: endWord.end_sec,
      text: selectedTranscriptWords.map((word) => word.text).join(" "),
      wordCount: selectedTranscriptWords.length,
    };
  }, [selectedTranscriptWords]);

  const transcriptIssueRegions = useMemo(() => {
    if (!transcript?.regions?.length) return [] as TranscriptRegion[];
    return transcript.regions.filter((region) => region.status !== "trusted");
  }, [transcript]);

  useEffect(() => {
    setRangeEditText(selectedTranscriptRange?.text ?? "");
  }, [selectedTranscriptRange?.startWordId, selectedTranscriptRange?.endWordId, selectedTranscriptRange?.text]);

  const lowConfidenceCount = useMemo(() => {
    if (!transcript) return 0;
    return transcript.words.filter(
      (word) => typeof word.confidence === "number" && word.confidence < LOW_CONFIDENCE_THRESHOLD
    ).length;
  }, [transcript]);
  const lowConfidenceRatio = useMemo(() => {
    if (!transcript || transcript.words.length === 0) return 0;
    return lowConfidenceCount / transcript.words.length;
  }, [transcript, lowConfidenceCount]);
  const shouldWarnLowConfidence = lowConfidenceCount >= LOW_CONFIDENCE_WARN_MIN_COUNT && lowConfidenceRatio >= LOW_CONFIDENCE_WARN_RATIO;

  const selectedVideoAssetUrl = useMemo(() => {
    if (!selectedVideoAsset) return null;
    return resolveMediaPath(selectedVideoAsset.storage_path);
  }, [selectedVideoAsset]);

  const previewSource = useMemo(() => {
    if (previewUrl) return previewUrl;
    if (!selectedVideoAsset) return null;
    return resolveMediaPath(selectedVideoAsset.storage_path);
  }, [previewUrl, selectedVideoAsset]);

  const previewShowsRenderedOutput = useMemo(() => {
    if (!previewUrl || !selectedVideoAssetUrl) return false;
    return previewUrl !== selectedVideoAssetUrl;
  }, [previewUrl, selectedVideoAssetUrl]);

  const previewRenderBusy = useMemo(() => {
    const status = previewJob?.status;
    return (
      queueingPreview ||
      applyingCut ||
      previewUpdateQueued ||
      status === "queued" ||
      status === "running"
    );
  }, [queueingPreview, applyingCut, previewUpdateQueued, previewJob?.status]);

  const previewStatusText = useMemo(() => {
    if (!previewJob) return "not queued";
    if (previewJob.status === "failed") return "failed";
    if (previewRenderBusy) {
      if (previewJob.status === "queued") return "queued...";
      if (previewJob.status === "running") {
        const progress = Math.max(0, Math.min(100, Math.round(previewJob.progress ?? 0)));
        return progress > 0 ? `rendering ${progress}%` : "rendering...";
      }
      return "updating latest edit...";
    }
    if (previewJob.status === "completed") return "up to date";
    return previewJob.status;
  }, [previewJob, previewRenderBusy]);

  const previewBusyDetail = useMemo(() => {
    if (!previewRenderBusy) return "";
    if (applyingCut) return "Applying cut and rendering...";
    if (previewUpdateQueued) return "Latest edit queued...";
    const message = previewJob?.message?.trim();
    if (message) return message;
    if (queueingPreview || previewJob?.status === "queued") return "Preparing render...";
    const progress = Math.max(0, Math.min(100, Math.round(previewJob?.progress ?? 0)));
    return progress > 0 ? `Rendering ${progress}%` : "Rendering...";
  }, [previewRenderBusy, applyingCut, queueingPreview, previewUpdateQueued, previewJob?.message, previewJob?.status, previewJob?.progress]);

  const previewProgress = useMemo(
    () => Math.max(0, Math.min(100, Math.round(previewJob?.progress ?? 0))),
    [previewJob?.progress]
  );

  const exportProgress = useMemo(
    () => Math.max(0, Math.min(100, Math.round(exportJob?.progress ?? 0))),
    [exportJob?.progress]
  );

  const exportStatusMessage = useMemo(() => {
    if (!exportJob) return "";
    if (exportJob.message?.trim()) return exportJob.message.trim();
    if (exportJob.status === "completed") return "Export completed.";
    if (exportJob.status === "failed") return exportJob.error ?? "Export failed.";
    return exportProgress > 0 ? `Rendering export ${exportProgress}%` : "Preparing export...";
  }, [exportJob, exportProgress]);

  useEffect(() => {
    const pending = pendingCaptionSeekRef.current;
    if (!pending || !previewJob || previewJob.status !== "completed" || previewJob.id !== pending.jobId) {
      return;
    }
    const element = videoRef.current;
    if (!element) return;
    const seekTarget = Math.max(0, pending.targetSec);
    const applySeek = () => {
      const duration = Number.isFinite(element.duration) ? element.duration : seekTarget + 0.1;
      const maxSeek = Math.max(0, duration - 0.05);
      const safeSeek = Math.min(seekTarget, maxSeek);
      element.currentTime = safeSeek;
      setCurrentTimeSec(safeSeek);
      pendingCaptionSeekRef.current = null;
    };
    if (element.readyState >= 1) {
      applySeek();
      return;
    }
    const onLoadedMetadata = () => applySeek();
    element.addEventListener("loadedmetadata", onLoadedMetadata, { once: true });
    return () => element.removeEventListener("loadedmetadata", onLoadedMetadata);
  }, [previewJob?.id, previewJob?.status, previewUrl]);

  // While a fresh preview is rendering, the visible player can still be the prior render.
  // In that state, keep transcript tracking on source-time so highlighting remains stable.
  const transcriptPlaybackTimeSec = useMemo(
    () => (previewRenderBusy ? Math.max(0, currentTimeSec) : mapEditedTimeToSourceTime(currentTimeSec, videoClips)),
    [previewRenderBusy, currentTimeSec, videoClips]
  );

  const activeWordId = useMemo(() => {
    if (!transcript) return null;

    const direct = transcript.words.find((word) => {
      if (!previewRenderBusy && deletedWordIds.has(word.id)) return false;
      return transcriptPlaybackTimeSec >= word.start_sec && transcriptPlaybackTimeSec <= word.end_sec;
    });
    if (direct) return direct.id;

    // If playhead sits inside a removed gap, snap highlight to the nearest kept word.
    const nextKept = transcript.words.find(
      (word) => !deletedWordIds.has(word.id) && word.start_sec >= transcriptPlaybackTimeSec
    );
    if (nextKept) return nextKept.id;

    for (let idx = transcript.words.length - 1; idx >= 0; idx -= 1) {
      const word = transcript.words[idx];
      if (deletedWordIds.has(word.id)) continue;
      if (word.end_sec <= transcriptPlaybackTimeSec) return word.id;
    }
    return null;
  }, [transcript, deletedWordIds, transcriptPlaybackTimeSec, previewRenderBusy]);

  const transcriptLanguageLabel = useMemo(
    () => TRANSCRIPT_LANGUAGE_OPTIONS.find((option) => option.value === transcriptLanguage)?.label ?? "Language: Auto",
    [transcriptLanguage]
  );

  const shouldShowLiveCaptionOverlay =
    (previewFrameAspectRatio === "16:9" && exportAspectRatio === "9:16") ||
    (!previewShowsRenderedOutput && (previewRenderBusy || previewUpdateQueued || !previewUrl));

  const livePreviewCaption = useMemo(() => {
    if (!project) return null;
    if (!shouldShowLiveCaptionOverlay) return null;
    const videoTrack = project.timeline.tracks.find((track) => track.kind === "video");
    if (!videoTrack) return null;

    const previewTimeSec = Math.max(0, currentTimeSec);
    for (const clip of videoTrack.clips) {
      const overlays = clip.text_overlays ?? [];
      if (!overlays.length) continue;
      for (const overlay of overlays) {
        const clipBaseSec = previewUrl ? clip.timeline_start_sec : clip.start_sec;
        const startSec = clipBaseSec + overlay.start_sec;
        const endSec = startSec + overlay.duration_sec;
        if (previewTimeSec < startSec || previewTimeSec > endSec) continue;
        return {
          text: overlay.text,
          fontName: overlay.font_name ?? "Arial",
          color: overlay.color,
          outlineColor: overlay.outline_color ?? "black@0.5",
          outlineWidth: overlay.outline_width ?? 2,
          shadow: overlay.shadow ?? 0,
          alignment: overlay.alignment ?? 2,
          marginV: overlay.margin_v ?? 80,
          fontSize: overlay.font_size ?? 40,
        };
      }
    }
    return null;
  }, [project, shouldShowLiveCaptionOverlay, previewUrl, currentTimeSec]);

  const selectedTimelineClipDetails = useMemo(() => {
    if (!selectedTimelineClip) return null;
    const lane = timelineLanes.find((item) => item.id === selectedTimelineClip.laneId);
    const clip = lane?.clips.find((item) => item.id === selectedTimelineClip.clipId);
    if (!lane || !clip) return null;
    const source = mediaById.get(clip.asset_id);
    return {
      lane,
      clip,
      source,
      durationSec: clipTimelineDurationSec(clip),
    };
  }, [selectedTimelineClip, timelineLanes, mediaById]);

  const selectedCaptionBlockDetails = useMemo(() => {
    if (!selectedCaptionBlock) return null;
    const lane = timelineLanes.find((item) => item.id === selectedCaptionBlock.laneId && item.kind === "video");
    const clip = lane?.clips.find((item) => item.id === selectedCaptionBlock.clipId);
    const overlay = clip?.text_overlays.find((item) => item.id === selectedCaptionBlock.overlayId);
    if (!lane || !clip || !overlay) return null;
    const speed = Math.max(clip.speed, 0.01);
    return {
      lane,
      clip,
      overlay,
      timelineStartSec: clip.timeline_start_sec + (overlay.start_sec / speed),
      durationSec: overlay.duration_sec / speed,
    };
  }, [selectedCaptionBlock, timelineLanes]);

  const selectedBrollClip = useMemo(() => {
    if (!selectedBrollClipId) return null;
    return sortedOverlayClips.find((clip) => clip.id === selectedBrollClipId) ?? null;
  }, [selectedBrollClipId, sortedOverlayClips]);

  const inspectorContext = useMemo(() => {
    if (selectedBrollClip) {
      return {
        kind: "broll_clip" as const,
        title: "B-roll Clip Selected",
        detail: `Start ${formatSeconds(selectedBrollClip.timeline_start_sec)} · ${formatSeconds(
          clipTimelineDurationSec(selectedBrollClip)
        )} duration`,
        suggestedTab: "broll_studio" as const,
      };
    }
    if (selectedCaptionBlockDetails) {
      return {
        kind: "caption_block" as const,
        title: "Caption Block Selected",
        detail: `${trimInlineText(selectedCaptionBlockDetails.overlay.text, 64)} · ${formatSeconds(
          selectedCaptionBlockDetails.timelineStartSec
        )} · ${formatSeconds(selectedCaptionBlockDetails.durationSec)}`,
        suggestedTab: "captions" as const,
      };
    }
    if (selectedTimelineClipDetails) {
      const { lane, clip } = selectedTimelineClipDetails;
      return {
        kind: "timeline_clip" as const,
        title: `${lane.label} Clip Selected`,
        detail: `Start ${formatSeconds(clip.timeline_start_sec)} · ${formatSeconds(
          clipTimelineDurationSec(clip)
        )} duration`,
        suggestedTab: "captions" as const,
      };
    }
    if (selectedWordIds.size > 0) {
      return {
        kind: "transcript" as const,
        title: "Transcript Selection",
        detail: `${selectedWordIds.size} word${selectedWordIds.size === 1 ? "" : "s"} selected`,
        suggestedTab: "captions" as const,
      };
    }
    return {
      kind: "project" as const,
      title: "Project Controls",
      detail: "Choose a clip or words to see contextual tools.",
      suggestedTab: "broll_studio" as const,
    };
  }, [selectedBrollClip, selectedCaptionBlockDetails, selectedTimelineClipDetails, selectedWordIds.size]);

  // Fetch waveform peaks whenever video asset changes
  useEffect(() => {
    if (!selectedVideoAsset) { setWaveformPeaks([]); return; }
    let cancelled = false;
    api.getWaveform(selectedVideoAsset.id).then((data) => {
      if (!cancelled) setWaveformPeaks(data.peaks);
    }).catch(() => { if (!cancelled) setWaveformPeaks([]); });
    return () => { cancelled = true; };
  }, [selectedVideoAsset]);

  useEffect(() => {
    const currentIds = new Set(brollSlots.map((slot) => slot.id));
    setExpandedBrollSlots((prev) => {
      const next: Record<string, boolean> = {};
      Object.entries(prev).forEach(([slotId, expanded]) => {
        if (expanded && currentIds.has(slotId)) {
          next[slotId] = true;
        }
      });
      if (Object.keys(next).length === Object.keys(prev).length) {
        return prev;
      }
      return next;
    });
  }, [brollSlots]);

  // Search matches
  const searchMatchIds = useMemo(() => {
    if (!transcript || !searchQuery.trim()) return [] as string[];
    const q = searchQuery.toLowerCase().trim();
    return transcript.words
      .filter((word) => word.text.toLowerCase().includes(q))
      .map((word) => word.id);
  }, [transcript, searchQuery]);

  // O(1) lookup set for search matches (avoids .includes() per word in render loop)
  const searchMatchIdSet = useMemo(() => new Set(searchMatchIds), [searchMatchIds]);

  // Filler word IDs
  const fillerWordIds = useMemo(() => {
    if (!transcript) return new Set<string>();
    return detectFillerWordIds(transcript.words);
  }, [transcript]);

  // ── Undo/Redo helpers ──────────────────────────────────────────────
  const pushUndo = useCallback(() => {
    undoStack.current.push({ deletedIds: new Set(deletedWordIds) });
    if (undoStack.current.length > MAX_UNDO) undoStack.current.shift();
    redoStack.current = [];
  }, [deletedWordIds]);

  function undo() {
    const entry = undoStack.current.pop();
    if (!entry) return;
    redoStack.current.push({ deletedIds: new Set(deletedWordIds) });
    setDeletedWordIds(entry.deletedIds);
  }

  function redo() {
    const entry = redoStack.current.pop();
    if (!entry) return;
    undoStack.current.push({ deletedIds: new Set(deletedWordIds) });
    setDeletedWordIds(entry.deletedIds);
  }

  // ── Core actions ───────────────────────────────────────────────────
  async function refreshMedia(projectId: string) {
    const items = await api.listMedia(projectId);
    setMedia(items);
    const firstVideo = items.find((asset) => asset.media_type === "video");
    if (!selectedAssetId && firstVideo) {
      setSelectedAssetId(firstVideo.id);
    }
  }

  async function refreshBrollSlots(projectId: string, transcriptId?: string) {
    if (!transcriptId) {
      setBrollSlots([]);
      return;
    }
    setLoadingBrollSlots(true);
    try {
      const slots = await api.listBrollSlots(projectId, transcriptId);
      setBrollSlots(slots);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoadingBrollSlots(false);
    }
  }

  async function queuePreview(
    force = false,
    overrides?: {
      aspectRatio?: ExportAspectRatio;
      fps?: 24 | 30 | 60;
    }
  ) {
    if (!project || queueingPreview) return;
    if (!force && previewJob && (previewJob.status === "queued" || previewJob.status === "running")) {
      pendingPreviewRefreshRef.current = true;
      setPreviewUpdateQueued(true);
      setNotice("Preview render in progress. Latest edit will render next.");
      return;
    }
    setQueueingPreview(true);
    setError(null);
    try {
      const job = await api.renderPreview(project.id, force, {
        aspect_ratio: overrides?.aspectRatio ?? exportAspectRatio,
        fps: overrides?.fps ?? exportFps,
      });
      setPreviewJob(job);
      setPreviewUpdateQueued(false);
      if (job.status === "completed" && job.output_path) {
        setPreviewUrl(resolveMediaPath(job.output_path));
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setQueueingPreview(false);
    }
  }

  async function applyTimelineOperations(
    operations: Array<{ op_type: string; params: Record<string, unknown>; source?: string }>,
    options?: { notice?: string | null; forcePreview?: boolean }
  ) {
    if (!project || !operations.length) return null;
    setError(null);
    try {
      const response = await api.applyOperations(project.id, operations);
      setProject((prev) => (prev ? { ...prev, timeline: response.timeline } : prev));
      if (options?.notice !== undefined) {
        setNotice(options.notice);
      }
      await queuePreview(!!options?.forcePreview);
      return response.timeline;
    } catch (err) {
      setError((err as Error).message);
      return null;
    }
  }

  const updateDeletedWords = useCallback((wordIds: string[], deleted: boolean) => {
    if (!wordIds.length) return;
    pushUndo();
    lastAutoCutFailedSignatureRef.current = null;
    setDeletedWordIds((prev) => {
      const next = new Set(prev);
      wordIds.forEach((id) => {
        if (deleted) {
          next.add(id);
        } else {
          next.delete(id);
        }
      });
      return next;
    });
  }, [pushUndo]);

  async function applyCut(signature: string, keptIds: string[], options?: { manual?: boolean }) {
    if (!project || !transcript || applyingCut) return;
    const manual = !!options?.manual;
    if (!manual && lastAutoCutFailedSignatureRef.current === signature) {
      return;
    }
    if (!keptIds.length) {
      setError("At least one word must remain. Restore some words before applying.");
      return;
    }

    setApplyingCut(true);
    setError(null);
    const previousDuration = project.timeline.duration_sec;
    try {
      const result = await api.applyTranscriptCut(project.id, transcript.id, keptIds, {
        contextSec: 0,
        mergeGapSec: 0.08,
        minRemovedSec: 0
      });
      setProject((prev) => (prev ? { ...prev, timeline: result.timeline } : prev));
      lastAppliedSignatureRef.current = signature;
      const nextDuration = result.timeline.duration_sec;
      const deltaSec = Math.max(0, previousDuration - nextDuration);
      const deltaLabel = deltaSec >= 0.01
        ? `Timeline shortened by ${deltaSec.toFixed(2)}s.`
        : "No additional timeline duration change.";
      setNotice(
        `Cut applied. Removed ${result.removed_word_count} word${result.removed_word_count === 1 ? "" : "s"}. ${deltaLabel}`
      );
      lastAutoCutFailedSignatureRef.current = null;
      await queuePreview(true);
    } catch (err) {
      setError((err as Error).message);
      if (!manual) {
        lastAutoCutFailedSignatureRef.current = signature;
        setNotice("Auto-cut paused after an error. Click Apply Cut to retry.");
      }
    } finally {
      setApplyingCut(false);
    }
  }

  async function createProject(
    name = BRAND.defaultProjectName,
    options?: { silent?: boolean }
  ) {
    const silent = !!options?.silent;
    setCreatingProject(true);
    setError(null);
    try {
      const created = await api.createProject(name.trim() || "Untitled Project");
      setProject(created);
      setMedia([]);
      setSelectedAssetId(null);
      setTranscript(null);
      setDeletedWordIds(new Set());
      setSelectedWordIds(new Set());
      setAnchorWordId(null);
      setPreviewJob(null);
      setPreviewUrl(null);
      setPreviewUpdateQueued(false);
      setBrollSlots([]);
      setBrollTimelineActionKey(null);
      setBrollDraftStartById({});
      setBrollDraftDurationById({});
      setBrollDraftOpacityById({});
      setSelectedTimelineClip(null);
      setSelectedBrollClipId(null);
      setSelectedCaptionBlock(null);
      setLockedLaneIds(new Set());
      undoStack.current = [];
      redoStack.current = [];
      setNotice(silent ? null : "Project created. Upload a video to start.");
      await refreshMedia(created.id);
    } catch (err) {
      setError((err as Error).message);
      setNotice(null);
    } finally {
      setCreatingProject(false);
    }
  }

  async function uploadVideo(file: File) {
    if (!project) return;
    setUploading(true);
    setError(null);
    try {
      const uploaded = await api.uploadMedia(project.id, file);
      setMedia((prev) => [uploaded, ...prev]);
      if (uploaded.media_type === "video") {
        setSelectedAssetId(uploaded.id);
        setPreviewUrl(resolveMediaPath(uploaded.storage_path));
      }
      setNotice("Video uploaded.");
    } catch (err) {
      setError((err as Error).message);
      setNotice(null);
    } finally {
      setUploading(false);
    }
  }

  async function generateTranscript() {
    if (!project || !selectedVideoAsset) return;
    const startedAtMs = Date.now();
    setGeneratingTranscript(true);
    setTranscriptStartedAtMs(startedAtMs);
    setTranscriptElapsedSec(0);
    setError(null);
    setNotice("Transcript generation started.");
    try {
      const language = transcriptLanguage === "auto" ? undefined : transcriptLanguage;
      const response = await api.generateTranscript(project.id, selectedVideoAsset.id, language, undefined);
      setTranscript(response.transcript);
      setProject((prev) => (prev ? { ...prev, timeline: response.timeline } : prev));
      setDeletedWordIds(new Set());
      setSelectedWordIds(new Set());
      setAnchorWordId(null);
      setBrollSlots([]);
      setBrollTimelineActionKey(null);
      setBrollDraftStartById({});
      setBrollDraftDurationById({});
      setBrollDraftOpacityById({});
      setSelectedTimelineClip(null);
      setSelectedBrollClipId(null);
      setSelectedCaptionBlock(null);
      lastAppliedSignatureRef.current = "";
      undoStack.current = [];
      redoStack.current = [];
      const reuseNotice =
        response.transcript.created_at !== undefined &&
          Date.now() - new Date(response.transcript.created_at).getTime() > 60000
          ? " (Reused existing transcript)"
          : "";

      setNotice(
        response.transcript.is_mock
          ? `Transcript generated (fallback mode). Install faster-whisper for higher accuracy.${reuseNotice}`
          : `Transcript generated with word timestamps.${reuseNotice}`
      );
      await refreshBrollSlots(project.id, response.transcript.id);
      await queuePreview();
    } catch (err) {
      setError((err as Error).message);
      setNotice(null);
    } finally {
      setGeneratingTranscript(false);
      setTranscriptStartedAtMs(null);
    }
  }

  async function runVibeAction(action: VibeAction) {
    if (!project) {
      setError("Create a project before applying actions.");
      return;
    }
    if (!selectedVideoAsset) {
      setError("Select a video before applying actions.");
      return;
    }
    if (runningAction) {
      setNotice("Another action is already running. Please wait.");
      return;
    }
    setRunningAction(action);
    setError(null);
    try {
      const options: Record<string, unknown> = {};
      if (action === "add_subtitles") {
        setNotice(
          transcript?.words?.length
            ? "Applying captions..."
            : "Generating transcript and applying captions. This may take over a minute for longer videos."
        );
        const selectedStyle = CAPTION_STYLE_PRESETS.find((item) => item.id === captionStyle) ?? CAPTION_STYLE_PRESETS[0];
        options.style = selectedStyle.id;
        options.caption_styles = CAPTION_STYLE_CONFIG_BY_ID;
        if (transcript?.words?.length) {
          // Use exactly what user sees in transcript UI, including an in-progress inline edit.
          const pendingEditText = editingWordText.trim();
          const subtitleWords: TranscriptWord[] = editingWordId && pendingEditText
            ? transcript.words.map((word) =>
              word.id === editingWordId ? { ...word, text: pendingEditText } : word
            )
            : transcript.words;
          options.words = subtitleWords;
        }
      }
      const response = await api.applyVibeAction(project.id, action, selectedVideoAsset.id, options);
      setProject((prev) => (prev ? { ...prev, timeline: response.timeline } : prev));
      if (action === "add_subtitles") {
        const firstCaptionTimeSec = findFirstCaptionTimeSec(response.timeline);
        if (firstCaptionTimeSec !== null) {
          pendingCaptionSeekRef.current = { jobId: response.preview_job.id, targetSec: firstCaptionTimeSec };
        }
        setCaptionResultInfo(response.details ?? null);
      }
      setPreviewJob(response.preview_job);
      if (response.preview_job.output_path) {
        setPreviewUrl(resolveMediaPath(response.preview_job.output_path));
      }
      if (response.transcript_id) {
        const latestTranscript = await api.getTranscript(project.id, response.transcript_id);
        setTranscript(latestTranscript);
        await refreshBrollSlots(project.id, latestTranscript.id);
      }
      setNotice(response.details ?? "Action applied.");
    } catch (err) {
      setError((err as Error).message);
      setNotice(null);
    } finally {
      setRunningAction(null);
    }
  }

  async function removeCaptions() {
    if (!project || !selectedVideoAsset || removingCaptions) return;
    setRemovingCaptions(true);
    setError(null);
    try {
      const response = await api.applyOperations(project.id, [
        { op_type: "clear_subtitles", params: { asset_id: selectedVideoAsset.id }, source: "ui" },
      ]);
      setProject((prev) => (prev ? { ...prev, timeline: response.timeline } : prev));
      setSelectedCaptionBlock(null);
      setCaptionResultInfo(null);
      setNotice("Captions removed.");
      await queuePreview(true);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setRemovingCaptions(false);
    }
  }

  async function exportVideo() {
    if (!project || exportingVideo) return;
    setExportingVideo(true);
    setError(null);
    try {
      const job = await api.renderExport(project.id, {
        format: exportFormat,
        aspect_ratio: exportAspectRatio,
        resolution: exportResolution,
        fps: exportFps,
        quality: exportQuality,
      });
      setExportJob(job);
      setNotice(`Export started (${exportAspectRatio}, ${exportResolution}, ${exportFormat}). Job ID: ${job.id}`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setExportingVideo(false);
    }
  }

  // ── Word interaction ───────────────────────────────────────────────
  const markSelectionDeleted = useCallback(() => {
    updateDeletedWords(Array.from(selectedWordIds), true);
  }, [selectedWordIds, updateDeletedWords]);

  const restoreSelection = useCallback(() => {
    updateDeletedWords(Array.from(selectedWordIds), false);
  }, [selectedWordIds, updateDeletedWords]);

  const restoreAllText = useCallback(() => {
    pushUndo();
    lastAutoCutFailedSignatureRef.current = null;
    setDeletedWordIds(new Set());
    setSelectedWordIds(new Set());
    setAnchorWordId(null);
  }, [pushUndo]);

  const removeFillerWords = useCallback(() => {
    if (!fillerWordIds.size) return;
    updateDeletedWords(Array.from(fillerWordIds), true);
    setNotice(`Marked ${fillerWordIds.size} filler word${fillerWordIds.size === 1 ? "" : "s"} as deleted.`);
  }, [fillerWordIds, updateDeletedWords]);

  const selectWord = useCallback((wordId: string, shiftHeld: boolean) => {
    if (!transcript) return;
    setSelectedTimelineClip(null);
    setSelectedBrollClipId(null);
    setSelectedCaptionBlock(null);
    if (!shiftHeld || !anchorWordId || !transcriptWordIndex.has(anchorWordId) || !transcriptWordIndex.has(wordId)) {
      setAnchorWordId(wordId);
      setSelectedWordIds(new Set([wordId]));
      return;
    }

    const anchorIndex = transcriptWordIndex.get(anchorWordId) ?? 0;
    const currentIndex = transcriptWordIndex.get(wordId) ?? 0;
    const minIndex = Math.min(anchorIndex, currentIndex);
    const maxIndex = Math.max(anchorIndex, currentIndex);
    const range = transcript.words.slice(minIndex, maxIndex + 1).map((word) => word.id);
    setSelectedWordIds(new Set(range));
  }, [transcript, anchorWordId, transcriptWordIndex]);

  const selectWordRange = useCallback((fromId: string, toId: string) => {
    if (!transcript) return;
    setSelectedTimelineClip(null);
    setSelectedBrollClipId(null);
    setSelectedCaptionBlock(null);
    const fromIdx = transcriptWordIndex.get(fromId) ?? 0;
    const toIdx = transcriptWordIndex.get(toId) ?? 0;
    const minIdx = Math.min(fromIdx, toIdx);
    const maxIdx = Math.max(fromIdx, toIdx);
    const range = transcript.words.slice(minIdx, maxIdx + 1).map((w) => w.id);
    setSelectedWordIds(new Set(range));
  }, [transcript, transcriptWordIndex]);

  const toggleBlock = useCallback((block: TextBlock) => {
    const allDeleted = block.wordIds.every((id) => deletedWordIds.has(id));
    updateDeletedWords(block.wordIds, !allDeleted);
  }, [deletedWordIds, updateDeletedWords]);

  const seekToWord = useCallback((word: TranscriptWord) => {
    if (!videoRef.current) return;
    const targetSec = previewRenderBusy ? Math.max(0, word.start_sec) : mapSourceTimeToEditedTime(word.start_sec, videoClips);
    videoRef.current.currentTime = targetSec;
    setCurrentTimeSec(targetSec);
  }, [previewRenderBusy, videoClips]);

  const seekToTranscriptTime = useCallback((sourceSec: number) => {
    if (!videoRef.current) return;
    const targetSec = previewRenderBusy ? Math.max(0, sourceSec) : mapSourceTimeToEditedTime(sourceSec, videoClips);
    videoRef.current.currentTime = targetSec;
    setCurrentTimeSec(targetSec);
  }, [previewRenderBusy, videoClips]);

  // ── Inline editing ─────────────────────────────────────────────────
  const startEditing = useCallback((word: TranscriptWord) => {
    setEditingWordId(word.id);
    setEditingWordText(word.text);
    setTimeout(() => editInputRef.current?.focus(), 0);
  }, []);

  const commitEdit = useCallback(() => {
    if (!editingWordId || !transcript) {
      setEditingWordId(null);
      return;
    }
    const trimmed = editingWordText.trim();
    if (!trimmed) {
      setEditingWordId(null);
      return;
    }
    // Update word text locally
    setTranscript((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        words: prev.words.map((w) =>
          w.id === editingWordId
            ? { ...w, text: trimmed, quality_label: "trusted", quality_score: 1, source_pass: "manual" }
            : w
        ),
        text: prev.words.map((w) => (w.id === editingWordId ? trimmed : w.text)).join(" ")
      };
    });
    // Fire-and-forget backend update
    if (project && transcript) {
      api.updateWordText(transcript.id, editingWordId, trimmed, project.id).catch(() => {
        /* ignore – local state is source of truth for now */
      });
    }
    setEditingWordId(null);
  }, [editingWordId, transcript, editingWordText, project]);

  const cancelEdit = useCallback(() => {
    setEditingWordId(null);
    setEditingWordText("");
  }, []);

  const applyTranscriptRangeUpdate = useCallback(async (
    mode: "replace" | "blank" | "preserve"
  ) => {
    if (!project || !transcript || !selectedTranscriptRange || updatingTranscriptRange) return;
    if (mode === "replace" && !rangeEditText.trim()) {
      setError("Range text cannot be empty.");
      return;
    }
    setUpdatingTranscriptRange(true);
    setError(null);
    try {
      const updated = await api.updateTranscriptRange(transcript.id, project.id, {
        start_word_id: selectedTranscriptRange.startWordId,
        end_word_id: selectedTranscriptRange.endWordId,
        mode,
        ...(mode === "replace" ? { text: rangeEditText.trim() } : {}),
      });
      setTranscript(updated);
      setDeletedWordIds(new Set());
      setSelectedWordIds(new Set());
      setAnchorWordId(null);
      undoStack.current = [];
      redoStack.current = [];
      setNotice(
        mode === "blank"
          ? "Marked selected transcript range as instrumental/blank."
          : mode === "preserve"
            ? "Selected repeated line will be treated as trusted."
            : "Updated selected transcript range."
      );
      await refreshBrollSlots(project.id, updated.id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setUpdatingTranscriptRange(false);
    }
  }, [project, transcript, selectedTranscriptRange, updatingTranscriptRange, rangeEditText, refreshBrollSlots]);

  const transcriptWordNodes = useMemo(() => {
    if (!transcript) return null;

    return transcript.words.map((word) => {
      const isDeleted = deletedWordIds.has(word.id);

      if (editingWordId === word.id) {
        return (
          <input
            key={word.id}
            ref={editInputRef}
            className="wordEditInput"
            value={editingWordText}
            onChange={(e) => setEditingWordText(e.target.value)}
            onBlur={commitEdit}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitEdit();
              if (e.key === "Escape") cancelEdit();
              e.stopPropagation();
            }}
            style={{ width: `${Math.max(editingWordText.length + 2, 4)}ch` }}
          />
        );
      }

      const isSelected = selectedWordIds.has(word.id);
      const isActive = activeWordId === word.id && (!isDeleted || previewRenderBusy);
      const isFiller = fillerWordIds.has(word.id) && !isDeleted;
      const isSearchMatch = searchMatchIdSet.has(word.id);
      const isCurrentMatch = searchMatchIds[searchMatchIndex] === word.id;
      const hasLowConfidence =
        !isDeleted && typeof word.confidence === "number" && word.confidence < LOW_CONFIDENCE_THRESHOLD;
      const isWeakRegionWord = !isDeleted && word.quality_label === "weak";

      return (
        <Word
          key={word.id}
          word={word}
          isDeleted={isDeleted}
          isSelected={isSelected}
          isActive={isActive}
          isFiller={isFiller}
          isSearchMatch={isSearchMatch}
          isCurrentMatch={isCurrentMatch}
          hasLowConfidence={hasLowConfidence}
          isWeakRegionWord={isWeakRegionWord}
          activeWordRef={activeWordRef}
          isDraggingRef={isDragging}
          dragStartWordIdRef={dragStartWordId}
          selectWord={selectWord}
          seekToWord={seekToWord}
          selectWordRange={selectWordRange}
          startEditing={startEditing}
        />
      );
    });
  }, [
    transcript,
    deletedWordIds,
    editingWordId,
    editingWordText,
    commitEdit,
    cancelEdit,
    selectedWordIds,
    activeWordId,
    previewRenderBusy,
    fillerWordIds,
    searchMatchIdSet,
    searchMatchIds,
    searchMatchIndex,
    selectWord,
    seekToWord,
    selectWordRange,
    startEditing,
  ]);

  const sentenceShortcutNodes = useMemo(() => {
    return sentenceBlocks.map((block) => {
      const allDeleted = block.wordIds.every((id) => deletedWordIds.has(id));
      return (
        <button
          key={block.id}
          type="button"
          className={`segmentBtn ${allDeleted ? "deleted" : ""}`}
          onClick={() => toggleBlock(block)}
          title={`${formatSeconds(block.startSec)} – ${formatSeconds(block.endSec)}`}
        >
          <span className="segTime">{formatSeconds(block.startSec)}</span>
          {block.text}
        </button>
      );
    });
  }, [sentenceBlocks, deletedWordIds, toggleBlock]);

  const paragraphShortcutNodes = useMemo(() => {
    return paragraphBlocks.map((block) => {
      const allDeleted = block.wordIds.every((id) => deletedWordIds.has(id));
      return (
        <button
          key={block.id}
          type="button"
          className={`segmentBtn ${allDeleted ? "deleted" : ""}`}
          onClick={() => toggleBlock(block)}
          title={`${formatSeconds(block.startSec)} – ${formatSeconds(block.endSec)}`}
        >
          <span className="segTime">{formatSeconds(block.startSec)}</span>
          {block.text}
        </button>
      );
    });
  }, [paragraphBlocks, deletedWordIds, toggleBlock]);

  const handleTimelineSeek = useCallback((sec: number) => {
    if (videoRef.current) videoRef.current.currentTime = sec;
    setCurrentTimeSec(sec);
  }, []);

  const clearEditorSelections = useCallback(() => {
    setSelectedTimelineClip(null);
    setSelectedBrollClipId(null);
    setSelectedCaptionBlock(null);
  }, []);

  const handleTimelineSelectWord = useCallback((id: string, shift: boolean) => {
    if (!transcript) return;
    selectWord(id, shift);
    const wd = transcript.words.find((w) => w.id === id);
    if (wd) seekToWord(wd);
  }, [transcript, selectWord, seekToWord]);

  const handleTimelineSelectWordsInRange = useCallback((startSec: number, endSec: number) => {
    if (!transcript) return;
    const ids = timelineAssistWords
      .filter((w) => w.start_sec >= startSec && w.end_sec <= endSec)
      .map((w) => w.id);
    setSelectedTimelineClip(null);
    setSelectedBrollClipId(null);
    setSelectedCaptionBlock(null);
    setSelectedWordIds(new Set(ids));
  }, [timelineAssistWords, transcript]);

  const handleTimelineMoveLaneClip = useCallback((selection: TimelineLaneClipSelection, timelineStartSec: number) => {
    if (lockedLaneIds.has(selection.laneId)) return;
    void applyTimelineOperations(
      [
        {
          op_type: "move_clip",
          params: {
            clip: selection.clipId,
            track_kind: selection.laneKind,
            timeline_start_sec: Number(Math.max(0, timelineStartSec).toFixed(3)),
            ripple: true,
          },
          source: "ui",
        },
      ],
      { notice: `${selection.laneLabel} clip moved.` }
    );
  }, [applyTimelineOperations, lockedLaneIds]);

  const handleTimelineTrimLaneClip = useCallback((selection: TimelineLaneClipSelection, nextRange: { startSec: number; endSec: number }) => {
    if (lockedLaneIds.has(selection.laneId)) return;
    void applyTimelineOperations(
      [
        {
          op_type: "trim_clip",
          params: {
            clip: selection.clipId,
            start_sec: nextRange.startSec,
            end_sec: nextRange.endSec,
          },
          source: "ui",
        },
      ],
      { notice: `${selection.laneLabel} clip trimmed.` }
    );
  }, [applyTimelineOperations, lockedLaneIds]);

  const handleTimelineToggleLaneMute = useCallback((lane: TimelineLane) => {
    void applyTimelineOperations(
      [
        {
          op_type: "set_volume",
          params: { track_id: lane.id, track_kind: lane.kind, mute: !lane.mute },
          source: "ui",
        },
      ],
      { notice: `${lane.label} ${lane.mute ? "unmuted" : "muted"}.` }
    );
  }, [applyTimelineOperations]);

  const handleTimelineToggleLaneSolo = useCallback((lane: TimelineLane) => {
    void applyTimelineOperations(
      [
        {
          op_type: "set_volume",
          params: { track_id: lane.id, track_kind: lane.kind, solo: !lane.solo },
          source: "ui",
        },
      ],
      { notice: `${lane.label} ${lane.solo ? "unsoloed" : "soloed"}.` }
    );
  }, [applyTimelineOperations]);

  const handleTimelineToggleLaneLock = useCallback((laneId: string) => {
    setLockedLaneIds((prev) => {
      const next = new Set(prev);
      if (next.has(laneId)) {
        next.delete(laneId);
      } else {
        next.add(laneId);
      }
      return next;
    });
  }, []);

  const handleTimelineMoveBrollClip = useCallback((clipId: string, timelineStartSec: number) => {
    if (brollTimelineActionKey) return;
    void setBrollClipStart(clipId, timelineStartSec);
  }, [brollTimelineActionKey]);

  const handleTimelineTrimBrollClip = useCallback((clipId: string, durationSec: number) => {
    if (brollTimelineActionKey) return;
    void setBrollClipDuration(clipId, durationSec);
  }, [brollTimelineActionKey]);

  const handleTimelineSetBrollOpacity = useCallback((clipId: string, opacity: number) => {
    if (brollTimelineActionKey) return;
    void setBrollClipOpacity(clipId, opacity);
  }, [brollTimelineActionKey]);

  const handleTimelineDeleteBrollClip = useCallback((clipId: string) => {
    if (brollTimelineActionKey) return;
    void removeBrollClipById(clipId);
  }, [brollTimelineActionKey]);

  const handleTimelineRerollBrollClip = useCallback((clipId: string) => {
    if (brollTimelineActionKey || brollActionKey) return;
    void rerollBrollFromTimelineClip(clipId);
  }, [brollTimelineActionKey, brollActionKey]);

  const handleTimelineSelectLaneClip = useCallback((selection: TimelineLaneClipSelection | null) => {
    setSelectedTimelineClip(selection);
    if (selection) {
      setSelectedBrollClipId(null);
      setSelectedCaptionBlock(null);
      setSelectedWordIds(new Set());
    }
  }, []);

  const handleTimelineSelectBrollClip = useCallback((clipId: string | null) => {
    setSelectedBrollClipId(clipId);
    if (clipId) {
      setSelectedTimelineClip(null);
      setSelectedCaptionBlock(null);
      setSelectedWordIds(new Set());
    }
  }, []);

  const handleTimelineSelectCaptionBlock = useCallback((selection: TimelineCaptionSelection | null) => {
    setSelectedCaptionBlock(selection);
    if (selection) {
      setSelectedTimelineClip(null);
      setSelectedBrollClipId(null);
      setSelectedWordIds(new Set());
    }
  }, []);

  const handleTimelineMoveCaptionBlock = useCallback((selection: TimelineCaptionSelection, startSec: number) => {
    void applyTimelineOperations(
      [
        {
          op_type: "move_text_overlay",
          params: {
            clip: selection.clipId,
            overlay: selection.overlayId,
            start_sec: startSec,
          },
          source: "ui",
        },
      ],
      { notice: "Caption block moved." }
    );
  }, [applyTimelineOperations]);

  const handleTimelineTrimCaptionBlock = useCallback((selection: TimelineCaptionSelection, startSec: number, durationSec: number) => {
    void applyTimelineOperations(
      [
        {
          op_type: "trim_text_overlay",
          params: {
            clip: selection.clipId,
            overlay: selection.overlayId,
            start_sec: startSec,
            duration_sec: durationSec,
          },
          source: "ui",
        },
      ],
      { notice: "Caption block trimmed." }
    );
  }, [applyTimelineOperations]);

  const deleteSelectedTimelineClip = useCallback(() => {
    if (!selectedTimelineClip) return;
    const currentSelection = selectedTimelineClip;
    void applyTimelineOperations(
      [
        {
          op_type: "delete_clip",
          params: { clip: currentSelection.clipId },
          source: "ui",
        },
      ],
      { notice: `${currentSelection.laneLabel} clip deleted.` }
    ).then(() => {
      setSelectedTimelineClip(null);
    });
  }, [applyTimelineOperations, selectedTimelineClip]);

  const splitSelectedTimelineClip = useCallback(() => {
    if (!selectedTimelineClipDetails) return;
    const clipStart = selectedTimelineClipDetails.clip.timeline_start_sec;
    const clipEnd = clipStart + selectedTimelineClipDetails.durationSec;
    if (currentTimeSec <= clipStart + 0.01 || currentTimeSec >= clipEnd - 0.01) {
      setNotice("Move the playhead inside the selected clip to split it.");
      return;
    }
    void applyTimelineOperations(
      [
        {
          op_type: "split_clip",
          params: {
            clip: selectedTimelineClipDetails.clip.id,
            at_sec: Number(currentTimeSec.toFixed(3)),
          },
          source: "ui",
        },
      ],
      { notice: `${selectedTimelineClipDetails.lane.label} clip split at playhead.` }
    );
  }, [applyTimelineOperations, currentTimeSec, selectedTimelineClipDetails]);

  const deleteSelectedCaptionBlock = useCallback(() => {
    if (!selectedCaptionBlock) return;
    const currentSelection = selectedCaptionBlock;
    void applyTimelineOperations(
      [
        {
          op_type: "delete_text_overlay",
          params: {
            clip: currentSelection.clipId,
            overlay: currentSelection.overlayId,
          },
          source: "ui",
        },
      ],
      { notice: "Caption block deleted." }
    ).then(() => {
      setSelectedCaptionBlock(null);
    });
  }, [applyTimelineOperations, selectedCaptionBlock]);

  const setSelectedTimelineClipSpeed = useCallback((speed: number) => {
    if (!selectedTimelineClipDetails) return;
    void applyTimelineOperations(
      [
        {
          op_type: "set_speed",
          params: {
            clip: selectedTimelineClipDetails.clip.id,
            speed,
          },
          source: "ui",
        },
      ],
      { notice: `${selectedTimelineClipDetails.lane.label} speed set to ${speed}x.` }
    );
  }, [applyTimelineOperations, selectedTimelineClipDetails]);

  const setSelectedTimelineClipVolume = useCallback((volume: number) => {
    if (!selectedTimelineClipDetails) return;
    void applyTimelineOperations(
      [
        {
          op_type: "set_volume",
          params: {
            clip: selectedTimelineClipDetails.clip.id,
            track_kind: selectedTimelineClipDetails.lane.kind,
            volume,
          },
          source: "ui",
        },
      ],
      { notice: `${selectedTimelineClipDetails.lane.label} volume updated.` }
    );
  }, [applyTimelineOperations, selectedTimelineClipDetails]);

  const toggleSelectedTimelineClipMute = useCallback(() => {
    if (!selectedTimelineClipDetails) return;
    const nextMute = !selectedTimelineClipDetails.clip.audio.mute;
    void applyTimelineOperations(
      [
        {
          op_type: "set_volume",
          params: {
            clip: selectedTimelineClipDetails.clip.id,
            track_kind: selectedTimelineClipDetails.lane.kind,
            mute: nextMute,
          },
          source: "ui",
        },
      ],
      { notice: `${selectedTimelineClipDetails.lane.label} clip ${nextMute ? "muted" : "unmuted"}.` }
    );
  }, [applyTimelineOperations, selectedTimelineClipDetails]);

  // ── Search navigation ──────────────────────────────────────────────
  function navigateSearch(direction: 1 | -1) {
    if (!searchMatchIds.length) return;
    const nextIdx = (searchMatchIndex + direction + searchMatchIds.length) % searchMatchIds.length;
    setSearchMatchIndex(nextIdx);
    // Scroll to matched word
    const wordEl = document.getElementById(`word-${searchMatchIds[nextIdx]}`);
    wordEl?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  // ── Effects ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!project?.id || !transcript?.id) {
      setBrollSlots([]);
      setBrollSuggestJob(null);
      setSuggestingBroll(false);
      return;
    }
    void refreshBrollSlots(project.id, transcript.id);
  }, [project?.id, transcript?.id]);

  useEffect(() => {
    setBrollDraftStartById(() => {
      const next: Record<string, string> = {};
      sortedOverlayClips.forEach((clip) => {
        next[clip.id] = formatFixedSec(clip.timeline_start_sec);
      });
      return next;
    });
    setBrollDraftDurationById(() => {
      const next: Record<string, string> = {};
      sortedOverlayClips.forEach((clip) => {
        next[clip.id] = formatFixedSec(clipTimelineDurationSec(clip));
      });
      return next;
    });
    setBrollDraftOpacityById(() => {
      const next: Record<string, number> = {};
      sortedOverlayClips.forEach((clip) => {
        const opacity = typeof clip.broll_opacity === "number" ? Math.max(0, Math.min(1, clip.broll_opacity)) : 1;
        next[clip.id] = opacity;
      });
      return next;
    });
  }, [sortedOverlayClips]);

  useEffect(() => {
    if (!selectedTimelineClip) return;
    const lane = timelineLanes.find((item) => item.id === selectedTimelineClip.laneId);
    const exists = !!lane?.clips.some((clip) => clip.id === selectedTimelineClip.clipId);
    if (!exists) {
      setSelectedTimelineClip(null);
    }
  }, [selectedTimelineClip, timelineLanes]);

  useEffect(() => {
    if (!selectedBrollClipId) return;
    const exists = sortedOverlayClips.some((clip) => clip.id === selectedBrollClipId);
    if (!exists) {
      setSelectedBrollClipId(null);
    }
  }, [selectedBrollClipId, sortedOverlayClips]);

  useEffect(() => {
    if (!selectedCaptionBlock) return;
    const exists = captionBlocks.some((block) => block.id === selectedCaptionBlock.overlayId && block.clipId === selectedCaptionBlock.clipId);
    if (!exists) {
      setSelectedCaptionBlock(null);
    }
  }, [selectedCaptionBlock, captionBlocks]);

  useEffect(() => {
    if (inspectorContext.kind === "project") return;
    if (activeFeatureTab !== inspectorContext.suggestedTab) {
      setActiveFeatureTab(inspectorContext.suggestedTab);
    }
  }, [inspectorContext.kind, inspectorContext.suggestedTab, activeFeatureTab]);

  useEffect(() => {
    if (!featureDrawerOpen) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setFeatureDrawerOpen(false);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [featureDrawerOpen]);

  useEffect(() => {
    let active = true;
    void api
      .health()
      .then(() => {
        if (active) setBackendStatus("ok");
      })
      .catch(() => {
        if (active) setBackendStatus("down");
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (project || creatingProject || autoCreateAttemptedRef.current || backendStatus === "down") {
      return;
    }
    autoCreateAttemptedRef.current = true;
    void createProject(BRAND.defaultProjectName, { silent: true });
  }, [project, creatingProject, backendStatus]);

  useEffect(() => {
    if (project?.name?.trim()) {
      document.title = `${project.name.trim()} | ${BRAND.productName}`;
      return;
    }
    document.title = BRAND.editorDocumentTitle;
  }, [project?.name]);

  useEffect(() => {
    if (!generatingTranscript || transcriptStartedAtMs === null) {
      return;
    }
    setTranscriptElapsedSec(Math.max(0, Math.floor((Date.now() - transcriptStartedAtMs) / 1000)));
    const interval = window.setInterval(() => {
      setTranscriptElapsedSec(Math.max(0, Math.floor((Date.now() - transcriptStartedAtMs) / 1000)));
    }, 1000);
    return () => window.clearInterval(interval);
  }, [generatingTranscript, transcriptStartedAtMs]);

  // Auto-apply cut debounce
  useEffect(() => {
    if (!project || !transcript) return;
    if (applyingCut) return;
    if (deletedSignature === lastAppliedSignatureRef.current) return;
    if (deletedSignature === lastAutoCutFailedSignatureRef.current) return;
    const handle = window.setTimeout(() => {
      void applyCut(deletedSignature, keptWordIds);
    }, 450);
    return () => window.clearTimeout(handle);
  }, [project?.id, transcript?.id, deletedSignature, keptWordIds, applyingCut]);

  // Preview polling — deps stabilized to [id, status] to prevent interval restart storms
  const previewJobId = previewJob?.id;
  const previewJobStatus = previewJob?.status;
  useEffect(() => {
    if (!previewJobId || (previewJobStatus !== "queued" && previewJobStatus !== "running")) return;
    const interval = window.setInterval(async () => {
      try {
        const refreshed = await api.getJob(previewJobId);
        if ((refreshed.status === "completed" || refreshed.status === "failed") && pendingPreviewRefreshRef.current) {
          pendingPreviewRefreshRef.current = false;
          setPreviewUpdateQueued(false);
          void queuePreview(true);
          return;
        }

        setPreviewJob(refreshed);
        if (refreshed.status === "completed" && refreshed.output_path) {
          setPreviewUrl(resolveMediaPath(refreshed.output_path));
        }
        if (refreshed.status === "failed") {
          setError(refreshed.error ?? "Preview render failed. Check logs.");
        }
      } catch {
        // Ignore transient polling errors
      }
    }, 1000);
    return () => window.clearInterval(interval);
  }, [previewJobId, previewJobStatus]);

  // B-roll suggest polling
  useEffect(() => {
    if (!project?.id || !brollSuggestJob || (brollSuggestJob.status !== "queued" && brollSuggestJob.status !== "running")) {
      return;
    }
    const interval = window.setInterval(async () => {
      try {
        const refreshed = await api.getJob(brollSuggestJob.id);
        setBrollSuggestJob(refreshed);
        if (refreshed.status === "completed") {
          const result = await api.getSuggestBrollResult(project.id, refreshed.id);
          setBrollSlots(result.slots);
          setSuggestingBroll(false);
          const reviewCount = result.slots.filter((slot) => slot.review_status === "needs_review").length;
          setNotice(
            `Generated ${result.created_slots} B-roll slot${result.created_slots === 1 ? "" : "s"}. ` +
            `${reviewCount} need review before sync.`
          );
        } else if (refreshed.status === "failed") {
          setSuggestingBroll(false);
          setError(refreshed.error ?? "B-roll generation failed.");
        }
      } catch {
        // Ignore transient polling errors.
      }
    }, 2500);
    return () => window.clearInterval(interval);
  }, [project?.id, brollSuggestJob]);

  // Export polling
  const exportJobId = exportJob?.id;
  const exportJobStatus = exportJob?.status;
  useEffect(() => {
    if (!exportJobId || (exportJobStatus !== "queued" && exportJobStatus !== "running")) {
      return;
    }
    const interval = window.setInterval(async () => {
      try {
        const refreshed = await api.getJob(exportJobId);
        setExportJob(refreshed);
        if (refreshed.status === "completed" && refreshed.output_path) {
          try {
            const fallbackFilename = `export.${exportFormat}`;
            const downloadedFilename = await api.downloadJobOutput(refreshed.id, fallbackFilename);
            setNotice(`Export completed. Downloaded ${downloadedFilename}.`);
          } catch (err) {
            setError((err as Error).message);
          }
        } else if (refreshed.status === "failed") {
          setError(refreshed.error ?? "Export video failed.");
        }
      } catch {
        // Ignore transient polling errors.
      }
    }, 1000);
    return () => window.clearInterval(interval);
  }, [exportJobId, exportJobStatus, exportFormat]);

  // Keyboard shortcuts
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const tagName = target?.tagName?.toLowerCase();
      const isEditableTarget =
        !!target?.closest("input,textarea,select,[contenteditable=\"true\"]") ||
        tagName === "input" ||
        tagName === "textarea" ||
        tagName === "select";

      if (!isEditableTarget && event.key === " ") {
        event.preventDefault();
        const player = videoRef.current;
        if (player) {
          if (player.paused) {
            void player.play();
          } else {
            player.pause();
          }
        }
      }

      if (!isEditableTarget && !editingWordId && (event.key === "Delete" || event.key === "Backspace")) {
        event.preventDefault();
        if (selectedCaptionBlock) {
          deleteSelectedCaptionBlock();
          return;
        }
        if (selectedBrollClipId) {
          void removeBrollClipById(selectedBrollClipId);
          return;
        }
        if (selectedTimelineClip) {
          deleteSelectedTimelineClip();
          return;
        }
        if (selectedWordIds.size > 0) {
          updateDeletedWords(Array.from(selectedWordIds), true);
        }
      }

      if (!isEditableTarget && !editingWordId && event.key.toLowerCase() === "s" && !event.ctrlKey && !event.metaKey) {
        if (selectedTimelineClip) {
          event.preventDefault();
          splitSelectedTimelineClip();
        }
      }

      if (event.key === "z" && (event.ctrlKey || event.metaKey) && !event.shiftKey) {
        event.preventDefault();
        undo();
      }

      if (event.key === "z" && (event.ctrlKey || event.metaKey) && event.shiftKey) {
        event.preventDefault();
        redo();
      }

      if (event.key === "y" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        redo();
      }

      if (event.key === "Escape") {
        if (editingWordId) {
          cancelEdit();
        } else {
          setSelectedWordIds(new Set());
          setAnchorWordId(null);
          clearEditorSelections();
        }
      }

      if (event.key === "f" && (event.ctrlKey || event.metaKey) && transcript) {
        event.preventDefault();
        document.getElementById("transcript-search")?.focus();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    clearEditorSelections,
    deleteSelectedCaptionBlock,
    deleteSelectedTimelineClip,
    editingWordId,
    removeBrollClipById,
    selectedBrollClipId,
    selectedCaptionBlock,
    selectedTimelineClip,
    selectedWordIds,
    splitSelectedTimelineClip,
    transcript,
    updateDeletedWords,
  ]);

  // Auto-scroll to active word during playback
  useEffect(() => {
    if (!activeWordId || editingWordId) return;
    const el = document.getElementById(`word-${activeWordId}`);
    if (el && transcriptBoxRef.current) {
      const box = transcriptBoxRef.current;
      const elTop = el.offsetTop - box.offsetTop;
      const elBottom = elTop + el.offsetHeight;
      const scrollTop = box.scrollTop;
      const boxHeight = box.clientHeight;
      if (elTop < scrollTop || elBottom > scrollTop + boxHeight) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  }, [activeWordId, editingWordId]);

  // Drag selection handler — mouseup
  useEffect(() => {
    function onMouseUp() {
      isDragging.current = false;
      dragStartWordId.current = null;
    }
    window.addEventListener("mouseup", onMouseUp);
    return () => window.removeEventListener("mouseup", onMouseUp);
  }, []);

  async function suggestBroll() {
    if (!project || !transcript || suggestingBroll) return;
    setSuggestingBroll(true);
    setError(null);
    try {
      const plan = resolveBrollGenerationPlan(
        project,
        transcript,
        brollIntensity,
        brollAutoMode,
        videoAssets.length
      );
      const queued = await api.suggestBrollAsync(project.id, {
        transcript_id: transcript.id,
        max_slots: plan.maxSlots,
        candidates_per_slot: plan.candidatesPerSlot,
        min_chunk_words: 4,
        replace_existing: true,
        include_project_assets: plan.includeProjectAssets,
        include_external_sources: plan.includeExternalSources,
        ai_rerank: plan.aiRerank,
      });
      setBrollSuggestJob(queued);
      setNotice(
        `${plan.modeLabel} B-roll queued (${queued.status}, ${queued.progress}%). ` +
        `Target runtime: ${plan.runtimeHint}.${plan.usedExternalFallback ? " Added external fallback due to limited local video assets." : ""}`
      );
    } catch (err) {
      setError((err as Error).message);
      setSuggestingBroll(false);
    }
  }

  async function autoApplyBroll() {
    if (!project || !transcript || autoApplyingBroll) return;
    setAutoApplyingBroll(true);
    setError(null);
    const plan = resolveBrollGenerationPlan(
      project,
      transcript,
      brollIntensity,
      brollAutoMode,
      videoAssets.length
    );
    setNotice(
      `Auto-applying ${plan.modeLabel} B-roll. Target runtime: ${plan.runtimeHint}.` +
      (plan.usedExternalFallback ? " Using external fallback due to limited local video assets." : "")
    );
    try {
      const response = await api.autoApplyBroll(project.id, {
        transcript_id: transcript.id,
        max_slots: plan.maxSlots,
        candidates_per_slot: plan.candidatesPerSlot,
        min_chunk_words: 4,
        replace_existing: true,
        include_project_assets: plan.includeProjectAssets,
        include_external_sources: plan.includeExternalSources,
        ai_rerank: plan.aiRerank,
        clear_existing_overlay: true,
        fallback_to_top_candidate: false,
        min_confidence: plan.minConfidence,
        overlay_opacity: 0.85,
      });
      setBrollSlots(response.slots);
      setProject((prev) => (prev ? { ...prev, timeline: response.timeline } : prev));
      await refreshMedia(project.id);
      setNotice(
        `Auto-applied B-roll: ${response.auto_chosen_slots} chosen, ${response.synced_clip_count} synced, ${response.skipped_slots} skipped (threshold ${(response.confidence_threshold * 100).toFixed(0)}%).`
      );
      await queuePreview();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setAutoApplyingBroll(false);
    }
  }

  async function chooseBroll(slotId: string, candidateId: string) {
    if (!project || brollActionKey) return;
    setBrollActionKey(`choose:${slotId}:${candidateId}`);
    setError(null);
    try {
      const updated = await api.chooseBrollCandidate(project.id, slotId, candidateId);
      setBrollSlots((prev) => prev.map((slot) => (slot.id === slotId ? updated : slot)));
      await refreshMedia(project.id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBrollActionKey(null);
    }
  }

  async function rejectBroll(slotId: string) {
    if (!project || brollActionKey) return;
    setBrollActionKey(`reject:${slotId}`);
    setError(null);
    try {
      const updated = await api.rejectBrollSlot(project.id, slotId);
      setBrollSlots((prev) => prev.map((slot) => (slot.id === slotId ? updated : slot)));
      setNotice("Rejected B-roll slot. It will be skipped in future syncs.");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBrollActionKey(null);
    }
  }

  async function rerollBroll(slotId: string) {
    if (!project || brollActionKey) return;
    const previousSlot = brollSlots.find((slot) => slot.id === slotId) ?? null;
    const previousCount = previousSlot?.candidates.length ?? 0;
    setBrollActionKey(`reroll:${slotId}`);
    setError(null);
    try {
      const updated = await api.rerollBrollSlot(project.id, slotId, {
        candidates_per_slot: 2,
        include_project_assets: true,
        include_external_sources: true,
        ai_rerank: true,
      });
      const nextCount = updated.candidates.length;
      const addedCount = Math.max(0, nextCount - previousCount);
      setBrollSlots((prev) => prev.map((slot) => (slot.id === slotId ? updated : slot)));
      setNotice(
        addedCount > 0
          ? `Added ${addedCount} new B-roll variant${addedCount === 1 ? "" : "s"} for this slot.`
          : "Rerolled slot candidates."
      );
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBrollActionKey(null);
    }
  }

  async function syncBrollToTimeline(slotIds?: string[]) {
    if (!project || syncingBroll) return;
    const chosenSlots = brollSlots
      .filter((slot) => slot.chosen_candidate_id && (!slotIds || slotIds.includes(slot.id)))
      .sort((a, b) => a.start_sec - b.start_sec);

    if (!chosenSlots.length) {
      setNotice("No chosen B-roll slots to sync.");
      return;
    }

    setSyncingBroll(true);
    setError(null);
    try {
      const clearExistingOverlay = brollSyncMode === "replace";
      const response = await api.syncBroll(project.id, {
        transcript_id: transcript?.id,
        clear_existing_overlay: clearExistingOverlay,
        overlay_opacity: 0.85,
        slot_ids: slotIds ?? [],
      });
      setProject((prev) => (prev ? { ...prev, timeline: response.timeline } : prev));
      setBrollSlots(response.slots);
      setNotice(`Synced ${response.synced_clip_count} B-roll clip${response.synced_clip_count === 1 ? "" : "s"} to timeline.`);
      await queuePreview();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSyncingBroll(false);
    }
  }

  async function undoBrollLayer() {
    if (!project || undoingBroll) return;
    setUndoingBroll(true);
    setError(null);
    try {
      const response = await api.undoBrollTransaction(project.id);
      setProject((prev) => (prev ? { ...prev, timeline: response.timeline } : prev));
      if (transcript) {
        await refreshBrollSlots(project.id, transcript.id);
      }
      setNotice(
        `Restored ${response.restored_clip_count} overlay clip${response.restored_clip_count === 1 ? "" : "s"} from previous B-roll transaction.`
      );
      await queuePreview();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setUndoingBroll(false);
    }
  }

  function isBrollTimelineClipBusy(clipId: string): boolean {
    return brollTimelineActionKey?.endsWith(`:${clipId}`) ?? false;
  }

  function getOverlayClipById(clipId: string): Clip | null {
    return sortedOverlayClips.find((clip) => clip.id === clipId) ?? null;
  }

  function findSlotForOverlayClip(clip: Clip): BrollSlot | null {
    const clipStart = clip.timeline_start_sec;
    const clipEnd = clip.timeline_start_sec + clipTimelineDurationSec(clip);
    const ranked = brollSlots
      .filter((slot) => !!slot.chosen_candidate_id)
      .map((slot) => {
        const chosen = slot.candidates.find((candidate) => candidate.id === slot.chosen_candidate_id) ?? null;
        const assetMatch = chosen?.asset_id === clip.asset_id ? 1 : 0;
        const overlap = Math.max(0, Math.min(slot.end_sec, clipEnd) - Math.max(slot.start_sec, clipStart));
        const startDelta = Math.abs(slot.start_sec - clipStart);
        const score = (assetMatch * 1000) + (overlap * 10) - startDelta;
        return { slot, score };
      })
      .sort((a, b) => b.score - a.score);
    return ranked[0]?.slot ?? null;
  }

  async function rerollBrollFromTimelineClip(clipId: string) {
    if (!project || brollActionKey) return;
    const clip = getOverlayClipById(clipId);
    if (!clip) return;
    const slot = findSlotForOverlayClip(clip);
    if (!slot) {
      setNotice("No linked B-roll slot found for this clip.");
      return;
    }

    setBrollActionKey(`reroll-clip:${clipId}`);
    setError(null);
    try {
      const localNext = slot.candidates.find((candidate) => candidate.id !== slot.chosen_candidate_id);
      const updatedSlot = localNext
        ? slot
        : await api.rerollBrollSlot(project.id, slot.id, {
          candidates_per_slot: 2,
          include_project_assets: true,
          include_external_sources: true,
          ai_rerank: true,
        });

      if (!localNext) {
        setBrollSlots((prev) => prev.map((item) => (item.id === slot.id ? updatedSlot : item)));
      }

      const chooseCandidateId = localNext?.id
        ?? updatedSlot.candidates.find((candidate) => candidate.id !== updatedSlot.chosen_candidate_id)?.id;
      if (!chooseCandidateId) {
        throw new Error("No alternate B-roll variant available for this clip.");
      }

      const chosenSlot = await api.chooseBrollCandidate(project.id, slot.id, chooseCandidateId);
      setBrollSlots((prev) => prev.map((item) => (item.id === slot.id ? chosenSlot : item)));
      const chosenCandidate = chosenSlot.candidates.find((candidate) => candidate.id === chosenSlot.chosen_candidate_id) ?? null;
      if (!chosenCandidate?.asset_id) {
        throw new Error("Chosen variant is missing a video asset.");
      }

      const latestMedia = await api.listMedia(project.id);
      setMedia(latestMedia);
      const latestMediaById = new Map(latestMedia.map((item) => [item.id, item]));
      const sourceDuration = clip.end_sec - clip.start_sec;
      const maybeDuration = latestMediaById.get(chosenCandidate.asset_id)?.duration_sec;
      const boundedDuration = (typeof maybeDuration === "number" && maybeDuration > 0)
        ? Math.min(maybeDuration, sourceDuration)
        : sourceDuration;
      const opacity = typeof clip.broll_opacity === "number" ? clip.broll_opacity : 0.85;

      await applyBrollTimelineOperations(
        clip.id,
        "reroll",
        [
          {
            op_type: "delete_broll_clip",
            params: { clip: clip.id },
            source: "ui",
          },
          {
            op_type: "add_broll_clip",
            params: {
              asset_id: chosenCandidate.asset_id,
              start_sec: 0,
              end_sec: Number(Math.max(0.1, boundedDuration).toFixed(3)),
              timeline_start_sec: Number(clip.timeline_start_sec.toFixed(3)),
              opacity: Number(Math.max(0, Math.min(1, opacity)).toFixed(3)),
            },
            source: "ui",
          },
        ],
        "Rerolled B-roll clip variant."
      );
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBrollActionKey(null);
    }
  }

  async function applyBrollTimelineOperations(
    clipId: string,
    action: "move" | "trim" | "opacity" | "delete" | "reroll",
    operations: Array<{ op_type: string; params: Record<string, unknown>; source?: string }>,
    noticeMessage: string
  ) {
    if (!project || !operations.length) return;
    setBrollTimelineActionKey(`${action}:${clipId}`);
    setError(null);
    try {
      const response = await api.applyOperations(project.id, operations);
      setProject((prev) => (prev ? { ...prev, timeline: response.timeline } : prev));
      setNotice(noticeMessage);
      await queuePreview();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBrollTimelineActionKey(null);
    }
  }

  async function setBrollClipStart(clipId: string, requestedStartSec: number) {
    const clip = getOverlayClipById(clipId);
    if (!clip) return;
    const nextStart = Number(Math.max(0, requestedStartSec).toFixed(3));
    const current = Number(clip.timeline_start_sec.toFixed(3));
    if (Math.abs(nextStart - current) < 0.001) {
      setBrollDraftStartById((prev) => ({ ...prev, [clip.id]: formatFixedSec(current) }));
      return;
    }

    await applyBrollTimelineOperations(
      clip.id,
      "move",
      [
        {
          op_type: "move_broll_clip",
          params: { clip: clip.id, timeline_start_sec: nextStart },
          source: "ui",
        },
      ],
      "Updated B-roll start time."
    );
  }

  async function setBrollClipDuration(clipId: string, requestedDurationSec: number) {
    const clip = getOverlayClipById(clipId);
    if (!clip) return;
    if (!Number.isFinite(requestedDurationSec) || requestedDurationSec <= 0) {
      setBrollDraftDurationById((prev) => ({ ...prev, [clip.id]: formatFixedSec(clipTimelineDurationSec(clip)) }));
      return;
    }

    const currentDuration = clipTimelineDurationSec(clip);
    if (Math.abs(requestedDurationSec - currentDuration) < 0.01) {
      setBrollDraftDurationById((prev) => ({ ...prev, [clip.id]: formatFixedSec(currentDuration) }));
      return;
    }

    const maxByAsset = mediaById.get(clip.asset_id)?.duration_sec ?? null;
    const proposedEnd = clip.start_sec + (requestedDurationSec * Math.max(clip.speed, 0.01));
    let boundedEnd = proposedEnd;
    if (typeof maxByAsset === "number" && maxByAsset > 0) {
      boundedEnd = Math.min(boundedEnd, maxByAsset);
    }
    boundedEnd = Math.max(clip.start_sec + 0.1, boundedEnd);

    await applyBrollTimelineOperations(
      clip.id,
      "trim",
      [
        {
          op_type: "trim_broll_clip",
          params: { clip: clip.id, start_sec: clip.start_sec, end_sec: Number(boundedEnd.toFixed(3)) },
          source: "ui",
        },
      ],
      "Updated B-roll duration."
    );
  }

  async function setBrollClipOpacity(clipId: string, nextOpacity: number) {
    const clip = getOverlayClipById(clipId);
    if (!clip) return;
    const clamped = Math.max(0, Math.min(1, nextOpacity));
    const current = typeof clip.broll_opacity === "number" ? clip.broll_opacity : 1;
    if (Math.abs(clamped - current) < 0.01) {
      setBrollDraftOpacityById((prev) => ({ ...prev, [clip.id]: clamped }));
      return;
    }

    await applyBrollTimelineOperations(
      clip.id,
      "opacity",
      [
        {
          op_type: "set_broll_opacity",
          params: { clip: clip.id, opacity: Number(clamped.toFixed(3)) },
          source: "ui",
        },
      ],
      "Updated B-roll opacity."
    );
  }

  async function removeBrollClipById(clipId: string) {
    const clip = getOverlayClipById(clipId);
    if (!clip) return;
    await applyBrollTimelineOperations(
      clip.id,
      "delete",
      [
        {
          op_type: "delete_broll_clip",
          params: { clip: clip.id },
          source: "ui",
        },
      ],
      "Removed B-roll clip from timeline."
    );
  }

  async function commitBrollStart(clip: Clip) {
    const raw = brollDraftStartById[clip.id] ?? formatFixedSec(clip.timeline_start_sec);
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed < 0) {
      setBrollDraftStartById((prev) => ({ ...prev, [clip.id]: formatFixedSec(clip.timeline_start_sec) }));
      return;
    }
    await setBrollClipStart(clip.id, parsed);
  }

  async function commitBrollDuration(clip: Clip) {
    const raw = brollDraftDurationById[clip.id] ?? formatFixedSec(clipTimelineDurationSec(clip));
    const parsed = Number(raw);
    await setBrollClipDuration(clip.id, parsed);
  }

  async function commitBrollOpacity(clip: Clip, nextOpacity: number) {
    await setBrollClipOpacity(clip.id, nextOpacity);
  }

  async function removeBrollClipFromTimeline(clip: Clip) {
    await removeBrollClipById(clip.id);
  }

  function openFeatureDrawer(tab: FeatureTabId) {
    setActiveFeatureTab(tab);
    setFeatureDrawerOpen(true);
  }

  return (
    <div className="appShell">
      {!project ? (
        <>
          <header className="topBar">
            <div className="headerLogos">
              <PlaySquare size={28} className="headerIcon" />
              <div>
                <p className="eyebrow">{BRAND.editorEyebrow}</p>
                <h1>{BRAND.loadingTitle}</h1>
              </div>
            </div>
            <div className="statusPill">
              <span className="statusIndicator" style={{ background: backendStatus === "ok" ? "var(--success)" : "var(--danger)" }}></span>
              Backend: {backendStatus}
            </div>
          </header>

          <section className="controls card">
            <p className="muted" style={{ margin: 0, marginRight: "auto" }}>
              {creatingProject ? "Preparing your workspace..." : "Starting studio..."}
            </p>
            <button
              className="primaryBtn"
              onClick={() => {
                autoCreateAttemptedRef.current = false;
                void createProject(BRAND.defaultProjectName, { silent: true });
              }}
              disabled={creatingProject}
            >
              <Wand2 size={16} />
              {creatingProject ? "Loading..." : "Retry"}
            </button>
          </section>

          {error && <div className="message error">{error}</div>}
          {notice && <div className="message notice">{notice}</div>}
        </>
      ) : (
        <>
          <header className="topBar">
            <div className="headerLogos">
              <PlaySquare size={28} className="headerIcon" />
              <div>
                <p className="eyebrow">{BRAND.editorEyebrow}</p>
                <h1>{project.name || BRAND.editorName}</h1>
              </div>
            </div>
            <div className="statusPill">
              <span className="statusIndicator" style={{ background: backendStatus === "ok" ? "var(--success)" : "var(--danger)" }}></span>
              Backend: {backendStatus}
            </div>
          </header>

          <section className="controls card creatorTopActions">
            <label className="uploadBtn primaryBtn">
              <input
                type="file"
                accept="video/*"
                disabled={uploading}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) {
                    void uploadVideo(file);
                  }
                  event.currentTarget.value = "";
                }}
              />
              <UploadCloud size={16} />
              {uploading ? "Uploading..." : "Upload Video"}
            </label>

            <button className="primaryBtn" onClick={generateTranscript} disabled={!selectedVideoAsset || generatingTranscript}>
              <Wand2 size={16} />
              {generatingTranscript ? `Generating ${formatSeconds(transcriptElapsedSec)}...` : "Generate"}
            </button>
            <button
              className="primaryBtn"
              onClick={() => void exportVideo()}
              disabled={!project || exportingVideo}
            >
              {exportingVideo ? "Exporting..." : "Export"}
            </button>
            <button onClick={() => setFeatureDrawerOpen(true)}>
              Tools
            </button>
            <p className="muted creatorTopMeta">
              <span>{selectedVideoAsset ? selectedVideoAsset.filename : "No video selected"}</span>
              <span>{transcriptLanguageLabel}</span>
              <span>{formatSeconds(project.timeline.duration_sec)}</span>
            </p>
          </section>

          {error && <div className="message error">{error}</div>}
          {notice && <div className="message notice">{notice}</div>}
          <section className="editorMainGrid">
            <section className="panel card editorPreviewDock">
              <div className="workspacePreviewBlock">
                <h2>Video Preview</h2>
                {!previewSource && <p className="muted">Upload a video to preview.</p>}
                {previewSource && (
                  <div
                    className={`previewStage previewStage${previewFrameAspectRatio === "9:16" ? "Portrait" : "Landscape"}`}
                  >
                    <video
                      ref={videoRef}
                      key={previewSource}
                      src={previewSource}
                      controls
                      crossOrigin="anonymous"
                      className="previewVideo"
                      onTimeUpdate={(event) => {
                        const time = event.currentTarget.currentTime;
                        if (!rafPendingRef.current) {
                          rafPendingRef.current = true;
                          requestAnimationFrame(() => {
                            rafPendingRef.current = false;
                            setCurrentTimeSec(time);
                          });
                        }
                      }}
                      onError={(e) => console.error("Video preview error:", e)}
                    />
                    {livePreviewCaption && (
                      <div
                        className="livePreviewCaption"
                        aria-hidden="true"
                        style={{
                          "--caption-safe-bottom": previewFrameAspectRatio === "9:16"
                            ? `clamp(72px, ${Math.max(8.8, Math.min(12.8, livePreviewCaption.marginV / 12))}%, 98px)`
                            : `${Math.max(
                              shouldShowLiveCaptionOverlay ? 84 : 54,
                              Math.min(
                                livePreviewCaption.marginV * (shouldShowLiveCaptionOverlay ? 0.8 : 0.58),
                                shouldShowLiveCaptionOverlay ? 156 : 116
                              )
                            )}px`,
                          "--caption-max-width": shouldShowLiveCaptionOverlay
                            ? previewFrameAspectRatio === "9:16"
                              ? "84%"
                              : "54%"
                            : previewFrameAspectRatio === "16:9"
                              ? "72%"
                              : "88%",
                        } as React.CSSProperties}
                      >
                        <span
                          className="livePreviewCaptionText"
                          style={{
                            color: assColorToCss(livePreviewCaption.color, "#ffffff"),
                            fontFamily: `${livePreviewCaption.fontName.replace("-", " ")}, sans-serif`,
                            WebkitTextStroke: `${Math.min(Math.max(livePreviewCaption.outlineWidth, 1), 3)}px ${assColorToCss(livePreviewCaption.outlineColor, "#000000")}`,
                            textShadow: livePreviewCaption.shadow > 0 ? `0 2px ${Math.min(livePreviewCaption.shadow * 2, 8)}px rgba(0,0,0,0.7)` : "0 1px 2px rgba(0,0,0,0.55)",
                            fontSize: `clamp(0.92rem, ${Math.max(
                              1.2,
                              Math.min(
                                livePreviewCaption.fontSize / 18,
                                previewFrameAspectRatio === "9:16" ? 2.3 : shouldShowLiveCaptionOverlay ? 2.1 : 1.85
                              )
                            )}vw, ${previewFrameAspectRatio === "9:16" ? "1.36rem" : shouldShowLiveCaptionOverlay ? "1.35rem" : "1.55rem"})`,
                            background: "transparent",
                          }}
                        >
                          {livePreviewCaption.text}
                        </span>
                      </div>
                    )}
                    {showExportFrameGuide && previewFrameAspectRatio === "16:9" && exportAspectRatio === "9:16" && (
                      <div className="previewFrameGuide" aria-hidden="true">
                        <div className="previewFrameGuideWindow" />
                      </div>
                    )}
                    {previewRenderBusy && (
                      <div className="previewBusyBadge" aria-live="polite">
                        <div className="previewBusyRow">
                          <span className="previewSpinner" aria-hidden="true" />
                          <span>{previewBusyDetail}</span>
                        </div>
                        <div className="jobProgressBar previewJobProgressBar" aria-hidden="true">
                          <span className="jobProgressFill" style={{ width: `${previewProgress}%` }} />
                        </div>
                      </div>
                    )}
                  </div>
                )}
                <div className="previewMeta">
                  <span>Playhead: {formatSeconds(currentTimeSec)}</span>
                  <span>Preview: {previewStatusText}</span>
                  <span>Editor frame: {previewFrameAspectRatio}</span>
                  {showExportFrameGuide && previewFrameAspectRatio === "16:9" && exportAspectRatio === "9:16" && <span>Portrait export guide on</span>}
                  {previewRenderBusy && previewSource && (
                    <span>Showing last rendered preview while update runs.</span>
                  )}
                  <span>
                    Job: {previewJob ? `${previewJob.status} (${previewProgress}%)` : "not queued"}
                    {previewUpdateQueued ? " · update queued" : ""}
                  </span>
                </div>
                {previewJob?.status === "failed" && (
                  <p className="warning">Preview failed: {previewJob.error ?? "Unknown render error"}</p>
                )}
                <div className="wordActions">
                  <div className="previewAspectToggle" role="group" aria-label="Preview frame aspect">
                    {(["16:9", "9:16"] as const).map((ratio) => (
                      <button
                        key={ratio}
                        className={`previewAspectBtn ${previewFrameAspectRatio === ratio ? "active" : ""}`}
                        onClick={() => setPreviewFrameAspectRatio(ratio)}
                        type="button"
                      >
                        {ratio}
                      </button>
                    ))}
                  </div>
                  <button onClick={() => void queuePreview()} disabled={!project || queueingPreview}>
                    {queueingPreview ? "Queueing..." : "Render Preview"}
                  </button>
                </div>
              </div>
            </section>

            <main className="twoPanel">
              <section className="panel card panelTranscript">
                <h2>Transcript Panel</h2>
                <div className="featureLauncher">
                  {FEATURE_TAB_ITEMS.map(({ id, label, icon: Icon }) => (
                    <button
                      key={id}
                      className={activeFeatureTab === id && featureDrawerOpen ? "active" : ""}
                      onClick={() => openFeatureDrawer(id)}
                    >
                      <Icon size={14} strokeWidth={1.9} aria-hidden="true" />
                      <span>{label}</span>
                    </button>
                  ))}
                </div>
                {!transcript && <p className="muted">Generate transcript from an uploaded video to start text-based editing.</p>}
                {transcript && (
                  <>
                    <p className="muted hint">
                      <strong>Click</strong> word to select & seek &nbsp;·&nbsp;
                      <strong>Shift+click</strong> range &nbsp;·&nbsp;
                      <strong>Drag</strong> to select &nbsp;·&nbsp;
                      <strong>Double-click</strong> to edit text &nbsp;·&nbsp;
                      <strong>Del/⌫</strong> delete &nbsp;·&nbsp;
                      <strong>Ctrl+Z</strong> undo
                    </p>

                    {transcript.is_mock && <p className="warning">Fallback transcript active; install faster-whisper for accurate speech text.</p>}
                    {shouldWarnLowConfidence && (
                      <p className="warning">
                        {lowConfidenceCount} low-confidence word{lowConfidenceCount === 1 ? "" : "s"} (~{(lowConfidenceRatio * 100).toFixed(0)}%).
                      </p>
                    )}
                    {transcriptIssueRegions.length > 0 && (
                      <div className="transcriptRegionSummary">
                        <div className="transcriptRegionSummaryHeader">
                          <strong>Transcript watchlist</strong>
                          <span>{transcriptIssueRegions.length} region{transcriptIssueRegions.length === 1 ? "" : "s"}</span>
                        </div>
                        <div className="transcriptRegionBar">
                          {transcriptIssueRegions.slice(0, 8).map((region, index) => (
                            <button
                              key={`${region.status}-${region.start_sec}-${region.end_sec}-${index}`}
                              type="button"
                              className={`transcriptRegionChip ${region.status}`}
                              onClick={() => seekToTranscriptTime(region.start_sec)}
                              title={`${transcriptRegionLabel(region)} · ${formatSeconds(region.start_sec)} – ${formatSeconds(region.end_sec)}${region.reason ? ` · ${region.reason}` : ""}`}
                            >
                              <span>{transcriptRegionLabel(region)}</span>
                              <span>{formatSeconds(region.start_sec)}–{formatSeconds(region.end_sec)}</span>
                            </button>
                          ))}
                          {transcriptIssueRegions.length > 8 && (
                            <span className="muted transcriptRegionOverflow">+{transcriptIssueRegions.length - 8} more</span>
                          )}
                        </div>
                      </div>
                    )}

                    {/* ── Search bar ────────────────────────────────── */}
                    <div className="searchBar">
                      <svg className="searchIcon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
                      </svg>
                      <input
                        id="transcript-search"
                        type="text"
                        placeholder="Search words... (Ctrl+F)"
                        value={searchQuery}
                        onChange={(e) => { setSearchQuery(e.target.value); setSearchMatchIndex(0); }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") navigateSearch(e.shiftKey ? -1 : 1);
                          if (e.key === "Escape") { setSearchQuery(""); (e.target as HTMLInputElement).blur(); }
                        }}
                      />
                      {searchQuery && (
                        <span className="searchCount">
                          {searchMatchIds.length ? `${searchMatchIndex + 1}/${searchMatchIds.length}` : "0 matches"}
                          <button className="searchNav" onClick={() => navigateSearch(-1)} title="Previous">▲</button>
                          <button className="searchNav" onClick={() => navigateSearch(1)} title="Next">▼</button>
                        </span>
                      )}
                    </div>

                    {/* ── Action toolbar ────────────────────────────── */}
                    <div className="wordActions toolbar">
                      <button onClick={markSelectionDeleted} disabled={!selectedWordIds.size} title="Delete selected words">
                        <Trash2 size={14} strokeWidth={1.9} aria-hidden="true" />
                        <span>Delete</span>
                      </button>
                      <button onClick={restoreSelection} disabled={!selectedWordIds.size} title="Restore selected words">
                        <RotateCcw size={14} strokeWidth={1.9} aria-hidden="true" />
                        <span>Restore</span>
                      </button>
                      <button onClick={restoreAllText} disabled={!deletedWordIds.size} title="Restore all deleted words">
                        <RefreshCw size={14} strokeWidth={1.9} aria-hidden="true" />
                        <span>Restore All</span>
                      </button>
                      <div className="toolbarSep" />
                      <button onClick={removeFillerWords} disabled={!fillerWordIds.size || applyingCut} title="Remove um, uh, like, etc.">
                        <Scissors size={14} strokeWidth={1.9} aria-hidden="true" />
                        <span>
                          Remove Fillers
                          {fillerWordIds.size > 0 && <span className="badge">{fillerWordIds.size}</span>}
                        </span>
                      </button>
                      <div className="toolbarSep" />
                      <button onClick={undo} disabled={!undoStack.current.length} title="Undo (Ctrl+Z)">
                        <Undo2 size={14} strokeWidth={1.9} aria-hidden="true" />
                        <span>Undo</span>
                      </button>
                      <button onClick={redo} disabled={!redoStack.current.length} title="Redo (Ctrl+Shift+Z)">
                        <Redo2 size={14} strokeWidth={1.9} aria-hidden="true" />
                        <span>Redo</span>
                      </button>
                      <div className="toolbarSep" />
                      <button onClick={() => void applyCut(deletedSignature, keptWordIds, { manual: true })} disabled={applyingCut || !transcript}>
                        {applyingCut ? (
                          "Applying..."
                        ) : (
                          <>
                            <ScissorsLineDashed size={14} strokeWidth={1.9} aria-hidden="true" />
                            <span>Apply Cut</span>
                          </>
                        )}
                      </button>
                    </div>

                    {selectedTranscriptRange && (
                      <div className="rangeEditor">
                        <div className="rangeEditorHeader">
                          <strong>Range Edit</strong>
                          <span>
                            {selectedTranscriptRange.wordCount} word{selectedTranscriptRange.wordCount === 1 ? "" : "s"} · {formatSeconds(selectedTranscriptRange.startSec)} – {formatSeconds(selectedTranscriptRange.endSec)}
                          </span>
                        </div>
                        <textarea
                          value={rangeEditText}
                          onChange={(e) => setRangeEditText(e.target.value)}
                          rows={2}
                          placeholder="Rewrite the selected lyric or dialogue range"
                        />
                        <div className="rangeEditorActions">
                          <button
                            type="button"
                            onClick={() => void applyTranscriptRangeUpdate("replace")}
                            disabled={updatingTranscriptRange || !rangeEditText.trim()}
                          >
                            {updatingTranscriptRange ? "Saving..." : "Replace Range"}
                          </button>
                          <button
                            type="button"
                            onClick={() => void applyTranscriptRangeUpdate("blank")}
                            disabled={updatingTranscriptRange}
                          >
                            Mark Instrumental
                          </button>
                          <button
                            type="button"
                            onClick={() => void applyTranscriptRangeUpdate("preserve")}
                            disabled={updatingTranscriptRange}
                          >
                            Preserve Repeat
                          </button>
                        </div>
                      </div>
                    )}

                    {/* ── Interactive word grid ─────────────────────── */}
                    <div
                      className="transcriptBox"
                      ref={transcriptBoxRef}
                      onMouseLeave={() => { isDragging.current = false; }}
                    >
                      {transcriptWordNodes}
                    </div>

                    {/* ── Sentence shortcuts ────────────────────────── */}
                    <details className="shortcutSection">
                      <summary><h3>Sentence Shortcuts ({sentenceBlocks.length})</h3></summary>
                      <div className="shortcutList">{sentenceShortcutNodes}</div>
                    </details>

                    <details className="shortcutSection">
                      <summary><h3>Paragraph Shortcuts ({paragraphBlocks.length})</h3></summary>
                      <div className="shortcutList">{paragraphShortcutNodes}</div>
                    </details>
                  </>
                )}
              </section>

              <button
                type="button"
                className={`featureDrawerBackdrop ${featureDrawerOpen ? "open" : ""}`}
                aria-label="Close tools panel"
                onClick={() => setFeatureDrawerOpen(false)}
              />

              <section
                className={`panel card panelFeatures featureDrawer ${featureDrawerOpen ? "open" : ""}`}
                aria-hidden={!featureDrawerOpen}
              >
                <div className="featureTabsContainer">
                  <div className="featureDrawerHeader">
                    <h3>Feature Tools</h3>
                    <button type="button" onClick={() => setFeatureDrawerOpen(false)}>
                      Close
                    </button>
                  </div>

                  <div className="inspectorContextCard">
                    <p className="inspectorEyebrow">Inspector</p>
                    <h3>{inspectorContext.title}</h3>
                    <p className="muted">{inspectorContext.detail}</p>
                  </div>

                  <div className="inspectorQuickSettings">
                    <label>
                      Video Source
                      <select
                        disabled={!videoAssets.length}
                        value={selectedVideoAsset?.id ?? ""}
                        onChange={(event) => {
                          const nextId = event.target.value;
                          setSelectedAssetId(nextId || null);
                          const selected = videoAssets.find((asset) => asset.id === nextId);
                          if (selected) {
                            setPreviewUrl(resolveMediaPath(selected.storage_path));
                          }
                        }}
                      >
                        {!videoAssets.length && <option value="">No uploaded videos</option>}
                        {videoAssets.map((asset) => (
                          <option key={asset.id} value={asset.id}>
                            {asset.filename}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Transcript Language
                      <select
                        value={transcriptLanguage}
                        disabled={generatingTranscript}
                        onChange={(event) => setTranscriptLanguage(event.target.value)}
                        title="Choose transcript language (Auto uses model detection)"
                      >
                        {TRANSCRIPT_LANGUAGE_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <p className="inspectorStats">
                      {project.timeline.resolution.width}x{project.timeline.resolution.height} · {Math.round(project.timeline.fps)} fps · {formatSeconds(project.timeline.duration_sec)}
                    </p>
                  </div>

                  {(selectedTimelineClipDetails || selectedCaptionBlockDetails) && (
                    <div className="creatorSelectionCard">
                      {selectedTimelineClipDetails && (
                        <>
                          <div className="creatorSelectionHead">
                            <div>
                              <p className="inspectorEyebrow">Clip Inspector</p>
                              <h4>
                                {selectedTimelineClipDetails.lane.label} · {selectedTimelineClipDetails.source?.filename ?? "Timeline clip"}
                              </h4>
                            </div>
                            <button type="button" className="secondaryBtn" onClick={toggleSelectedTimelineClipMute}>
                              {selectedTimelineClipDetails.clip.audio.mute ? "Unmute" : "Mute"}
                            </button>
                          </div>
                          <p className="muted">
                            {formatSeconds(selectedTimelineClipDetails.clip.timeline_start_sec)} start ·{" "}
                            {formatSeconds(selectedTimelineClipDetails.durationSec)} duration
                          </p>
                          <div className="creatorSelectionActions">
                            <button type="button" className="secondaryBtn" onClick={splitSelectedTimelineClip}>
                              Split at Playhead
                            </button>
                            <button type="button" className="secondaryBtn dangerBtn" onClick={deleteSelectedTimelineClip}>
                              Delete Clip
                            </button>
                          </div>
                          <div className="creatorSelectionFields">
                            <label>
                              Speed
                              <div className="chipRow">
                                {[0.75, 1, 1.25, 1.5, 2].map((speed) => (
                                  <button
                                    key={speed}
                                    type="button"
                                    className={`chipBtn ${Math.abs(selectedTimelineClipDetails.clip.speed - speed) < 0.01 ? "active" : ""}`}
                                    onClick={() => setSelectedTimelineClipSpeed(speed)}
                                  >
                                    {speed}x
                                  </button>
                                ))}
                              </div>
                            </label>
                            <label>
                              Volume {(selectedTimelineClipDetails.clip.audio.volume * 100).toFixed(0)}%
                              <div className="chipRow">
                                {[0, 0.5, 1, 1.25, 1.5].map((volume) => (
                                  <button
                                    key={volume}
                                    type="button"
                                    className={`chipBtn ${Math.abs(selectedTimelineClipDetails.clip.audio.volume - volume) < 0.01 ? "active" : ""}`}
                                    onClick={() => setSelectedTimelineClipVolume(volume)}
                                  >
                                    {(volume * 100).toFixed(0)}%
                                  </button>
                                ))}
                              </div>
                            </label>
                          </div>
                        </>
                      )}

                      {selectedCaptionBlockDetails && (
                        <>
                          <div className="creatorSelectionHead">
                            <div>
                              <p className="inspectorEyebrow">Caption Inspector</p>
                              <h4>{trimInlineText(selectedCaptionBlockDetails.overlay.text, 72)}</h4>
                            </div>
                            <button
                              type="button"
                              className="secondaryBtn dangerBtn"
                              onClick={deleteSelectedCaptionBlock}
                            >
                              Delete Caption
                            </button>
                          </div>
                          <p className="muted">
                            Lower-third safe area · {formatSeconds(selectedCaptionBlockDetails.timelineStartSec)} start ·{" "}
                            {formatSeconds(selectedCaptionBlockDetails.durationSec)} duration
                          </p>
                          <div className="creatorSelectionActions">
                            <button
                              type="button"
                              className="secondaryBtn"
                              onClick={() => handleTimelineSeek(selectedCaptionBlockDetails.timelineStartSec)}
                            >
                              Jump to Caption
                            </button>
                          </div>
                        </>
                      )}
                    </div>
                  )}

                  <div className="featureTabs">
                    {FEATURE_TAB_ITEMS.map(({ id, label, icon: Icon }) => (
                      <button
                        key={id}
                        className={`featureTab ${activeFeatureTab === id ? "active" : ""}`}
                        onClick={() => setActiveFeatureTab(id)}
                      >
                        <span className="featureTabInner">
                          <Icon className="featureTabIcon" size={15} strokeWidth={1.9} aria-hidden="true" />
                          <span>{label}</span>
                        </span>
                      </button>
                    ))}
                  </div>

                  <div className="featureTabContent">
                    {/* ── AI Actions Tab ─────────────────────────── */}
                    {activeFeatureTab === "ai_actions" && (
                      <section className="aiPanel aiQuickPanel active">
                        <h3>AI Editing Tools</h3>
                        <p className="muted">Only backend-ready tools are shown here, so every action in this section is wired to a real API route.</p>
                        <div className="actionGrid actionGridWide">
                          {AI_ACTION_ITEMS.map(({ action, label, desc, icon: Icon, primary }) => (
                            <button
                              key={action}
                              className={`actionCard ${primary ? "actionCardPrimary" : ""}`}
                              onClick={() => void runVibeAction(action)}
                              disabled={!selectedVideoAsset || runningAction !== null}
                            >
                              <span className="actionIcon">
                                <Icon size={18} strokeWidth={1.9} aria-hidden="true" />
                              </span>
                              <span className="actionLabel">{runningAction === action ? "Applying..." : label}</span>
                              <span className="actionDesc">{desc}</span>
                            </button>
                          ))}
                        </div>
                      </section>
                    )}

                    {/* ── Captions Tab ───────────────────────────── */}
                    {activeFeatureTab === "captions" && (
                      <section className="aiPanel captionsPanel active">
                        <h3>Caption Styles</h3>
                        <p className="muted">Select a style and apply. 9 curated presets with tuned timing, color, and typography.</p>

                        <div className="captionStyleGrid">
                          {CAPTION_STYLE_PRESETS.map((style) => {
                            const isActive = captionStyle === style.id;
                            // Parse ASS color (&H00BBGGRR) to CSS hex for preview
                            const primaryHex = style.config.primary_color.startsWith("&H")
                              ? (() => {
                                const raw = style.config.primary_color.replace("&H", "").replace("00", "");
                                const b = raw.slice(0, 2);
                                const g = raw.slice(2, 4);
                                const r = raw.slice(4, 6);
                                return `#${r || "FF"}${g || "FF"}${b || "FF"}`;
                              })()
                              : "#ffffff";
                            return (
                              <button
                                key={style.id}
                                className={`captionStyleCard ${isActive ? "active" : ""}`}
                                onClick={() => setCaptionStyle(style.id)}
                                style={{ "--caption-accent": style.color } as React.CSSProperties}
                              >
                                <div className="captionPreviewBox">
                                  <div className={`captionPreviewScene ${style.preview_class}`}>
                                    <span
                                      className="captionPreviewText"
                                      style={{
                                        color: primaryHex,
                                        fontFamily: style.config.font_name.replace("-", " ") + ", sans-serif",
                                        fontSize: `${Math.min(style.config.font_size * 0.55, 14)}px`,
                                        textShadow: style.config.shadow > 0 ? `0 1px ${style.config.shadow}px rgba(0,0,0,0.7)` : "none",
                                        WebkitTextStroke: style.config.outline_width > 0 ? `${Math.min(style.config.outline_width * 0.3, 1)}px rgba(0,0,0,0.5)` : "none",
                                      }}
                                    >
                                      <span>{style.preview_words[0]}</span>
                                      <span className="captionPreviewHighlight">{style.preview_words[1]}</span>
                                      <span>{style.preview_words[2]}</span>
                                    </span>
                                    <span className="captionPreviewPulse" aria-hidden="true" />
                                  </div>
                                </div>
                                <span className="captionStyleName">{style.name}</span>
                                <span className="captionStyleDesc">{style.desc}</span>
                                {isActive && (
                                  <span className="captionActiveCheck">
                                    <Check size={12} strokeWidth={2.4} aria-hidden="true" />
                                  </span>
                                )}
                              </button>
                            );
                          })}
                        </div>

                        {captionResultInfo && (
                          <div className="captionResultBadge">
                            <span className="captionResultIcon">
                              <Check size={14} strokeWidth={2.4} aria-hidden="true" />
                            </span>
                            <span>{captionResultInfo}</span>
                          </div>
                        )}

                        <div className="captionApplyRow">
                          <button
                            className="primaryBtn captionApplyBtn"
                            onClick={() => { setCaptionResultInfo(null); void runVibeAction("add_subtitles"); }}
                            disabled={!selectedVideoAsset || runningAction !== null || removingCaptions}
                          >
                            {runningAction === "add_subtitles" ? (
                              <>
                                <span className="captionSpinner" />
                                Generating...
                              </>
                            ) : (
                              <>
                                <Captions size={16} strokeWidth={1.9} aria-hidden="true" />
                                <span>{`Apply "${selectedCaptionStyleName}" Captions`}</span>
                              </>
                            )}
                          </button>
                          <button
                            className="secondaryBtn captionRemoveBtn"
                            onClick={() => void removeCaptions()}
                            disabled={!selectedVideoAsset || runningAction !== null || removingCaptions}
                            title="Remove all captions from the video"
                          >
                            {removingCaptions ? (
                              "Removing..."
                            ) : (
                              <>
                                <Trash2 size={15} strokeWidth={1.9} aria-hidden="true" />
                                <span>Remove</span>
                              </>
                            )}
                          </button>
                        </div>
                      </section>
                    )}

                    {/* ── Export Tab ──────────────────────────────── */}
                    {activeFeatureTab === "export" && (
                      <section className="aiPanel exportPanel active">
                        <h3>Export Video</h3>
                        <p className="muted">Render your final video. Choose format, resolution, and quality.</p>
                        <div className="exportSettings">
                          <div className="exportField">
                            <label className="exportLabel">Aspect Ratio</label>
                            <div className="exportOptions">
                              {(["16:9", "9:16"] as const).map((ratio) => (
                                <button
                                  key={ratio}
                                  className={`exportOption ${exportAspectRatio === ratio ? "active" : ""}`}
                                  onClick={() => {
                                    setExportAspectRatio(ratio);
                                    setPreviewFrameAspectRatio(ratio);
                                    void queuePreview(false, { aspectRatio: ratio });
                                  }}
                                >
                                  {ratio}
                                </button>
                              ))}
                            </div>
                          </div>
                          <div className="exportField">
                            <label className="exportLabel">Format</label>
                            <div className="exportOptions">
                              {(["mp4", "mov", "webm"] as const).map((fmt) => (
                                <button
                                  key={fmt}
                                  className={`exportOption ${exportFormat === fmt ? "active" : ""}`}
                                  onClick={() => setExportFormat(fmt)}
                                >
                                  {fmt.toUpperCase()}
                                </button>
                              ))}
                            </div>
                          </div>
                          <div className="exportField">
                            <label className="exportLabel">Resolution</label>
                            <div className="exportOptions">
                              {(["720p", "1080p", "4k"] as const).map((res) => (
                                <button
                                  key={res}
                                  className={`exportOption ${exportResolution === res ? "active" : ""}`}
                                  onClick={() => setExportResolution(res)}
                                >
                                  {res}
                                </button>
                              ))}
                            </div>
                          </div>
                          <div className="exportField">
                            <label className="exportLabel">Frame Rate</label>
                            <div className="exportOptions">
                              {([24, 30, 60] as const).map((fps) => (
                                <button
                                  key={fps}
                                  className={`exportOption ${exportFps === fps ? "active" : ""}`}
                                  onClick={() => {
                                    setExportFps(fps);
                                    void queuePreview(false, { fps });
                                  }}
                                >
                                  {fps} fps
                                </button>
                              ))}
                            </div>
                          </div>
                          <div className="exportField">
                            <label className="exportLabel">Quality</label>
                            <div className="exportOptions">
                              {(["low", "medium", "high", "max"] as const).map((q) => (
                                <button
                                  key={q}
                                  className={`exportOption ${exportQuality === q ? "active" : ""}`}
                                  onClick={() => setExportQuality(q)}
                                >
                                  {q.charAt(0).toUpperCase() + q.slice(1)}
                                </button>
                              ))}
                            </div>
                          </div>
                        </div>
                        {exportJob && (
                          <div className="exportJobStatus">
                            <div className="exportJobTop">
                              <span className="exportJobLabel">Export Job:</span>
                              <span className={`exportJobBadge ${exportJob.status}`}>{exportJob.status}</span>
                              <span className="exportJobProgress">{exportProgress}%</span>
                            </div>
                            <div className="jobProgressBar" aria-hidden="true">
                              <span className="jobProgressFill" style={{ width: `${exportProgress}%` }} />
                            </div>
                            <span className="exportJobMessage">{exportStatusMessage}</span>
                          </div>
                        )}
                        <div className="exportGuideRow">
                          <button
                            className={`exportGuideBtn ${showExportFrameGuide ? "active" : ""}`}
                            onClick={() => setShowExportFrameGuide((prev) => !prev)}
                            type="button"
                          >
                            {showExportFrameGuide ? "Hide export guide" : "Show export guide"}
                          </button>
                          <span className="muted exportGuideHint">
                            {previewFrameAspectRatio === "16:9"
                              ? `Overlay the final ${exportAspectRatio} frame on the wide editor preview.`
                              : `Switch preview back to 16:9 to see the ${exportAspectRatio} export guide.`}
                          </span>
                        </div>
                        <div className="exportApplyRow">
                          <button
                            className="primaryBtn exportApplyBtn"
                            onClick={() => void exportVideo()}
                            disabled={!project || exportingVideo}
                          >
                            {exportingVideo ? (
                              "Exporting..."
                            ) : (
                              <>
                                <Download size={16} strokeWidth={1.9} aria-hidden="true" />
                                <span>{`Export ${exportAspectRatio} ${exportResolution} ${exportFormat.toUpperCase()}`}</span>
                              </>
                            )}
                          </button>
                        </div>
                      </section>
                    )}

                    {activeFeatureTab === "broll_studio" && (
                      <section className="aiPanel brollPanel brollStudio active">
                        <h3>B-roll Studio</h3>
                        <p className="muted">Generate visual cutaway suggestions from transcript chunks. Transcript edits stay unchanged.</p>
                        <div className="wordActions">
                          <button
                            className="primaryBtn"
                            onClick={() => void autoApplyBroll()}
                            disabled={!project || !transcript || autoApplyingBroll || loadingBrollSlots || suggestingBroll || syncingBroll || undoingBroll}
                            title="Generate slots, auto-pick confident candidates, and sync to timeline in one step."
                          >
                            {autoApplyingBroll ? (
                              "Auto-applying..."
                            ) : (
                              <>
                                <Wand2 size={16} strokeWidth={1.9} aria-hidden="true" />
                                <span>Auto B-roll (1-click)</span>
                              </>
                            )}
                          </button>
                          <button
                            onClick={() => void suggestBroll()}
                            disabled={!project || !transcript || suggestingBroll || loadingBrollSlots || autoApplyingBroll}
                          >
                            {suggestingBroll ? (
                              "Suggesting..."
                            ) : (
                              <>
                                <Sparkles size={16} strokeWidth={1.9} aria-hidden="true" />
                                <span>Suggest B-roll</span>
                              </>
                            )}
                          </button>
                          <button
                            onClick={() => project && transcript && void refreshBrollSlots(project.id, transcript.id)}
                            disabled={!project || !transcript || loadingBrollSlots}
                          >
                            {loadingBrollSlots ? (
                              "Refreshing..."
                            ) : (
                              <>
                                <RefreshCw size={16} strokeWidth={1.9} aria-hidden="true" />
                                <span>Refresh</span>
                              </>
                            )}
                          </button>
                          <button
                            onClick={() => void syncBrollToTimeline()}
                            disabled={!project || syncingBroll || autoApplyingBroll || undoingBroll}
                            title={
                              brollSyncMode === "replace"
                                ? "Replaces existing overlay B-roll clips with the currently chosen slots."
                                : "Keeps existing overlay B-roll clips and adds the currently chosen slots on top."
                            }
                          >
                            {syncingBroll ? (
                              "Syncing..."
                            ) : (
                              <>
                                <Clapperboard size={16} strokeWidth={1.9} aria-hidden="true" />
                                <span>Sync to Timeline</span>
                              </>
                            )}
                          </button>
                          <button
                            onClick={() => void undoBrollLayer()}
                            disabled={!project || undoingBroll || syncingBroll || autoApplyingBroll}
                            title="Undo only the latest B-roll transaction while preserving other timeline layers."
                          >
                            {undoingBroll ? (
                              "Undoing..."
                            ) : (
                              <>
                                <RotateCcw size={16} strokeWidth={1.9} aria-hidden="true" />
                                <span>Undo B-roll Layer</span>
                              </>
                            )}
                          </button>
                        </div>
                        <div className="brollControlsRow">
                          <div className="brollIntensity">
                            <span className="brollSyncLabel">Auto mode:</span>
                            <select
                              value={brollAutoMode}
                              onChange={(event) => setBrollAutoMode(event.target.value as BrollAutoMode)}
                              disabled={autoApplyingBroll || suggestingBroll}
                            >
                              <option value="fast">Fast (&lt;1 min target)</option>
                              <option value="balanced">Balanced</option>
                              <option value="creative">Creative (quality)</option>
                            </select>
                          </div>
                          <div className="brollSyncMode">
                            <span className="brollSyncLabel">Sync mode:</span>
                            <button
                              type="button"
                              className={brollSyncMode === "replace" ? "brollSyncOption active" : "brollSyncOption"}
                              onClick={() => setBrollSyncMode("replace")}
                              disabled={syncingBroll}
                            >
                              Replace
                            </button>
                            <button
                              type="button"
                              className={brollSyncMode === "append" ? "brollSyncOption active" : "brollSyncOption"}
                              onClick={() => setBrollSyncMode("append")}
                              disabled={syncingBroll}
                            >
                              Append
                            </button>
                          </div>
                          <div className="brollIntensity">
                            <span className="brollSyncLabel">B-roll intensity:</span>
                            <select
                              value={brollIntensity}
                              onChange={(event) => setBrollIntensity(event.target.value as BrollIntensity)}
                              disabled={autoApplyingBroll || suggestingBroll}
                            >
                              <option value="low">Light</option>
                              <option value="medium">Balanced</option>
                              <option value="high">Rich</option>
                            </select>
                          </div>
                        </div>
                        <p className="muted brollMeta">
                          Mode: {brollAutoMode} · Slots: {brollSlots.length} · Ready: {brollSlots.filter((slot) => slot.review_status === "ready" || slot.review_status === "approved").length} ·
                          Needs review: {brollSlots.filter((slot) => (slot.review_status ?? "needs_review") === "needs_review").length} · Timeline overlay clips: {overlayClips.length}
                        </p>
                        <div className="brollSlots">
                          {!brollSlots.length && (
                            <p className="muted">
                              {loadingBrollSlots || suggestingBroll || autoApplyingBroll
                                ? "Looking for B-roll moments in your transcript..."
                                : !transcript
                                  ? "No B-roll slots yet. Generate a transcript, then click Suggest or Auto B-roll."
                                  : "No B-roll slots yet. Click Suggest B-roll or Auto B-roll to generate them."}
                            </p>
                          )}
                          {[...brollSlots]
                            .sort((left, right) => {
                              const leftReview = (left.review_status ?? "needs_review") === "needs_review" ? 0 : 1;
                              const rightReview = (right.review_status ?? "needs_review") === "needs_review" ? 0 : 1;
                              if (leftReview !== rightReview) return leftReview - rightReview;
                              return left.start_sec - right.start_sec;
                            })
                            .map((slot, slotIndex) => {
                            const chosenCandidate = slot.candidates.find((candidate) => candidate.id === slot.chosen_candidate_id) ?? null;
                            const primaryCandidate = chosenCandidate ?? slot.candidates[0] ?? null;
                            const primaryReason = primaryCandidate?.reason;
                            const anchorText = slot.anchor_word_ids
                              .map((wordId) => transcriptWordsById.get(wordId)?.text?.trim() ?? "")
                              .filter((word) => !!word)
                              .join(" ")
                              .trim();
                            const slotContextText = anchorText || slot.concept_text || "general scene";
                            const slotMeta = [
                              readReasonText(primaryReason, "section_label"),
                              readReasonText(primaryReason, "shot_style")
                                ? `${humanizeBrollMeta(readReasonText(primaryReason, "shot_style") ?? "")} shot`
                                : null,
                              readReasonText(primaryReason, "source_strategy")
                                ? humanizeBrollMeta(readReasonText(primaryReason, "source_strategy") ?? "")
                                : null,
                              readReasonNumber(primaryReason, "planner_confidence") !== null
                                ? `plan ${(readReasonNumber(primaryReason, "planner_confidence")! * 100).toFixed(0)}%`
                                : null,
                            ]
                              .filter((item): item is string => !!item)
                              .join(" · ");
                            const defaultVisibleCount = slotIndex === 0 ? 5 : 3;
                            const expanded = !!expandedBrollSlots[slot.id];
                            const visibleCandidates = expanded ? slot.candidates : slot.candidates.slice(0, defaultVisibleCount);
                            return (
                              <article
                                key={slot.id}
                                className={`brollSlotCard ${slot.status} ${slot.locked ? "locked" : ""} ${slot.chosen_candidate_id ? "hasChosen" : ""
                                  }`}
                              >
                                <div className="brollSlotHead">
                                  <span className="brollTime">{formatSeconds(slot.start_sec)}-{formatSeconds(slot.end_sec)}</span>
                                  <span className={`brollStatus ${slot.review_status ?? "needs_review"}`}>{reviewStatusLabel(slot.review_status ?? "needs_review")}</span>
                                </div>
                                <p className="brollConcept">{slot.concept_text || "general scene"}</p>
                                {!!slotMeta && <p className="muted brollPlanMeta">{slotMeta}</p>}
                                <p className="muted brollReviewMeta">
                                  Intent: {slot.visual_intent ?? "support"}{slot.review_summary ? ` · ${slot.review_summary}` : ""}
                                </p>
                                {!!(slot.weak_reason_codes?.length ?? 0) && (
                                  <p className="brollWeakReasons">
                                    {(slot.weak_reason_codes ?? []).map((code) => reasonCodeLabel(code)).join(" · ")}
                                  </p>
                                )}
                                {!!anchorText && (
                                  <p className="brollAnchorText">"{slotContextText}"</p>
                                )}
                                {chosenCandidate && (
                                  <p className="brollChosen">
                                    Chosen: {chosenCandidate.source_label ?? chosenCandidate.asset_id ?? "candidate"}
                                  </p>
                                )}
                                <div className="brollCandidates">
                                  {visibleCandidates.map((candidate) => {
                                    const busyChoose = brollActionKey === `choose:${slot.id}:${candidate.id}`;
                                    const isChosen = slot.chosen_candidate_id === candidate.id;
                                    const confidence = typeof candidate.confidence === "number" ? candidate.confidence : null;
                                    const confidencePercent = confidence !== null ? `${(confidence * 100).toFixed(0)}%` : null;
                                    const confidenceTier = confidenceLabel(confidence);
                                    const candidateReason = candidate.reason ?? {};
                                    const breakdownChips = [
                                      ...candidateBreakdownChips(candidate.score_breakdown ?? {}),
                                      ...(candidate.weak_reason_codes ?? []).map((code) => reasonCodeLabel(code)),
                                    ].slice(0, 4);
                                    const scoreLabel = `match ${(candidate.score * 100).toFixed(0)}%`;
                                    const shotStyle = readReasonText(candidateReason, "shot_style") ?? readReasonText(candidateReason, "shot_type");
                                    const queryMode = readReasonText(candidateReason, "query_mode");
                                    const stockability = readReasonText(candidateReason, "stockability");
                                    const metaLine = [
                                      candidateSourceTag(candidate.source_type),
                                      candidate.visual_intent ? humanizeBrollMeta(candidate.visual_intent) : null,
                                      shotStyle ? `${humanizeBrollMeta(shotStyle)} shot` : null,
                                      queryMode ? humanizeBrollMeta(queryMode) : null,
                                      stockability,
                                    ]
                                      .filter((item): item is string => !!item)
                                      .join(" · ");
                                    const previewParams = resolveBrollCandidatePreviewParams(candidate, mediaById);
                                    return (
                                      <BrollCandidateCard
                                        key={candidate.id}
                                        label={candidate.source_label ?? candidate.asset_id ?? "asset"}
                                        sourceTag={candidateSourceTag(candidate.source_type)}
                                        metaLine={metaLine || null}
                                        confidencePercent={confidencePercent}
                                        confidenceTier={confidenceTier}
                                        scoreLabel={scoreLabel}
                                        breakdownChips={breakdownChips}
                                        previewUrl={previewParams?.url ?? null}
                                        previewType={previewParams?.type ?? "video"}
                                        chosen={isChosen}
                                        busy={busyChoose}
                                        locked={slot.locked}
                                        onClick={() => void chooseBroll(slot.id, candidate.id)}
                                      />
                                    );
                                  })}
                                </div>
                                {slot.candidates.length > defaultVisibleCount && (
                                  <button
                                    type="button"
                                    className="brollToggleCandidates"
                                    onClick={() =>
                                      setExpandedBrollSlots((prev) => ({ ...prev, [slot.id]: !expanded }))
                                    }
                                    disabled={!!brollActionKey}
                                  >
                                    {expanded ? "Show less" : `Show all (${slot.candidates.length})`}
                                  </button>
                                )}
                                <div className="brollSlotActions">
                                  <button
                                    type="button"
                                    onClick={() => void rerollBroll(slot.id)}
                                    disabled={!!brollActionKey || slot.locked}
                                  >
                                    {brollActionKey === `reroll:${slot.id}` ? "Rerolling..." : "Re-roll"}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => void rejectBroll(slot.id)}
                                    disabled={!!brollActionKey || slot.locked}
                                  >
                                    {brollActionKey === `reject:${slot.id}` ? "Rejecting..." : "Reject"}
                                  </button>
                                </div>
                              </article>
                            );
                          })}
                        </div>

                        <div className="brollTimelineEditor">
                          <h4>Timeline B-roll Edits</h4>
                          {!sortedOverlayClips.length && (
                            <p className="muted">
                              {syncingBroll
                                ? "Applying B-roll clips to the timeline..."
                                : "No B-roll clips in timeline yet. Choose slots, then sync to timeline."}
                            </p>
                          )}
                          {sortedOverlayClips.map((clip, index) => {
                            const clipBusy = isBrollTimelineClipBusy(clip.id);
                            const clipDuration = clipTimelineDurationSec(clip);
                            const clipOpacity = typeof clip.broll_opacity === "number" ? clip.broll_opacity : 1;
                            const draftStart = brollDraftStartById[clip.id] ?? formatFixedSec(clip.timeline_start_sec);
                            const draftDuration = brollDraftDurationById[clip.id] ?? formatFixedSec(clipDuration);
                            const draftOpacity = brollDraftOpacityById[clip.id] ?? clipOpacity;
                            const source = mediaById.get(clip.asset_id);
                            return (
                              <article key={clip.id} className="brollTimelineCard">
                                <div className="brollTimelineHead">
                                  <span>B{index + 1}</span>
                                  <span>{source?.filename ?? clip.asset_id}</span>
                                </div>
                                <div className="brollTimelineFields">
                                  <label>
                                    Start
                                    <input
                                      type="number"
                                      min={0}
                                      step={0.05}
                                      value={draftStart}
                                      disabled={clipBusy}
                                      onChange={(event) =>
                                        setBrollDraftStartById((prev) => ({ ...prev, [clip.id]: event.target.value }))
                                      }
                                      onBlur={() => void commitBrollStart(clip)}
                                      onKeyDown={(event) => {
                                        if (event.key === "Enter") {
                                          event.currentTarget.blur();
                                        }
                                      }}
                                    />
                                  </label>
                                  <label>
                                    Duration
                                    <input
                                      type="number"
                                      min={0.1}
                                      step={0.05}
                                      value={draftDuration}
                                      disabled={clipBusy}
                                      onChange={(event) =>
                                        setBrollDraftDurationById((prev) => ({ ...prev, [clip.id]: event.target.value }))
                                      }
                                      onBlur={() => void commitBrollDuration(clip)}
                                      onKeyDown={(event) => {
                                        if (event.key === "Enter") {
                                          event.currentTarget.blur();
                                        }
                                      }}
                                    />
                                  </label>
                                </div>
                                <label className="brollOpacityField">
                                  Opacity {(draftOpacity * 100).toFixed(0)}%
                                  <input
                                    type="range"
                                    min={0}
                                    max={1}
                                    step={0.01}
                                    value={draftOpacity}
                                    disabled={clipBusy}
                                    onChange={(event) =>
                                      setBrollDraftOpacityById((prev) => ({
                                        ...prev,
                                        [clip.id]: Number(event.target.value),
                                      }))
                                    }
                                    onMouseUp={(event) =>
                                      void commitBrollOpacity(clip, Number(event.currentTarget.value))
                                    }
                                    onTouchEnd={(event) =>
                                      void commitBrollOpacity(clip, Number(event.currentTarget.value))
                                    }
                                    onBlur={(event) =>
                                      void commitBrollOpacity(clip, Number(event.currentTarget.value))
                                    }
                                  />
                                </label>
                                <div className="brollTimelineActions">
                                  <button
                                    type="button"
                                    disabled={clipBusy}
                                    onClick={() => {
                                      if (videoRef.current) {
                                        videoRef.current.currentTime = clip.timeline_start_sec;
                                      }
                                      setCurrentTimeSec(clip.timeline_start_sec);
                                    }}
                                  >
                                    Jump
                                  </button>
                                  <button
                                    type="button"
                                    disabled={clipBusy || !!brollActionKey}
                                    onClick={() => void rerollBrollFromTimelineClip(clip.id)}
                                  >
                                    {brollActionKey === `reroll-clip:${clip.id}` ? "Rerolling..." : "Re-roll"}
                                  </button>
                                  <button
                                    type="button"
                                    disabled={clipBusy}
                                    onClick={() => void removeBrollClipFromTimeline(clip)}
                                  >
                                    {clipBusy ? "Working..." : "Remove"}
                                  </button>
                                </div>
                              </article>
                            );
                          })}
                        </div>
                      </section>
                    )}
                  </div>
                </div>
              </section>
            </main>
          </section>

          {/* ── Visual Timeline ─────────────────────────── */}
          <Timeline
            words={timelineAssistWords}
            timelineLanes={timelineLanes}
            assetUrlById={assetUrlById}
            assetDurationById={assetDurationById}
            captionBlocks={captionBlocks}
            durationSec={transcript?.duration_sec || project.timeline.duration_sec}
            currentTimeSec={currentTimeSec}
            deletedWordIds={deletedWordIds}
            selectedWordIds={selectedWordIds}
            activeWordId={activeWordId}
            waveformPeaks={waveformPeaks}
            overlayClips={overlayClips}
            selectedLaneClipId={selectedTimelineClip?.clipId ?? null}
            selectedBrollClipId={selectedBrollClipId}
            selectedCaptionId={selectedCaptionBlock?.overlayId ?? null}
            lockedLaneIds={lockedLaneIds}
            onSeek={handleTimelineSeek}
            onSelectWord={handleTimelineSelectWord}
            onSelectWordsInRange={handleTimelineSelectWordsInRange}
            onDeleteSelected={markSelectionDeleted}
            onRestoreSelected={restoreSelection}
            onMoveLaneClip={handleTimelineMoveLaneClip}
            onTrimLaneClip={handleTimelineTrimLaneClip}
            onToggleLaneMute={handleTimelineToggleLaneMute}
            onToggleLaneSolo={handleTimelineToggleLaneSolo}
            onToggleLaneLock={handleTimelineToggleLaneLock}
            onMoveBrollClip={handleTimelineMoveBrollClip}
            onTrimBrollClip={handleTimelineTrimBrollClip}
            onSetBrollOpacity={handleTimelineSetBrollOpacity}
            onDeleteBrollClip={handleTimelineDeleteBrollClip}
            onRerollBrollClip={handleTimelineRerollBrollClip}
            onSelectLaneClip={handleTimelineSelectLaneClip}
            onSelectBrollClip={handleTimelineSelectBrollClip}
            onSelectCaptionBlock={handleTimelineSelectCaptionBlock}
            onMoveCaptionBlock={handleTimelineMoveCaptionBlock}
            onTrimCaptionBlock={handleTimelineTrimCaptionBlock}
            brollEditBusy={!!brollTimelineActionKey}
          />

        </>
      )}
    </div>
  );
}

export default App;
