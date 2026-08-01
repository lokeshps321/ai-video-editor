import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Captions,
  ChevronDown,
  ChevronUp,
  Check,
  Clapperboard,
  Download,
  Film,
  Globe,
  Redo2,
  RefreshCw,
  RotateCcw,
  Scissors,
  ScissorsLineDashed,
  Sparkles,
  Trash2,
  Undo2,
  Wand2,
  X,
} from "lucide-react";
import { api } from "./lib/api";
import {
  consumePendingUploadFile,
  filenameToProjectName,
  hasPendingUploadFile,
  peekPendingUploadName,
} from "./lib/pendingUpload";
import type {
  BrollCandidate,
  BrollConfig,
  BrollSlot,
  Clip,
  ExportAspectRatio,
  Job,
  MediaAsset,
  Project,
  Timeline as ProjectTimeline,
  TimelineOperation,
  Transcript,
  TranscriptGenerateResponse,
  TranscriptMode,
  TranscriptRegion,
  TranscriptSpeed,
  TranscriptWord,
  VibeAction,
} from "./types";
import Timeline, {
  type TimelineCaptionBlock,
  type TimelineCaptionSelection,
  type TimelineDropTarget,
  type TimelineLane,
  type TimelineLaneClipSelection,
} from "./components/Timeline";
import {
  ExportCompletionCard,
  QuickEditSummaryCard,
} from "./components/CompletionCards";
import { EditorHeader } from "./components/EditorHeader";
import { EditorTopActions } from "./components/EditorTopActions";
import { KeyboardShortcutsModal } from "./components/KeyboardShortcutsModal";
import { PreviewDock } from "./components/PreviewDock";
import { ProjectReopenPanel } from "./components/ProjectReopenPanel";
import { TranscriptWordButton } from "./components/transcript/TranscriptWordButton";
import "./components/features/AiActionsPanel.css";
import "./components/features/BrollStudioPanel.css";
import "./components/features/CaptionStylesPanel.css";
import "./components/features/ExportPanel.css";
import "./components/features/FeatureDrawer.css";
import "./components/transcript/TranscriptPanel.css";
import {
  readLockedLaneIds,
  selectTranscriptWordIdsInRange,
  writeLockedLaneIds,
} from "./utils/timelineSelection";
import { BrollCandidateCard } from "./components/BrollCandidateCard";
import { BrollTrustSummary } from "./components/features/BrollTrustSummary";
import { TranscriptQualityPanel } from "./components/features/TranscriptQualityPanel";
import { BRAND } from "./config/brand";
import { createTimelineClock } from "./timeline/clock";
import {
  compileTimelineKeyboardCommand,
  decideTimelineArrowHandling,
  resolveCanonicalFps,
} from "./timeline/integration";
import { frameToSeconds, secondsToFrame } from "./timeline/timebase";
import {
  createTimelineMutationCoordinator,
  TimelineMutationProjectChangedError,
} from "./timeline/mutationCoordinator";
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
  TRANSCRIPT_MODE_OPTIONS,
  TRANSCRIPT_SPEED_OPTIONS,
  type FeatureTabId,
} from "./config/editor";

const TIMELINE_CORE_V2 = import.meta.env.VITE_TIMELINE_CORE_V2 !== "false";

type TextBlock = {
  id: string;
  wordIds: string[];
  text: string;
  startSec: number;
  endSec: number;
};

type BrollIntensity = "low" | "medium" | "high";
type BrollAutoMode = "fast" | "balanced" | "creative";
type BrollSuggestionSource = "full" | "selection";

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

type QuickEditSummary = {
  cutDetails: string | null;
  captionDetails: string | null;
  removedDurationSec: number | null;
  removedWordCount: number | null;
  captionBlockCount: number | null;
  captionsAdded: boolean;
  finalDurationSec: number;
  nextStep: string;
};

type ExportSettingsSnapshot = {
  format: "mp4" | "mov" | "webm";
  aspectRatio: ExportAspectRatio;
  resolution: "720p" | "1080p" | "4k";
  fps: 24 | 30 | 60;
  quality: "low" | "medium" | "high" | "max";
  autoFrame: boolean;
};

type ExportCompletionSummary = ExportSettingsSnapshot & {
  jobId: string;
  filename: string;
  outputPath: string | null;
  downloadError: string | null;
};

const WORKSPACE_READY_NOTICE = "Workspace ready — upload a video to start.";

type InspectorTimelineSelection = TimelineLaneClipSelection;
type InspectorCaptionSelection = TimelineCaptionSelection;

function formatSeconds(value: number): string {
  if (!Number.isFinite(value)) return "0:00";
  const mins = Math.floor(value / 60);
  const secs = Math.floor(value % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function transcriptStageCeiling(
  stage: string | null | undefined,
  status: Job["status"] | undefined,
): number {
  if (status === "queued") return 10;
  switch ((stage ?? "").trim()) {
    case "prepare":
      return 14;
    case "prepare_audio":
      return 30;
    case "recognize":
      return 76;
    case "lyrics":
      return 84;
    case "refine":
      return 92;
    case "weak_retry":
      return 95;
    case "timeline":
      return 99;
    case "reuse":
      return 92;
    default:
      return 80;
  }
}

function transcriptStageRate(
  stage: string | null | undefined,
  status: Job["status"] | undefined,
): number {
  if (status === "queued") return 1.5;
  switch ((stage ?? "").trim()) {
    case "prepare":
      return 2.2;
    case "prepare_audio":
      return 1.6;
    case "recognize":
      return 0.6;
    case "lyrics":
      return 1.0;
    case "refine":
      return 0.9;
    case "weak_retry":
      return 0.7;
    case "timeline":
      return 0.5;
    case "reuse":
      return 1.4;
    default:
      return 0.5;
  }
}

function formatPreciseSeconds(value: number): string {
  if (!Number.isFinite(value)) return "0.00s";
  if (Math.abs(value) < 60) return `${value.toFixed(2)}s`;
  const mins = Math.floor(value / 60);
  const secs = (value - mins * 60).toFixed(2).padStart(5, "0");
  return `${mins}:${secs}`;
}

function formatFixedSec(value: number): string {
  if (!Number.isFinite(value)) return "0.00";
  return value.toFixed(2);
}

function estimateTranscriptRuntimeLabel(
  mode: TranscriptMode,
  durationSec: number | null | undefined,
  speed: TranscriptSpeed = "normal",
): string {
  const duration =
    typeof durationSec === "number" && Number.isFinite(durationSec)
      ? durationSec
      : null;
  if (speed === "fast") {
    if (duration === null || duration <= 0) {
      return "about 30-90 sec for most clips";
    }
    if (duration <= 45) return "usually under 45 sec";
    if (duration <= 180) return "about 45-90 sec";
    return "about 1-3 min";
  }
  if (duration === null || duration <= 0) {
    return mode === "song"
      ? "about 2-4 min for lyric-heavy clips"
      : "about 1-3 min for most clips";
  }
  if (duration <= 45) {
    return mode === "song" ? "about 1-2 min" : "usually under 1 min";
  }
  if (duration <= 180) {
    return mode === "song" ? "about 2-4 min" : "about 1-3 min";
  }
  return mode === "song" ? "about 4-8+ min" : "about 3-6 min";
}

function transcriptModeDetail(
  mode: TranscriptMode,
  speed: TranscriptSpeed = "normal",
): string {
  if (speed === "fast") {
    return "Fast skips voice isolation for quicker transcripts.";
  }
  if (mode === "song")
    return "Song mode spends extra time on lyric-safe passes and uses voice isolation when needed.";
  if (mode === "speech")
    return "Speech mode is the fastest reliable choice for talking clips.";
  return "Auto may add time when it needs to detect speech vs. song; Normal uses voice isolation for songs when needed.";
}

function estimateQuickEditRuntimeLabel(
  mode: TranscriptMode,
  durationSec: number | null | undefined,
  hasTranscript: boolean,
  speed: TranscriptSpeed = "normal",
): string {
  if (hasTranscript) return "usually under 1 min because transcript is ready";
  return `${estimateTranscriptRuntimeLabel(mode, durationSec, speed)} + caption render`;
}

function estimateExportRuntimeLabel(
  settings: ExportSettingsSnapshot,
  durationSec: number | null | undefined,
): string {
  const duration =
    typeof durationSec === "number" && Number.isFinite(durationSec)
      ? durationSec
      : 0;
  const qualityBoost =
    settings.quality === "max" || settings.resolution === "4k"
      ? 1.8
      : settings.resolution === "1080p"
        ? 1.25
        : 1;
  const fpsBoost = settings.fps === 60 ? 1.25 : 1;
  const estimatedSec = Math.max(30, duration * qualityBoost * fpsBoost);
  if (estimatedSec < 75) return "about 1 min";
  if (estimatedSec < 210) return "about 2-3 min";
  if (estimatedSec < 420) return "about 4-7 min";
  return "7+ min for this quality";
}

function parseDetailFloat(
  details: string | null | undefined,
  pattern: RegExp,
): number | null {
  const match = details?.match(pattern);
  if (!match?.[1]) return null;
  const value = Number(match[1]);
  return Number.isFinite(value) ? value : null;
}

function parseDetailInt(
  details: string | null | undefined,
  pattern: RegExp,
): number | null {
  const match = details?.match(pattern);
  if (!match?.[1]) return null;
  const value = Number.parseInt(match[1], 10);
  return Number.isFinite(value) ? value : null;
}

function countCaptionOverlays(timeline: ProjectTimeline): number {
  return timeline.tracks.reduce((total, track) => {
    if (track.kind !== "video") return total;
    return (
      total +
      track.clips.reduce(
        (clipTotal, clip) => clipTotal + (clip.text_overlays?.length ?? 0),
        0,
      )
    );
  }, 0);
}

function buildQuickEditSummary(
  cutDetails: string | null,
  captionDetails: string | null,
  finalTimeline: ProjectTimeline,
): QuickEditSummary {
  const removedDurationSec = parseDetailFloat(
    cutDetails,
    /Removed\s+([\d.]+)s/i,
  );
  const removedWordCount = parseDetailInt(
    cutDetails,
    /Removed\s+(\d+)\s+filler word/i,
  );
  const parsedCaptionBlocks = parseDetailInt(
    captionDetails,
    /Added\s+(\d+)\s+caption overlay/i,
  );
  const timelineCaptionBlocks = countCaptionOverlays(finalTimeline);
  const captionsSkipped =
    captionDetails?.toLowerCase().startsWith("captions skipped") ?? false;
  const captionBlockCount =
    parsedCaptionBlocks ??
    (!captionsSkipped && timelineCaptionBlocks > 0
      ? timelineCaptionBlocks
      : null);

  return {
    cutDetails,
    captionDetails,
    removedDurationSec,
    removedWordCount,
    captionBlockCount,
    captionsAdded: captionBlockCount !== null && captionBlockCount > 0,
    finalDurationSec: finalTimeline.duration_sec,
    nextStep: "Review the preview, then add B-roll or export.",
  };
}

function fallbackExportFilename(settings: ExportSettingsSnapshot): string {
  const ratio = settings.aspectRatio.replace(":", "x");
  return `export-${ratio}-${settings.resolution}.${settings.format}`;
}

function trimInlineText(value: string, maxLength = 32): string {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength - 1).trimEnd()}…`;
}

function clipTimelineDurationSec(clip: Clip): number {
  return Math.max(
    (clip.end_sec - clip.start_sec) / Math.max(clip.speed, 0.01),
    0.1,
  );
}

function transcriptDisplayText(
  word: Pick<TranscriptWord, "text" | "display_text">,
  preferRomanized: boolean,
): string {
  const romanized =
    typeof word.display_text === "string" ? word.display_text.trim() : "";
  return preferRomanized && romanized ? romanized : word.text;
}

const SCRIPT_TAG_LABELS: Record<
  NonNullable<TranscriptWord["script_tag"]>,
  string
> = {
  latin: "Latin",
  indic: "Indic",
  arabic: "Arabic",
  mixed: "Mixed",
  other: "Other",
};

function hasRomanizedTranscript(
  words: TranscriptWord[] | null | undefined,
): boolean {
  if (!words?.length) return false;
  return words.some((word) => {
    const romanized =
      typeof word.display_text === "string" ? word.display_text.trim() : "";
    return !!romanized && romanized !== word.text;
  });
}

type LivePreviewCaptionWord = {
  text: string;
  key: string;
  isActive: boolean;
  isPast: boolean;
};

function findFirstCaptionTimeSec(timeline: ProjectTimeline): number | null {
  let first: number | null = null;
  for (const track of timeline.tracks) {
    if (track.kind !== "video") continue;
    for (const clip of track.clips) {
      for (const overlay of clip.text_overlays ?? []) {
        const absoluteStart = Math.max(
          (clip.timeline_start_sec ?? 0) + (overlay.start_sec ?? 0),
          0,
        );
        if (first === null || absoluteStart < first) {
          first = absoluteStart;
        }
      }
    }
  }
  return first;
}

function computeBrollSlotBudget(
  project: Project,
  transcript: Transcript,
): number {
  const fallbackDuration = Math.max(
    1,
    transcript.words.length
      ? transcript.words[transcript.words.length - 1].end_sec
      : 0,
  );
  const durationSec =
    Number.isFinite(transcript.duration_sec) && transcript.duration_sec > 0
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
  projectVideoCount: number,
): BrollGenerationPlan {
  const baseMaxSlots = computeBrollSlotBudget(project, transcript);
  const intensityMultiplier =
    intensity === "low" ? 0.82 : intensity === "high" ? 1.2 : 1.0;
  const isVertical = project.height >= project.width;
  const hasEnoughLocalBroll = projectVideoCount >= 2;

  if (mode === "fast") {
    const maxSlots = clampInt(
      Math.round(baseMaxSlots * 0.4 * intensityMultiplier),
      3,
      isVertical ? 6 : 5,
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
      minConfidence:
        intensity === "low" ? 0.82 : intensity === "high" ? 0.72 : 0.76,
      usedExternalFallback,
    };
  }

  if (mode === "balanced") {
    const maxSlots = clampInt(
      Math.round(baseMaxSlots * 0.72 * intensityMultiplier),
      isVertical ? 4 : 3,
      isVertical ? 10 : 8,
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
      minConfidence:
        intensity === "low" ? 0.86 : intensity === "high" ? 0.74 : 0.8,
      usedExternalFallback: false,
    };
  }

  const maxSlots = clampInt(
    Math.round(baseMaxSlots * 1.08 * intensityMultiplier),
    isVertical ? 6 : 4,
    isVertical ? 16 : 10,
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
    minConfidence:
      intensity === "low" ? 0.88 : intensity === "high" ? 0.76 : 0.82,
    usedExternalFallback: false,
  };
}

function mapEditedTimeToSourceTime(
  editedSec: number,
  timelineSortedClips: Clip[],
): number {
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
      return (
        clip.start_sec + (timelineSec - clipStart) * Math.max(clip.speed, 0.01)
      );
    }
  }

  return timelineSortedClips[timelineSortedClips.length - 1].end_sec;
}

function mapSourceTimeToEditedTime(
  sourceSec: number,
  timelineSortedClips: Clip[],
): number {
  if (!Number.isFinite(sourceSec)) return 0;
  if (!timelineSortedClips.length) return Math.max(sourceSec, 0);

  const sourceTime = Math.max(sourceSec, 0);
  for (const clip of timelineSortedClips) {
    if (sourceTime >= clip.start_sec && sourceTime <= clip.end_sec) {
      return (
        clip.timeline_start_sec +
        (sourceTime - clip.start_sec) / Math.max(clip.speed, 0.01)
      );
    }
  }

  // If source time falls in a removed gap, snap to the nearest kept edge.
  let nearestEdited = timelineSortedClips[0].timeline_start_sec;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const clip of timelineSortedClips) {
    const clipDuration = clipTimelineDurationSec(clip);
    const candidates = [
      { source: clip.start_sec, edited: clip.timeline_start_sec },
      { source: clip.end_sec, edited: clip.timeline_start_sec + clipDuration },
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
  if (
    sourceType === "generated_video" ||
    sourceType === "generated_image_video"
  )
    return "GenAI";
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
  const keys = [
    "semantic",
    "alignment",
    "specificity",
    "diversity",
    "content",
    "entity",
    "metadata",
    "crop",
    "duration",
  ];
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
    meaning_uncertain: "meaning needs review",
    no_candidates: "no candidates",
    semantic_weak: "semantic weak",
    specificity_low: "generic match",
    talking_head_risk: "talking-head risk",
  };
  if (customLabels[code]) return customLabels[code];
  return code.replace(/_/g, " ");
}

function autoApplySkipReasonLabel(reason: string): string {
  if (reason === "no_candidates") return "no candidates found";
  if (reason === "needs_review") return "below confidence threshold";
  if (reason === "materialize_failed") return "stock download failed";
  if (reason === "meaning_uncertain") return "meaning needs review";
  return reason.replace(/_/g, " ");
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
  mediaById: Map<string, MediaAsset>,
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

function readReasonText(
  reason: Record<string, unknown> | undefined,
  key: string,
): string | null {
  const value = reason?.[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function readReasonNumber(
  reason: Record<string, unknown> | undefined,
  key: string,
): number | null {
  const value = reason?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function readReasonStringList(
  reason: Record<string, unknown> | undefined,
  key: string,
): string[] {
  const value = reason?.[key];
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is string => typeof item === "string" && !!item.trim())
    .map((item) => item.trim());
}

function getSlotTranscriptText(
  slot: BrollSlot,
  wordsById: Map<string, TranscriptWord>,
  allWords: TranscriptWord[],
  preferRomanized: boolean,
): string {
  const anchorText = slot.anchor_word_ids
    .map((wordId) => {
      const word = wordsById.get(wordId);
      if (!word) return "";
      return transcriptDisplayText(word, preferRomanized).trim();
    })
    .filter(Boolean)
    .join(" ")
    .trim();
  if (anchorText) return anchorText;

  return allWords
    .filter(
      (word) => word.start_sec < slot.end_sec && word.end_sec > slot.start_sec,
    )
    .map((word) => transcriptDisplayText(word, preferRomanized).trim())
    .filter(Boolean)
    .join(" ")
    .trim();
}

function buildBrollRerollPayload(englishGlossOverride?: string) {
  const trimmed = englishGlossOverride?.trim();
  return {
    candidates_per_slot: 2,
    include_project_assets: true,
    include_external_sources: true,
    ai_rerank: true,
    ...(trimmed ? { english_gloss_override: trimmed } : {}),
  };
}

function humanizeBrollMeta(value: string): string {
  return value.replace(/_/g, " ");
}

function buildSpeakerLegend(
  words: TranscriptWord[],
): Array<{ speakerId: string; label: string; slot: number }> {
  const seen: string[] = [];
  for (const word of words) {
    if (word.speaker_id && !seen.includes(word.speaker_id)) {
      seen.push(word.speaker_id);
    }
  }
  return seen.map((speakerId, slot) => {
    const label =
      words.find((entry) => entry.speaker_id === speakerId)?.speaker_label ||
      `Speaker ${slot + 1}`;
    return { speakerId, label, slot };
  });
}

function speakerSlotForWord(
  word: TranscriptWord,
  speakerIds: string[],
): number | null {
  if (!word.speaker_id) return null;
  const index = speakerIds.indexOf(word.speaker_id);
  return index >= 0 ? index : null;
}

function looksLikeDuetFilename(filename: string): boolean {
  return /\bfeat(?:uring)?\.?\b|\bft\.?\b|\s&\s|\s+x\s+|\bduet\b/i.test(
    filename,
  );
}

function buildSentenceBlocks(
  words: TranscriptWord[],
  preferRomanized = false,
): TextBlock[] {
  if (!words.length) return [];
  const blocks: TextBlock[] = [];
  let current: TranscriptWord[] = [];

  const flush = () => {
    if (!current.length) return;
    const id = `sent-${blocks.length + 1}`;
    blocks.push({
      id,
      wordIds: current.map((word) => word.id),
      text: current
        .map((word) => transcriptDisplayText(word, preferRomanized))
        .join(" "),
      startSec: current[0].start_sec,
      endSec: current[current.length - 1].end_sec,
    });
    current = [];
  };

  for (let idx = 0; idx < words.length; idx += 1) {
    const word = words[idx];
    const prev = idx > 0 ? words[idx - 1] : null;
    if (current.length > 0 && prev && word.start_sec - prev.end_sec > 1.1) {
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
      endSec: current[current.length - 1].endSec,
    });
    current = [];
  };

  for (let idx = 0; idx < sentences.length; idx += 1) {
    const sentence = sentences[idx];
    const prev = idx > 0 ? sentences[idx - 1] : null;
    if (current.length > 0 && prev && sentence.startSec - prev.endSec > 1.5) {
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
  return text
    .toLowerCase()
    .replace(/^[^a-z0-9']+|[^a-z0-9']+$/g, "")
    .trim();
}

function detectFillerWordIds(words: TranscriptWord[]): Set<string> {
  const result = new Set<string>();
  if (!words.length) return result;

  const tokens = words.map((word) => normalizeFillerToken(word.text));
  const singleWordSet = ENABLE_AGGRESSIVE_FILLER_SINGLE_WORDS
    ? new Set([
        ...FILLER_SINGLE_WORDS_CONSERVATIVE,
        ...FILLER_SINGLE_WORDS_AGGRESSIVE,
      ])
    : FILLER_SINGLE_WORDS_CONSERVATIVE;

  for (let idx = 0; idx < words.length; idx += 1) {
    for (const phrase of FILLER_MULTI_WORD_PHRASES) {
      if (idx + phrase.length > words.length) continue;
      const matches = phrase.every(
        (token, offset) => tokens[idx + offset] === token,
      );
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
type UndoEntry =
  | { kind: "selection"; deletedIds: Set<string> }
  | { kind: "cut"; transcript: Transcript; timeline: ProjectTimeline };
const MAX_UNDO = 80;

function App() {
  const [project, setProject] = useState<Project | null>(null);
  const timelineVersionRef = useRef(0);
  const timelineMutationCoordinatorRef = useRef(
    createTimelineMutationCoordinator(),
  );
  const [media, setMedia] = useState<MediaAsset[]>([]);
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>(null);
  const [transcriptMode, setTranscriptMode] = useState<TranscriptMode>("auto");
  const [transcriptSpeed, setTranscriptSpeed] =
    useState<TranscriptSpeed>("normal");
  const [transcriptLanguage, setTranscriptLanguage] = useState(() => {
    try {
      return localStorage.getItem("clipmind_transcript_lang") || "auto";
    } catch {
      return "auto";
    }
  });
  const [transcriptStageStartedAtMs, setTranscriptStageStartedAtMs] = useState<
    number | null
  >(null);
  const [transcriptStageBaseProgress, setTranscriptStageBaseProgress] =
    useState(0);
  const [transcript, setTranscript] = useState<Transcript | null>(null);

  useLayoutEffect(() => {
    timelineMutationCoordinatorRef.current.activate(project?.id ?? null);
    timelineVersionRef.current = project?.timeline_version ?? 0;
  }, [project?.id, project?.timeline_version]);

  const [deletedWordIds, setDeletedWordIds] = useState<Set<string>>(new Set());
  const [selectedWordIds, setSelectedWordIds] = useState<Set<string>>(
    new Set(),
  );
  const [anchorWordId, setAnchorWordId] = useState<string | null>(null);
  const [weakReviewIndex, setWeakReviewIndex] = useState(0);

  // Undo / Redo
  const undoStack = useRef<UndoEntry[]>([]);
  const redoStack = useRef<UndoEntry[]>([]);
  const [undoStackSize, setUndoStackSize] = useState(0);
  const [redoStackSize, setRedoStackSize] = useState(0);
  const [timelineCanUndo, setTimelineCanUndo] = useState(false);
  const [timelineCanRedo, setTimelineCanRedo] = useState(false);

  // Inline editing
  const [editingWordId, setEditingWordId] = useState<string | null>(null);
  const [editingWordText, setEditingWordText] = useState("");
  const [editingCaptionOverlayId, setEditingCaptionOverlayId] = useState<
    string | null
  >(null);
  const [editingCaptionText, setEditingCaptionText] = useState("");
  const captionEditInputRef = useRef<HTMLInputElement | null>(null);
  const [updatingTranscriptRange, setUpdatingTranscriptRange] = useState(false);
  const [showRomanizedTranscript, setShowRomanizedTranscript] = useState(false);
  const editInputRef = useRef<HTMLInputElement | null>(null);
  const committingWordEditRef = useRef(false);

  // Search
  const [searchQuery, setSearchQuery] = useState("");
  const [searchMatchIndex, setSearchMatchIndex] = useState(0);

  // Drag selection
  const isDragging = useRef(false);
  const dragStartWordId = useRef<string | null>(null);

  // Loading states
  const [creatingProject, setCreatingProject] = useState(false);
  const [recentProjects, setRecentProjects] = useState<Project[]>([]);
  const [projectsPanelOpen, setProjectsPanelOpen] = useState(false);
  const [loadingProjects, setLoadingProjects] = useState(false);
  const [openingProjectId, setOpeningProjectId] = useState<string | null>(null);
  const [submittingRenameId, setSubmittingRenameId] = useState<string | null>(
    null,
  );
  const [deletingProjectId, setDeletingProjectId] = useState<string | null>(
    null,
  );
  const [showNewProjectForm, setShowNewProjectForm] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [ingestingUrl, setIngestingUrl] = useState(false);
  const [ingestProgress, setIngestProgress] = useState(0);
  const [ingestStatusMessage, setIngestStatusMessage] = useState("");
  const [generatingTranscript, setGeneratingTranscript] = useState(false);
  const [transcriptStartedAtMs, setTranscriptStartedAtMs] = useState<
    number | null
  >(null);
  const [transcriptElapsedSec, setTranscriptElapsedSec] = useState(0);
  const [transcriptJob, setTranscriptJob] = useState<Job | null>(null);
  const transcriptStageKeyRef = useRef<string | null>(null);
  const [applyingCut, setApplyingCut] = useState(false);
  const [queueingPreview, setQueueingPreview] = useState(false);
  const [runningAction, setRunningAction] = useState<VibeAction | null>(null);
  const [brollSlots, setBrollSlots] = useState<BrollSlot[]>([]);
  const [brollConfig, setBrollConfig] = useState<BrollConfig | null>(null);
  const [lastBrollAutoApplySkips, setLastBrollAutoApplySkips] = useState<
    Array<{
      slot_id: string;
      concept_text: string;
      reason: string;
      detail?: string | null;
    }>
  >([]);
  const [loadingBrollSlots, setLoadingBrollSlots] = useState(false);
  const [suggestingBroll, setSuggestingBroll] = useState(false);
  const [suggestingBrollSelection, setSuggestingBrollSelection] =
    useState(false);
  const [autoApplyingBroll, setAutoApplyingBroll] = useState(false);
  const [syncingBroll, setSyncingBroll] = useState(false);
  const [undoingBroll, setUndoingBroll] = useState(false);
  const [brollActionKey, setBrollActionKey] = useState<string | null>(null);
  const [brollTimelineActionKey, setBrollTimelineActionKey] = useState<
    string | null
  >(null);
  const [brollDraftStartById, setBrollDraftStartById] = useState<
    Record<string, string>
  >({});
  const [brollDraftDurationById, setBrollDraftDurationById] = useState<
    Record<string, string>
  >({});
  const [brollDraftOpacityById, setBrollDraftOpacityById] = useState<
    Record<string, number>
  >({});
  const [brollSyncMode, setBrollSyncMode] = useState<"replace" | "append">(
    "replace",
  );
  const [brollAutoMode, setBrollAutoMode] = useState<BrollAutoMode>("fast");
  const [brollDefaultOpacity, setBrollDefaultOpacity] = useState(1);
  const [brollIntensity, setBrollIntensity] =
    useState<BrollIntensity>("medium");
  const [brollSuggestJob, setBrollSuggestJob] = useState<Job | null>(null);
  const [brollSuggestionSource, setBrollSuggestionSource] =
    useState<BrollSuggestionSource | null>(null);
  const [brollSelectionLabel, setBrollSelectionLabel] = useState("");
  const [expandedBrollSlots, setExpandedBrollSlots] = useState<
    Record<string, boolean>
  >({});
  const [brollMeaningDrafts, setBrollMeaningDrafts] = useState<
    Record<string, string>
  >({});
  const [activeFeatureTab, setActiveFeatureTab] =
    useState<FeatureTabId>("broll_studio");
  const [featureDrawerOpen, setFeatureDrawerOpen] = useState(false);
  const [mobileWorkspaceTab, setMobileWorkspaceTab] = useState<
    "preview" | "transcript" | "timeline"
  >("preview");
  const [selectedTimelineClip, setSelectedTimelineClip] =
    useState<InspectorTimelineSelection | null>(null);
  const [clipClipboard, setClipClipboard] = useState<{
    clip: Clip;
    laneKind: "video" | "audio";
    laneLabel: string;
  } | null>(null);
  const [selectedBrollClipId, setSelectedBrollClipId] = useState<string | null>(
    null,
  );
  const [selectedBrollSlotId, setSelectedBrollSlotId] = useState<string | null>(
    null,
  );
  const [selectedCaptionBlock, setSelectedCaptionBlock] =
    useState<InspectorCaptionSelection | null>(null);
  const [lockedLaneIds, setLockedLaneIds] = useState<Set<string>>(
    () => new Set(),
  );

  // Captions
  const [captionStyle, setCaptionStyle] = useState<string>("basic_white");
  const [captionResultInfo, setCaptionResultInfo] = useState<string | null>(
    null,
  );
  const [removingCaptions, setRemovingCaptions] = useState(false);
  const selectedCaptionStyleName = useMemo(
    () =>
      CAPTION_STYLE_PRESETS.find((item) => item.id === captionStyle)?.name ??
      captionStyle,
    [captionStyle],
  );

  // Export
  const [exportFormat, setExportFormat] = useState<"mp4" | "mov" | "webm">(
    "mp4",
  );
  const [exportAspectRatio, setExportAspectRatio] =
    useState<ExportAspectRatio>("9:16");
  const [exportResolution, setExportResolution] = useState<
    "720p" | "1080p" | "4k"
  >("1080p");
  const [exportFps, setExportFps] = useState<24 | 30 | 60>(30);
  const [exportQuality, setExportQuality] = useState<
    "low" | "medium" | "high" | "max"
  >("high");
  const [previewFrameAspectRatio, setPreviewFrameAspectRatio] =
    useState<ExportAspectRatio>("16:9");
  const [showExportFrameGuide, setShowExportFrameGuide] = useState(false);
  const [autoFraming, setAutoFraming] = useState(false);
  const [smartReframing, setSmartReframing] = useState(false);
  const [exportingVideo, setExportingVideo] = useState(false);
  const [exportJob, setExportJob] = useState<Job | null>(null);
  const [exportCompletion, setExportCompletion] =
    useState<ExportCompletionSummary | null>(null);
  const [downloadingExport, setDownloadingExport] = useState(false);

  const [previewJob, setPreviewJob] = useState<Job | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewUpdateQueued, setPreviewUpdateQueued] = useState(false);
  const [currentTimeSec, setCurrentTimeSec] = useState(0);
  const timelineClockRef = useRef(createTimelineClock(0));
  const lastAppClockRenderMsRef = useRef(0);
  const canonicalTimelineFps = resolveCanonicalFps(
    project?.timeline.fps,
    project?.fps,
  );

  // Waveform data for timeline
  const [waveformPeaks, setWaveformPeaks] = useState<number[]>([]);

  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<
    "checking" | "ok" | "down"
  >("checking");
  const [quickEditing, setQuickEditing] = useState(false);
  const [quickEditStage, setQuickEditStage] = useState("");
  const [quickEditSummary, setQuickEditSummary] =
    useState<QuickEditSummary | null>(null);
  const [showShortcutsHelp, setShowShortcutsHelp] = useState(false);

  // Quick edit state machine phase ref — drives the useEffect pipeline
  // "idle" | "transcribing" | "cutting" | "captioning" | "done"
  const quickEditPhaseRef = useRef<
    "idle" | "transcribing" | "cutting" | "captioning" | "done"
  >("idle");

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const lastAppliedSignatureRef = useRef<string>("");
  const lastAutoCutFailedSignatureRef = useRef<string | null>(null);
  const latestDeletedWordIdsRef = useRef<Set<string>>(new Set());
  const pendingPreviewRefreshRef = useRef(false);
  const transcriptBoxRef = useRef<HTMLDivElement | null>(null);
  const transcriptFollowPlaybackRef = useRef(true);
  const transcriptProgrammaticScrollRef = useRef(false);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const activeWordRef = useRef<HTMLButtonElement | null>(null);
  const autoCreateAttemptedRef = useRef(false);
  const pendingCaptionSeekRef = useRef<{
    jobId: string;
    targetSec: number;
  } | null>(null);
  const transcriptJobResultHandledRef = useRef<string | null>(null);
  const lastTranscriptRequestRef = useRef<{ forceRegenerate: boolean }>({
    forceRegenerate: false,
  });
  const playbackSyncFrameRef = useRef<number | null>(null);
  const exportSettingsRef = useRef<ExportSettingsSnapshot | null>(null);
  const exportCompletionHandledRef = useRef<string | null>(null);

  const videoAssets = useMemo(
    () => media.filter((asset) => asset.media_type === "video"),
    [media],
  );

  const selectedVideoAsset = useMemo(() => {
    if (!selectedAssetId) return videoAssets[0] ?? null;
    return videoAssets.find((asset) => asset.id === selectedAssetId) ?? null;
  }, [selectedAssetId, videoAssets]);

  const recentProjectItems = useMemo(
    () => recentProjects.slice(0, 8),
    [recentProjects],
  );

  const brollSetupWarning = useMemo(() => {
    if (!brollConfig) return null;
    if (videoAssets.length >= 2) return null;
    if (brollConfig.stock_search_available) return null;
    return {
      title: "Limited B-roll sources",
      detail:
        "Only one video is uploaded and no Pexels/Pixabay API keys are configured on the server. Upload extra cutaway clips, or set PEXELS_API_KEY / PIXABAY_API_KEY for stock footage.",
    };
  }, [brollConfig, videoAssets.length]);

  const selectedAssetDurationSec = useMemo(() => {
    if (
      typeof selectedVideoAsset?.duration_sec === "number" &&
      selectedVideoAsset.duration_sec > 0
    ) {
      return selectedVideoAsset.duration_sec;
    }
    return project?.timeline.duration_sec ?? null;
  }, [project?.timeline.duration_sec, selectedVideoAsset?.duration_sec]);

  const transcriptRuntimeHint = useMemo(
    () =>
      estimateTranscriptRuntimeLabel(
        transcriptMode,
        selectedAssetDurationSec,
        transcriptSpeed,
      ),
    [selectedAssetDurationSec, transcriptMode, transcriptSpeed],
  );

  const quickEditRuntimeHint = useMemo(
    () =>
      estimateQuickEditRuntimeLabel(
        transcriptMode,
        selectedAssetDurationSec,
        !!transcript?.words?.length,
        transcriptSpeed,
      ),
    [
      selectedAssetDurationSec,
      transcript?.words?.length,
      transcriptMode,
      transcriptSpeed,
    ],
  );

  const selectedBrollPlan = useMemo(() => {
    if (!project || !transcript) return null;
    return resolveBrollGenerationPlan(
      project,
      transcript,
      brollIntensity,
      brollAutoMode,
      videoAssets.length,
    );
  }, [brollAutoMode, brollIntensity, project, transcript, videoAssets.length]);

  const videoClips = useMemo<Clip[]>(() => {
    if (!project) return [];
    const videoTrack = project.timeline.tracks.find(
      (track) => track.kind === "video",
    );
    return (videoTrack?.clips ?? [])
      .slice()
      .sort((a, b) => a.timeline_start_sec - b.timeline_start_sec);
  }, [project]);

  const timelineLanes = useMemo<TimelineLane[]>(() => {
    if (!project) return [];
    let videoIndex = 0;
    let audioIndex = 0;
    const lanes: TimelineLane[] = [];
    for (const track of project.timeline.tracks) {
      if (track.kind !== "video" && track.kind !== "audio") continue;
      const label =
        track.kind === "video" ? `V${++videoIndex}` : `A${++audioIndex}`;
      lanes.push({
        id: track.id,
        label,
        kind: track.kind,
        clips: (track.clips ?? [])
          .slice()
          .sort((a, b) => a.timeline_start_sec - b.timeline_start_sec),
        mute: track.mute,
        solo: track.solo,
        volume: track.volume,
      });
    }
    return lanes;
  }, [project]);

  const overlayClips = useMemo<Clip[]>(() => {
    if (!project) return [];
    const overlayTrack = project.timeline.tracks.find(
      (track) => track.kind === "overlay",
    );
    return overlayTrack?.clips ?? [];
  }, [project]);

  const sortedOverlayClips = useMemo<Clip[]>(
    () =>
      overlayClips
        .slice()
        .sort((a, b) => a.timeline_start_sec - b.timeline_start_sec),
    [overlayClips],
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

  const assetNameById = useMemo(() => {
    const index = new Map<string, string>();
    media.forEach((item) => {
      index.set(item.id, item.filename);
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
            startSec: clip.timeline_start_sec + overlay.start_sec / speed,
            durationSec: overlay.duration_sec / speed,
            clipTimelineStartSec: clip.timeline_start_sec,
            clipSourceDurationSec: Math.max(
              clip.end_sec - clip.start_sec,
              0.05,
            ),
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

  const transcriptHasRomanization = useMemo(
    () => hasRomanizedTranscript(transcript?.words),
    [transcript?.words],
  );
  const transcriptScriptSummary = useMemo(() => {
    const tags = (transcript?.script_tags ?? []).filter(Boolean);
    return tags.map(
      (tag) => SCRIPT_TAG_LABELS[tag as keyof typeof SCRIPT_TAG_LABELS] ?? tag,
    );
  }, [transcript?.script_tags]);

  const timelineAssistWords = useMemo<TranscriptWord[]>(() => {
    if (!transcript) return [];
    return transcript.words
      .map((word) => {
        const editedStart = mapSourceTimeToEditedTime(
          word.start_sec,
          videoClips,
        );
        const editedEnd = mapSourceTimeToEditedTime(word.end_sec, videoClips);
        return {
          ...word,
          start_sec: editedStart,
          end_sec: Math.max(editedEnd, editedStart),
        };
      })
      .sort((a, b) => a.start_sec - b.start_sec);
  }, [transcript, videoClips]);

  const sentenceBlocks = useMemo(
    () => buildSentenceBlocks(transcript?.words ?? [], showRomanizedTranscript),
    [transcript?.words, showRomanizedTranscript],
  );
  const paragraphBlocks = useMemo(
    () => buildParagraphBlocks(sentenceBlocks),
    [sentenceBlocks],
  );

  const deletedSignature = useMemo(
    () => Array.from(deletedWordIds).sort().join(","),
    [deletedWordIds],
  );

  useEffect(() => {
    latestDeletedWordIdsRef.current = new Set(deletedWordIds);
  }, [deletedWordIds]);

  const keptWordIds = useMemo(() => {
    if (!transcript) return [] as string[];
    return transcript.words
      .filter((word) => !deletedWordIds.has(word.id))
      .map((word) => word.id);
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

  const transcriptIssueWordIdSet = useMemo(() => {
    const ids = new Set<string>();
    transcriptIssueRegions.forEach((region) => {
      region.word_ids.forEach((wordId) => ids.add(wordId));
    });
    return ids;
  }, [transcriptIssueRegions]);

  useEffect(() => {
    if (!transcriptHasRomanization) {
      setShowRomanizedTranscript(false);
      return;
    }
    setShowRomanizedTranscript(true);
  }, [transcript?.id, transcriptHasRomanization]);

  useEffect(() => {
    setWeakReviewIndex(0);
  }, [transcript?.id]);

  const lowConfidenceCount = useMemo(() => {
    if (!transcript) return 0;
    return transcript.words.filter(
      (word) =>
        typeof word.confidence === "number" &&
        word.confidence < LOW_CONFIDENCE_THRESHOLD,
    ).length;
  }, [transcript]);
  const lowConfidenceRatio = useMemo(() => {
    if (!transcript || transcript.words.length === 0) return 0;
    return lowConfidenceCount / transcript.words.length;
  }, [transcript, lowConfidenceCount]);
  const shouldWarnLowConfidence =
    lowConfidenceCount >= LOW_CONFIDENCE_WARN_MIN_COUNT &&
    lowConfidenceRatio >= LOW_CONFIDENCE_WARN_RATIO;

  const weakQualityCount = useMemo(() => {
    if (!transcript) return 0;
    return transcript.words.filter(
      (word) =>
        !deletedWordIds.has(word.id) &&
        (word.quality_label === "weak" ||
          transcriptIssueWordIdSet.has(word.id)),
    ).length;
  }, [deletedWordIds, transcript, transcriptIssueWordIdSet]);

  const transcriptReviewWordIds = useMemo(() => {
    if (!transcript) return [] as string[];
    return transcript.words
      .filter((word) => {
        const lowConfidence =
          typeof word.confidence === "number" &&
          word.confidence < LOW_CONFIDENCE_THRESHOLD;
        return (
          !deletedWordIds.has(word.id) &&
          (lowConfidence ||
            word.quality_label === "weak" ||
            transcriptIssueWordIdSet.has(word.id))
        );
      })
      .map((word) => word.id);
  }, [deletedWordIds, transcript, transcriptIssueWordIdSet]);

  const reviewWordCount = transcriptReviewWordIds.length;
  const lowConfidenceOnlyCount = Math.max(
    0,
    reviewWordCount - weakQualityCount,
  );

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
        const progress = Math.max(
          0,
          Math.min(100, Math.round(previewJob.progress ?? 0)),
        );
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
    if (queueingPreview || previewJob?.status === "queued")
      return "Preparing render...";
    const progress = Math.max(
      0,
      Math.min(100, Math.round(previewJob?.progress ?? 0)),
    );
    return progress > 0 ? `Rendering ${progress}%` : "Rendering...";
  }, [
    previewRenderBusy,
    applyingCut,
    queueingPreview,
    previewUpdateQueued,
    previewJob?.message,
    previewJob?.status,
    previewJob?.progress,
  ]);

  const previewProgress = useMemo(
    () => Math.max(0, Math.min(100, Math.round(previewJob?.progress ?? 0))),
    [previewJob?.progress],
  );

  const exportProgress = useMemo(
    () => Math.max(0, Math.min(100, Math.round(exportJob?.progress ?? 0))),
    [exportJob?.progress],
  );

  const brollGenerationActive =
    suggestingBroll ||
    suggestingBrollSelection ||
    brollSuggestJob?.status === "queued" ||
    brollSuggestJob?.status === "running";

  const brollSuggestionProgress = useMemo(
    () =>
      Math.max(
        0,
        Math.min(
          100,
          Math.max(
            brollGenerationActive ? 5 : 0,
            Math.round(brollSuggestJob?.progress ?? 0),
          ),
        ),
      ),
    [brollGenerationActive, brollSuggestJob?.progress],
  );

  const brollSuggestionMessage = useMemo(() => {
    if (brollSuggestJob?.message?.trim()) return brollSuggestJob.message;
    if (brollSuggestJob?.status === "queued") return "B-roll generation queued...";
    return brollSuggestionSource === "selection"
      ? "Finding B-roll candidates for your selection..."
      : "Finding B-roll moments and visual candidates...";
  }, [brollSuggestJob?.message, brollSuggestJob?.status, brollSuggestionSource]);

  const transcriptActualProgress = useMemo(
    () => Math.max(0, Math.min(100, Math.round(transcriptJob?.progress ?? 0))),
    [transcriptJob?.progress],
  );

  const transcriptProgress = useMemo(() => {
    const actual = transcriptActualProgress;
    if (
      !transcriptJob ||
      (transcriptJob.status !== "queued" && transcriptJob.status !== "running")
    ) {
      return actual;
    }
    if (transcriptStageStartedAtMs === null) {
      return actual;
    }
    const elapsedSec = Math.max(
      0,
      Math.floor((Date.now() - transcriptStageStartedAtMs) / 1000),
    );
    const animated =
      transcriptStageBaseProgress +
      elapsedSec *
        transcriptStageRate(transcriptJob.stage, transcriptJob.status);
    const ceiling = transcriptStageCeiling(
      transcriptJob.stage,
      transcriptJob.status,
    );
    return Math.max(actual, Math.min(ceiling, Math.round(animated)));
  }, [
    transcriptActualProgress,
    transcriptElapsedSec,
    transcriptJob,
    transcriptStageBaseProgress,
    transcriptStageStartedAtMs,
  ]);

  const transcriptStageLabel = useMemo(() => {
    const stage = (transcriptJob?.stage ?? "").trim();
    switch (stage) {
      case "prepare":
        return "Preparing audio file...";
      case "prepare_audio":
        return "Isolating vocals from background music...";
      case "recognize":
        return "Recognizing speech...";
      case "lyrics":
        return "Detecting song lyrics...";
      case "refine":
        return "Refining word timings...";
      case "weak_retry":
        return "Re-analyzing unclear sections...";
      case "timeline":
        return "Building timeline...";
      case "reuse":
        return "Loading cached transcript...";
      default:
        return null;
    }
  }, [transcriptJob?.stage]);

  const transcriptStatusMessage = useMemo(() => {
    if (!transcriptJob) return "";
    if (transcriptJob.status === "completed") return "";
    if (transcriptJob.status === "failed")
      return transcriptJob.error ?? "Transcript generation failed.";
    if (transcriptStageLabel) return transcriptStageLabel;
    if (transcriptJob.message?.trim()) return transcriptJob.message.trim();
    return transcriptProgress > 0
      ? `Generating transcript ${transcriptProgress}%`
      : "Preparing transcript...";
  }, [transcriptJob, transcriptProgress, transcriptStageLabel]);

  const exportStatusMessage = useMemo(() => {
    if (!exportJob) return "";
    if (exportJob.message?.trim()) return exportJob.message.trim();
    if (exportJob.status === "completed") return "Export completed.";
    if (exportJob.status === "failed")
      return exportJob.error ?? "Export failed.";
    return exportProgress > 0
      ? `Rendering export ${exportProgress}%`
      : "Preparing export...";
  }, [exportJob, exportProgress]);

  const syncVideoTimeOnce = useCallback(() => {
    const element = videoRef.current;
    if (!element) return;
    const nextTime = element.currentTime;
    timelineClockRef.current.setTime(nextTime);
    setCurrentTimeSec((prev) =>
      Math.abs(prev - nextTime) >= 0.005 ? nextTime : prev,
    );
  }, []);

  const syncVideoTimeIfPlaying = useCallback(() => {
    const element = videoRef.current;
    if (!element || element.paused || element.ended) return;
    syncVideoTimeOnce();
  }, [syncVideoTimeOnce]);

  const stopPlaybackSync = useCallback(() => {
    if (playbackSyncFrameRef.current !== null) {
      window.cancelAnimationFrame(playbackSyncFrameRef.current);
      playbackSyncFrameRef.current = null;
    }
  }, []);

  const startPlaybackSync = useCallback(() => {
    if (playbackSyncFrameRef.current !== null) return;
    const tick = () => {
      const element = videoRef.current;
      if (!element) {
        playbackSyncFrameRef.current = null;
        return;
      }
      const nextTime = element.currentTime;
      timelineClockRef.current.setTime(nextTime);
      if (
        !TIMELINE_CORE_V2 ||
        performance.now() - lastAppClockRenderMsRef.current >= 100
      ) {
        lastAppClockRenderMsRef.current = performance.now();
        setCurrentTimeSec((prev) =>
          Math.abs(prev - nextTime) >= 0.005 ? nextTime : prev,
        );
      }
      if (!element.paused && !element.ended) {
        playbackSyncFrameRef.current = window.requestAnimationFrame(tick);
      } else {
        playbackSyncFrameRef.current = null;
      }
    };
    playbackSyncFrameRef.current = window.requestAnimationFrame(tick);
  }, []);

  useEffect(() => {
    timelineClockRef.current.setTime(currentTimeSec);
  }, [currentTimeSec]);

  useEffect(() => {
    if (!previewJob || previewJob.status !== "completed") {
      return;
    }
    const pending = pendingCaptionSeekRef.current;
    if (pending && previewJob.id !== pending.jobId) {
      return;
    }
    const element = videoRef.current;
    if (!element) return;
    // Without a pending caption seek, still nudge the freshly mounted
    // <video> off t=0 — a paused remounted element never decodes a frame,
    // leaving the preview black until something forces a paint. Only do so
    // while the element is parked at the start, so a completing render can't
    // yank the playhead mid-viewing.
    if (!pending && (!element.paused || element.currentTime > 0.05)) {
      return;
    }
    const seekTarget = pending ? Math.max(0, pending.targetSec) : 0.001;
    const applySeek = () => {
      const duration = Number.isFinite(element.duration)
        ? element.duration
        : seekTarget + 0.1;
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
    element.addEventListener("loadedmetadata", onLoadedMetadata, {
      once: true,
    });
    return () =>
      element.removeEventListener("loadedmetadata", onLoadedMetadata);
  }, [previewJob?.id, previewJob?.status, previewUrl]);

  useEffect(() => stopPlaybackSync, [stopPlaybackSync]);

  // While a fresh preview is rendering, the visible player can still be the prior render.
  // In that state, keep transcript tracking on source-time so highlighting remains stable.
  const transcriptPlaybackTimeSec = useMemo(
    () =>
      previewRenderBusy
        ? Math.max(0, currentTimeSec)
        : mapEditedTimeToSourceTime(currentTimeSec, videoClips),
    [previewRenderBusy, currentTimeSec, videoClips],
  );

  const activeWordId = useMemo(() => {
    if (!transcript) return null;

    const direct = transcript.words.find((word) => {
      if (!previewRenderBusy && deletedWordIds.has(word.id)) return false;
      return (
        transcriptPlaybackTimeSec >= word.start_sec &&
        transcriptPlaybackTimeSec <= word.end_sec
      );
    });
    if (direct) return direct.id;

    // If playhead sits inside a removed gap, snap highlight to the nearest kept word.
    const nextKept = transcript.words.find(
      (word) =>
        !deletedWordIds.has(word.id) &&
        word.start_sec >= transcriptPlaybackTimeSec,
    );
    if (nextKept) return nextKept.id;

    for (let idx = transcript.words.length - 1; idx >= 0; idx -= 1) {
      const word = transcript.words[idx];
      if (deletedWordIds.has(word.id)) continue;
      if (word.end_sec <= transcriptPlaybackTimeSec) return word.id;
    }
    return null;
  }, [
    transcript,
    deletedWordIds,
    transcriptPlaybackTimeSec,
    previewRenderBusy,
  ]);

  const shouldShowLiveCaptionOverlay =
    !previewShowsRenderedOutput &&
    ((previewFrameAspectRatio === "16:9" && exportAspectRatio === "9:16") ||
      previewRenderBusy ||
      previewUpdateQueued ||
      !previewUrl);

  const livePreviewCaption = useMemo(() => {
    if (!project) return null;
    if (!shouldShowLiveCaptionOverlay) return null;
    const videoTrack = project.timeline.tracks.find(
      (track) => track.kind === "video",
    );
    if (!videoTrack) return null;

    const previewTimeSec = Math.max(0, currentTimeSec);
    const usesTimelineTime = previewShowsRenderedOutput;
    for (const clip of videoTrack.clips) {
      const overlays = clip.text_overlays ?? [];
      if (!overlays.length) continue;
      const speed = Math.max(clip.speed, 0.01);
      for (const overlay of overlays) {
        const startSec = usesTimelineTime
          ? clip.timeline_start_sec + overlay.start_sec / speed
          : clip.start_sec + overlay.start_sec;
        const endSec = usesTimelineTime
          ? clip.timeline_start_sec +
            (overlay.start_sec + overlay.duration_sec) / speed
          : clip.start_sec + overlay.start_sec + overlay.duration_sec;
        if (previewTimeSec < startSec || previewTimeSec > endSec) continue;

        const rawWordTimings = Array.isArray(overlay.word_timings)
          ? overlay.word_timings
          : [];
        const displayTokens = overlay.text.trim().split(/\s+/).filter(Boolean);
        const captionWords: LivePreviewCaptionWord[] =
          rawWordTimings.length > 0
            ? rawWordTimings
                .map((word, index) => {
                  const sourceStartSec = Number.isFinite(word.start_sec)
                    ? word.start_sec
                    : startSec;
                  const sourceEndSec = Number.isFinite(word.end_sec)
                    ? word.end_sec
                    : sourceStartSec;
                  const wordStartSec = usesTimelineTime
                    ? clip.timeline_start_sec +
                      Math.max(sourceStartSec - clip.start_sec, 0) / speed
                    : sourceStartSec;
                  const wordEndSec = usesTimelineTime
                    ? clip.timeline_start_sec +
                      Math.max(sourceEndSec - clip.start_sec, 0) / speed
                    : sourceEndSec;
                  return {
                    text:
                      displayTokens.length === rawWordTimings.length
                        ? displayTokens[index]
                        : String(word.text ?? "").trim() ||
                          displayTokens[index] ||
                          "",
                    key: `${overlay.id}-word-${index}`,
                    isActive:
                      previewTimeSec >= wordStartSec &&
                      previewTimeSec <= wordEndSec,
                    isPast: previewTimeSec > wordEndSec,
                  };
                })
                .filter((word) => word.text)
            : [];

        return {
          text: overlay.text,
          words: captionWords,
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
  }, [
    project,
    shouldShowLiveCaptionOverlay,
    previewShowsRenderedOutput,
    currentTimeSec,
  ]);

  const selectedTimelineClipDetails = useMemo(() => {
    if (!selectedTimelineClip) return null;
    const lane = timelineLanes.find(
      (item) => item.id === selectedTimelineClip.laneId,
    );
    const clip = lane?.clips.find(
      (item) => item.id === selectedTimelineClip.clipId,
    );
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
    const lane = timelineLanes.find(
      (item) =>
        item.id === selectedCaptionBlock.laneId && item.kind === "video",
    );
    const clip = lane?.clips.find(
      (item) => item.id === selectedCaptionBlock.clipId,
    );
    const overlay = clip?.text_overlays.find(
      (item) => item.id === selectedCaptionBlock.overlayId,
    );
    if (!lane || !clip || !overlay) return null;
    const speed = Math.max(clip.speed, 0.01);
    return {
      lane,
      clip,
      overlay,
      timelineStartSec: clip.timeline_start_sec + overlay.start_sec / speed,
      durationSec: overlay.duration_sec / speed,
    };
  }, [selectedCaptionBlock, timelineLanes]);

  const selectedBrollClip = useMemo(() => {
    if (!selectedBrollClipId) return null;
    return (
      sortedOverlayClips.find((clip) => clip.id === selectedBrollClipId) ?? null
    );
  }, [selectedBrollClipId, sortedOverlayClips]);

  const inspectorContext = useMemo(() => {
    if (selectedBrollClip) {
      return {
        kind: "broll_clip" as const,
        title: "B-roll Clip Selected",
        detail: `Start ${formatSeconds(selectedBrollClip.timeline_start_sec)} · ${formatSeconds(
          clipTimelineDurationSec(selectedBrollClip),
        )} duration`,
        suggestedTab: "broll_studio" as const,
      };
    }
    if (selectedCaptionBlockDetails) {
      return {
        kind: "caption_block" as const,
        title: "Caption Block Selected",
        detail: `${trimInlineText(selectedCaptionBlockDetails.overlay.text, 64)} · ${formatSeconds(
          selectedCaptionBlockDetails.timelineStartSec,
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
          clipTimelineDurationSec(clip),
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
  }, [
    selectedBrollClip,
    selectedCaptionBlockDetails,
    selectedTimelineClipDetails,
    selectedWordIds.size,
  ]);

  // Fetch waveform peaks whenever video asset changes
  useEffect(() => {
    if (!selectedVideoAsset) {
      setWaveformPeaks([]);
      return;
    }
    let cancelled = false;
    api
      .getWaveform(selectedVideoAsset.id)
      .then((data) => {
        if (!cancelled) setWaveformPeaks(data.peaks);
      })
      .catch(() => {
        if (!cancelled) setWaveformPeaks([]);
      });
    return () => {
      cancelled = true;
    };
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

  useEffect(() => {
    setBrollMeaningDrafts((prev) => {
      const next: Record<string, string> = {};
      let changed = false;
      for (const slot of brollSlots) {
        if (prev[slot.id] !== undefined) {
          next[slot.id] = prev[slot.id];
          continue;
        }
        const gloss =
          slot.meaning?.english_gloss ??
          readReasonText(slot.candidates[0]?.reason, "english_gloss");
        next[slot.id] = gloss ?? "";
        changed = true;
      }
      if (Object.keys(prev).length !== Object.keys(next).length) {
        changed = true;
      }
      return changed ? next : prev;
    });
  }, [brollSlots]);

  // Search matches
  const searchMatchIds = useMemo(() => {
    if (!transcript || !searchQuery.trim()) return [] as string[];
    const q = searchQuery.toLowerCase().trim();
    return transcript.words
      .filter((word) => {
        const original = word.text.toLowerCase();
        const romanized =
          typeof word.display_text === "string"
            ? word.display_text.toLowerCase()
            : "";
        return original.includes(q) || romanized.includes(q);
      })
      .map((word) => word.id);
  }, [transcript, searchQuery]);

  // O(1) lookup set for search matches (avoids .includes() per word in render loop)
  const searchMatchIdSet = useMemo(
    () => new Set(searchMatchIds),
    [searchMatchIds],
  );

  // Filler word IDs
  const fillerWordIds = useMemo(() => {
    if (!transcript) return new Set<string>();
    return detectFillerWordIds(transcript.words);
  }, [transcript]);

  // ── Undo/Redo helpers ──────────────────────────────────────────────
  const syncLocalUndoRedoState = useCallback(() => {
    setUndoStackSize(undoStack.current.length);
    setRedoStackSize(redoStack.current.length);
  }, []);

  const applyProjectFromServer = useCallback((nextProject: Project) => {
    setProject(nextProject);
    setTimelineCanUndo(!!nextProject.timeline_can_undo);
    setTimelineCanRedo(!!nextProject.timeline_can_redo);
  }, []);

  const resetEditorStateForProject = useCallback(
    (nextProject: Project) => {
      applyProjectFromServer(nextProject);
      setMedia([]);
      setSelectedAssetId(null);
      setTranscript(null);
      setDeletedWordIds(new Set());
      setSelectedWordIds(new Set());
      setAnchorWordId(null);
      setWeakReviewIndex(0);
      setPreviewJob(null);
      setPreviewUrl(null);
      setPreviewUpdateQueued(false);
      pendingPreviewRefreshRef.current = false;
      setCurrentTimeSec(0);
      setExportJob(null);
      setExportCompletion(null);
      setQuickEditSummary(null);
      quickEditPhaseRef.current = "idle";
      setQuickEditing(false);
      setQuickEditStage("");
      setBrollSlots([]);
      setLastBrollAutoApplySkips([]);
      setBrollSuggestJob(null);
      setBrollSuggestionSource(null);
      setBrollSelectionLabel("");
      setBrollActionKey(null);
      setBrollTimelineActionKey(null);
      setBrollDraftStartById({});
      setBrollDraftDurationById({});
      setBrollDraftOpacityById({});
      setExpandedBrollSlots({});
      setBrollMeaningDrafts({});
      setSelectedTimelineClip(null);
      setSelectedBrollClipId(null);
      setSelectedBrollSlotId(null);
      setSelectedCaptionBlock(null);
      setSearchQuery("");
      setSearchMatchIndex(0);
      setLockedLaneIds(readLockedLaneIds(nextProject.id));
      lastAppliedSignatureRef.current = "";
      lastAutoCutFailedSignatureRef.current = null;
      transcriptJobResultHandledRef.current = null;
      undoStack.current = [];
      redoStack.current = [];
      syncLocalUndoRedoState();
    },
    [applyProjectFromServer, syncLocalUndoRedoState],
  );

  const pushUndo = useCallback(() => {
    undoStack.current.push({
      kind: "selection",
      deletedIds: new Set(deletedWordIds),
    });
    if (undoStack.current.length > MAX_UNDO) undoStack.current.shift();
    redoStack.current = [];
    syncLocalUndoRedoState();
  }, [deletedWordIds, syncLocalUndoRedoState]);

  // ── Core actions ───────────────────────────────────────────────────
  const refreshMedia = useCallback(
    async (projectId: string) => {
      const items = await api.listMedia(projectId);
      setMedia(items);
      const firstVideo = items.find((asset) => asset.media_type === "video");
      if (!selectedAssetId && firstVideo) {
        setSelectedAssetId(firstVideo.id);
      }
    },
    [selectedAssetId],
  );

  const refreshBrollSlots = useCallback(
    async (projectId: string, transcriptId?: string) => {
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
    },
    [],
  );

  const refreshProjectList = useCallback(async () => {
    setLoadingProjects(true);
    try {
      const projects = await api.listProjects();
      setRecentProjects(projects);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoadingProjects(false);
    }
  }, []);

  const queuePreview = useCallback(
    async (
      force = false,
      overrides?: {
        aspectRatio?: ExportAspectRatio;
        fps?: 24 | 30 | 60;
        autoFrame?: boolean;
      },
    ) => {
      if (!project || queueingPreview) return;
      if (
        !force &&
        previewJob &&
        (previewJob.status === "queued" || previewJob.status === "running")
      ) {
        pendingPreviewRefreshRef.current = true;
        setPreviewUpdateQueued(true);
        setNotice("Preview render in progress. Latest edit will render next.");
        return;
      }
      setQueueingPreview(true);
      setError(null);
      try {
        const previewAspectRatio =
          overrides?.aspectRatio ?? previewFrameAspectRatio;
        const job = await api.renderPreview(project.id, force, {
          // The preview frame is independent from the export preset. This
          // prevents a portrait export crop from appearing in a 16:9 editor
          // preview.
          aspect_ratio: previewAspectRatio,
          fps: overrides?.fps ?? exportFps,
          auto_frame:
            previewAspectRatio === "9:16" &&
            (overrides?.autoFrame ?? autoFraming),
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
    },
    [
      project,
      queueingPreview,
      previewJob,
      previewFrameAspectRatio,
      exportFps,
      autoFraming,
    ],
  );

  const changePreviewFrameAspectRatio = useCallback(
    (ratio: ExportAspectRatio) => {
      if (ratio === previewFrameAspectRatio) return;
      setPreviewFrameAspectRatio(ratio);
      void queuePreview(false, { aspectRatio: ratio });
    },
    [previewFrameAspectRatio, queuePreview],
  );

  const undo = useCallback(async () => {
    const localEntry = undoStack.current[undoStack.current.length - 1];
    if (localEntry?.kind === "selection") {
      undoStack.current.pop();
      redoStack.current.push({
        kind: "selection",
        deletedIds: new Set(deletedWordIds),
      });
      setDeletedWordIds(localEntry.deletedIds);
      syncLocalUndoRedoState();
      setNotice("Word selection restored.");
      return;
    }

    if (project && timelineCanUndo) {
      setError(null);
      try {
        const restored = await api.undo(project.id);
        applyProjectFromServer(restored);
        setNotice("Timeline edit undone.");
        await queuePreview(true);
        return;
      } catch (err) {
        setError((err as Error).message);
        return;
      }
    }

    const cutEntry = undoStack.current.pop();
    if (!cutEntry || cutEntry.kind !== "cut" || !project || !transcript) {
      if (cutEntry) undoStack.current.push(cutEntry);
      syncLocalUndoRedoState();
      return;
    }

    redoStack.current.push({
      kind: "cut",
      transcript,
      timeline: project.timeline,
    });
    syncLocalUndoRedoState();
    setError(null);
    try {
      const restored = await api.restoreTranscriptSnapshot(
        project.id,
        transcript.id,
        {
          words: cutEntry.transcript.words,
          timeline: cutEntry.timeline,
        },
      );
      setTranscript(restored.transcript);
      applyProjectFromServer({ ...project, timeline: restored.timeline });
      setDeletedWordIds(new Set());
      setSelectedWordIds(new Set());
      setAnchorWordId(null);
      lastAppliedSignatureRef.current = "";
      lastAutoCutFailedSignatureRef.current = null;
      setNotice("Transcript cut undone.");
      await refreshBrollSlots(project.id, restored.transcript.id);
      await queuePreview(true);
    } catch (err) {
      setError((err as Error).message);
      undoStack.current.push(cutEntry);
      redoStack.current.pop();
      syncLocalUndoRedoState();
    }
  }, [
    applyProjectFromServer,
    deletedWordIds,
    project,
    queuePreview,
    refreshBrollSlots,
    syncLocalUndoRedoState,
    timelineCanUndo,
    transcript,
  ]);

  const redo = useCallback(async () => {
    const localEntry = redoStack.current[redoStack.current.length - 1];
    if (localEntry?.kind === "selection") {
      redoStack.current.pop();
      undoStack.current.push({
        kind: "selection",
        deletedIds: new Set(deletedWordIds),
      });
      setDeletedWordIds(localEntry.deletedIds);
      syncLocalUndoRedoState();
      setNotice("Word selection restored.");
      return;
    }

    if (project && timelineCanRedo) {
      setError(null);
      try {
        const restored = await api.redo(project.id);
        applyProjectFromServer(restored);
        setNotice("Timeline edit redone.");
        await queuePreview(true);
        return;
      } catch (err) {
        setError((err as Error).message);
        return;
      }
    }

    const cutEntry = redoStack.current.pop();
    if (!cutEntry || cutEntry.kind !== "cut" || !project || !transcript) {
      if (cutEntry) redoStack.current.push(cutEntry);
      syncLocalUndoRedoState();
      return;
    }

    undoStack.current.push({
      kind: "cut",
      transcript,
      timeline: project.timeline,
    });
    syncLocalUndoRedoState();
    setError(null);
    try {
      const restored = await api.restoreTranscriptSnapshot(
        project.id,
        transcript.id,
        {
          words: cutEntry.transcript.words,
          timeline: cutEntry.timeline,
        },
      );
      setTranscript(restored.transcript);
      applyProjectFromServer({ ...project, timeline: restored.timeline });
      setDeletedWordIds(new Set());
      setSelectedWordIds(new Set());
      setAnchorWordId(null);
      setNotice("Transcript cut redone.");
      await refreshBrollSlots(project.id, restored.transcript.id);
      await queuePreview(true);
    } catch (err) {
      setError((err as Error).message);
      redoStack.current.push(cutEntry);
      undoStack.current.pop();
      syncLocalUndoRedoState();
    }
  }, [
    applyProjectFromServer,
    deletedWordIds,
    project,
    queuePreview,
    refreshBrollSlots,
    syncLocalUndoRedoState,
    timelineCanRedo,
    transcript,
  ]);

  const canUndoAction = undoStackSize > 0 || timelineCanUndo;
  const canRedoAction = redoStackSize > 0 || timelineCanRedo;

  const applyTranscriptGenerationResult = useCallback(
    async (response: TranscriptGenerateResponse) => {
      if (!project) return;
      setTranscript(response.transcript);
      setProject((prev) =>
        prev ? { ...prev, timeline: response.timeline } : prev,
      );
      setDeletedWordIds(new Set());
      setSelectedWordIds(new Set());
      setAnchorWordId(null);
      setWeakReviewIndex(0);
      setBrollSlots([]);
      setBrollTimelineActionKey(null);
      setBrollDraftStartById({});
      setBrollDraftDurationById({});
      setBrollDraftOpacityById({});
      setSelectedTimelineClip(null);
      setSelectedBrollClipId(null);
      setSelectedBrollSlotId(null);
      setSelectedCaptionBlock(null);
      lastAppliedSignatureRef.current = "";
      undoStack.current = [];
      redoStack.current = [];
      syncLocalUndoRedoState();
      const reuseNotice = response.reused_transcript
        ? " (Reused cached transcript)"
        : "";
      const sourceText = (response.transcript.source || "").toLowerCase();
      const lyricsNotice = sourceText.includes("lyrics_ref")
        ? " Reference lyrics matched."
        : "";

      setNotice(
        response.transcript.is_mock
          ? `Transcript generated (fallback mode). Install faster-whisper for higher accuracy.${reuseNotice}`
          : `Transcript generated with word timestamps.${lyricsNotice}${reuseNotice}`,
      );
      await refreshBrollSlots(project.id, response.transcript.id);
      await queuePreview();
    },
    [project, refreshBrollSlots, queuePreview],
  );

  async function requestTimelineOperations(
    projectId: string,
    operations: TimelineOperation[],
  ) {
    return timelineMutationCoordinatorRef.current.run(
      projectId,
      () => timelineVersionRef.current,
      (expectedVersion) =>
        api.applyOperations(projectId, operations, expectedVersion),
      (response) => {
        timelineVersionRef.current = response.version;
        setProject((prev) => {
          if (
            !prev ||
            prev.id !== projectId ||
            response.version < (prev.timeline_version ?? 0)
          ) {
            return prev;
          }
          return {
            ...prev,
            timeline: response.timeline,
            timeline_version: response.version,
            timeline_can_undo: response.timeline_can_undo,
            timeline_can_redo: response.timeline_can_redo,
          };
        });
        setTimelineCanUndo(!!response.timeline_can_undo);
        setTimelineCanRedo(!!response.timeline_can_redo);
      },
    );
  }

  async function applyTimelineOperations(
    operations: TimelineOperation[],
    options?: { notice?: string | null; forcePreview?: boolean },
  ) {
    if (!project || !operations.length) return null;
    setError(null);
    try {
      const response = await requestTimelineOperations(project.id, operations);
      if (options?.notice !== undefined) {
        setNotice(options.notice);
      }
      await queuePreview(!!options?.forcePreview);
      return response.timeline;
    } catch (err) {
      if (err instanceof TimelineMutationProjectChangedError) return null;
      setError((err as Error).message);
      return null;
    }
  }

  async function toggleAutoFraming() {
    const nextAutoFraming = !autoFraming;
    if (!nextAutoFraming) {
      setAutoFraming(false);
      await queuePreview(false, { autoFrame: false });
      return;
    }
    if (!project) return;

    setSmartReframing(true);
    setError(null);
    try {
      const response = await api.smartReframe(project.id);
      setProject((prev) =>
        prev
          ? {
              ...prev,
              timeline: response.timeline,
              timeline_version: response.version,
              timeline_can_undo: response.timeline_can_undo,
              timeline_can_redo: response.timeline_can_redo,
            }
          : prev,
      );
      setTimelineCanUndo(response.timeline_can_undo);
      setTimelineCanRedo(response.timeline_can_redo);
      setAutoFraming(true);

      const detail =
        response.tracked_clip_count > 0
          ? ` ${response.tracked_clip_count} clip${response.tracked_clip_count === 1 ? "" : "s"} follow${response.tracked_clip_count === 1 ? "s" : ""} the detected subject.`
          : " Centre framing will be used where no subject track is available.";
      setNotice(
        response.reframed_clip_count > 0
          ? `Auto Frame is on for ${response.reframed_clip_count} wide clip${response.reframed_clip_count === 1 ? "" : "s"}.${detail}`
          : "Auto Frame is on. No wide main-video clips needed a subject crop.",
      );
      await queuePreview(true, { autoFrame: true });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSmartReframing(false);
    }
  }

  const updateDeletedWords = useCallback(
    (wordIds: string[], deleted: boolean) => {
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
    },
    [pushUndo],
  );

  async function applyCut(
    signature: string,
    keptIds: string[],
    options?: { manual?: boolean },
  ) {
    if (!project || !transcript || applyingCut) return;
    const manual = !!options?.manual;
    if (!manual && lastAutoCutFailedSignatureRef.current === signature) {
      return;
    }
    if (!keptIds.length) {
      setError(
        "At least one word must remain. Restore some words before applying.",
      );
      return;
    }

    setApplyingCut(true);
    setError(null);
    const previousDuration = project.timeline.duration_sec;
    const undoEntry: UndoEntry = {
      kind: "cut",
      transcript,
      timeline: project.timeline,
    };
    try {
      const result = await api.applyTranscriptCut(
        project.id,
        transcript.id,
        keptIds,
        {
          contextSec: 0,
          mergeGapSec: 0.08,
          minRemovedSec: 0,
        },
      );
      const refreshedProject = await api.getProject(project.id);
      applyProjectFromServer({
        ...refreshedProject,
        timeline: result.timeline,
      });
      const latestTranscript = await api.getTranscript(
        project.id,
        transcript.id,
      );
      setTranscript(latestTranscript);
      const remainingWordIds = new Set(
        latestTranscript.words.map((word) => word.id),
      );
      const stillPendingDeletedIds = new Set(
        Array.from(latestDeletedWordIdsRef.current).filter((wordId) =>
          remainingWordIds.has(wordId),
        ),
      );
      setDeletedWordIds(stillPendingDeletedIds);
      setSelectedWordIds(new Set());
      setAnchorWordId(null);
      undoStack.current.push(undoEntry);
      if (undoStack.current.length > MAX_UNDO) undoStack.current.shift();
      redoStack.current.push({
        kind: "cut",
        transcript: latestTranscript,
        timeline: result.timeline,
      });
      syncLocalUndoRedoState();
      lastAppliedSignatureRef.current = "";
      const nextDuration = result.timeline.duration_sec;
      const deltaSec = Math.max(0, previousDuration - nextDuration);
      const deltaLabel =
        deltaSec >= 0.01
          ? `Timeline shortened by ${deltaSec.toFixed(2)}s.`
          : "No additional timeline duration change.";
      setNotice(
        `Cut applied. Removed ${result.removed_word_count} word${result.removed_word_count === 1 ? "" : "s"}. ${deltaLabel}`,
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

  async function restoreProjectPreview(
    projectId: string,
    nextProject: Project,
    hasVideoSource: boolean,
  ) {
    try {
      const savedPreview = await api.getLatestProjectPreview(projectId);
      if (savedPreview?.output_path) {
        setPreviewJob(savedPreview);
        setPreviewUrl(resolveMediaPath(savedPreview.output_path));
        return;
      }

      const hasRenderableTimeline = nextProject.timeline.tracks.some(
        (track) => track.kind === "video" && track.clips.length > 0,
      );
      if (!hasVideoSource || !hasRenderableTimeline) return;

      // There is no current render on disk (for example after a deployment),
      // so rebuild it from the persisted timeline rather than showing the
      // unedited source as if it were the saved edit.
      const job = await api.renderPreview(projectId, false, {
        aspect_ratio: previewFrameAspectRatio,
        fps: exportFps,
        auto_frame: autoFraming,
      });
      setPreviewJob(job);
      if (job.status === "completed" && job.output_path) {
        setPreviewUrl(resolveMediaPath(job.output_path));
      }
    } catch {
      // The source-video fallback remains visible. The polling/render UI will
      // surface an actionable error if a newly queued preview later fails.
    }
  }

  async function openProject(projectId: string) {
    if (!projectId || openingProjectId) return;
    setOpeningProjectId(projectId);
    setError(null);
    try {
      const nextProject = await api.getProject(projectId);
      resetEditorStateForProject(nextProject);

      const items = await api.listMedia(projectId);
      setMedia(items);
      const primaryTimelineAssetId = nextProject.timeline.tracks
        .filter((track) => track.kind === "video")
        .flatMap((track) => track.clips)
        .sort((left, right) => left.timeline_start_sec - right.timeline_start_sec)[0]
        ?.asset_id;
      const activeVideo =
        items.find((asset) => asset.id === primaryTimelineAssetId) ??
        items.find((asset) => asset.media_type === "video") ??
        null;
      setSelectedAssetId(activeVideo?.id ?? null);
      setPreviewUrl(
        activeVideo ? resolveMediaPath(activeVideo.storage_path) : null,
      );

      try {
        const latestTranscript = await api.getTranscript(projectId);
        setTranscript(latestTranscript);
        await refreshBrollSlots(projectId, latestTranscript.id);
      } catch {
        setTranscript(null);
        setBrollSlots([]);
      }

      await restoreProjectPreview(projectId, nextProject, !!activeVideo);
      setProjectsPanelOpen(false);
      setNotice(`Opened ${nextProject.name || "project"}.`);
      await refreshProjectList();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setOpeningProjectId(null);
    }
  }

  async function createProject(
    name: string = BRAND.defaultProjectName,
    options?: { silent?: boolean; notice?: string | null },
  ) {
    const silent = !!options?.silent;
    setCreatingProject(true);
    setError(null);
    try {
      const created = await api.createProject(
        name.trim() || "Untitled Project",
      );
      resetEditorStateForProject(created);
      setNotice(
        options?.notice !== undefined
          ? options.notice
          : silent
            ? WORKSPACE_READY_NOTICE
            : "Project created. Upload a video to start.",
      );
      await refreshMedia(created.id);
      await refreshProjectList();
    } catch (err) {
      setError((err as Error).message);
      setNotice(null);
    } finally {
      setCreatingProject(false);
    }
  }

  async function handleRenameProject(projectId: string, newName: string) {
    const trimmed = newName.trim();
    if (!trimmed) return;
    setSubmittingRenameId(projectId);
    try {
      const updated = await api.renameProject(projectId, trimmed);
      // Update recent projects list in-place
      setRecentProjects((prev) =>
        prev.map((p) =>
          p.id === projectId ? { ...p, name: updated.name } : p,
        ),
      );
      // If this is the currently open project, update its name
      if (project && project.id === projectId) {
        setProject((prev) => (prev ? { ...prev, name: updated.name } : prev));
      }
      setNotice(`Renamed to "${updated.name}".`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmittingRenameId(null);
    }
  }

  async function handleDeleteProject(projectId: string) {
    setDeletingProjectId(projectId);
    try {
      await api.deleteProject(projectId);
      setRecentProjects((prev) => prev.filter((p) => p.id !== projectId));
      const wasCurrentProject = project && project.id === projectId;
      if (wasCurrentProject) {
        // Clear current project state without auto-creating a new one
        setProject(null);
        setMedia([]);
        setTranscript(null);
        setBrollSlots([]);
        setSelectedAssetId(null);
        setPreviewUrl(null);
      }
      // Refresh the list from the backend to get the true count
      const updatedProjects = await api.listProjects();
      setRecentProjects(updatedProjects);
      if (wasCurrentProject) {
        if (updatedProjects.length > 0) {
          // Switch to the first remaining project instead of creating a new one
          await openProject(updatedProjects[0].id);
          setNotice("Project deleted. Switched to next project.");
        } else {
          // No projects left – show the welcome screen; the user creates the
          // next project explicitly.
          setNotice("Project deleted.");
        }
      } else {
        setNotice("Project deleted.");
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setDeletingProjectId(null);
    }
  }

  async function attachVideoAsset(uploaded: MediaAsset): Promise<boolean> {
    if (!project) return false;
    setMedia((prev) =>
      prev.some((item) => item.id === uploaded.id) ? prev : [uploaded, ...prev],
    );
    let addedToTimeline = false;
    if (uploaded.media_type === "video") {
        setSelectedAssetId(uploaded.id);
        setPreviewUrl(resolveMediaPath(uploaded.storage_path));
        const durationSec = uploaded.duration_sec ?? 0;
        const hasVideoClip = project.timeline.tracks.some(
          (track) => track.kind === "video" && (track.clips ?? []).length > 0,
        );
        if (!hasVideoClip && Number.isFinite(durationSec) && durationSec > 0) {
          try {
            await requestTimelineOperations(project.id, [
              {
                op_type: "add_clip",
                source: "ui",
                params: {
                  asset_id: uploaded.id,
                  start_sec: 0,
                  duration_sec: durationSec,
                  timeline_start_sec: 0,
                },
              },
            ]);
            addedToTimeline = true;
          } catch (timelineError) {
            if (!(timelineError instanceof TimelineMutationProjectChangedError)) {
              setError(
                `Video uploaded, but timeline setup failed: ${(timelineError as Error).message}`,
              );
            }
          }
        }
      }
    return addedToTimeline;
  }

  async function uploadVideo(file: File) {
    if (!project) return;
    setUploading(true);
    setError(null);
    setQuickEditSummary(null);
    setExportCompletion(null);
    try {
      const uploaded = await api.uploadMedia(project.id, file);
      const addedToTimeline = await attachVideoAsset(uploaded);
      setNotice(
        addedToTimeline ? "Video uploaded. Timeline ready." : "Video uploaded.",
      );
    } catch (err) {
      setError((err as Error).message);
      setNotice(null);
    } finally {
      setUploading(false);
    }
  }

  async function ingestVideoFromUrl(url: string) {
    if (!project || ingestingUrl) return;
    setIngestingUrl(true);
    setIngestProgress(5);
    setIngestStatusMessage("Starting video fetch...");
    setError(null);
    setQuickEditSummary(null);
    setExportCompletion(null);
    const knownAssetIds = new Set(media.map((item) => item.id));
    try {
      setNotice("Fetching video from URL...");
      let job = await api.ingestUrl(project.id, url.trim());
      setIngestProgress(Math.max(5, Math.round(job.progress ?? 5)));
      setIngestStatusMessage(job.message || "Queued for download...");
      while (job.status === "queued" || job.status === "running") {
        await new Promise((resolve) => window.setTimeout(resolve, 1000));
        job = await api.getJob(job.id);
        const pct = Math.max(0, Math.min(99, Math.round(job.progress ?? 0)));
        setIngestProgress(pct);
        const statusText =
          job.message?.trim() ||
          (job.status === "queued"
            ? "Waiting in queue..."
            : "Fetching video...");
        setIngestStatusMessage(statusText);
        setNotice(`${statusText} ${pct}%`);
      }
      if (job.status !== "completed") {
        throw new Error(job.error ?? job.message ?? "URL ingestion failed.");
      }
      setIngestProgress(100);
      setIngestStatusMessage(job.message || "Video ready");
      const assets = await api.listMedia(project.id);
      const ingested =
        assets.find(
          (item) => !knownAssetIds.has(item.id) && item.media_type === "video",
        ) ?? assets.find((item) => item.media_type === "video");
      if (!ingested) {
        throw new Error("Ingestion finished but no video was found.");
      }
      const addedToTimeline = await attachVideoAsset(ingested);
      setNotice(
        addedToTimeline
          ? "Video added from URL. Timeline ready."
          : "Video added from URL.",
      );
    } catch (err) {
      setError((err as Error).message);
      setNotice(null);
      setIngestStatusMessage("");
      setIngestProgress(0);
    } finally {
      setIngestingUrl(false);
    }
  }

  async function generateTranscript(options?: { forceRegenerate?: boolean }) {
    if (!project || !selectedVideoAsset || generatingTranscript) {
      return;
    }
    const forceRegenerate = !!options?.forceRegenerate;
    lastTranscriptRequestRef.current = { forceRegenerate };
    const startedAtMs = Date.now();
    setGeneratingTranscript(true);
    setTranscriptStartedAtMs(startedAtMs);
    setTranscriptElapsedSec(0);
    setTranscriptStageStartedAtMs(startedAtMs);
    setTranscriptStageBaseProgress(0);
    transcriptStageKeyRef.current = null;
    setTranscriptJob(null);
    transcriptJobResultHandledRef.current = null;
    setError(null);
    setNotice(
      forceRegenerate
        ? "Regenerating transcript from scratch..."
        : "Transcript generation started.",
    );
    try {
      const language =
        transcriptLanguage === "auto" ? undefined : transcriptLanguage;
      const job = await api.generateTranscriptAsync(
        project.id,
        selectedVideoAsset.id,
        transcriptMode,
        language,
        undefined,
        false,
        { forceRegenerate, speed: transcriptSpeed },
      );
      setTranscriptJob(job);
      if (job.status === "completed") {
        transcriptJobResultHandledRef.current = job.id;
        const response = await api.getTranscriptGenerateResult(
          project.id,
          job.id,
        );
        await applyTranscriptGenerationResult(response);
        setGeneratingTranscript(false);
        setTranscriptStartedAtMs(null);
        setTranscriptStageStartedAtMs(null);
        setTranscriptStageBaseProgress(0);
      }
    } catch (err) {
      setError((err as Error).message);
      setNotice(null);
      setGeneratingTranscript(false);
      setTranscriptStartedAtMs(null);
      setTranscriptStageStartedAtMs(null);
      setTranscriptStageBaseProgress(0);
    } finally {
      setTranscriptElapsedSec(
        Math.max(0, Math.floor((Date.now() - startedAtMs) / 1000)),
      );
    }
  }

  function retryTranscriptGeneration() {
    void generateTranscript(lastTranscriptRequestRef.current);
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
            : "Generating transcript and applying captions. This may take over a minute for longer videos.",
        );
        const selectedStyle =
          CAPTION_STYLE_PRESETS.find((item) => item.id === captionStyle) ??
          CAPTION_STYLE_PRESETS[0];
        options.style = selectedStyle.id;
        options.caption_styles = CAPTION_STYLE_CONFIG_BY_ID;
        if (transcriptLanguage !== "auto") {
          options.transcript_language = transcriptLanguage;
        }
        if (transcriptHasRomanization && showRomanizedTranscript) {
          options.transliterate = true;
        }
        if (transcript?.words?.length) {
          // Use exactly what user sees in transcript UI, including an in-progress inline edit.
          // Filter out deleted words so captions are only generated for kept words.
          const pendingEditText = editingWordText.trim();
          const subtitleWords: TranscriptWord[] = (
            editingWordId && pendingEditText
              ? transcript.words.map((word) =>
                  word.id === editingWordId
                    ? { ...word, text: pendingEditText }
                    : word,
                )
              : transcript.words
          ).filter((word) => !deletedWordIds.has(word.id));
          options.words = subtitleWords;
        }
      }
      const response = await api.applyVibeAction(
        project.id,
        action,
        selectedVideoAsset.id,
        options,
      );
      setProject((prev) =>
        prev ? { ...prev, timeline: response.timeline } : prev,
      );
      if (action === "add_subtitles") {
        const firstCaptionTimeSec = findFirstCaptionTimeSec(response.timeline);
        if (firstCaptionTimeSec !== null) {
          pendingCaptionSeekRef.current = {
            jobId: response.preview_job.id,
            targetSec: firstCaptionTimeSec,
          };
        }
        setCaptionResultInfo(response.details ?? null);
      }
      setPreviewJob(response.preview_job);
      if (response.preview_job.output_path) {
        setPreviewUrl(resolveMediaPath(response.preview_job.output_path));
      }
      if (response.transcript_id) {
        const latestTranscript = await api.getTranscript(
          project.id,
          response.transcript_id,
        );
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
      await requestTimelineOperations(project.id, [
        {
          op_type: "clear_subtitles",
          params: { asset_id: selectedVideoAsset.id },
          source: "ui",
        },
      ]);
      setSelectedCaptionBlock(null);
      setCaptionResultInfo(null);
      setNotice("Captions removed.");
      await queuePreview(true);
    } catch (err) {
      if (!(err instanceof TimelineMutationProjectChangedError)) {
        setError((err as Error).message);
      }
    } finally {
      setRemovingCaptions(false);
    }
  }

  const currentExportSettings = useCallback(
    (): ExportSettingsSnapshot => ({
      format: exportFormat,
      aspectRatio: exportAspectRatio,
      resolution: exportResolution,
      fps: exportFps,
      quality: exportQuality,
      autoFrame: autoFraming,
    }),
    [
      exportAspectRatio,
      autoFraming,
      exportFormat,
      exportFps,
      exportQuality,
      exportResolution,
    ],
  );

  const exportRuntimeHint = useMemo(
    () =>
      estimateExportRuntimeLabel(
        currentExportSettings(),
        project?.timeline.duration_sec ?? selectedAssetDurationSec,
      ),
    [
      currentExportSettings,
      project?.timeline.duration_sec,
      selectedAssetDurationSec,
    ],
  );

  const handleCompletedExport = useCallback(
    async (job: Job) => {
      if (exportCompletionHandledRef.current === job.id) return;
      exportCompletionHandledRef.current = job.id;

      const settings = exportSettingsRef.current ?? currentExportSettings();
      const fallbackFilename = fallbackExportFilename(settings);
      let filename = fallbackFilename;
      let downloadError: string | null = null;

      if (job.output_path) {
        try {
          filename = await api.downloadJobOutput(job.id, fallbackFilename);
          setNotice(`Export completed. Downloaded ${filename}.`);
        } catch (err) {
          downloadError = (err as Error).message;
          setError(downloadError);
          setNotice("Export completed. Download is ready to retry.");
        }
      } else {
        setNotice("Export completed.");
      }

      setExportCompletion({
        ...settings,
        jobId: job.id,
        filename,
        outputPath: job.output_path,
        downloadError,
      });
    },
    [currentExportSettings],
  );

  async function downloadCompletedExport() {
    if (
      !exportCompletion?.jobId ||
      !exportCompletion.outputPath ||
      downloadingExport
    )
      return;
    setDownloadingExport(true);
    setError(null);
    try {
      const filename = await api.downloadJobOutput(
        exportCompletion.jobId,
        fallbackExportFilename(exportCompletion),
      );
      setExportCompletion((prev) =>
        prev ? { ...prev, filename, downloadError: null } : prev,
      );
      setNotice(`Downloaded ${filename}.`);
    } catch (err) {
      const message = (err as Error).message;
      setExportCompletion((prev) =>
        prev ? { ...prev, downloadError: message } : prev,
      );
      setError(message);
    } finally {
      setDownloadingExport(false);
    }
  }

  async function exportVideo() {
    if (!project || exportingVideo) return;
    const settings = currentExportSettings();
    exportSettingsRef.current = settings;
    exportCompletionHandledRef.current = null;
    setExportingVideo(true);
    setExportCompletion(null);
    setError(null);
    try {
      const job = await api.renderExport(project.id, {
        format: settings.format,
        aspect_ratio: settings.aspectRatio,
        resolution: settings.resolution,
        fps: settings.fps,
        quality: settings.quality,
        auto_frame: settings.aspectRatio === "9:16" && settings.autoFrame,
      });
      setExportJob(job);
      setNotice(
        `Export started (${settings.aspectRatio}, ${settings.resolution}, ${settings.format}). Job ID: ${job.id}`,
      );
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setExportingVideo(false);
    }
  }

  // ── Quick Edit: one-click workflow (state-machine, no blocking loops) ──
  // Phase 1: start transcription (or skip if transcript exists), set phase to "transcribing"
  // Phase 2: useEffect below watches for transcript completion → runs auto-cut + captions inline
  function quickEdit() {
    if (!project || !selectedVideoAsset || quickEditing) return;
    setError(null);
    setQuickEditSummary(null);

    if (transcript?.words?.length) {
      // Transcript already exists — skip straight to cut+captions
      quickEditPhaseRef.current = "cutting";
      setQuickEditing(true);
      setQuickEditStage("Removing pauses & filler...");
      void runQuickEditCutAndCaptions();
      return;
    }

    // Need to generate transcript first
    quickEditPhaseRef.current = "transcribing";
    setQuickEditing(true);
    setQuickEditStage("Generating transcript...");
    void generateTranscript();
  }

  // Called once transcript is available (either pre-existing or freshly generated)
  async function runQuickEditCutAndCaptions() {
    if (!project || !selectedVideoAsset) {
      quickEditPhaseRef.current = "idle";
      setQuickEditing(false);
      setQuickEditStage("");
      return;
    }
    let cutDetails: string | null = null;
    let captionDetails: string | null = null;
    let finalTimeline: ProjectTimeline = project.timeline;
    try {
      // Step A: Auto-cut pauses (fire-and-forget preview; returns quickly)
      quickEditPhaseRef.current = "cutting";
      setQuickEditStage("Removing pauses & filler...");
      try {
        const cutResponse = await api.applyVibeAction(
          project.id,
          "auto_cut_pauses",
          selectedVideoAsset.id,
          {},
        );
        cutDetails = cutResponse.details ?? "Auto-cut applied.";
        finalTimeline = cutResponse.timeline;
        setProject((prev) =>
          prev ? { ...prev, timeline: cutResponse.timeline } : prev,
        );
        // Accept the queued preview job from the backend
        setPreviewJob(cutResponse.preview_job);
        if (cutResponse.preview_job?.output_path) {
          setPreviewUrl(resolveMediaPath(cutResponse.preview_job.output_path));
        }
        if (cutResponse.transcript_id) {
          const latestTranscript = await api.getTranscript(
            project.id,
            cutResponse.transcript_id,
          );
          setTranscript(latestTranscript);
        }
      } catch {
        // Auto-cut is non-fatal; some videos have no meaningful pauses
        cutDetails = "Auto-cut skipped: no significant pauses detected.";
        setNotice(cutDetails);
      }

      // Step B: Add captions with the selected style
      quickEditPhaseRef.current = "captioning";
      setQuickEditStage("Adding captions...");
      try {
        const selectedStyle =
          CAPTION_STYLE_PRESETS.find((item) => item.id === captionStyle) ??
          CAPTION_STYLE_PRESETS[0];
        const captionOptions: Record<string, unknown> = {
          style: selectedStyle.id,
          caption_styles: CAPTION_STYLE_CONFIG_BY_ID,
        };
        if (transcriptLanguage !== "auto") {
          captionOptions.transcript_language = transcriptLanguage;
        }
        const captionResponse = await api.applyVibeAction(
          project.id,
          "add_subtitles",
          selectedVideoAsset.id,
          captionOptions,
        );
        captionDetails = captionResponse.details ?? "Captions added.";
        finalTimeline = captionResponse.timeline;
        setProject((prev) =>
          prev ? { ...prev, timeline: captionResponse.timeline } : prev,
        );
        setPreviewJob(captionResponse.preview_job);
        if (captionResponse.preview_job?.output_path) {
          setPreviewUrl(
            resolveMediaPath(captionResponse.preview_job.output_path),
          );
        }
        setCaptionResultInfo(captionResponse.details ?? null);
      } catch {
        captionDetails =
          "Captions skipped: generate a transcript first or retry.";
        setNotice(captionDetails);
      }

      quickEditPhaseRef.current = "done";
      setQuickEditStage("");
      const summary = buildQuickEditSummary(
        cutDetails,
        captionDetails,
        finalTimeline,
      );
      setQuickEditSummary(summary);
      setNotice(
        summary.captionsAdded
          ? "Quick Edit complete: transcript cut and captions applied. Add B-roll separately in B-roll Studio when ready."
          : "Quick Edit complete: transcript cut applied. Captions need another try.",
      );
    } catch (err) {
      setError((err as Error).message);
      setQuickEditStage("");
      quickEditPhaseRef.current = "idle";
    } finally {
      setQuickEditing(false);
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
    setNotice(
      `Marked ${fillerWordIds.size} filler word${fillerWordIds.size === 1 ? "" : "s"} as deleted.`,
    );
  }, [fillerWordIds, updateDeletedWords]);

  const selectWord = useCallback(
    (wordId: string, shiftHeld: boolean) => {
      if (!transcript) return;
      setSelectedTimelineClip(null);
      setSelectedBrollClipId(null);
      setSelectedBrollSlotId(null);
      setSelectedCaptionBlock(null);
      if (
        !shiftHeld ||
        !anchorWordId ||
        !transcriptWordIndex.has(anchorWordId) ||
        !transcriptWordIndex.has(wordId)
      ) {
        setAnchorWordId(wordId);
        setSelectedWordIds(new Set([wordId]));
        return;
      }

      const anchorIndex = transcriptWordIndex.get(anchorWordId) ?? 0;
      const currentIndex = transcriptWordIndex.get(wordId) ?? 0;
      const minIndex = Math.min(anchorIndex, currentIndex);
      const maxIndex = Math.max(anchorIndex, currentIndex);
      const range = transcript.words
        .slice(minIndex, maxIndex + 1)
        .map((word) => word.id);
      setSelectedWordIds(new Set(range));
    },
    [transcript, anchorWordId, transcriptWordIndex],
  );

  const selectWordRange = useCallback(
    (fromId: string, toId: string) => {
      if (!transcript) return;
      setSelectedTimelineClip(null);
      setSelectedBrollClipId(null);
      setSelectedBrollSlotId(null);
      setSelectedCaptionBlock(null);
      const fromIdx = transcriptWordIndex.get(fromId) ?? 0;
      const toIdx = transcriptWordIndex.get(toId) ?? 0;
      const minIdx = Math.min(fromIdx, toIdx);
      const maxIdx = Math.max(fromIdx, toIdx);
      const range = transcript.words.slice(minIdx, maxIdx + 1).map((w) => w.id);
      setSelectedWordIds(new Set(range));
    },
    [transcript, transcriptWordIndex],
  );

  const toggleBlock = useCallback(
    async (block: TextBlock) => {
      const allDeleted = block.wordIds.every((id) => deletedWordIds.has(id));
      if (allDeleted) {
        // Restoring a deleted block — use the undo-based path
        updateDeletedWords(block.wordIds, false);
        return;
      }
      // Deleting the block — directly call the backend so the video timeline
      // is updated alongside the transcript panel.
      if (!project || !transcript || updatingTranscriptRange) return;
      const firstWordId = block.wordIds[0];
      const lastWordId = block.wordIds[block.wordIds.length - 1];
      if (!firstWordId || !lastWordId) return;
      pushUndo();
      setUpdatingTranscriptRange(true);
      setError(null);
      try {
        const updated = await api.updateTranscriptRange(
          transcript.id,
          project.id,
          {
            start_word_id: firstWordId,
            end_word_id: lastWordId,
            mode: "delete",
          },
        );
        setTranscript(updated.transcript);
        setProject((prev) =>
          prev ? { ...prev, timeline: updated.timeline } : prev,
        );
        // Clean up any local deleted/selected state for the removed words
        setDeletedWordIds((prev) => {
          const next = new Set(prev);
          block.wordIds.forEach((id) => next.delete(id));
          return next;
        });
        setSelectedWordIds((prev) => {
          const next = new Set(prev);
          block.wordIds.forEach((id) => next.delete(id));
          return next;
        });
        lastAppliedSignatureRef.current = "";
        undoStack.current = [];
        redoStack.current = [];
        setNotice(
          `Removed ${block.wordIds.length} word${block.wordIds.length === 1 ? "" : "s"} from transcript and video.`,
        );
        await refreshBrollSlots(project.id, updated.transcript.id);
        await queuePreview(true);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setUpdatingTranscriptRange(false);
      }
    },
    [
      deletedWordIds,
      updateDeletedWords,
      project,
      transcript,
      updatingTranscriptRange,
      pushUndo,
      refreshBrollSlots,
      queuePreview,
    ],
  );

  const seekToWord = useCallback(
    (word: TranscriptWord) => {
      if (!videoRef.current) return;
      const targetSec = previewRenderBusy
        ? Math.max(0, word.start_sec)
        : mapSourceTimeToEditedTime(word.start_sec, videoClips);
      videoRef.current.currentTime = targetSec;
      setCurrentTimeSec(targetSec);
    },
    [previewRenderBusy, videoClips],
  );

  const seekToTranscriptTime = useCallback(
    (sourceSec: number) => {
      if (!videoRef.current) return;
      const targetSec = previewRenderBusy
        ? Math.max(0, sourceSec)
        : mapSourceTimeToEditedTime(sourceSec, videoClips);
      videoRef.current.currentTime = targetSec;
      setCurrentTimeSec(targetSec);
    },
    [previewRenderBusy, videoClips],
  );

  const scrollTranscriptWordIntoView = useCallback((wordId: string) => {
    window.setTimeout(() => {
      document
        .getElementById(`word-${wordId}`)
        ?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 0);
  }, []);

  const focusTranscriptWordIds = useCallback(
    (wordIds: string[], fallbackSourceSec?: number, noticeMessage?: string) => {
      if (!transcript) {
        if (typeof fallbackSourceSec === "number")
          seekToTranscriptTime(fallbackSourceSec);
        return;
      }
      const validIds = wordIds.filter((wordId) =>
        transcriptWordIndex.has(wordId),
      );
      setSelectedTimelineClip(null);
      setSelectedCaptionBlock(null);
      if (!validIds.length) {
        if (typeof fallbackSourceSec === "number")
          seekToTranscriptTime(fallbackSourceSec);
        if (noticeMessage) setNotice(noticeMessage);
        return;
      }
      const firstWord =
        transcript.words[transcriptWordIndex.get(validIds[0]) ?? 0];
      setSelectedWordIds(new Set(validIds));
      setAnchorWordId(validIds[0]);
      if (firstWord) {
        seekToWord(firstWord);
        scrollTranscriptWordIntoView(firstWord.id);
      }
      if (noticeMessage) setNotice(noticeMessage);
    },
    [
      scrollTranscriptWordIntoView,
      seekToTranscriptTime,
      seekToWord,
      transcript,
      transcriptWordIndex,
    ],
  );

  const focusTranscriptRegion = useCallback(
    (region: TranscriptRegion) => {
      const regionWordIds = region.word_ids?.length
        ? region.word_ids
        : (transcript?.words ?? [])
            .filter(
              (word) =>
                word.start_sec < region.end_sec &&
                word.end_sec > region.start_sec,
            )
            .map((word) => word.id);
      focusTranscriptWordIds(
        regionWordIds,
        region.start_sec,
        `${transcriptRegionLabel(region)} transcript region selected.`,
      );
    },
    [focusTranscriptWordIds, transcript?.words],
  );

  const reviewWeakWords = useCallback(
    (index: number) => {
      if (!transcriptReviewWordIds.length) return;
      const boundedIndex =
        ((index % transcriptReviewWordIds.length) +
          transcriptReviewWordIds.length) %
        transcriptReviewWordIds.length;
      const wordId = transcriptReviewWordIds[boundedIndex];
      setWeakReviewIndex(boundedIndex);
      focusTranscriptWordIds(
        [wordId],
        undefined,
        `Reviewing word ${boundedIndex + 1} of ${transcriptReviewWordIds.length}.`,
      );
    },
    [focusTranscriptWordIds, transcriptReviewWordIds],
  );

  const reviewNextWeakWord = useCallback(() => {
    if (!transcriptReviewWordIds.length) return;
    const selectedReviewId = Array.from(selectedWordIds).find((wordId) =>
      transcriptReviewWordIds.includes(wordId),
    );
    const currentIndex = selectedReviewId
      ? transcriptReviewWordIds.indexOf(selectedReviewId)
      : weakReviewIndex - 1;
    reviewWeakWords(currentIndex + 1);
  }, [
    reviewWeakWords,
    selectedWordIds,
    transcriptReviewWordIds,
    weakReviewIndex,
  ]);

  const focusBrollSlot = useCallback(
    (slot: BrollSlot, noticeMessage?: string) => {
      setSelectedBrollSlotId(slot.id);
      const wordIds = slot.anchor_word_ids.length
        ? slot.anchor_word_ids
        : (transcript?.words ?? [])
            .filter(
              (word) =>
                word.start_sec < slot.end_sec && word.end_sec > slot.start_sec,
            )
            .map((word) => word.id);
      focusTranscriptWordIds(
        wordIds,
        slot.start_sec,
        noticeMessage ?? "B-roll transcript region selected.",
      );
    },
    [focusTranscriptWordIds, transcript?.words],
  );

  // ── Inline editing ─────────────────────────────────────────────────
  const startEditing = useCallback((word: TranscriptWord) => {
    setEditingWordId(word.id);
    setEditingWordText(word.text);
    setTimeout(() => editInputRef.current?.focus(), 0);
  }, []);

  const commitEdit = useCallback(async () => {
    if (!editingWordId || !transcript) {
      setEditingWordId(null);
      return;
    }
    if (committingWordEditRef.current) {
      return;
    }
    committingWordEditRef.current = true;
    const trimmed = editingWordText.trim();
    const projectId = project?.id ?? null;
    const transcriptId = transcript.id;
    const wordId = editingWordId;
    try {
      if (!trimmed) {
        setEditingWordId(null);
        setEditingWordText("");
        if (!projectId) {
          return;
        }
        setUpdatingTranscriptRange(true);
        setError(null);
        try {
          const updated = await api.updateTranscriptRange(
            transcriptId,
            projectId,
            {
              start_word_id: wordId,
              end_word_id: wordId,
              mode: "delete",
            },
          );
          setTranscript(updated.transcript);
          setProject((prev) =>
            prev ? { ...prev, timeline: updated.timeline } : prev,
          );
          setDeletedWordIds((prev) => {
            if (!prev.has(wordId)) return prev;
            const next = new Set(prev);
            next.delete(wordId);
            return next;
          });
          setSelectedWordIds((prev) => {
            if (!prev.has(wordId)) return prev;
            const next = new Set(prev);
            next.delete(wordId);
            return next;
          });
          setAnchorWordId((prev) => (prev === wordId ? null : prev));
          undoStack.current = [];
          redoStack.current = [];
          setNotice("Word removed from transcript.");
          await refreshBrollSlots(projectId, updated.transcript.id);
          await queuePreview(true);
        } catch (err) {
          setError((err as Error).message);
        } finally {
          setUpdatingTranscriptRange(false);
        }
        return;
      }

      // Update word text locally
      setTranscript((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          words: prev.words.map((w) =>
            w.id === wordId
              ? {
                  ...w,
                  text: trimmed,
                  display_text: null,
                  quality_label: "trusted",
                  quality_score: 1,
                  source_pass: "manual",
                }
              : w,
          ),
          text: prev.words
            .map((w) => (w.id === wordId ? trimmed : w.text))
            .join(" "),
        };
      });
      setEditingWordId(null);
      setEditingWordText("");
      if (!projectId) {
        return;
      }
      try {
        const updated = await api.updateWordText(
          transcriptId,
          wordId,
          trimmed,
          projectId,
        );
        setTranscript(updated.transcript);
        setProject((prev) =>
          prev ? { ...prev, timeline: updated.timeline } : prev,
        );
        await refreshBrollSlots(projectId, updated.transcript.id);
        await queuePreview(true);
      } catch (err) {
        setError((err as Error).message);
        setNotice("Word text updated locally, but transcript sync failed.");
      }
    } finally {
      committingWordEditRef.current = false;
    }
  }, [
    editingWordId,
    transcript,
    editingWordText,
    project?.id,
    refreshBrollSlots,
    queuePreview,
  ]);

  const cancelEdit = useCallback(() => {
    setEditingWordId(null);
    setEditingWordText("");
  }, []);

  const applyTranscriptRangeUpdate = useCallback(
    async (mode: "delete") => {
      if (
        !project ||
        !transcript ||
        !selectedTranscriptRange ||
        updatingTranscriptRange
      )
        return;
      setUpdatingTranscriptRange(true);
      setError(null);
      try {
        const updated = await api.updateTranscriptRange(
          transcript.id,
          project.id,
          {
            start_word_id: selectedTranscriptRange.startWordId,
            end_word_id: selectedTranscriptRange.endWordId,
            mode,
          },
        );
        setTranscript(updated.transcript);
        setProject((prev) =>
          prev ? { ...prev, timeline: updated.timeline } : prev,
        );
        setDeletedWordIds(new Set());
        setSelectedWordIds(new Set());
        setAnchorWordId(null);
        undoStack.current = [];
        redoStack.current = [];
        setNotice("Deleted selected transcript text.");
        await refreshBrollSlots(project.id, updated.transcript.id);
        await queuePreview(true);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setUpdatingTranscriptRange(false);
      }
    },
    [
      project,
      transcript,
      selectedTranscriptRange,
      updatingTranscriptRange,
      refreshBrollSlots,
      queuePreview,
    ],
  );

  // Build a set of word IDs that start new sentences (after the first sentence)
  const sentenceStartIds = useMemo(() => {
    const ids = new Set<string>();
    sentenceBlocks.forEach((block, idx) => {
      if (idx > 0 && block.wordIds.length > 0) {
        ids.add(block.wordIds[0]);
      }
    });
    return ids;
  }, [sentenceBlocks]);

  const speakerLegend = useMemo(
    () => (transcript ? buildSpeakerLegend(transcript.words) : []),
    [transcript],
  );
  const speakerIds = useMemo(
    () => speakerLegend.map((entry) => entry.speakerId),
    [speakerLegend],
  );
  const selectedAssetLooksLikeDuet = useMemo(
    () =>
      Boolean(
        selectedVideoAsset?.filename &&
        looksLikeDuetFilename(selectedVideoAsset.filename),
      ),
    [selectedVideoAsset],
  );

  const transcriptWordNodes = useMemo(() => {
    if (!transcript) return null;

    const nodes: React.ReactNode[] = [];
    transcript.words.forEach((word) => {
      // Insert a visual line break before sentence-starting words
      if (sentenceStartIds.has(word.id)) {
        nodes.push(
          <span
            key={`brk-${word.id}`}
            className="sentenceBreak"
            aria-hidden="true"
          />,
        );
      }

      const isDeleted = deletedWordIds.has(word.id);
      const displayText = transcriptDisplayText(word, showRomanizedTranscript);

      if (editingWordId === word.id) {
        nodes.push(
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
          />,
        );
        return;
      }

      const isSelected = selectedWordIds.has(word.id);
      const isActive =
        activeWordId === word.id && (!isDeleted || previewRenderBusy);
      const isFiller = fillerWordIds.has(word.id) && !isDeleted;
      const isSearchMatch = searchMatchIdSet.has(word.id);
      const isCurrentMatch = searchMatchIds[searchMatchIndex] === word.id;
      const hasLowConfidence =
        !isDeleted &&
        typeof word.confidence === "number" &&
        word.confidence < LOW_CONFIDENCE_THRESHOLD;
      const isWeakRegionWord =
        !isDeleted &&
        (word.quality_label === "weak" ||
          transcriptIssueWordIdSet.has(word.id));
      const speakerSlot = speakerSlotForWord(word, speakerIds);

      nodes.push(
        <TranscriptWordButton
          key={word.id}
          word={word}
          displayText={displayText}
          showRomanized={showRomanizedTranscript}
          isDeleted={isDeleted}
          isSelected={isSelected}
          isActive={isActive}
          isFiller={isFiller}
          isSearchMatch={isSearchMatch}
          isCurrentMatch={isCurrentMatch}
          hasLowConfidence={hasLowConfidence}
          isWeakRegionWord={isWeakRegionWord}
          speakerSlot={speakerSlot}
          activeWordRef={activeWordRef}
          isDraggingRef={isDragging}
          dragStartWordIdRef={dragStartWordId}
          selectWord={selectWord}
          seekToWord={seekToWord}
          selectWordRange={selectWordRange}
          startEditing={startEditing}
          formatPreciseSeconds={formatPreciseSeconds}
        />,
      );
    });
    return nodes;
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
    showRomanizedTranscript,
    sentenceStartIds,
    speakerIds,
    transcriptIssueWordIdSet,
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
    timelineClockRef.current.setTime(sec);
    setCurrentTimeSec(sec);
  }, []);

  const clearEditorSelections = useCallback(() => {
    setSelectedTimelineClip(null);
    setSelectedBrollClipId(null);
    setSelectedBrollSlotId(null);
    setSelectedCaptionBlock(null);
  }, []);

  const handleTimelineSelectWord = useCallback(
    (id: string, shift: boolean) => {
      if (!transcript) return;
      selectWord(id, shift);
      const wd = transcript.words.find((w) => w.id === id);
      if (wd) seekToWord(wd);
    },
    [transcript, selectWord, seekToWord],
  );

  const handleTimelineSelectWordsInRange = useCallback(
    (startSec: number, endSec: number) => {
      if (!transcript) return;
      const ids = selectTranscriptWordIdsInRange(
        timelineAssistWords,
        startSec,
        endSec,
      );
      setSelectedTimelineClip(null);
      setSelectedBrollClipId(null);
      setSelectedCaptionBlock(null);
      setSelectedWordIds(new Set(ids));
    },
    [timelineAssistWords, transcript],
  );

  const handleTimelineMoveLaneClip = useCallback(
    (
      selection: TimelineLaneClipSelection,
      timelineStartSec: number,
      dropTarget?: TimelineDropTarget,
    ) => {
      if (lockedLaneIds.has(selection.laneId)) return;
      const crossTrack = !!dropTarget && dropTarget.laneId !== selection.laneId;
      const destinationKind = crossTrack ? dropTarget.kind : selection.laneKind;
      void applyTimelineOperations(
        [
          {
            op_type: "move_clip",
            params: {
              clip: selection.clipId,
              track_kind: destinationKind,
              ...(crossTrack && dropTarget.kind !== "overlay"
                ? { track_id: dropTarget.laneId }
                : {}),
              timeline_start_sec: Number(
                Math.max(0, timelineStartSec).toFixed(3),
              ),
              // The overlay (B-roll) track keeps clips freely positioned;
              // video/audio tracks stay packed.
              ripple: destinationKind !== "overlay",
              source_ripple: crossTrack,
            },
            source: "ui",
          },
        ],
        {
          notice: crossTrack
            ? `Clip moved to ${dropTarget.label}.`
            : `${selection.laneLabel} clip moved.`,
        },
      ).then(() => {
        if (crossTrack) setSelectedTimelineClip(null);
      });
    },
    [applyTimelineOperations, lockedLaneIds],
  );

  const handleTimelineMoveBrollClipToLane = useCallback(
    (
      clipId: string,
      timelineStartSec: number,
      dropTarget: TimelineDropTarget,
    ) => {
      if (dropTarget.kind === "overlay") return;
      if (lockedLaneIds.has(dropTarget.laneId)) return;
      void applyTimelineOperations(
        [
          {
            op_type: "move_clip",
            params: {
              clip: clipId,
              track_kind: dropTarget.kind,
              track_id: dropTarget.laneId,
              timeline_start_sec: Number(
                Math.max(0, timelineStartSec).toFixed(3),
              ),
              ripple: true,
              source_ripple: false,
            },
            source: "ui",
          },
        ],
        { notice: `B-roll clip moved to ${dropTarget.label}.` },
      ).then(() => setSelectedBrollClipId(null));
    },
    [applyTimelineOperations, lockedLaneIds],
  );

  const handleTimelineTrimLaneClip = useCallback(
    (
      selection: TimelineLaneClipSelection,
      nextRange: { startSec: number; endSec: number },
    ) => {
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
        { notice: `${selection.laneLabel} clip trimmed.` },
      );
    },
    [applyTimelineOperations, lockedLaneIds],
  );

  const handleTimelineToggleLaneMute = useCallback(
    (lane: TimelineLane) => {
      void applyTimelineOperations(
        [
          {
            op_type: "set_volume",
            params: {
              track_id: lane.id,
              track_kind: lane.kind,
              mute: !lane.mute,
            },
            source: "ui",
          },
        ],
        { notice: `${lane.label} ${lane.mute ? "unmuted" : "muted"}.` },
      );
    },
    [applyTimelineOperations],
  );

  const handleTimelineToggleLaneSolo = useCallback(
    (lane: TimelineLane) => {
      void applyTimelineOperations(
        [
          {
            op_type: "set_volume",
            params: {
              track_id: lane.id,
              track_kind: lane.kind,
              solo: !lane.solo,
            },
            source: "ui",
          },
        ],
        { notice: `${lane.label} ${lane.solo ? "unsoloed" : "soloed"}.` },
      );
    },
    [applyTimelineOperations],
  );

  const handleTimelineToggleLaneLock = useCallback(
    (laneId: string) => {
      setLockedLaneIds((prev) => {
        const next = new Set(prev);
        if (next.has(laneId)) {
          next.delete(laneId);
        } else {
          next.add(laneId);
        }
        if (project?.id) {
          writeLockedLaneIds(project.id, next);
        }
        return next;
      });
    },
    [project?.id],
  );

  useEffect(() => {
    if (!project?.id) return;
    setLockedLaneIds(readLockedLaneIds(project.id));
  }, [project?.id]);

  const handleTimelineMoveBrollClip = useCallback(
    (clipId: string, timelineStartSec: number) => {
      if (brollTimelineActionKey) return;
      void setBrollClipStart(clipId, timelineStartSec);
    },
    [brollTimelineActionKey],
  );

  const handleTimelineTrimBrollClip = useCallback(
    (clipId: string, durationSec: number) => {
      if (brollTimelineActionKey) return;
      void setBrollClipDuration(clipId, durationSec);
    },
    [brollTimelineActionKey],
  );

  const handleTimelineSetBrollOpacity = useCallback(
    (clipId: string, opacity: number) => {
      if (brollTimelineActionKey) return;
      void setBrollClipOpacity(clipId, opacity);
    },
    [brollTimelineActionKey],
  );

  const handleTimelineDeleteBrollClip = useCallback(
    (clipId: string) => {
      if (brollTimelineActionKey) return;
      void removeBrollClipById(clipId);
    },
    [brollTimelineActionKey],
  );

  const handleTimelineRerollBrollClip = useCallback(
    (clipId: string) => {
      if (brollTimelineActionKey || brollActionKey) return;
      void rerollBrollFromTimelineClip(clipId);
    },
    [brollTimelineActionKey, brollActionKey],
  );

  const findSlotForOverlayClip = useCallback(
    (clip: Clip): BrollSlot | null => {
      const clipStart = clip.timeline_start_sec;
      const clipEnd = clip.timeline_start_sec + clipTimelineDurationSec(clip);
      const ranked = brollSlots
        .filter((slot) => !!slot.chosen_candidate_id)
        .map((slot) => {
          const chosen =
            slot.candidates.find(
              (candidate) => candidate.id === slot.chosen_candidate_id,
            ) ?? null;
          const assetMatch = chosen?.asset_id === clip.asset_id ? 1 : 0;
          const overlap = Math.max(
            0,
            Math.min(slot.end_sec, clipEnd) -
              Math.max(slot.start_sec, clipStart),
          );
          const startDelta = Math.abs(slot.start_sec - clipStart);
          const score = assetMatch * 1000 + overlap * 10 - startDelta;
          return { slot, score };
        })
        .sort((a, b) => b.score - a.score);
      return ranked[0]?.slot ?? null;
    },
    [brollSlots],
  );

  const handleTimelineSelectLaneClip = useCallback(
    (selection: TimelineLaneClipSelection | null) => {
      setSelectedTimelineClip(selection);
      if (selection) {
        setSelectedBrollClipId(null);
        setSelectedBrollSlotId(null);
        setSelectedCaptionBlock(null);
        setSelectedWordIds(new Set());
      }
    },
    [],
  );

  const handleTimelineSelectBrollClip = useCallback(
    (clipId: string | null) => {
      setSelectedBrollClipId(clipId);
      if (!clipId) {
        setSelectedBrollSlotId(null);
        return;
      }
      if (clipId) {
        setSelectedTimelineClip(null);
        setSelectedCaptionBlock(null);
        const clip =
          sortedOverlayClips.find((item) => item.id === clipId) ?? null;
        const slot = clip ? findSlotForOverlayClip(clip) : null;
        if (slot) {
          focusBrollSlot(
            slot,
            "B-roll clip selected with its transcript region.",
          );
        } else if (clip && transcript) {
          const ids = selectTranscriptWordIdsInRange(
            timelineAssistWords,
            clip.timeline_start_sec,
            clip.timeline_start_sec + clipTimelineDurationSec(clip),
          );
          setSelectedBrollSlotId(null);
          focusTranscriptWordIds(
            ids,
            undefined,
            "B-roll clip selected. No linked slot was found.",
          );
        } else {
          setSelectedWordIds(new Set());
          setSelectedBrollSlotId(null);
        }
      }
    },
    [
      findSlotForOverlayClip,
      focusBrollSlot,
      focusTranscriptWordIds,
      sortedOverlayClips,
      timelineAssistWords,
      transcript,
    ],
  );

  const handleTimelineSelectCaptionBlock = useCallback(
    (selection: TimelineCaptionSelection | null) => {
      setSelectedCaptionBlock(selection);
      if (selection) {
        setSelectedTimelineClip(null);
        setSelectedBrollClipId(null);
        setSelectedBrollSlotId(null);
        setSelectedWordIds(new Set());
      }
    },
    [],
  );

  const handleTimelineMoveCaptionBlock = useCallback(
    (selection: TimelineCaptionSelection, startSec: number) => {
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
        { notice: "Caption block moved." },
      );
    },
    [applyTimelineOperations],
  );

  const handleTimelineTrimCaptionBlock = useCallback(
    (
      selection: TimelineCaptionSelection,
      startSec: number,
      durationSec: number,
    ) => {
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
        { notice: "Caption block trimmed." },
      );
    },
    [applyTimelineOperations],
  );

  const handleTimelineDeleteLaneClip = useCallback(
    (selection: TimelineLaneClipSelection) => {
      if (lockedLaneIds.has(selection.laneId)) return;
      void applyTimelineOperations(
        [
          {
            op_type: "delete_clip",
            params: { clip: selection.clipId },
            source: "ui",
          },
        ],
        { notice: `${selection.laneLabel} clip deleted.` },
      ).then(() => setSelectedTimelineClip(null));
    },
    [applyTimelineOperations, lockedLaneIds],
  );

  const handleTimelineSplitLaneClip = useCallback(
    (selection: TimelineLaneClipSelection) => {
      if (lockedLaneIds.has(selection.laneId)) return;
      void applyTimelineOperations(
        [
          {
            op_type: "split_clip",
            params: {
              clip: selection.clipId,
              at_sec: Number(currentTimeSec.toFixed(3)),
            },
            source: "ui",
          },
        ],
        { notice: `${selection.laneLabel} clip split at playhead.` },
      );
    },
    [applyTimelineOperations, currentTimeSec, lockedLaneIds],
  );

  const handleTimelineEditWord = useCallback(
    (wordId: string) => {
      const word = transcript?.words.find((w) => w.id === wordId);
      if (word) startEditing(word);
    },
    [transcript, startEditing],
  );

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
      { notice: `${currentSelection.laneLabel} clip deleted.` },
    ).then(() => {
      setSelectedTimelineClip(null);
    });
  }, [applyTimelineOperations, selectedTimelineClip]);

  const copySelectedTimelineClip = useCallback(() => {
    if (!selectedTimelineClipDetails) return;
    const { lane, clip } = selectedTimelineClipDetails;
    setClipClipboard({
      clip: JSON.parse(JSON.stringify(clip)) as Clip,
      laneKind: lane.kind,
      laneLabel: lane.label,
    });
    setNotice(`${lane.label} clip copied.`);
  }, [selectedTimelineClipDetails]);

  const pasteTimelineClip = useCallback(
    (atSec?: number) => {
      if (!clipClipboard) return;
      const { id: _copiedId, ...clipPayload } = clipClipboard.clip;
      void applyTimelineOperations(
        [
          {
            op_type: "paste_clip",
            params: {
              clip: clipPayload,
              track_kind: clipClipboard.laneKind,
              timeline_start_sec: Number(
                Math.max(0, atSec ?? currentTimeSec).toFixed(3),
              ),
              ripple: true,
            },
            source: "ui",
          },
        ],
        { notice: `${clipClipboard.laneLabel} clip pasted.` },
      );
    },
    [applyTimelineOperations, clipClipboard, currentTimeSec],
  );

  const duplicateSelectedTimelineClip = useCallback(() => {
    if (!selectedTimelineClipDetails) return;
    const { lane, clip, durationSec } = selectedTimelineClipDetails;
    const { id: _copiedId, ...clipPayload } = clip;
    // Land just before the next clip's start so the duplicate sorts in
    // directly after the original before the track ripples.
    const insertAtSec = Math.max(
      clip.timeline_start_sec + 0.001,
      clip.timeline_start_sec + durationSec - 0.001,
    );
    void applyTimelineOperations(
      [
        {
          op_type: "paste_clip",
          params: {
            clip: clipPayload,
            track_kind: lane.kind,
            track_id: lane.id,
            timeline_start_sec: Number(insertAtSec.toFixed(3)),
            ripple: true,
          },
          source: "ui",
        },
      ],
      { notice: `${lane.label} clip duplicated.` },
    );
  }, [applyTimelineOperations, selectedTimelineClipDetails]);

  const splitSelectedTimelineClip = useCallback(() => {
    if (!selectedTimelineClipDetails) return;
    const clipStart = selectedTimelineClipDetails.clip.timeline_start_sec;
    const clipEnd = clipStart + selectedTimelineClipDetails.durationSec;
    if (
      currentTimeSec <= clipStart + 0.01 ||
      currentTimeSec >= clipEnd - 0.01
    ) {
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
      {
        notice: `${selectedTimelineClipDetails.lane.label} clip split at playhead.`,
      },
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
      { notice: "Caption block deleted." },
    ).then(() => {
      setSelectedCaptionBlock(null);
      setEditingCaptionOverlayId(null);
      setEditingCaptionText("");
    });
  }, [applyTimelineOperations, selectedCaptionBlock]);

  const startCaptionEditing = useCallback(
    (selection: TimelineCaptionSelection) => {
      setEditingCaptionOverlayId(selection.overlayId);
      setEditingCaptionText(selection.text);
      setSelectedCaptionBlock({
        overlayId: selection.overlayId,
        clipId: selection.clipId,
        laneId: selection.laneId,
        laneLabel: selection.laneLabel,
        text: selection.text,
        style: selection.style,
      });
      setSelectedTimelineClip(null);
      setSelectedBrollClipId(null);
      setTimeout(() => captionEditInputRef.current?.focus(), 0);
    },
    [],
  );

  const commitCaptionEdit = useCallback(async () => {
    if (!editingCaptionOverlayId || !selectedCaptionBlockDetails) {
      setEditingCaptionOverlayId(null);
      setEditingCaptionText("");
      return;
    }
    const trimmed = editingCaptionText.trim();
    const currentText = selectedCaptionBlockDetails.overlay.text;
    if (!trimmed || trimmed === currentText) {
      setEditingCaptionOverlayId(null);
      setEditingCaptionText("");
      return;
    }
    const overlayId = editingCaptionOverlayId;
    const { clip } = selectedCaptionBlockDetails;
    setEditingCaptionOverlayId(null);
    setEditingCaptionText("");
    await applyTimelineOperations(
      [
        {
          op_type: "update_text_overlay",
          params: {
            clip: clip.id,
            overlay: overlayId,
            text: trimmed,
          },
          source: "ui",
        },
      ],
      { notice: "Caption text updated.", forcePreview: true },
    );
    setSelectedCaptionBlock((prev) =>
      prev && prev.overlayId === overlayId ? { ...prev, text: trimmed } : prev,
    );
  }, [
    applyTimelineOperations,
    editingCaptionOverlayId,
    editingCaptionText,
    selectedCaptionBlockDetails,
  ]);

  const cancelCaptionEdit = useCallback(() => {
    setEditingCaptionOverlayId(null);
    setEditingCaptionText("");
  }, []);

  const setSelectedTimelineClipSpeed = useCallback(
    (speed: number) => {
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
        {
          notice: `${selectedTimelineClipDetails.lane.label} speed set to ${speed}x.`,
        },
      );
    },
    [applyTimelineOperations, selectedTimelineClipDetails],
  );

  const setSelectedTimelineClipVolume = useCallback(
    (volume: number) => {
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
        { notice: `${selectedTimelineClipDetails.lane.label} volume updated.` },
      );
    },
    [applyTimelineOperations, selectedTimelineClipDetails],
  );

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
      {
        notice: `${selectedTimelineClipDetails.lane.label} clip ${nextMute ? "muted" : "unmuted"}.`,
      },
    );
  }, [applyTimelineOperations, selectedTimelineClipDetails]);

  // ── Search navigation ──────────────────────────────────────────────
  function navigateSearch(direction: 1 | -1) {
    if (!searchMatchIds.length) return;
    const nextIdx =
      (searchMatchIndex + direction + searchMatchIds.length) %
      searchMatchIds.length;
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
      setSuggestingBrollSelection(false);
      setBrollSuggestionSource(null);
      setBrollSelectionLabel("");
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
        const opacity =
          typeof clip.broll_opacity === "number"
            ? Math.max(0, Math.min(1, clip.broll_opacity))
            : 1;
        next[clip.id] = opacity;
      });
      return next;
    });
  }, [sortedOverlayClips]);

  useEffect(() => {
    if (!selectedTimelineClip) return;
    const lane = timelineLanes.find(
      (item) => item.id === selectedTimelineClip.laneId,
    );
    const exists = !!lane?.clips.some(
      (clip) => clip.id === selectedTimelineClip.clipId,
    );
    if (!exists) {
      setSelectedTimelineClip(null);
    }
  }, [selectedTimelineClip, timelineLanes]);

  useEffect(() => {
    if (!selectedBrollClipId) return;
    const exists = sortedOverlayClips.some(
      (clip) => clip.id === selectedBrollClipId,
    );
    if (!exists) {
      setSelectedBrollClipId(null);
    }
  }, [selectedBrollClipId, sortedOverlayClips]);

  useEffect(() => {
    if (!selectedCaptionBlock) return;
    const exists = captionBlocks.some(
      (block) =>
        block.id === selectedCaptionBlock.overlayId &&
        block.clipId === selectedCaptionBlock.clipId,
    );
    if (!exists) {
      setSelectedCaptionBlock(null);
    }
  }, [selectedCaptionBlock, captionBlocks]);

  // Jump to the tab a selection suggests, but only once per new selection.
  // Tracking the last handled context kind lets the user freely switch tabs
  // afterwards instead of being snapped back on every manual click.
  const lastInspectorKindRef = useRef<string | null>(null);
  useEffect(() => {
    if (!featureDrawerOpen) {
      lastInspectorKindRef.current = null;
      return;
    }
    if (inspectorContext.kind === "project") return;
    if (lastInspectorKindRef.current === inspectorContext.kind) return;
    lastInspectorKindRef.current = inspectorContext.kind;
    setActiveFeatureTab(inspectorContext.suggestedTab);
  }, [featureDrawerOpen, inspectorContext.kind, inspectorContext.suggestedTab]);

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

  // ? key to toggle keyboard shortcuts help
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const tag = (event.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (event.key === "?" || (event.shiftKey && event.key === "/")) {
        event.preventDefault();
        setShowShortcutsHelp((prev) => !prev);
      }
      if (event.key === "Escape" && showShortcutsHelp) {
        setShowShortcutsHelp(false);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [showShortcutsHelp]);

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
    if (backendStatus !== "ok") return;
    let active = true;
    void api
      .getBrollConfig()
      .then((config) => {
        if (active) setBrollConfig(config);
      })
      .catch(() => {
        if (active) setBrollConfig(null);
      });
    return () => {
      active = false;
    };
  }, [backendStatus]);

  useEffect(() => {
    if (backendStatus !== "ok") return;
    void refreshProjectList();
  }, [backendStatus, refreshProjectList]);

  useEffect(() => {
    if (
      project ||
      creatingProject ||
      autoCreateAttemptedRef.current ||
      backendStatus !== "ok"
    ) {
      return;
    }
    autoCreateAttemptedRef.current = true;
    void (async () => {
      // A file dropped on the landing page needs a project to receive the
      // upload, so creating one here is the user's explicit intent.
      if (hasPendingUploadFile()) {
        await createProject(
          filenameToProjectName(
            peekPendingUploadName(),
            BRAND.defaultProjectName,
          ),
          { silent: true },
        );
        return;
      }
      try {
        const projects = await api.listProjects();
        setRecentProjects(projects);
        if (projects.length > 0) {
          await openProject(projects[0].id);
        }
        // No projects yet: stay on the welcome screen and wait for the user
        // to create one explicitly.
      } catch (err) {
        setError((err as Error).message);
      }
    })();
  }, [project, creatingProject, backendStatus]);

  // Auto-upload pending file from landing page drop zone
  const pendingUploadHandledRef = useRef(false);
  useEffect(() => {
    if (!project || uploading || pendingUploadHandledRef.current) return;
    const file = consumePendingUploadFile();
    if (!file) return;
    pendingUploadHandledRef.current = true;
    void uploadVideo(file);
  }, [project, uploading]);

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
    setTranscriptElapsedSec(
      Math.max(0, Math.floor((Date.now() - transcriptStartedAtMs) / 1000)),
    );
    const interval = window.setInterval(() => {
      setTranscriptElapsedSec(
        Math.max(0, Math.floor((Date.now() - transcriptStartedAtMs) / 1000)),
      );
    }, 1000);
    return () => window.clearInterval(interval);
  }, [generatingTranscript, transcriptStartedAtMs]);

  const transcriptJobId = transcriptJob?.id;
  const transcriptJobStatus = transcriptJob?.status;
  useEffect(() => {
    if (
      !generatingTranscript ||
      !transcriptJob ||
      (transcriptJob.status !== "queued" && transcriptJob.status !== "running")
    ) {
      transcriptStageKeyRef.current = null;
      setTranscriptStageStartedAtMs(null);
      setTranscriptStageBaseProgress(0);
      return;
    }

    const nextKey = [
      transcriptJob.id,
      transcriptJob.status,
      transcriptJob.stage ?? "",
      transcriptJob.message ?? "",
    ].join(":");
    if (transcriptStageKeyRef.current === nextKey) {
      return;
    }

    transcriptStageKeyRef.current = nextKey;
    setTranscriptStageStartedAtMs(Date.now());
    setTranscriptStageBaseProgress(
      Math.max(0, Math.min(100, Math.round(transcriptJob.progress ?? 0))),
    );
  }, [generatingTranscript, transcriptJob]);


  useEffect(() => {
    if (!project?.id || !transcriptJobId) return;

    if (transcriptJobStatus === "completed") {
      if (transcriptJobResultHandledRef.current === transcriptJobId) return;
      transcriptJobResultHandledRef.current = transcriptJobId;
      void (async () => {
        try {
          const response = await api.getTranscriptGenerateResult(
            project.id,
            transcriptJobId,
          );
          await applyTranscriptGenerationResult(response);
        } catch (err) {
          setError((err as Error).message);
          setNotice(null);
        } finally {
          setGeneratingTranscript(false);
          setTranscriptStartedAtMs(null);
          setTranscriptStageStartedAtMs(null);
          setTranscriptStageBaseProgress(0);
        }
      })();
      return;
    }

    if (transcriptJobStatus === "failed") {
      setGeneratingTranscript(false);
      setTranscriptStartedAtMs(null);
      setTranscriptStageStartedAtMs(null);
      setTranscriptStageBaseProgress(0);
      if (transcriptJob.error) {
        setError(transcriptJob.error);
      }
      return;
    }

    if (transcriptJobStatus !== "queued" && transcriptJobStatus !== "running") {
      return;
    }

    const interval = window.setInterval(async () => {
      try {
        const refreshed = await api.getJob(transcriptJobId);
        setTranscriptJob(refreshed);
      } catch {
        // Ignore transient polling errors.
      }
    }, 1000);
    return () => window.clearInterval(interval);
  }, [
    project?.id,
    transcriptJobId,
    transcriptJobStatus,
    transcriptJob?.error,
    applyTranscriptGenerationResult,
  ]);

  // Auto-apply cut debounce
  useEffect(() => {
    if (!project || !transcript) return;
    if (applyingCut) return;
    if (deletedWordIds.size === 0) return;
    if (deletedSignature === lastAppliedSignatureRef.current) return;
    if (deletedSignature === lastAutoCutFailedSignatureRef.current) return;
    const handle = window.setTimeout(() => {
      void applyCut(deletedSignature, keptWordIds);
    }, 450);
    return () => window.clearTimeout(handle);
  }, [
    project?.id,
    transcript?.id,
    deletedWordIds.size,
    deletedSignature,
    keptWordIds,
    applyingCut,
  ]);

  // Quick edit pipeline continuation — when transcript generation completes while in quickEdit
  // "transcribing" phase, automatically advance to cut+captions
  useEffect(() => {
    if (quickEditPhaseRef.current !== "transcribing") return;
    if (generatingTranscript) return; // still in progress


    if (!transcript?.words?.length) {
      // Transcript generation finished but no words — it failed or was empty
      quickEditPhaseRef.current = "idle";
      setQuickEditing(false);
      setQuickEditStage("");
      return;
    }

    // Transcript just became available — advance to cut+captions
    quickEditPhaseRef.current = "cutting";
    setQuickEditStage("Removing pauses & filler...");
    void runQuickEditCutAndCaptions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [generatingTranscript, transcript?.words?.length]);

  // Preview polling — deps stabilized to [id, status] to prevent interval restart storms
  const previewJobId = previewJob?.id;
  const previewJobStatus = previewJob?.status;
  useEffect(() => {
    if (
      !previewJobId ||
      (previewJobStatus !== "queued" && previewJobStatus !== "running")
    )
      return;
    const interval = window.setInterval(async () => {
      try {
        const refreshed = await api.getJob(previewJobId);
        if (
          (refreshed.status === "completed" || refreshed.status === "failed") &&
          pendingPreviewRefreshRef.current
        ) {
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
    if (
      !project?.id ||
      !brollSuggestJob ||
      (brollSuggestJob.status !== "queued" &&
        brollSuggestJob.status !== "running")
    ) {
      return;
    }
    const interval = window.setInterval(async () => {
      try {
        const refreshed = await api.getJob(brollSuggestJob.id);
        setBrollSuggestJob(refreshed);
        if (refreshed.status === "completed") {
          const result = await api.getSuggestBrollResult(
            project.id,
            refreshed.id,
          );
          const wasSelectionRequest = brollSuggestionSource === "selection";
          if (wasSelectionRequest) {
            setBrollSlots((prev) => {
              const existingIds = new Set(prev.map((slot) => slot.id));
              const newSlots = result.slots.filter(
                (slot) => !existingIds.has(slot.id),
              );
              return [...prev, ...newSlots];
            });
            const firstSlot = result.slots[0];
            if (firstSlot) {
              setExpandedBrollSlots((prev) => ({
                ...prev,
                [firstSlot.id]: true,
              }));
              setActiveFeatureTab("broll_studio");
              setFeatureDrawerOpen(true);
            }
          } else {
            setBrollSlots(result.slots);
          }
          setSuggestingBroll(false);
          setSuggestingBrollSelection(false);
          const reviewCount = result.slots.filter(
            (slot) => slot.review_status === "needs_review",
          ).length;
          const completionNotice =
            wasSelectionRequest && result.slots.length === 0
              ? "No B-roll candidates found for the selected words."
              : `${wasSelectionRequest && brollSelectionLabel ? `B-roll suggestions ready for \"${brollSelectionLabel}\". ` : "Generated "}${result.created_slots} B-roll slot${result.created_slots === 1 ? "" : "s"}. ${reviewCount} need review before sync.`;
          setNotice(completionNotice);
          setBrollSuggestionSource(null);
          setBrollSelectionLabel("");
        } else if (refreshed.status === "failed") {
          setSuggestingBroll(false);
          setSuggestingBrollSelection(false);
          setBrollSuggestionSource(null);
          setBrollSelectionLabel("");
          setError(refreshed.error ?? "B-roll generation failed.");
        }
      } catch {
        // Ignore transient polling errors.
      }
    }, 2500);
    return () => window.clearInterval(interval);
  }, [
    project?.id,
    brollSelectionLabel,
    brollSuggestJob,
    brollSuggestionSource,
  ]);

  // Export polling
  const exportJobId = exportJob?.id;
  const exportJobStatus = exportJob?.status;
  useEffect(() => {
    if (
      !exportJobId ||
      (exportJobStatus !== "queued" && exportJobStatus !== "running")
    ) {
      return;
    }
    const interval = window.setInterval(async () => {
      try {
        const refreshed = await api.getJob(exportJobId);
        setExportJob(refreshed);
        if (refreshed.status === "failed") {
          setError(refreshed.error ?? "Export video failed.");
        }
      } catch {
        // Ignore transient polling errors.
      }
    }, 1000);
    return () => window.clearInterval(interval);
  }, [exportJobId, exportJobStatus]);

  useEffect(() => {
    if (exportJob?.status === "completed") {
      void handleCompletedExport(exportJob);
    }
    if (exportJob?.status === "failed" && exportJob.error) {
      setError(exportJob.error);
    }
  }, [exportJob, handleCompletedExport]);

  // Keyboard shortcuts
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const tagName = target?.tagName?.toLowerCase();
      const isEditableTarget =
        !!target?.closest('input,textarea,select,[contenteditable="true"]') ||
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

      if (
        !isEditableTarget &&
        (event.key === "ArrowLeft" || event.key === "ArrowRight")
      ) {
        const player = videoRef.current;
        const arrowHandling = decideTimelineArrowHandling({
          key: event.key,
          timelineCoreV2: TIMELINE_CORE_V2,
          altKey: event.altKey,
          shiftKey: event.shiftKey,
          ctrlKey: event.ctrlKey,
          metaKey: event.metaKey,
        });
        if (arrowHandling === "v2") {
          const durationSec =
            (player && Number.isFinite(player.duration) ? player.duration : 0) ||
            transcript?.duration_sec ||
            project?.timeline.duration_sec ||
            0;
          const command = compileTimelineKeyboardCommand({
            key: event.key,
            altKey: event.altKey,
            shiftKey: event.shiftKey,
            ctrlKey: event.ctrlKey,
            metaKey: event.metaKey,
            currentFrame: secondsToFrame(
              timelineClockRef.current.getSnapshot(),
              canonicalTimelineFps,
            ),
            durationFrames: secondsToFrame(
              Math.max(0, durationSec),
              canonicalTimelineFps,
            ),
            selectedClipStartFrame: selectedTimelineClipDetails
              ? secondsToFrame(
                  selectedTimelineClipDetails.clip.timeline_start_sec,
                  canonicalTimelineFps,
                )
              : null,
          });
          if (command) {
            event.preventDefault();
            const seconds = frameToSeconds(command.frame, canonicalTimelineFps);
            if (command.kind === "seek") {
              handleTimelineSeek(seconds);
            } else if (selectedTimelineClip) {
              handleTimelineMoveLaneClip(selectedTimelineClip, seconds);
            }
          }
        } else if (arrowHandling === "legacy" && player) {
          event.preventDefault();
          const step = event.shiftKey ? 1 : 5;
          const next =
            event.key === "ArrowLeft"
              ? Math.max(0, player.currentTime - step)
              : Math.min(
                  player.duration || Infinity,
                  player.currentTime + step,
                );
          player.currentTime = next;
          setCurrentTimeSec(next);
        }
      }

      if (
        !isEditableTarget &&
        !editingWordId &&
        !editingCaptionOverlayId &&
        (event.key === "Delete" || event.key === "Backspace")
      ) {
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

      if (
        !isEditableTarget &&
        !editingWordId &&
        event.key.toLowerCase() === "s" &&
        !event.ctrlKey &&
        !event.metaKey
      ) {
        if (selectedTimelineClip) {
          event.preventDefault();
          splitSelectedTimelineClip();
        }
      }

      if (
        !isEditableTarget &&
        (event.ctrlKey || event.metaKey) &&
        event.key.toLowerCase() === "c" &&
        selectedTimelineClip &&
        !window.getSelection()?.toString()
      ) {
        event.preventDefault();
        copySelectedTimelineClip();
      }

      if (
        !isEditableTarget &&
        (event.ctrlKey || event.metaKey) &&
        event.key.toLowerCase() === "v" &&
        clipClipboard
      ) {
        event.preventDefault();
        pasteTimelineClip();
      }

      if (
        !isEditableTarget &&
        (event.ctrlKey || event.metaKey) &&
        event.key.toLowerCase() === "d" &&
        selectedTimelineClip
      ) {
        event.preventDefault();
        duplicateSelectedTimelineClip();
      }

      if (
        event.key === "z" &&
        (event.ctrlKey || event.metaKey) &&
        !event.shiftKey
      ) {
        event.preventDefault();
        void undo();
      }

      if (
        event.key === "z" &&
        (event.ctrlKey || event.metaKey) &&
        event.shiftKey
      ) {
        event.preventDefault();
        void redo();
      }

      if (event.key === "y" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        void redo();
      }

      if (event.key === "Escape") {
        if (editingWordId) {
          cancelEdit();
        } else if (editingCaptionOverlayId) {
          cancelCaptionEdit();
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
    cancelCaptionEdit,
    cancelEdit,
    clearEditorSelections,
    clipClipboard,
    canonicalTimelineFps,
    copySelectedTimelineClip,
    deleteSelectedCaptionBlock,
    deleteSelectedTimelineClip,
    duplicateSelectedTimelineClip,
    editingCaptionOverlayId,
    editingWordId,
    handleTimelineMoveLaneClip,
    handleTimelineSeek,
    pasteTimelineClip,
    redo,
    removeBrollClipById,
    selectedBrollClipId,
    selectedCaptionBlock,
    selectedTimelineClip,
    selectedTimelineClipDetails,
    selectedWordIds,
    splitSelectedTimelineClip,
    transcript,
    project?.timeline.duration_sec,
    undo,
    updateDeletedWords,
  ]);

  // Auto-scroll to active word during playback (paused after manual scroll)
  useEffect(() => {
    if (!isVideoPlaying) return;
    if (!activeWordId || editingWordId) return;
    if (!transcriptFollowPlaybackRef.current) return;
    const el = document.getElementById(`word-${activeWordId}`);
    const box = transcriptBoxRef.current;
    if (!el || !box) return;
    const elTop = el.offsetTop - box.offsetTop;
    const elBottom = elTop + el.offsetHeight;
    const scrollTop = box.scrollTop;
    const boxHeight = box.clientHeight;
    if (elTop < scrollTop || elBottom > scrollTop + boxHeight) {
      transcriptProgrammaticScrollRef.current = true;
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      window.setTimeout(() => {
        transcriptProgrammaticScrollRef.current = false;
      }, 400);
    }
  }, [activeWordId, editingWordId, isVideoPlaying]);

  useEffect(() => {
    if (isVideoPlaying) {
      transcriptFollowPlaybackRef.current = true;
    }
  }, [isVideoPlaying]);

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
    if (!project || !transcript || suggestingBroll || suggestingBrollSelection)
      return;
    setSuggestingBroll(true);
    setBrollSuggestionSource("full");
    setBrollSelectionLabel("");
    setBrollSuggestJob(null);
    setError(null);
    try {
      const plan = resolveBrollGenerationPlan(
        project,
        transcript,
        brollIntensity,
        brollAutoMode,
        videoAssets.length,
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
          `Target runtime: ${plan.runtimeHint}.${plan.usedExternalFallback ? " Added external fallback due to limited local video assets." : ""}`,
      );
    } catch (err) {
      setError((err as Error).message);
      setSuggestingBroll(false);
      setBrollSuggestionSource(null);
    }
  }

  async function suggestBrollForSelection() {
    if (
      !project ||
      !transcript ||
      !selectedTranscriptRange ||
      suggestingBrollSelection ||
      suggestingBroll
    )
      return;
    const selectionText = selectedTranscriptRange.text;
    const selectionLabel = `${selectionText.slice(0, 40)}${selectionText.length > 40 ? "…" : ""}`;
    setSuggestingBrollSelection(true);
    setBrollSuggestionSource("selection");
    setBrollSelectionLabel(selectionLabel);
    setBrollSuggestJob(null);
    setError(null);
    try {
      const wordIds = Array.from(selectedWordIds);
      const queued = await api.suggestBrollAsync(project.id, {
        transcript_id: transcript.id,
        candidates_per_slot: 4,
        replace_existing: false,
        include_project_assets: true,
        include_external_sources: true,
        ai_rerank: true,
        anchor_word_ids: wordIds,
      });
      setBrollSuggestJob(queued);
      setNotice(`Finding B-roll candidates for "${selectionLabel}"...`);
    } catch (err) {
      setError((err as Error).message);
      setSuggestingBrollSelection(false);
      setBrollSuggestionSource(null);
      setBrollSelectionLabel("");
    }
  }

  async function autoApplyBroll() {
    if (
      !project ||
      !transcript ||
      autoApplyingBroll ||
      suggestingBroll ||
      suggestingBrollSelection
    )
      return;
    setAutoApplyingBroll(true);
    setError(null);
    const plan = resolveBrollGenerationPlan(
      project,
      transcript,
      brollIntensity,
      brollAutoMode,
      videoAssets.length,
    );
    setNotice(
      `Auto-applying ${plan.modeLabel} B-roll. Target runtime: ${plan.runtimeHint}.` +
        (plan.usedExternalFallback
          ? " Using external fallback due to limited local video assets."
          : ""),
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
        fallback_to_top_candidate: brollAutoMode === "fast",
        min_confidence: plan.minConfidence,
        overlay_opacity: brollDefaultOpacity,
      });
      setBrollSlots(response.slots);
      setLastBrollAutoApplySkips(response.skipped_slot_summaries ?? []);
      setProject((prev) =>
        prev ? { ...prev, timeline: response.timeline } : prev,
      );
      await refreshMedia(project.id);
      let noticeText = `Auto-applied B-roll: ${response.auto_chosen_slots} chosen, ${response.synced_clip_count} synced to timeline.`;
      if (response.skipped_slots > 0) {
        noticeText += ` ${response.skipped_slots} slot${response.skipped_slots === 1 ? "" : "s"} skipped — review below in B-roll Studio.`;
        const summaries = response.skipped_slot_summaries ?? [];
        if (summaries.length > 0) {
          const preview = summaries
            .slice(0, 2)
            .map((item) => autoApplySkipReasonLabel(item.reason))
            .join(", ");
          noticeText += ` Reasons: ${preview}${summaries.length > 2 ? ", ..." : ""}.`;
        }
      } else {
        noticeText += ` Confidence threshold ${(response.confidence_threshold * 100).toFixed(0)}%.`;
      }
      if (brollAutoMode === "fast") {
        noticeText +=
          " Fast mode uses best available match when confidence is low.";
      }
      setNotice(noticeText);
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
      const updated = await api.chooseBrollCandidate(
        project.id,
        slotId,
        candidateId,
      );
      setBrollSlots((prev) =>
        prev.map((slot) => (slot.id === slotId ? updated : slot)),
      );
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
      setBrollSlots((prev) =>
        prev.map((slot) => (slot.id === slotId ? updated : slot)),
      );
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
      const updated = await api.rerollBrollSlot(
        project.id,
        slotId,
        buildBrollRerollPayload(brollMeaningDrafts[slotId]),
      );
      const nextCount = updated.candidates.length;
      const addedCount = Math.max(0, nextCount - previousCount);
      setBrollSlots((prev) =>
        prev.map((slot) => (slot.id === slotId ? updated : slot)),
      );
      setNotice(
        addedCount > 0
          ? `Added ${addedCount} new B-roll variant${addedCount === 1 ? "" : "s"} for this slot.`
          : "Rerolled slot candidates.",
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
      .filter(
        (slot) =>
          slot.chosen_candidate_id && (!slotIds || slotIds.includes(slot.id)),
      )
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
        overlay_opacity: brollDefaultOpacity,
        slot_ids: slotIds ?? [],
      });
      setProject((prev) =>
        prev ? { ...prev, timeline: response.timeline } : prev,
      );
      setBrollSlots(response.slots);
      setNotice(
        `Synced ${response.synced_clip_count} B-roll clip${response.synced_clip_count === 1 ? "" : "s"} to timeline.`,
      );
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
      setProject((prev) =>
        prev ? { ...prev, timeline: response.timeline } : prev,
      );
      if (transcript) {
        await refreshBrollSlots(project.id, transcript.id);
      }
      setNotice(
        `Restored ${response.restored_clip_count} overlay clip${response.restored_clip_count === 1 ? "" : "s"} from previous B-roll transaction.`,
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
      const localNext = slot.candidates.find(
        (candidate) => candidate.id !== slot.chosen_candidate_id,
      );
      const updatedSlot = localNext
        ? slot
        : await api.rerollBrollSlot(
            project.id,
            slot.id,
            buildBrollRerollPayload(brollMeaningDrafts[slot.id]),
          );

      if (!localNext) {
        setBrollSlots((prev) =>
          prev.map((item) => (item.id === slot.id ? updatedSlot : item)),
        );
      }

      const chooseCandidateId =
        localNext?.id ??
        updatedSlot.candidates.find(
          (candidate) => candidate.id !== updatedSlot.chosen_candidate_id,
        )?.id;
      if (!chooseCandidateId) {
        throw new Error("No alternate B-roll variant available for this clip.");
      }

      const chosenSlot = await api.chooseBrollCandidate(
        project.id,
        slot.id,
        chooseCandidateId,
      );
      setBrollSlots((prev) =>
        prev.map((item) => (item.id === slot.id ? chosenSlot : item)),
      );
      const chosenCandidate =
        chosenSlot.candidates.find(
          (candidate) => candidate.id === chosenSlot.chosen_candidate_id,
        ) ?? null;
      if (!chosenCandidate?.asset_id) {
        throw new Error("Chosen variant is missing a video asset.");
      }

      const latestMedia = await api.listMedia(project.id);
      setMedia(latestMedia);
      const latestMediaById = new Map(
        latestMedia.map((item) => [item.id, item]),
      );
      const sourceDuration = clip.end_sec - clip.start_sec;
      const maybeDuration = latestMediaById.get(
        chosenCandidate.asset_id,
      )?.duration_sec;
      const boundedDuration =
        typeof maybeDuration === "number" && maybeDuration > 0
          ? Math.min(maybeDuration, sourceDuration)
          : sourceDuration;
      const opacity =
        typeof clip.broll_opacity === "number" ? clip.broll_opacity : 1;

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
        "Rerolled B-roll clip variant.",
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
    operations: TimelineOperation[],
    noticeMessage: string,
  ) {
    if (!project || !operations.length) return;
    setBrollTimelineActionKey(`${action}:${clipId}`);
    setError(null);
    try {
      await requestTimelineOperations(project.id, operations);
      setNotice(noticeMessage);
      await queuePreview();
    } catch (err) {
      if (!(err instanceof TimelineMutationProjectChangedError)) {
        setError((err as Error).message);
      }
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
      setBrollDraftStartById((prev) => ({
        ...prev,
        [clip.id]: formatFixedSec(current),
      }));
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
      "Updated B-roll start time.",
    );
  }

  async function setBrollClipDuration(
    clipId: string,
    requestedDurationSec: number,
  ) {
    const clip = getOverlayClipById(clipId);
    if (!clip) return;
    if (!Number.isFinite(requestedDurationSec) || requestedDurationSec <= 0) {
      setBrollDraftDurationById((prev) => ({
        ...prev,
        [clip.id]: formatFixedSec(clipTimelineDurationSec(clip)),
      }));
      return;
    }

    const currentDuration = clipTimelineDurationSec(clip);
    if (Math.abs(requestedDurationSec - currentDuration) < 0.01) {
      setBrollDraftDurationById((prev) => ({
        ...prev,
        [clip.id]: formatFixedSec(currentDuration),
      }));
      return;
    }

    const maxByAsset = mediaById.get(clip.asset_id)?.duration_sec ?? null;
    const proposedEnd =
      clip.start_sec + requestedDurationSec * Math.max(clip.speed, 0.01);
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
          params: {
            clip: clip.id,
            start_sec: clip.start_sec,
            end_sec: Number(boundedEnd.toFixed(3)),
          },
          source: "ui",
        },
      ],
      "Updated B-roll duration.",
    );
  }

  async function setBrollClipOpacity(clipId: string, nextOpacity: number) {
    const clip = getOverlayClipById(clipId);
    if (!clip) return;
    const clamped = Math.max(0, Math.min(1, nextOpacity));
    const current =
      typeof clip.broll_opacity === "number" ? clip.broll_opacity : 1;
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
      "Updated B-roll opacity.",
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
      "Removed B-roll clip from timeline.",
    );
  }

  async function commitBrollStart(clip: Clip) {
    const raw =
      brollDraftStartById[clip.id] ?? formatFixedSec(clip.timeline_start_sec);
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed < 0) {
      setBrollDraftStartById((prev) => ({
        ...prev,
        [clip.id]: formatFixedSec(clip.timeline_start_sec),
      }));
      return;
    }
    await setBrollClipStart(clip.id, parsed);
  }

  async function commitBrollDuration(clip: Clip) {
    const raw =
      brollDraftDurationById[clip.id] ??
      formatFixedSec(clipTimelineDurationSec(clip));
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
    <div className={`appShell${project ? " appShellEditor" : ""}`}>
      {!project ? (
        <>
          <EditorHeader title={BRAND.loadingTitle} />

          <section className="controls card">
            <div style={{ display: "grid", gap: 10, width: "100%" }}>
              <p className="muted" style={{ margin: 0 }}>
                {creatingProject
                  ? "Preparing your workspace..."
                  : loadingProjects || openingProjectId
                    ? "Loading your projects..."
                    : recentProjects.length > 0
                      ? "Open a recent project or create a new one."
                      : "No projects yet. Create one to start editing."}
              </p>
              {showNewProjectForm ? (
                <div className="newProjectInputRow">
                  <input
                    autoFocus
                    type="text"
                    className="controlInput newProjectInput"
                    placeholder="Project name (optional)"
                    value={newProjectName}
                    onChange={(event) => setNewProjectName(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        void createProject(
                          newProjectName.trim() || BRAND.defaultProjectName,
                        );
                      }
                      if (event.key === "Escape") {
                        setShowNewProjectForm(false);
                        setNewProjectName("");
                      }
                    }}
                    disabled={creatingProject}
                  />
                  <button
                    type="button"
                    className="primaryBtn"
                    onClick={() =>
                      void createProject(
                        newProjectName.trim() || BRAND.defaultProjectName,
                      )
                    }
                    disabled={creatingProject}
                  >
                    {creatingProject ? "Creating..." : "Create project"}
                  </button>
                  <button
                    type="button"
                    className="projectActionBtnClose"
                    onClick={() => {
                      setShowNewProjectForm(false);
                      setNewProjectName("");
                    }}
                    disabled={creatingProject}
                    title="Cancel"
                  >
                    <X size={14} />
                  </button>
                </div>
              ) : (
                <button
                  className="primaryBtn"
                  onClick={() => setShowNewProjectForm(true)}
                  disabled={creatingProject || !!openingProjectId}
                  style={{ justifySelf: "start" }}
                >
                  <Wand2 size={16} />
                  New Project
                </button>
              )}
            </div>
          </section>

          {recentProjects.length > 0 && (
            <section className="controls card">
              <div style={{ display: "grid", gap: 8, width: "100%" }}>
                {recentProjects.slice(0, 8).map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className="brollSyncOption"
                    style={{ justifyContent: "flex-start", textAlign: "left" }}
                    onClick={() => void openProject(item.id)}
                    disabled={!!openingProjectId || creatingProject}
                  >
                    {openingProjectId === item.id
                      ? `Opening ${item.name}...`
                      : item.name}
                  </button>
                ))}
              </div>
            </section>
          )}

          {error && <div className="message error">{error}</div>}
          {notice && (
            <div
              className={`message notice${notice === WORKSPACE_READY_NOTICE ? " workspaceReadyNotice" : ""}`}
            >
              {notice}
            </div>
          )}
        </>
      ) : (
        <>
          <EditorHeader title={project.name || BRAND.editorName} />

          <EditorTopActions
            uploading={uploading}
            selectedVideoFilename={selectedVideoAsset?.filename ?? null}
            timelineDurationSec={project.timeline.duration_sec}
            quickEditing={quickEditing}
            quickEditStage={quickEditStage}
            quickEditRuntimeHint={quickEditRuntimeHint}
            generatingTranscript={generatingTranscript}
            runningAction={runningAction}
            exportingVideo={exportingVideo}
            onUploadVideo={(file) => void uploadVideo(file)}
            onToggleProjects={() => {
              setProjectsPanelOpen((open) => !open);
              void refreshProjectList();
            }}
            onQuickEdit={() => quickEdit()}
            onExport={() => void exportVideo()}
            onOpenFeatureDrawer={openFeatureDrawer}
            onShowShortcuts={() => setShowShortcutsHelp(true)}
            formatSeconds={formatSeconds}
          />

          <nav className="mobileWorkspaceTabs" aria-label="Editor views">
            <button
              type="button"
              className={mobileWorkspaceTab === "preview" ? "active" : ""}
              onClick={() => setMobileWorkspaceTab("preview")}
            >
              Preview
            </button>
            <button
              type="button"
              className={mobileWorkspaceTab === "transcript" ? "active" : ""}
              onClick={() => setMobileWorkspaceTab("transcript")}
            >
              Edit
            </button>
            <button
              type="button"
              className={mobileWorkspaceTab === "timeline" ? "active" : ""}
              onClick={() => setMobileWorkspaceTab("timeline")}
            >
              Timeline
            </button>
          </nav>

          {projectsPanelOpen && (
            <ProjectReopenPanel
              project={project}
              recentProjects={recentProjectItems}
              loadingProjects={loadingProjects}
              openingProjectId={openingProjectId}
              submittingRenameId={submittingRenameId}
              deletingProjectId={deletingProjectId}
              creatingProject={creatingProject}
              defaultProjectName={BRAND.defaultProjectName}
              onRefresh={() => void refreshProjectList()}
              onCreate={(name) => void createProject(name)}
              onOpen={(projectId) => void openProject(projectId)}
              onRename={(projectId, name) => void handleRenameProject(projectId, name)}
              onDelete={(projectId) => void handleDeleteProject(projectId)}
              formatSeconds={formatSeconds}
            />
          )}

          {error && <div className="message error">{error}</div>}
          {notice && (
            <div
              className={`message notice${notice === WORKSPACE_READY_NOTICE ? " workspaceReadyNotice" : ""}`}
            >
              {notice}
            </div>
          )}
          {quickEditSummary && (
            <QuickEditSummaryCard
              quickEditSummary={quickEditSummary}
              formatSeconds={formatSeconds}
              formatFixedSec={formatFixedSec}
            />
          )}
          {exportCompletion && (
            <ExportCompletionCard
              exportCompletion={exportCompletion}
              downloadingExport={downloadingExport}
              onDownload={() => void downloadCompletedExport()}
            />
          )}
          <div className={`editorWorkspace mobileTab-${mobileWorkspaceTab}`}>
          <section className="editorMainGrid">
            <div className="mobilePane mobilePanePreview">
            <PreviewDock
              previewSource={previewSource}
              uploading={uploading}
              videoRef={videoRef}
              previewFrameAspectRatio={previewFrameAspectRatio}
              exportAspectRatio={exportAspectRatio}
              livePreviewCaption={livePreviewCaption}
              shouldShowLiveCaptionOverlay={shouldShowLiveCaptionOverlay}
              showExportFrameGuide={showExportFrameGuide}
              previewRenderBusy={previewRenderBusy}
              previewBusyDetail={previewBusyDetail}
              previewProgress={previewProgress}
              currentTimeSec={currentTimeSec}
              previewStatusText={previewStatusText}
              previewJob={previewJob}
              previewUpdateQueued={previewUpdateQueued}
              queueingPreview={queueingPreview}
              canRenderPreview={!!project}
              ingestingUrl={ingestingUrl}
              ingestProgress={ingestProgress}
              ingestStatusMessage={ingestStatusMessage}
              onUploadVideo={(file) => void uploadVideo(file)}
              onIngestUrl={(url) => void ingestVideoFromUrl(url)}
              onLoadedMetadata={() => setCurrentTimeSec(0)}
              onPlay={() => {
                setIsVideoPlaying(true);
                startPlaybackSync();
              }}
              onPause={() => {
                setIsVideoPlaying(false);
                stopPlaybackSync();
                syncVideoTimeOnce();
              }}
              onSeeked={syncVideoTimeOnce}
              onEnded={() => {
                setIsVideoPlaying(false);
                stopPlaybackSync();
                syncVideoTimeOnce();
              }}
              onTimeUpdate={syncVideoTimeIfPlaying}
              onFrameAspectRatioChange={changePreviewFrameAspectRatio}
              onQueuePreview={() => void queuePreview()}
              formatPreciseSeconds={formatPreciseSeconds}
            />
            </div>
            <main className="twoPanel mobilePane mobilePaneTranscript">
              <section className="panel card panelTranscript">
                <div className="transcriptPanelHead">
                  <div>
                    <h2>Transcript Panel</h2>
                    <p className="muted transcriptPanelMeta">
                      Pick the language, generate the transcript, then edit
                      word-by-word with exact timings.
                    </p>
                    {selectedAssetLooksLikeDuet && (
                      <p className="duetHint muted">
                        Multi-voice song detected — speaker labels are enabled
                        when diarization runs.
                      </p>
                    )}
                    {speakerLegend.length > 1 && (
                      <div
                        className="speakerLegend"
                        aria-label="Speaker legend"
                      >
                        {speakerLegend.map((entry) => (
                          <span
                            key={entry.speakerId}
                            className={`speakerLegendItem ${
                              entry.slot === 0
                                ? "speakerA"
                                : entry.slot === 1
                                  ? "speakerB"
                                  : "speakerExtra"
                            }`}
                          >
                            {entry.label}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  {transcriptHasRomanization && (
                    <div
                      className="transcriptViewToggle"
                      role="group"
                      aria-label="Transcript display mode"
                    >
                      <button
                        type="button"
                        className={!showRomanizedTranscript ? "active" : ""}
                        onClick={() => setShowRomanizedTranscript(false)}
                        title="Show transcript in original script"
                      >
                        <span className="scriptHint" aria-hidden="true">
                          ಅ
                        </span>{" "}
                        Original
                      </button>
                      <button
                        type="button"
                        className={showRomanizedTranscript ? "active" : ""}
                        onClick={() => setShowRomanizedTranscript(true)}
                        title="Show transcript in Roman/Latin characters"
                      >
                        <span className="scriptHint" aria-hidden="true">
                          A
                        </span>{" "}
                        Romanized
                      </button>
                    </div>
                  )}
                </div>
                <div className="transcriptControlBar">
                  <label className="transcriptControlField">
                    <span>Transcript Mode</span>
                    <select
                      value={transcriptMode}
                      disabled={generatingTranscript}
                      onChange={(event) =>
                        setTranscriptMode(event.target.value as TranscriptMode)
                      }
                      title="Auto adapts to the clip. Song mode prefers lyric-safe transcription."
                    >
                      {TRANSCRIPT_MODE_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="transcriptControlField">
                    <span>Speed</span>
                    <select
                      value={transcriptSpeed}
                      disabled={generatingTranscript}
                      onChange={(event) =>
                        setTranscriptSpeed(
                          event.target.value as TranscriptSpeed,
                        )
                      }
                      title="Fast skips voice isolation. Normal uses voice isolation for songs when needed."
                    >
                      {TRANSCRIPT_SPEED_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="transcriptControlField">
                    <span>Transcript Language</span>
                    <select
                      value={transcriptLanguage}
                      disabled={generatingTranscript}
                      onChange={(event) => {
                        setTranscriptLanguage(event.target.value);
                        try {
                          localStorage.setItem(
                            "clipmind_transcript_lang",
                            event.target.value,
                          );
                        } catch {}
                      }}
                      title="Choose transcript language (Auto uses model detection)"
                    >
                      {TRANSCRIPT_LANGUAGE_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button
                    className="primaryBtn transcriptGenerateBtn"
                    onClick={() => void generateTranscript()}
                    disabled={!selectedVideoAsset || generatingTranscript}
                  >
                    <Wand2 size={16} />
                    {generatingTranscript
                      ? `Generating ${transcriptProgress}%`
                      : "Generate Transcript"}
                  </button>
                  {transcript && (
                    <button
                      type="button"
                      className="transcriptRegenerateBtn"
                      onClick={() =>
                        void generateTranscript({ forceRegenerate: true })
                      }
                      disabled={!selectedVideoAsset || generatingTranscript}
                      title="Run full speech recognition again (ignores cached transcript)"
                    >
                      <RefreshCw size={16} />
                      Regenerate
                    </button>
                  )}
                </div>
                <p className="muted transcriptEstimateHint">
                  Estimated transcript time:{" "}
                  <strong>{transcriptRuntimeHint}</strong>.{" "}
                  {transcriptModeDetail(transcriptMode, transcriptSpeed)}
                  {transcriptJob?.status === "running" && transcriptStageLabel
                    ? ` Current stage: ${transcriptStageLabel}`
                    : ""}
                </p>
                {transcriptJob &&
                  (generatingTranscript ||
                    transcriptJob.status === "failed") && (
                    <div className="transcriptJobCard" aria-live="polite">
                      <div className="transcriptJobTop">
                        <span
                          className={`transcriptJobBadge ${transcriptJob.status}`}
                        >
                          {transcriptJob.status}
                        </span>
                        <span className="transcriptJobProgress">
                          {transcriptProgress}%
                        </span>
                        <span className="transcriptJobElapsed">
                          {formatSeconds(transcriptElapsedSec)}
                        </span>
                      </div>
                      <div className="jobProgressBar" aria-hidden="true">
                        <span
                          className="jobProgressFill"
                          style={{ width: `${transcriptProgress}%` }}
                        />
                      </div>
                      <p className="transcriptJobMessage">
                        {transcriptJob.status === "completed" ? (
                          <>
                            <Check size={14} aria-hidden="true" />
                            <span>Transcript ready.</span>
                          </>
                        ) : (
                          transcriptStatusMessage
                        )}
                      </p>
                      {transcriptJob.status === "failed" && (
                        <div className="jobRetryRow">
                          <button
                            type="button"
                            className="jobRetryBtn"
                            onClick={retryTranscriptGeneration}
                            disabled={
                              !selectedVideoAsset || generatingTranscript
                            }
                          >
                            <RefreshCw size={14} aria-hidden="true" />
                            Retry transcript
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                {transcript?.language && !generatingTranscript && (
                  <div
                    className="detectedLanguageBadge"
                    title="Detected transcript language"
                  >
                    <span className="detectedLanguageLabel">
                      <Globe size={12} aria-hidden="true" />
                      Detected:
                    </span>
                    <span className="detectedLanguageValue">
                      {transcript.language.charAt(0).toUpperCase() +
                        transcript.language.slice(1)}
                    </span>
                    {transcript.source?.toLowerCase().includes("lyrics_ref") ? (
                      <span className="detectedLanguageHint">
                        {" "}
                        · Reference lyrics
                      </span>
                    ) : transcript.source?.toLowerCase().includes("groq") ? (
                      <span className="detectedLanguageHint"> · Raw ASR</span>
                    ) : null}
                    {transcriptLanguage === "auto" && (
                      <button
                        type="button"
                        className="lockLanguageBtn"
                        title={`Pin language to ${transcript.language} for future transcriptions`}
                        onClick={() => {
                          setTranscriptLanguage(transcript.language!);
                          try {
                            localStorage.setItem(
                              "clipmind_transcript_lang",
                              transcript.language!,
                            );
                          } catch {}
                        }}
                      >
                        Pin
                      </button>
                    )}
                  </div>
                )}
                {!!transcript &&
                  transcript.mixed_script &&
                  transcriptScriptSummary.length > 1 && (
                    <div
                      className="detectedLanguageBadge transcriptScriptBadge"
                      title="Transcript contains more than one script family. This is common in mixed-language or code-switched videos."
                    >
                      <span className="detectedLanguageLabel">
                        <Globe size={12} aria-hidden="true" />
                        Script mix:
                      </span>
                      <span className="detectedLanguageValue">
                        {transcriptScriptSummary.join(" + ")}
                      </span>
                      <span className="detectedLanguageHint">
                        {" "}
                        · review mixed-language regions carefully
                      </span>
                    </div>
                  )}
                <div className="featureLauncher">
                  {FEATURE_TAB_ITEMS.map(({ id, label, icon: Icon }) => (
                    <button
                      key={id}
                      className={
                        activeFeatureTab === id && featureDrawerOpen
                          ? "active"
                          : ""
                      }
                      onClick={() => openFeatureDrawer(id)}
                    >
                      <Icon size={14} strokeWidth={1.9} aria-hidden="true" />
                      <span>{label}</span>
                    </button>
                  ))}
                </div>
                {!transcript && (
                  <p className="muted">
                    Generate transcript from an uploaded video to start
                    text-based editing.
                  </p>
                )}
                {transcript && (
                  <>
                    <p className="muted hint">
                      <strong>Click</strong> word to select & seek &nbsp;·&nbsp;
                      <strong>Shift+click</strong> range &nbsp;·&nbsp;
                      <strong>Drag</strong> to select &nbsp;·&nbsp;
                      <strong>Double-click</strong> to edit text &nbsp;·&nbsp;
                      <strong>Del/Backspace</strong> delete &nbsp;·&nbsp;
                      <strong>Ctrl+Z</strong> undo
                    </p>

                    {transcript.is_mock && (
                      <p className="warning">
                        Cloud transcription failed — check your API keys
                        (GROQ_API_KEY / SARVAM_API_KEY) and network, then
                        regenerate.
                      </p>
                    )}
                    <TranscriptQualityPanel
                      transcript={transcript}
                      selectedLanguage={transcriptLanguage}
                      scriptSummary={transcriptScriptSummary}
                      captionBlockCount={captionBlocks.length}
                      reviewWordCount={reviewWordCount}
                      weakQualityCount={weakQualityCount}
                      lowConfidenceOnlyCount={lowConfidenceOnlyCount}
                      lowConfidenceCount={lowConfidenceCount}
                      lowConfidenceRatio={lowConfidenceRatio}
                      shouldWarnLowConfidence={shouldWarnLowConfidence}
                      issueRegions={transcriptIssueRegions}
                      onReviewWeakWords={() => reviewWeakWords(weakReviewIndex)}
                      onReviewNextWeakWord={reviewNextWeakWord}
                      onFocusRegion={focusTranscriptRegion}
                      formatSeconds={formatSeconds}
                      regionLabel={transcriptRegionLabel}
                    />

                    {/* ── Search bar ────────────────────────────────── */}
                    <div className="searchBar">
                      <svg
                        className="searchIcon"
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <circle cx="11" cy="11" r="8" />
                        <path d="m21 21-4.35-4.35" />
                      </svg>
                      <input
                        id="transcript-search"
                        type="text"
                        placeholder="Search words... (Ctrl+F)"
                        value={searchQuery}
                        onChange={(e) => {
                          setSearchQuery(e.target.value);
                          setSearchMatchIndex(0);
                        }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter")
                            navigateSearch(e.shiftKey ? -1 : 1);
                          if (e.key === "Escape") {
                            setSearchQuery("");
                            (e.target as HTMLInputElement).blur();
                          }
                        }}
                      />
                      {searchQuery && (
                        <span className="searchCount">
                          {searchMatchIds.length
                            ? `${searchMatchIndex + 1}/${searchMatchIds.length}`
                            : "0 matches"}
                          <button
                            className="searchNav"
                            onClick={() => navigateSearch(-1)}
                            title="Previous"
                            aria-label="Previous match"
                          >
                            <ChevronUp size={12} aria-hidden="true" />
                          </button>
                          <button
                            className="searchNav"
                            onClick={() => navigateSearch(1)}
                            title="Next"
                            aria-label="Next match"
                          >
                            <ChevronDown size={12} aria-hidden="true" />
                          </button>
                        </span>
                      )}
                    </div>
                    {selectedTranscriptRange && (
                      <div className="transcriptSelectionMeta">
                        <span>
                          {selectedTranscriptRange.wordCount} word
                          {selectedTranscriptRange.wordCount === 1
                            ? ""
                            : "s"}{" "}
                          selected
                        </span>
                        <span>
                          {formatPreciseSeconds(
                            selectedTranscriptRange.startSec,
                          )}{" "}
                          -{" "}
                          {formatPreciseSeconds(selectedTranscriptRange.endSec)}
                        </span>
                      </div>
                    )}

                    {/* ── Action toolbar ────────────────────────────── */}
                    <div className="wordActions toolbar">
                      <button
                        onClick={markSelectionDeleted}
                        disabled={!selectedWordIds.size}
                        title="Delete selected words"
                      >
                        <Trash2
                          size={14}
                          strokeWidth={1.9}
                          aria-hidden="true"
                        />
                        <span>Delete</span>
                      </button>
                      <button
                        onClick={() =>
                          void applyTranscriptRangeUpdate("delete")
                        }
                        disabled={
                          !selectedTranscriptRange || updatingTranscriptRange
                        }
                        title="Remove selected text from transcript without cutting the timeline"
                      >
                        <Trash2
                          size={14}
                          strokeWidth={1.9}
                          aria-hidden="true"
                        />
                        <span>
                          {updatingTranscriptRange
                            ? "Deleting..."
                            : "Delete Text"}
                        </span>
                      </button>
                      <button
                        onClick={restoreSelection}
                        disabled={!selectedWordIds.size}
                        title="Restore selected words"
                      >
                        <RotateCcw
                          size={14}
                          strokeWidth={1.9}
                          aria-hidden="true"
                        />
                        <span>Restore</span>
                      </button>
                      <button
                        onClick={restoreAllText}
                        disabled={!deletedWordIds.size}
                        title="Restore all deleted words"
                      >
                        <RefreshCw
                          size={14}
                          strokeWidth={1.9}
                          aria-hidden="true"
                        />
                        <span>Restore All</span>
                      </button>
                      <div className="toolbarSep" />
                      <button
                        className="brollSelectionBtn"
                        onClick={() => void suggestBrollForSelection()}
                        disabled={
                          !selectedWordIds.size ||
                          suggestingBrollSelection ||
                          suggestingBroll ||
                          !transcript
                        }
                        title="Get AI B-roll suggestions for the selected transcript words"
                      >
                        <Film
                          size={14}
                          strokeWidth={1.9}
                          aria-hidden="true"
                        />
                        <span>
                          {suggestingBrollSelection
                            ? "Finding B-roll…"
                            : "Get B-roll"}
                        </span>
                      </button>
                      <div className="toolbarSep" />
                      <button
                        onClick={removeFillerWords}
                        disabled={!fillerWordIds.size || applyingCut}
                        title="Remove um, uh, like, etc."
                      >
                        <Scissors
                          size={14}
                          strokeWidth={1.9}
                          aria-hidden="true"
                        />
                        <span>
                          Remove Fillers
                          {fillerWordIds.size > 0 && (
                            <span className="badge">{fillerWordIds.size}</span>
                          )}
                        </span>
                      </button>
                      <div className="toolbarSep" />
                      <button
                        onClick={() => void undo()}
                        disabled={!canUndoAction}
                        title="Undo (Ctrl+Z)"
                      >
                        <Undo2 size={14} strokeWidth={1.9} aria-hidden="true" />
                        <span>Undo</span>
                      </button>
                      <button
                        onClick={() => void redo()}
                        disabled={!canRedoAction}
                        title="Redo (Ctrl+Shift+Z)"
                      >
                        <Redo2 size={14} strokeWidth={1.9} aria-hidden="true" />
                        <span>Redo</span>
                      </button>
                      <div className="toolbarSep" />
                      <button
                        onClick={() =>
                          void applyCut(deletedSignature, keptWordIds, {
                            manual: true,
                          })
                        }
                        disabled={applyingCut || !transcript}
                      >
                        {applyingCut ? (
                          "Applying..."
                        ) : (
                          <>
                            <ScissorsLineDashed
                              size={14}
                              strokeWidth={1.9}
                              aria-hidden="true"
                            />
                            <span>Apply Cut</span>
                          </>
                        )}
                      </button>
                    </div>

                    {suggestingBrollSelection && (
                      <div
                        className="brollSelectionProgress"
                        role="status"
                        aria-live="polite"
                      >
                        <div className="brollSelectionProgressTop">
                          <span>{brollSuggestionMessage}</span>
                          <strong>{brollSuggestionProgress}%</strong>
                        </div>
                        <div
                          className="jobProgressBar"
                          role="progressbar"
                          aria-label="Get B-roll progress"
                          aria-valuemin={0}
                          aria-valuemax={100}
                          aria-valuenow={brollSuggestionProgress}
                        >
                          <span
                            className="jobProgressFill"
                            style={{ width: `${brollSuggestionProgress}%` }}
                          />
                        </div>
                      </div>
                    )}

                    {/* ── Interactive word grid ─────────────────────── */}
                    <div
                      className="transcriptBox"
                      ref={transcriptBoxRef}
                      onMouseLeave={() => {
                        isDragging.current = false;
                      }}
                      onWheel={() => {
                        if (!transcriptProgrammaticScrollRef.current) {
                          transcriptFollowPlaybackRef.current = false;
                        }
                      }}
                      onTouchMove={() => {
                        if (!transcriptProgrammaticScrollRef.current) {
                          transcriptFollowPlaybackRef.current = false;
                        }
                      }}
                      onScroll={() => {
                        if (!transcriptProgrammaticScrollRef.current) {
                          transcriptFollowPlaybackRef.current = false;
                        }
                      }}
                    >
                      {transcriptWordNodes}
                    </div>

                    {/* ── Sentence shortcuts ────────────────────────── */}
                    <details className="shortcutSection">
                      <summary>
                        <h3>Sentence Shortcuts ({sentenceBlocks.length})</h3>
                      </summary>
                      <div className="shortcutList">
                        {sentenceShortcutNodes}
                      </div>
                    </details>

                    <details className="shortcutSection">
                      <summary>
                        <h3>Paragraph Shortcuts ({paragraphBlocks.length})</h3>
                      </summary>
                      <div className="shortcutList">
                        {paragraphShortcutNodes}
                      </div>
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
                    <button
                      type="button"
                      onClick={() => setFeatureDrawerOpen(false)}
                    >
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
                          const selected = videoAssets.find(
                            (asset) => asset.id === nextId,
                          );
                          if (selected) {
                            setPreviewUrl(
                              resolveMediaPath(selected.storage_path),
                            );
                          }
                        }}
                      >
                        {!videoAssets.length && (
                          <option value="">No uploaded videos</option>
                        )}
                        {videoAssets.map((asset) => (
                          <option key={asset.id} value={asset.id}>
                            {asset.filename}
                          </option>
                        ))}
                      </select>
                    </label>
                    <p className="inspectorStats">
                      {project.timeline.resolution.width}x
                      {project.timeline.resolution.height} ·{" "}
                      {Math.round(project.timeline.fps)} fps ·{" "}
                      {formatSeconds(project.timeline.duration_sec)}
                    </p>
                  </div>

                  {(selectedTimelineClipDetails ||
                    selectedCaptionBlockDetails ||
                    selectedBrollClip) && (
                    <div className="creatorSelectionCard">
                      {selectedBrollClip &&
                        (() => {
                          const clip = selectedBrollClip;
                          const clipBusy = isBrollTimelineClipBusy(clip.id);
                          const clipDuration = clipTimelineDurationSec(clip);
                          const clipOpacity =
                            typeof clip.broll_opacity === "number"
                              ? clip.broll_opacity
                              : 1;
                          const draftStart =
                            brollDraftStartById[clip.id] ??
                            formatFixedSec(clip.timeline_start_sec);
                          const draftDuration =
                            brollDraftDurationById[clip.id] ??
                            formatFixedSec(clipDuration);
                          const draftOpacity =
                            brollDraftOpacityById[clip.id] ?? clipOpacity;
                          const source = mediaById.get(clip.asset_id);
                          return (
                            <>
                              <div className="creatorSelectionHead">
                                <div>
                                  <p className="inspectorEyebrow">
                                    B-roll Inspector
                                  </p>
                                  <h4>{source?.filename ?? clip.asset_id}</h4>
                                </div>
                                <button
                                  type="button"
                                  className="secondaryBtn dangerBtn"
                                  disabled={clipBusy}
                                  onClick={() =>
                                    void removeBrollClipById(clip.id)
                                  }
                                >
                                  Remove
                                </button>
                              </div>
                              <p className="muted">
                                {formatSeconds(clip.timeline_start_sec)} start ·{" "}
                                {formatSeconds(clipDuration)} duration ·{" "}
                                {(draftOpacity * 100).toFixed(0)}% opacity
                              </p>
                              <div className="creatorSelectionFields">
                                <label>
                                  Start (sec)
                                  <input
                                    type="number"
                                    min={0}
                                    step={0.05}
                                    value={draftStart}
                                    disabled={clipBusy}
                                    onChange={(event) =>
                                      setBrollDraftStartById((prev) => ({
                                        ...prev,
                                        [clip.id]: event.target.value,
                                      }))
                                    }
                                    onBlur={() => void commitBrollStart(clip)}
                                    onKeyDown={(event) => {
                                      if (event.key === "Enter")
                                        event.currentTarget.blur();
                                    }}
                                  />
                                </label>
                                <label>
                                  Duration (sec)
                                  <input
                                    type="number"
                                    min={0.1}
                                    step={0.05}
                                    value={draftDuration}
                                    disabled={clipBusy}
                                    onChange={(event) =>
                                      setBrollDraftDurationById((prev) => ({
                                        ...prev,
                                        [clip.id]: event.target.value,
                                      }))
                                    }
                                    onBlur={() =>
                                      void commitBrollDuration(clip)
                                    }
                                    onKeyDown={(event) => {
                                      if (event.key === "Enter")
                                        event.currentTarget.blur();
                                    }}
                                  />
                                </label>
                                <label>
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
                                      void commitBrollOpacity(
                                        clip,
                                        Number(event.currentTarget.value),
                                      )
                                    }
                                    onTouchEnd={(event) =>
                                      void commitBrollOpacity(
                                        clip,
                                        Number(event.currentTarget.value),
                                      )
                                    }
                                  />
                                </label>
                              </div>
                              <div className="creatorSelectionActions">
                                <button
                                  type="button"
                                  className="secondaryBtn"
                                  onClick={() =>
                                    handleTimelineSeek(clip.timeline_start_sec)
                                  }
                                >
                                  Jump to Clip
                                </button>
                                <button
                                  type="button"
                                  className="secondaryBtn"
                                  disabled={clipBusy || !!brollActionKey}
                                  onClick={() =>
                                    void rerollBrollFromTimelineClip(clip.id)
                                  }
                                >
                                  {brollActionKey === `reroll-clip:${clip.id}`
                                    ? "Rerolling..."
                                    : "Re-roll"}
                                </button>
                              </div>
                            </>
                          );
                        })()}
                      {selectedTimelineClipDetails && (
                        <>
                          <div className="creatorSelectionHead">
                            <div>
                              <p className="inspectorEyebrow">Clip Inspector</p>
                              <h4>
                                {selectedTimelineClipDetails.lane.label} ·{" "}
                                {selectedTimelineClipDetails.source?.filename ??
                                  "Timeline clip"}
                              </h4>
                            </div>
                            <button
                              type="button"
                              className="secondaryBtn"
                              onClick={toggleSelectedTimelineClipMute}
                            >
                              {selectedTimelineClipDetails.clip.audio.mute
                                ? "Unmute"
                                : "Mute"}
                            </button>
                          </div>
                          <p className="muted">
                            {formatSeconds(
                              selectedTimelineClipDetails.clip
                                .timeline_start_sec,
                            )}{" "}
                            start ·{" "}
                            {formatSeconds(
                              selectedTimelineClipDetails.durationSec,
                            )}{" "}
                            duration
                          </p>
                          <div className="creatorSelectionActions">
                            <button
                              type="button"
                              className="secondaryBtn"
                              onClick={splitSelectedTimelineClip}
                            >
                              Split at Playhead
                            </button>
                            <button
                              type="button"
                              className="secondaryBtn dangerBtn"
                              onClick={deleteSelectedTimelineClip}
                            >
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
                                    onClick={() =>
                                      setSelectedTimelineClipSpeed(speed)
                                    }
                                  >
                                    {speed}x
                                  </button>
                                ))}
                              </div>
                            </label>
                            <label>
                              Volume{" "}
                              {(
                                selectedTimelineClipDetails.clip.audio.volume *
                                100
                              ).toFixed(0)}
                              %
                              <div className="chipRow">
                                {[0, 0.5, 1, 1.25, 1.5].map((volume) => (
                                  <button
                                    key={volume}
                                    type="button"
                                    className={`chipBtn ${Math.abs(selectedTimelineClipDetails.clip.audio.volume - volume) < 0.01 ? "active" : ""}`}
                                    onClick={() =>
                                      setSelectedTimelineClipVolume(volume)
                                    }
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
                              <p className="inspectorEyebrow">
                                Caption Inspector
                              </p>
                              <h4>Caption Block</h4>
                            </div>
                            <button
                              type="button"
                              className="secondaryBtn dangerBtn"
                              onClick={deleteSelectedCaptionBlock}
                            >
                              Delete Caption
                            </button>
                          </div>
                          <label className="creatorSelectionFields">
                            Caption text
                            <textarea
                              rows={3}
                              value={
                                editingCaptionOverlayId ===
                                selectedCaptionBlockDetails.overlay.id
                                  ? editingCaptionText
                                  : selectedCaptionBlockDetails.overlay.text
                              }
                              onChange={(event) => {
                                if (
                                  editingCaptionOverlayId !==
                                  selectedCaptionBlockDetails.overlay.id
                                ) {
                                  startCaptionEditing({
                                    overlayId:
                                      selectedCaptionBlockDetails.overlay.id,
                                    clipId: selectedCaptionBlockDetails.clip.id,
                                    laneId: selectedCaptionBlockDetails.lane.id,
                                    laneLabel:
                                      selectedCaptionBlockDetails.lane.label,
                                    text: selectedCaptionBlockDetails.overlay
                                      .text,
                                    style:
                                      selectedCaptionBlockDetails.overlay.style,
                                  });
                                }
                                setEditingCaptionText(event.target.value);
                              }}
                              onBlur={() => void commitCaptionEdit()}
                              onKeyDown={(event) => {
                                if (event.key === "Enter" && !event.shiftKey) {
                                  event.preventDefault();
                                  void commitCaptionEdit();
                                }
                                if (event.key === "Escape") {
                                  event.preventDefault();
                                  cancelCaptionEdit();
                                }
                              }}
                            />
                          </label>
                          <p className="muted">
                            Lower-third safe area ·{" "}
                            {formatSeconds(
                              selectedCaptionBlockDetails.timelineStartSec,
                            )}{" "}
                            start ·{" "}
                            {formatSeconds(
                              selectedCaptionBlockDetails.durationSec,
                            )}{" "}
                            duration
                          </p>
                          <div className="creatorSelectionActions">
                            <button
                              type="button"
                              className="secondaryBtn"
                              onClick={() =>
                                handleTimelineSeek(
                                  selectedCaptionBlockDetails.timelineStartSec,
                                )
                              }
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
                          <Icon
                            className="featureTabIcon"
                            size={15}
                            strokeWidth={1.9}
                            aria-hidden="true"
                          />
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
                        <p className="muted">
                          Only backend-ready tools are shown here, so every
                          action in this section is wired to a real API route.
                        </p>
                        <div className="actionGrid actionGridWide">
                          {AI_ACTION_ITEMS.map(
                            ({ action, label, desc, icon: Icon, primary }) => (
                              <button
                                key={action}
                                className={`actionCard ${primary ? "actionCardPrimary" : ""}`}
                                onClick={() => void runVibeAction(action)}
                                disabled={
                                  !selectedVideoAsset || runningAction !== null
                                }
                              >
                                <span className="actionIcon">
                                  <Icon
                                    size={18}
                                    strokeWidth={1.9}
                                    aria-hidden="true"
                                  />
                                </span>
                                <span className="actionLabel">
                                  {runningAction === action
                                    ? "Applying..."
                                    : label}
                                </span>
                                <span className="actionDesc">{desc}</span>
                              </button>
                            ),
                          )}
                        </div>
                      </section>
                    )}

                    {/* ── Captions Tab ───────────────────────────── */}
                    {activeFeatureTab === "captions" && (
                      <section className="aiPanel captionsPanel active">
                        <h3>Caption Styles</h3>
                        <p className="muted">
                          Select a style and apply. 9 curated presets with tuned
                          timing, color, and typography.
                        </p>

                        <TranscriptQualityPanel
                          transcript={transcript}
                          selectedLanguage={transcriptLanguage}
                          scriptSummary={transcriptScriptSummary}
                          captionBlockCount={captionBlocks.length}
                          reviewWordCount={reviewWordCount}
                          weakQualityCount={weakQualityCount}
                          lowConfidenceOnlyCount={lowConfidenceOnlyCount}
                          lowConfidenceCount={lowConfidenceCount}
                          lowConfidenceRatio={lowConfidenceRatio}
                          shouldWarnLowConfidence={shouldWarnLowConfidence}
                          issueRegions={transcriptIssueRegions}
                          surface="captions"
                          compact
                          formatSeconds={formatSeconds}
                          regionLabel={transcriptRegionLabel}
                        />

                        <div className="captionStyleGrid">
                          {CAPTION_STYLE_PRESETS.map((style) => {
                            const isActive = captionStyle === style.id;
                            // Parse ASS color (&H00BBGGRR) to CSS hex for preview
                            const primaryHex =
                              style.config.primary_color.startsWith("&H")
                                ? (() => {
                                    const raw = style.config.primary_color
                                      .replace("&H", "")
                                      .replace("00", "");
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
                                style={
                                  {
                                    "--caption-accent": style.color,
                                  } as React.CSSProperties
                                }
                              >
                                <div className="captionPreviewBox">
                                  <div
                                    className={`captionPreviewScene ${style.preview_class}`}
                                  >
                                    <span
                                      className="captionPreviewText"
                                      style={{
                                        color: primaryHex,
                                        fontFamily:
                                          style.config.font_name.replace(
                                            "-",
                                            " ",
                                          ) + ", sans-serif",
                                        fontSize: `${Math.min(style.config.font_size * 0.7, 18)}px`,
                                        textShadow:
                                          style.config.shadow > 0
                                            ? `0 1px ${style.config.shadow}px rgba(0,0,0,0.7)`
                                            : "none",
                                        WebkitTextStroke:
                                          style.config.outline_width > 0
                                            ? `${Math.min(style.config.outline_width * 0.3, 1)}px rgba(0,0,0,0.5)`
                                            : "none",
                                      }}
                                    >
                                      <span>{style.preview_words[0]}</span>
                                      <span className="captionPreviewHighlight">
                                        {style.preview_words[1]}
                                      </span>
                                      <span>{style.preview_words[2]}</span>
                                    </span>
                                    <span
                                      className="captionPreviewPulse"
                                      aria-hidden="true"
                                    />
                                  </div>
                                </div>
                                <span className="captionStyleName">
                                  {style.name}
                                </span>
                                <span className="captionStyleDesc">
                                  {style.desc}
                                </span>
                                {isActive && (
                                  <span className="captionActiveCheck">
                                    <Check
                                      size={12}
                                      strokeWidth={2.4}
                                      aria-hidden="true"
                                    />
                                  </span>
                                )}
                              </button>
                            );
                          })}
                        </div>

                        {captionResultInfo && (
                          <div className="captionResultBadge">
                            <span className="captionResultIcon">
                              <Check
                                size={14}
                                strokeWidth={2.4}
                                aria-hidden="true"
                              />
                            </span>
                            <span>{captionResultInfo}</span>
                          </div>
                        )}

                        <div className="captionApplyRow">
                          <button
                            className="primaryBtn captionApplyBtn"
                            onClick={() => {
                              setCaptionResultInfo(null);
                              void runVibeAction("add_subtitles");
                            }}
                            disabled={
                              !selectedVideoAsset ||
                              runningAction !== null ||
                              removingCaptions
                            }
                          >
                            {runningAction === "add_subtitles" ? (
                              <>
                                <span className="captionSpinner" />
                                Generating...
                              </>
                            ) : (
                              <>
                                <Captions
                                  size={16}
                                  strokeWidth={1.9}
                                  aria-hidden="true"
                                />
                                <span>{`Apply "${selectedCaptionStyleName}" Captions`}</span>
                              </>
                            )}
                          </button>
                          <button
                            className="secondaryBtn captionRemoveBtn"
                            onClick={() => void removeCaptions()}
                            disabled={
                              !selectedVideoAsset ||
                              runningAction !== null ||
                              removingCaptions
                            }
                            title="Remove all captions from the video"
                          >
                            {removingCaptions ? (
                              "Removing..."
                            ) : (
                              <>
                                <Trash2
                                  size={15}
                                  strokeWidth={1.9}
                                  aria-hidden="true"
                                />
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
                        <p className="muted">
                          Render your final video. Estimated render time:{" "}
                          <strong>{exportRuntimeHint}</strong>.
                        </p>
                        <TranscriptQualityPanel
                          transcript={transcript}
                          selectedLanguage={transcriptLanguage}
                          scriptSummary={transcriptScriptSummary}
                          captionBlockCount={captionBlocks.length}
                          reviewWordCount={reviewWordCount}
                          weakQualityCount={weakQualityCount}
                          lowConfidenceOnlyCount={lowConfidenceOnlyCount}
                          lowConfidenceCount={lowConfidenceCount}
                          lowConfidenceRatio={lowConfidenceRatio}
                          shouldWarnLowConfidence={shouldWarnLowConfidence}
                          issueRegions={transcriptIssueRegions}
                          surface="export"
                          compact
                          formatSeconds={formatSeconds}
                          regionLabel={transcriptRegionLabel}
                        />
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
                                  }}
                                >
                                  {ratio}
                                </button>
                              ))}
                            </div>
                          </div>
                          {exportAspectRatio === "9:16" && (
                            <div className="smartReframeCard">
                              <div>
                                <strong>Auto Frame</strong>
                                <p>
                                  Fill a 9:16 Reel by cropping wide clips. It
                                  never crops a 16:9 preview or export.
                                </p>
                              </div>
                              <button
                                type="button"
                                className={`smartReframeBtn ${autoFraming ? "active" : ""}`}
                                aria-pressed={autoFraming}
                                onClick={() => void toggleAutoFraming()}
                                disabled={!project || smartReframing || exportingVideo}
                              >
                                {smartReframing
                                  ? "Analysing framing..."
                                  : autoFraming
                                    ? "Auto Frame: On"
                                    : "Auto Frame: Off"}
                              </button>
                            </div>
                          )}
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
                              {(["low", "medium", "high", "max"] as const).map(
                                (q) => (
                                  <button
                                    key={q}
                                    className={`exportOption ${exportQuality === q ? "active" : ""}`}
                                    onClick={() => setExportQuality(q)}
                                  >
                                    {q.charAt(0).toUpperCase() + q.slice(1)}
                                  </button>
                                ),
                              )}
                            </div>
                          </div>
                        </div>
                        {exportJob && (
                          <div className="exportJobStatus">
                            <div className="exportJobTop">
                              <span className="exportJobLabel">
                                Export Job:
                              </span>
                              <span
                                className={`exportJobBadge ${exportJob.status}`}
                              >
                                {exportJob.status}
                              </span>
                              <span className="exportJobProgress">
                                {exportProgress}%
                              </span>
                            </div>
                            <div className="jobProgressBar" aria-hidden="true">
                              <span
                                className="jobProgressFill"
                                style={{ width: `${exportProgress}%` }}
                              />
                            </div>
                            <span className="exportJobMessage">
                              {exportStatusMessage}
                            </span>
                            {exportJob.status === "failed" && (
                              <div className="jobRetryRow">
                                <button
                                  type="button"
                                  className="jobRetryBtn"
                                  onClick={() => void exportVideo()}
                                  disabled={!project || exportingVideo}
                                >
                                  <RefreshCw size={14} aria-hidden="true" />
                                  Retry export
                                </button>
                              </div>
                            )}
                          </div>
                        )}
                        <div className="exportGuideRow">
                          <button
                            className={`exportGuideBtn ${showExportFrameGuide ? "active" : ""}`}
                            onClick={() =>
                              setShowExportFrameGuide((prev) => !prev)
                            }
                            type="button"
                          >
                            {showExportFrameGuide
                              ? "Hide export guide"
                              : "Show export guide"}
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
                                <Download
                                  size={16}
                                  strokeWidth={1.9}
                                  aria-hidden="true"
                                />
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
                        <p className="muted">
                          Add cutaway visuals after your transcript edit. Quick
                          Edit does not include B-roll — use this studio when
                          you are ready.
                        </p>
                        {brollSetupWarning && (
                          <div className="brollSetupBanner" role="status">
                            <strong>{brollSetupWarning.title}</strong>
                            <p>{brollSetupWarning.detail}</p>
                          </div>
                        )}
                        {lastBrollAutoApplySkips.length > 0 && (
                          <div className="brollSkipSummary">
                            <strong>
                              {lastBrollAutoApplySkips.length} slot
                              {lastBrollAutoApplySkips.length === 1
                                ? ""
                                : "s"}{" "}
                              skipped in last auto-apply
                            </strong>
                            <ul>
                              {lastBrollAutoApplySkips
                                .slice(0, 4)
                                .map((item) => (
                                  <li key={item.slot_id}>
                                    {trimInlineText(
                                      item.concept_text || "Untitled slot",
                                      48,
                                    )}{" "}
                                    — {autoApplySkipReasonLabel(item.reason)}
                                    {item.detail ? `: ${item.detail}` : ""}
                                  </li>
                                ))}
                            </ul>
                          </div>
                        )}
                        <div className="wordActions">
                          <button
                            className="primaryBtn"
                            onClick={() => void autoApplyBroll()}
                            disabled={
                              !project ||
                              !transcript ||
                              autoApplyingBroll ||
                              loadingBrollSlots ||
                              suggestingBroll ||
                              suggestingBrollSelection ||
                              syncingBroll ||
                              undoingBroll
                            }
                            title="Generate slots, auto-pick confident candidates, and sync to timeline in one step."
                          >
                            {autoApplyingBroll ? (
                              "Auto-applying..."
                            ) : (
                              <>
                                <Wand2
                                  size={16}
                                  strokeWidth={1.9}
                                  aria-hidden="true"
                                />
                                <span>Auto B-roll (1-click)</span>
                              </>
                            )}
                          </button>
                          <button
                            onClick={() => void suggestBroll()}
                            disabled={
                              !project ||
                              !transcript ||
                              suggestingBroll ||
                              suggestingBrollSelection ||
                              loadingBrollSlots ||
                              autoApplyingBroll
                            }
                          >
                            {suggestingBroll ? (
                              "Suggesting..."
                            ) : (
                              <>
                                <Sparkles
                                  size={16}
                                  strokeWidth={1.9}
                                  aria-hidden="true"
                                />
                                <span>Suggest B-roll</span>
                              </>
                            )}
                          </button>
                          <button
                            onClick={() =>
                              project &&
                              transcript &&
                              void refreshBrollSlots(project.id, transcript.id)
                            }
                            disabled={
                              !project || !transcript || loadingBrollSlots
                            }
                          >
                            {loadingBrollSlots ? (
                              "Refreshing..."
                            ) : (
                              <>
                                <RefreshCw
                                  size={16}
                                  strokeWidth={1.9}
                                  aria-hidden="true"
                                />
                                <span>Refresh</span>
                              </>
                            )}
                          </button>
                          <button
                            onClick={() => void syncBrollToTimeline()}
                            disabled={
                              !project ||
                              syncingBroll ||
                              autoApplyingBroll ||
                              undoingBroll
                            }
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
                                <Clapperboard
                                  size={16}
                                  strokeWidth={1.9}
                                  aria-hidden="true"
                                />
                                <span>Sync to Timeline</span>
                              </>
                            )}
                          </button>
                          <button
                            onClick={() => void undoBrollLayer()}
                            disabled={
                              !project ||
                              undoingBroll ||
                              syncingBroll ||
                              autoApplyingBroll
                            }
                            title="Undo only the latest B-roll transaction while preserving other timeline layers."
                          >
                            {undoingBroll ? (
                              "Undoing..."
                            ) : (
                              <>
                                <RotateCcw
                                  size={16}
                                  strokeWidth={1.9}
                                  aria-hidden="true"
                                />
                                <span>Undo B-roll Layer</span>
                              </>
                            )}
                          </button>
                        </div>
                        {brollGenerationActive && (
                          <div
                            className="brollGenerationProgress"
                            role="status"
                            aria-live="polite"
                          >
                            <div className="brollGenerationProgressTop">
                              <span>{brollSuggestionMessage}</span>
                              <strong>{brollSuggestionProgress}%</strong>
                            </div>
                            <div
                              className="jobProgressBar"
                              role="progressbar"
                              aria-label="B-roll generation progress"
                              aria-valuemin={0}
                              aria-valuemax={100}
                              aria-valuenow={brollSuggestionProgress}
                            >
                              <span
                                className="jobProgressFill"
                                style={{
                                  width: `${brollSuggestionProgress}%`,
                                }}
                              />
                            </div>
                          </div>
                        )}
                        <div className="brollControlsRow">
                          <div className="brollIntensity">
                            <span className="brollSyncLabel">Auto mode:</span>
                            <select
                              value={brollAutoMode}
                              onChange={(event) =>
                                setBrollAutoMode(
                                  event.target.value as BrollAutoMode,
                                )
                              }
                              disabled={
                                autoApplyingBroll ||
                                suggestingBroll ||
                                suggestingBrollSelection
                              }
                            >
                              <option value="fast">
                                Fast (&lt;1 min target)
                              </option>
                              <option value="balanced">Balanced</option>
                              <option value="creative">
                                Creative (quality)
                              </option>
                            </select>
                          </div>
                          <div className="brollSyncMode">
                            <span className="brollSyncLabel">Sync mode:</span>
                            <button
                              type="button"
                              className={
                                brollSyncMode === "replace"
                                  ? "brollSyncOption active"
                                  : "brollSyncOption"
                              }
                              onClick={() => setBrollSyncMode("replace")}
                              disabled={syncingBroll}
                            >
                              Replace
                            </button>
                            <button
                              type="button"
                              className={
                                brollSyncMode === "append"
                                  ? "brollSyncOption active"
                                  : "brollSyncOption"
                              }
                              onClick={() => setBrollSyncMode("append")}
                              disabled={syncingBroll}
                            >
                              Append
                            </button>
                          </div>
                          <div className="brollIntensity">
                            <span className="brollSyncLabel">
                              B-roll opacity:
                            </span>
                            <select
                              value={String(brollDefaultOpacity)}
                              onChange={(event) =>
                                setBrollDefaultOpacity(
                                  Number(event.target.value),
                                )
                              }
                              disabled={autoApplyingBroll || syncingBroll}
                            >
                              <option value="1">100% (Full frame)</option>
                              <option value="0.85">85%</option>
                              <option value="0.7">70%</option>
                              <option value="0.5">50%</option>
                            </select>
                          </div>
                          <div className="brollIntensity">
                            <span className="brollSyncLabel">
                              B-roll intensity:
                            </span>
                            <select
                              value={brollIntensity}
                              onChange={(event) =>
                                setBrollIntensity(
                                  event.target.value as BrollIntensity,
                                )
                              }
                              disabled={
                                autoApplyingBroll ||
                                suggestingBroll ||
                                suggestingBrollSelection
                              }
                            >
                              <option value="low">Light</option>
                              <option value="medium">Balanced</option>
                              <option value="high">Rich</option>
                            </select>
                          </div>
                        </div>
                        <BrollTrustSummary
                          slots={brollSlots}
                          overlayClipCount={overlayClips.length}
                          selectedPlan={selectedBrollPlan}
                          formatSeconds={formatSeconds}
                          reasonCodeLabel={reasonCodeLabel}
                          onFocusSlot={focusBrollSlot}
                        />
                        <div className="brollSlots">
                          {!brollSlots.length && (
                            <p className="muted">
                              {loadingBrollSlots ||
                              suggestingBroll ||
                              autoApplyingBroll
                                ? "Looking for B-roll moments in your transcript..."
                                : !transcript
                                  ? "No B-roll slots yet. Generate a transcript, then click Suggest or Auto B-roll."
                                  : "No B-roll slots yet. Click Suggest B-roll or Auto B-roll to generate them."}
                            </p>
                          )}
                          {[...brollSlots]
                            .sort((left, right) => {
                              const leftReview =
                                (left.review_status ?? "needs_review") ===
                                "needs_review"
                                  ? 0
                                  : 1;
                              const rightReview =
                                (right.review_status ?? "needs_review") ===
                                "needs_review"
                                  ? 0
                                  : 1;
                              if (leftReview !== rightReview)
                                return leftReview - rightReview;
                              return left.start_sec - right.start_sec;
                            })
                            .map((slot, slotIndex) => {
                              const chosenCandidate =
                                slot.candidates.find(
                                  (candidate) =>
                                    candidate.id === slot.chosen_candidate_id,
                                ) ?? null;
                              const primaryCandidate =
                                chosenCandidate ?? slot.candidates[0] ?? null;
                              const primaryReason = primaryCandidate?.reason;
                              const heardText = getSlotTranscriptText(
                                slot,
                                transcriptWordsById,
                                transcript?.words ?? [],
                                showRomanizedTranscript,
                              );
                              const englishGloss =
                                slot.meaning?.english_gloss ??
                                readReasonText(primaryReason, "english_gloss");
                              const searchConcept =
                                slot.meaning?.search_concept ??
                                readReasonText(primaryReason, "search_concept") ??
                                slot.concept_text;
                              const searchQueries =
                                slot.meaning?.search_queries?.length
                                  ? slot.meaning.search_queries
                                  : readReasonStringList(
                                      primaryReason,
                                      "search_queries",
                                    );
                              const languageMix =
                                slot.meaning?.source_languages ?? [];
                              const meaningDraft =
                                brollMeaningDrafts[slot.id] ?? "";
                              const hasMeaningDraft = !!meaningDraft.trim();
                              const weakNeedsMeaningHint = (
                                slot.weak_reason_codes ?? []
                              ).some(
                                (code) =>
                                  code === "specificity_low" ||
                                  code === "confidence_low" ||
                                  code === "semantic_weak",
                              );
                              const slotMeta = [
                                readReasonText(primaryReason, "section_label"),
                                readReasonText(primaryReason, "shot_style")
                                  ? `${humanizeBrollMeta(readReasonText(primaryReason, "shot_style") ?? "")} shot`
                                  : null,
                                readReasonText(primaryReason, "source_strategy")
                                  ? humanizeBrollMeta(
                                      readReasonText(
                                        primaryReason,
                                        "source_strategy",
                                      ) ?? "",
                                    )
                                  : null,
                                readReasonNumber(
                                  primaryReason,
                                  "planner_confidence",
                                ) !== null
                                  ? `plan ${(readReasonNumber(primaryReason, "planner_confidence")! * 100).toFixed(0)}%`
                                  : null,
                              ]
                                .filter((item): item is string => !!item)
                                .join(" · ");
                              const defaultVisibleCount =
                                slotIndex === 0 ? 5 : 3;
                              const expanded = !!expandedBrollSlots[slot.id];
                              const visibleCandidates = expanded
                                ? slot.candidates
                                : slot.candidates.slice(0, defaultVisibleCount);
                              return (
                                <article
                                  key={slot.id}
                                  className={`brollSlotCard ${slot.status} ${slot.locked ? "locked" : ""} ${
                                    slot.chosen_candidate_id ? "hasChosen" : ""
                                  } ${selectedBrollSlotId === slot.id ? "focused" : ""}`}
                                >
                                  <div className="brollSlotHead">
                                    <button
                                      type="button"
                                      className="brollTime brollFocusTime"
                                      onClick={() => focusBrollSlot(slot)}
                                      title="Highlight the related transcript words"
                                    >
                                      {formatSeconds(slot.start_sec)}-
                                      {formatSeconds(slot.end_sec)}
                                    </button>
                                    <span
                                      className={`brollStatus ${slot.review_status ?? "needs_review"}`}
                                    >
                                      {reviewStatusLabel(
                                        slot.review_status ?? "needs_review",
                                      )}
                                    </span>
                                  </div>
                                  <p className="brollHeardText">
                                    <span className="brollContextLabel">
                                      Heard
                                    </span>
                                    {heardText
                                      ? `"${heardText}"`
                                      : "No transcript words in this range"}
                                  </p>
                                  {englishGloss && (
                                    <p className="brollMeaningText">
                                      <span className="brollContextLabel">
                                        Means (English)
                                      </span>
                                      {englishGloss}
                                    </p>
                                  )}
                                  {slot.meaning?.meaning_review_required && (
                                    <p className="brollWeakHint">
                                      Romanized lyric needs review — {slot.meaning.meaning_warning ?? "confirm the literal meaning before using this B-roll."}
                                    </p>
                                  )}
                                  {!!slot.meaning?.normalized_source_text && (
                                    <p className="brollLanguageMix muted">
                                      <span className="brollContextLabel">
                                        Normalized script
                                      </span>
                                      {slot.meaning.normalized_source_text}
                                    </p>
                                  )}
                                  <p className="brollSearchText">
                                    <span className="brollContextLabel">
                                      Search
                                    </span>
                                    {searchConcept || "general scene"}
                                  </p>
                                  {slot.meaning?.code_switched &&
                                    languageMix.length > 1 && (
                                      <p className="brollLanguageMix muted">
                                        <span className="brollContextLabel">
                                          Language mix
                                        </span>
                                        {languageMix.join(" + ")}
                                      </p>
                                    )}
                                  {!!(slot.meaning?.translation_provider ||
                                    slot.meaning?.planner_provider) && (
                                    <p className="brollLanguageMix muted">
                                      <span className="brollContextLabel">
                                        AI pipeline
                                      </span>
                                      {[
                                        slot.meaning?.translation_provider
                                          ? `Meaning: ${slot.meaning.translation_provider}`
                                          : "",
                                        slot.meaning?.planner_provider
                                          ? `B-roll: ${slot.meaning.planner_provider}`
                                          : "",
                                      ]
                                        .filter(Boolean)
                                        .join(" · ")}
                                    </p>
                                  )}
                                  {!!searchQueries.length && (
                                    <p className="brollQueryText muted">
                                      <span className="brollContextLabel">
                                        Queries
                                      </span>
                                      {searchQueries.slice(0, 4).join(" · ")}
                                    </p>
                                  )}
                                  {weakNeedsMeaningHint && (
                                    <p className="brollWeakHint muted">
                                      Generic stock match — edit meaning below
                                      and re-roll.
                                    </p>
                                  )}
                                  {!!slotMeta && (
                                    <p className="muted brollPlanMeta">
                                      {slotMeta}
                                    </p>
                                  )}
                                  <p className="muted brollReviewMeta">
                                    Intent: {slot.visual_intent ?? "support"}
                                    {slot.review_summary
                                      ? ` · ${slot.review_summary}`
                                      : ""}
                                  </p>
                                  {!!(slot.weak_reason_codes?.length ?? 0) && (
                                    <p className="brollWeakReasons">
                                      {(slot.weak_reason_codes ?? [])
                                        .map((code) => reasonCodeLabel(code))
                                        .join(" · ")}
                                    </p>
                                  )}
                                  <label className="brollMeaningField">
                                    <span className="brollContextLabel">
                                      Correct meaning (English)
                                    </span>
                                    <textarea
                                      className="brollMeaningInput"
                                      rows={2}
                                      value={meaningDraft}
                                      placeholder="If wrong, describe what this line means in English..."
                                      disabled={!!brollActionKey || slot.locked}
                                      onChange={(event) =>
                                        setBrollMeaningDrafts((prev) => ({
                                          ...prev,
                                          [slot.id]: event.target.value,
                                        }))
                                      }
                                    />
                                    <span className="brollMeaningPersistHint">
                                      Saved with this B-roll slot when you
                                      re-roll.
                                    </span>
                                  </label>
                                  {chosenCandidate && (
                                    <p className="brollChosen">
                                      Chosen:{" "}
                                      {chosenCandidate.source_label ??
                                        chosenCandidate.asset_id ??
                                        "candidate"}
                                    </p>
                                  )}
                                  <div className="brollCandidates">
                                    {visibleCandidates.map((candidate) => {
                                      const busyChoose =
                                        brollActionKey ===
                                        `choose:${slot.id}:${candidate.id}`;
                                      const isChosen =
                                        slot.chosen_candidate_id ===
                                        candidate.id;
                                      const confidence =
                                        typeof candidate.confidence === "number"
                                          ? candidate.confidence
                                          : null;
                                      const confidencePercent =
                                        confidence !== null
                                          ? `${(confidence * 100).toFixed(0)}%`
                                          : null;
                                      const confidenceTier =
                                        confidenceLabel(confidence);
                                      const candidateReason =
                                        candidate.reason ?? {};
                                      const breakdownChips = [
                                        ...candidateBreakdownChips(
                                          candidate.score_breakdown ?? {},
                                        ),
                                        ...(
                                          candidate.weak_reason_codes ?? []
                                        ).map((code) => reasonCodeLabel(code)),
                                      ].slice(0, 4);
                                      const scoreLabel = `match ${(candidate.score * 100).toFixed(0)}%`;
                                      const shotStyle =
                                        readReasonText(
                                          candidateReason,
                                          "shot_style",
                                        ) ??
                                        readReasonText(
                                          candidateReason,
                                          "shot_type",
                                        );
                                      const queryMode = readReasonText(
                                        candidateReason,
                                        "query_mode",
                                      );
                                      const stockability = readReasonText(
                                        candidateReason,
                                        "stockability",
                                      );
                                      const metaLine = [
                                        candidateSourceTag(
                                          candidate.source_type,
                                        ),
                                        candidate.visual_intent
                                          ? humanizeBrollMeta(
                                              candidate.visual_intent,
                                            )
                                          : null,
                                        shotStyle
                                          ? `${humanizeBrollMeta(shotStyle)} shot`
                                          : null,
                                        queryMode
                                          ? humanizeBrollMeta(queryMode)
                                          : null,
                                        stockability,
                                      ]
                                        .filter(
                                          (item): item is string => !!item,
                                        )
                                        .join(" · ");
                                      const previewParams =
                                        resolveBrollCandidatePreviewParams(
                                          candidate,
                                          mediaById,
                                        );
                                      return (
                                        <BrollCandidateCard
                                          key={candidate.id}
                                          label={
                                            candidate.source_label ??
                                            candidate.asset_id ??
                                            "asset"
                                          }
                                          sourceTag={candidateSourceTag(
                                            candidate.source_type,
                                          )}
                                          metaLine={metaLine || null}
                                          confidencePercent={confidencePercent}
                                          confidenceTier={confidenceTier}
                                          scoreLabel={scoreLabel}
                                          breakdownChips={breakdownChips}
                                          previewUrl={
                                            previewParams?.url ?? null
                                          }
                                          previewType={
                                            previewParams?.type ?? "video"
                                          }
                                          chosen={isChosen}
                                          busy={busyChoose}
                                          locked={slot.locked}
                                          onClick={() =>
                                            void chooseBroll(
                                              slot.id,
                                              candidate.id,
                                            )
                                          }
                                        />
                                      );
                                    })}
                                  </div>
                                  {slot.candidates.length >
                                    defaultVisibleCount && (
                                    <button
                                      type="button"
                                      className="brollToggleCandidates"
                                      onClick={() =>
                                        setExpandedBrollSlots((prev) => ({
                                          ...prev,
                                          [slot.id]: !expanded,
                                        }))
                                      }
                                      disabled={!!brollActionKey}
                                    >
                                      {expanded
                                        ? "Show less"
                                        : `Show all (${slot.candidates.length})`}
                                    </button>
                                  )}
                                  <div className="brollSlotActions">
                                    <button
                                      type="button"
                                      onClick={() => focusBrollSlot(slot)}
                                      disabled={!transcript}
                                    >
                                      Focus words
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => void rerollBroll(slot.id)}
                                      disabled={!!brollActionKey || slot.locked}
                                    >
                                      {brollActionKey === `reroll:${slot.id}`
                                        ? "Rerolling..."
                                        : hasMeaningDraft
                                          ? "Re-roll with this meaning"
                                          : "Re-roll"}
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => void rejectBroll(slot.id)}
                                      disabled={!!brollActionKey || slot.locked}
                                    >
                                      {brollActionKey === `reject:${slot.id}`
                                        ? "Rejecting..."
                                        : "Reject"}
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
                            const clipOpacity =
                              typeof clip.broll_opacity === "number"
                                ? clip.broll_opacity
                                : 1;
                            const draftStart =
                              brollDraftStartById[clip.id] ??
                              formatFixedSec(clip.timeline_start_sec);
                            const draftDuration =
                              brollDraftDurationById[clip.id] ??
                              formatFixedSec(clipDuration);
                            const draftOpacity =
                              brollDraftOpacityById[clip.id] ?? clipOpacity;
                            const source = mediaById.get(clip.asset_id);
                            return (
                              <article
                                key={clip.id}
                                className="brollTimelineCard"
                              >
                                <div className="brollTimelineHead">
                                  <span>B{index + 1}</span>
                                  <span>
                                    {source?.filename ?? clip.asset_id}
                                  </span>
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
                                        setBrollDraftStartById((prev) => ({
                                          ...prev,
                                          [clip.id]: event.target.value,
                                        }))
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
                                        setBrollDraftDurationById((prev) => ({
                                          ...prev,
                                          [clip.id]: event.target.value,
                                        }))
                                      }
                                      onBlur={() =>
                                        void commitBrollDuration(clip)
                                      }
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
                                      void commitBrollOpacity(
                                        clip,
                                        Number(event.currentTarget.value),
                                      )
                                    }
                                    onTouchEnd={(event) =>
                                      void commitBrollOpacity(
                                        clip,
                                        Number(event.currentTarget.value),
                                      )
                                    }
                                    onBlur={(event) =>
                                      void commitBrollOpacity(
                                        clip,
                                        Number(event.currentTarget.value),
                                      )
                                    }
                                  />
                                </label>
                                <div className="brollTimelineActions">
                                  <button
                                    type="button"
                                    disabled={clipBusy}
                                    onClick={() => {
                                      if (videoRef.current) {
                                        videoRef.current.currentTime =
                                          clip.timeline_start_sec;
                                      }
                                      setCurrentTimeSec(
                                        clip.timeline_start_sec,
                                      );
                                    }}
                                  >
                                    Jump
                                  </button>
                                  <button
                                    type="button"
                                    disabled={clipBusy}
                                    onClick={() => {
                                      const slot = findSlotForOverlayClip(clip);
                                      if (slot) {
                                        focusBrollSlot(
                                          slot,
                                          "Timeline B-roll clip selected with its transcript region.",
                                        );
                                      } else {
                                        const ids =
                                          selectTranscriptWordIdsInRange(
                                            timelineAssistWords,
                                            clip.timeline_start_sec,
                                            clip.timeline_start_sec +
                                              clipTimelineDurationSec(clip),
                                          );
                                        focusTranscriptWordIds(
                                          ids,
                                          undefined,
                                          "Highlighted transcript words under this B-roll clip.",
                                        );
                                      }
                                    }}
                                  >
                                    Words
                                  </button>
                                  <button
                                    type="button"
                                    disabled={clipBusy || !!brollActionKey}
                                    onClick={() =>
                                      void rerollBrollFromTimelineClip(clip.id)
                                    }
                                  >
                                    {brollActionKey === `reroll-clip:${clip.id}`
                                      ? "Rerolling..."
                                      : "Re-roll"}
                                  </button>
                                  <button
                                    type="button"
                                    disabled={clipBusy}
                                    onClick={() =>
                                      void removeBrollClipFromTimeline(clip)
                                    }
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
          <div className="mobilePane mobilePaneTimeline">
          <Timeline
            words={timelineAssistWords}
            timelineLanes={timelineLanes}
            assetUrlById={assetUrlById}
            assetNameById={assetNameById}
            assetDurationById={assetDurationById}
            captionBlocks={captionBlocks}
            durationSec={
              transcript?.duration_sec || project.timeline.duration_sec
            }
            currentTimeSec={currentTimeSec}
            isPlaying={isVideoPlaying}
            fps={canonicalTimelineFps}
            timelineClock={
              TIMELINE_CORE_V2 ? timelineClockRef.current : undefined
            }
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
            onMoveBrollClipToLane={handleTimelineMoveBrollClipToLane}
            onTrimBrollClip={handleTimelineTrimBrollClip}
            onSetBrollOpacity={handleTimelineSetBrollOpacity}
            onDeleteBrollClip={handleTimelineDeleteBrollClip}
            onRerollBrollClip={handleTimelineRerollBrollClip}
            onSelectLaneClip={handleTimelineSelectLaneClip}
            onSelectBrollClip={handleTimelineSelectBrollClip}
            onSelectCaptionBlock={handleTimelineSelectCaptionBlock}
            onMoveCaptionBlock={handleTimelineMoveCaptionBlock}
            onTrimCaptionBlock={handleTimelineTrimCaptionBlock}
            onEditWord={handleTimelineEditWord}
            onStartEditCaption={startCaptionEditing}
            editingCaptionId={editingCaptionOverlayId}
            editingCaptionText={editingCaptionText}
            onCaptionTextChange={setEditingCaptionText}
            onCommitCaptionEdit={() => void commitCaptionEdit()}
            onCancelCaptionEdit={cancelCaptionEdit}
            captionEditInputRef={captionEditInputRef}
            onDeleteLaneClip={handleTimelineDeleteLaneClip}
            onSplitLaneClip={handleTimelineSplitLaneClip}
            onCopyLaneClip={copySelectedTimelineClip}
            onDuplicateLaneClip={duplicateSelectedTimelineClip}
            onPasteLaneClip={pasteTimelineClip}
            canPasteLaneClip={!!clipClipboard}
            brollEditBusy={!!brollTimelineActionKey}
          />
          </div>
          </div>
        </>
      )}

      {showShortcutsHelp && (
        <KeyboardShortcutsModal onClose={() => setShowShortcutsHelp(false)} />
      )}
    </div>
  );
}

export default App;
