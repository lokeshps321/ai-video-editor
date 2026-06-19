import { memo, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type MouseEvent as ReactMouseEvent, type RefObject } from "react";
import { Captions, Lock, LockOpen, MapPin, Minus, Pencil, Plus, RefreshCw, RotateCcw, Scissors, Trash2, Volume2, VolumeX } from "lucide-react";
import type { Clip, TranscriptWord } from "../types";

export type TimelineLane = {
  id: string;
  label: string;
  kind: "video" | "audio";
  clips: Clip[];
  mute?: boolean;
  solo?: boolean;
  volume?: number;
};

export type TimelineLaneClipSelection = {
  clipId: string;
  laneId: string;
  laneLabel: string;
  laneKind: "video" | "audio";
};

export type TimelineCaptionSelection = {
  overlayId: string;
  clipId: string;
  laneId: string;
  laneLabel: string;
  text: string;
  style: string;
};

export type TimelineCaptionBlock = {
  id: string;
  clipId: string;
  laneId: string;
  laneLabel: string;
  text: string;
  style: string;
  startSec: number;
  durationSec: number;
  clipTimelineStartSec: number;
  clipSourceDurationSec: number;
  clipSpeed: number;
};

export type TimelineProps = {
  words: TranscriptWord[];
  timelineLanes: TimelineLane[];
  assetUrlById: Map<string, string>;
  assetDurationById: Map<string, number | null>;
  overlayClips: Clip[];
  captionBlocks: TimelineCaptionBlock[];
  durationSec: number;
  currentTimeSec: number;
  deletedWordIds: Set<string>;
  selectedWordIds: Set<string>;
  activeWordId: string | null;
  waveformPeaks: number[];
  selectedLaneClipId: string | null;
  selectedBrollClipId: string | null;
  selectedCaptionId: string | null;
  lockedLaneIds: Set<string>;
  onSeek: (sec: number) => void;
  onSelectWord: (id: string, shift: boolean) => void;
  onSelectWordsInRange: (startSec: number, endSec: number) => void;
  onDeleteSelected: () => void;
  onRestoreSelected: () => void;
  onMoveLaneClip: (selection: TimelineLaneClipSelection, timelineStartSec: number) => void;
  onTrimLaneClip: (selection: TimelineLaneClipSelection, nextRange: { startSec: number; endSec: number }) => void;
  onToggleLaneMute?: (lane: TimelineLane) => void;
  onToggleLaneSolo?: (lane: TimelineLane) => void;
  onToggleLaneLock?: (laneId: string) => void;
  onMoveBrollClip: (clipId: string, timelineStartSec: number) => void;
  onTrimBrollClip: (clipId: string, durationSec: number) => void;
  onSetBrollOpacity: (clipId: string, opacity: number) => void;
  onDeleteBrollClip: (clipId: string) => void;
  onRerollBrollClip?: (clipId: string) => void;
  onSelectLaneClip?: (selection: TimelineLaneClipSelection | null) => void;
  onSelectBrollClip?: (clipId: string | null) => void;
  onSelectCaptionBlock?: (selection: TimelineCaptionSelection | null) => void;
  onMoveCaptionBlock?: (selection: TimelineCaptionSelection, startSec: number) => void;
  onTrimCaptionBlock?: (selection: TimelineCaptionSelection, startSec: number, durationSec: number) => void;
  onEditWord?: (wordId: string) => void;
  editingCaptionId?: string | null;
  editingCaptionText?: string;
  onStartEditCaption?: (selection: TimelineCaptionSelection) => void;
  onCaptionTextChange?: (text: string) => void;
  onCommitCaptionEdit?: () => void;
  onCancelCaptionEdit?: () => void;
  captionEditInputRef?: RefObject<HTMLInputElement | null>;
  onDeleteLaneClip?: (selection: TimelineLaneClipSelection) => void;
  onSplitLaneClip?: (selection: TimelineLaneClipSelection) => void;
  brollEditBusy: boolean;
};

const MIN_PX_PER_SEC = 15;
const MAX_PX_PER_SEC = 250;
const DEFAULT_PX_PER_SEC = 40;
const TRACK_LEFT_MARGIN = 96;
const MIN_BROLL_DURATION_SEC = 0.1;
const MIN_CLIP_SOURCE_DURATION_SEC = 0.05;

type LaneClipThumbProps = {
  src: string;
  seekSec: number;
};

type SnapGuide = {
  ownerKey: string;
  timeSec: number;
};

type BrollDragState = {
  clipId: string;
  mode: "move" | "resize-end";
  startClientX: number;
  initialStartSec: number;
  initialDurationSec: number;
  currentStartSec: number;
  currentDurationSec: number;
};

type LaneDragState = {
  selection: TimelineLaneClipSelection;
  mode: "move" | "trim-start" | "trim-end";
  startClientX: number;
  speed: number;
  sourceMaxEndSec: number;
  initialTimelineStartSec: number;
  initialTimelineEndSec: number;
  currentTimelineStartSec: number;
  initialStartSec: number;
  initialEndSec: number;
  currentStartSec: number;
  currentEndSec: number;
};

type CaptionDragState = {
  selection: TimelineCaptionSelection;
  mode: "move" | "trim-start" | "trim-end";
  startClientX: number;
  clipTimelineStartSec: number;
  clipSourceDurationSec: number;
  clipSpeed: number;
  initialStartSec: number;
  currentStartSec: number;
  initialDurationSec: number;
  currentDurationSec: number;
};

function LaneClipThumb({ src, seekSec }: LaneClipThumbProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const seekToFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    const duration = Number.isFinite(video.duration) ? Math.max(video.duration - 0.05, 0) : seekSec;
    const target = Math.max(0, Math.min(seekSec, duration));
    try {
      video.currentTime = target;
    } catch {
      // Ignore transient metadata / seek races.
    }
  }, [seekSec]);

  return (
    <video
      ref={videoRef}
      className="timelineLaneThumb"
      src={src}
      muted
      playsInline
      preload="metadata"
      onLoadedMetadata={seekToFrame}
      onCanPlay={seekToFrame}
    />
  );
}

function formatTimecode(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatDuration(sec: number): string {
  if (sec < 1) return `${Math.round(sec * 1000)}ms`;
  return `${sec.toFixed(1)}s`;
}

function clipTimelineDuration(clip: Clip): number {
  return Math.max((clip.end_sec - clip.start_sec) / Math.max(clip.speed, 0.01), MIN_BROLL_DURATION_SEC);
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function trimInlineText(text: string, maxLength = 36): string {
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 1).trimEnd()}…`;
}

function buildSpeakerIdOrder(words: TranscriptWord[]): string[] {
  const seen: string[] = [];
  for (const word of words) {
    if (word.speaker_id && !seen.includes(word.speaker_id)) {
      seen.push(word.speaker_id);
    }
  }
  return seen;
}

function speakerSlotForWord(word: TranscriptWord, speakerIdOrder: string[]): number | null {
  if (!word.speaker_id) return null;
  const index = speakerIdOrder.indexOf(word.speaker_id);
  return index >= 0 ? index : null;
}

function isInteractiveTarget(target: EventTarget | null): boolean {
  const node = target as HTMLElement | null;
  if (!node) return false;
  return !!node.closest("button,input,select,textarea");
}

function Timeline({
  words,
  timelineLanes,
  assetUrlById,
  assetDurationById,
  overlayClips,
  captionBlocks,
  durationSec,
  currentTimeSec,
  deletedWordIds,
  selectedWordIds,
  activeWordId,
  waveformPeaks,
  selectedLaneClipId,
  selectedBrollClipId,
  selectedCaptionId,
  lockedLaneIds,
  onSeek,
  onSelectWord,
  onSelectWordsInRange,
  onDeleteSelected,
  onRestoreSelected,
  onMoveLaneClip,
  onTrimLaneClip,
  onToggleLaneMute,
  onToggleLaneSolo,
  onToggleLaneLock,
  onMoveBrollClip,
  onTrimBrollClip,
  onSetBrollOpacity,
  onDeleteBrollClip,
  onRerollBrollClip,
  onSelectLaneClip,
  onSelectBrollClip,
  onSelectCaptionBlock,
  onMoveCaptionBlock,
  onTrimCaptionBlock,
  onEditWord,
  editingCaptionId = null,
  editingCaptionText = "",
  onStartEditCaption,
  onCaptionTextChange,
  onCommitCaptionEdit,
  onCancelCaptionEdit,
  captionEditInputRef,
  onDeleteLaneClip,
  onSplitLaneClip,
  brollEditBusy,
}: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [pxPerSec, setPxPerSec] = useState(DEFAULT_PX_PER_SEC);
  const [showTranscriptAssist, setShowTranscriptAssist] = useState(words.length > 0);

  type DragMode = "none" | "seek" | "range";
  const [dragMode, setDragMode] = useState<DragMode>("none");
  const [rangeStart, setRangeStart] = useState<number | null>(null);
  const [rangeEnd, setRangeEnd] = useState<number | null>(null);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null);
  const [brollOpacityDraftById, setBrollOpacityDraftById] = useState<Record<string, number>>({});
  const opacityCommitTimersRef = useRef<Record<string, number>>({});
  const [brollDragState, setBrollDragState] = useState<BrollDragState | null>(null);
  const [laneDragState, setLaneDragState] = useState<LaneDragState | null>(null);
  const [captionDragState, setCaptionDragState] = useState<CaptionDragState | null>(null);

  const totalWidth = Math.max(durationSec * pxPerSec, 200);

  useEffect(() => {
    if (!words.length) {
      setShowTranscriptAssist(false);
      return;
    }
    setShowTranscriptAssist((prev) => prev || words.length > 0);
  }, [words.length]);

  const ticks = useMemo(() => {
    let interval = 1;
    if (pxPerSec < 20) interval = 10;
    else if (pxPerSec < 40) interval = 5;
    else if (pxPerSec < 80) interval = 2;
    else if (pxPerSec < 150) interval = 1;
    else interval = 0.5;

    const result: { sec: number; x: number; label: string; major: boolean }[] = [];
    for (let sec = 0; sec <= durationSec; sec += interval) {
      result.push({
        sec,
        x: sec * pxPerSec,
        label: formatTimecode(sec),
        major: sec % (interval >= 1 ? 5 : 1) === 0,
      });
    }
    return result;
  }, [durationSec, pxPerSec]);

  const waveformBars = useMemo(() => {
    if (!waveformPeaks.length) return [];
    const barWidth = totalWidth / waveformPeaks.length;
    return waveformPeaks.map((peak, i) => ({
      x: i * barWidth,
      width: Math.max(barWidth - 0.5, 0.5),
      height: peak,
    }));
  }, [waveformPeaks, totalWidth]);

  const deletedRegions = useMemo(() => {
    if (!words.length) return [];
    const sorted = [...words].sort((a, b) => a.start_sec - b.start_sec);
    const regions: { startSec: number; endSec: number }[] = [];
    let regionStart: number | null = null;

    for (const word of sorted) {
      if (deletedWordIds.has(word.id)) {
        if (regionStart === null) regionStart = word.start_sec;
      } else if (regionStart !== null) {
        const previousDeleted = sorted.find((candidate) => candidate.end_sec <= word.start_sec && deletedWordIds.has(candidate.id));
        regions.push({
          startSec: regionStart,
          endSec: previousDeleted ? previousDeleted.end_sec : word.start_sec,
        });
        regionStart = null;
      }
    }

    if (regionStart !== null) {
      const lastWord = sorted[sorted.length - 1];
      regions.push({ startSec: regionStart, endSec: lastWord.end_sec });
    }
    return regions;
  }, [words, deletedWordIds]);

  const wordBlocks = useMemo(() => {
    const speakerIdOrder = buildSpeakerIdOrder(words);
    return words.map((word) => {
      const x = word.start_sec * pxPerSec;
      const w = Math.max((word.end_sec - word.start_sec) * pxPerSec, 3);
      const speakerSlot = speakerSlotForWord(word, speakerIdOrder);
      return {
        word,
        x,
        w,
        speakerSlot,
        isDeleted: deletedWordIds.has(word.id),
        isSelected: selectedWordIds.has(word.id),
        isActive: activeWordId === word.id,
      };
    });
  }, [words, pxPerSec, deletedWordIds, selectedWordIds, activeWordId]);

  const speakerLegend = useMemo(() => {
    const speakerIdOrder = buildSpeakerIdOrder(words);
    return speakerIdOrder.map((speakerId, slot) => {
      const label = words.find((word) => word.speaker_id === speakerId)?.speaker_label || `Speaker ${slot + 1}`;
      return { speakerId, label, slot };
    });
  }, [words]);

  const snapGuides = useMemo(() => {
    const guides: SnapGuide[] = [
      { ownerKey: "system:start", timeSec: 0 },
      { ownerKey: "system:playhead", timeSec: currentTimeSec },
      { ownerKey: "system:end", timeSec: durationSec },
    ];

    timelineLanes.forEach((lane) => {
      lane.clips.forEach((clip) => {
        const startSec = clip.timeline_start_sec;
        const endSec = clip.timeline_start_sec + clipTimelineDuration(clip);
        guides.push({ ownerKey: `lane:${clip.id}:start`, timeSec: startSec });
        guides.push({ ownerKey: `lane:${clip.id}:end`, timeSec: endSec });
      });
    });
    overlayClips.forEach((clip) => {
      const startSec = clip.timeline_start_sec;
      const endSec = clip.timeline_start_sec + clipTimelineDuration(clip);
      guides.push({ ownerKey: `broll:${clip.id}:start`, timeSec: startSec });
      guides.push({ ownerKey: `broll:${clip.id}:end`, timeSec: endSec });
    });
    captionBlocks.forEach((block) => {
      guides.push({ ownerKey: `caption:${block.id}:start`, timeSec: block.startSec });
      guides.push({ ownerKey: `caption:${block.id}:end`, timeSec: block.startSec + block.durationSec });
    });
    return guides;
  }, [timelineLanes, overlayClips, captionBlocks, currentTimeSec, durationSec]);

  const snapThresholdSec = 10 / pxPerSec;

  const resolveEdgeSnap = useCallback(
    (rawSec: number, ownerPrefix: string) => {
      let best = rawSec;
      let bestDistance = snapThresholdSec + 1;
      for (const guide of snapGuides) {
        if (guide.ownerKey.startsWith(ownerPrefix)) continue;
        const distance = Math.abs(guide.timeSec - rawSec);
        if (distance <= snapThresholdSec && distance < bestDistance) {
          bestDistance = distance;
          best = guide.timeSec;
        }
      }
      return best;
    },
    [snapGuides, snapThresholdSec]
  );

  const resolveBlockSnap = useCallback(
    (rawStartSec: number, blockDurationSec: number, ownerPrefix: string) => {
      let best = rawStartSec;
      let bestDistance = snapThresholdSec + 1;
      for (const guide of snapGuides) {
        if (guide.ownerKey.startsWith(ownerPrefix)) continue;
        const startDistance = Math.abs(guide.timeSec - rawStartSec);
        if (startDistance <= snapThresholdSec && startDistance < bestDistance) {
          bestDistance = startDistance;
          best = guide.timeSec;
        }
        const rawEndSec = rawStartSec + blockDurationSec;
        const endDistance = Math.abs(guide.timeSec - rawEndSec);
        if (endDistance <= snapThresholdSec && endDistance < bestDistance) {
          bestDistance = endDistance;
          best = guide.timeSec - blockDurationSec;
        }
      }
      return best;
    },
    [snapGuides, snapThresholdSec]
  );

  const brollBlocks = useMemo(() => {
    return overlayClips
      .slice()
      .sort((a, b) => a.timeline_start_sec - b.timeline_start_sec)
      .map((clip) => {
        const dragPreview = brollDragState?.clipId === clip.id ? brollDragState : null;
        const timelineStartSec = dragPreview ? dragPreview.currentStartSec : clip.timeline_start_sec;
        const duration = dragPreview ? dragPreview.currentDurationSec : clipTimelineDuration(clip);
        const x = timelineStartSec * pxPerSec;
        const w = Math.max(duration * pxPerSec, 4);
        const clipOpacity = typeof clip.broll_opacity === "number" ? clip.broll_opacity : 1;
        const opacity = brollOpacityDraftById[clip.id] ?? clipOpacity;
        return {
          clip,
          x,
          w,
          opacity,
          timelineStartSec,
          duration,
          isDragging: !!dragPreview,
        };
      });
  }, [overlayClips, pxPerSec, brollDragState, brollOpacityDraftById]);

  const laneBlocks = useMemo(() => {
    return timelineLanes.map((lane) => {
      const isLocked = lockedLaneIds.has(lane.id);
      return {
        ...lane,
        isLocked,
        blocks: lane.clips
          .slice()
          .sort((a, b) => a.timeline_start_sec - b.timeline_start_sec)
          .map((clip) => {
            const dragPreview = laneDragState?.selection.clipId === clip.id ? laneDragState : null;
            const sourceStartSec = dragPreview ? dragPreview.currentStartSec : clip.start_sec;
            const sourceEndSec = dragPreview ? dragPreview.currentEndSec : clip.end_sec;
            const timelineStartSec = dragPreview ? dragPreview.currentTimelineStartSec : clip.timeline_start_sec;
            const duration = Math.max((sourceEndSec - sourceStartSec) / Math.max(clip.speed, 0.01), MIN_BROLL_DURATION_SEC);
            const clipWidthPx = Math.max(duration * pxPerSec, 4);
            const thumbCount = Math.max(1, Math.min(8, Math.floor(clipWidthPx / 54)));
            const sourceDuration = Math.max(sourceEndSec - sourceStartSec, MIN_CLIP_SOURCE_DURATION_SEC);
            const thumbTimes = Array.from({ length: thumbCount }, (_unused, idx) => {
              if (thumbCount <= 1) {
                return sourceStartSec + Math.min(0.35, sourceDuration * 0.25);
              }
              const t = idx / (thumbCount - 1);
              return sourceStartSec + (sourceDuration * t);
            });
            return {
              clip,
              x: timelineStartSec * pxPerSec,
              w: clipWidthPx,
              duration,
              sourceStartSec,
              sourceEndSec,
              timelineStartSec,
              thumbSrc: assetUrlById.get(clip.asset_id) ?? null,
              thumbTimes,
              isSelected: selectedLaneClipId === clip.id,
              isDragging: !!dragPreview,
            };
          }),
      };
    });
  }, [timelineLanes, lockedLaneIds, laneDragState, pxPerSec, assetUrlById, selectedLaneClipId]);

  const renderedCaptionBlocks = useMemo(() => {
    return captionBlocks
      .slice()
      .sort((a, b) => a.startSec - b.startSec)
      .map((block) => {
        const dragPreview = captionDragState?.selection.overlayId === block.id ? captionDragState : null;
        const startSec = dragPreview
          ? dragPreview.clipTimelineStartSec + (dragPreview.currentStartSec / Math.max(dragPreview.clipSpeed, 0.01))
          : block.startSec;
        const duration = dragPreview
          ? dragPreview.currentDurationSec / Math.max(dragPreview.clipSpeed, 0.01)
          : block.durationSec;
        return {
          ...block,
          x: startSec * pxPerSec,
          w: Math.max(duration * pxPerSec, 8),
          renderedStartSec: startSec,
          renderedDurationSec: duration,
          isSelected: selectedCaptionId === block.id,
          isDragging: !!dragPreview,
        };
      });
  }, [captionBlocks, captionDragState, pxPerSec, selectedCaptionId]);

  useEffect(() => {
    setBrollOpacityDraftById(() => {
      const next: Record<string, number> = {};
      overlayClips.forEach((clip) => {
        next[clip.id] = clamp(typeof clip.broll_opacity === "number" ? clip.broll_opacity : 1, 0, 1);
      });
      return next;
    });
  }, [overlayClips]);

  useEffect(() => {
    return () => {
      Object.values(opacityCommitTimersRef.current).forEach((timer) => window.clearTimeout(timer));
      opacityCommitTimersRef.current = {};
    };
  }, []);

  const playheadX = currentTimeSec * pxPerSec;

  const rangeLeft = rangeStart !== null && rangeEnd !== null ? Math.min(rangeStart, rangeEnd) * pxPerSec : null;
  const rangeWidth = rangeStart !== null && rangeEnd !== null ? Math.abs(rangeEnd - rangeStart) * pxPerSec : null;
  const rangeDuration = rangeStart !== null && rangeEnd !== null ? Math.abs(rangeEnd - rangeStart) : null;

  useEffect(() => {
    if (!containerRef.current || dragMode !== "none" || brollDragState || laneDragState || captionDragState) return;
    const element = containerRef.current;
    const viewLeft = element.scrollLeft;
    const viewRight = viewLeft + element.clientWidth;
    if (playheadX < viewLeft + 40 || playheadX > viewRight - 40) {
      element.scrollLeft = playheadX - element.clientWidth / 3;
    }
  }, [playheadX, dragMode, brollDragState, laneDragState, captionDragState]);

  const secFromClientX = useCallback(
    (clientX: number) => {
      if (!containerRef.current) return 0;
      const rect = containerRef.current.getBoundingClientRect();
      const x = clientX - rect.left + containerRef.current.scrollLeft - TRACK_LEFT_MARGIN;
      return Math.max(0, Math.min(x / pxPerSec, durationSec));
    },
    [pxPerSec, durationSec]
  );

  const secFromEvent = useCallback((event: ReactMouseEvent) => secFromClientX(event.clientX), [secFromClientX]);

  function scheduleOpacityCommit(clipId: string, opacity: number) {
    const previous = opacityCommitTimersRef.current[clipId];
    if (typeof previous === "number") {
      window.clearTimeout(previous);
    }
    opacityCommitTimersRef.current[clipId] = window.setTimeout(() => {
      onSetBrollOpacity(clipId, opacity);
      delete opacityCommitTimersRef.current[clipId];
    }, 180);
  }

  function clearAllSelections() {
    onSelectLaneClip?.(null);
    onSelectBrollClip?.(null);
    onSelectCaptionBlock?.(null);
  }

  function startBrollDrag(event: ReactMouseEvent, clip: Clip, mode: "move" | "resize-end") {
    if (event.button !== 0 || brollEditBusy) return;
    event.preventDefault();
    event.stopPropagation();
    clearAllSelections();
    onSelectBrollClip?.(clip.id);
    const duration = clipTimelineDuration(clip);
    setBrollDragState({
      clipId: clip.id,
      mode,
      startClientX: event.clientX,
      initialStartSec: clip.timeline_start_sec,
      initialDurationSec: duration,
      currentStartSec: clip.timeline_start_sec,
      currentDurationSec: duration,
    });
    onSeek(clip.timeline_start_sec);
  }

  function startLaneDrag(
    event: ReactMouseEvent,
    lane: TimelineLane,
    clip: Clip,
    mode: LaneDragState["mode"]
  ) {
    if (event.button !== 0 || isInteractiveTarget(event.target)) return;
    event.preventDefault();
    event.stopPropagation();
    const selection: TimelineLaneClipSelection = {
      clipId: clip.id,
      laneId: lane.id,
      laneLabel: lane.label,
      laneKind: lane.kind,
    };
    onSelectLaneClip?.(selection);
    onSelectBrollClip?.(null);
    onSelectCaptionBlock?.(null);
    onSeek(clip.timeline_start_sec);
    if (lockedLaneIds.has(lane.id)) return;

    const duration = clipTimelineDuration(clip);
    const assetDurationSec = assetDurationById.get(clip.asset_id) ?? null;
    const sourceMaxEndSec = assetDurationSec && assetDurationSec > 0 ? assetDurationSec : clip.end_sec + 30;
    setLaneDragState({
      selection,
      mode,
      startClientX: event.clientX,
      speed: Math.max(clip.speed, 0.01),
      sourceMaxEndSec,
      initialTimelineStartSec: clip.timeline_start_sec,
      initialTimelineEndSec: clip.timeline_start_sec + duration,
      currentTimelineStartSec: clip.timeline_start_sec,
      initialStartSec: clip.start_sec,
      initialEndSec: clip.end_sec,
      currentStartSec: clip.start_sec,
      currentEndSec: clip.end_sec,
    });
  }

  function startCaptionDrag(
    event: ReactMouseEvent,
    block: TimelineCaptionBlock,
    mode: CaptionDragState["mode"]
  ) {
    if (event.button !== 0 || isInteractiveTarget(event.target)) return;
    event.preventDefault();
    event.stopPropagation();
    const selection: TimelineCaptionSelection = {
      overlayId: block.id,
      clipId: block.clipId,
      laneId: block.laneId,
      laneLabel: block.laneLabel,
      text: block.text,
      style: block.style,
    };
    onSelectCaptionBlock?.(selection);
    onSelectLaneClip?.(null);
    onSelectBrollClip?.(null);
    onSeek(block.startSec);
    setCaptionDragState({
      selection,
      mode,
      startClientX: event.clientX,
      clipTimelineStartSec: block.clipTimelineStartSec,
      clipSourceDurationSec: block.clipSourceDurationSec,
      clipSpeed: Math.max(block.clipSpeed, 0.01),
      initialStartSec: (block.startSec - block.clipTimelineStartSec) * Math.max(block.clipSpeed, 0.01),
      currentStartSec: (block.startSec - block.clipTimelineStartSec) * Math.max(block.clipSpeed, 0.01),
      initialDurationSec: block.durationSec * Math.max(block.clipSpeed, 0.01),
      currentDurationSec: block.durationSec * Math.max(block.clipSpeed, 0.01),
    });
  }

  function handleMouseDown(event: ReactMouseEvent) {
    if (event.button !== 0 || isInteractiveTarget(event.target)) return;
    setContextMenu(null);
    clearAllSelections();

    if (event.altKey || event.shiftKey) {
      const sec = secFromEvent(event);
      setDragMode("range");
      setRangeStart(sec);
      setRangeEnd(sec);
      return;
    }

    setDragMode("seek");
    setRangeStart(null);
    setRangeEnd(null);
    onSeek(secFromEvent(event));
  }

  function handleMouseMove(event: ReactMouseEvent) {
    if (dragMode === "seek") {
      onSeek(secFromEvent(event));
    } else if (dragMode === "range") {
      setRangeEnd(secFromEvent(event));
    }
  }

  useEffect(() => {
    function onWindowUp() {
      if (dragMode === "range" && rangeStart !== null && rangeEnd !== null) {
        const lo = Math.min(rangeStart, rangeEnd);
        const hi = Math.max(rangeStart, rangeEnd);
        if (hi - lo > 0.05) {
          onSelectWordsInRange(lo, hi);
        }
      }
      setDragMode("none");
    }
    window.addEventListener("mouseup", onWindowUp);
    return () => window.removeEventListener("mouseup", onWindowUp);
  }, [dragMode, rangeStart, rangeEnd, onSelectWordsInRange]);

  useEffect(() => {
    if (!brollDragState) return;

    function onMove(event: MouseEvent) {
      setBrollDragState((prev) => {
        if (!prev) return prev;
        const deltaSec = (event.clientX - prev.startClientX) / pxPerSec;
        if (prev.mode === "move") {
          const rawStart = clamp(prev.initialStartSec + deltaSec, 0, Math.max(durationSec, prev.initialStartSec + 30));
          const snappedStart = clamp(
            resolveBlockSnap(rawStart, prev.initialDurationSec, `broll:${prev.clipId}:`),
            0,
            Math.max(durationSec, prev.initialStartSec + 30)
          );
          return { ...prev, currentStartSec: snappedStart };
        }
        const maxDuration = Math.max(durationSec - prev.initialStartSec, prev.initialDurationSec + 30, MIN_BROLL_DURATION_SEC);
        const rawDuration = clamp(prev.initialDurationSec + deltaSec, MIN_BROLL_DURATION_SEC, maxDuration);
        const rawEnd = prev.initialStartSec + rawDuration;
        const snappedEnd = clamp(resolveEdgeSnap(rawEnd, `broll:${prev.clipId}:`), prev.initialStartSec + MIN_BROLL_DURATION_SEC, prev.initialStartSec + maxDuration);
        return { ...prev, currentDurationSec: Math.max(snappedEnd - prev.initialStartSec, MIN_BROLL_DURATION_SEC) };
      });
    }

    function onUp() {
      setBrollDragState((prev) => {
        if (!prev) return prev;
        if (prev.mode === "move") {
          if (Math.abs(prev.currentStartSec - prev.initialStartSec) >= 0.01) {
            onMoveBrollClip(prev.clipId, Number(prev.currentStartSec.toFixed(3)));
          }
        } else if (Math.abs(prev.currentDurationSec - prev.initialDurationSec) >= 0.01) {
          onTrimBrollClip(prev.clipId, Number(prev.currentDurationSec.toFixed(3)));
        }
        return null;
      });
    }

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [brollDragState, durationSec, onMoveBrollClip, onTrimBrollClip, pxPerSec, resolveBlockSnap, resolveEdgeSnap]);

  useEffect(() => {
    if (!laneDragState) return;

    function onMove(event: MouseEvent) {
      setLaneDragState((prev) => {
        if (!prev) return prev;
        const deltaTimelineSec = (event.clientX - prev.startClientX) / pxPerSec;
        const ownerPrefix = `lane:${prev.selection.clipId}:`;

        if (prev.mode === "move") {
          const duration = (prev.initialEndSec - prev.initialStartSec) / prev.speed;
          const rawStart = clamp(prev.initialTimelineStartSec + deltaTimelineSec, 0, Math.max(durationSec, prev.initialTimelineStartSec + 30));
          const snappedStart = clamp(resolveBlockSnap(rawStart, duration, ownerPrefix), 0, Math.max(durationSec, prev.initialTimelineStartSec + 30));
          return { ...prev, currentTimelineStartSec: snappedStart };
        }

        if (prev.mode === "trim-start") {
          const minTimelineStart = prev.initialTimelineEndSec - (MIN_CLIP_SOURCE_DURATION_SEC / prev.speed);
          const rawTimelineStart = clamp(prev.initialTimelineStartSec + deltaTimelineSec, 0, minTimelineStart);
          const snappedTimelineStart = clamp(resolveEdgeSnap(rawTimelineStart, ownerPrefix), 0, minTimelineStart);
          const nextStartSec = clamp(
            prev.initialStartSec + ((snappedTimelineStart - prev.initialTimelineStartSec) * prev.speed),
            0,
            prev.initialEndSec - MIN_CLIP_SOURCE_DURATION_SEC
          );
          const nextTimelineStartSec = prev.initialTimelineStartSec + ((nextStartSec - prev.initialStartSec) / prev.speed);
          return {
            ...prev,
            currentStartSec: nextStartSec,
            currentTimelineStartSec: nextTimelineStartSec,
          };
        }

        const maxTimelineEnd = prev.initialTimelineStartSec + ((prev.sourceMaxEndSec - prev.initialStartSec) / prev.speed);
        const rawTimelineEnd = clamp(
          prev.initialTimelineEndSec + deltaTimelineSec,
          prev.initialTimelineStartSec + (MIN_CLIP_SOURCE_DURATION_SEC / prev.speed),
          maxTimelineEnd
        );
        const snappedTimelineEnd = clamp(
          resolveEdgeSnap(rawTimelineEnd, ownerPrefix),
          prev.initialTimelineStartSec + (MIN_CLIP_SOURCE_DURATION_SEC / prev.speed),
          maxTimelineEnd
        );
        const nextEndSec = clamp(
          prev.initialEndSec + ((snappedTimelineEnd - prev.initialTimelineEndSec) * prev.speed),
          prev.initialStartSec + MIN_CLIP_SOURCE_DURATION_SEC,
          prev.sourceMaxEndSec
        );
        return { ...prev, currentEndSec: nextEndSec };
      });
    }

    function onUp() {
      setLaneDragState((prev) => {
        if (!prev) return prev;
        if (prev.mode === "move") {
          if (Math.abs(prev.currentTimelineStartSec - prev.initialTimelineStartSec) >= 0.01) {
            onMoveLaneClip(prev.selection, Number(prev.currentTimelineStartSec.toFixed(3)));
          }
        } else if (
          Math.abs(prev.currentStartSec - prev.initialStartSec) >= 0.01 ||
          Math.abs(prev.currentEndSec - prev.initialEndSec) >= 0.01
        ) {
          onTrimLaneClip(prev.selection, {
            startSec: Number(prev.currentStartSec.toFixed(3)),
            endSec: Number(prev.currentEndSec.toFixed(3)),
          });
        }
        return null;
      });
    }

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [durationSec, laneDragState, onMoveLaneClip, onTrimLaneClip, pxPerSec, resolveBlockSnap, resolveEdgeSnap]);

  useEffect(() => {
    if (!captionDragState) return;

    function onMove(event: MouseEvent) {
      setCaptionDragState((prev) => {
        if (!prev) return prev;
        const deltaTimelineSec = (event.clientX - prev.startClientX) / pxPerSec;
        const ownerPrefix = `caption:${prev.selection.overlayId}:`;

        if (prev.mode === "move") {
          const rawStart = prev.clipTimelineStartSec + prev.initialStartSec / prev.clipSpeed + deltaTimelineSec;
          const maxStart = prev.clipTimelineStartSec + Math.max(0, (prev.clipSourceDurationSec - prev.initialDurationSec) / prev.clipSpeed);
          const snappedStart = clamp(
            resolveBlockSnap(rawStart, prev.initialDurationSec / prev.clipSpeed, ownerPrefix),
            prev.clipTimelineStartSec,
            maxStart
          );
          const nextStartSec = clamp(
            (snappedStart - prev.clipTimelineStartSec) * prev.clipSpeed,
            0,
            Math.max(0, prev.clipSourceDurationSec - prev.initialDurationSec)
          );
          return { ...prev, currentStartSec: nextStartSec };
        }

        if (prev.mode === "trim-start") {
          const rawStart = prev.clipTimelineStartSec + prev.initialStartSec / prev.clipSpeed + deltaTimelineSec;
          const maxStart = prev.clipTimelineStartSec + ((prev.initialStartSec + prev.initialDurationSec - MIN_CLIP_SOURCE_DURATION_SEC) / prev.clipSpeed);
          const snappedStart = clamp(resolveEdgeSnap(rawStart, ownerPrefix), prev.clipTimelineStartSec, maxStart);
          const nextStartSec = clamp(
            (snappedStart - prev.clipTimelineStartSec) * prev.clipSpeed,
            0,
            prev.initialStartSec + prev.initialDurationSec - MIN_CLIP_SOURCE_DURATION_SEC
          );
          const nextDurationSec = Math.max(prev.initialStartSec + prev.initialDurationSec - nextStartSec, MIN_CLIP_SOURCE_DURATION_SEC);
          return {
            ...prev,
            currentStartSec: nextStartSec,
            currentDurationSec: nextDurationSec,
          };
        }

        const rawEnd = prev.clipTimelineStartSec + ((prev.initialStartSec + prev.initialDurationSec) / prev.clipSpeed) + deltaTimelineSec;
        const minEnd = prev.clipTimelineStartSec + ((prev.initialStartSec + MIN_CLIP_SOURCE_DURATION_SEC) / prev.clipSpeed);
        const maxEnd = prev.clipTimelineStartSec + (prev.clipSourceDurationSec / prev.clipSpeed);
        const snappedEnd = clamp(resolveEdgeSnap(rawEnd, ownerPrefix), minEnd, maxEnd);
        const nextEndSourceSec = clamp(
          (snappedEnd - prev.clipTimelineStartSec) * prev.clipSpeed,
          prev.initialStartSec + MIN_CLIP_SOURCE_DURATION_SEC,
          prev.clipSourceDurationSec
        );
        return {
          ...prev,
          currentDurationSec: Math.max(nextEndSourceSec - prev.initialStartSec, MIN_CLIP_SOURCE_DURATION_SEC),
        };
      });
    }

    function onUp() {
      setCaptionDragState((prev) => {
        if (!prev) return prev;
        if (prev.mode === "move") {
          if (Math.abs(prev.currentStartSec - prev.initialStartSec) >= 0.01) {
            onMoveCaptionBlock?.(prev.selection, Number(prev.currentStartSec.toFixed(3)));
          }
        } else if (
          Math.abs(prev.currentStartSec - prev.initialStartSec) >= 0.01 ||
          Math.abs(prev.currentDurationSec - prev.initialDurationSec) >= 0.01
        ) {
          onTrimCaptionBlock?.(
            prev.selection,
            Number(prev.currentStartSec.toFixed(3)),
            Number(prev.currentDurationSec.toFixed(3))
          );
        }
        return null;
      });
    }

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [captionDragState, onMoveCaptionBlock, onTrimCaptionBlock, pxPerSec, resolveBlockSnap, resolveEdgeSnap]);

  function handleContextMenu(event: ReactMouseEvent) {
    event.preventDefault();
    setContextMenu({ x: event.clientX, y: event.clientY });
  }

  useEffect(() => {
    function close() {
      setContextMenu(null);
    }
    window.addEventListener("click", close);
    return () => window.removeEventListener("click", close);
  }, []);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    function onWheel(event: WheelEvent) {
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        setPxPerSec((prev) => {
          const delta = event.deltaY > 0 ? 0.85 : 1.18;
          return Math.max(MIN_PX_PER_SEC, Math.min(MAX_PX_PER_SEC, prev * delta));
        });
      }
    }
    element.addEventListener("wheel", onWheel, { passive: false });
    return () => element.removeEventListener("wheel", onWheel);
  }, []);

  const selectedCount = selectedWordIds.size;
  const selectedHasDeleted = useMemo(() => {
    for (const id of selectedWordIds) {
      if (deletedWordIds.has(id)) return true;
    }
    return false;
  }, [selectedWordIds, deletedWordIds]);

  const sectionStyle = useMemo(
    () => ({ "--timeline-rail-width": `${TRACK_LEFT_MARGIN}px` } as CSSProperties),
    []
  );

  return (
    <section className="timeline card" style={sectionStyle}>
      <div className="timelineHeader">
        <div className="timelineHeaderCopy">
          <h3>Timeline</h3>
          <span className="tlHint">
            Drag clips to move, drag edges to trim, press <kbd>S</kbd> to split, use <kbd>Delete</kbd> to remove.
          </span>
          {speakerLegend.length > 1 && (
            <div className="timelineSpeakerLegend" aria-label="Speaker legend">
              {speakerLegend.map((entry) => (
                <span
                  key={entry.speakerId}
                  className={`timelineSpeakerLegendItem ${
                    entry.slot === 0 ? "speakerA" : entry.slot === 1 ? "speakerB" : "speakerExtra"
                  }`}
                >
                  {entry.label}
                </span>
              ))}
            </div>
          )}
        </div>
        <div className="timelineHeaderActions">
          <button
            type="button"
            className={`timelineAssistToggle ${showTranscriptAssist ? "active" : ""}`}
            disabled={!words.length}
            onClick={() => setShowTranscriptAssist((prev) => !prev)}
            title="Show or hide the transcript assist lane"
          >
            Transcript Assist
          </button>
          <div className="zoomControls">
            <button className="zoomBtn" onClick={() => setPxPerSec((prev) => Math.max(MIN_PX_PER_SEC, prev * 0.7))} title="Zoom out" aria-label="Zoom out">
              <Minus size={14} aria-hidden="true" />
            </button>
            <input
              type="range"
              min={MIN_PX_PER_SEC}
              max={MAX_PX_PER_SEC}
              step={1}
              value={pxPerSec}
              onChange={(event) => setPxPerSec(Number(event.target.value))}
              className="zoomSlider"
            />
            <button className="zoomBtn" onClick={() => setPxPerSec((prev) => Math.min(MAX_PX_PER_SEC, prev * 1.4))} title="Zoom in" aria-label="Zoom in">
              <Plus size={14} aria-hidden="true" />
            </button>
            <span className="zoomLabel">{Math.round(pxPerSec)}px/s</span>
          </div>
        </div>
      </div>

      <div
        className="timelineScroll"
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onContextMenu={handleContextMenu}
      >
        <div className="timelineCanvas" style={{ width: totalWidth }}>
          <div className="timeRuler">
            {ticks.map((tick) => (
              <div key={`${tick.sec}-${tick.major ? "major" : "minor"}`} className={`tick ${tick.major ? "major" : ""}`} style={{ left: tick.x }}>
                {tick.major && <span className="tickLabel">{tick.label}</span>}
              </div>
            ))}
          </div>

          {laneBlocks.map((lane) => (
            <div key={lane.id} className={`timelineLane ${lane.kind} ${lane.isLocked ? "locked" : ""}`}>
              <div className="trackRail">
                <span className="trackRailLabel">{lane.label}</span>
                <div className="trackRailActions">
                  <button
                    type="button"
                    className={`trackRailBtn ${lane.mute ? "active danger" : ""}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      onToggleLaneMute?.(lane);
                    }}
                    title={lane.mute ? "Unmute track" : "Mute track"}
                  >
                    {lane.mute ? <VolumeX size={12} strokeWidth={2} aria-hidden="true" /> : <Volume2 size={12} strokeWidth={2} aria-hidden="true" />}
                  </button>
                  <button
                    type="button"
                    className={`trackRailBtn ${lane.solo ? "active" : ""}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      onToggleLaneSolo?.(lane);
                    }}
                    title={lane.solo ? "Disable solo" : "Solo track"}
                  >
                    S
                  </button>
                  <button
                    type="button"
                    className={`trackRailBtn ${lane.isLocked ? "active" : ""}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      onToggleLaneLock?.(lane.id);
                    }}
                    title={lane.isLocked ? "Unlock lane" : "Lock lane"}
                  >
                    {lane.isLocked ? <Lock size={12} strokeWidth={2} aria-hidden="true" /> : <LockOpen size={12} strokeWidth={2} aria-hidden="true" />}
                  </button>
                </div>
              </div>

              {lane.blocks.length === 0 && <div className="laneEmpty">No clips</div>}

              {lane.blocks.map(({ clip, x, w, duration, timelineStartSec, thumbSrc, thumbTimes, isSelected, isDragging }) => (
                <div
                  key={clip.id}
                  className={[
                    "timelineLaneClip",
                    lane.kind,
                    isSelected ? "selected" : "",
                    isDragging ? "dragging" : "",
                    lane.isLocked ? "locked" : "",
                  ].filter(Boolean).join(" ")}
                  style={{ left: x, width: w }}
                  onMouseDown={(event) => startLaneDrag(event, lane, clip, "move")}
                  onClick={(event) => {
                    event.stopPropagation();
                    const selection: TimelineLaneClipSelection = {
                      clipId: clip.id,
                      laneId: lane.id,
                      laneLabel: lane.label,
                      laneKind: lane.kind,
                    };
                    onSeek(timelineStartSec);
                    onSelectLaneClip?.(selection);
                    onSelectBrollClip?.(null);
                    onSelectCaptionBlock?.(null);
                  }}
                  title={`${lane.label} · ${formatTimecode(timelineStartSec)} · ${formatDuration(duration)}`}
                >
                  <div className="laneClipHandle start" onMouseDown={(event) => startLaneDrag(event, lane, clip, "trim-start")} title="Trim clip in" />
                  <div className="laneClipHandle end" onMouseDown={(event) => startLaneDrag(event, lane, clip, "trim-end")} title="Trim clip out" />
                  {lane.kind === "video" && thumbSrc && w > 36 && (
                    <div className="timelineLaneFilmstrip">
                      {thumbTimes.map((seekSec, index) => (
                        <LaneClipThumb key={`${clip.id}-thumb-${index}`} src={thumbSrc} seekSec={seekSec} />
                      ))}
                    </div>
                  )}
                  <span className="laneClipMeta">{w > 72 ? formatDuration(duration) : ""}</span>
                </div>
              ))}
            </div>
          ))}

          <div className="captionTrack">
            <div className="trackRail">
              <span className="trackRailLabel">
                <Captions size={12} strokeWidth={1.9} aria-hidden="true" />
                <span>CC</span>
              </span>
            </div>

            {renderedCaptionBlocks.length === 0 && <div className="laneEmpty">No caption blocks</div>}

            {renderedCaptionBlocks.map((block) => (
              <div
                key={block.id}
                className={[
                  "captionBlock",
                  block.isSelected ? "selected" : "",
                  block.isDragging ? "dragging" : "",
                ].filter(Boolean).join(" ")}
                style={{ left: block.x, width: block.w }}
                onMouseDown={(event) => startCaptionDrag(event, block, "move")}
                onClick={(event) => {
                  event.stopPropagation();
                  onSeek(block.renderedStartSec);
                  onSelectCaptionBlock?.({
                    overlayId: block.id,
                    clipId: block.clipId,
                    laneId: block.laneId,
                    laneLabel: block.laneLabel,
                    text: block.text,
                    style: block.style,
                  });
                  onSelectLaneClip?.(null);
                  onSelectBrollClip?.(null);
                }}
                onDoubleClick={(event) => {
                  event.stopPropagation();
                  if (!onStartEditCaption) return;
                  onStartEditCaption({
                    overlayId: block.id,
                    clipId: block.clipId,
                    laneId: block.laneId,
                    laneLabel: block.laneLabel,
                    text: block.text,
                    style: block.style,
                  });
                }}
                title={`Caption · ${formatTimecode(block.renderedStartSec)} · ${formatDuration(block.renderedDurationSec)} · double-click to edit`}
              >
                <div className="captionBlockHandle start" onMouseDown={(event) => startCaptionDrag(event, block, "trim-start")} title="Trim caption in" />
                {editingCaptionId === block.id ? (
                  <input
                    ref={captionEditInputRef}
                    className="captionBlockEditInput wordEditInput"
                    value={editingCaptionText}
                    onClick={(event) => event.stopPropagation()}
                    onMouseDown={(event) => event.stopPropagation()}
                    onChange={(event) => onCaptionTextChange?.(event.target.value)}
                    onBlur={() => onCommitCaptionEdit?.()}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        event.preventDefault();
                        onCommitCaptionEdit?.();
                      }
                      if (event.key === "Escape") {
                        event.preventDefault();
                        onCancelCaptionEdit?.();
                      }
                    }}
                  />
                ) : (
                  <span className="captionBlockText">{block.w > 40 ? trimInlineText(block.text, 44) : ""}</span>
                )}
                <div className="captionBlockHandle end" onMouseDown={(event) => startCaptionDrag(event, block, "trim-end")} title="Trim caption out" />
              </div>
            ))}
          </div>

          <div className="waveformTrack">
            <div className="trackRail">
              <span className="trackRailLabel">WFM</span>
            </div>
            <svg className="waveformSvg" width={totalWidth} height={50} preserveAspectRatio="none">
              {waveformBars.map((bar, index) => (
                <rect key={index} x={bar.x} y={50 - bar.height * 46} width={bar.width} height={bar.height * 46} rx={1} />
              ))}
            </svg>
            {deletedRegions.map((region, index) => (
              <div
                key={`del-wave-${index}`}
                className="deletedOverlay"
                style={{ left: region.startSec * pxPerSec, width: (region.endSec - region.startSec) * pxPerSec }}
              />
            ))}
          </div>

          <div className="brollTrack">
            <div className="trackRail">
              <span className="trackRailLabel">B</span>
            </div>
            {brollBlocks.length === 0 && <div className="brollEmpty">No overlay clips</div>}
            {brollBlocks.map(({ clip, x, w, opacity, timelineStartSec, duration, isDragging }) => {
              const isSelected = selectedBrollClipId === clip.id;
              return (
                <div
                  key={clip.id}
                  className={[
                    "brollBlock",
                    isSelected ? "selected" : "",
                    isDragging ? "dragging" : "",
                    brollEditBusy ? "disabled" : "",
                  ].filter(Boolean).join(" ")}
                  style={{ left: x, width: w, opacity: Math.max(0.28, Math.min(opacity, 1)) }}
                  onMouseDown={(event) => startBrollDrag(event, clip, "move")}
                  onClick={(event) => {
                    event.stopPropagation();
                    clearAllSelections();
                    onSelectBrollClip?.(clip.id);
                    onSeek(timelineStartSec);
                  }}
                  onWheel={(event) => {
                    if (!event.altKey || brollEditBusy) return;
                    event.preventDefault();
                    event.stopPropagation();
                    clearAllSelections();
                    onSelectBrollClip?.(clip.id);
                    const currentOpacity = brollOpacityDraftById[clip.id] ?? clamp(typeof clip.broll_opacity === "number" ? clip.broll_opacity : 1, 0, 1);
                    const step = event.deltaY < 0 ? 0.04 : -0.04;
                    const nextOpacity = clamp(currentOpacity + step, 0, 1);
                    setBrollOpacityDraftById((prev) => ({ ...prev, [clip.id]: nextOpacity }));
                    scheduleOpacityCommit(clip.id, Number(nextOpacity.toFixed(3)));
                  }}
                  title={`B-roll ${formatTimecode(timelineStartSec)} · ${formatDuration(duration)} · opacity ${(opacity * 100).toFixed(0)}%`}
                >
                  {w > 74 ? `${(opacity * 100).toFixed(0)}%` : ""}
                  <button
                    type="button"
                    className="brollRerollBtn"
                    disabled={brollEditBusy}
                    onMouseDown={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                    }}
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      onRerollBrollClip?.(clip.id);
                    }}
                    title="Re-roll B-roll clip"
                  >
                    <RefreshCw size={10} strokeWidth={2.1} aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    className="brollDeleteBtn"
                    disabled={brollEditBusy}
                    onMouseDown={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                    }}
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      onDeleteBrollClip(clip.id);
                    }}
                    title="Remove B-roll clip"
                  >
                    <Trash2 size={10} strokeWidth={2.1} aria-hidden="true" />
                  </button>
                  <div className="brollResizeHandle" onMouseDown={(event) => startBrollDrag(event, clip, "resize-end")} title="Trim B-roll duration" />
                </div>
              );
            })}
          </div>

          {showTranscriptAssist && (
            <div className="wordTrack">
              <div className="trackRail">
                <span className="trackRailLabel">TXT</span>
              </div>
              {deletedRegions.map((region, index) => (
                <div
                  key={`del-word-${index}`}
                  className="deletedOverlay wordDeletedOverlay"
                  style={{ left: region.startSec * pxPerSec, width: (region.endSec - region.startSec) * pxPerSec }}
                />
              ))}
              {wordBlocks.map(({ word, x, w, speakerSlot, isDeleted, isSelected, isActive }) => (
                <div
                  key={word.id}
                  className={[
                    "tlWord",
                    isDeleted ? "deleted" : "",
                    isSelected ? "selected" : "",
                    isActive ? "active" : "",
                    speakerSlot === 0 ? "speakerA" : "",
                    speakerSlot === 1 ? "speakerB" : "",
                    speakerSlot !== null && speakerSlot >= 2 ? "speakerExtra" : "",
                  ].filter(Boolean).join(" ")}
                  style={{ left: x, width: w }}
                  onMouseDown={(event) => {
                    event.stopPropagation();
                  }}
                  onClick={(event) => {
                    event.stopPropagation();
                    onSelectLaneClip?.(null);
                    onSelectBrollClip?.(null);
                    onSelectCaptionBlock?.(null);
                    onSelectWord(word.id, event.shiftKey);
                  }}
                  onDoubleClick={(event) => {
                    event.stopPropagation();
                    if (!isDeleted && onEditWord) {
                      onSeek(word.start_sec);
                      onEditWord(word.id);
                    }
                  }}
                  title={
                    isDeleted
                      ? `${word.text} (deleted — select and use Restore, or right-click Restore Selected Words)`
                      : `${word.speaker_label ? `${word.speaker_label}: ` : ""}${word.text} · double-click to edit`
                  }
                >
                  {w > 24 ? word.text : ""}
                  {!isDeleted && w > 24 && (
                    <Pencil size={8} className="tlWordEditHint" aria-hidden="true" />
                  )}
                </div>
              ))}
            </div>
          )}

          {rangeLeft !== null && rangeWidth !== null && rangeWidth > 2 && (
            <div className="rangeSelection" style={{ left: rangeLeft + TRACK_LEFT_MARGIN, width: rangeWidth }}>
              {rangeDuration !== null && rangeDuration > 0.1 && <span className="rangeLabel">{formatDuration(rangeDuration)}</span>}
            </div>
          )}

          <div className="timeline-playhead" style={{ left: playheadX }}>
            <div className="timeline-playheadHead" />
            <div className="timeline-playheadLine" />
          </div>
        </div>
      </div>

      {contextMenu && (
        <div className="tlContextMenu" style={{ left: contextMenu.x, top: contextMenu.y }}>
          {/* ── Word actions ── */}
          <button disabled={!selectedCount} onClick={() => { onDeleteSelected(); setContextMenu(null); }}>
            <Trash2 size={14} strokeWidth={1.9} aria-hidden="true" />
            <span>Delete Selected Words ({selectedCount})</span>
          </button>
          <button disabled={!selectedHasDeleted} onClick={() => { onRestoreSelected(); setContextMenu(null); }}>
            <RotateCcw size={14} strokeWidth={1.9} aria-hidden="true" />
            <span>Restore Selected Words</span>
          </button>
          <hr />
          {/* ── Clip actions ── */}
          {selectedLaneClipId && (() => {
            const selectedBlock = laneBlocks.flatMap((lane) => lane.blocks).find((b) => b.clip.id === selectedLaneClipId);
            const selectedLane = laneBlocks.find((lane) => lane.blocks.some((b) => b.clip.id === selectedLaneClipId));
            const sel = selectedBlock && selectedLane ? {
              clipId: selectedBlock.clip.id,
              laneId: selectedLane.id,
              laneLabel: selectedLane.label,
              laneKind: selectedLane.kind,
            } as TimelineLaneClipSelection : null;
            if (!sel) return null;
            const clipStart = selectedBlock!.timelineStartSec;
            const clipEnd = clipStart + selectedBlock!.duration;
            const canSplit = currentTimeSec > clipStart + 0.01 && currentTimeSec < clipEnd - 0.01;
            return (
              <>
                <button onClick={() => { if (selectedBlock) onSeek(selectedBlock.timelineStartSec); setContextMenu(null); }}>
                  <MapPin size={14} strokeWidth={1.9} aria-hidden="true" />
                  <span>Jump to Clip Start</span>
                </button>
                <button
                  disabled={!canSplit || !onSplitLaneClip}
                  onClick={() => { if (sel && canSplit) { onSplitLaneClip?.(sel); } setContextMenu(null); }}
                  title={canSplit ? "Split clip at current playhead position" : "Move playhead inside the clip to split"}
                >
                  <Scissors size={14} strokeWidth={1.9} aria-hidden="true" />
                  <span>Split Clip at Playhead</span>
                </button>
                <button
                  className="danger"
                  onClick={() => { onDeleteLaneClip?.(sel); setContextMenu(null); }}
                >
                  <Trash2 size={14} strokeWidth={1.9} aria-hidden="true" />
                  <span>Delete Clip</span>
                </button>
              </>
            );
          })()}
          {!selectedLaneClipId && (
            <button onClick={() => { onSeek(currentTimeSec); setContextMenu(null); }}>
              <MapPin size={14} strokeWidth={1.9} aria-hidden="true" />
              <span>Seek to Playhead ({formatTimecode(currentTimeSec)})</span>
            </button>
          )}
        </div>
      )}
    </section>
  );
}

Timeline.displayName = "Timeline";

export default memo(Timeline);
