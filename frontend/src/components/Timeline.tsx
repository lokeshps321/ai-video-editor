import {
  memo,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
} from "react";
import {
  Captions,
  ClipboardPaste,
  Copy,
  CopyPlus,
  Lock,
  LockOpen,
  MapPin,
  Minus,
  Pencil,
  Plus,
  RefreshCw,
  RotateCcw,
  Scissors,
  Trash2,
  Volume2,
  VolumeX,
} from "lucide-react";
import type { Clip, TranscriptWord } from "../types";
import { api } from "../lib/api";
import {
  createTimelineClock,
  type TimelineClock,
  type TimelineClockPreview,
} from "../timeline/clock";
import { createGestureController } from "../timeline/gestureController";
import {
  canClaimGesturePointer,
  minimumDurationFrames,
  shouldCommitFrameChange,
  snapBlockStartFrame,
  snapEdgeFrame,
  sourceBoundaryFrame,
  sourceFrameAfterTimelineDelta,
} from "../timeline/integration";
import { frameToSeconds, secondsToFrame } from "../timeline/timebase";
import type { SnapGuide as FrameSnapGuide } from "../timeline/snapping";
import { zoomViewportAtCursor } from "../timeline/viewport";
import "./Timeline.css";

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

export type TimelineDropTarget = {
  laneId: string;
  kind: "video" | "audio" | "overlay";
  label: string;
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
  assetNameById?: Map<string, string>;
  assetDurationById: Map<string, number | null>;
  overlayClips: Clip[];
  captionBlocks: TimelineCaptionBlock[];
  durationSec: number;
  currentTimeSec: number;
  fps?: number;
  timelineClock?: TimelineClock;
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
  onMoveLaneClip: (
    selection: TimelineLaneClipSelection,
    timelineStartSec: number,
    dropTarget?: TimelineDropTarget,
  ) => void;
  onTrimLaneClip: (
    selection: TimelineLaneClipSelection,
    nextRange: { startSec: number; endSec: number },
  ) => void;
  onToggleLaneMute?: (lane: TimelineLane) => void;
  onToggleLaneSolo?: (lane: TimelineLane) => void;
  onToggleLaneLock?: (laneId: string) => void;
  onMoveBrollClip: (clipId: string, timelineStartSec: number) => void;
  onMoveBrollClipToLane?: (
    clipId: string,
    timelineStartSec: number,
    dropTarget: TimelineDropTarget,
  ) => void;
  onTrimBrollClip: (clipId: string, durationSec: number) => void;
  onSetBrollOpacity: (clipId: string, opacity: number) => void;
  onDeleteBrollClip: (clipId: string) => void;
  onRerollBrollClip?: (clipId: string) => void;
  onSelectLaneClip?: (selection: TimelineLaneClipSelection | null) => void;
  onSelectBrollClip?: (clipId: string | null) => void;
  onSelectCaptionBlock?: (selection: TimelineCaptionSelection | null) => void;
  onMoveCaptionBlock?: (
    selection: TimelineCaptionSelection,
    startSec: number,
  ) => void;
  onTrimCaptionBlock?: (
    selection: TimelineCaptionSelection,
    startSec: number,
    durationSec: number,
  ) => void;
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
  onCopyLaneClip?: (selection: TimelineLaneClipSelection) => void;
  onDuplicateLaneClip?: (selection: TimelineLaneClipSelection) => void;
  onPasteLaneClip?: (atSec: number) => void;
  canPasteLaneClip?: boolean;
  brollEditBusy: boolean;
};

const MIN_PX_PER_SEC = 15;
const MAX_PX_PER_SEC = 250;
const DEFAULT_PX_PER_SEC = 40;
const TRACK_LEFT_MARGIN = 96;
const MIN_BROLL_DURATION_SEC = 0.1;
const MIN_CLIP_SOURCE_DURATION_SEC = 0.05;
const BROLL_LANE_ID = "__broll__";
const SNAP_INDICATOR_EPSILON_SEC = 0.002;
const TIMELINE_CORE_V2 = import.meta.env.VITE_TIMELINE_CORE_V2 !== "false";

type DropLaneRect = {
  laneId: string;
  kind: "video" | "audio" | "overlay";
  label: string;
  locked: boolean;
  top: number;
  bottom: number;
};

type LaneClipThumbProps = {
  assetId: string;
  seekSec: number;
};

const FILMSTRIP_THUMB_WIDTH = 160;

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

type ActiveV2Gesture = {
  controller: ReturnType<typeof createGestureController>;
  clockPreview?: TimelineClockPreview;
};

const noopSubscribe = () => () => undefined;

const TimelinePlayhead = memo(function TimelinePlayhead({
  currentTimeSec,
  pxPerSec,
  timelineClock,
  containerRef,
  autoScrollEnabledRef,
}: {
  currentTimeSec: number;
  pxPerSec: number;
  timelineClock?: TimelineClock;
  containerRef: RefObject<HTMLDivElement | null>;
  autoScrollEnabledRef: RefObject<boolean>;
}) {
  const clockTime = useSyncExternalStore(
    timelineClock?.subscribe ?? noopSubscribe,
    timelineClock?.getSnapshot ?? (() => currentTimeSec),
    timelineClock?.getSnapshot ?? (() => currentTimeSec),
  );
  const timeSec = timelineClock ? clockTime : currentTimeSec;
  const playheadX = timeSec * pxPerSec;
  const lastAutoScrollTimeRef = useRef<number | null>(null);

  useEffect(() => {
    if (!autoScrollEnabledRef.current) return;
    if (
      lastAutoScrollTimeRef.current !== null &&
      lastAutoScrollTimeRef.current === timeSec
    ) {
      return;
    }
    lastAutoScrollTimeRef.current = timeSec;
    const element = containerRef.current;
    if (!element) return;
    const viewLeft = element.scrollLeft;
    const viewRight = viewLeft + element.clientWidth;
    if (playheadX < viewLeft + 40 || playheadX > viewRight - 40) {
      element.scrollLeft = playheadX - element.clientWidth / 3;
    }
  }, [autoScrollEnabledRef, containerRef, playheadX, timeSec]);

  return (
    <div
      className="timeline-playhead"
      style={{ "--timeline-playhead-x": `${playheadX}px` } as CSSProperties}
    >
      <div className="timeline-playheadHead" />
      <div className="timeline-playheadLine" />
    </div>
  );
});

const LaneClipThumb = memo(function LaneClipThumb({
  assetId,
  seekSec,
}: LaneClipThumbProps) {
  const [loadedSrc, setLoadedSrc] = useState<string | null>(null);
  const src = useMemo(
    () => api.mediaThumbnailUrl(assetId, seekSec, FILMSTRIP_THUMB_WIDTH),
    [assetId, seekSec],
  );
  // Compare against the URL that actually loaded instead of resetting state
  // in an effect. After transcript generation the V1 clip is rebuilt with
  // the same thumbnail URLs. Cached images can fire `load` before that
  // effect runs, causing the old code to immediately hide valid frames.
  const isReady = loadedSrc === src;

  return (
    <div className="timelineLaneThumb">
      <img
        className={`timelineLaneThumbVideo ${isReady ? "ready" : ""}`}
        src={src}
        loading="lazy"
        decoding="async"
        draggable={false}
        onLoad={() => setLoadedSrc(src)}
        onError={() => setLoadedSrc(null)}
        alt=""
        aria-hidden="true"
      />
    </div>
  );
});

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
  return Math.max(
    (clip.end_sec - clip.start_sec) / Math.max(clip.speed, 0.01),
    MIN_BROLL_DURATION_SEC,
  );
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

function speakerSlotForWord(
  word: TranscriptWord,
  speakerIdOrder: string[],
): number | null {
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
  assetNameById,
  assetDurationById,
  overlayClips,
  captionBlocks,
  durationSec,
  currentTimeSec,
  fps = 30,
  timelineClock,
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
  onMoveBrollClipToLane,
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
  onCopyLaneClip,
  onDuplicateLaneClip,
  onPasteLaneClip,
  canPasteLaneClip = false,
  brollEditBusy,
}: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const internalTimelineClockRef = useRef(createTimelineClock(currentTimeSec));
  const activeTimelineClock = timelineClock ?? internalTimelineClockRef.current;
  const [pxPerSec, setPxPerSec] = useState(DEFAULT_PX_PER_SEC);
  const pxPerSecRef = useRef(DEFAULT_PX_PER_SEC);
  const zoomAnimationFrameRef = useRef<number | null>(null);
  const pendingZoomScrollRef = useRef<number | null>(null);
  const playheadAutoScrollEnabledRef = useRef(true);
  const [showTranscriptAssist, setShowTranscriptAssist] = useState(
    words.length > 0,
  );

  type DragMode = "none" | "seek" | "range";
  const [dragMode, setDragMode] = useState<DragMode>("none");
  const [rangeStart, setRangeStart] = useState<number | null>(null);
  const [rangeEnd, setRangeEnd] = useState<number | null>(null);
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [brollOpacityDraftById, setBrollOpacityDraftById] = useState<
    Record<string, number>
  >({});
  const opacityCommitTimersRef = useRef<Record<string, number>>({});
  const [brollDragState, setBrollDragState] = useState<BrollDragState | null>(
    null,
  );
  const [laneDragState, setLaneDragState] = useState<LaneDragState | null>(
    null,
  );
  const [captionDragState, setCaptionDragState] =
    useState<CaptionDragState | null>(null);
  const [dropTarget, setDropTarget] = useState<TimelineDropTarget | null>(
    null,
  );
  const dropLaneRectsRef = useRef<DropLaneRect[]>([]);
  const activeV2GestureRef = useRef<ActiveV2Gesture | null>(null);
  const suppressClickRef = useRef(false);

  const totalWidth = Math.max(durationSec * pxPerSec, 200);

  useEffect(() => {
    if (TIMELINE_CORE_V2) activeTimelineClock.setTime(currentTimeSec);
  }, [activeTimelineClock, currentTimeSec]);

  useEffect(() => {
    pxPerSecRef.current = pxPerSec;
  }, [pxPerSec]);

  useLayoutEffect(() => {
    const pendingScroll = pendingZoomScrollRef.current;
    const container = containerRef.current;
    if (pendingScroll === null || !container) return;
    container.scrollLeft = pendingScroll;
    pendingZoomScrollRef.current = null;
  }, [pxPerSec]);

  useLayoutEffect(() => {
    if (import.meta.env.VITE_TIMELINE_TEST_HARNESS === "true") {
      window.dispatchEvent(new CustomEvent("timeline:v2-react-commit"));
    }
  });

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

    const result: { sec: number; x: number; label: string; major: boolean }[] =
      [];
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
  const waveformRects = useMemo(
    () =>
      waveformBars.map((bar, index) => (
        <rect
          key={index}
          x={bar.x}
          y={50 - bar.height * 46}
          width={bar.width}
          height={bar.height * 46}
          rx={1}
        />
      )),
    [waveformBars],
  );

  const deletedRegions = useMemo(() => {
    if (!words.length) return [];
    const sorted = [...words].sort((a, b) => a.start_sec - b.start_sec);
    const regions: { startSec: number; endSec: number }[] = [];
    let regionStart: number | null = null;

    for (const word of sorted) {
      if (deletedWordIds.has(word.id)) {
        if (regionStart === null) regionStart = word.start_sec;
      } else if (regionStart !== null) {
        const previousDeleted = sorted.find(
          (candidate) =>
            candidate.end_sec <= word.start_sec &&
            deletedWordIds.has(candidate.id),
        );
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
      const label =
        words.find((word) => word.speaker_id === speakerId)?.speaker_label ||
        `Speaker ${slot + 1}`;
      return { speakerId, label, slot };
    });
  }, [words]);

  const legacyPlayheadGuideTime = TIMELINE_CORE_V2 ? null : currentTimeSec;
  const snapGuides = useMemo(() => {
    const guides: SnapGuide[] = [
      { ownerKey: "system:start", timeSec: 0 },
      { ownerKey: "system:end", timeSec: durationSec },
    ];
    if (legacyPlayheadGuideTime !== null) {
      guides.splice(1, 0, {
        ownerKey: "system:playhead",
        timeSec: legacyPlayheadGuideTime,
      });
    }

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
      guides.push({
        ownerKey: `caption:${block.id}:start`,
        timeSec: block.startSec,
      });
      guides.push({
        ownerKey: `caption:${block.id}:end`,
        timeSec: block.startSec + block.durationSec,
      });
    });
    return guides;
  }, [
    timelineLanes,
    overlayClips,
    captionBlocks,
    legacyPlayheadGuideTime,
    durationSec,
  ]);

  const frameSnapGuides = useMemo<FrameSnapGuide[]>(
    () =>
      snapGuides.map((guide) => ({
        id: guide.ownerKey,
        frame: secondsToFrame(guide.timeSec, fps),
      })),
    [fps, snapGuides],
  );

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
    [snapGuides, snapThresholdSec],
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
    [snapGuides, snapThresholdSec],
  );

  const captureDropLanes = useCallback(() => {
    const container = containerRef.current;
    if (!container) {
      dropLaneRectsRef.current = [];
      return;
    }
    dropLaneRectsRef.current = Array.from(
      container.querySelectorAll<HTMLElement>("[data-drop-lane-id]"),
    ).map((element) => {
      const rect = element.getBoundingClientRect();
      return {
        laneId: element.dataset.dropLaneId ?? "",
        kind: (element.dataset.dropLaneKind ?? "video") as DropLaneRect["kind"],
        label: element.dataset.dropLaneLabel ?? "",
        locked: element.dataset.dropLaneLocked === "1",
        top: rect.top,
        bottom: rect.bottom,
      };
    });
  }, []);

  const resolveDropTarget = useCallback(
    (clientY: number, sourceLaneId: string): TimelineDropTarget | null => {
      const hit = dropLaneRectsRef.current.find(
        (lane) => clientY >= lane.top && clientY <= lane.bottom,
      );
      if (!hit || hit.locked || hit.laneId === sourceLaneId) return null;
      return { laneId: hit.laneId, kind: hit.kind, label: hit.label };
    },
    [],
  );

  // Vertical guide shown while a dragged block edge is aligned with another
  // clip edge, the playhead, or the timeline bounds.
  const snapIndicatorSec = useMemo(() => {
    let ownerPrefix: string | null = null;
    const edges: number[] = [];
    if (laneDragState) {
      ownerPrefix = `lane:${laneDragState.selection.clipId}:`;
      const timelineEnd =
        laneDragState.currentTimelineStartSec +
        (laneDragState.currentEndSec - laneDragState.currentStartSec) /
          laneDragState.speed;
      edges.push(laneDragState.currentTimelineStartSec, timelineEnd);
    } else if (brollDragState) {
      ownerPrefix = `broll:${brollDragState.clipId}:`;
      edges.push(
        brollDragState.currentStartSec,
        brollDragState.currentStartSec + brollDragState.currentDurationSec,
      );
    } else if (captionDragState) {
      ownerPrefix = `caption:${captionDragState.selection.overlayId}:`;
      const speed = Math.max(captionDragState.clipSpeed, 0.01);
      const startSec =
        captionDragState.clipTimelineStartSec +
        captionDragState.currentStartSec / speed;
      edges.push(startSec, startSec + captionDragState.currentDurationSec / speed);
    }
    if (!ownerPrefix || !edges.length) return null;
    for (const guide of snapGuides) {
      if (guide.ownerKey.startsWith(ownerPrefix)) continue;
      for (const edge of edges) {
        if (Math.abs(guide.timeSec - edge) <= SNAP_INDICATOR_EPSILON_SEC) {
          return guide.timeSec;
        }
      }
    }
    return null;
  }, [laneDragState, brollDragState, captionDragState, snapGuides]);

  const brollBlocks = useMemo(() => {
    return overlayClips
      .slice()
      .sort((a, b) => a.timeline_start_sec - b.timeline_start_sec)
      .map((clip) => {
        const dragPreview =
          brollDragState?.clipId === clip.id ? brollDragState : null;
        const timelineStartSec = dragPreview
          ? dragPreview.currentStartSec
          : clip.timeline_start_sec;
        const duration = dragPreview
          ? dragPreview.currentDurationSec
          : clipTimelineDuration(clip);
        const x = timelineStartSec * pxPerSec;
        const w = Math.max(duration * pxPerSec, 4);
        const clipOpacity =
          typeof clip.broll_opacity === "number" ? clip.broll_opacity : 1;
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
            const dragPreview =
              laneDragState?.selection.clipId === clip.id
                ? laneDragState
                : null;
            const sourceStartSec = dragPreview
              ? dragPreview.currentStartSec
              : clip.start_sec;
            const sourceEndSec = dragPreview
              ? dragPreview.currentEndSec
              : clip.end_sec;
            const timelineStartSec = dragPreview
              ? dragPreview.currentTimelineStartSec
              : clip.timeline_start_sec;
            const duration = Math.max(
              (sourceEndSec - sourceStartSec) / Math.max(clip.speed, 0.01),
              MIN_BROLL_DURATION_SEC,
            );
            const clipWidthPx = Math.max(duration * pxPerSec, 4);
            // One frame roughly every ~120px so the filmstrip stays crisp
            // across the full clip width instead of stretching a handful of
            // frames over a very long clip. Lazy loading keeps off-screen
            // frames from loading until scrolled into view.
            const thumbCount = Math.max(
              1,
              Math.min(80, Math.ceil(clipWidthPx / 120)),
            );
            const sourceDuration = Math.max(
              sourceEndSec - sourceStartSec,
              MIN_CLIP_SOURCE_DURATION_SEC,
            );
            const thumbTimes = Array.from(
              { length: thumbCount },
              (_unused, idx) => {
                // Sample the midpoint of each segment instead of the exact
                // clip edges. Videos commonly fade in/out from black, so
                // sampling at t=start or t=end produces black filmstrip slots.
                const frac = (idx + 0.5) / thumbCount;
                return sourceStartSec + sourceDuration * frac;
              },
            );
            return {
              clip,
              x: timelineStartSec * pxPerSec,
              w: clipWidthPx,
              duration,
              sourceStartSec,
              sourceEndSec,
              timelineStartSec,
              thumbSrc: assetUrlById.get(clip.asset_id) ?? null,
              assetId: clip.asset_id,
              clipName: (() => {
                const raw = assetNameById?.get(clip.asset_id);
                return raw ? raw.replace(/\.[^.]+$/, "") : null;
              })(),
              thumbTimes,
              isSelected: selectedLaneClipId === clip.id,
              isDragging: !!dragPreview,
            };
          }),
      };
    });
  }, [
    timelineLanes,
    lockedLaneIds,
    laneDragState,
    pxPerSec,
    assetUrlById,
    assetNameById,
    selectedLaneClipId,
  ]);

  const renderedCaptionBlocks = useMemo(() => {
    return captionBlocks
      .slice()
      .sort((a, b) => a.startSec - b.startSec)
      .map((block) => {
        const dragPreview =
          captionDragState?.selection.overlayId === block.id
            ? captionDragState
            : null;
        const startSec = dragPreview
          ? dragPreview.clipTimelineStartSec +
            dragPreview.currentStartSec / Math.max(dragPreview.clipSpeed, 0.01)
          : block.startSec;
        const duration = dragPreview
          ? dragPreview.currentDurationSec /
            Math.max(dragPreview.clipSpeed, 0.01)
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
        next[clip.id] = clamp(
          typeof clip.broll_opacity === "number" ? clip.broll_opacity : 1,
          0,
          1,
        );
      });
      return next;
    });
  }, [overlayClips]);

  useEffect(() => {
    return () => {
      Object.values(opacityCommitTimersRef.current).forEach((timer) =>
        window.clearTimeout(timer),
      );
      opacityCommitTimersRef.current = {};
    };
  }, []);

  const rangeLeft =
    rangeStart !== null && rangeEnd !== null
      ? Math.min(rangeStart, rangeEnd) * pxPerSec
      : null;
  const rangeWidth =
    rangeStart !== null && rangeEnd !== null
      ? Math.abs(rangeEnd - rangeStart) * pxPerSec
      : null;
  const rangeDuration =
    rangeStart !== null && rangeEnd !== null
      ? Math.abs(rangeEnd - rangeStart)
      : null;

  useEffect(() => {
    if (TIMELINE_CORE_V2) return;
    if (
      !containerRef.current ||
      dragMode !== "none" ||
      brollDragState ||
      laneDragState ||
      captionDragState
    )
      return;
    const element = containerRef.current;
    const playheadX = currentTimeSec * pxPerSec;
    const viewLeft = element.scrollLeft;
    const viewRight = viewLeft + element.clientWidth;
    if (playheadX < viewLeft + 40 || playheadX > viewRight - 40) {
      element.scrollLeft = playheadX - element.clientWidth / 3;
    }
  }, [
    currentTimeSec,
    pxPerSec,
    dragMode,
    brollDragState,
    laneDragState,
    captionDragState,
  ]);

  const secFromClientX = useCallback(
    (clientX: number) => {
      if (!containerRef.current) return 0;
      const rect = containerRef.current.getBoundingClientRect();
      const x =
        clientX -
        rect.left +
        containerRef.current.scrollLeft -
        TRACK_LEFT_MARGIN;
      return Math.max(0, Math.min(x / pxPerSec, durationSec));
    },
    [pxPerSec, durationSec],
  );

  const secFromEvent = useCallback(
    (event: ReactMouseEvent) => secFromClientX(event.clientX),
    [secFromClientX],
  );

  const gestureThresholdFrames = Math.max(
    1,
    Math.round((10 / pxPerSec) * fps),
  );
  const durationFrames = secondsToFrame(durationSec, fps);

  const computeBrollPreview = useCallback(
    (initial: BrollDragState, deltaX: number): BrollDragState => {
      const deltaFrames = Math.round((deltaX / pxPerSec) * fps);
      const initialStartFrame = secondsToFrame(initial.initialStartSec, fps);
      const initialDurationFrames = Math.max(
        minimumDurationFrames(MIN_BROLL_DURATION_SEC, fps),
        secondsToFrame(initial.initialDurationSec, fps),
      );
      const minimumFrames = minimumDurationFrames(
        MIN_BROLL_DURATION_SEC,
        fps,
      );
      const maxFrame = Math.max(
        durationFrames,
        initialStartFrame + secondsToFrame(30, fps),
      );
      if (initial.mode === "move") {
        const snapped = snapBlockStartFrame(
          Math.max(0, initialStartFrame + deltaFrames),
          initialDurationFrames,
          frameSnapGuides,
          gestureThresholdFrames,
          `broll:${initial.clipId}:`,
          0,
          maxFrame,
        );
        return {
          ...initial,
          currentStartSec: frameToSeconds(snapped.frame, fps),
        };
      }
      const rawEndFrame = initialStartFrame + initialDurationFrames + deltaFrames;
      const snapped = snapEdgeFrame(
        rawEndFrame,
        frameSnapGuides,
        gestureThresholdFrames,
        `broll:${initial.clipId}:`,
        initialStartFrame + minimumFrames,
        Math.max(maxFrame, rawEndFrame),
      );
      return {
        ...initial,
        currentDurationSec: frameToSeconds(
          snapped.frame - initialStartFrame,
          fps,
        ),
      };
    },
    [
      durationFrames,
      fps,
      frameSnapGuides,
      gestureThresholdFrames,
      pxPerSec,
    ],
  );

  const computeLanePreview = useCallback(
    (initial: LaneDragState, deltaX: number): LaneDragState => {
      const deltaFrames = Math.round((deltaX / pxPerSec) * fps);
      const ownerPrefix = `lane:${initial.selection.clipId}:`;
      const minimumSourceFrames = minimumDurationFrames(
        MIN_CLIP_SOURCE_DURATION_SEC,
        fps,
      );
      const initialSourceStartFrame = sourceBoundaryFrame(
        initial.initialStartSec,
        fps,
        "start",
      );
      const initialSourceEndFrame = Math.max(
        initialSourceStartFrame + minimumSourceFrames,
        sourceBoundaryFrame(initial.initialEndSec, fps, "end"),
      );
      const startFrame = secondsToFrame(initial.initialTimelineStartSec, fps);
      const endFrame =
        startFrame +
        Math.max(
          1,
          Math.round(
            (initialSourceEndFrame - initialSourceStartFrame) / initial.speed,
          ),
        );
      const maxFrame = Math.max(
        durationFrames,
        startFrame + secondsToFrame(30, fps),
      );
      if (initial.mode === "move") {
        const snapped = snapBlockStartFrame(
          Math.max(0, startFrame + deltaFrames),
          endFrame - startFrame,
          frameSnapGuides,
          gestureThresholdFrames,
          ownerPrefix,
          0,
          maxFrame,
        );
        return {
          ...initial,
          currentTimelineStartSec: frameToSeconds(snapped.frame, fps),
        };
      }
      if (initial.mode === "trim-start") {
        const maxStartFrame = Math.max(startFrame, endFrame - 1);
        const snapped = snapEdgeFrame(
          startFrame + deltaFrames,
          frameSnapGuides,
          gestureThresholdFrames,
          ownerPrefix,
          0,
          maxStartFrame,
        );
        const nextStartFrame = clamp(
          sourceFrameAfterTimelineDelta(
            initialSourceStartFrame,
            snapped.frame - startFrame,
            initial.speed,
            fps,
            "start",
          ),
          0,
          initialSourceEndFrame - minimumSourceFrames,
        );
        const nextStartSec = frameToSeconds(nextStartFrame, fps);
        return {
          ...initial,
          currentStartSec: nextStartSec,
          currentEndSec: frameToSeconds(initialSourceEndFrame, fps),
          currentTimelineStartSec:
            initial.initialTimelineStartSec +
            (nextStartFrame - initialSourceStartFrame) / (fps * initial.speed),
        };
      }
      const maxEndFrame = secondsToFrame(
        initial.initialTimelineStartSec +
          (initial.sourceMaxEndSec - initial.initialStartSec) / initial.speed,
        fps,
      );
      const snapped = snapEdgeFrame(
        endFrame + deltaFrames,
        frameSnapGuides,
        gestureThresholdFrames,
        ownerPrefix,
        startFrame + 1,
        Math.max(startFrame + 1, maxEndFrame),
      );
      const nextEndFrame = clamp(
        sourceFrameAfterTimelineDelta(
          initialSourceEndFrame,
          snapped.frame - endFrame,
          initial.speed,
          fps,
          "end",
        ),
        initialSourceStartFrame + minimumSourceFrames,
        sourceBoundaryFrame(initial.sourceMaxEndSec, fps, "end"),
      );
      return {
        ...initial,
        currentStartSec: frameToSeconds(initialSourceStartFrame, fps),
        currentEndSec: frameToSeconds(nextEndFrame, fps),
      };
    },
    [
      durationFrames,
      fps,
      frameSnapGuides,
      gestureThresholdFrames,
      pxPerSec,
    ],
  );

  const computeCaptionPreview = useCallback(
    (initial: CaptionDragState, deltaX: number): CaptionDragState => {
      const deltaFrames = Math.round((deltaX / pxPerSec) * fps);
      const ownerPrefix = `caption:${initial.selection.overlayId}:`;
      const minimumSourceFrames = minimumDurationFrames(
        MIN_CLIP_SOURCE_DURATION_SEC,
        fps,
      );
      const initialSourceStartFrame = sourceBoundaryFrame(
        initial.initialStartSec,
        fps,
        "start",
      );
      const initialSourceEndFrame = Math.max(
        initialSourceStartFrame + minimumSourceFrames,
        sourceBoundaryFrame(
          initial.initialStartSec + initial.initialDurationSec,
          fps,
          "end",
        ),
      );
      const sourceDurationFrames =
        initialSourceEndFrame - initialSourceStartFrame;
      const blockStartSec =
        initial.clipTimelineStartSec +
        frameToSeconds(initialSourceStartFrame, fps) / initial.clipSpeed;
      const startFrame = secondsToFrame(blockStartSec, fps);
      const durationFrameCount = Math.max(
        1,
        Math.ceil(sourceDurationFrames / initial.clipSpeed),
      );
      const clipStartFrame = secondsToFrame(initial.clipTimelineStartSec, fps);
      const clipEndFrame = secondsToFrame(
        initial.clipTimelineStartSec +
          initial.clipSourceDurationSec / initial.clipSpeed,
        fps,
      );
      let nextTimelineStartFrame = startFrame;
      let nextTimelineEndFrame = startFrame + durationFrameCount;
      let nextSourceStartFrame = initialSourceStartFrame;
      let nextSourceEndFrame = initialSourceEndFrame;
      if (initial.mode === "move") {
        const snapped = snapBlockStartFrame(
          startFrame + deltaFrames,
          durationFrameCount,
          frameSnapGuides,
          gestureThresholdFrames,
          ownerPrefix,
          clipStartFrame,
          Math.max(clipStartFrame, clipEndFrame - durationFrameCount),
        );
        nextTimelineStartFrame = snapped.frame;
        nextTimelineEndFrame = snapped.frame + durationFrameCount;
        nextSourceStartFrame = clamp(
          sourceFrameAfterTimelineDelta(
            initialSourceStartFrame,
            nextTimelineStartFrame - startFrame,
            initial.clipSpeed,
            fps,
            "start",
          ),
          0,
          Math.max(
            0,
            sourceBoundaryFrame(initial.clipSourceDurationSec, fps, "end") -
              sourceDurationFrames,
          ),
        );
        nextSourceEndFrame = nextSourceStartFrame + sourceDurationFrames;
      } else if (initial.mode === "trim-start") {
        nextTimelineStartFrame = snapEdgeFrame(
          startFrame + deltaFrames,
          frameSnapGuides,
          gestureThresholdFrames,
          ownerPrefix,
          clipStartFrame,
          nextTimelineEndFrame - 1,
        ).frame;
        nextSourceStartFrame = clamp(
          sourceFrameAfterTimelineDelta(
            initialSourceStartFrame,
            nextTimelineStartFrame - startFrame,
            initial.clipSpeed,
            fps,
            "start",
          ),
          0,
          initialSourceEndFrame - minimumSourceFrames,
        );
      } else {
        nextTimelineEndFrame = snapEdgeFrame(
          nextTimelineEndFrame + deltaFrames,
          frameSnapGuides,
          gestureThresholdFrames,
          ownerPrefix,
          nextTimelineStartFrame + 1,
          clipEndFrame,
        ).frame;
        nextSourceEndFrame = clamp(
          sourceFrameAfterTimelineDelta(
            initialSourceEndFrame,
            nextTimelineEndFrame - (startFrame + durationFrameCount),
            initial.clipSpeed,
            fps,
            "end",
          ),
          initialSourceStartFrame + minimumSourceFrames,
          sourceBoundaryFrame(initial.clipSourceDurationSec, fps, "end"),
        );
      }
      return {
        ...initial,
        currentStartSec: frameToSeconds(nextSourceStartFrame, fps),
        currentDurationSec: frameToSeconds(
          nextSourceEndFrame - nextSourceStartFrame,
          fps,
        ),
      };
    },
    [fps, frameSnapGuides, gestureThresholdFrames, pxPerSec],
  );

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

  function startBrollDrag(
    event: ReactMouseEvent,
    clip: Clip,
    mode: "move" | "resize-end",
  ) {
    if (event.button !== 0 || brollEditBusy) return;
    event.preventDefault();
    event.stopPropagation();
    clearAllSelections();
    onSelectBrollClip?.(clip.id);
    if (mode === "move") captureDropLanes();
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
    mode: LaneDragState["mode"],
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

    if (mode === "move") captureDropLanes();
    const duration = clipTimelineDuration(clip);
    const assetDurationSec = assetDurationById.get(clip.asset_id) ?? null;
    const sourceMaxEndSec =
      assetDurationSec && assetDurationSec > 0
        ? assetDurationSec
        : clip.end_sec + 30;
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
    mode: CaptionDragState["mode"],
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
      initialStartSec:
        (block.startSec - block.clipTimelineStartSec) *
        Math.max(block.clipSpeed, 0.01),
      currentStartSec:
        (block.startSec - block.clipTimelineStartSec) *
        Math.max(block.clipSpeed, 0.01),
      initialDurationSec: block.durationSec * Math.max(block.clipSpeed, 0.01),
      currentDurationSec: block.durationSec * Math.max(block.clipSpeed, 0.01),
    });
  }

  function pointerData(event: ReactPointerEvent) {
    return {
      type: event.type as
        | "pointerdown"
        | "pointermove"
        | "pointerup"
        | "pointercancel",
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      button: event.button,
      isPrimary: event.isPrimary,
    };
  }

  function activateV2Gesture(
    event: ReactPointerEvent,
    options: Parameters<typeof createGestureController>[0],
    clockPreview?: TimelineClockPreview,
  ): boolean {
    if (
      !canClaimGesturePointer(
        activeV2GestureRef.current !== null,
        pointerData(event),
      )
    ) {
      return false;
    }
    const controller = createGestureController(options);
    activeV2GestureRef.current = { controller, clockPreview };
    controller.pointerDown(pointerData(event), event.currentTarget);
    return controller.isActive();
  }

  function finishV2Preview() {
    setLaneDragState(null);
    setBrollDragState(null);
    setCaptionDragState(null);
    setDropTarget(null);
    activeV2GestureRef.current = null;
  }

  function publishV2Preview(kind: string, expectedSec: number) {
    window.dispatchEvent(
      new CustomEvent("timeline:v2-preview", {
        detail: { kind, expectedSec },
      }),
    );
  }

  function setDirectDropTarget(target: TimelineDropTarget | null) {
    const container = containerRef.current;
    if (!container) return;
    container
      .querySelectorAll<HTMLElement>("[data-drop-lane-id].dropTarget")
      .forEach((element) => element.classList.remove("dropTarget"));
    if (target) {
      container
        .querySelector<HTMLElement>(
          `[data-drop-lane-id="${CSS.escape(target.laneId)}"]`,
        )
        ?.classList.add("dropTarget");
    }
  }

  function applyDirectBlockPreview(
    element: HTMLElement,
    leftPx: number,
    widthPx: number,
    kind: string,
  ) {
    element.classList.add("dragging");
    element.style.left = `${leftPx}px`;
    element.style.width = `${Math.max(widthPx, 4)}px`;
    publishV2Preview(kind, leftPx / pxPerSec);
  }

  function startBrollDragV2(
    event: ReactPointerEvent,
    clip: Clip,
    mode: "move" | "resize-end",
  ) {
    if (
      event.button !== 0 ||
      brollEditBusy ||
      isInteractiveTarget(event.target) ||
      !canClaimGesturePointer(
        activeV2GestureRef.current !== null,
        pointerData(event),
      )
    ) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    clearAllSelections();
    onSelectBrollClip?.(clip.id);
    if (mode === "move") captureDropLanes();
    const duration = clipTimelineDuration(clip);
    const previewElement = (
      event.currentTarget as HTMLElement
    ).closest<HTMLElement>(".brollBlock");
    if (!previewElement) return;
    const originalStyle = previewElement.getAttribute("style");
    const clearDirectPreview = (restoreStyle: boolean) => {
      if (restoreStyle) {
        if (originalStyle === null) previewElement.removeAttribute("style");
        else previewElement.setAttribute("style", originalStyle);
      }
      previewElement.classList.remove("dragging");
      setDirectDropTarget(null);
      finishV2Preview();
    };
    const initial: BrollDragState = {
      clipId: clip.id,
      mode,
      startClientX: event.clientX,
      initialStartSec: clip.timeline_start_sec,
      initialDurationSec: duration,
      currentStartSec: clip.timeline_start_sec,
      currentDurationSec: duration,
    };
    activateV2Gesture(event, {
      thresholdPx: 3,
      onPreview(update) {
        const preview = computeBrollPreview(initial, update.deltaX);
        applyDirectBlockPreview(
          previewElement,
          preview.currentStartSec * pxPerSec,
          preview.currentDurationSec * pxPerSec,
          `broll-${mode}`,
        );
        if (mode === "move") {
          setDirectDropTarget(resolveDropTarget(update.clientY, BROLL_LANE_ID));
        }
      },
      onCommit(update) {
        const preview = computeBrollPreview(initial, update.deltaX);
        const target =
          mode === "move"
            ? resolveDropTarget(update.clientY, BROLL_LANE_ID)
            : null;
        suppressClickRef.current = true;
        clearDirectPreview(false);
        if (mode === "move") {
          const initialFrame = secondsToFrame(initial.initialStartSec, fps);
          const finalFrame = secondsToFrame(preview.currentStartSec, fps);
          if (
            target &&
            onMoveBrollClipToLane &&
            shouldCommitFrameChange(initialFrame, finalFrame, true)
          ) {
            onMoveBrollClipToLane(clip.id, preview.currentStartSec, target);
          } else if (
            shouldCommitFrameChange(initialFrame, finalFrame, false)
          ) {
            onMoveBrollClip(clip.id, preview.currentStartSec);
          }
        } else if (
          shouldCommitFrameChange(
            secondsToFrame(initial.initialDurationSec, fps),
            secondsToFrame(preview.currentDurationSec, fps),
            false,
          )
        ) {
          onTrimBrollClip(clip.id, preview.currentDurationSec);
        }
      },
      onCancel: () => clearDirectPreview(true),
    });
  }

  function startLaneDragV2(
    event: ReactPointerEvent,
    lane: TimelineLane,
    clip: Clip,
    mode: LaneDragState["mode"],
  ) {
    if (
      event.button !== 0 ||
      isInteractiveTarget(event.target) ||
      !canClaimGesturePointer(
        activeV2GestureRef.current !== null,
        pointerData(event),
      )
    ) {
      return;
    }
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
    if (lockedLaneIds.has(lane.id)) return;
    if (mode === "move") captureDropLanes();
    const duration = clipTimelineDuration(clip);
    const previewElement = (
      event.currentTarget as HTMLElement
    ).closest<HTMLElement>(".timelineLaneClip");
    if (!previewElement) return;
    const originalStyle = previewElement.getAttribute("style");
    const clearDirectPreview = (restoreStyle: boolean) => {
      if (restoreStyle) {
        if (originalStyle === null) previewElement.removeAttribute("style");
        else previewElement.setAttribute("style", originalStyle);
      }
      previewElement.classList.remove("dragging");
      setDirectDropTarget(null);
      finishV2Preview();
    };
    const assetDurationSec = assetDurationById.get(clip.asset_id) ?? null;
    const initial: LaneDragState = {
      selection,
      mode,
      startClientX: event.clientX,
      speed: Math.max(clip.speed, 0.01),
      sourceMaxEndSec:
        assetDurationSec && assetDurationSec > 0
          ? assetDurationSec
          : clip.end_sec + 30,
      initialTimelineStartSec: clip.timeline_start_sec,
      initialTimelineEndSec: clip.timeline_start_sec + duration,
      currentTimelineStartSec: clip.timeline_start_sec,
      initialStartSec: clip.start_sec,
      initialEndSec: clip.end_sec,
      currentStartSec: clip.start_sec,
      currentEndSec: clip.end_sec,
    };
    activateV2Gesture(event, {
      thresholdPx: 3,
      onPreview(update) {
        const preview = computeLanePreview(initial, update.deltaX);
        applyDirectBlockPreview(
          previewElement,
          preview.currentTimelineStartSec * pxPerSec,
          ((preview.currentEndSec - preview.currentStartSec) / preview.speed) *
            pxPerSec,
          `lane-${mode}`,
        );
        if (mode === "move") {
          setDirectDropTarget(resolveDropTarget(update.clientY, lane.id));
        }
      },
      onCommit(update) {
        const preview = computeLanePreview(initial, update.deltaX);
        const target =
          mode === "move" ? resolveDropTarget(update.clientY, lane.id) : null;
        suppressClickRef.current = true;
        clearDirectPreview(false);
        if (mode === "move") {
          if (
            shouldCommitFrameChange(
              secondsToFrame(initial.initialTimelineStartSec, fps),
              secondsToFrame(preview.currentTimelineStartSec, fps),
              target !== null,
            )
          ) {
            onMoveLaneClip(
              selection,
              preview.currentTimelineStartSec,
              target ?? undefined,
            );
          }
        } else if (
          shouldCommitFrameChange(
            sourceBoundaryFrame(initial.initialStartSec, fps, "start"),
            sourceBoundaryFrame(preview.currentStartSec, fps, "start"),
            false,
          ) ||
          shouldCommitFrameChange(
            sourceBoundaryFrame(initial.initialEndSec, fps, "end"),
            sourceBoundaryFrame(preview.currentEndSec, fps, "end"),
            false,
          )
        ) {
          onTrimLaneClip(selection, {
            startSec: preview.currentStartSec,
            endSec: preview.currentEndSec,
          });
        }
      },
      onCancel: () => clearDirectPreview(true),
    });
  }

  function startCaptionDragV2(
    event: ReactPointerEvent,
    block: TimelineCaptionBlock,
    mode: CaptionDragState["mode"],
  ) {
    if (
      event.button !== 0 ||
      isInteractiveTarget(event.target) ||
      !canClaimGesturePointer(
        activeV2GestureRef.current !== null,
        pointerData(event),
      )
    ) {
      return;
    }
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
    const previewElement = (
      event.currentTarget as HTMLElement
    ).closest<HTMLElement>(".captionBlock");
    if (!previewElement) return;
    const originalStyle = previewElement.getAttribute("style");
    const clearDirectPreview = (restoreStyle: boolean) => {
      if (restoreStyle) {
        if (originalStyle === null) previewElement.removeAttribute("style");
        else previewElement.setAttribute("style", originalStyle);
      }
      previewElement.classList.remove("dragging");
      finishV2Preview();
    };
    const speed = Math.max(block.clipSpeed, 0.01);
    const initial: CaptionDragState = {
      selection,
      mode,
      startClientX: event.clientX,
      clipTimelineStartSec: block.clipTimelineStartSec,
      clipSourceDurationSec: block.clipSourceDurationSec,
      clipSpeed: speed,
      initialStartSec: (block.startSec - block.clipTimelineStartSec) * speed,
      currentStartSec: (block.startSec - block.clipTimelineStartSec) * speed,
      initialDurationSec: block.durationSec * speed,
      currentDurationSec: block.durationSec * speed,
    };
    activateV2Gesture(event, {
      thresholdPx: 3,
      onPreview(update) {
        const preview = computeCaptionPreview(initial, update.deltaX);
        const timelineStart =
          initial.clipTimelineStartSec +
          preview.currentStartSec / preview.clipSpeed;
        const timelineDuration = preview.currentDurationSec / preview.clipSpeed;
        applyDirectBlockPreview(
          previewElement,
          timelineStart * pxPerSec,
          timelineDuration * pxPerSec,
          `caption-${mode}`,
        );
      },
      onCommit(update) {
        const preview = computeCaptionPreview(initial, update.deltaX);
        suppressClickRef.current = true;
        clearDirectPreview(false);
        if (mode === "move") {
          if (
            shouldCommitFrameChange(
              sourceBoundaryFrame(initial.initialStartSec, fps, "start"),
              sourceBoundaryFrame(preview.currentStartSec, fps, "start"),
              false,
            )
          ) {
            onMoveCaptionBlock?.(selection, preview.currentStartSec);
          }
        } else if (
          shouldCommitFrameChange(
            sourceBoundaryFrame(initial.initialStartSec, fps, "start"),
            sourceBoundaryFrame(preview.currentStartSec, fps, "start"),
            false,
          ) ||
          shouldCommitFrameChange(
            sourceBoundaryFrame(
              initial.initialStartSec + initial.initialDurationSec,
              fps,
              "end",
            ),
            sourceBoundaryFrame(
              preview.currentStartSec + preview.currentDurationSec,
              fps,
              "end",
            ),
            false,
          )
        ) {
          onTrimCaptionBlock?.(
            selection,
            preview.currentStartSec,
            preview.currentDurationSec,
          );
        }
      },
      onCancel: () => clearDirectPreview(true),
    });
  }

  function startScrubV2(event: ReactPointerEvent) {
    if (
      event.button !== 0 ||
      event.altKey ||
      event.shiftKey ||
      isInteractiveTarget(event.target) ||
      !canClaimGesturePointer(
        activeV2GestureRef.current !== null,
        pointerData(event),
      )
    ) {
      return;
    }
    setContextMenu(null);
    // Keep clip selection during scrub so Split (S) still works after
    // positioning the playhead inside the selected clip.
    onSelectBrollClip?.(null);
    onSelectCaptionBlock?.(null);
    const initialSnapshot = activeTimelineClock.getSnapshot();
    const clockPreview = activeTimelineClock.beginPreview();
    if (!clockPreview) return;
    const preview = (clientX: number) => {
      const frame = secondsToFrame(secFromClientX(clientX), fps);
      const seconds = frameToSeconds(frame, fps);
      clockPreview.setTime(seconds);
      return seconds;
    };
    preview(event.clientX);
    const activated = activateV2Gesture(
      event,
      {
        thresholdPx: 0,
        onPreview(update) {
          publishV2Preview("scrub", preview(update.clientX));
        },
        onCommit(update) {
          const frame = secondsToFrame(secFromClientX(update.clientX), fps);
          const seconds = frameToSeconds(frame, fps);
          const initialFrame = secondsToFrame(initialSnapshot, fps);
          clockPreview.commit(
            frame === initialFrame ? initialSnapshot : seconds,
          );
          activeV2GestureRef.current = null;
          if (frame !== initialFrame) onSeek(seconds);
        },
        onCancel() {
          clockPreview.cancel();
          activeV2GestureRef.current = null;
        },
      },
      clockPreview,
    );
    if (!activated) clockPreview.cancel();
  }

  function handleV2PointerMove(event: ReactPointerEvent) {
    activeV2GestureRef.current?.controller.pointerMove(pointerData(event));
  }

  function handleV2PointerUp(event: ReactPointerEvent) {
    const active = activeV2GestureRef.current;
    active?.controller.pointerUp(pointerData(event));
    if (active && !active.controller.isActive()) {
      activeV2GestureRef.current = null;
    }
  }

  function handleV2PointerCancel(event: ReactPointerEvent) {
    activeV2GestureRef.current?.controller.pointerCancel(pointerData(event));
  }

  function consumeSuppressedClick(event: ReactMouseEvent): boolean {
    if (!suppressClickRef.current) return false;
    event.preventDefault();
    event.stopPropagation();
    suppressClickRef.current = false;
    return true;
  }

  useEffect(() => {
    if (!TIMELINE_CORE_V2) return;
    const onKeyDown = (event: KeyboardEvent) => {
      activeV2GestureRef.current?.controller.keyDown(event);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

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
    if (TIMELINE_CORE_V2) return;
    if (!brollDragState) return;

    function onMove(event: MouseEvent) {
      if (brollDragState?.mode === "move") {
        setDropTarget(resolveDropTarget(event.clientY, BROLL_LANE_ID));
      }
      setBrollDragState((prev) => {
        if (!prev) return prev;
        const deltaSec = (event.clientX - prev.startClientX) / pxPerSec;
        if (prev.mode === "move") {
          const rawStart = clamp(
            prev.initialStartSec + deltaSec,
            0,
            Math.max(durationSec, prev.initialStartSec + 30),
          );
          const snappedStart = clamp(
            resolveBlockSnap(
              rawStart,
              prev.initialDurationSec,
              `broll:${prev.clipId}:`,
            ),
            0,
            Math.max(durationSec, prev.initialStartSec + 30),
          );
          return { ...prev, currentStartSec: snappedStart };
        }
        const maxDuration = Math.max(
          durationSec - prev.initialStartSec,
          prev.initialDurationSec + 30,
          MIN_BROLL_DURATION_SEC,
        );
        const rawDuration = clamp(
          prev.initialDurationSec + deltaSec,
          MIN_BROLL_DURATION_SEC,
          maxDuration,
        );
        const rawEnd = prev.initialStartSec + rawDuration;
        const snappedEnd = clamp(
          resolveEdgeSnap(rawEnd, `broll:${prev.clipId}:`),
          prev.initialStartSec + MIN_BROLL_DURATION_SEC,
          prev.initialStartSec + maxDuration,
        );
        return {
          ...prev,
          currentDurationSec: Math.max(
            snappedEnd - prev.initialStartSec,
            MIN_BROLL_DURATION_SEC,
          ),
        };
      });
    }

    function onUp() {
      const target = dropTarget;
      setDropTarget(null);
      setBrollDragState((prev) => {
        if (!prev) return prev;
        if (prev.mode === "move") {
          if (target && onMoveBrollClipToLane) {
            onMoveBrollClipToLane(
              prev.clipId,
              Number(prev.currentStartSec.toFixed(3)),
              target,
            );
          } else if (
            Math.abs(prev.currentStartSec - prev.initialStartSec) >= 0.01
          ) {
            onMoveBrollClip(
              prev.clipId,
              Number(prev.currentStartSec.toFixed(3)),
            );
          }
        } else if (
          Math.abs(prev.currentDurationSec - prev.initialDurationSec) >= 0.01
        ) {
          onTrimBrollClip(
            prev.clipId,
            Number(prev.currentDurationSec.toFixed(3)),
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
  }, [
    brollDragState,
    dropTarget,
    durationSec,
    onMoveBrollClip,
    onMoveBrollClipToLane,
    onTrimBrollClip,
    pxPerSec,
    resolveBlockSnap,
    resolveDropTarget,
    resolveEdgeSnap,
  ]);

  useEffect(() => {
    if (TIMELINE_CORE_V2) return;
    if (!laneDragState) return;

    function onMove(event: MouseEvent) {
      if (laneDragState?.mode === "move") {
        setDropTarget(
          resolveDropTarget(event.clientY, laneDragState.selection.laneId),
        );
      }
      setLaneDragState((prev) => {
        if (!prev) return prev;
        const deltaTimelineSec = (event.clientX - prev.startClientX) / pxPerSec;
        const ownerPrefix = `lane:${prev.selection.clipId}:`;

        if (prev.mode === "move") {
          const duration =
            (prev.initialEndSec - prev.initialStartSec) / prev.speed;
          const rawStart = clamp(
            prev.initialTimelineStartSec + deltaTimelineSec,
            0,
            Math.max(durationSec, prev.initialTimelineStartSec + 30),
          );
          const snappedStart = clamp(
            resolveBlockSnap(rawStart, duration, ownerPrefix),
            0,
            Math.max(durationSec, prev.initialTimelineStartSec + 30),
          );
          return { ...prev, currentTimelineStartSec: snappedStart };
        }

        if (prev.mode === "trim-start") {
          const minTimelineStart =
            prev.initialTimelineEndSec -
            MIN_CLIP_SOURCE_DURATION_SEC / prev.speed;
          const rawTimelineStart = clamp(
            prev.initialTimelineStartSec + deltaTimelineSec,
            0,
            minTimelineStart,
          );
          const snappedTimelineStart = clamp(
            resolveEdgeSnap(rawTimelineStart, ownerPrefix),
            0,
            minTimelineStart,
          );
          const nextStartSec = clamp(
            prev.initialStartSec +
              (snappedTimelineStart - prev.initialTimelineStartSec) *
                prev.speed,
            0,
            prev.initialEndSec - MIN_CLIP_SOURCE_DURATION_SEC,
          );
          const nextTimelineStartSec =
            prev.initialTimelineStartSec +
            (nextStartSec - prev.initialStartSec) / prev.speed;
          return {
            ...prev,
            currentStartSec: nextStartSec,
            currentTimelineStartSec: nextTimelineStartSec,
          };
        }

        const maxTimelineEnd =
          prev.initialTimelineStartSec +
          (prev.sourceMaxEndSec - prev.initialStartSec) / prev.speed;
        const rawTimelineEnd = clamp(
          prev.initialTimelineEndSec + deltaTimelineSec,
          prev.initialTimelineStartSec +
            MIN_CLIP_SOURCE_DURATION_SEC / prev.speed,
          maxTimelineEnd,
        );
        const snappedTimelineEnd = clamp(
          resolveEdgeSnap(rawTimelineEnd, ownerPrefix),
          prev.initialTimelineStartSec +
            MIN_CLIP_SOURCE_DURATION_SEC / prev.speed,
          maxTimelineEnd,
        );
        const nextEndSec = clamp(
          prev.initialEndSec +
            (snappedTimelineEnd - prev.initialTimelineEndSec) * prev.speed,
          prev.initialStartSec + MIN_CLIP_SOURCE_DURATION_SEC,
          prev.sourceMaxEndSec,
        );
        return { ...prev, currentEndSec: nextEndSec };
      });
    }

    function onUp() {
      const target = dropTarget;
      setDropTarget(null);
      setLaneDragState((prev) => {
        if (!prev) return prev;
        if (prev.mode === "move") {
          if (
            target ||
            Math.abs(
              prev.currentTimelineStartSec - prev.initialTimelineStartSec,
            ) >= 0.01
          ) {
            onMoveLaneClip(
              prev.selection,
              Number(prev.currentTimelineStartSec.toFixed(3)),
              target ?? undefined,
            );
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
  }, [
    dropTarget,
    durationSec,
    laneDragState,
    onMoveLaneClip,
    onTrimLaneClip,
    pxPerSec,
    resolveBlockSnap,
    resolveDropTarget,
    resolveEdgeSnap,
  ]);

  useEffect(() => {
    if (TIMELINE_CORE_V2) return;
    if (!captionDragState) return;

    function onMove(event: MouseEvent) {
      setCaptionDragState((prev) => {
        if (!prev) return prev;
        const deltaTimelineSec = (event.clientX - prev.startClientX) / pxPerSec;
        const ownerPrefix = `caption:${prev.selection.overlayId}:`;

        if (prev.mode === "move") {
          const rawStart =
            prev.clipTimelineStartSec +
            prev.initialStartSec / prev.clipSpeed +
            deltaTimelineSec;
          const maxStart =
            prev.clipTimelineStartSec +
            Math.max(
              0,
              (prev.clipSourceDurationSec - prev.initialDurationSec) /
                prev.clipSpeed,
            );
          const snappedStart = clamp(
            resolveBlockSnap(
              rawStart,
              prev.initialDurationSec / prev.clipSpeed,
              ownerPrefix,
            ),
            prev.clipTimelineStartSec,
            maxStart,
          );
          const nextStartSec = clamp(
            (snappedStart - prev.clipTimelineStartSec) * prev.clipSpeed,
            0,
            Math.max(0, prev.clipSourceDurationSec - prev.initialDurationSec),
          );
          return { ...prev, currentStartSec: nextStartSec };
        }

        if (prev.mode === "trim-start") {
          const rawStart =
            prev.clipTimelineStartSec +
            prev.initialStartSec / prev.clipSpeed +
            deltaTimelineSec;
          const maxStart =
            prev.clipTimelineStartSec +
            (prev.initialStartSec +
              prev.initialDurationSec -
              MIN_CLIP_SOURCE_DURATION_SEC) /
              prev.clipSpeed;
          const snappedStart = clamp(
            resolveEdgeSnap(rawStart, ownerPrefix),
            prev.clipTimelineStartSec,
            maxStart,
          );
          const nextStartSec = clamp(
            (snappedStart - prev.clipTimelineStartSec) * prev.clipSpeed,
            0,
            prev.initialStartSec +
              prev.initialDurationSec -
              MIN_CLIP_SOURCE_DURATION_SEC,
          );
          const nextDurationSec = Math.max(
            prev.initialStartSec + prev.initialDurationSec - nextStartSec,
            MIN_CLIP_SOURCE_DURATION_SEC,
          );
          return {
            ...prev,
            currentStartSec: nextStartSec,
            currentDurationSec: nextDurationSec,
          };
        }

        const rawEnd =
          prev.clipTimelineStartSec +
          (prev.initialStartSec + prev.initialDurationSec) / prev.clipSpeed +
          deltaTimelineSec;
        const minEnd =
          prev.clipTimelineStartSec +
          (prev.initialStartSec + MIN_CLIP_SOURCE_DURATION_SEC) /
            prev.clipSpeed;
        const maxEnd =
          prev.clipTimelineStartSec +
          prev.clipSourceDurationSec / prev.clipSpeed;
        const snappedEnd = clamp(
          resolveEdgeSnap(rawEnd, ownerPrefix),
          minEnd,
          maxEnd,
        );
        const nextEndSourceSec = clamp(
          (snappedEnd - prev.clipTimelineStartSec) * prev.clipSpeed,
          prev.initialStartSec + MIN_CLIP_SOURCE_DURATION_SEC,
          prev.clipSourceDurationSec,
        );
        return {
          ...prev,
          currentDurationSec: Math.max(
            nextEndSourceSec - prev.initialStartSec,
            MIN_CLIP_SOURCE_DURATION_SEC,
          ),
        };
      });
    }

    function onUp() {
      setCaptionDragState((prev) => {
        if (!prev) return prev;
        if (prev.mode === "move") {
          if (Math.abs(prev.currentStartSec - prev.initialStartSec) >= 0.01) {
            onMoveCaptionBlock?.(
              prev.selection,
              Number(prev.currentStartSec.toFixed(3)),
            );
          }
        } else if (
          Math.abs(prev.currentStartSec - prev.initialStartSec) >= 0.01 ||
          Math.abs(prev.currentDurationSec - prev.initialDurationSec) >= 0.01
        ) {
          onTrimCaptionBlock?.(
            prev.selection,
            Number(prev.currentStartSec.toFixed(3)),
            Number(prev.currentDurationSec.toFixed(3)),
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
  }, [
    captionDragState,
    onMoveCaptionBlock,
    onTrimCaptionBlock,
    pxPerSec,
    resolveBlockSnap,
    resolveEdgeSnap,
  ]);

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

  const setZoomAt = useCallback(
    (requested: number, clientX?: number) => {
      if (zoomAnimationFrameRef.current !== null) {
        window.cancelAnimationFrame(zoomAnimationFrameRef.current);
      }
      zoomAnimationFrameRef.current = window.requestAnimationFrame(() => {
        zoomAnimationFrameRef.current = null;
        const container = containerRef.current;
        const previous = pxPerSecRef.current;
        const next = clamp(requested, MIN_PX_PER_SEC, MAX_PX_PER_SEC);
        if (!container || next === previous) return;
        const rect = container.getBoundingClientRect();
        const anchorClientX = clientX ?? rect.left + rect.width / 2;
        const nextWidth = Math.max(durationSec * next, 200);
        const maxScrollLeft = Math.max(0, nextWidth - container.clientWidth);
        const result = zoomViewportAtCursor({
          scrollLeft: clamp(
            container.scrollLeft,
            0,
            Math.max(0, container.scrollWidth - container.clientWidth),
          ),
          viewportLeft: rect.left + TRACK_LEFT_MARGIN,
          cursorClientX: anchorClientX,
          oldPixelsPerFrame: previous / fps,
          newPixelsPerFrame: next / fps,
          maxScrollLeft,
        });
        // Expand/shrink scrollable width before applying anchor scroll so the
        // browser does not clamp against the previous canvas size.
        const canvas =
          container.querySelector<HTMLElement>(".timelineCanvas");
        if (canvas) canvas.style.width = `${nextWidth}px`;
        pxPerSecRef.current = next;
        playheadAutoScrollEnabledRef.current = false;
        pendingZoomScrollRef.current = result.scrollLeft;
        container.scrollLeft = result.scrollLeft;
        setPxPerSec(next);
        window.requestAnimationFrame(() => {
          const pending = pendingZoomScrollRef.current;
          if (pending !== null) {
            container.scrollLeft = pending;
            pendingZoomScrollRef.current = null;
          }
          playheadAutoScrollEnabledRef.current = true;
        });
      });
    },
    [durationSec, fps],
  );

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    function onWheel(event: WheelEvent) {
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        const container = containerRef.current;
        const delta = event.deltaY > 0 ? 0.85 : 1.18;
        if (TIMELINE_CORE_V2) {
          setZoomAt(
            clamp(
              pxPerSecRef.current * delta,
              MIN_PX_PER_SEC,
              MAX_PX_PER_SEC,
            ),
            event.clientX,
          );
          return;
        }
        setPxPerSec((prev) => {
          const next = Math.max(
            MIN_PX_PER_SEC,
            Math.min(MAX_PX_PER_SEC, prev * delta),
          );
          if (container && next !== prev && !TIMELINE_CORE_V2) {
            // Keep the time under the cursor fixed while zooming.
            const rect = container.getBoundingClientRect();
            const viewportX = event.clientX - rect.left;
            const anchorSec =
              (viewportX + container.scrollLeft - TRACK_LEFT_MARGIN) / prev;
            window.requestAnimationFrame(() => {
              container.scrollLeft =
                anchorSec * next + TRACK_LEFT_MARGIN - viewportX;
            });
          }
          return next;
        });
      }
    }
    element.addEventListener("wheel", onWheel, { passive: false });
    return () => element.removeEventListener("wheel", onWheel);
  }, [setZoomAt]);

  const selectedCount = selectedWordIds.size;
  const selectedHasDeleted = useMemo(() => {
    for (const id of selectedWordIds) {
      if (deletedWordIds.has(id)) return true;
    }
    return false;
  }, [selectedWordIds, deletedWordIds]);

  const sectionStyle = useMemo(
    () =>
      ({ "--timeline-rail-width": `${TRACK_LEFT_MARGIN}px` }) as CSSProperties,
    [],
  );

  return (
    <section className="timeline card" style={sectionStyle}>
      <div className="timelineHeader">
        <div className="timelineHeaderCopy">
          <h3>Timeline</h3>
          <span className="tlHint">
            Drag clips to move (up/down to change track), drag edges to trim.
            Split: select clip → scrub playhead inside it → press <kbd>S</kbd>.
            <kbd>Ctrl</kbd>+<kbd>C</kbd>/<kbd>V</kbd> copy/paste,{" "}
            <kbd>Ctrl</kbd>+<kbd>D</kbd> duplicate, <kbd>Delete</kbd> remove.
          </span>
          {speakerLegend.length > 1 && (
            <div className="timelineSpeakerLegend" aria-label="Speaker legend">
              {speakerLegend.map((entry) => (
                <span
                  key={entry.speakerId}
                  className={`timelineSpeakerLegendItem ${
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
            <button
              className="zoomBtn"
              onClick={() =>
                TIMELINE_CORE_V2
                  ? setZoomAt(pxPerSec * 0.7)
                  : setPxPerSec((prev) =>
                      Math.max(MIN_PX_PER_SEC, prev * 0.7),
                    )
              }
              title="Zoom out"
              aria-label="Zoom out"
            >
              <Minus size={14} aria-hidden="true" />
            </button>
            <input
              type="range"
              min={MIN_PX_PER_SEC}
              max={MAX_PX_PER_SEC}
              step={1}
              value={pxPerSec}
              onChange={(event) =>
                TIMELINE_CORE_V2
                  ? setZoomAt(Number(event.target.value))
                  : setPxPerSec(Number(event.target.value))
              }
              className="zoomSlider"
            />
            <button
              className="zoomBtn"
              onClick={() =>
                TIMELINE_CORE_V2
                  ? setZoomAt(pxPerSec * 1.4)
                  : setPxPerSec((prev) =>
                      Math.min(MAX_PX_PER_SEC, prev * 1.4),
                    )
              }
              title="Zoom in"
              aria-label="Zoom in"
            >
              <Plus size={14} aria-hidden="true" />
            </button>
            <span className="zoomLabel">{Math.round(pxPerSec)}px/s</span>
          </div>
        </div>
      </div>

      <div
        className="timelineScroll"
        ref={containerRef}
        onMouseDown={(event) => {
          if (!TIMELINE_CORE_V2 || event.altKey || event.shiftKey) {
            handleMouseDown(event);
          }
        }}
        onMouseMove={(event) => {
          if (!TIMELINE_CORE_V2 || dragMode === "range") {
            handleMouseMove(event);
          }
        }}
        onPointerDown={(event) => {
          if (TIMELINE_CORE_V2) startScrubV2(event);
        }}
        onPointerMove={(event) => {
          if (TIMELINE_CORE_V2) handleV2PointerMove(event);
        }}
        onPointerUp={(event) => {
          if (TIMELINE_CORE_V2) handleV2PointerUp(event);
        }}
        onPointerCancel={(event) => {
          if (TIMELINE_CORE_V2) handleV2PointerCancel(event);
        }}
        onContextMenu={handleContextMenu}
      >
        <div className="timelineCanvas" style={{ width: totalWidth }}>
          <div className="timeRuler">
            {ticks.map((tick) => (
              <div
                key={`${tick.sec}-${tick.major ? "major" : "minor"}`}
                className={`tick ${tick.major ? "major" : ""}`}
                style={{ left: tick.x }}
              >
                {tick.major && <span className="tickLabel">{tick.label}</span>}
              </div>
            ))}
          </div>

          {laneBlocks.map((lane) => (
            <div
              key={lane.id}
              className={`timelineLane ${lane.kind} ${lane.isLocked ? "locked" : ""} ${
                dropTarget?.laneId === lane.id ? "dropTarget" : ""
              }`}
              data-drop-lane-id={lane.id}
              data-drop-lane-kind={lane.kind}
              data-drop-lane-label={lane.label}
              data-drop-lane-locked={lane.isLocked ? "1" : "0"}
            >
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
                    {lane.mute ? (
                      <VolumeX size={12} strokeWidth={2} aria-hidden="true" />
                    ) : (
                      <Volume2 size={12} strokeWidth={2} aria-hidden="true" />
                    )}
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
                    {lane.isLocked ? (
                      <Lock size={12} strokeWidth={2} aria-hidden="true" />
                    ) : (
                      <LockOpen size={12} strokeWidth={2} aria-hidden="true" />
                    )}
                  </button>
                </div>
              </div>

              {lane.blocks.length === 0 && (
                <div className="laneEmpty">No clips</div>
              )}

              {lane.blocks.map(
                ({
                  clip,
                  x,
                  w,
                  duration,
                  timelineStartSec,
                  thumbSrc,
                  assetId,
                  thumbTimes,
                  clipName,
                  isSelected,
                  isDragging,
                }) => (
                  <div
                    key={clip.id}
                    className={[
                      "timelineLaneClip",
                      lane.kind,
                      isSelected ? "selected" : "",
                      isDragging ? "dragging" : "",
                      lane.isLocked ? "locked" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    style={{ left: x, width: w }}
                    onMouseDown={
                      TIMELINE_CORE_V2
                        ? undefined
                        : (event) => startLaneDrag(event, lane, clip, "move")
                    }
                    onPointerDown={
                      TIMELINE_CORE_V2
                        ? (event) => startLaneDragV2(event, lane, clip, "move")
                        : undefined
                    }
                    onClick={(event) => {
                      if (consumeSuppressedClick(event)) return;
                      event.stopPropagation();
                      const selection: TimelineLaneClipSelection = {
                        clipId: clip.id,
                        laneId: lane.id,
                        laneLabel: lane.label,
                        laneKind: lane.kind,
                      };
                      // Keep playhead if already inside this clip so Split (S)
                      // works; only jump to start when outside.
                      const playheadSec = TIMELINE_CORE_V2
                        ? activeTimelineClock.getSnapshot()
                        : currentTimeSec;
                      const clipEndSec = timelineStartSec + duration;
                      const playheadInside =
                        playheadSec > timelineStartSec + 0.01 &&
                        playheadSec < clipEndSec - 0.01;
                      if (!playheadInside) onSeek(timelineStartSec);
                      onSelectLaneClip?.(selection);
                      onSelectBrollClip?.(null);
                      onSelectCaptionBlock?.(null);
                    }}
                    title={`${lane.label} · ${formatTimecode(timelineStartSec)} · ${formatDuration(duration)}`}
                  >
                    <div
                      className="laneClipHandle start"
                      onMouseDown={
                        TIMELINE_CORE_V2
                          ? undefined
                          : (event) =>
                              startLaneDrag(event, lane, clip, "trim-start")
                      }
                      onPointerDown={
                        TIMELINE_CORE_V2
                          ? (event) =>
                              startLaneDragV2(event, lane, clip, "trim-start")
                          : undefined
                      }
                      title="Trim clip in"
                    />
                    <div
                      className="laneClipHandle end"
                      onMouseDown={
                        TIMELINE_CORE_V2
                          ? undefined
                          : (event) =>
                              startLaneDrag(event, lane, clip, "trim-end")
                      }
                      onPointerDown={
                        TIMELINE_CORE_V2
                          ? (event) =>
                              startLaneDragV2(event, lane, clip, "trim-end")
                          : undefined
                      }
                      title="Trim clip out"
                    />
                    {lane.kind === "video" && thumbSrc && w > 36 && (
                      <div className="timelineLaneFilmstrip">
                        {thumbTimes.map((seekSec, index) => (
                          <LaneClipThumb
                            key={`${clip.id}-thumb-${index}`}
                            assetId={assetId}
                            seekSec={seekSec}
                          />
                        ))}
                      </div>
                    )}
                    {w > 44 && (
                      <div className="laneClipLabel">
                        {clipName && w > 80 && (
                          <span className="laneClipName">{clipName}</span>
                        )}
                        <span className="laneClipDuration">
                          {formatDuration(duration)}
                        </span>
                      </div>
                    )}
                  </div>
                ),
              )}
            </div>
          ))}

          <div className="captionTrack">
            <div className="trackRail">
              <span className="trackRailLabel">
                <Captions size={12} strokeWidth={1.9} aria-hidden="true" />
                <span>CC</span>
              </span>
            </div>

            {renderedCaptionBlocks.length === 0 && (
              <div className="laneEmpty">No caption blocks</div>
            )}

            {renderedCaptionBlocks.map((block) => (
              <div
                key={block.id}
                className={[
                  "captionBlock",
                  block.isSelected ? "selected" : "",
                  block.isDragging ? "dragging" : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                style={{ left: block.x, width: block.w }}
                onMouseDown={
                  TIMELINE_CORE_V2
                    ? undefined
                    : (event) => startCaptionDrag(event, block, "move")
                }
                onPointerDown={
                  TIMELINE_CORE_V2
                    ? (event) => startCaptionDragV2(event, block, "move")
                    : undefined
                }
                onClick={(event) => {
                  if (consumeSuppressedClick(event)) return;
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
                <div
                  className="captionBlockHandle start"
                  onMouseDown={
                    TIMELINE_CORE_V2
                      ? undefined
                      : (event) =>
                          startCaptionDrag(event, block, "trim-start")
                  }
                  onPointerDown={
                    TIMELINE_CORE_V2
                      ? (event) =>
                          startCaptionDragV2(event, block, "trim-start")
                      : undefined
                  }
                  title="Trim caption in"
                />
                {editingCaptionId === block.id ? (
                  <input
                    ref={captionEditInputRef}
                    className="captionBlockEditInput wordEditInput"
                    value={editingCaptionText}
                    onClick={(event) => event.stopPropagation()}
                    onMouseDown={(event) => event.stopPropagation()}
                    onChange={(event) =>
                      onCaptionTextChange?.(event.target.value)
                    }
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
                  <span className="captionBlockText">
                    {block.w > 40 ? trimInlineText(block.text, 44) : ""}
                  </span>
                )}
                <div
                  className="captionBlockHandle end"
                  onMouseDown={
                    TIMELINE_CORE_V2
                      ? undefined
                      : (event) => startCaptionDrag(event, block, "trim-end")
                  }
                  onPointerDown={
                    TIMELINE_CORE_V2
                      ? (event) =>
                          startCaptionDragV2(event, block, "trim-end")
                      : undefined
                  }
                  title="Trim caption out"
                />
              </div>
            ))}
          </div>

          <div className="waveformTrack">
            <div className="trackRail">
              <span className="trackRailLabel">WFM</span>
            </div>
            <svg
              className="waveformSvg"
              width={totalWidth}
              height={50}
              preserveAspectRatio="none"
            >
              {waveformRects}
            </svg>
            {deletedRegions.map((region, index) => (
              <div
                key={`del-wave-${index}`}
                className="deletedOverlay"
                style={{
                  left: region.startSec * pxPerSec,
                  width: (region.endSec - region.startSec) * pxPerSec,
                }}
              />
            ))}
          </div>

          <div
            className={`brollTrack ${
              dropTarget?.laneId === BROLL_LANE_ID ? "dropTarget" : ""
            }`}
            data-drop-lane-id={BROLL_LANE_ID}
            data-drop-lane-kind="overlay"
            data-drop-lane-label="B-roll"
            data-drop-lane-locked="0"
          >
            <div className="trackRail">
              <span className="trackRailLabel">B</span>
            </div>
            {brollBlocks.length === 0 && (
              <div className="brollEmpty">No overlay clips</div>
            )}
            {brollBlocks.map(
              ({
                clip,
                x,
                w,
                opacity,
                timelineStartSec,
                duration,
                isDragging,
              }) => {
                const isSelected = selectedBrollClipId === clip.id;
                return (
                  <div
                    key={clip.id}
                    className={[
                      "brollBlock",
                      isSelected ? "selected" : "",
                      isDragging ? "dragging" : "",
                      brollEditBusy ? "disabled" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    style={{
                      left: x,
                      width: w,
                      opacity: Math.max(0.28, Math.min(opacity, 1)),
                    }}
                    onMouseDown={
                      TIMELINE_CORE_V2
                        ? undefined
                        : (event) => startBrollDrag(event, clip, "move")
                    }
                    onPointerDown={
                      TIMELINE_CORE_V2
                        ? (event) => startBrollDragV2(event, clip, "move")
                        : undefined
                    }
                    onClick={(event) => {
                      if (consumeSuppressedClick(event)) return;
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
                      const currentOpacity =
                        brollOpacityDraftById[clip.id] ??
                        clamp(
                          typeof clip.broll_opacity === "number"
                            ? clip.broll_opacity
                            : 1,
                          0,
                          1,
                        );
                      const step = event.deltaY < 0 ? 0.04 : -0.04;
                      const nextOpacity = clamp(currentOpacity + step, 0, 1);
                      setBrollOpacityDraftById((prev) => ({
                        ...prev,
                        [clip.id]: nextOpacity,
                      }));
                      scheduleOpacityCommit(
                        clip.id,
                        Number(nextOpacity.toFixed(3)),
                      );
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
                      <RefreshCw
                        size={10}
                        strokeWidth={2.1}
                        aria-hidden="true"
                      />
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
                    <div
                      className="brollResizeHandle"
                      onMouseDown={
                        TIMELINE_CORE_V2
                          ? undefined
                          : (event) =>
                              startBrollDrag(event, clip, "resize-end")
                      }
                      onPointerDown={
                        TIMELINE_CORE_V2
                          ? (event) =>
                              startBrollDragV2(event, clip, "resize-end")
                          : undefined
                      }
                      title="Trim B-roll duration"
                    />
                  </div>
                );
              },
            )}
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
                  style={{
                    left: region.startSec * pxPerSec,
                    width: (region.endSec - region.startSec) * pxPerSec,
                  }}
                />
              ))}
              {wordBlocks.map(
                ({
                  word,
                  x,
                  w,
                  speakerSlot,
                  isDeleted,
                  isSelected,
                  isActive,
                }) => (
                  <div
                    key={word.id}
                    className={[
                      "tlWord",
                      isDeleted ? "deleted" : "",
                      isSelected ? "selected" : "",
                      isActive ? "active" : "",
                      speakerSlot === 0 ? "speakerA" : "",
                      speakerSlot === 1 ? "speakerB" : "",
                      speakerSlot !== null && speakerSlot >= 2
                        ? "speakerExtra"
                        : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
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
                      <Pencil
                        size={8}
                        className="tlWordEditHint"
                        aria-hidden="true"
                      />
                    )}
                  </div>
                ),
              )}
            </div>
          )}

          {rangeLeft !== null && rangeWidth !== null && rangeWidth > 2 && (
            <div
              className="rangeSelection"
              style={{ left: rangeLeft + TRACK_LEFT_MARGIN, width: rangeWidth }}
            >
              {rangeDuration !== null && rangeDuration > 0.1 && (
                <span className="rangeLabel">
                  {formatDuration(rangeDuration)}
                </span>
              )}
            </div>
          )}

          {snapIndicatorSec !== null && (
            <div
              className="snapIndicator"
              style={{ left: snapIndicatorSec * pxPerSec }}
            />
          )}

          <TimelinePlayhead
            currentTimeSec={currentTimeSec}
            pxPerSec={pxPerSec}
            timelineClock={TIMELINE_CORE_V2 ? activeTimelineClock : undefined}
            containerRef={containerRef}
            autoScrollEnabledRef={playheadAutoScrollEnabledRef}
          />
        </div>
      </div>

      {contextMenu && (
        <div
          className="tlContextMenu"
          style={{ left: contextMenu.x, top: contextMenu.y }}
        >
          {/* ── Word actions ── */}
          <button
            disabled={!selectedCount}
            onClick={() => {
              onDeleteSelected();
              setContextMenu(null);
            }}
          >
            <Trash2 size={14} strokeWidth={1.9} aria-hidden="true" />
            <span>Delete Selected Words ({selectedCount})</span>
          </button>
          <button
            disabled={!selectedHasDeleted}
            onClick={() => {
              onRestoreSelected();
              setContextMenu(null);
            }}
          >
            <RotateCcw size={14} strokeWidth={1.9} aria-hidden="true" />
            <span>Restore Selected Words</span>
          </button>
          <hr />
          {/* ── Clip actions ── */}
          {selectedLaneClipId &&
            (() => {
              const selectedBlock = laneBlocks
                .flatMap((lane) => lane.blocks)
                .find((b) => b.clip.id === selectedLaneClipId);
              const selectedLane = laneBlocks.find((lane) =>
                lane.blocks.some((b) => b.clip.id === selectedLaneClipId),
              );
              const sel =
                selectedBlock && selectedLane
                  ? ({
                      clipId: selectedBlock.clip.id,
                      laneId: selectedLane.id,
                      laneLabel: selectedLane.label,
                      laneKind: selectedLane.kind,
                    } as TimelineLaneClipSelection)
                  : null;
              if (!sel) return null;
              const clipStart = selectedBlock!.timelineStartSec;
              const clipEnd = clipStart + selectedBlock!.duration;
              const canSplit =
                currentTimeSec > clipStart + 0.01 &&
                currentTimeSec < clipEnd - 0.01;
              return (
                <>
                  <button
                    onClick={() => {
                      if (selectedBlock) onSeek(selectedBlock.timelineStartSec);
                      setContextMenu(null);
                    }}
                  >
                    <MapPin size={14} strokeWidth={1.9} aria-hidden="true" />
                    <span>Jump to Clip Start</span>
                  </button>
                  <button
                    disabled={!canSplit || !onSplitLaneClip}
                    onClick={() => {
                      if (sel && canSplit) {
                        onSplitLaneClip?.(sel);
                      }
                      setContextMenu(null);
                    }}
                    title={
                      canSplit
                        ? "Split clip at current playhead position"
                        : "Move playhead inside the clip to split"
                    }
                  >
                    <Scissors size={14} strokeWidth={1.9} aria-hidden="true" />
                    <span>Split Clip at Playhead</span>
                  </button>
                  <button
                    disabled={!onCopyLaneClip}
                    onClick={() => {
                      onCopyLaneClip?.(sel);
                      setContextMenu(null);
                    }}
                  >
                    <Copy size={14} strokeWidth={1.9} aria-hidden="true" />
                    <span>Copy Clip</span>
                  </button>
                  <button
                    disabled={!onDuplicateLaneClip}
                    onClick={() => {
                      onDuplicateLaneClip?.(sel);
                      setContextMenu(null);
                    }}
                  >
                    <CopyPlus size={14} strokeWidth={1.9} aria-hidden="true" />
                    <span>Duplicate Clip</span>
                  </button>
                  <button
                    className="danger"
                    onClick={() => {
                      onDeleteLaneClip?.(sel);
                      setContextMenu(null);
                    }}
                  >
                    <Trash2 size={14} strokeWidth={1.9} aria-hidden="true" />
                    <span>Delete Clip</span>
                  </button>
                </>
              );
            })()}
          {canPasteLaneClip && onPasteLaneClip && (
            <button
              onClick={() => {
                onPasteLaneClip(currentTimeSec);
                setContextMenu(null);
              }}
            >
              <ClipboardPaste size={14} strokeWidth={1.9} aria-hidden="true" />
              <span>Paste Clip at Playhead</span>
            </button>
          )}
          {!selectedLaneClipId && (
            <button
              onClick={() => {
                onSeek(currentTimeSec);
                setContextMenu(null);
              }}
            >
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
