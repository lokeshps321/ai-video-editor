import {
  Profiler,
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ProfilerOnRenderCallback,
} from "react";
import { createRoot } from "react-dom/client";
import Timeline, {
  type TimelineCaptionSelection,
  type TimelineLane,
  type TimelineLaneClipSelection,
} from "../../src/components/Timeline";
import type { Clip } from "../../src/types";
import { compileTimelineKeyboardCommand } from "../../src/timeline/integration";
import {
  frameToSeconds,
  secondsToFrame,
} from "../../src/timeline/timebase";
import { createLongProjectTimelineFixture } from "../fixtures/longProjectTimeline";

type CommitSample = {
  frame: number;
  commitTimeMs: number;
  actualDurationMs: number;
  phase: string;
};

type FrameSample = {
  frame: number;
  timeMs: number;
  expectedSec: number;
  actualSec: number;
  deltaSec: number;
};

type FrameSamplingConfig = {
  mode: "scrub" | "drag";
  startClientX: number;
  fps: number;
  initialTimelineSec?: number;
};

type LongTaskSample = {
  durationMs: number;
  startMs: number;
};

type FramePublicationSample = {
  frame: number;
  timeMs: number;
  kind: string;
};

type MeasurementResult = {
  label: string;
  durationMs: number;
  commitCount: number;
  framesWithCommits: number;
  maxCommitsPerFrame: number;
  framesOverOneCommit: number;
  commits: CommitSample[];
  frameSamples: FrameSample[];
  seekCallbackCount: number;
  longTasks: LongTaskSample[];
  previewPublications: FramePublicationSample[];
  renderedMutations: FramePublicationSample[];
};

type BaselineApi = {
  startMeasurement: (label: string) => void;
  startFrameSampling: (config: FrameSamplingConfig) => void;
  stopMeasurement: () => MeasurementResult;
  getState: () => {
    currentTimeSec: number;
    pxPerSec: number;
    scrollLeft: number;
  };
  getCallbackCounts: () => Record<string, number>;
  resetCallbackCounts: () => void;
  setAuthoritativeDelay: (delayMs: number) => void;
  waitForAuthoritative: () => Promise<void>;
};

type BaselineWindow = Window & {
  timelineBaseline: BaselineApi;
};

const fixture = createLongProjectTimelineFixture();
const requestedFps = Number(
  new URLSearchParams(window.location.search).get("fps") ?? "30",
);
const harnessFps = [24, 30, 60].includes(requestedFps) ? requestedFps : 30;
let activeLabel = "";
let activeStartMs = 0;
let active = false;
let animationFrame = 0;
let commitSamples: CommitSample[] = [];
let frameSamples: FrameSample[] = [];
let frameSamplingConfig: FrameSamplingConfig | null = null;
let latestPointerX = 0;
let latestPublishedExpectedSec: number | null = null;
let seekCallbackTimes: number[] = [];
let longTaskSamples: LongTaskSample[] = [];
let previewPublicationSamples: FramePublicationSample[] = [];
let renderedMutationSamples: FramePublicationSample[] = [];
let authoritativeDelayMs = 0;
let authoritativeUpdates: Promise<void>[] = [];
let callbackCounts = {
  seek: 0,
  moveLane: 0,
  trimLane: 0,
  moveBroll: 0,
  trimBroll: 0,
  moveCaption: 0,
  trimCaption: 0,
  stateMutation: 0,
};

function applyAuthoritativeUpdate(update: () => void): void {
  const pending = new Promise<void>((resolve) => {
    window.setTimeout(() => {
      callbackCounts.stateMutation += 1;
      update();
      resolve();
    }, authoritativeDelayMs);
  });
  authoritativeUpdates.push(pending);
}

window.addEventListener("mousemove", (event) => {
  latestPointerX = event.clientX;
});
window.addEventListener("pointermove", (event) => {
  latestPointerX = event.clientX;
});
window.addEventListener("timeline:v2-preview", (event) => {
  if (!active) return;
  if (
    event instanceof CustomEvent &&
    Number.isFinite(event.detail?.expectedSec)
  ) {
    latestPublishedExpectedSec = event.detail.expectedSec;
  }
  previewPublicationSamples.push({
    frame: animationFrame,
    timeMs: Number(performance.now().toFixed(3)),
    kind:
      event instanceof CustomEvent && typeof event.detail?.kind === "string"
        ? event.detail.kind
        : "unknown",
  });
  window.setTimeout(() => sampleTimelineFrame(performance.now()), 0);
});
window.addEventListener("timeline:v2-react-commit", () => {
  if (!active) return;
  commitSamples.push({
    frame: animationFrame,
    commitTimeMs: Number(performance.now().toFixed(3)),
    actualDurationMs: 0,
    phase: "production-layout-effect",
  });
});

const renderedMutationObserver = new MutationObserver((entries) => {
  if (!active) return;
  for (const entry of entries) {
    const element = entry.target;
    if (
      element instanceof HTMLElement &&
      element.matches(
        ".timeline-playhead, .timelineLaneClip, .brollBlock, .captionBlock, .snapGuide",
      )
    ) {
      renderedMutationSamples.push({
        frame: animationFrame,
        timeMs: Number(performance.now().toFixed(3)),
        kind: element.className,
      });
    }
  }
});
renderedMutationObserver.observe(document.documentElement, {
  attributes: true,
  attributeFilter: ["style"],
  subtree: true,
});

function pxPerSecFromDom(): number {
  const canvas = document.querySelector<HTMLElement>(".timelineCanvas");
  return Number.parseFloat(canvas?.style.width ?? "NaN") / 1_800;
}

function sampleTimelineFrame(timeMs: number): void {
  if (!active || !frameSamplingConfig) return;
  const pxPerSec = pxPerSecFromDom();
  let expectedSec = Number.NaN;
  let actualSec = Number.NaN;

  if (frameSamplingConfig.mode === "scrub") {
    const scroll = document.querySelector<HTMLElement>(".timelineScroll");
    const playhead = document.querySelector<HTMLElement>(".timeline-playhead");
    if (scroll && playhead) {
      const rect = scroll.getBoundingClientRect();
      const pointerSec = Math.max(
        0,
        Math.min(
          1_800,
          (latestPointerX - rect.left + scroll.scrollLeft - 96) / pxPerSec,
        ),
      );
      expectedSec =
        latestPublishedExpectedSec ??
        Math.round(pointerSec * frameSamplingConfig.fps) /
          frameSamplingConfig.fps;
      actualSec =
        Number.parseFloat(
          playhead.style.getPropertyValue("--timeline-playhead-x"),
        ) / pxPerSec;
    }
  } else {
    const draggingClip = document.querySelector<HTMLElement>(
      ".timelineLaneClip.dragging",
    );
    if (
      draggingClip &&
      typeof frameSamplingConfig.initialTimelineSec === "number"
    ) {
      const pointerSec =
        frameSamplingConfig.initialTimelineSec +
        (latestPointerX - frameSamplingConfig.startClientX) / pxPerSec;
      expectedSec =
        latestPublishedExpectedSec ??
        Math.round(pointerSec * frameSamplingConfig.fps) /
          frameSamplingConfig.fps;
      actualSec = Number.parseFloat(draggingClip.style.left) / pxPerSec;
    }
  }

  if (
    Number.isFinite(expectedSec) &&
    Number.isFinite(actualSec) &&
    Number.isFinite(timeMs)
  ) {
    frameSamples.push({
      frame: animationFrame,
      timeMs: Number(timeMs.toFixed(3)),
      expectedSec: Number(expectedSec.toFixed(6)),
      actualSec: Number(actualSec.toFixed(6)),
      deltaSec: Number((actualSec - expectedSec).toFixed(6)),
    });
  }
}

function frameLoop(timeMs: number) {
  animationFrame += 1;
  window.setTimeout(() => sampleTimelineFrame(timeMs), 0);
  window.requestAnimationFrame(frameLoop);
}
window.requestAnimationFrame(frameLoop);

const recordTimelineCommit: ProfilerOnRenderCallback = (
  _id,
  phase,
  actualDuration,
  _baseDuration,
  _startTime,
  commitTime,
) => {
  if (!active) return;
  commitSamples.push({
    frame: animationFrame,
    commitTimeMs: Number(commitTime.toFixed(3)),
    actualDurationMs: Number(actualDuration.toFixed(3)),
    phase,
  });
};

const longTaskObserver =
  "PerformanceObserver" in window
    ? new PerformanceObserver((entries) => {
        if (!active) return;
        for (const entry of entries.getEntries()) {
          longTaskSamples.push({
            durationMs: Number(entry.duration.toFixed(3)),
            startMs: Number(entry.startTime.toFixed(3)),
          });
        }
      })
    : null;

try {
  longTaskObserver?.observe({ type: "longtask", buffered: false });
} catch {
  // Chromium supports longtask; keep the harness usable in other browsers.
}

function currentTimelineState() {
  const scroll = document.querySelector<HTMLElement>(".timelineScroll");
  const zoomText =
    document.querySelector<HTMLElement>(".zoomLabel")?.textContent ?? "0";
  return {
    currentTimeSec: Number(
      document.querySelector<HTMLElement>("[data-current-time-sec]")?.dataset
        .currentTimeSec ?? 0,
    ),
    pxPerSec: Number.parseFloat(zoomText),
    scrollLeft: scroll?.scrollLeft ?? 0,
  };
}

(window as unknown as BaselineWindow).timelineBaseline = {
  startMeasurement(label) {
    activeLabel = label;
    activeStartMs = performance.now();
    commitSamples = [];
    frameSamples = [];
    frameSamplingConfig = null;
    seekCallbackTimes = [];
    longTaskSamples = [];
    previewPublicationSamples = [];
    renderedMutationSamples = [];
    latestPublishedExpectedSec = null;
    active = true;
  },
  startFrameSampling(config) {
    frameSamplingConfig = config;
    latestPointerX = config.startClientX;
  },
  stopMeasurement() {
    active = false;
    frameSamplingConfig = null;
    const commitsByFrame = new Map<number, number>();
    for (const sample of commitSamples) {
      commitsByFrame.set(
        sample.frame,
        (commitsByFrame.get(sample.frame) ?? 0) + 1,
      );
    }
    const commitsPerFrame = [...commitsByFrame.values()];
    return {
      label: activeLabel,
      durationMs: Number((performance.now() - activeStartMs).toFixed(3)),
      commitCount: commitSamples.length,
      framesWithCommits: commitsByFrame.size,
      maxCommitsPerFrame: Math.max(0, ...commitsPerFrame),
      framesOverOneCommit: commitsPerFrame.filter((count) => count > 1).length,
      commits: commitSamples,
      frameSamples,
      seekCallbackCount: seekCallbackTimes.length,
      longTasks: longTaskSamples,
      previewPublications: previewPublicationSamples,
      renderedMutations: renderedMutationSamples,
    };
  },
  getState: currentTimelineState,
  getCallbackCounts: () => ({ ...callbackCounts }),
  resetCallbackCounts: () => {
    callbackCounts = Object.fromEntries(
      Object.keys(callbackCounts).map((key) => [key, 0]),
    ) as typeof callbackCounts;
  },
  setAuthoritativeDelay(delayMs) {
    authoritativeDelayMs = Math.max(0, delayMs);
  },
  async waitForAuthoritative() {
    const pending = authoritativeUpdates;
    authoritativeUpdates = [];
    await Promise.all(pending);
  },
};

function replaceClip(
  lanes: TimelineLane[],
  selection: TimelineLaneClipSelection,
  update: (clip: Clip) => Clip,
): TimelineLane[] {
  return lanes.map((lane) =>
    lane.id !== selection.laneId
      ? lane
      : {
          ...lane,
          clips: lane.clips.map((clip) =>
            clip.id === selection.clipId ? update(clip) : clip,
          ),
        },
  );
}

function Harness() {
  const [currentTimeSec, setCurrentTimeSec] = useState(
    fixture.timelineProps.currentTimeSec,
  );
  const [timelineLanes, setTimelineLanes] = useState(
    fixture.timelineProps.timelineLanes,
  );
  const [overlayClips, setOverlayClips] = useState(
    fixture.timelineProps.overlayClips,
  );
  const [selectedLaneClipId, setSelectedLaneClipId] = useState<string | null>(
    null,
  );
  const [selectedBrollClipId, setSelectedBrollClipId] = useState<string | null>(
    null,
  );
  const [selectedCaptionId, setSelectedCaptionId] = useState<string | null>(
    null,
  );
  const [lockedLaneIds, setLockedLaneIds] = useState(
    fixture.timelineProps.lockedLaneIds,
  );

  const onSeek = useCallback((sec: number) => {
    callbackCounts.seek += 1;
    if (active) seekCallbackTimes.push(performance.now());
    setCurrentTimeSec(sec);
  }, []);

  const onMoveLaneClip = useCallback(
    (selection: TimelineLaneClipSelection, timelineStartSec: number) => {
      callbackCounts.moveLane += 1;
      applyAuthoritativeUpdate(() => {
        setTimelineLanes((lanes) =>
          replaceClip(lanes, selection, (clip) => ({
            ...clip,
            timeline_start_sec: timelineStartSec,
          })),
        );
      });
    },
    [],
  );

  const onTrimLaneClip = useCallback(
    (
      selection: TimelineLaneClipSelection,
      range: { startSec: number; endSec: number },
    ) => {
      callbackCounts.trimLane += 1;
      applyAuthoritativeUpdate(() => {
        setTimelineLanes((lanes) =>
          replaceClip(lanes, selection, (clip) => ({
            ...clip,
            start_sec: range.startSec,
            end_sec: range.endSec,
          })),
        );
      });
    },
    [],
  );

  const callbacks = useMemo(
    () => ({
      onSelectWord: () => undefined,
      onSelectWordsInRange: () => undefined,
      onDeleteSelected: () => undefined,
      onRestoreSelected: () => undefined,
      onMoveLaneClip,
      onTrimLaneClip,
      onMoveBrollClip: (clipId: string, timelineStartSec: number) => {
        callbackCounts.moveBroll += 1;
        setOverlayClips((clips) =>
          clips.map((clip) =>
            clip.id === clipId
              ? { ...clip, timeline_start_sec: timelineStartSec }
              : clip,
          ),
        );
      },
      onTrimBrollClip: (clipId: string, durationSec: number) => {
        callbackCounts.trimBroll += 1;
        setOverlayClips((clips) =>
          clips.map((clip) =>
            clip.id === clipId ? { ...clip, end_sec: durationSec } : clip,
          ),
        );
      },
      onMoveCaptionBlock: () => {
        callbackCounts.moveCaption += 1;
      },
      onTrimCaptionBlock: () => {
        callbackCounts.trimCaption += 1;
      },
      onSetBrollOpacity: (clipId: string, opacity: number) =>
        setOverlayClips((clips) =>
          clips.map((clip) =>
            clip.id === clipId ? { ...clip, broll_opacity: opacity } : clip,
          ),
        ),
      onDeleteBrollClip: (clipId: string) =>
        setOverlayClips((clips) => clips.filter((clip) => clip.id !== clipId)),
    }),
    [onMoveLaneClip, onTrimLaneClip],
  );

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const selected = timelineLanes
        .flatMap((lane) =>
          lane.clips.map((clip) => ({ lane, clip })),
        )
        .find(({ clip }) => clip.id === selectedLaneClipId);
      const command = compileTimelineKeyboardCommand({
        key: event.key,
        altKey: event.altKey,
        shiftKey: event.shiftKey,
        ctrlKey: event.ctrlKey,
        metaKey: event.metaKey,
        currentFrame: secondsToFrame(currentTimeSec, harnessFps),
        durationFrames: secondsToFrame(
          fixture.timelineProps.durationSec,
          harnessFps,
        ),
        selectedClipStartFrame: selected
          ? secondsToFrame(
              selected.clip.timeline_start_sec,
              harnessFps,
            )
          : null,
      });
      if (!command) return;
      event.preventDefault();
      if (command.kind === "seek") {
        onSeek(frameToSeconds(command.frame, harnessFps));
      } else if (selected) {
        onMoveLaneClip(
          {
            clipId: selected.clip.id,
            laneId: selected.lane.id,
            laneLabel: selected.lane.label,
            laneKind: selected.lane.kind,
          },
          frameToSeconds(command.frame, harnessFps),
        );
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    currentTimeSec,
    onMoveLaneClip,
    onSeek,
    selectedLaneClipId,
    timelineLanes,
  ]);

  return (
    <main
      data-current-time-sec={currentTimeSec.toFixed(6)}
      data-testid="timeline-baseline-harness"
    >
      <Profiler id="Timeline" onRender={recordTimelineCommit}>
        <Timeline
          {...fixture.timelineProps}
          fps={harnessFps}
          {...callbacks}
          currentTimeSec={currentTimeSec}
          timelineLanes={timelineLanes}
          overlayClips={overlayClips}
          selectedLaneClipId={selectedLaneClipId}
          selectedBrollClipId={selectedBrollClipId}
          selectedCaptionId={selectedCaptionId}
          lockedLaneIds={lockedLaneIds}
          onSeek={onSeek}
          onToggleLaneLock={(laneId) =>
            setLockedLaneIds((ids) => {
              const next = new Set(ids);
              if (next.has(laneId)) next.delete(laneId);
              else next.add(laneId);
              return next;
            })
          }
          onSelectLaneClip={(selection) =>
            setSelectedLaneClipId(selection?.clipId ?? null)
          }
          onSelectBrollClip={setSelectedBrollClipId}
          onSelectCaptionBlock={(selection: TimelineCaptionSelection | null) =>
            setSelectedCaptionId(selection?.overlayId ?? null)
          }
          brollEditBusy={false}
        />
      </Profiler>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<Harness />);
