import { findSnap, type SnapGuide } from "./snapping";
import {
  clampFrame,
  secondsToFrame,
  secondsToTrimFrame,
  stepFrame,
  type TrimEdge,
} from "./timebase";

export interface PointerClaimInput {
  pointerId: number;
  button: number;
  isPrimary: boolean;
}

export function canClaimGesturePointer(
  hasActiveOwner: boolean,
  input: PointerClaimInput,
): boolean {
  return (
    !hasActiveOwner &&
    input.isPrimary &&
    input.button === 0 &&
    Number.isInteger(input.pointerId)
  );
}

export function minimumDurationFrames(seconds: number, fps: number): number {
  return Math.max(1, secondsToFrame(seconds, fps, "ceil"));
}

export function sourceBoundaryFrame(
  seconds: number,
  fps: number,
  edge: TrimEdge,
): number {
  return secondsToTrimFrame(seconds, fps, edge);
}

export function sourceFrameAfterTimelineDelta(
  initialSourceFrame: number,
  deltaTimelineFrames: number,
  speed: number,
  fps: number,
  edge: TrimEdge,
): number {
  if (
    !Number.isInteger(initialSourceFrame) ||
    !Number.isInteger(deltaTimelineFrames) ||
    initialSourceFrame < 0 ||
    !Number.isFinite(speed) ||
    speed <= 0
  ) {
    throw new RangeError("source frame, timeline delta, and speed must be valid");
  }
  return sourceBoundaryFrame(
    (initialSourceFrame + deltaTimelineFrames * speed) / fps,
    fps,
    edge,
  );
}

export function shouldCommitFrameChange(
  initialFrame: number,
  finalFrame: number,
  changedLane: boolean,
): boolean {
  return changedLane || initialFrame !== finalFrame;
}

export function resolveCanonicalFps(
  timelineFps: number | null | undefined,
  projectFps: number | null | undefined,
): number {
  if (Number.isFinite(timelineFps) && (timelineFps ?? 0) > 0) {
    return timelineFps as number;
  }
  if (Number.isFinite(projectFps) && (projectFps ?? 0) > 0) {
    return projectFps as number;
  }
  return 30;
}

export interface SnappedFrame {
  frame: number;
  guideFrame: number | null;
}

export function snapEdgeFrame(
  rawFrame: number,
  guides: readonly SnapGuide[],
  thresholdFrames: number,
  ownerPrefix: string,
  minFrame: number,
  maxFrame: number,
): SnappedFrame {
  const frame = clampFrame(rawFrame, minFrame, maxFrame);
  const eligible = guides.filter((guide) => !guide.id.startsWith(ownerPrefix));
  const snap = findSnap(frame, eligible, thresholdFrames);
  return snap
    ? { frame: clampFrame(snap.frame, minFrame, maxFrame), guideFrame: snap.frame }
    : { frame, guideFrame: null };
}

export function snapBlockStartFrame(
  rawStartFrame: number,
  durationFrames: number,
  guides: readonly SnapGuide[],
  thresholdFrames: number,
  ownerPrefix: string,
  minFrame: number,
  maxFrame: number,
): SnappedFrame {
  const frame = clampFrame(rawStartFrame, minFrame, maxFrame);
  const candidates = guides
    .filter((guide) => !guide.id.startsWith(ownerPrefix))
    .flatMap((guide) => {
      const starts = [
        { frame: guide.frame, id: `${guide.id}:start`, priority: guide.priority },
        {
          frame: guide.frame - durationFrames,
          id: `${guide.id}:end`,
          priority: guide.priority,
        },
      ];
      return starts.filter((candidate) => candidate.frame >= 0);
    });
  const snap = findSnap(frame, candidates, thresholdFrames);
  if (!snap) return { frame, guideFrame: null };
  const guideFrame = snap.guide.id.endsWith(":end")
    ? snap.frame + durationFrames
    : snap.frame;
  return {
    frame: clampFrame(snap.frame, minFrame, maxFrame),
    guideFrame,
  };
}

export type TimelineKeyboardCommand =
  | { kind: "seek"; frame: number }
  | { kind: "nudge-selected-clip"; frame: number };

export type TimelineArrowHandling = "blocked" | "v2" | "legacy" | "ignore";

export interface TimelineArrowDecisionInput {
  key: string;
  timelineCoreV2: boolean;
  altKey: boolean;
  shiftKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
}

export function decideTimelineArrowHandling(
  input: TimelineArrowDecisionInput,
): TimelineArrowHandling {
  if (input.key !== "ArrowLeft" && input.key !== "ArrowRight") return "ignore";
  if (input.ctrlKey || input.metaKey) return "blocked";
  return input.timelineCoreV2 && !input.shiftKey ? "v2" : "legacy";
}

export interface TimelineKeyboardInput {
  key: string;
  altKey: boolean;
  shiftKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  currentFrame: number;
  durationFrames: number;
  selectedClipStartFrame: number | null;
}

export function compileTimelineKeyboardCommand(
  input: TimelineKeyboardInput,
): TimelineKeyboardCommand | null {
  if (
    input.shiftKey ||
    input.ctrlKey ||
    input.metaKey ||
    (input.key !== "ArrowLeft" && input.key !== "ArrowRight")
  ) {
    return null;
  }
  const delta = input.key === "ArrowLeft" ? -1 : 1;
  if (input.altKey && input.selectedClipStartFrame !== null) {
    return {
      kind: "nudge-selected-clip",
      frame: stepFrame(
        input.selectedClipStartFrame,
        delta,
        0,
        input.durationFrames,
      ),
    };
  }
  if (input.altKey) return null;
  return {
    kind: "seek",
    frame: stepFrame(input.currentFrame, delta, 0, input.durationFrames),
  };
}
