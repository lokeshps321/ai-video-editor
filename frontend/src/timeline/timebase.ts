export const SUPPORTED_FPS = [24, 30, 60] as const;
export type TimelineFps = (typeof SUPPORTED_FPS)[number];
export type FrameRounding = "nearest" | "floor" | "ceil";
export type TrimEdge = "start" | "end";

const INTEGER_EPSILON = 1e-9;

function requireFinite(value: number, name: string): void {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${name} must be finite`);
  }
}

function requireInteger(value: number, name: string): void {
  requireFinite(value, name);
  if (!Number.isInteger(value)) {
    throw new RangeError(`${name} must be an integer`);
  }
}

function requireNonNegativeFrame(frame: number): void {
  requireInteger(frame, "frame");
  if (frame < 0) {
    throw new RangeError("frame must be non-negative");
  }
}

function stabilizeFrame(value: number): number {
  const nearest = Math.round(value);
  return Math.abs(value - nearest) <= INTEGER_EPSILON ? nearest : value;
}

export function validateFps(fps: number): asserts fps is TimelineFps {
  if (!SUPPORTED_FPS.some((supported) => supported === fps)) {
    throw new RangeError(`fps must be one of ${SUPPORTED_FPS.join(", ")}`);
  }
}

export function secondsToFrame(
  seconds: number,
  fps: number,
  rounding: FrameRounding = "nearest"
): number {
  requireFinite(seconds, "seconds");
  if (seconds < 0) {
    throw new RangeError("seconds must be non-negative");
  }
  validateFps(fps);

  const frame = stabilizeFrame(seconds * fps);
  if (rounding === "floor") return Math.floor(frame);
  if (rounding === "ceil") return Math.ceil(frame);
  return Math.round(frame);
}

export function frameToSeconds(frame: number, fps: number): number {
  requireNonNegativeFrame(frame);
  validateFps(fps);
  return frame / fps;
}

export function secondsToTrimFrame(
  seconds: number,
  fps: number,
  edge: TrimEdge
): number {
  return secondsToFrame(seconds, fps, edge === "start" ? "ceil" : "floor");
}

export function clampFrame(frame: number, minFrame: number, maxFrame: number): number {
  requireInteger(frame, "frame");
  requireInteger(minFrame, "minFrame");
  requireInteger(maxFrame, "maxFrame");
  if (minFrame < 0 || maxFrame < minFrame) {
    throw new RangeError("frame bounds must be non-negative and ordered");
  }
  return Math.min(maxFrame, Math.max(minFrame, frame));
}

export function stepFrame(
  frame: number,
  deltaFrames: number,
  minFrame: number,
  maxFrame: number
): number {
  requireInteger(deltaFrames, "deltaFrames");
  return clampFrame(frame + deltaFrames, minFrame, maxFrame);
}
