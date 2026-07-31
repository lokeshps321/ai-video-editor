export interface CursorZoomInput {
  scrollLeft: number;
  viewportLeft: number;
  cursorClientX: number;
  oldPixelsPerFrame: number;
  newPixelsPerFrame: number;
  maxScrollLeft: number;
}

export interface CursorZoomResult {
  scrollLeft: number;
  anchorFrame: number;
}

export interface AutoScrollInput {
  pointerClientX: number;
  viewportLeft: number;
  viewportWidth: number;
  edgeSizePx: number;
  maxSpeedPxPerSecond: number;
  elapsedMs: number;
  scrollLeft: number;
  maxScrollLeft: number;
}

export interface AutoScrollResult {
  delta: number;
  scrollLeft: number;
}

function requireFinite(value: number, name: string): void {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${name} must be finite`);
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function zoomViewportAtCursor(input: CursorZoomInput): CursorZoomResult {
  Object.entries(input).forEach(([name, value]) => requireFinite(value, name));
  if (
    input.scrollLeft < 0 ||
    input.maxScrollLeft < 0 ||
    input.scrollLeft > input.maxScrollLeft ||
    input.oldPixelsPerFrame <= 0 ||
    input.newPixelsPerFrame <= 0
  ) {
    throw new RangeError("zoom scales and scroll bounds must be positive and valid");
  }

  const cursorOffset = input.cursorClientX - input.viewportLeft;
  const anchorFrame = (input.scrollLeft + cursorOffset) / input.oldPixelsPerFrame;
  const desiredScrollLeft =
    anchorFrame * input.newPixelsPerFrame - cursorOffset;

  return {
    scrollLeft: clamp(desiredScrollLeft, 0, input.maxScrollLeft),
    anchorFrame
  };
}

export function computeAutoScroll(input: AutoScrollInput): AutoScrollResult {
  Object.entries(input).forEach(([name, value]) => requireFinite(value, name));
  if (
    input.viewportWidth <= 0 ||
    input.edgeSizePx <= 0 ||
    input.edgeSizePx > input.viewportWidth / 2 ||
    input.maxSpeedPxPerSecond < 0 ||
    input.elapsedMs < 0 ||
    input.scrollLeft < 0 ||
    input.maxScrollLeft < 0 ||
    input.scrollLeft > input.maxScrollLeft
  ) {
    throw new RangeError("auto-scroll dimensions, timing, and bounds must be valid");
  }

  const localPointer = input.pointerClientX - input.viewportLeft;
  let intensity = 0;
  if (localPointer < input.edgeSizePx) {
    intensity = -clamp(
      (input.edgeSizePx - localPointer) / input.edgeSizePx,
      0,
      1
    );
  } else if (localPointer > input.viewportWidth - input.edgeSizePx) {
    intensity = clamp(
      (localPointer - (input.viewportWidth - input.edgeSizePx)) /
        input.edgeSizePx,
      0,
      1
    );
  }

  const requestedDelta =
    intensity * input.maxSpeedPxPerSecond * (input.elapsedMs / 1_000);
  const scrollLeft = clamp(
    input.scrollLeft + requestedDelta,
    0,
    input.maxScrollLeft
  );
  return { delta: scrollLeft - input.scrollLeft, scrollLeft };
}
