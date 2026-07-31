export interface GesturePointerEvent {
  type: "pointerdown" | "pointermove" | "pointerup" | "pointercancel";
  pointerId: number;
  clientX: number;
  clientY: number;
  button: number;
  isPrimary: boolean;
}

export interface PointerCaptureTarget {
  setPointerCapture(pointerId: number): void;
  releasePointerCapture(pointerId: number): void;
  hasPointerCapture?(pointerId: number): boolean;
}

export interface AnimationFrameScheduler {
  request(callback: FrameRequestCallback): number;
  cancel(id: number): void;
}

export interface GestureUpdate {
  pointerId: number;
  startX: number;
  startY: number;
  clientX: number;
  clientY: number;
  deltaX: number;
  deltaY: number;
}

export interface GestureControllerOptions {
  thresholdPx: number;
  raf?: AnimationFrameScheduler;
  onPreview?: (update: GestureUpdate) => void;
  onCommit?: (update: GestureUpdate) => void;
  onCancel?: () => void;
}

export interface GestureController {
  pointerDown(event: GesturePointerEvent, target: PointerCaptureTarget): void;
  pointerMove(event: GesturePointerEvent): void;
  pointerUp(event: GesturePointerEvent): void;
  pointerCancel(event: GesturePointerEvent): void;
  keyDown(event: { key: string }): void;
  isActive(): boolean;
}

interface ActiveGesture {
  pointerId: number;
  startX: number;
  startY: number;
  target: PointerCaptureTarget;
  dragging: boolean;
  latest: GestureUpdate;
}

function browserRaf(): AnimationFrameScheduler {
  return {
    request: (callback) => globalThis.requestAnimationFrame(callback),
    cancel: (id) => globalThis.cancelAnimationFrame(id)
  };
}

function toUpdate(active: ActiveGesture, event: GesturePointerEvent): GestureUpdate {
  return {
    pointerId: active.pointerId,
    startX: active.startX,
    startY: active.startY,
    clientX: event.clientX,
    clientY: event.clientY,
    deltaX: event.clientX - active.startX,
    deltaY: event.clientY - active.startY
  };
}

export function createGestureController(
  options: GestureControllerOptions
): GestureController {
  if (!Number.isFinite(options.thresholdPx) || options.thresholdPx < 0) {
    throw new RangeError("thresholdPx must be finite and non-negative");
  }

  const raf = options.raf ?? browserRaf();
  let active: ActiveGesture | null = null;
  let previewFrame: number | null = null;

  const clearPreview = (): void => {
    if (previewFrame !== null) {
      raf.cancel(previewFrame);
      previewFrame = null;
    }
  };

  const releaseCapture = (gesture: ActiveGesture): void => {
    if (
      gesture.target.hasPointerCapture === undefined ||
      gesture.target.hasPointerCapture(gesture.pointerId)
    ) {
      gesture.target.releasePointerCapture(gesture.pointerId);
    }
  };

  const finishCancellation = (): void => {
    if (!active) return;
    clearPreview();
    releaseCapture(active);
    active = null;
    options.onCancel?.();
  };

  const schedulePreview = (): void => {
    if (previewFrame !== null) return;
    previewFrame = raf.request(() => {
      previewFrame = null;
      if (active?.dragging) {
        options.onPreview?.(active.latest);
      }
    });
  };

  return {
    pointerDown(event, target) {
      if (active || !event.isPrimary || event.button !== 0) return;
      const latest: GestureUpdate = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        clientX: event.clientX,
        clientY: event.clientY,
        deltaX: 0,
        deltaY: 0
      };
      active = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        target,
        dragging: false,
        latest
      };
      target.setPointerCapture(event.pointerId);
    },

    pointerMove(event) {
      if (!active || event.pointerId !== active.pointerId) return;
      active.latest = toUpdate(active, event);
      if (
        !active.dragging &&
        Math.hypot(active.latest.deltaX, active.latest.deltaY) >= options.thresholdPx
      ) {
        active.dragging = true;
      }
      if (active.dragging) schedulePreview();
    },

    pointerUp(event) {
      if (!active || event.pointerId !== active.pointerId) return;
      const completed = active;
      completed.latest = toUpdate(completed, event);
      if (
        !completed.dragging &&
        Math.hypot(completed.latest.deltaX, completed.latest.deltaY) >=
          options.thresholdPx
      ) {
        completed.dragging = true;
      }
      clearPreview();
      releaseCapture(completed);
      active = null;
      if (completed.dragging) options.onCommit?.(completed.latest);
    },

    pointerCancel(event) {
      if (active && event.pointerId === active.pointerId) finishCancellation();
    },

    keyDown(event) {
      if (event.key === "Escape") finishCancellation();
    },

    isActive() {
      return active !== null;
    }
  };
}
