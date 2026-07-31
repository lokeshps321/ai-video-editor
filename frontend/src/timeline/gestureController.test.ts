import { describe, expect, it, vi } from "vitest";

import {
  createGestureController,
  type AnimationFrameScheduler,
  type GesturePointerEvent,
  type PointerCaptureTarget
} from "./gestureController";

function pointer(
  type: GesturePointerEvent["type"],
  clientX: number,
  clientY = 0,
  pointerId = 7
): GesturePointerEvent {
  return { type, clientX, clientY, pointerId, button: 0, isPrimary: true };
}

function createRaf(): AnimationFrameScheduler & { flush: () => void; pending: () => number } {
  let nextId = 1;
  const callbacks = new Map<number, FrameRequestCallback>();

  return {
    request(callback) {
      const id = nextId++;
      callbacks.set(id, callback);
      return id;
    },
    cancel(id) {
      callbacks.delete(id);
    },
    flush() {
      const queued = [...callbacks.values()];
      callbacks.clear();
      queued.forEach((callback) => callback(16));
    },
    pending() {
      return callbacks.size;
    }
  };
}

function createTarget(): PointerCaptureTarget & {
  captured: Set<number>;
  setPointerCapture: ReturnType<typeof vi.fn>;
  releasePointerCapture: ReturnType<typeof vi.fn>;
} {
  const captured = new Set<number>();
  return {
    captured,
    setPointerCapture: vi.fn((pointerId: number) => captured.add(pointerId)),
    releasePointerCapture: vi.fn((pointerId: number) => captured.delete(pointerId)),
    hasPointerCapture: (pointerId: number) => captured.has(pointerId)
  };
}

describe("pointer gesture controller", () => {
  it("captures the pointer and starts only after the drag threshold", () => {
    const raf = createRaf();
    const target = createTarget();
    const onPreview = vi.fn();
    const controller = createGestureController({ thresholdPx: 4, raf, onPreview });

    controller.pointerDown(pointer("pointerdown", 10), target);
    controller.pointerMove(pointer("pointermove", 13));
    raf.flush();

    expect(target.setPointerCapture).toHaveBeenCalledWith(7);
    expect(onPreview).not.toHaveBeenCalled();

    controller.pointerMove(pointer("pointermove", 14));
    raf.flush();
    expect(onPreview).toHaveBeenCalledWith(
      expect.objectContaining({ clientX: 14, deltaX: 4, pointerId: 7 })
    );
  });

  it("coalesces multiple moves into exactly one RAF preview using the latest pointer", () => {
    const raf = createRaf();
    const onPreview = vi.fn();
    const controller = createGestureController({ thresholdPx: 2, raf, onPreview });

    controller.pointerDown(pointer("pointerdown", 0), createTarget());
    controller.pointerMove(pointer("pointermove", 3));
    controller.pointerMove(pointer("pointermove", 8));
    controller.pointerMove(pointer("pointermove", 13));

    expect(raf.pending()).toBe(1);
    expect(onPreview).not.toHaveBeenCalled();
    raf.flush();
    expect(onPreview).toHaveBeenCalledTimes(1);
    expect(onPreview).toHaveBeenCalledWith(expect.objectContaining({ clientX: 13, deltaX: 13 }));
  });

  it("commits once only on a completed pointer release", () => {
    const raf = createRaf();
    const target = createTarget();
    const onCommit = vi.fn();
    const controller = createGestureController({ thresholdPx: 2, raf, onCommit });

    controller.pointerDown(pointer("pointerdown", 10), target);
    controller.pointerMove(pointer("pointermove", 20));
    raf.flush();
    controller.pointerUp(pointer("pointerup", 22));
    controller.pointerUp(pointer("pointerup", 25));

    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(onCommit).toHaveBeenCalledWith(expect.objectContaining({ clientX: 22, deltaX: 12 }));
    expect(target.releasePointerCapture).toHaveBeenCalledWith(7);
  });

  it("does not commit a click that never crosses the threshold", () => {
    const onCommit = vi.fn();
    const controller = createGestureController({
      thresholdPx: 5,
      raf: createRaf(),
      onCommit
    });

    controller.pointerDown(pointer("pointerdown", 10), createTarget());
    controller.pointerUp(pointer("pointerup", 13));

    expect(onCommit).not.toHaveBeenCalled();
  });

  it.each(["escape", "pointercancel"] as const)(
    "cancels pending preview and never commits after %s",
    (reason) => {
      const raf = createRaf();
      const target = createTarget();
      const onPreview = vi.fn();
      const onCommit = vi.fn();
      const onCancel = vi.fn();
      const controller = createGestureController({
        thresholdPx: 2,
        raf,
        onPreview,
        onCommit,
        onCancel
      });

      controller.pointerDown(pointer("pointerdown", 0), target);
      controller.pointerMove(pointer("pointermove", 10));
      if (reason === "escape") {
        controller.keyDown({ key: "Escape" });
      } else {
        controller.pointerCancel(pointer("pointercancel", 10));
      }
      raf.flush();
      controller.pointerUp(pointer("pointerup", 10));

      expect(onPreview).not.toHaveBeenCalled();
      expect(onCommit).not.toHaveBeenCalled();
      expect(onCancel).toHaveBeenCalledTimes(1);
      expect(target.captured.size).toBe(0);
    }
  );

  it("ignores secondary and unrelated pointers", () => {
    const onPreview = vi.fn();
    const controller = createGestureController({
      thresholdPx: 1,
      raf: createRaf(),
      onPreview
    });

    controller.pointerDown({ ...pointer("pointerdown", 0), button: 2 }, createTarget());
    controller.pointerMove(pointer("pointermove", 10));
    controller.pointerDown(pointer("pointerdown", 0), createTarget());
    controller.pointerMove(pointer("pointermove", 10, 0, 99));

    expect(onPreview).not.toHaveBeenCalled();
  });
});
