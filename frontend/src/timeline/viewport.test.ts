import { describe, expect, it } from "vitest";

import { computeAutoScroll, zoomViewportAtCursor } from "./viewport";

describe("timeline viewport math", () => {
  it("keeps the frame under the cursor anchored within 0.5px while zooming", () => {
    const viewportLeft = 100;
    const cursorClientX = 350;
    const result = zoomViewportAtCursor({
      scrollLeft: 1_250,
      viewportLeft,
      cursorClientX,
      oldPixelsPerFrame: 10,
      newPixelsPerFrame: 25,
      maxScrollLeft: 10_000
    });

    const anchoredClientX =
      viewportLeft + result.anchorFrame * 25 - result.scrollLeft;
    expect(Math.abs(anchoredClientX - cursorClientX)).toBeLessThanOrEqual(0.5);
    expect(result.anchorFrame).toBe(150);
  });

  it("clamps cursor-anchored zoom to valid scroll bounds", () => {
    expect(
      zoomViewportAtCursor({
        scrollLeft: 0,
        viewportLeft: 100,
        cursorClientX: 50,
        oldPixelsPerFrame: 10,
        newPixelsPerFrame: 20,
        maxScrollLeft: 500
      }).scrollLeft
    ).toBe(0);

    expect(
      zoomViewportAtCursor({
        scrollLeft: 500,
        viewportLeft: 0,
        cursorClientX: 500,
        oldPixelsPerFrame: 1,
        newPixelsPerFrame: 10,
        maxScrollLeft: 600
      }).scrollLeft
    ).toBe(600);
  });

  it("computes deterministic proportional auto-scroll at both edges", () => {
    expect(
      computeAutoScroll({
        pointerClientX: 110,
        viewportLeft: 100,
        viewportWidth: 500,
        edgeSizePx: 50,
        maxSpeedPxPerSecond: 1_000,
        elapsedMs: 100,
        scrollLeft: 200,
        maxScrollLeft: 1_000
      })
    ).toEqual({ delta: -80, scrollLeft: 120 });

    expect(
      computeAutoScroll({
        pointerClientX: 590,
        viewportLeft: 100,
        viewportWidth: 500,
        edgeSizePx: 50,
        maxSpeedPxPerSecond: 1_000,
        elapsedMs: 100,
        scrollLeft: 200,
        maxScrollLeft: 1_000
      })
    ).toEqual({ delta: 80, scrollLeft: 280 });
  });

  it("returns no auto-scroll in the viewport interior", () => {
    expect(
      computeAutoScroll({
        pointerClientX: 300,
        viewportLeft: 100,
        viewportWidth: 500,
        edgeSizePx: 50,
        maxSpeedPxPerSecond: 1_000,
        elapsedMs: 100,
        scrollLeft: 200,
        maxScrollLeft: 1_000
      })
    ).toEqual({ delta: 0, scrollLeft: 200 });
  });

  it("clamps auto-scroll and reports the applied delta", () => {
    expect(
      computeAutoScroll({
        pointerClientX: 100,
        viewportLeft: 100,
        viewportWidth: 500,
        edgeSizePx: 50,
        maxSpeedPxPerSecond: 1_000,
        elapsedMs: 100,
        scrollLeft: 20,
        maxScrollLeft: 1_000
      })
    ).toEqual({ delta: -20, scrollLeft: 0 });
  });

  it("rejects invalid zoom and auto-scroll inputs", () => {
    expect(() =>
      zoomViewportAtCursor({
        scrollLeft: 0,
        viewportLeft: 0,
        cursorClientX: 0,
        oldPixelsPerFrame: 0,
        newPixelsPerFrame: 10,
        maxScrollLeft: 100
      })
    ).toThrow(RangeError);
    expect(() =>
      computeAutoScroll({
        pointerClientX: 0,
        viewportLeft: 0,
        viewportWidth: 100,
        edgeSizePx: 60,
        maxSpeedPxPerSecond: 100,
        elapsedMs: -1,
        scrollLeft: 0,
        maxScrollLeft: 100
      })
    ).toThrow(RangeError);
  });
});
