import { describe, expect, it } from "vitest";

import {
  clampFrame,
  frameToSeconds,
  secondsToFrame,
  secondsToTrimFrame,
  stepFrame,
  validateFps
} from "./timebase";

describe("timeline timebase", () => {
  it.each([24, 30, 60] as const)("round-trips integer frames without drift at %i fps", (fps) => {
    for (let frame = 0; frame <= fps * 60 * 10; frame += 137) {
      expect(secondsToFrame(frameToSeconds(frame, fps), fps)).toBe(frame);
    }
  });

  it("keeps persisted command-boundary seconds as an unrounded float", () => {
    expect(frameToSeconds(1, 24)).toBe(1 / 24);
    expect(frameToSeconds(17, 30)).toBe(17 / 30);
  });

  it.each([
    [-1, 30],
    [Number.NaN, 30],
    [Number.POSITIVE_INFINITY, 30]
  ])("rejects invalid seconds %s", (seconds, fps) => {
    expect(() => secondsToFrame(seconds, fps)).toThrow(RangeError);
  });

  it.each([-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY])(
    "rejects invalid frame %s",
    (frame) => {
      expect(() => frameToSeconds(frame, 30)).toThrow(RangeError);
    }
  );

  it.each([0, -30, 25, 29.97, 120, Number.NaN])("rejects unsupported fps %s", (fps) => {
    expect(() => validateFps(fps)).toThrow(RangeError);
  });

  it("rounds trim boundaries inward in the requested direction", () => {
    const betweenFrames = 10.4 / 30;

    expect(secondsToTrimFrame(betweenFrames, 30, "start")).toBe(11);
    expect(secondsToTrimFrame(betweenFrames, 30, "end")).toBe(10);
  });

  it("leaves exact-frame trim boundaries unchanged", () => {
    expect(secondsToTrimFrame(12 / 24, 24, "start")).toBe(12);
    expect(secondsToTrimFrame(12 / 24, 24, "end")).toBe(12);
  });

  it("clamps frames and frame stepping to integer bounds", () => {
    expect(clampFrame(-5, 0, 100)).toBe(0);
    expect(clampFrame(105, 0, 100)).toBe(100);
    expect(stepFrame(99, 5, 0, 100)).toBe(100);
    expect(stepFrame(1, -5, 0, 100)).toBe(0);
  });

  it("rejects invalid clamp ranges and fractional frame steps", () => {
    expect(() => clampFrame(10, 20, 5)).toThrow(RangeError);
    expect(() => stepFrame(10, 0.5, 0, 20)).toThrow(RangeError);
  });
});
