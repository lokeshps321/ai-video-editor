import { describe, expect, it, vi } from "vitest";

import { createTimelineClock } from "./clock";

describe("timeline clock", () => {
  it("publishes time without requiring React state", () => {
    const clock = createTimelineClock(1);
    const listener = vi.fn();
    const unsubscribe = clock.subscribe(listener);

    clock.setTime(2.5);
    clock.setTime(2.5);

    expect(clock.getSnapshot()).toBe(2.5);
    expect(listener).toHaveBeenCalledTimes(1);

    unsubscribe();
    clock.setTime(3);
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("clamps invalid or negative values to zero", () => {
    const clock = createTimelineClock(4);
    clock.setTime(-1);
    expect(clock.getSnapshot()).toBe(0);
    clock.setTime(Number.NaN);
    expect(clock.getSnapshot()).toBe(0);
  });

  it("arbitrates a scrub preview against playback and restores its exact snapshot", () => {
    const clock = createTimelineClock(1.234567);
    const preview = clock.beginPreview();
    expect(preview).not.toBeNull();

    preview?.setTime(8);
    clock.setTime(3);
    expect(clock.getSnapshot()).toBe(8);

    preview?.cancel();
    expect(clock.getSnapshot()).toBe(1.234567);
    clock.setTime(3);
    expect(clock.getSnapshot()).toBe(3);
  });

  it("allows only one preview owner and commits its final value", () => {
    const clock = createTimelineClock(2);
    const preview = clock.beginPreview();
    expect(clock.beginPreview()).toBeNull();
    preview?.setTime(5);
    preview?.commit(6);
    expect(clock.getSnapshot()).toBe(6);
    expect(clock.beginPreview()).not.toBeNull();
  });
});
