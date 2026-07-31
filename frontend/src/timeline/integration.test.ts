import { describe, expect, it } from "vitest";

import {
  canClaimGesturePointer,
  compileTimelineKeyboardCommand,
  decideTimelineArrowHandling,
  minimumDurationFrames,
  resolveCanonicalFps,
  shouldCommitFrameChange,
  sourceFrameAfterTimelineDelta,
  sourceBoundaryFrame,
  snapBlockStartFrame,
  snapEdgeFrame,
} from "./integration";
import { frameToSeconds } from "./timebase";

describe("timeline integration adapters", () => {
  it("resolves canonical FPS in timeline, project, fallback order", () => {
    expect(resolveCanonicalFps(24, 60)).toBe(24);
    expect(resolveCanonicalFps(undefined, 60)).toBe(60);
    expect(resolveCanonicalFps(Number.NaN, 0)).toBe(30);
  });

  it("snaps block edges with integer frame math and excludes its owner", () => {
    const guides = [
      { id: "lane:a:start", frame: 30 },
      { id: "lane:b:start", frame: 45 },
      { id: "system:end", frame: 300 },
    ];
    expect(
      snapBlockStartFrame(43, 10, guides, 3, "lane:a:", 0, 300),
    ).toEqual({ frame: 45, guideFrame: 45 });
    expect(snapEdgeFrame(31, guides, 2, "lane:a:", 0, 300)).toEqual({
      frame: 31,
      guideFrame: null,
    });
  });

  it("compiles frame-step and selected-clip nudge commands without overlap", () => {
    expect(
      compileTimelineKeyboardCommand({
        key: "ArrowRight",
        altKey: false,
        shiftKey: false,
        ctrlKey: false,
        metaKey: false,
        currentFrame: 10,
        durationFrames: 20,
        selectedClipStartFrame: null,
      }),
    ).toEqual({ kind: "seek", frame: 11 });
    expect(
      compileTimelineKeyboardCommand({
        key: "ArrowLeft",
        altKey: true,
        shiftKey: false,
        ctrlKey: false,
        metaKey: false,
        currentFrame: 10,
        durationFrames: 20,
        selectedClipStartFrame: 0,
      }),
    ).toEqual({ kind: "nudge-selected-clip", frame: 0 });
    expect(
      compileTimelineKeyboardCommand({
        key: "ArrowRight",
        altKey: false,
        shiftKey: true,
        ctrlKey: false,
        metaKey: false,
        currentFrame: 10,
        durationFrames: 20,
        selectedClipStartFrame: 5,
      }),
    ).toBeNull();
  });

  it.each([24, 30, 60] as const)(
    "compiles seek and nudge commands as one canonical frame at %i fps",
    (fps) => {
      const currentFrame = fps * 2;
      const seek = compileTimelineKeyboardCommand({
        key: "ArrowRight",
        altKey: false,
        shiftKey: false,
        ctrlKey: false,
        metaKey: false,
        currentFrame,
        durationFrames: fps * 10,
        selectedClipStartFrame: currentFrame,
      });
      const nudge = compileTimelineKeyboardCommand({
        key: "ArrowLeft",
        altKey: true,
        shiftKey: false,
        ctrlKey: false,
        metaKey: false,
        currentFrame,
        durationFrames: fps * 10,
        selectedClipStartFrame: currentFrame,
      });

      expect(seek).toEqual({ kind: "seek", frame: currentFrame + 1 });
      expect(nudge).toEqual({
        kind: "nudge-selected-clip",
        frame: currentFrame - 1,
      });
      expect(frameToSeconds(currentFrame + 1, fps)).toBe(
        (currentFrame + 1) / fps,
      );
    },
  );

  it("does not claim a competing, secondary, or non-primary pointer", () => {
    expect(
      canClaimGesturePointer(false, { pointerId: 1, button: 0, isPrimary: true }),
    ).toBe(true);
    expect(
      canClaimGesturePointer(true, { pointerId: 2, button: 0, isPrimary: true }),
    ).toBe(false);
    expect(
      canClaimGesturePointer(false, {
        pointerId: 2,
        button: 0,
        isPrimary: false,
      }),
    ).toBe(false);
    expect(
      canClaimGesturePointer(false, { pointerId: 2, button: 2, isPrimary: true }),
    ).toBe(false);
  });

  it("preserves minimum durations and aligns source boundaries to frames", () => {
    expect(minimumDurationFrames(0.1, 24)).toBe(3);
    expect(minimumDurationFrames(0.05, 30)).toBe(2);
    expect(sourceBoundaryFrame(1.051, 30, "start")).toBe(32);
    expect(sourceBoundaryFrame(1.099, 30, "end")).toBe(32);
    expect(sourceFrameAfterTimelineDelta(30, 1, 1.5, 30, "start")).toBe(
      32,
    );
    expect(sourceFrameAfterTimelineDelta(60, -1, 1.5, 30, "end")).toBe(58);
  });

  it("suppresses frame-identical releases except real cross-lane changes", () => {
    expect(shouldCommitFrameChange(12, 12, false)).toBe(false);
    expect(shouldCommitFrameChange(12, 12, true)).toBe(true);
    expect(shouldCommitFrameChange(12, 13, false)).toBe(true);
  });

  it("reserves Ctrl and Meta arrows for existing shortcuts", () => {
    for (const modifiers of [
      { ctrlKey: true, metaKey: false },
      { ctrlKey: false, metaKey: true },
    ]) {
      expect(
        compileTimelineKeyboardCommand({
          key: "ArrowRight",
          altKey: false,
          shiftKey: false,
          ...modifiers,
          currentFrame: 10,
          durationFrames: 20,
          selectedClipStartFrame: 5,
        }),
      ).toBeNull();
    }
  });

  it("blocks modified arrows before either V2 or legacy keyboard handling", () => {
    for (const timelineCoreV2 of [false, true]) {
      for (const modifiers of [
        { ctrlKey: true, metaKey: false, shiftKey: true, altKey: false },
        { ctrlKey: false, metaKey: true, shiftKey: true, altKey: true },
      ]) {
        expect(
          decideTimelineArrowHandling({
            key: "ArrowLeft",
            timelineCoreV2,
            ...modifiers,
          }),
        ).toBe("blocked");
      }
    }

    expect(
      decideTimelineArrowHandling({
        key: "z",
        timelineCoreV2: true,
        ctrlKey: true,
        metaKey: false,
        shiftKey: false,
        altKey: false,
      }),
    ).toBe("ignore");
    expect(
      decideTimelineArrowHandling({
        key: "ArrowRight",
        timelineCoreV2: true,
        ctrlKey: false,
        metaKey: false,
        shiftKey: true,
        altKey: false,
      }),
    ).toBe("legacy");
  });
});
