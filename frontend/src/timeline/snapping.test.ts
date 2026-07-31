import { describe, expect, it } from "vitest";

import { findSnap } from "./snapping";

describe("timeline snapping", () => {
  it("snaps to a frame guide inside the inclusive threshold", () => {
    expect(
      findSnap(102, [{ frame: 105, id: "clip-end", priority: 1 }], 3)
    ).toEqual({
      frame: 105,
      deltaFrames: 3,
      guide: { frame: 105, id: "clip-end", priority: 1 }
    });
  });

  it("does not snap outside the frame threshold", () => {
    expect(findSnap(100, [{ frame: 104, id: "guide", priority: 0 }], 3)).toBeNull();
  });

  it("breaks equal-distance ties by priority, then frame, then stable id", () => {
    const guides = [
      { frame: 98, id: "z", priority: 1 },
      { frame: 102, id: "b", priority: 0 },
      { frame: 98, id: "c", priority: 0 },
      { frame: 98, id: "a", priority: 0 }
    ];

    expect(findSnap(100, guides, 4)?.guide.id).toBe("a");
  });

  it("is deterministic when guide input order changes", () => {
    const guides = [
      { frame: 52, id: "right", priority: 2 },
      { frame: 48, id: "left", priority: 2 }
    ];

    expect(findSnap(50, guides, 2)?.guide.id).toBe("left");
    expect(findSnap(50, [...guides].reverse(), 2)?.guide.id).toBe("left");
  });

  it("uses locale-independent code-unit ordering for distinct Unicode ids", () => {
    const decomposed = "e\u0301";
    const composed = "\u00e9";
    const guides = [
      { frame: 50, id: composed },
      { frame: 50, id: decomposed }
    ];

    expect(composed).not.toBe(decomposed);
    expect(composed.localeCompare(decomposed)).toBe(0);
    expect(findSnap(50, guides, 0)?.guide.id).toBe(decomposed);
    expect(findSnap(50, [...guides].reverse(), 0)?.guide.id).toBe(decomposed);
  });

  it("rejects fractional frames, duplicate ids, and invalid thresholds", () => {
    expect(() => findSnap(1.5, [], 2)).toThrow(RangeError);
    expect(() =>
      findSnap(
        10,
        [
          { frame: 9, id: "same" },
          { frame: 11, id: "same" }
        ],
        2
      )
    ).toThrow(RangeError);
    expect(() => findSnap(10, [], -1)).toThrow(RangeError);
  });
});
