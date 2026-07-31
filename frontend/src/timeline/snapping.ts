export interface SnapGuide {
  frame: number;
  id: string;
  priority?: number;
}

export interface SnapResult {
  frame: number;
  deltaFrames: number;
  guide: SnapGuide;
}

interface RankedGuide {
  guide: SnapGuide;
  distance: number;
  priority: number;
}

function requireFrame(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative integer`);
  }
}

function compareCodeUnits(left: string, right: string): number {
  if (left === right) return 0;
  return left < right ? -1 : 1;
}

function rankGuides(left: RankedGuide, right: RankedGuide): number {
  return (
    left.distance - right.distance ||
    left.priority - right.priority ||
    left.guide.frame - right.guide.frame ||
    compareCodeUnits(left.guide.id, right.guide.id)
  );
}

export function findSnap(
  frame: number,
  guides: readonly SnapGuide[],
  thresholdFrames: number
): SnapResult | null {
  requireFrame(frame, "frame");
  requireFrame(thresholdFrames, "thresholdFrames");

  const ids = new Set<string>();
  const ranked = guides.map((guide): RankedGuide => {
    requireFrame(guide.frame, "guide.frame");
    if (guide.id.length === 0 || ids.has(guide.id)) {
      throw new RangeError("guide ids must be non-empty and unique");
    }
    ids.add(guide.id);

    const priority = guide.priority ?? 0;
    if (!Number.isInteger(priority)) {
      throw new RangeError("guide.priority must be an integer");
    }
    return { guide, distance: Math.abs(guide.frame - frame), priority };
  });

  const winner = ranked
    .filter((candidate) => candidate.distance <= thresholdFrames)
    .sort(rankGuides)[0];

  if (!winner) return null;
  return {
    frame: winner.guide.frame,
    deltaFrames: winner.guide.frame - frame,
    guide: winner.guide
  };
}
