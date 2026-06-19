import type { TranscriptWord } from "../types";

export function selectTranscriptWordIdsInRange(
  words: TranscriptWord[],
  startSec: number,
  endSec: number
): string[] {
  const lo = Math.min(startSec, endSec);
  const hi = Math.max(startSec, endSec);
  return words
    .filter((word) => word.start_sec < hi && word.end_sec > lo)
    .map((word) => word.id);
}

export function lockedLaneStorageKey(projectId: string): string {
  return `clipmind_locked_lanes_${projectId}`;
}

export function readLockedLaneIds(projectId: string): Set<string> {
  try {
    const raw = localStorage.getItem(lockedLaneStorageKey(projectId));
    if (!raw) return new Set();
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return new Set();
    return new Set(parsed.filter((value) => typeof value === "string"));
  } catch {
    return new Set();
  }
}

export function writeLockedLaneIds(projectId: string, laneIds: Set<string>): void {
  try {
    localStorage.setItem(lockedLaneStorageKey(projectId), JSON.stringify(Array.from(laneIds)));
  } catch {
    // Ignore storage failures.
  }
}
