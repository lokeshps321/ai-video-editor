export interface TimelineClock {
  getSnapshot(): number;
  subscribe(listener: () => void): () => void;
  setTime(seconds: number): void;
  beginPreview(): TimelineClockPreview | null;
}

export interface TimelineClockPreview {
  setTime(seconds: number): void;
  commit(seconds: number): void;
  cancel(): void;
}

function normalizeTime(seconds: number): number {
  return Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
}

export function createTimelineClock(initialSeconds = 0): TimelineClock {
  let current = normalizeTime(initialSeconds);
  let activePreview: symbol | null = null;
  const listeners = new Set<() => void>();
  const publish = (seconds: number): void => {
    const next = normalizeTime(seconds);
    if (Object.is(next, current)) return;
    current = next;
    listeners.forEach((listener) => listener());
  };

  return {
    getSnapshot: () => current,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    setTime(seconds) {
      if (activePreview) return;
      publish(seconds);
    },
    beginPreview() {
      if (activePreview) return null;
      const owner = Symbol("timeline-clock-preview");
      const snapshot = current;
      activePreview = owner;
      const release = (seconds: number) => {
        if (activePreview !== owner) return;
        activePreview = null;
        publish(seconds);
      };
      return {
        setTime(seconds) {
          if (activePreview === owner) publish(seconds);
        },
        commit(seconds) {
          release(seconds);
        },
        cancel() {
          release(snapshot);
        },
      };
    },
  };
}
