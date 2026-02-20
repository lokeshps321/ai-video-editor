import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Clip, TranscriptWord } from "../types";

export type TimelineProps = {
    words: TranscriptWord[];
    overlayClips: Clip[];
    durationSec: number;
    currentTimeSec: number;
    deletedWordIds: Set<string>;
    selectedWordIds: Set<string>;
    activeWordId: string | null;
    waveformPeaks: number[];
    onSeek: (sec: number) => void;
    onSelectWord: (id: string, shift: boolean) => void;
    onSelectWordsInRange: (startSec: number, endSec: number) => void;
    onDeleteSelected: () => void;
    onRestoreSelected: () => void;
    onMoveBrollClip: (clipId: string, timelineStartSec: number) => void;
    onTrimBrollClip: (clipId: string, durationSec: number) => void;
    onSetBrollOpacity: (clipId: string, opacity: number) => void;
    onDeleteBrollClip: (clipId: string) => void;
    brollEditBusy: boolean;
};

const MIN_PX_PER_SEC = 15;
const MAX_PX_PER_SEC = 250;
const DEFAULT_PX_PER_SEC = 40;
const TRACK_LEFT_MARGIN = 52;
const MIN_BROLL_DURATION_SEC = 0.1;

function formatTimecode(sec: number): string {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatDuration(sec: number): string {
    if (sec < 1) return `${Math.round(sec * 1000)}ms`;
    return `${sec.toFixed(1)}s`;
}

function clipTimelineDuration(clip: Clip): number {
    return Math.max((clip.end_sec - clip.start_sec) / Math.max(clip.speed, 0.01), MIN_BROLL_DURATION_SEC);
}

function clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
}

export default function Timeline({
    words,
    overlayClips,
    durationSec,
    currentTimeSec,
    deletedWordIds,
    selectedWordIds,
    activeWordId,
    waveformPeaks,
    onSeek,
    onSelectWord,
    onSelectWordsInRange,
    onDeleteSelected,
    onRestoreSelected,
    onMoveBrollClip,
    onTrimBrollClip,
    onSetBrollOpacity,
    onDeleteBrollClip,
    brollEditBusy,
}: TimelineProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [pxPerSec, setPxPerSec] = useState(DEFAULT_PX_PER_SEC);

    // ── Drag modes ──────────────────────────────────────────────
    type DragMode = "none" | "seek" | "range";
    const [dragMode, setDragMode] = useState<DragMode>("none");
    const [rangeStart, setRangeStart] = useState<number | null>(null);
    const [rangeEnd, setRangeEnd] = useState<number | null>(null);

    // ── Context menu ────────────────────────────────────────────
    const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null);
    const [selectedBrollId, setSelectedBrollId] = useState<string | null>(null);
    const [brollOpacityDraftById, setBrollOpacityDraftById] = useState<Record<string, number>>({});
    const opacityCommitTimersRef = useRef<Record<string, number>>({});

    type BrollDragState = {
        clipId: string;
        mode: "move" | "resize-end";
        startClientX: number;
        initialStartSec: number;
        initialDurationSec: number;
        currentStartSec: number;
        currentDurationSec: number;
    };
    const [brollDragState, setBrollDragState] = useState<BrollDragState | null>(null);

    const totalWidth = Math.max(durationSec * pxPerSec, 200);

    // ── Ruler ticks ────────────────────────────────────────────
    const ticks = useMemo(() => {
        let interval = 1;
        if (pxPerSec < 20) interval = 10;
        else if (pxPerSec < 40) interval = 5;
        else if (pxPerSec < 80) interval = 2;
        else if (pxPerSec < 150) interval = 1;
        else interval = 0.5;

        const result: { sec: number; x: number; label: string; major: boolean }[] = [];
        for (let sec = 0; sec <= durationSec; sec += interval) {
            result.push({
                sec,
                x: sec * pxPerSec,
                label: formatTimecode(sec),
                major: sec % (interval >= 1 ? 5 : 1) === 0,
            });
        }
        return result;
    }, [durationSec, pxPerSec]);

    // ── Waveform bars ─────────────────────────────────────────
    const waveformBars = useMemo(() => {
        if (!waveformPeaks.length) return [];
        const barWidth = totalWidth / waveformPeaks.length;
        return waveformPeaks.map((peak, i) => ({
            x: i * barWidth,
            width: Math.max(barWidth - 0.5, 0.5),
            height: peak,
        }));
    }, [waveformPeaks, totalWidth]);

    // ── Deleted region overlays ───────────────────────────────
    const deletedRegions = useMemo(() => {
        if (!words.length) return [];
        const sorted = [...words].sort((a, b) => a.start_sec - b.start_sec);
        const regions: { startSec: number; endSec: number }[] = [];
        let regionStart: number | null = null;

        for (const w of sorted) {
            if (deletedWordIds.has(w.id)) {
                if (regionStart === null) regionStart = w.start_sec;
            } else {
                if (regionStart !== null) {
                    // end the region at the start of this kept word
                    const prevDeleted = sorted.find(
                        (pw) => pw.end_sec <= w.start_sec && deletedWordIds.has(pw.id)
                    );
                    regions.push({
                        startSec: regionStart,
                        endSec: prevDeleted ? prevDeleted.end_sec : w.start_sec,
                    });
                    regionStart = null;
                }
            }
        }
        // Trailing deleted region
        if (regionStart !== null) {
            const lastWord = sorted[sorted.length - 1];
            regions.push({ startSec: regionStart, endSec: lastWord.end_sec });
        }
        return regions;
    }, [words, deletedWordIds]);

    // ── Word blocks ───────────────────────────────────────────
    const wordBlocks = useMemo(() => {
        return words.map((word) => {
            const x = word.start_sec * pxPerSec;
            const w = Math.max((word.end_sec - word.start_sec) * pxPerSec, 3);
            const isDeleted = deletedWordIds.has(word.id);
            const isSelected = selectedWordIds.has(word.id);
            const isActive = activeWordId === word.id;
            return { word, x, w, isDeleted, isSelected, isActive };
        });
    }, [words, pxPerSec, deletedWordIds, selectedWordIds, activeWordId]);

    // ── B-roll overlay blocks ────────────────────────────────
    const brollBlocks = useMemo(() => {
        return overlayClips
            .slice()
            .sort((a, b) => a.timeline_start_sec - b.timeline_start_sec)
            .map((clip) => {
                const dragPreview = brollDragState?.clipId === clip.id ? brollDragState : null;
                const timelineStartSec = dragPreview ? dragPreview.currentStartSec : clip.timeline_start_sec;
                const duration = dragPreview ? dragPreview.currentDurationSec : clipTimelineDuration(clip);
                const x = timelineStartSec * pxPerSec;
                const w = Math.max(duration * pxPerSec, 4);
                const clipOpacity = typeof clip.broll_opacity === "number" ? clip.broll_opacity : 1;
                const opacity = brollOpacityDraftById[clip.id] ?? clipOpacity;
                return {
                    clip,
                    x,
                    w,
                    opacity,
                    timelineStartSec,
                    duration,
                    isDragging: !!dragPreview,
                };
            });
    }, [overlayClips, pxPerSec, brollDragState, brollOpacityDraftById]);

    useEffect(() => {
        setBrollOpacityDraftById(() => {
            const next: Record<string, number> = {};
            overlayClips.forEach((clip) => {
                next[clip.id] = clamp(typeof clip.broll_opacity === "number" ? clip.broll_opacity : 1, 0, 1);
            });
            return next;
        });
        setSelectedBrollId((prev) => (prev && overlayClips.some((clip) => clip.id === prev) ? prev : null));
    }, [overlayClips]);

    useEffect(() => {
        return () => {
            Object.values(opacityCommitTimersRef.current).forEach((timer) => window.clearTimeout(timer));
            opacityCommitTimersRef.current = {};
        };
    }, []);

    // ── Playhead position ─────────────────────────────────────
    const playheadX = currentTimeSec * pxPerSec;

    // ── Range selection bounds ────────────────────────────────
    const rangeLeft = rangeStart !== null && rangeEnd !== null
        ? Math.min(rangeStart, rangeEnd) * pxPerSec
        : null;
    const rangeWidth = rangeStart !== null && rangeEnd !== null
        ? Math.abs(rangeEnd - rangeStart) * pxPerSec
        : null;
    const rangeDuration = rangeStart !== null && rangeEnd !== null
        ? Math.abs(rangeEnd - rangeStart)
        : null;

    // ── Auto-scroll to playhead ───────────────────────────────
    useEffect(() => {
        if (!containerRef.current || dragMode !== "none" || brollDragState) return;
        const el = containerRef.current;
        const viewLeft = el.scrollLeft;
        const viewRight = viewLeft + el.clientWidth;
        if (playheadX < viewLeft + 40 || playheadX > viewRight - 40) {
            el.scrollLeft = playheadX - el.clientWidth / 3;
        }
    }, [playheadX, dragMode, brollDragState]);

    // ── Convert mouse event → seconds ────────────────────────
    const secFromEvent = useCallback(
        (e: React.MouseEvent) => {
            if (!containerRef.current) return 0;
            const rect = containerRef.current.getBoundingClientRect();
            const x = e.clientX - rect.left + containerRef.current.scrollLeft - TRACK_LEFT_MARGIN;
            return Math.max(0, Math.min(x / pxPerSec, durationSec));
        },
        [pxPerSec, durationSec]
    );

    function scheduleOpacityCommit(clipId: string, opacity: number) {
        const prev = opacityCommitTimersRef.current[clipId];
        if (typeof prev === "number") {
            window.clearTimeout(prev);
        }
        opacityCommitTimersRef.current[clipId] = window.setTimeout(() => {
            onSetBrollOpacity(clipId, opacity);
            delete opacityCommitTimersRef.current[clipId];
        }, 180);
    }

    function startBrollDrag(
        event: React.MouseEvent,
        clip: Clip,
        mode: "move" | "resize-end"
    ) {
        if (event.button !== 0 || brollEditBusy) return;
        event.preventDefault();
        event.stopPropagation();
        const durationSec = clipTimelineDuration(clip);
        setSelectedBrollId(clip.id);
        setBrollDragState({
            clipId: clip.id,
            mode,
            startClientX: event.clientX,
            initialStartSec: clip.timeline_start_sec,
            initialDurationSec: durationSec,
            currentStartSec: clip.timeline_start_sec,
            currentDurationSec: durationSec,
        });
    }

    // ── Mouse handlers ────────────────────────────────────────
    function handleMouseDown(e: React.MouseEvent) {
        if (e.button !== 0) return;
        setContextMenu(null);

        if (e.altKey || e.shiftKey) {
            // Alt+click or Shift+click starts range selection
            const sec = secFromEvent(e);
            setDragMode("range");
            setRangeStart(sec);
            setRangeEnd(sec);
        } else {
            // Regular click = seek
            setDragMode("seek");
            setRangeStart(null);
            setRangeEnd(null);
            onSeek(secFromEvent(e));
        }
    }

    function handleMouseMove(e: React.MouseEvent) {
        if (dragMode === "seek") {
            onSeek(secFromEvent(e));
        } else if (dragMode === "range") {
            setRangeEnd(secFromEvent(e));
        }
    }

    useEffect(() => {
        function up() {
            if (dragMode === "range" && rangeStart !== null && rangeEnd !== null) {
                const lo = Math.min(rangeStart, rangeEnd);
                const hi = Math.max(rangeStart, rangeEnd);
                if (hi - lo > 0.05) {
                    onSelectWordsInRange(lo, hi);
                }
            }
            setDragMode("none");
        }
        window.addEventListener("mouseup", up);
        return () => window.removeEventListener("mouseup", up);
    }, [dragMode, rangeStart, rangeEnd, onSelectWordsInRange]);

    useEffect(() => {
        if (!brollDragState) return;

        function onMove(event: MouseEvent) {
            setBrollDragState((prev) => {
                if (!prev) return prev;
                const deltaSec = (event.clientX - prev.startClientX) / pxPerSec;
                if (prev.mode === "move") {
                    const maxStart = Math.max(durationSec, prev.initialStartSec + 30);
                    return {
                        ...prev,
                        currentStartSec: clamp(prev.initialStartSec + deltaSec, 0, maxStart),
                    };
                }
                const maxDuration = Math.max(
                    durationSec - prev.initialStartSec,
                    prev.initialDurationSec + 30,
                    MIN_BROLL_DURATION_SEC
                );
                return {
                    ...prev,
                    currentDurationSec: clamp(
                        prev.initialDurationSec + deltaSec,
                        MIN_BROLL_DURATION_SEC,
                        maxDuration
                    ),
                };
            });
        }

        function onUp() {
            setBrollDragState((prev) => {
                if (!prev) return prev;
                if (prev.mode === "move") {
                    if (Math.abs(prev.currentStartSec - prev.initialStartSec) >= 0.01) {
                        onMoveBrollClip(prev.clipId, Number(prev.currentStartSec.toFixed(3)));
                    }
                } else if (Math.abs(prev.currentDurationSec - prev.initialDurationSec) >= 0.01) {
                    onTrimBrollClip(prev.clipId, Number(prev.currentDurationSec.toFixed(3)));
                }
                return null;
            });
        }

        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
        return () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
        };
    }, [brollDragState, pxPerSec, durationSec, onMoveBrollClip, onTrimBrollClip]);

    // ── Right-click context menu ──────────────────────────────
    function handleContextMenu(e: React.MouseEvent) {
        e.preventDefault();
        setContextMenu({ x: e.clientX, y: e.clientY });
    }

    useEffect(() => {
        function close() { setContextMenu(null); }
        window.addEventListener("click", close);
        return () => window.removeEventListener("click", close);
    }, []);

    // ── Zoom with scroll wheel ────────────────────────────────
    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;
        function onWheel(e: WheelEvent) {
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                setPxPerSec((prev) => {
                    const delta = e.deltaY > 0 ? 0.85 : 1.18;
                    return Math.max(MIN_PX_PER_SEC, Math.min(MAX_PX_PER_SEC, prev * delta));
                });
            }
        }
        el.addEventListener("wheel", onWheel, { passive: false });
        return () => el.removeEventListener("wheel", onWheel);
    }, []);

    // ── Count selected for context menu label ─────────────────
    const selectedCount = selectedWordIds.size;
    const selectedHasDeleted = useMemo(() => {
        for (const id of selectedWordIds) {
            if (deletedWordIds.has(id)) return true;
        }
        return false;
    }, [selectedWordIds, deletedWordIds]);

    return (
        <section className="timeline card">
            <div className="timelineHeader">
                <h3>
                    Timeline
                    <span className="tlHint">
                        Click to seek · Alt+drag to select range · Drag B-roll to move · Alt+wheel B-roll opacity
                    </span>
                </h3>
                <div className="zoomControls">
                    <button
                        className="zoomBtn"
                        onClick={() => setPxPerSec((v) => Math.max(MIN_PX_PER_SEC, v * 0.7))}
                        title="Zoom out"
                    >
                        −
                    </button>
                    <input
                        type="range"
                        min={MIN_PX_PER_SEC}
                        max={MAX_PX_PER_SEC}
                        step={1}
                        value={pxPerSec}
                        onChange={(e) => setPxPerSec(Number(e.target.value))}
                        className="zoomSlider"
                    />
                    <button
                        className="zoomBtn"
                        onClick={() => setPxPerSec((v) => Math.min(MAX_PX_PER_SEC, v * 1.4))}
                        title="Zoom in"
                    >
                        +
                    </button>
                    <span className="zoomLabel">{Math.round(pxPerSec)}px/s</span>
                </div>
            </div>

            <div
                className="timelineScroll"
                ref={containerRef}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onContextMenu={handleContextMenu}
            >
                <div className="timelineCanvas" style={{ width: totalWidth }}>
                    {/* ── TIME RULER ───────────────────────── */}
                    <div className="timeRuler">
                        {ticks.map((tick, i) => (
                            <div
                                key={i}
                                className={`tick ${tick.major ? "major" : ""}`}
                                style={{ left: tick.x }}
                            >
                                {tick.major && <span className="tickLabel">{tick.label}</span>}
                            </div>
                        ))}
                    </div>

                    {/* ── WAVEFORM TRACK ───────────────────── */}
                    <div className="waveformTrack">
                        <span className="trackLabel">Audio</span>
                        <svg className="waveformSvg" width={totalWidth} height={50} preserveAspectRatio="none">
                            {waveformBars.map((bar, i) => (
                                <rect
                                    key={i}
                                    x={bar.x}
                                    y={50 - bar.height * 46}
                                    width={bar.width}
                                    height={bar.height * 46}
                                    rx={1}
                                />
                            ))}
                        </svg>
                        {/* Deleted region overlays on waveform */}
                        {deletedRegions.map((r, i) => (
                            <div
                                key={`del-wave-${i}`}
                                className="deletedOverlay"
                                style={{
                                    left: r.startSec * pxPerSec,
                                    width: (r.endSec - r.startSec) * pxPerSec,
                                }}
                            />
                        ))}
                    </div>

                    {/* ── B-ROLL TRACK ─────────────────────── */}
                    <div className="brollTrack">
                        <span className="trackLabel">B-roll</span>
                        {brollBlocks.length === 0 && (
                            <div className="brollEmpty">No overlay clips</div>
                        )}
                        {brollBlocks.map(({ clip, x, w, opacity, timelineStartSec, duration, isDragging }) => {
                            const isSelected = selectedBrollId === clip.id;
                            return (
                                <div
                                    key={clip.id}
                                    className={[
                                        "brollBlock",
                                        isSelected ? "selected" : "",
                                        isDragging ? "dragging" : "",
                                        brollEditBusy ? "disabled" : "",
                                    ].filter(Boolean).join(" ")}
                                    style={{ left: x, width: w, opacity: Math.max(0.28, Math.min(opacity, 1)) }}
                                    onMouseDown={(event) => startBrollDrag(event, clip, "move")}
                                    onClick={(event) => {
                                        event.stopPropagation();
                                        setSelectedBrollId(clip.id);
                                    }}
                                    onDoubleClick={(event) => {
                                        event.stopPropagation();
                                        onSeek(timelineStartSec);
                                    }}
                                    onWheel={(event) => {
                                        if (!event.altKey || brollEditBusy) return;
                                        event.preventDefault();
                                        event.stopPropagation();
                                        setSelectedBrollId(clip.id);
                                        const current = brollOpacityDraftById[clip.id] ?? clamp(
                                            typeof clip.broll_opacity === "number" ? clip.broll_opacity : 1,
                                            0,
                                            1
                                        );
                                        const step = event.deltaY < 0 ? 0.04 : -0.04;
                                        const next = clamp(current + step, 0, 1);
                                        setBrollOpacityDraftById((prev) => ({ ...prev, [clip.id]: next }));
                                        scheduleOpacityCommit(clip.id, Number(next.toFixed(3)));
                                    }}
                                    title={`B-roll ${formatTimecode(timelineStartSec)} · ${formatDuration(duration)} · opacity ${(opacity * 100).toFixed(0)}%`}
                                >
                                    {w > 74 ? `${(opacity * 100).toFixed(0)}%` : ""}
                                    <button
                                        type="button"
                                        className="brollDeleteBtn"
                                        disabled={brollEditBusy}
                                        onMouseDown={(event) => {
                                            event.preventDefault();
                                            event.stopPropagation();
                                        }}
                                        onClick={(event) => {
                                            event.preventDefault();
                                            event.stopPropagation();
                                            onDeleteBrollClip(clip.id);
                                        }}
                                        title="Remove B-roll clip"
                                    >
                                        ×
                                    </button>
                                    <div
                                        className="brollResizeHandle"
                                        onMouseDown={(event) => startBrollDrag(event, clip, "resize-end")}
                                        title="Drag to trim B-roll duration"
                                    />
                                </div>
                            );
                        })}
                    </div>

                    {/* ── WORD TRACK ───────────────────────── */}
                    <div className="wordTrack">
                        <span className="trackLabel">Words</span>
                        {/* Deleted region overlays on word track */}
                        {deletedRegions.map((r, i) => (
                            <div
                                key={`del-word-${i}`}
                                className="deletedOverlay wordDeletedOverlay"
                                style={{
                                    left: r.startSec * pxPerSec,
                                    width: (r.endSec - r.startSec) * pxPerSec,
                                }}
                            />
                        ))}
                        {wordBlocks.map(({ word, x, w, isDeleted, isSelected, isActive }) => (
                            <div
                                key={word.id}
                                className={[
                                    "tlWord",
                                    isDeleted ? "deleted" : "",
                                    isSelected ? "selected" : "",
                                    isActive ? "active" : "",
                                ]
                                    .filter(Boolean)
                                    .join(" ")}
                                style={{ left: x, width: w }}
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onSelectWord(word.id, e.shiftKey);
                                }}
                                title={word.text}
                            >
                                {w > 24 ? word.text : ""}
                            </div>
                        ))}
                    </div>

                    {/* ── RANGE SELECTION OVERLAY ───────────── */}
                    {rangeLeft !== null && rangeWidth !== null && rangeWidth > 2 && (
                        <div
                            className="rangeSelection"
                            style={{ left: rangeLeft + TRACK_LEFT_MARGIN, width: rangeWidth }}
                        >
                            {rangeDuration !== null && rangeDuration > 0.1 && (
                                <span className="rangeLabel">
                                    {formatDuration(rangeDuration)}
                                </span>
                            )}
                        </div>
                    )}

                    {/* ── PLAYHEAD ─────────────────────────── */}
                    <div className="playhead" style={{ left: playheadX }}>
                        <div className="playheadHead" />
                        <div className="playheadLine" />
                    </div>
                </div>
            </div>

            {/* ── CONTEXT MENU ──────────────────────────── */}
            {contextMenu && (
                <div
                    className="tlContextMenu"
                    style={{ left: contextMenu.x, top: contextMenu.y }}
                >
                    <button
                        disabled={!selectedCount}
                        onClick={() => { onDeleteSelected(); setContextMenu(null); }}
                    >
                        🗑 Delete Selected ({selectedCount})
                    </button>
                    <button
                        disabled={!selectedHasDeleted}
                        onClick={() => { onRestoreSelected(); setContextMenu(null); }}
                    >
                        ↩ Restore Selected
                    </button>
                    <hr />
                    <button onClick={() => { onSeek(currentTimeSec); setContextMenu(null); }}>
                        📍 Seek to Playhead ({formatTimecode(currentTimeSec)})
                    </button>
                </div>
            )}
        </section>
    );
}
