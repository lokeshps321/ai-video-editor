import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./lib/api";
import type { Job, MediaAsset, Project, Transcript, TranscriptWord, VibeAction } from "./types";
import Timeline from "./components/Timeline";

type TextBlock = {
  id: string;
  wordIds: string[];
  text: string;
  startSec: number;
  endSec: number;
};

const LOW_CONFIDENCE_THRESHOLD = 0.6;
const LOW_CONFIDENCE_WARN_RATIO = 0.18;
const LOW_CONFIDENCE_WARN_MIN_COUNT = 30;

const FILLER_WORDS = new Set([
  "um", "uh", "uhm", "umm", "hmm", "hm", "mm",
  "ah", "er", "erm", "eh", "huh", "mhm",
  "like", "basically", "literally", "actually",
  "you know", "i mean", "sort of", "kind of",
  "right", "okay", "so", "well", "yeah",
]);

function formatSeconds(value: number): string {
  if (!Number.isFinite(value)) return "0:00";
  const mins = Math.floor(value / 60);
  const secs = Math.floor(value % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function resolveMediaPath(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${api.apiBase}${path}`;
}

function buildSentenceBlocks(words: TranscriptWord[]): TextBlock[] {
  if (!words.length) return [];
  const blocks: TextBlock[] = [];
  let current: TranscriptWord[] = [];

  const flush = () => {
    if (!current.length) return;
    const id = `sent-${blocks.length + 1}`;
    blocks.push({
      id,
      wordIds: current.map((word) => word.id),
      text: current.map((word) => word.text).join(" "),
      startSec: current[0].start_sec,
      endSec: current[current.length - 1].end_sec
    });
    current = [];
  };

  for (let idx = 0; idx < words.length; idx += 1) {
    const word = words[idx];
    const prev = idx > 0 ? words[idx - 1] : null;
    if (current.length > 0 && prev && (word.start_sec - prev.end_sec) > 1.1) {
      flush();
    }

    current.push(word);
    const token = word.text.trim();
    const endsSentence = /[.!?]$/.test(token);
    const maxWordsReached = current.length >= 12;
    if (endsSentence || maxWordsReached) {
      flush();
    }
  }
  flush();
  return blocks;
}

function buildParagraphBlocks(sentences: TextBlock[]): TextBlock[] {
  if (!sentences.length) return [];
  const blocks: TextBlock[] = [];
  let current: TextBlock[] = [];

  const flush = () => {
    if (!current.length) return;
    const id = `para-${blocks.length + 1}`;
    blocks.push({
      id,
      wordIds: current.flatMap((item) => item.wordIds),
      text: current.map((item) => item.text).join(" "),
      startSec: current[0].startSec,
      endSec: current[current.length - 1].endSec
    });
    current = [];
  };

  for (let idx = 0; idx < sentences.length; idx += 1) {
    const sentence = sentences[idx];
    const prev = idx > 0 ? sentences[idx - 1] : null;
    if (current.length > 0 && prev && (sentence.startSec - prev.endSec) > 1.5) {
      flush();
    }
    current.push(sentence);
    if (current.length >= 3) {
      flush();
    }
  }
  flush();
  return blocks;
}

function isFillerWord(text: string): boolean {
  return FILLER_WORDS.has(text.toLowerCase().replace(/[^a-z\s]/g, "").trim());
}

// ── Undo / Redo history ──────────────────────────────────────────────
type UndoEntry = { deletedIds: Set<string> };
const MAX_UNDO = 80;

function App() {
  const [projectName, setProjectName] = useState("Text + Vibe Editor");
  const [project, setProject] = useState<Project | null>(null);
  const [media, setMedia] = useState<MediaAsset[]>([]);
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<Transcript | null>(null);

  const [deletedWordIds, setDeletedWordIds] = useState<Set<string>>(new Set());
  const [selectedWordIds, setSelectedWordIds] = useState<Set<string>>(new Set());
  const [anchorWordId, setAnchorWordId] = useState<string | null>(null);

  // Undo / Redo
  const undoStack = useRef<UndoEntry[]>([]);
  const redoStack = useRef<UndoEntry[]>([]);

  // Inline editing
  const [editingWordId, setEditingWordId] = useState<string | null>(null);
  const [editingWordText, setEditingWordText] = useState("");
  const editInputRef = useRef<HTMLInputElement | null>(null);

  // Search
  const [searchQuery, setSearchQuery] = useState("");
  const [searchMatchIndex, setSearchMatchIndex] = useState(0);

  // Drag selection
  const isDragging = useRef(false);
  const dragStartWordId = useRef<string | null>(null);

  // Loading states
  const [creatingProject, setCreatingProject] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [generatingTranscript, setGeneratingTranscript] = useState(false);
  const [applyingCut, setApplyingCut] = useState(false);
  const [queueingPreview, setQueueingPreview] = useState(false);
  const [runningAction, setRunningAction] = useState<VibeAction | null>(null);

  const [previewJob, setPreviewJob] = useState<Job | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [currentTimeSec, setCurrentTimeSec] = useState(0);

  // Waveform data for timeline
  const [waveformPeaks, setWaveformPeaks] = useState<number[]>([]);

  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<"checking" | "ok" | "down">("checking");

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const lastAppliedSignatureRef = useRef<string>("");
  const pendingPreviewRefreshRef = useRef(false);
  const transcriptBoxRef = useRef<HTMLDivElement | null>(null);
  const activeWordRef = useRef<HTMLButtonElement | null>(null);

  const videoAssets = useMemo(() => media.filter((asset) => asset.media_type === "video"), [media]);

  const selectedVideoAsset = useMemo(() => {
    if (!selectedAssetId) return videoAssets[0] ?? null;
    return videoAssets.find((asset) => asset.id === selectedAssetId) ?? null;
  }, [selectedAssetId, videoAssets]);

  const transcriptWordIndex = useMemo(() => {
    const index = new Map<string, number>();
    transcript?.words.forEach((word, idx) => {
      index.set(word.id, idx);
    });
    return index;
  }, [transcript]);

  const sentenceBlocks = useMemo(() => buildSentenceBlocks(transcript?.words ?? []), [transcript?.words]);
  const paragraphBlocks = useMemo(() => buildParagraphBlocks(sentenceBlocks), [sentenceBlocks]);

  const deletedSignature = useMemo(() => Array.from(deletedWordIds).sort().join(","), [deletedWordIds]);

  const keptWordIds = useMemo(() => {
    if (!transcript) return [] as string[];
    return transcript.words.filter((word) => !deletedWordIds.has(word.id)).map((word) => word.id);
  }, [transcript, deletedWordIds]);

  const activeWordId = useMemo(() => {
    if (!transcript) return null;
    return transcript.words.find((word) => currentTimeSec >= word.start_sec && currentTimeSec <= word.end_sec)?.id ?? null;
  }, [transcript, currentTimeSec]);

  const lowConfidenceCount = useMemo(() => {
    if (!transcript) return 0;
    return transcript.words.filter(
      (word) => typeof word.confidence === "number" && word.confidence < LOW_CONFIDENCE_THRESHOLD
    ).length;
  }, [transcript]);
  const lowConfidenceRatio = useMemo(() => {
    if (!transcript || transcript.words.length === 0) return 0;
    return lowConfidenceCount / transcript.words.length;
  }, [transcript, lowConfidenceCount]);
  const shouldWarnLowConfidence = lowConfidenceCount >= LOW_CONFIDENCE_WARN_MIN_COUNT && lowConfidenceRatio >= LOW_CONFIDENCE_WARN_RATIO;

  const previewSource = useMemo(() => {
    if (previewUrl) return previewUrl;
    if (!selectedVideoAsset) return null;
    return resolveMediaPath(selectedVideoAsset.storage_path);
  }, [previewUrl, selectedVideoAsset]);

  // Fetch waveform peaks whenever video asset changes
  useEffect(() => {
    if (!selectedVideoAsset) { setWaveformPeaks([]); return; }
    let cancelled = false;
    api.getWaveform(selectedVideoAsset.id).then((data) => {
      if (!cancelled) setWaveformPeaks(data.peaks);
    }).catch(() => { if (!cancelled) setWaveformPeaks([]); });
    return () => { cancelled = true; };
  }, [selectedVideoAsset]);

  // Search matches
  const searchMatchIds = useMemo(() => {
    if (!transcript || !searchQuery.trim()) return [] as string[];
    const q = searchQuery.toLowerCase().trim();
    return transcript.words
      .filter((word) => word.text.toLowerCase().includes(q))
      .map((word) => word.id);
  }, [transcript, searchQuery]);

  // Filler word IDs
  const fillerWordIds = useMemo(() => {
    if (!transcript) return new Set<string>();
    return new Set(transcript.words.filter((w) => isFillerWord(w.text)).map((w) => w.id));
  }, [transcript]);

  // ── Undo/Redo helpers ──────────────────────────────────────────────
  const pushUndo = useCallback(() => {
    undoStack.current.push({ deletedIds: new Set(deletedWordIds) });
    if (undoStack.current.length > MAX_UNDO) undoStack.current.shift();
    redoStack.current = [];
  }, [deletedWordIds]);

  function undo() {
    const entry = undoStack.current.pop();
    if (!entry) return;
    redoStack.current.push({ deletedIds: new Set(deletedWordIds) });
    setDeletedWordIds(entry.deletedIds);
  }

  function redo() {
    const entry = redoStack.current.pop();
    if (!entry) return;
    undoStack.current.push({ deletedIds: new Set(deletedWordIds) });
    setDeletedWordIds(entry.deletedIds);
  }

  // ── Core actions ───────────────────────────────────────────────────
  async function refreshMedia(projectId: string) {
    const items = await api.listMedia(projectId);
    setMedia(items);
    const firstVideo = items.find((asset) => asset.media_type === "video");
    if (!selectedAssetId && firstVideo) {
      setSelectedAssetId(firstVideo.id);
    }
  }

  async function queuePreview(force = false) {
    if (!project || queueingPreview) return;
    if (!force && previewJob && (previewJob.status === "queued" || previewJob.status === "running")) {
      pendingPreviewRefreshRef.current = true;
      setNotice("Preview render in progress. Latest edit will render next.");
      return;
    }
    setQueueingPreview(true);
    setError(null);
    try {
      const job = await api.renderPreview(project.id, force);
      setPreviewJob(job);
      if (job.status === "completed" && job.output_path) {
        setPreviewUrl(resolveMediaPath(job.output_path));
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setQueueingPreview(false);
    }
  }

  function updateDeletedWords(wordIds: string[], deleted: boolean) {
    if (!wordIds.length) return;
    pushUndo();
    setDeletedWordIds((prev) => {
      const next = new Set(prev);
      wordIds.forEach((id) => {
        if (deleted) {
          next.add(id);
        } else {
          next.delete(id);
        }
      });
      return next;
    });
  }

  async function applyCut(signature: string, keptIds: string[]) {
    if (!project || !transcript || applyingCut) return;
    if (!keptIds.length) {
      setError("At least one word must remain. Restore some words before applying.");
      return;
    }

    setApplyingCut(true);
    setError(null);
    try {
      const result = await api.applyTranscriptCut(project.id, transcript.id, keptIds);
      setProject((prev) => (prev ? { ...prev, timeline: result.timeline } : prev));
      lastAppliedSignatureRef.current = signature;
      setNotice(`Cut applied. Removed ${result.removed_word_count} word${result.removed_word_count === 1 ? "" : "s"}.`);
      await queuePreview();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setApplyingCut(false);
    }
  }

  async function createProject() {
    setCreatingProject(true);
    setError(null);
    try {
      const created = await api.createProject(projectName.trim() || "Untitled Project");
      setProject(created);
      setMedia([]);
      setSelectedAssetId(null);
      setTranscript(null);
      setDeletedWordIds(new Set());
      setSelectedWordIds(new Set());
      setAnchorWordId(null);
      setPreviewJob(null);
      setPreviewUrl(null);
      undoStack.current = [];
      redoStack.current = [];
      setNotice("Project created. Upload a video to start.");
      await refreshMedia(created.id);
    } catch (err) {
      setError((err as Error).message);
      setNotice(null);
    } finally {
      setCreatingProject(false);
    }
  }

  async function uploadVideo(file: File) {
    if (!project) return;
    setUploading(true);
    setError(null);
    try {
      const uploaded = await api.uploadMedia(project.id, file);
      setMedia((prev) => [uploaded, ...prev]);
      if (uploaded.media_type === "video") {
        setSelectedAssetId(uploaded.id);
        setPreviewUrl(resolveMediaPath(uploaded.storage_path));
      }
      setNotice("Video uploaded.");
    } catch (err) {
      setError((err as Error).message);
      setNotice(null);
    } finally {
      setUploading(false);
    }
  }

  async function generateTranscript() {
    if (!project || !selectedVideoAsset) return;
    setGeneratingTranscript(true);
    setError(null);
    try {
      const response = await api.generateTranscript(project.id, selectedVideoAsset.id);
      setTranscript(response.transcript);
      setProject((prev) => (prev ? { ...prev, timeline: response.timeline } : prev));
      setDeletedWordIds(new Set());
      setSelectedWordIds(new Set());
      setAnchorWordId(null);
      lastAppliedSignatureRef.current = "";
      undoStack.current = [];
      redoStack.current = [];
      setNotice(
        response.transcript.is_mock
          ? "Transcript generated (fallback mode). Install faster-whisper for higher accuracy."
          : "Transcript generated with word timestamps."
      );
      await queuePreview();
    } catch (err) {
      setError((err as Error).message);
      setNotice(null);
    } finally {
      setGeneratingTranscript(false);
    }
  }

  async function runVibeAction(action: VibeAction) {
    if (!project || !selectedVideoAsset || runningAction) return;
    setRunningAction(action);
    setError(null);
    try {
      const response = await api.applyVibeAction(project.id, action, selectedVideoAsset.id);
      setProject((prev) => (prev ? { ...prev, timeline: response.timeline } : prev));
      setPreviewJob(response.preview_job);
      if (response.preview_job.output_path) {
        setPreviewUrl(resolveMediaPath(response.preview_job.output_path));
      }
      if (response.transcript_id) {
        const latestTranscript = await api.getTranscript(project.id, response.transcript_id);
        setTranscript(latestTranscript);
      }
      setNotice(response.details ?? "Action applied.");
    } catch (err) {
      setError((err as Error).message);
      setNotice(null);
    } finally {
      setRunningAction(null);
    }
  }

  // ── Word interaction ───────────────────────────────────────────────
  function markSelectionDeleted() {
    updateDeletedWords(Array.from(selectedWordIds), true);
  }

  function restoreSelection() {
    updateDeletedWords(Array.from(selectedWordIds), false);
  }

  function restoreAllText() {
    pushUndo();
    setDeletedWordIds(new Set());
    setSelectedWordIds(new Set());
    setAnchorWordId(null);
  }

  function removeFillerWords() {
    if (!fillerWordIds.size) return;
    updateDeletedWords(Array.from(fillerWordIds), true);
    setNotice(`Marked ${fillerWordIds.size} filler word${fillerWordIds.size === 1 ? "" : "s"} as deleted.`);
  }

  function selectWord(wordId: string, shiftHeld: boolean) {
    if (!transcript) return;
    if (!shiftHeld || !anchorWordId || !transcriptWordIndex.has(anchorWordId) || !transcriptWordIndex.has(wordId)) {
      setAnchorWordId(wordId);
      setSelectedWordIds(new Set([wordId]));
      return;
    }

    const anchorIndex = transcriptWordIndex.get(anchorWordId) ?? 0;
    const currentIndex = transcriptWordIndex.get(wordId) ?? 0;
    const minIndex = Math.min(anchorIndex, currentIndex);
    const maxIndex = Math.max(anchorIndex, currentIndex);
    const range = transcript.words.slice(minIndex, maxIndex + 1).map((word) => word.id);
    setSelectedWordIds(new Set(range));
  }

  function selectWordRange(fromId: string, toId: string) {
    if (!transcript) return;
    const fromIdx = transcriptWordIndex.get(fromId) ?? 0;
    const toIdx = transcriptWordIndex.get(toId) ?? 0;
    const minIdx = Math.min(fromIdx, toIdx);
    const maxIdx = Math.max(fromIdx, toIdx);
    const range = transcript.words.slice(minIdx, maxIdx + 1).map((w) => w.id);
    setSelectedWordIds(new Set(range));
  }

  function toggleWordDeleted(wordId: string) {
    pushUndo();
    setDeletedWordIds((prev) => {
      const next = new Set(prev);
      if (next.has(wordId)) {
        next.delete(wordId);
      } else {
        next.add(wordId);
      }
      return next;
    });
  }

  function toggleBlock(block: TextBlock) {
    const allDeleted = block.wordIds.every((id) => deletedWordIds.has(id));
    updateDeletedWords(block.wordIds, !allDeleted);
  }

  function seekToWord(word: TranscriptWord) {
    if (!videoRef.current) return;
    videoRef.current.currentTime = Math.max(0, word.start_sec);
  }

  // ── Inline editing ─────────────────────────────────────────────────
  function startEditing(word: TranscriptWord) {
    setEditingWordId(word.id);
    setEditingWordText(word.text);
    setTimeout(() => editInputRef.current?.focus(), 0);
  }

  function commitEdit() {
    if (!editingWordId || !transcript) {
      setEditingWordId(null);
      return;
    }
    const trimmed = editingWordText.trim();
    if (!trimmed) {
      setEditingWordId(null);
      return;
    }
    // Update word text locally
    setTranscript((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        words: prev.words.map((w) =>
          w.id === editingWordId ? { ...w, text: trimmed } : w
        ),
        text: prev.words.map((w) => (w.id === editingWordId ? trimmed : w.text)).join(" ")
      };
    });
    // Fire-and-forget backend update
    if (project && transcript) {
      api.updateWordText(transcript.id, editingWordId, trimmed, project.id).catch(() => {
        /* ignore – local state is source of truth for now */
      });
    }
    setEditingWordId(null);
  }

  function cancelEdit() {
    setEditingWordId(null);
    setEditingWordText("");
  }

  // ── Search navigation ──────────────────────────────────────────────
  function navigateSearch(direction: 1 | -1) {
    if (!searchMatchIds.length) return;
    const nextIdx = (searchMatchIndex + direction + searchMatchIds.length) % searchMatchIds.length;
    setSearchMatchIndex(nextIdx);
    // Scroll to matched word
    const wordEl = document.getElementById(`word-${searchMatchIds[nextIdx]}`);
    wordEl?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  // ── Effects ────────────────────────────────────────────────────────
  useEffect(() => {
    let active = true;
    void api
      .health()
      .then(() => {
        if (active) setBackendStatus("ok");
      })
      .catch(() => {
        if (active) setBackendStatus("down");
      });
    return () => {
      active = false;
    };
  }, []);

  // Auto-apply cut debounce
  useEffect(() => {
    if (!project || !transcript) return;
    if (applyingCut) return;
    if (deletedSignature === lastAppliedSignatureRef.current) return;
    const handle = window.setTimeout(() => {
      void applyCut(deletedSignature, keptWordIds);
    }, 450);
    return () => window.clearTimeout(handle);
  }, [project?.id, transcript?.id, deletedSignature, keptWordIds, applyingCut]);

  // Preview polling
  useEffect(() => {
    if (!previewJob || (previewJob.status !== "queued" && previewJob.status !== "running")) return;
    const interval = window.setInterval(async () => {
      try {
        const refreshed = await api.getJob(previewJob.id);
        setPreviewJob(refreshed);
        if (refreshed.status === "completed" && refreshed.output_path) {
          setPreviewUrl(resolveMediaPath(refreshed.output_path));
        }
        if (refreshed.status === "failed") {
          setError(refreshed.error ?? "Preview render failed. Check logs.");
        }
        if ((refreshed.status === "completed" || refreshed.status === "failed") && pendingPreviewRefreshRef.current) {
          pendingPreviewRefreshRef.current = false;
          void queuePreview(true);
        }
      } catch {
        // Ignore transient polling errors
      }
    }, 1000);
    return () => window.clearInterval(interval);
  }, [previewJob]);

  // Keyboard shortcuts
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      // Delete/Backspace — delete selected words
      if ((event.key === "Delete" || event.key === "Backspace") && selectedWordIds.size > 0 && !editingWordId) {
        event.preventDefault();
        updateDeletedWords(Array.from(selectedWordIds), true);
      }
      // Ctrl+Z — undo
      if (event.key === "z" && (event.ctrlKey || event.metaKey) && !event.shiftKey) {
        event.preventDefault();
        undo();
      }
      // Ctrl+Shift+Z — redo
      if (event.key === "z" && (event.ctrlKey || event.metaKey) && event.shiftKey) {
        event.preventDefault();
        redo();
      }
      // Ctrl+Y — redo (alternative)
      if (event.key === "y" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        redo();
      }
      // Escape — deselect all / cancel edit
      if (event.key === "Escape") {
        if (editingWordId) {
          cancelEdit();
        } else {
          setSelectedWordIds(new Set());
          setAnchorWordId(null);
        }
      }
      // Ctrl+F — focus search
      if (event.key === "f" && (event.ctrlKey || event.metaKey) && transcript) {
        event.preventDefault();
        document.getElementById("transcript-search")?.focus();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectedWordIds, editingWordId, transcript, deletedWordIds, pushUndo]);

  // Auto-scroll to active word during playback
  useEffect(() => {
    if (!activeWordId || editingWordId) return;
    const el = document.getElementById(`word-${activeWordId}`);
    if (el && transcriptBoxRef.current) {
      const box = transcriptBoxRef.current;
      const elTop = el.offsetTop - box.offsetTop;
      const elBottom = elTop + el.offsetHeight;
      const scrollTop = box.scrollTop;
      const boxHeight = box.clientHeight;
      if (elTop < scrollTop || elBottom > scrollTop + boxHeight) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  }, [activeWordId, editingWordId]);

  // Drag selection handler — mouseup
  useEffect(() => {
    function onMouseUp() {
      isDragging.current = false;
      dragStartWordId.current = null;
    }
    window.addEventListener("mouseup", onMouseUp);
    return () => window.removeEventListener("mouseup", onMouseUp);
  }, []);

  return (
    <div className="appShell">
      <header className="topBar">
        <div>
          <p className="eyebrow">AI Video Editor</p>
          <h1>Text-Based Video Editor</h1>
          <p className="subhead">Edit video by editing text. Click, select, delete words — your video updates automatically.</p>
        </div>
        <div className="statusPill">Backend: {backendStatus === "checking" ? "checking" : backendStatus}</div>
      </header>

      <section className="controls card">
        <input
          value={projectName}
          onChange={(event) => setProjectName(event.target.value)}
          placeholder="Project name"
          className="controlInput"
        />
        <button onClick={createProject} disabled={creatingProject}>
          {creatingProject ? "Creating..." : "Create Project"}
        </button>
        <label className="uploadBtn">
          <input
            type="file"
            accept="video/*"
            disabled={!project || uploading}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                void uploadVideo(file);
              }
              event.currentTarget.value = "";
            }}
          />
          {uploading ? "Uploading..." : "Upload Video"}
        </label>
        <select
          disabled={!project || !videoAssets.length}
          value={selectedVideoAsset?.id ?? ""}
          onChange={(event) => {
            const nextId = event.target.value;
            setSelectedAssetId(nextId || null);
            const selected = videoAssets.find((asset) => asset.id === nextId);
            if (selected) {
              setPreviewUrl(resolveMediaPath(selected.storage_path));
            }
          }}
        >
          {!videoAssets.length && <option value="">No uploaded videos</option>}
          {videoAssets.map((asset) => (
            <option key={asset.id} value={asset.id}>
              {asset.filename}
            </option>
          ))}
        </select>
        <button onClick={generateTranscript} disabled={!project || !selectedVideoAsset || generatingTranscript}>
          {generatingTranscript ? "Generating..." : "Generate Transcript"}
        </button>
      </section>

      {project && (
        <section className="projectMeta card">
          <span>{project.name}</span>
          <span>{project.timeline.resolution.width}x{project.timeline.resolution.height}</span>
          <span>{project.timeline.fps} fps</span>
          <span>{formatSeconds(project.timeline.duration_sec)} duration</span>
        </section>
      )}

      {error && <div className="message error">{error}</div>}
      {notice && <div className="message notice">{notice}</div>}

      {!project && <p className="empty">Create a project, upload a video, and generate transcript to start.</p>}

      {project && (
        <>
          <main className="twoPanel">
            <section className="panel card">
              <h2>Transcript Editor</h2>
              {!transcript && <p className="muted">Generate transcript from an uploaded video to start text-based editing.</p>}
              {transcript && (
                <>
                  <p className="muted hint">
                    <strong>Click</strong> word to select & seek &nbsp;·&nbsp;
                    <strong>Shift+click</strong> range &nbsp;·&nbsp;
                    <strong>Drag</strong> to select &nbsp;·&nbsp;
                    <strong>Double-click</strong> to edit text &nbsp;·&nbsp;
                    <strong>Del/⌫</strong> delete &nbsp;·&nbsp;
                    <strong>Ctrl+Z</strong> undo
                  </p>

                  {transcript.is_mock && <p className="warning">Fallback transcript active; install faster-whisper for accurate speech text.</p>}
                  {shouldWarnLowConfidence && (
                    <p className="warning">
                      {lowConfidenceCount} low-confidence word{lowConfidenceCount === 1 ? "" : "s"} (~{(lowConfidenceRatio * 100).toFixed(0)}%).
                    </p>
                  )}

                  {/* ── Search bar ────────────────────────────────── */}
                  <div className="searchBar">
                    <svg className="searchIcon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
                    </svg>
                    <input
                      id="transcript-search"
                      type="text"
                      placeholder="Search words... (Ctrl+F)"
                      value={searchQuery}
                      onChange={(e) => { setSearchQuery(e.target.value); setSearchMatchIndex(0); }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") navigateSearch(e.shiftKey ? -1 : 1);
                        if (e.key === "Escape") { setSearchQuery(""); (e.target as HTMLInputElement).blur(); }
                      }}
                    />
                    {searchQuery && (
                      <span className="searchCount">
                        {searchMatchIds.length ? `${searchMatchIndex + 1}/${searchMatchIds.length}` : "0 matches"}
                        <button className="searchNav" onClick={() => navigateSearch(-1)} title="Previous">▲</button>
                        <button className="searchNav" onClick={() => navigateSearch(1)} title="Next">▼</button>
                      </span>
                    )}
                  </div>

                  {/* ── Action toolbar ────────────────────────────── */}
                  <div className="wordActions toolbar">
                    <button onClick={markSelectionDeleted} disabled={!selectedWordIds.size} title="Delete selected words">
                      🗑 Delete
                    </button>
                    <button onClick={restoreSelection} disabled={!selectedWordIds.size} title="Restore selected words">
                      ↩ Restore
                    </button>
                    <button onClick={restoreAllText} disabled={!deletedWordIds.size} title="Restore all deleted words">
                      ↻ Restore All
                    </button>
                    <div className="toolbarSep" />
                    <button onClick={removeFillerWords} disabled={!fillerWordIds.size || applyingCut} title="Remove um, uh, like, etc.">
                      ✂ Remove Fillers {fillerWordIds.size > 0 && <span className="badge">{fillerWordIds.size}</span>}
                    </button>
                    <div className="toolbarSep" />
                    <button onClick={undo} disabled={!undoStack.current.length} title="Undo (Ctrl+Z)">
                      ↶ Undo
                    </button>
                    <button onClick={redo} disabled={!redoStack.current.length} title="Redo (Ctrl+Shift+Z)">
                      ↷ Redo
                    </button>
                    <div className="toolbarSep" />
                    <button onClick={() => void applyCut(deletedSignature, keptWordIds)} disabled={applyingCut || !transcript}>
                      {applyingCut ? "Applying..." : "▶ Apply Cut"}
                    </button>
                  </div>

                  {/* ── Interactive word grid ─────────────────────── */}
                  <div
                    className="transcriptBox"
                    ref={transcriptBoxRef}
                    onMouseLeave={() => { isDragging.current = false; }}
                  >
                    {transcript.words.map((word) => {
                      const isDeleted = deletedWordIds.has(word.id);
                      const isSelected = selectedWordIds.has(word.id);
                      const isActive = activeWordId === word.id && !isDeleted;
                      const isFiller = fillerWordIds.has(word.id) && !isDeleted;
                      const isSearchMatch = searchMatchIds.includes(word.id);
                      const isCurrentMatch = searchMatchIds[searchMatchIndex] === word.id;
                      const hasLowConfidence =
                        !isDeleted && typeof word.confidence === "number" && word.confidence < LOW_CONFIDENCE_THRESHOLD;

                      const className = [
                        "word",
                        isDeleted ? "deleted" : "",
                        isSelected ? "selected" : "",
                        isActive ? "active" : "",
                        isFiller ? "filler" : "",
                        isSearchMatch ? "searchMatch" : "",
                        isCurrentMatch ? "currentMatch" : "",
                        hasLowConfidence ? "lowConfidence" : ""
                      ]
                        .filter(Boolean)
                        .join(" ");

                      const confidenceHint =
                        typeof word.confidence === "number" ? ` · ${(word.confidence * 100).toFixed(0)}%` : "";

                      // Inline editing mode for this word
                      if (editingWordId === word.id) {
                        return (
                          <input
                            key={word.id}
                            ref={editInputRef}
                            className="wordEditInput"
                            value={editingWordText}
                            onChange={(e) => setEditingWordText(e.target.value)}
                            onBlur={commitEdit}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") commitEdit();
                              if (e.key === "Escape") cancelEdit();
                              e.stopPropagation();
                            }}
                            style={{ width: `${Math.max(editingWordText.length + 2, 4)}ch` }}
                          />
                        );
                      }

                      return (
                        <button
                          key={word.id}
                          id={`word-${word.id}`}
                          type="button"
                          className={className}
                          ref={isActive ? activeWordRef : undefined}
                          onMouseDown={(event) => {
                            if (event.detail >= 2) return; // let double-click handle
                            isDragging.current = true;
                            dragStartWordId.current = word.id;
                            selectWord(word.id, event.shiftKey);
                            seekToWord(word);
                          }}
                          onMouseEnter={() => {
                            if (isDragging.current && dragStartWordId.current) {
                              selectWordRange(dragStartWordId.current, word.id);
                            }
                          }}
                          onDoubleClick={() => startEditing(word)}
                          title={`${formatSeconds(word.start_sec)} – ${formatSeconds(word.end_sec)}${confidenceHint}`}
                        >
                          {word.text}
                        </button>
                      );
                    })}
                  </div>

                  {/* ── Sentence shortcuts ────────────────────────── */}
                  <details className="shortcutSection">
                    <summary><h3>Sentence Shortcuts ({sentenceBlocks.length})</h3></summary>
                    <div className="shortcutList">
                      {sentenceBlocks.map((block) => {
                        const allDeleted = block.wordIds.every((id) => deletedWordIds.has(id));
                        return (
                          <button
                            key={block.id}
                            type="button"
                            className={`segmentBtn ${allDeleted ? "deleted" : ""}`}
                            onClick={() => toggleBlock(block)}
                            title={`${formatSeconds(block.startSec)} – ${formatSeconds(block.endSec)}`}
                          >
                            <span className="segTime">{formatSeconds(block.startSec)}</span>
                            {block.text}
                          </button>
                        );
                      })}
                    </div>
                  </details>

                  <details className="shortcutSection">
                    <summary><h3>Paragraph Shortcuts ({paragraphBlocks.length})</h3></summary>
                    <div className="shortcutList">
                      {paragraphBlocks.map((block) => {
                        const allDeleted = block.wordIds.every((id) => deletedWordIds.has(id));
                        return (
                          <button
                            key={block.id}
                            type="button"
                            className={`segmentBtn ${allDeleted ? "deleted" : ""}`}
                            onClick={() => toggleBlock(block)}
                            title={`${formatSeconds(block.startSec)} – ${formatSeconds(block.endSec)}`}
                          >
                            <span className="segTime">{formatSeconds(block.startSec)}</span>
                            {block.text}
                          </button>
                        );
                      })}
                    </div>
                  </details>
                </>
              )}
            </section>

            <section className="panel card">
              <h2>Video Preview + AI Actions</h2>
              {!previewSource && <p className="muted">Upload a video to preview.</p>}
              {previewSource && (
                <video
                  ref={videoRef}
                  key={previewSource}
                  src={previewSource}
                  controls
                  className="previewVideo"
                  onTimeUpdate={(event) => setCurrentTimeSec(event.currentTarget.currentTime)}
                />
              )}
              <div className="previewMeta">
                <span>Playhead: {formatSeconds(currentTimeSec)}</span>
                <span>Preview: {previewJob ? `${previewJob.status} (${previewJob.progress}%)` : "not queued"}</span>
              </div>
              {previewJob?.status === "failed" && (
                <p className="warning">Preview failed: {previewJob.error ?? "Unknown render error"}</p>
              )}
              <div className="wordActions">
                <button onClick={() => void queuePreview()} disabled={!project || queueingPreview}>
                  {queueingPreview ? "Queueing..." : "Render Preview"}
                </button>
              </div>

              <section className="aiPanel">
                <h3>AI Action Panel</h3>
                <p className="muted">One-click AI improvements. No manual timeline editing needed.</p>
                <div className="actionGrid">
                  <button
                    onClick={() => void runVibeAction("add_subtitles")}
                    disabled={!selectedVideoAsset || runningAction !== null}
                  >
                    {runningAction === "add_subtitles" ? "Applying..." : "📝 Add Subtitles"}
                  </button>
                  <button
                    onClick={() => void runVibeAction("auto_cut_pauses")}
                    disabled={!selectedVideoAsset || runningAction !== null}
                  >
                    {runningAction === "auto_cut_pauses" ? "Applying..." : "✂️ Auto Cut Pauses"}
                  </button>
                  <button
                    onClick={() => void runVibeAction("trim_start_end")}
                    disabled={!selectedVideoAsset || runningAction !== null}
                  >
                    {runningAction === "trim_start_end" ? "Applying..." : "🔪 Trim Start & End"}
                  </button>
                </div>
              </section>
            </section>
          </main>

          {/* ── Visual Timeline ─────────────────────────── */}
          {transcript && (
            <Timeline
              words={transcript.words}
              durationSec={transcript.duration_sec || project.timeline.duration_sec}
              currentTimeSec={currentTimeSec}
              deletedWordIds={deletedWordIds}
              selectedWordIds={selectedWordIds}
              activeWordId={activeWordId}
              waveformPeaks={waveformPeaks}
              onSeek={(sec) => {
                if (videoRef.current) videoRef.current.currentTime = sec;
                setCurrentTimeSec(sec);
              }}
              onSelectWord={(id, shift) => {
                selectWord(id, shift);
                const wd = transcript.words.find((w) => w.id === id);
                if (wd) seekToWord(wd);
              }}
              onSelectWordsInRange={(startSec, endSec) => {
                const ids = transcript.words
                  .filter((w) => w.start_sec >= startSec && w.end_sec <= endSec)
                  .map((w) => w.id);
                setSelectedWordIds(new Set(ids));
              }}
              onDeleteSelected={markSelectionDeleted}
              onRestoreSelected={restoreSelection}
            />
          )}
        </>
      )}
    </div >
  );
}

export default App;
